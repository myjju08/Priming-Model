"""Conditioning, paired randomness, and released-backbone regression checks."""
from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models import Priming
from models import priming
from models.adaln import AdaLNOST, DiTTimestepEmbedder


def small_model(*, conditioned=True, t_min=.8, t_max=.8):
    kwargs = dict(img_size=16, patch_size=8, num_classes=3, class_tokens=2,
                  guidance_tokens=0, interval_tokens=0, label_drop_prob=0.,
                  t_dist='uniform', t_min=t_min, t_max=t_max)
    if conditioned:
        with patch('models.adaln.AdaLNOST', partial(AdaLNOST, depth=2, hidden_size=32, num_heads=4)):
            return Priming(backbone='OST_B_ADALN', **kwargs)
    with patch.dict(priming.OST_models, OST_B=partial(priming.OST, depth=2, hidden_size=32, num_heads=4)):
        return Priming(backbone='OST_B', **kwargs)


def activate_conditioning(net):
    """Make zero-initialized paths observable without relying on an optimizer."""
    with torch.no_grad():
        nn.init.normal_(net.final_layer.linear.weight, std=.1)
        nn.init.normal_(net.final_layer.adaLN_modulation[-1].weight, std=.1)
        for block in net.blocks:
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=.1)


class AdaLNModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_initialization_and_affine_free_layer_norm(self):
        model = small_model()
        x, y = torch.randn(2, 3, 16, 16), torch.tensor([0, 2])
        output = model.net(x, y, t=torch.tensor([.2, .8]))
        self.assertTrue(torch.equal(output, torch.zeros_like(x)))
        sequence = torch.randn(2, 6, 32)
        condition = torch.randn(2, 32)
        for block in model.net.blocks:
            self.assertFalse(block.norm1.elementwise_affine)
            self.assertFalse(block.norm2.elementwise_affine)
            self.assertTrue(torch.equal(block(sequence, model.net.rope_freqs, condition), sequence))
            self.assertEqual(torch.count_nonzero(block.adaLN_modulation[-1].weight).item(), 0)
            self.assertNotIn('attn_scale', dict(block.named_parameters()))
        self.assertFalse(model.net.final_layer.norm.elementwise_affine)

    def test_timestep_changes_output_and_receives_finite_gradients(self):
        torch.manual_seed(5)
        model = small_model()
        activate_conditioning(model.net)
        x, y = torch.randn(2, 3, 16, 16), torch.tensor([0, 1])
        early = model.net(x, y, t=torch.full((2,), .2))
        late = model.net(x, y, t=torch.full((2,), .8))
        self.assertGreater((early-late).abs().max().item(), 1e-5)
        late.square().mean().backward()
        gradient = model.net.t_embedder.mlp[0].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_corruption_time_and_generation_endpoint_are_explicit(self):
        model = small_model()
        captured = []
        hook = model.net.register_forward_pre_hook(
            lambda module, args, kwargs: captured.append(kwargs['t'].detach().clone()), with_kwargs=True)
        x, y = torch.randn(2, 3, 16, 16), torch.tensor([0, 1])
        loss, metrics = model(x, y)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(captured[-1], torch.full((2, 1, 1, 1), .8)))
        model.sample_images_with_grad(x, y)
        self.assertTrue(torch.equal(captured[-1], torch.ones(2)))
        model.generate(2, y, cfg=1., args=SimpleNamespace(same_noise=False))
        self.assertTrue(torch.equal(captured[-1], torch.ones(2)))
        model(x, y, endpoint_generation=True)
        self.assertTrue(torch.equal(captured[-1], torch.ones(2)))
        hook.remove()

    def test_paired_arms_have_identical_initial_weights_and_noise(self):
        torch.manual_seed(19)
        fixed = small_model()
        torch.manual_seed(19)
        varying = small_model(t_min=0., t_max=1.)
        self.assertEqual(set(fixed.state_dict()), set(varying.state_dict()))
        self.assertTrue(all(torch.equal(v, varying.state_dict()[k]) for k,v in fixed.state_dict().items()))
        torch.manual_seed(21)
        fixed_t = fixed.sample_t(32, 'cpu', torch.float32)
        fixed_noise = torch.randn(32, 3, 16, 16)
        torch.manual_seed(21)
        varying_t = varying.sample_t(32, 'cpu', torch.float32)
        varying_noise = torch.randn(32, 3, 16, 16)
        self.assertTrue(torch.equal(fixed_t, torch.full_like(fixed_t, .8)))
        self.assertTrue(((varying_t >= 0) & (varying_t < 1)).all())
        self.assertGreater(varying_t.std().item(), .1)
        self.assertTrue(torch.equal(fixed_noise, varying_noise))

    def test_legacy_fixed_time_keeps_rng_and_legacy_dispatch(self):
        model = small_model(conditioned=False)
        before = torch.get_rng_state()
        t = model.sample_t(2, 'cpu', torch.float32)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(t, torch.full_like(t, .8)))
        self.assertFalse(any('adaLN' in k or 't_embedder' in k for k in model.state_dict()))
        x, y = torch.randn(2, 3, 16, 16), torch.tensor([0, 1])
        torch.manual_seed(31)
        actual, metrics = model(x,y)
        torch.manual_seed(31)
        labels = model.drop_labels(y)
        noise = torch.randn_like(x)
        prediction = model.net(.2*x+.8*noise, labels)
        mse = (prediction-x).square().mean((1,2,3))
        expected = (mse/(mse.detach()+model.norm_eps)).mean()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(metrics['loss_mse'], mse.mean()))
        self.assertTrue(torch.equal(model.sample_images_with_grad(x,y), model.net(x,y)))

    def test_timestep_embedding_uses_dit_frequency_scale(self):
        embedder = DiTTimestepEmbedder(8, frequency_embedding_size=4)
        captured = []
        hook = embedder.mlp.register_forward_pre_hook(lambda m,a: captured.append(a[0].detach().clone()))
        embedder(torch.tensor([0., .8, 1.]))
        values = torch.tensor([0., 800., 1000.])
        angles = values[:,None] * torch.tensor([1., .01])[None,:]
        expected = torch.cat((angles.cos(), angles.sin()), dim=1)
        torch.testing.assert_close(captured[0], expected, atol=1e-5, rtol=1e-5)
        hook.remove()

    def test_implicit_or_wrong_batch_timestep_is_rejected(self):
        net = small_model().net
        x, y = torch.randn(2,3,16,16), torch.tensor([0,1])
        with self.assertRaises(TypeError):
            net(x,y)
        with self.assertRaisesRegex(ValueError, 'one timestep'):
            net(x,y,t=torch.tensor([.8]))


if __name__ == '__main__':
    unittest.main()
