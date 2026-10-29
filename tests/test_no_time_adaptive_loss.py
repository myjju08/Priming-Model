"""CPU/math contracts for original adaptive MSE with only the noise interval changed."""
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
import unittest
from unittest.mock import patch

import torch
from torch import nn

import pretrain
from losses.reconstruction import reconstruction_loss
from models import priming
from models.encoders import perceptual as encoders


class StubLPIPS(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.5))

    def forward(self, x, y):
        return self.scale * (x-y).square().mean((1, 2, 3))


class StubConvNext(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.7))

    def forward(self, x):
        return self.scale * x.mean((2, 3))


class NoTimeAdaptiveLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, name='b10_diffusion_no_time_adaptive_perceptual'):
        return pretrain.load_config(f'configs/pretrain/{name}.yaml')[0]

    def compact(self, config):
        config = deepcopy(config)
        config['data'].update(img_size=8, num_classes=3)
        config['model']['patch_size'] = 4
        with patch.dict(priming.OST_models,
                        OST_B=partial(priming.OST, depth=2, hidden_size=32, num_heads=4)):
            return pretrain.make_model(config, device='cpu')

    def auxiliary(self, config):
        # Keep the actual production wrapper, clamp/normalization, paired random
        # crop, masks and adaptive perceptual formula; only pretrained encoders
        # are replaced with small frozen differentiable CPU stubs.
        with patch.object(encoders, 'LPIPS', StubLPIPS), \
             patch.object(encoders, 'ConvNextFeatureExtractor', StubConvNext):
            return pretrain.build_auxiliary(config, 'b', 'unused.pth', device='cpu')

    def test_adaptive_value_and_gradient_use_detached_per_image_denominator(self):
        prediction = torch.tensor([.1, .2, -.1, .4, 2., -1., 3., .5],
                                  dtype=torch.float64).reshape(2, 1, 2, 2).requires_grad_()
        target = torch.zeros_like(prediction)
        epsilon = 1e-4
        mse = (prediction-target).square().mean((1, 2, 3))
        actual, metrics = reconstruction_loss(prediction, target, norm_p=1., norm_eps=epsilon)
        expected_value = (mse/(mse.detach()+epsilon)).mean()
        torch.testing.assert_close(actual, expected_value, rtol=0, atol=0)
        torch.testing.assert_close(metrics['loss_mse'], mse.mean(), rtol=0, atol=0)
        gradient, = torch.autograd.grad(actual, prediction, retain_graph=True)
        expected_gradient = 2*(prediction-target)/(2*4*(mse.detach()+epsilon)[:, None, None, None])
        torch.testing.assert_close(gradient, expected_gradient, rtol=1e-14, atol=1e-14)
        wrong = (mse/(mse+epsilon)).mean()
        wrong_gradient, = torch.autograd.grad(wrong, prediction, retain_graph=True)
        self.assertGreater((gradient-wrong_gradient).abs().max().item(), .1)
        # First-order gradient equivalence does not imply equal logged values.
        log_gradient, = torch.autograd.grad((mse+epsilon).log().mean(), prediction)
        torch.testing.assert_close(gradient, log_gradient, rtol=1e-14, atol=1e-14)
        self.assertFalse(torch.equal(actual.detach(), (mse+epsilon).log().mean().detach()))

    def test_exact_reconstruction_has_finite_zero_loss_and_gradient(self):
        prediction = torch.tensor([0., 1., -1., .5]).reshape(1, 1, 2, 2).requires_grad_()
        loss, _ = reconstruction_loss(prediction, prediction.detach().clone(), norm_p=1., norm_eps=1e-4)
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))

    def test_matched_recipes_differ_only_in_noise_interval_and_variant(self):
        variable = self.config()
        control = self.config('b10_fixed_noise_no_time_adaptive_perceptual')
        self.assertEqual(variable['experiment']['loss_weighting'], 'adaptive_residual')
        self.assertEqual(variable['experiment']['conditioning'], 'none')
        self.assertEqual(variable['loss']['norm_p'], 1.)
        self.assertEqual(variable['loss']['norm_eps'], 1e-4)
        self.assertEqual(variable['loss']['aux'], self.config('b160_perceptual')['loss']['aux'])
        self.assertEqual(variable['loss']['perceptual_threshold'], 1.)
        self.assertEqual(variable['train']['epochs']*variable['train']['steps_per_epoch'], 12510)
        control['experiment']['variant'] = 'variable_noise'
        control['model'].update(t_min=0., t_max=1.)
        self.assertEqual(variable, control)

    def test_original_model_schema_parameters_and_initialization_are_preserved(self):
        original, variable = self.config('b160_perceptual'), self.config()
        torch.manual_seed(501)
        before = self.compact(original)
        original_rng = torch.get_rng_state()
        torch.manual_seed(501)
        after = self.compact(variable)
        self.assertTrue(torch.equal(torch.get_rng_state(), original_rng))
        for name, value in before.state_dict().items():
            torch.testing.assert_close(after.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(set(before.state_dict()), set(after.state_dict()))
        self.assertFalse(after.timestep_conditioning)
        self.assertFalse(hasattr(after.net, 'noise_t_embedder'))
        self.assertFalse(any('adaLN' in name for name in after.state_dict()))
        with torch.device('meta'):
            full_before = pretrain.make_model(original, device='meta')
            full_after = pretrain.make_model(variable, device='meta')
        self.assertEqual(sum(p.numel() for p in full_after.parameters()), 115063936)
        self.assertEqual({name: value.shape for name, value in full_before.state_dict().items()},
                         {name: value.shape for name, value in full_after.state_dict().items()})

    def test_controlled_t08_matches_original_loss_parameter_gradients_and_rng_with_aux(self):
        configs = [self.config('b160_perceptual'), self.config(),
                   self.config('b10_fixed_noise_no_time_adaptive_perceptual')]
        models = [self.compact(config) for config in configs]
        with torch.no_grad():
            for parameter in models[0].parameters():
                parameter.normal_(std=.1)
        for model in models[1:]:
            model.load_state_dict(models[0].state_dict(), strict=True)
        auxiliaries = [self.auxiliary(config) for config in configs]
        images = torch.linspace(-.9, .9, 384).reshape(2, 3, 8, 8)
        labels, times = torch.tensor([0, 2]), torch.full((2, 1, 1, 1), .8)
        results = []
        for index, (model, auxiliary) in enumerate(zip(models, auxiliaries)):
            calls = []
            hook = model.net.register_forward_pre_hook(lambda _, args, kwargs: calls.append(kwargs), with_kwargs=True)
            time_control = patch.object(model, 'sample_t', return_value=times) if index == 1 else nullcontext()
            try:
                torch.manual_seed(503)
                with time_control:
                    loss, metrics = model(images, labels, aux_loss_fn=auxiliary)
                loss.backward()
                results.append((loss.detach(), metrics, torch.get_rng_state(),
                                {name: parameter.grad.clone() for name, parameter in model.named_parameters()}))
            finally:
                hook.remove()
            self.assertEqual(calls, [{}])
        reference = results[0]
        for loss, metrics, rng, gradients in results[1:]:
            torch.testing.assert_close(loss, reference[0], rtol=0, atol=0)
            for name, value in reference[1].items():
                torch.testing.assert_close(metrics[name], value, rtol=0, atol=0)
            self.assertTrue(torch.equal(rng, reference[2]))
            for name, gradient in gradients.items():
                torch.testing.assert_close(gradient, reference[3][name], rtol=0, atol=0)
            self.assertEqual(metrics['perceptual_active_fraction'].item(), 1.)
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in reference[3].values()), 0.)

    def test_random_noise_keeps_adaptive_mse_and_auxiliary_active_at_every_t(self):
        config = self.config()
        model = self.compact(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(std=.1)
        auxiliary = self.auxiliary(config)
        images, labels = torch.randn(4, 3, 8, 8).clamp(-1, 1), torch.tensor([0, 1, 2, 0])
        times = torch.tensor([0., .2, .9, 1.]).reshape(4, 1, 1, 1)
        noise = torch.randn_like(images)
        captured = []

        def record(prediction, target, mask):
            terms = auxiliary(prediction, target, mask)
            captured.append((prediction, target, mask, terms))
            return terms

        inputs = []
        hook = model.net.register_forward_pre_hook(lambda _, args, kwargs: inputs.append((args[0], kwargs)), with_kwargs=True)
        try:
            with patch.object(model, 'sample_t', return_value=times), \
                 patch('models.priming.torch.randn_like', return_value=noise):
                actual, metrics = model(images, labels, aux_loss_fn=record)
        finally:
            hook.remove()
        self.assertEqual(inputs[0][1], {})
        torch.testing.assert_close(inputs[0][0], (1-times)*images+times*noise)
        prediction, target, mask, (aux, lpips, convnext) = captured[0]
        self.assertIs(target, images)
        self.assertTrue(mask.all())
        self.assertTrue((lpips > 0).all())
        self.assertTrue((convnext > 0).all())
        mse = (prediction-images).square().mean((1, 2, 3))
        expected_aux = .4*lpips/(lpips.detach()+.01)+.1*convnext/(convnext.detach()+.01)
        expected = (mse/(mse.detach()+1e-4)+expected_aux).mean()
        torch.testing.assert_close(aux, expected_aux)
        torch.testing.assert_close(actual, expected)
        actual_grad, = torch.autograd.grad(actual, prediction, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, prediction)
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertEqual(metrics['perceptual_active_fraction'].item(), 1.)


if __name__ == '__main__':
    unittest.main()
