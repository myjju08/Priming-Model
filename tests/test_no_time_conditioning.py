"""CPU/meta checks for original Priming with an unobserved corruption level."""
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

from losses.perceptual import PerceptualLoss
from models import Priming, priming
from models.encoders import perceptual as encoders
from utils.diffusion_diagnostics import diffusion_panels
from utils.ema import EMA


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


class NoTimeConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, **overrides):
        kwargs = dict(img_size=8, patch_size=4, num_classes=3, class_tokens=2,
                      guidance_tokens=0, interval_tokens=0, label_drop_prob=0.,
                      t_dist="uniform", t_min=0., t_max=1., norm_p=0.,
                      perceptual_threshold=1., timestep_conditioning=False)
        kwargs.update(overrides)
        with patch.dict(priming.OST_models,
                        OST_B=partial(priming.OST, depth=2, hidden_size=32, num_heads=4)):
            return Priming(**kwargs)

    def randomize(self, model):
        # Exercise the full learned path instead of only its initially zero head.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(std=.1)
        return model

    def test_original_initialization_rng_schema_and_forward_are_unchanged(self):
        torch.manual_seed(301)
        fixed = self.model(t_min=.8, t_max=.8, norm_p=1., perceptual_threshold=.8)
        baseline_rng = torch.get_rng_state()
        torch.manual_seed(301)
        mixed = self.model(log_noise_metrics=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), baseline_rng))
        self.assertEqual(set(mixed.state_dict()), set(fixed.state_dict()))
        for name, value in fixed.state_dict().items():
            torch.testing.assert_close(mixed.state_dict()[name], value, rtol=0, atol=0)
        self.assertFalse(any("noise_t" in name or "adaLN" in name for name in mixed.state_dict()))
        self.randomize(fixed)
        mixed.load_state_dict(fixed.state_dict(), strict=True)
        image, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        expected = fixed.net(image, labels)
        torch.testing.assert_close(mixed.net(image, labels), expected, rtol=0, atol=0)
        sequence = mixed.net.pos_embed_func(torch.cat([
            mixed.net.class_tokens + mixed.net.y_embedder(labels).unsqueeze(1),
            mixed.net.x_embedder(image)], dim=1))
        for block in mixed.net.blocks:
            sequence = block(sequence, mixed.net.rope_freqs)
        manual = mixed.net.unpatchify(mixed.net.final_layer(sequence[:, mixed.net.prefix_tokens:]))
        torch.testing.assert_close(expected, manual, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "requires timestep_conditioning"):
            mixed.net(image, labels, noise_t=.8)

    def test_full_original_b_parameter_count_and_schema(self):
        with torch.device("meta"):
            baseline = Priming(guidance_tokens=0, interval_tokens=0)
            mixed = Priming(guidance_tokens=0, interval_tokens=0, timestep_conditioning=False,
                           t_dist="uniform", t_min=0., t_max=1., norm_p=0., log_noise_metrics=True)
        self.assertEqual(sum(p.numel() for p in mixed.parameters()), 115_063_936)
        self.assertEqual({name: value.shape for name, value in mixed.state_dict().items()},
                         {name: value.shape for name, value in baseline.state_dict().items()})
        self.assertEqual(mixed.net.pos_embed.shape, (1, 264, 768))
        self.assertEqual(mixed.net.class_tokens.shape, (8, 768))
        self.assertFalse(hasattr(mixed.net, "noise_t_embedder"))
        self.assertFalse(any(isinstance(module, nn.LayerNorm) for module in mixed.modules()))

    def test_noise_only_changes_input_plain_mse_and_perceptual_cover_every_t(self):
        torch.manual_seed(303)
        model = self.randomize(self.model(log_noise_metrics=True))
        with patch.object(encoders, "LPIPS", StubLPIPS), \
             patch.object(encoders, "ConvNextFeatureExtractor", StubConvNext):
            perceptual = PerceptualLoss(lpips_weight=.4, convnext_weight=.1,
                                       norm_p=1., norm_eps=.01, random_crop=False)
        image = torch.randn(4, 3, 8, 8)
        labels, times = torch.tensor([0, 1, 2, 0]), torch.tensor([0., .2, .9, 1.]).reshape(4, 1, 1, 1)
        noise = torch.randn_like(image)
        calls, outputs, auxiliary_calls = [], [], []
        before = model.net.register_forward_pre_hook(lambda _, args, kwargs: calls.append((args, kwargs)), with_kwargs=True)
        after = model.net.register_forward_hook(lambda _, args, output: outputs.append(output))

        def auxiliary(prediction, target, mask):
            result = perceptual(prediction, target, mask)
            auxiliary_calls.append((prediction, target, mask, result))
            return result

        try:
            with patch.object(model, "sample_t", return_value=times), \
                 patch("models.priming.torch.randn_like", return_value=noise):
                loss, metrics = model(image, labels, aux_loss_fn=auxiliary)
        finally:
            before.remove()
            after.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], {})
        torch.testing.assert_close(calls[0][0][0], (1-times)*image + times*noise)
        prediction, target, mask, (aux, lpips, convnext) = auxiliary_calls[0]
        self.assertIs(target, image)
        self.assertIs(prediction, outputs[0])
        self.assertTrue(mask.all())
        self.assertTrue((lpips > 0).all())
        self.assertTrue((convnext > 0).all())
        expected_aux = .4 * lpips / (lpips.detach()+.01) + .1 * convnext / (convnext.detach()+.01)
        torch.testing.assert_close(aux, expected_aux)
        expected = ((prediction-image).square().mean((1, 2, 3)) + expected_aux).mean()
        torch.testing.assert_close(loss, expected)
        actual_grad, = torch.autograd.grad(loss, prediction, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, prediction, retain_graph=True)
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertTrue((actual_grad.flatten(1).abs().sum(1) > 0).all())
        loss.backward()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in perceptual.parameters()))
        self.assertGreater(model.net.final_layer.linear.linear.weight.grad.abs().sum().item(), 0.)
        torch.testing.assert_close(metrics["noise_t_mean"], times.mean())
        self.assertEqual(metrics["perceptual_active_fraction"].item(), 1.)

    def test_diagnostic_flag_changes_only_returned_metrics_not_rng_loss_or_gradients(self):
        baseline = self.randomize(self.model())
        logged = self.model(log_noise_metrics=True)
        logged.load_state_dict(baseline.state_dict(), strict=True)
        image, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        results = []
        for model in (baseline, logged):
            torch.manual_seed(307)
            loss, metrics = model(image, labels)
            loss.backward()
            results.append((loss.detach(), metrics, torch.get_rng_state(),
                            {name: p.grad.clone() for name, p in model.named_parameters()}))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        self.assertTrue(torch.equal(results[0][2], results[1][2]))
        self.assertEqual(set(results[0][1]), {"loss_mse"})
        self.assertEqual(set(results[1][1]), {"loss_mse", "noise_t_mean"})
        for name, gradient in results[0][3].items():
            torch.testing.assert_close(gradient, results[1][3][name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "log_noise_metrics must be a boolean"):
            self.model(log_noise_metrics="true")

    def test_uniform_sampler_uses_per_image_noise_levels(self):
        model = self.model()
        torch.manual_seed(309)
        times = model.sample_t(4096, "cpu", torch.float32)
        self.assertEqual(times.shape, (4096, 1, 1, 1))
        self.assertTrue(((times >= 0.) & (times < 1.)).all())
        self.assertLess(abs(times.mean().item()-.5), .02)
        self.assertGreater(times.flatten().unique().numel(), 4000)

    def test_generation_consumes_pure_noise_once_without_time_and_preserves_gradients(self):
        model = self.randomize(self.model())
        noise, labels = torch.randn(2, 3, 8, 8, requires_grad=True), torch.tensor([0, 2])
        expected = model.net(noise, labels)
        calls = []
        hook = model.net.register_forward_pre_hook(lambda _, args, kwargs: calls.append((args, kwargs)), with_kwargs=True)
        try:
            generated = model(noise, labels, endpoint_generation=True)
        finally:
            hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0][0], noise)
        self.assertEqual(calls[0][1], {})
        torch.testing.assert_close(generated, expected, rtol=0, atol=0)
        torch.testing.assert_close(model.sample_images_with_grad(noise, labels), expected, rtol=0, atol=0)
        torch.testing.assert_close(model.generate(2, labels, z_t=noise), expected, rtol=0, atol=0)
        generated.square().mean().backward()
        self.assertGreater(noise.grad.abs().sum().item(), 0.)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_unconditioned_panels_route_no_time_and_restore_rng_modes_weights(self):
        model = self.randomize(self.model())
        ema = EMA(model)
        model.train()
        model.net.eval()
        rows = [(torch.ones(3, 8, 8)*.25, 0), (torch.ones(3, 8, 8)*-.25, 1)]
        rng, state = torch.get_rng_state().clone(), deepcopy(model.state_dict())
        calls = []
        hook = model.net.register_forward_pre_hook(lambda _, args, kwargs: calls.append((args[0].clone(), kwargs)), with_kwargs=True)
        try:
            with TemporaryDirectory() as directory:
                report, path = diffusion_panels(directory, 0, model, ema, rows, device="cpu")
                self.assertTrue(path.is_file())
                saved = json.loads((Path(directory)/"visualization/step_0000000.json").read_text())
                self.assertEqual(saved["conditioning"], "none")
        finally:
            hook.remove()
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(not kwargs for _, kwargs in calls))
        image = torch.stack([x for x, _ in rows])
        noise = calls[3][0]
        for index, t in enumerate((.2, .5, .8, 1.)):
            torch.testing.assert_close(calls[index][0], (1-t)*image+t*noise)
        self.assertIn("unobserved", report["rows"][1])
        self.assertEqual(report["nfe_per_prediction"], 1)
        self.assertIn("no timestep", report["interpretation"])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(model.training)
        self.assertFalse(model.net.training)
        for name, value in state.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        unknown = nn.Module()
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            diffusion_panels("unused", 0, unknown, ema, rows, device="cpu")

    def test_no_time_training_supports_fullgraph_compilation(self):
        model = self.randomize(self.model(log_noise_metrics=True))
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        image, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        torch.manual_seed(311)
        expected, expected_metrics = model(image, labels)
        torch.manual_seed(311)
        actual, actual_metrics = compiled(image, labels)
        torch.testing.assert_close(actual, expected)
        for name, value in expected_metrics.items():
            torch.testing.assert_close(actual_metrics[name], value)
        actual.backward()
        self.assertIsNotNone(model.net.final_layer.linear.linear.weight.grad)


if __name__ == "__main__":
    unittest.main()
