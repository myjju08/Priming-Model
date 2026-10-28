"""CPU contracts for optional noise-level conditioning and one-step generation."""
from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from losses.reconstruction import reconstruction_loss
from models import Priming, priming


class NoiseConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, **overrides):
        kwargs = dict(img_size=8, patch_size=4, num_classes=3, class_tokens=2,
                      guidance_tokens=0, interval_tokens=0, label_drop_prob=0.,
                      t_dist="uniform", t_min=0., t_max=1., norm_p=1.)
        kwargs.update(overrides)
        with patch.dict(priming.OST_models,
                        OST_B=partial(priming.OST, depth=2, hidden_size=32, num_heads=4)):
            return Priming(**kwargs)

    def randomize(self, model):
        # Residual scales and the output head start at zero in the actual model.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter) * .1)
        return model

    def test_disabled_default_preserves_state_rng_and_baseline_sequence(self):
        torch.manual_seed(17)
        implicit = self.model()
        implicit_rng = torch.get_rng_state()
        torch.manual_seed(17)
        explicit = self.model(timestep_conditioning=False, generation_t=.2)
        self.assertTrue(torch.equal(implicit_rng, torch.get_rng_state()))
        for name, value in implicit.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]), name)
        self.assertFalse(any("noise_t" in name for name in implicit.state_dict()))
        x, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        expected = implicit.net.pos_embed_func(torch.cat([
            implicit.net.class_tokens + implicit.net.y_embedder(labels).unsqueeze(1),
            implicit.net.x_embedder(x)], dim=1))
        self.assertTrue(torch.equal(implicit.net._build_sequence(x, labels), expected))
        loss, metrics = implicit(x, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(metrics), {"loss_mse"})
        with self.assertRaisesRegex(ValueError, "requires timestep_conditioning"):
            implicit.net(x, labels, noise_t=.8)

    def test_matched_control_has_identical_initial_state_and_added_keys_only(self):
        torch.manual_seed(29)
        variable = self.model(timestep_conditioning=True)
        torch.manual_seed(29)
        control = self.model(timestep_conditioning=True, t_min=.8, t_max=.8)
        for name, value in variable.state_dict().items():
            self.assertTrue(torch.equal(value, control.state_dict()[name]), name)
        baseline = self.model()
        previous, current = baseline.state_dict(), variable.state_dict()
        added = set(current) - set(previous)
        self.assertEqual(added, {"net.noise_t_embedder.mlp.0.linear.weight", "net.noise_t_embedder.mlp.0.linear.bias",
                                "net.noise_t_embedder.mlp.2.linear.weight", "net.noise_t_embedder.mlp.2.linear.bias"})
        for name, value in previous.items():
            self.assertEqual(value.shape, current[name].shape, name)
        with self.assertRaises(RuntimeError):
            baseline.load_state_dict(current, strict=True)
        with self.assertRaises(RuntimeError):
            variable.load_state_dict(previous, strict=True)

    def test_noise_time_is_per_image_scaled_and_changes_only_class_tokens(self):
        model = self.model(timestep_conditioning=True)
        x, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        times = torch.tensor([.2, .8])
        seen = []
        hook = model.net.noise_t_embedder.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
        try:
            actual = model.net._build_sequence(x, labels, noise_t=times)
        finally:
            hook.remove()
        torch.testing.assert_close(seen[0], times * 1000.)
        y_embed = model.net.y_embedder(labels) + model.net.noise_t_embedder(times * 1000.)
        expected = model.net.pos_embed_func(torch.cat([
            model.net.class_tokens + y_embed.unsqueeze(1), model.net.x_embedder(x)], dim=1))
        torch.testing.assert_close(actual, expected)
        different = model.net._build_sequence(x, labels, noise_t=1. - times)
        self.assertFalse(torch.equal(actual[:, :model.net.prefix_tokens], different[:, :model.net.prefix_tokens]))
        self.assertTrue(torch.equal(actual[:, model.net.prefix_tokens:], different[:, model.net.prefix_tokens:]))
        with self.assertRaisesRegex(ValueError, "one noise fraction per image"):
            model.net(x, labels, noise_t=torch.tensor([.2, .3, .4]))

    def test_sampled_time_reaches_network_and_original_loss_is_retained(self):
        model = self.randomize(self.model(timestep_conditioning=True))
        x, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        noise, times = torch.randn_like(x), torch.tensor([.2, .9]).reshape(2, 1, 1, 1)
        calls, predictions, masks = [], [], []
        before = model.net.register_forward_pre_hook(lambda _, args, kwargs: calls.append((args, kwargs)), with_kwargs=True)
        after = model.net.register_forward_hook(lambda _, args, output: predictions.append(output))

        def auxiliary(prediction, target, mask):
            masks.append(mask)
            error = (prediction - target).square().mean((1, 2, 3))
            lpips, convnext = .4 * error * mask, .1 * error * mask
            return lpips + convnext, lpips, convnext

        try:
            with patch.object(model, "sample_t", return_value=times), \
                 patch("models.priming.torch.randn_like", return_value=noise):
                loss, metrics = model(x, labels, aux_loss_fn=auxiliary)
        finally:
            before.remove()
            after.remove()
        torch.testing.assert_close(calls[0][0][0], (1 - times) * x + times * noise)
        torch.testing.assert_close(calls[0][1]["noise_t"], times.reshape(-1))
        self.assertTrue(torch.equal(masks[0], torch.tensor([True, False])))
        expected, _ = reconstruction_loss(predictions[0], x, norm_p=1.,
            aux_loss_fn=auxiliary, aux_target=x, times=times, perceptual_threshold=.8)
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(metrics["noise_t_mean"], times.mean())
        self.assertEqual(metrics["perceptual_active_fraction"].item(), .5)
        self.assertTrue({"loss_mse", "aux_loss", "aux_loss_lpips", "aux_loss_convnext"} <= set(metrics))
        loss.backward()
        gradients = [parameter.grad for parameter in model.net.noise_t_embedder.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.)

    def test_one_step_generation_defaults_to_configured_time_and_keeps_gradients(self):
        model = self.randomize(self.model(timestep_conditioning=True, generation_t=1.))
        z, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        expected = model.net(z, labels, noise_t=torch.ones(2))
        self.assertTrue(torch.equal(model.net(z, labels), expected))
        self.assertTrue(torch.equal(model(z, labels, endpoint_generation=True), expected))
        self.assertTrue(torch.equal(model.sample_images_with_grad(z, labels), expected))
        self.assertTrue(torch.equal(model.generate(2, labels, z_t=z), expected))
        generated = model.sample_images_with_grad(z.requires_grad_(), labels)
        generated.square().mean().backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(z.grad.abs().sum().item(), 0.)
        lower = model.net(z.detach(), labels, noise_t=.4)
        self.assertTrue(torch.equal(model.sample_images_with_grad(z.detach(), labels, {"noise_t": .4}), lower))
        self.assertTrue(torch.equal(model.generate(2, labels, z_t=z.detach(), args=SimpleNamespace(noise_t=.4)), lower))
        self.assertFalse(torch.equal(lower, expected))

    def test_guidance_teacher_and_student_receive_matching_sampled_time(self):
        model = self.model(timestep_conditioning=True, guidance_tokens=2, interval_tokens=1)
        x, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        times = torch.tensor([.25, .75]).reshape(2, 1, 1, 1)
        recorded = []
        hook = model.net.register_forward_pre_hook(
            lambda _, args, kwargs: recorded.append(kwargs["noise_t"]), with_kwargs=True)
        try:
            with patch.object(model, "sample_t", return_value=times):
                model(x, labels)
        finally:
            hook.remove()
        torch.testing.assert_close(recorded[0], times.reshape(-1).repeat(2))
        torch.testing.assert_close(recorded[1], times.reshape(-1))

    def test_conditioned_forward_supports_fullgraph_compilation(self):
        model = self.randomize(self.model(timestep_conditioning=True))
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        images, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        torch.manual_seed(53)
        expected, expected_metrics = model(images, labels)
        torch.manual_seed(53)
        actual, actual_metrics = compiled(images, labels)
        torch.testing.assert_close(actual, expected)
        for name, reference in expected_metrics.items():
            torch.testing.assert_close(actual_metrics[name], reference)
        actual.backward()
        self.assertIsNotNone(model.net.noise_t_embedder.mlp[0].linear.weight.grad)

    def test_invalid_conditioning_configuration_is_rejected(self):
        for kwargs in ({"timestep_conditioning": "true"},
                       {"timestep_conditioning": True, "class_tokens": 0},
                       {"generation_t": -.1}, {"generation_t": 1.1}, {"generation_t": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs)


if __name__ == "__main__":
    unittest.main()
