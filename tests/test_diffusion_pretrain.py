"""CPU checks for objective isolation, matched recipes, and validation diagnostics."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

import pretrain
from losses.reconstruction import reconstruction_loss
from utils.diffusion_diagnostics import diffusion_panels
from utils.ema import EMA


class DiffusionPretrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, name="b10_diffusion"):
        return pretrain.load_config(f"configs/pretrain/{name}.yaml")[0]

    def validate(self, config):
        with patch("utils.config.load_yaml", return_value=config):
            return pretrain.load_config("unused.yaml")

    def test_matched_control_differs_only_in_noise_interval_and_variant(self):
        diffusion, control = self.config(), self.config("b10_fixed_noise_control")
        self.assertEqual(diffusion["train"]["epochs"]*diffusion["train"]["steps_per_epoch"], 12510)
        self.assertEqual(diffusion["loss"]["norm_p"], 0)
        self.assertFalse(diffusion["loss"]["aux"]["enabled"])
        self.assertEqual(diffusion["model"]["generation_t"], 1.)
        control["experiment"]["variant"] = "variable_noise"
        control["model"].update(t_min=0., t_max=1.)
        self.assertEqual(diffusion, control)

    def test_rejects_mislabelled_or_unimplemented_objectives(self):
        base = self.config()
        for section, key, value in (
            ("experiment", "prediction_target", "epsilon"),
            ("experiment", "loss_weighting", "snr"),
            ("experiment", "noise_schedule", "cosine"),
            ("experiment", "auxiliary", "mae"),
            ("model", "t_min", .8), ("model", "generation_t", .8),
            ("model", "timestep_conditioning", False),
            ("loss", "norm_p", 1.), ("train", "epochs", 160),
        ):
            config = deepcopy(base)
            config[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                self.validate(config)

    def test_optional_perceptual_loss_must_cover_all_noise_levels(self):
        config = self.config()
        config["experiment"]["auxiliary"] = "perceptual"
        config["loss"]["aux"]["enabled"] = True
        self.validate(config)
        config["loss"]["perceptual_threshold"] = .8
        with self.assertRaisesRegex(ValueError, "every sampled noise level"):
            self.validate(config)

    def test_perceptual_memory_fallback_keeps_global_batch_and_matched_control(self):
        diffusion = self.config("b10_diffusion_perceptual")
        control = self.config("b10_fixed_noise_control_perceptual")
        self.assertEqual((diffusion["train"]["micro_batch_size"],
                          diffusion["train"]["gradient_accumulation_steps"]), (32, 4))
        self.assertEqual(8 * 32 * 4, 1024)
        control["experiment"]["variant"] = "variable_noise"
        control["model"].update(t_min=0., t_max=1.)
        self.assertEqual(diffusion, control)
        for name, micro, accumulation in (("b10_diffusion", 32, 4),
                ("b160_perceptual", 32, 4), ("b10_diffusion_perceptual", 16, 8),
                ("b10_diffusion_perceptual", 32, 2)):
            config = self.config(name)
            config["train"].update(micro_batch_size=micro, gradient_accumulation_steps=accumulation)
            with self.subTest(name=name, micro=micro, accumulation=accumulation), self.assertRaises(ValueError):
                self.validate(config)
        diffusion["train"].update(micro_batch_size=64, gradient_accumulation_steps=2)
        self.validate(diffusion)
        diffusion["train"]["dtype"] = "fp32"
        with self.assertRaises(ValueError):
            self.validate(diffusion)

    def test_profiling_and_warmup_batches_use_recipe_shapes(self):
        for name, micro, accumulation in (("b10_diffusion", 64, 2),
                                         ("b10_diffusion_perceptual", 32, 4)):
            config = self.config(name)
            config["data"]["img_size"] = 4
            for include_indices in (False, True):
                batches = pretrain.dummy_batches(config, device="cpu", include_indices=include_indices)
                self.assertEqual(len(batches), accumulation)
                for batch in batches:
                    self.assertEqual(batch[0].shape, (micro, 3, 4, 4))
                    self.assertEqual(batch[1].shape, (micro,))
                    self.assertEqual(len(batch), 3 if include_indices else 2)
                    if include_indices:
                        self.assertEqual(batch[2].shape, (micro, 2))

    def test_accumulation_four_has_same_average_loss_and_gradient_as_two(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(.7, dtype=torch.float64))

            def forward(self, images, labels, aux_loss_fn=None):
                loss = (self.scale * images - labels[:, None, None, None]).square().mean()
                return loss, {"loss_mse": loss}

        images = torch.linspace(-1., 1., 128 * 12, dtype=torch.float64).reshape(128, 3, 2, 2)
        labels = (torch.arange(128) % 3).to(torch.float64)
        reference = ToyModel()
        expected, _ = reference(images, labels)
        expected.backward()
        for micro, accumulation in ((64, 2), (32, 4)):
            config = self.config()
            config["train"].update(micro_batch_size=micro, gradient_accumulation_steps=accumulation)
            model = ToyModel()
            batches = list(zip(images.split(micro), labels.split(micro)))
            actual, metrics = pretrain.train_step(config, model, batches, None, 1,
                                                  family="b", rank=0, device="cpu")
            torch.testing.assert_close(actual.double(), expected.detach(), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(metrics["loss_mse"], expected.detach())
            torch.testing.assert_close(model.scale.grad, reference.scale.grad, atol=1e-12, rtol=1e-12)

    def test_plain_mse_values_gradients_and_no_perceptual_construction(self):
        config = self.config()
        self.assertIsNone(pretrain.build_auxiliary(config, "b", "does-not-exist", device="cpu"))
        prediction, target = torch.randn(3, 3, 4, 4, requires_grad=True), torch.randn(3, 3, 4, 4)
        loss, metrics = reconstruction_loss(prediction, target, norm_p=config["loss"]["norm_p"])
        expected = (prediction-target).square().mean()
        self.assertTrue(torch.equal(loss, metrics["loss_mse"]))
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(torch.autograd.grad(loss, prediction, retain_graph=True)[0],
                                   torch.autograd.grad(expected, prediction)[0])

    def test_original_recipes_still_use_fixed_noise_and_original_loss(self):
        for name in ("b160_perceptual", "b160_no_perceptual", "l160_perceptual"):
            config = self.config(name)
            self.assertEqual(config["train"]["epochs"], 160)
            self.assertEqual((config["model"]["t_min"], config["model"]["t_max"]), (.8, .8))
            self.assertEqual(config["loss"]["norm_p"], 1.)
            self.assertFalse(config["model"].get("timestep_conditioning", False))

    def test_full_pretraining_and_posttraining_parameter_schemas_match(self):
        import posttrain
        config = posttrain.load_config("configs/posttrain/b160.yaml")
        config["model"].update(timestep_conditioning=True, generation_t=1.)
        with torch.device("meta"):
            before = pretrain.make_model(self.config(), device="meta")
            after = posttrain.make_model(config, device="meta")
        self.assertEqual({k: v.shape for k, v in before.state_dict().items()},
                         {k: v.shape for k, v in after.state_dict().items()})
        self.assertEqual(sum(p.numel() for p in before.parameters()), 115851904)
        self.assertTrue(all(p.requires_grad for p in after.parameters()))

    def test_panels_use_correct_times_preserve_rng_modes_and_weights(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(.5))
                self.calls = []

            def forward(self, x, labels, noise_t):
                self.calls.append((float(noise_t[0]), x.detach().clone()))
                return x*self.scale + noise_t[:, None, None, None]

        model = nn.Module()
        model.net = Net()
        ema = EMA(model, decay=.9999, batch_size=1024)
        model.train()
        model.net.eval()
        rows = [(torch.ones(3, 4, 4)*.25, 0), (torch.ones(3, 4, 4)*-.25, 1)]
        rng = torch.get_rng_state().clone()
        weights = deepcopy(model.state_dict())
        with TemporaryDirectory() as directory:
            report, path = diffusion_panels(directory, 0, model, ema, rows, device="cpu")
            self.assertTrue(path.is_file())
            self.assertTrue((Path(directory)/"visualization/step_0000000.json").is_file())
        torch.testing.assert_close(torch.tensor([t for t, _ in model.net.calls]),
                                   torch.tensor([.2, .5, .8, 1., 1.]*2))
        self.assertEqual(report["nfe_per_prediction"], 1)
        images = torch.stack([x for x, _ in rows])
        pure_noise = model.net.calls[3][1]
        torch.testing.assert_close(model.net.calls[0][1], .8*images+.2*pure_noise)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(model.training)
        self.assertFalse(model.net.training)
        for key, value in weights.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
