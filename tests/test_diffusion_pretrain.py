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
