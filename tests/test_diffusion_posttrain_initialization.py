"""Actual CPU checkpoint handoff and differentiable fixed-time generation tests."""
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

import posttrain
import pretrain
from models import priming
from utils.ema import EMA
from utils.posttrain_checkpoint import initialize_posttraining, validate_diffusion_pretraining_contract


class DiffusionPosttrainInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'diffusion.pth'
        self.preconfig = pretrain.load_config('configs/pretrain/b10_diffusion.yaml')[0]
        self.postconfig = posttrain.load_config('configs/posttrain/b160.yaml')
        self.postconfig['model'].update(timestep_conditioning=True, generation_t=1.)
        self.options = dict(optimizer='adamw', lr=.001, beta1=.9, beta2=.95, weight_decay=0.)

    def model(self, config=None):
        # Preserve RGB256/B16 interfaces, reducing only depth/width for CPU tests.
        with patch.dict(priming.OST_models, OST_B=partial(priming.OST, depth=2, hidden_size=32, num_heads=4)):
            return pretrain.make_model(config or self.preconfig, device='cpu')

    def checkpoint(self, model=None, config=None):
        config = deepcopy(config or self.preconfig)
        model = model or self.model(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter) * .1)
        ema = EMA(model, decay=.9999)
        for parameter in ema.shadow.values():
            parameter.add_(.02)
        ema.step_count = 12510
        contract = {section: deepcopy(config[section]) for section in ('model', 'data', 'loss', 'train', 'optimizer')}
        if 'experiment' in config:
            contract['experiment'] = deepcopy(config['experiment'])
        source = dict(model=deepcopy(model.state_dict()), model_ema=ema.state_dict(),
                      config=config, contract=contract, step=12510, current_step=12511,
                      optimizer={'must_not_be_loaded': True})
        torch.save(source, self.path)
        return source

    def test_actual_handoff_keeps_online_ema_and_updates_time_embedding_at_t1(self):
        source = self.checkpoint()
        model = self.model()
        original_module = id(model.net)
        optimizer, ema = initialize_posttraining(model, self.path, self.options, config=self.postconfig)
        self.assertEqual(id(model.net), original_module)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertFalse(optimizer.state)
        self.assertEqual(ema.label, '0.999')
        self.assertEqual(ema.step_count, 12510)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, source['model'][name]), name)
            self.assertTrue(torch.equal(ema.shadow[name], source['model_ema']['shadows']['0.9999'][name]), name)
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
        self.assertEqual(optimized, {id(parameter) for parameter in model.parameters()})
        times = []
        handle = model.net.noise_t_embedder.register_forward_pre_hook(lambda _, args: times.append(args[0].detach().clone()))
        try:
            noise = torch.randn(2, 3, 256, 256)
            generated = model.sample_images_with_grad(noise, torch.tensor([0, 1]), self.postconfig['sampling'])
            # A differentiable output loss exercises the same generator/optimizer
            # path as OT without introducing encoders or a GPU-only solver.
            (generated.square().mean() + .2 * generated.mean()).backward()
        finally:
            handle.remove()
        self.assertEqual(len(times), 1)
        torch.testing.assert_close(times[0], torch.full((2,), 1000.))
        embedding = dict(model.net.noise_t_embedder.named_parameters())
        before = {name: value.detach().clone() for name, value in embedding.items()}
        for parameter in embedding.values():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        ema.step(model)
        for name, value in embedding.items():
            self.assertFalse(torch.equal(before[name], value), name)
        self.assertEqual(ema.step_count, 12511)

    def test_both_declared_auxiliary_variants_and_fixed_noise_control_are_accepted(self):
        for name in ('b10_diffusion', 'b10_diffusion_perceptual', 'b10_fixed_noise_control', 'b10_fixed_noise_control_perceptual'):
            with self.subTest(recipe=name):
                config = pretrain.load_config(f'configs/pretrain/{name}.yaml')[0]
                source = self.checkpoint(config=config)
                self.assertTrue(validate_diffusion_pretraining_contract(source))
                optimizer, _ = initialize_posttraining(self.model(config), self.path, self.options, config=self.postconfig)
                self.assertFalse(optimizer.state)

    def test_direct_initializer_rejects_flag_mismatch_and_missing_ema_parameters(self):
        source = self.checkpoint()
        self.postconfig['model']['timestep_conditioning'] = False
        with self.assertRaisesRegex(ValueError, 'timestep_conditioning differs'):
            initialize_posttraining(self.model(), self.path, self.options, config=self.postconfig)
        self.postconfig['model']['timestep_conditioning'] = True
        shadow = source['model_ema']['shadows']['0.9999']
        for name in list(shadow):
            if name.startswith('net.noise_t_embedder.'):
                del shadow[name]
        torch.save(source, self.path)
        with self.assertRaisesRegex(ValueError, 'ema timestep tensors'):
            initialize_posttraining(self.model(), self.path, self.options, config=self.postconfig)

    def test_nonperceptual_legacy_checkpoint_still_fails(self):
        config = deepcopy(self.preconfig)
        del config['experiment']
        config['model']['timestep_conditioning'] = False
        self.postconfig['model']['timestep_conditioning'] = False
        model = self.model(config)
        self.checkpoint(model=model, config=config)
        with self.assertRaisesRegex(ValueError, 'requires perceptual pretraining'):
            initialize_posttraining(model, self.path, self.options, config=self.postconfig)

    def test_legacy_perceptual_initialization_remains_supported(self):
        config = pretrain.load_config('configs/pretrain/b160_perceptual.yaml')[0]
        self.postconfig['model']['timestep_conditioning'] = False
        model = self.model(config)
        source = self.checkpoint(model=model, config=config)
        optimizer, ema = initialize_posttraining(model, self.path, self.options, config=self.postconfig)
        self.assertFalse(optimizer.state)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, source['model'][name]))
            self.assertTrue(torch.equal(ema.shadow[name], source['model_ema']['shadows']['0.9999'][name]))

    def test_profile_cannot_be_relabelled_or_relaxed(self):
        original = self.checkpoint()
        for section, key, value in (
                ('experiment', 'prediction_target', 'epsilon'),
                ('experiment', 'noise_schedule', 'cosine'),
                ('experiment', 'loss_weighting', 'snr'),
                ('experiment', 'auxiliary', 'mae'),
                ('model', 'noise_scale', 2.), ('model', 't_min', .1),
                ('model', 'generation_t', .8), ('loss', 'norm_p', 1.),
                ('train', 'epochs', 160)):
            bad = deepcopy(original)
            bad['contract'][section][key] = value
            bad['config'][section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_diffusion_pretraining_contract(bad)
        bad = deepcopy(original)
        del bad['contract']['experiment']
        with self.assertRaisesRegex(ValueError, 'experiment must agree'):
            validate_diffusion_pretraining_contract(bad)

    def test_freezing_time_embedding_is_rejected(self):
        self.checkpoint()
        model = self.model()
        model.net.noise_t_embedder.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'all trainable parameters'):
            initialize_posttraining(model, self.path, self.options, config=self.postconfig)


if __name__ == '__main__':
    unittest.main()
