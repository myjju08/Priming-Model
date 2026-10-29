"""CPU checks for EMA-only Drive evaluation without GPU or asset downloads."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from metrics import drive_evaluation as drive
from scripts import evaluate_pretrain_drive as runner
from utils.ema import EMA


class ToyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.1))
        self.calls = []

    def generate(self, count, labels, *, cfg, args, verbose):
        self.calls.append((self.weight.item(), labels.clone()))
        return self.weight + torch.randn(count, 3, 2, 2) * .1 + labels[:, None, None, None] * .001


class ToyInception(nn.Module):
    def forward(self, pixels):
        assert pixels.dtype == torch.uint8
        means = pixels.float().mean((2, 3)) / 255.
        return means[:, :2], means * 4.


class PretrainEMAEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def evaluate(self, *, weight_labels=None, seed=None, explicit=False):
        model = ToyGenerator()
        ema = EMA(model)
        ema.shadow['weight'].fill_(.5)
        with TemporaryDirectory() as directory:
            reference = Path(directory)/'reference.npz'
            np.savez(reference, ref_mu=np.zeros(2), ref_sigma=np.eye(2), num_samples=50000)
            config = dict(fid_reference=str(reference), prc_reference='unused.pt',
                          num_images=20, batch_size=5, prc_k=3, prc_batch_size=1000)
            kwargs = dict(seed=seed)
            if explicit:
                kwargs['weight_labels'] = weight_labels
            def inception_factory(*args, **kw):
                torch.rand(37)  # Encoder construction must precede the optional seed reset.
                return ToyInception()
            with patch.object(drive, 'load_inception', side_effect=inception_factory), \
                 patch.object(drive.torch, 'load', return_value=torch.empty(10000, 2048)), \
                 patch.object(drive.torch, 'autocast', side_effect=lambda *a, **k: nullcontext()), \
                 patch.object(drive, 'compute_precision_recall', return_value=(.75, .25)) as pr:
                rows = drive.evaluate(model, ema, config, dict(cfg=1., t_min=.1, t_max=1.),
                                      {'FID(ADM)': 'unused.pth'}, **kwargs)
                for call in pr.call_args_list:
                    self.assertEqual(call.args[0].shape, (10000, 2048))
                    self.assertEqual(call.kwargs, dict(k=3, batch_size=1000))
        self.assertAlmostEqual(model.weight.item(), .1)
        return rows, model.calls

    def test_default_online_then_ema_is_exactly_unchanged_by_explicit_selection(self):
        torch.manual_seed(401)
        implicit, implicit_calls = self.evaluate()
        torch.manual_seed(401)
        explicit, explicit_calls = self.evaluate(explicit=True, weight_labels=('online', '0.9999'))
        self.assertEqual(implicit, explicit)
        self.assertEqual([row['ema_label'] for row in implicit], ['online', '0.9999'])
        self.assertEqual(len(implicit_calls), 8)
        for (weight_a, labels_a), (weight_b, labels_b) in zip(implicit_calls, explicit_calls):
            self.assertEqual(weight_a, weight_b)
            torch.testing.assert_close(labels_a, labels_b)

    def test_only_selected_ema_generates_and_seed_resets_after_encoder_setup(self):
        torch.manual_seed(403)
        first, calls = self.evaluate(explicit=True, weight_labels=('0.9999',), seed=1)
        torch.manual_seed(999)
        second, _ = self.evaluate(explicit=True, weight_labels=('0.9999',), seed=1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]['ema_label'], '0.9999')
        self.assertEqual(first[0]['num_samples'], 20)
        self.assertEqual(first[0]['real_num_samples'], 10000)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(weight == .5 for weight, _ in calls))
        torch.testing.assert_close(torch.cat([labels for _, labels in calls]), torch.arange(20))

    def test_invalid_weight_selections_fail_before_generation(self):
        self.assertEqual(drive.select_weight_labels('0.9999'), ('online', '0.9999'))
        for selection in ((), ('0.999',), ('online', 'online'), '0.9999', ('unknown',)):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                drive.select_weight_labels('0.9999', selection)

    def test_load_ema_preserves_online_tensors_without_aliasing_and_freezes_afterward(self):
        source = ToyGenerator()
        source_ema = EMA(source)
        source_ema.shadow['weight'].fill_(.7)
        state = dict(model=deepcopy(source.state_dict()), model_ema=source_ema.state_dict())
        target = ToyGenerator()
        ema = runner.load_online_and_ema(target, state)
        self.assertFalse(target.weight.requires_grad)
        self.assertNotEqual(ema.shadow['weight'].data_ptr(), target.weight.data_ptr())
        torch.testing.assert_close(target.weight, state['model']['weight'])
        with ema.swap(target):
            torch.testing.assert_close(target.weight, source_ema.shadow['weight'])
        torch.testing.assert_close(target.weight, state['model']['weight'])
        with self.assertRaisesRegex(ValueError, 'before freezing'):
            runner.load_online_and_ema(target, state)

    def fixture(self):
        from pretrain import load_config
        config = load_config('configs/pretrain/b10_diffusion_no_time_perceptual.yaml')[0]
        return dict(config=config, contract=dict(version=2, world_size=8, global_batch=1024,
                    **{key: deepcopy(config[key]) for key in ('experiment', 'model', 'data', 'loss', 'train', 'optimizer')}),
                    model={}, world_size=8, step=12510, current_step=12511, samples_seen=12510*1024,
                    model_ema=dict(shadows={'0.9999': {}}, step_count=12510, schedule='const_schedule',
                                   batch_size=1024, schedules=[('0.9999', .9999)]))

    def test_checkpoint_metadata_rejects_partial_or_wrong_ema(self):
        source = self.fixture()
        self.assertEqual(runner.validate_checkpoint(source), source['config'])
        for key, value in (('step', 12509), ('current_step', 12510), ('samples_seen', 0), ('world_size', 1)):
            invalid = deepcopy(source)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_checkpoint(invalid)
        for key, value in (('step_count', 0), ('schedules', [('0.999', .999)]),
                           ('shadows', {'0.999': {}})):
            invalid = deepcopy(source)
            invalid['model_ema'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_checkpoint(invalid)

    def test_fixed_evaluation_protocol_is_enforced(self):
        config = dict(model=dict(timestep_conditioning=False), sampling=dict(cfg=1., t_min=.1, t_max=1., num_steps=1),
                      evaluation=dict(protocol='drive_fid_is_pr', num_images=50000, batch_size=32,
                                      real_num_images=10000, prc_k=3, prc_batch_size=1000))
        runner.validate_evaluation(config)
        for key, value in (('num_images', 1000), ('real_num_images', 5000), ('prc_k', 5), ('batch_size', 64)):
            invalid = deepcopy(config)
            invalid['evaluation'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_evaluation(invalid)

    def test_only_complete_finite_ema_metrics_can_be_published(self):
        row = dict(ema_label='0.9999', num_samples=50000, real_num_samples=10000,
                   fid=50., **{'is': 2.}, is_std=.1, precision=.1, recall=.2)
        runner.validate_results([row])
        for key, value in (('ema_label', 'online'), ('num_samples', 1000), ('fid', float('nan')),
                           ('is', float('inf')), ('precision', 1.1), ('recall', -.1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_results([dict(row, **{key: value})])


if __name__ == '__main__':
    unittest.main()
