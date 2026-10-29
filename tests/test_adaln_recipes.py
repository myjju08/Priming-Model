"""Check paired experiment identities and actual pretrain-to-posttrain inheritance."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import yaml

import pretrain
import posttrain
from utils.config import load_yaml
from utils.posttrain_checkpoint import initialize_posttraining

ROOT = Path(__file__).resolve().parents[1]


class AdaLNRecipeTests(unittest.TestCase):
    def setUp(self):
        self.pre = {arm: pretrain.load_config(ROOT/f'configs/pretrain/adaln_{arm}_10ep.yaml')[0]
                    for arm in ('fixed', 'uniform')}
        self.post = {arm: posttrain.load_config(ROOT/f'configs/posttrain/adaln_{arm}_post5.yaml')
                     for arm in ('fixed', 'uniform')}

    def test_paired_pretraining_changes_only_time_range_and_arm(self):
        a, b = deepcopy(self.pre['fixed']), deepcopy(self.pre['uniform'])
        self.assertEqual((a['model']['t_min'], a['model']['t_max']), (.8, .8))
        self.assertEqual((b['model']['t_min'], b['model']['t_max']), (0., 1.))
        for config in (a, b):
            config['experiment'].pop('arm')
            config['model'].pop('t_min')
            config['model'].pop('t_max')
        self.assertEqual(a, b)
        original = load_yaml(ROOT/'configs/pretrain/b160_perceptual.yaml')
        self.assertEqual(a['loss'], original['loss'])
        self.assertEqual(a['optimizer'], original['optimizer'])
        self.assertEqual(a['train']['epochs']*a['train']['steps_per_epoch'], 12510)
        self.assertEqual(8*a['train']['micro_batch_size']*a['train']['gradient_accumulation_steps'], 1024)

    def test_posttraining_changes_only_pretraining_source_and_arm(self):
        a, b = deepcopy(self.post['fixed']), deepcopy(self.post['uniform'])
        for config in (a, b):
            config['experiment'].pop('arm')
            config['initialization'].pop('checkpoint')
        self.assertEqual(a, b)
        self.assertEqual(a['train']['steps'], 12515)
        self.assertEqual(a['evaluation']['every'], 2503)
        self.assertEqual(a['evaluation']['num_images'], 50000)
        self.assertEqual(a['evaluation']['real_num_images'], 10000)
        self.assertEqual(a['matching']['stack'], 'convnext_supervised_rn50')

    def test_scientific_drift_rejected(self):
        mutations = [('model', 't_min', .1), ('model', 'corruption', 'masking'),
                     ('optimizer', 'lr', .002), ('train', 'epochs', 9), ('train', 'seed', 2)]
        for section, key, value in mutations:
            altered = deepcopy(self.pre['uniform'])
            altered[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                pretrain.validate_adaln_pretraining(altered)
        for key in ('lpips_weight', 'convnext_weight', 'enabled', 'random_crop'):
            altered = deepcopy(self.pre['fixed'])
            altered['loss']['aux'][key] = 0
            with self.subTest(loss_key=key), self.assertRaises(ValueError):
                pretrain.validate_adaln_pretraining(altered)

    def test_legacy_validation_is_not_relaxed(self):
        self.assertEqual(pretrain.load_config(ROOT/'configs/pretrain/b160_perceptual.yaml')[0]['train']['epochs'], 160)
        self.assertEqual(posttrain.load_config(ROOT/'configs/posttrain/b160.yaml')['train']['steps'], 25000)
        with TemporaryDirectory() as directory:
            altered = deepcopy(self.pre['uniform'])
            altered.pop('experiment')
            path = Path(directory)/'recipe.yaml'
            path.write_text(yaml.safe_dump(altered))
            with self.assertRaises(ValueError):
                pretrain.load_config(path)
            altered = deepcopy(self.post['fixed'])
            altered['experiment']['profile'] = 'encoder_comparison'
            path.write_text(yaml.safe_dump(altered))
            with self.assertRaises(ValueError):
                posttrain.load_config(path)

    def checkpoint(self, model, *, arm='fixed', step=2):
        config = deepcopy(self.pre[arm])
        online = {k: torch.full_like(v, .125) for k, v in model.state_dict().items()}
        shadow = {k: torch.full_like(v, .25) for k, v in model.state_dict().items()}
        return dict(model=online, model_ema=dict(schedule='const_schedule',
            schedules=[('0.9999', .9999)], step_count=step, shadows={'0.9999': shadow}),
            step=step, current_step=step+1, samples_seen=step*1024, config=config,
            contract={key: deepcopy(config[key]) for key in ('model', 'data', 'loss', 'optimizer', 'train')})

    def test_partial_smoke_and_terminal_checkpoint_inherit_online_ema_and_fresh_optimizer(self):
        for step in (2, 12510):
            model = torch.nn.Linear(3, 2)
            source = self.checkpoint(model, step=step)
            with self.subTest(step=step), TemporaryDirectory() as directory:
                path = Path(directory)/'checkpoint.pth'
                torch.save(source, path)
                optimizer, ema = initialize_posttraining(model, path, self.post['fixed']['optimizer'],
                                                         config=self.post['fixed'])
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, source['model'][key]))
                    self.assertTrue(torch.equal(ema.shadow[key], source['model_ema']['shadows']['0.9999'][key]))
                self.assertEqual(ema.step_count, step)
                self.assertEqual(ema.decay, .999)
                self.assertFalse(optimizer.state)
                self.assertEqual({id(p) for group in optimizer.param_groups for p in group['params']},
                                 {id(p) for p in model.parameters()})

    def test_mismatched_arm_architecture_and_corrupted_contract_rejected(self):
        for kind in ('arm', 'architecture', 'contract', 'loss', 'sample_counter', 'too_many_steps'):
            model = torch.nn.Linear(3, 2)
            source = self.checkpoint(model)
            if kind == 'arm':
                source = self.checkpoint(model, arm='uniform')
            elif kind == 'architecture':
                source['contract']['model']['model'] = 'OST_B'
            elif kind == 'contract':
                source['contract']['optimizer']['lr'] = .002
            elif kind == 'loss':
                source['config']['loss']['aux']['lpips_weight'] = .2
                source['contract']['loss']['aux']['lpips_weight'] = .2
            elif kind == 'sample_counter':
                source['samples_seen'] += 1
            else:
                source = self.checkpoint(model, step=12511)
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                path = Path(directory)/'checkpoint.pth'
                torch.save(source, path)
                with self.assertRaises(ValueError):
                    initialize_posttraining(model, path, self.post['fixed']['optimizer'], config=self.post['fixed'])


if __name__ == '__main__':
    unittest.main()
