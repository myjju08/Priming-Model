"""Validate the requested population/schedule and full evaluation counts."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import yaml

from data.imagenet import ClasswiseOTStream
from posttrain import load_config
from utils.evaluation import validate_rows


class CustomPosttrainTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).resolve().parents[1]/'configs/posttrain/b160_mae640_32_40epochs.yaml'
        self.config = load_config(self.path)

    def test_population_and_epoch_schedule(self):
        config = self.config
        self.assertEqual(config['train']['steps'], 40*2503)
        self.assertEqual(config['evaluation']['every'], 5*2503)
        total = 0
        for rank in range(8):
            stream = ClasswiseOTStream(class_counts=[1000]*1000, global_classes_per_step=16,
                samples_per_class=32, real_references_per_class=32,
                rank=rank, world_size=8, seed=7919, device='cpu')
            batch = stream.next_batch()
            self.assertEqual(len(batch.generated_labels), 64)
            self.assertEqual(batch.target_indices.numel(), 64)
            total += len(batch.generated_labels)
        self.assertEqual(total, config['train']['global_batch_size'])

    def test_modified_production_recipe_still_rejected(self):
        config = deepcopy(self.config)
        config.pop('experiment')
        with TemporaryDirectory() as directory:
            path = Path(directory)/'bad.yaml'
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_full_evaluation_counts_required(self):
        rows = [dict(ema_label=label, num_samples=50000, real_num_samples=10000,
                cfg=1., fid=2., **{'is':200.}, precision=.7, recall=.6)
                for label in ('online','0.999')]
        validate_rows(rows, self.config)
        for key, value in [('num_samples', 49999), ('real_num_samples', 9999), ('fid',float('nan'))]:
            altered = deepcopy(rows)
            altered[0][key] = value
            with self.assertRaises((ValueError, FloatingPointError)):
                validate_rows(altered, self.config)

    def test_original_production_config_unchanged(self):
        config = load_config(self.path.parent/'b160.yaml')
        self.assertEqual(config['matching']['samples_per_class'],64)
        self.assertEqual(config['train']['steps'],25000)


if __name__ == '__main__':
    unittest.main()
