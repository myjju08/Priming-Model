"""Portable recipe schedules, encoder selection, resume identity and calibration."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml

from scripts.experiment import configuration
from scripts.prepare_feature_encoder import median_distance, resolve_checkpoint
from utils.experiments import recipe, validate_schedule
from posttrain import load_config


class EncoderExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_variants_validate_without_assets(self):
        for encoder in ('mae640','supcon','spark','supervised'):
            for epochs in (5,10,40):
                with self.subTest(encoder=encoder, epochs=epochs), TemporaryDirectory() as root:
                    c = recipe(encoder, root, epochs=epochs)
                    p = Path(root)/'config.yaml'
                    p.write_text(yaml.safe_dump(c))
                    loaded = load_config(p)
                    self.assertEqual(loaded['train']['steps'], epochs*2503)
                    self.assertEqual(loaded['evaluation']['every'], 12515)
                    self.assertEqual(loaded['matching']['samples_per_class'],32)
                    self.assertEqual(loaded['matching']['real_references_per_class'],32)
                    self.assertNotIn('tracking', loaded)
                    self.assertEqual(loaded['matching']['mae_bf16_channels_last'],encoder=='mae640')

    def test_only_encoder_settings_differ(self):
        a, b = recipe('mae640','/tmp/storage'), recipe('supcon','/tmp/storage')
        for c in (a,b):
            c['matching'].pop('stack'); c['matching'].pop('mae_bf16_channels_last')
            c['data'].pop('encoder_cache'); c['experiment'].pop('encoder')
        self.assertEqual(a,b)

    def test_supervised_preserves_rn50_recipes_for_both_evaluations(self):
        for evaluation in ('native', 'drive'):
            reference = None
            for encoder in ('supcon', 'spark', 'supervised'):
                config = recipe(encoder, '/tmp/storage', evaluation=evaluation)
                config['matching'].pop('stack')
                config['data'].pop('encoder_cache')
                config['experiment'].pop('encoder')
                if reference is None:
                    reference = config
                self.assertEqual(config, reference)

    def test_supervised_static_recipe_preserves_existing_control(self):
        reference = load_config('configs/posttrain/b160_spark_32_40epochs.yaml')
        actual = load_config('configs/posttrain/b160_supervised_32_40epochs.yaml')
        for config in (reference, actual):
            config['matching'].pop('stack')
            config['data'].pop('encoder_cache')
            config['experiment'].pop('profile')
        self.assertEqual(actual, reference)

    def test_invalid_schedules_rejected(self):
        for epochs, interval in ((0,5),(4,5),(40,0)):
            with self.assertRaises(ValueError):
                recipe('spark','/tmp/storage', epochs=epochs,eval_every=interval)
        c = recipe('spark','/tmp/storage')
        c['train']['save_every'] += 1
        with self.assertRaises(ValueError): validate_schedule(c)

    def test_resume_preserves_run_identity_and_rejects_encoder_change(self):
        with TemporaryDirectory() as root:
            config = recipe('supcon', root, epochs=10, evaluation='drive')
            config['tracking'] = dict(project='test', name='original', id='stable')
            (Path(root)/'run.json').write_text(json.dumps(dict(config=config)))
            args = argparse.Namespace(resume=True, output=Path(root), encoder='supcon')
            self.assertEqual(configuration(args),config)
            args.encoder = 'spark'
            with self.assertRaises(ValueError): configuration(args)

    def test_dry_run_does_not_create_output_or_require_cuda(self):
        with TemporaryDirectory() as root:
            output = Path(root)/'run'
            result = subprocess.run([sys.executable,'scripts/experiment.py','train','--encoder','supervised',
                '--storage',root,'--output',str(output),'--dry-run'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--nproc_per_node=8',result.stdout)
            self.assertIn('convnext_supervised_rn50',result.stdout)
            self.assertFalse(output.exists())
            self.assertFalse((Path(root)/'recipes').exists())

    def test_calibration_counts_each_unordered_pair_once(self):
        features = np.array([[0.,0.],[3.,0.],[0.,4.],[3.,4.],[8.,0.]],dtype=np.float32)
        actual = median_distance(features,'cpu',block=2)
        distances = torch.pdist(torch.from_numpy(features))
        self.assertAlmostEqual(actual,torch.quantile(distances,.5).item(),places=5)
        with self.assertRaises(ValueError): median_distance(features[:1],'cpu')
        with self.assertRaises(ValueError): median_distance(np.zeros((3,2),dtype=np.float32),'cpu')

    def test_backbone_loading_is_strict_and_accepts_classifier(self):
        from torchvision.models import resnet50
        from models.encoders.rn50 import load_backbone
        state = resnet50(weights=None).state_dict()
        with patch('models.encoders.rn50.torch.load', return_value={'state_dict':{'module.'+k:v for k,v in state.items()}}):
            model = load_backbone('unused')
            self.assertIsInstance(model.fc,torch.nn.Identity)
        del state['layer1.0.conv1.weight']
        with patch('models.encoders.rn50.torch.load', return_value=state):
            with self.assertRaises(RuntimeError): load_backbone('unused')

    def test_supervised_download_is_pinned_and_corruption_is_rejected(self):
        from models.encoders.rn50 import SUPERVISED_URL, SUPERVISED_SHA256
        with TemporaryDirectory() as root:
            with patch('torch.hub.download_url_to_file') as download, \
                 patch('utils.files.sha256', return_value=SUPERVISED_SHA256):
                path, digest = resolve_checkpoint(root, 'supervised')
            download.assert_called_once_with(SUPERVISED_URL, str(path),
                                              hash_prefix=SUPERVISED_SHA256, progress=True)
            self.assertEqual(digest, SUPERVISED_SHA256)
            path.write_bytes(b'corrupt checkpoint')
            with patch('torch.hub.download_url_to_file') as download:
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    resolve_checkpoint(root, 'supervised')
                download.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                resolve_checkpoint(root, 'supervised', checkpoint=path)

    def test_supervised_features_match_rn50_extraction_and_keep_input_gradients(self):
        from torchvision.models import resnet50
        from models.encoders.rn50 import RN50ConvNext
        encoder = RN50ConvNext.__new__(RN50ConvNext)
        torch.nn.Module.__init__(encoder)
        encoder.backbone = resnet50(weights=None).eval().requires_grad_(False)
        image = torch.rand(1, 3, 64, 64, requires_grad=True)
        encoder.family = 'supervised'
        supervised = encoder.encode_family(image, 'supervised')
        self.assertEqual(len(supervised), 12)
        self.assertTrue(all(value.dtype == torch.float32 for value in supervised.values()))
        for family in ('supcon', 'spark'):
            encoder.family = family
            with torch.no_grad():
                rows = encoder.encode_family(image, family)
            for key, value in supervised.items():
                torch.testing.assert_close(value, rows[key.replace('supervised__', family+'__')], rtol=0, atol=0)
        sum(value.square().mean() for value in supervised.values()).backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))


if __name__ == '__main__': unittest.main()
