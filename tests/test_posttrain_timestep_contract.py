"""CPU checks that a timestep-conditioned pretrain cannot silently change models."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

import posttrain


class PosttrainTimestepContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'source.pth'
        self.config = dict(model=dict(timestep_conditioning=False, generation_t=1.))

    def checkpoint(self, *, conditioned=False, weights=None, generation_t=1., legacy=False):
        model = {} if legacy else dict(timestep_conditioning=conditioned, generation_t=generation_t)
        parameters = {'net.base.weight': torch.ones(2, 2)}
        has_weights = conditioned if weights is None else weights
        if has_weights:
            parameters['net.noise_t_embedder.mlp.0.weight'] = torch.ones(2, 2)
        source = dict(contract=dict(model=model), model=parameters,
                      model_ema=dict(shadows={'0.9999': deepcopy(parameters)}))
        torch.save(source, self.path)
        return source

    def test_legacy_and_matching_time_conditioned_contracts_are_accepted(self):
        self.checkpoint(legacy=True)
        posttrain.validate_pretraining_timestep_contract(self.config, self.path)
        self.config['model']['timestep_conditioning'] = True
        self.checkpoint(conditioned=True)
        posttrain.validate_pretraining_timestep_contract(self.config, self.path)

    def test_architecture_mismatch_is_rejected_in_both_directions(self):
        self.checkpoint(conditioned=True)
        with self.assertRaisesRegex(ValueError, 'timestep_conditioning differs'):
            posttrain.validate_pretraining_timestep_contract(self.config, self.path)
        self.config['model']['timestep_conditioning'] = True
        self.checkpoint(legacy=True)
        with self.assertRaisesRegex(ValueError, 'timestep_conditioning differs'):
            posttrain.validate_pretraining_timestep_contract(self.config, self.path)

    def test_missing_or_undeclared_timestep_weights_are_rejected(self):
        for conditioned in (False, True):
            with self.subTest(conditioned=conditioned):
                self.config['model']['timestep_conditioning'] = conditioned
                self.checkpoint(conditioned=conditioned, weights=not conditioned)
                with self.assertRaisesRegex(ValueError, 'tensors disagree'):
                    posttrain.validate_pretraining_timestep_contract(self.config, self.path)

    def test_ema_cannot_drop_timestep_parameters(self):
        self.config['model']['timestep_conditioning'] = True
        source = self.checkpoint(conditioned=True)
        del source['model_ema']['shadows']['0.9999']['net.noise_t_embedder.mlp.0.weight']
        torch.save(source, self.path)
        with self.assertRaisesRegex(ValueError, 'ema timestep tensors'):
            posttrain.validate_pretraining_timestep_contract(self.config, self.path)

    def test_generation_t_must_match_pure_noise_input(self):
        self.config['model']['timestep_conditioning'] = True
        self.checkpoint(conditioned=True, generation_t=.8)
        with self.assertRaisesRegex(ValueError, 'generation_t=1'):
            posttrain.validate_pretraining_timestep_contract(self.config, self.path)

    def test_config_accepts_explicit_conditioning_but_rejects_wrong_endpoint(self):
        original = posttrain.load_config('configs/posttrain/b160.yaml')
        configured = deepcopy(original)
        configured['model'].update(timestep_conditioning=True, generation_t=1.)
        with patch('utils.config.load_yaml', return_value=configured):
            self.assertTrue(posttrain.load_config('unused.yaml')['model']['timestep_conditioning'])
        for key, value in (('timestep_conditioning', 'true'), ('generation_t', .8), ('generation_t', True)):
            invalid = deepcopy(configured)
            invalid['model'][key] = value
            with self.subTest(key=key, value=value), patch('utils.config.load_yaml', return_value=invalid), self.assertRaises(ValueError):
                posttrain.load_config('unused.yaml')


if __name__ == '__main__':
    unittest.main()
