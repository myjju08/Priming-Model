"""Strict objective labels and fresh post-training handoff for adaptive pixel MSE."""
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

import pretrain
from models import priming
from scripts.run_diffusion_pretrain import post_command
from utils.ema import EMA
from utils.experiments import recipe
from utils.posttrain_checkpoint import initialize_posttraining, validate_diffusion_pretraining_contract


class NoTimeAdaptiveContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'adaptive.pth'
        self.config = self.load('b10_diffusion_no_time_adaptive_perceptual')
        self.postconfig = recipe('supervised',self.directory.name,evaluation='drive')
        self.postconfig['model'].update(timestep_conditioning=False,generation_t=1.)

    def load(self,name):
        return pretrain.load_config(f'configs/pretrain/{name}.yaml')[0]

    def validate(self,config):
        with patch('utils.config.load_yaml',return_value=config):
            return pretrain.load_config('unused.yaml')

    def tiny_model(self,config):
        with patch.dict(priming.OST_models,OST_B=partial(priming.OST,depth=2,hidden_size=32,num_heads=4)):
            return pretrain.make_model(config,device='cpu')

    def checkpoint(self,config=None):
        config = deepcopy(config or self.config)
        model = self.tiny_model(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter)*.1)
        ema = EMA(model,decay=.9999)
        for value in ema.shadow.values():
            value.add_(.02)
        ema.step_count = 12510
        contract = {key:deepcopy(config[key]) for key in ('model','data','loss','train','optimizer','experiment')}
        state = dict(model=deepcopy(model.state_dict()),model_ema=ema.state_dict(),config=config,
                     contract=contract,step=12510,current_step=12511,optimizer={'must_not_load':True})
        torch.save(state,self.path)
        return state

    def test_adaptive_recipe_differs_from_plain_recipe_only_in_pixel_weighting(self):
        plain = self.load('b10_diffusion_no_time_perceptual')
        changed = deepcopy(self.config)
        self.assertEqual(changed['experiment']['loss_weighting'],'adaptive_residual')
        self.assertEqual((changed['loss']['norm_p'],changed['loss']['norm_eps']),(1.,1e-4))
        changed['experiment']['loss_weighting'] = 'uniform'
        changed['loss']['norm_p'] = 0.
        self.assertEqual(changed,plain)

    def test_old_uniform_recipes_remain_strictly_plain_mse(self):
        for name in ('b10_diffusion','b10_diffusion_perceptual','b10_diffusion_no_time_perceptual'):
            config = self.load(name)
            self.assertEqual((config['experiment']['loss_weighting'],config['loss']['norm_p']),('uniform',0.))
            config['loss']['norm_p'] = 1.
            with self.subTest(recipe=name),self.assertRaisesRegex(ValueError,'match declared loss_weighting'):
                self.validate(config)

    def test_adaptive_label_cannot_hide_changed_pixel_or_auxiliary_losses(self):
        edits = [('loss','norm_p',0.),('loss','norm_p',.5),('loss','norm_p',True),
                 ('loss','norm_eps',.01),('loss','perceptual_threshold',.8),
                 ('experiment','loss_weighting','uniform'),('experiment','loss_weighting','snr')]
        for section,key,value in edits:
            invalid = deepcopy(self.config)
            invalid[section][key] = value
            with self.subTest(section=section,key=key,value=value),self.assertRaises(ValueError):
                self.validate(invalid)
        for key,value in (('lpips_weight',0.),('convnext_weight',0.),('norm_p',0.),('norm_eps',1e-4)):
            invalid = deepcopy(self.config)
            invalid['loss']['aux'][key] = value
            with self.subTest(auxiliary=key,value=value),self.assertRaises(ValueError):
                self.validate(invalid)
        invalid = deepcopy(self.config)
        invalid['experiment']['auxiliary'] = 'none'
        invalid['loss']['aux']['enabled'] = False
        with self.assertRaisesRegex(ValueError,'retains the original perceptual'):
            self.validate(invalid)
        invalid = deepcopy(self.config)
        invalid['experiment']['conditioning'] = 'timestep'
        invalid['model']['timestep_conditioning'] = True
        with self.assertRaisesRegex(ValueError,'explicitly unconditioned'):
            self.validate(invalid)

    def test_saved_contract_must_agree_with_adaptive_weighting_and_config(self):
        source = self.checkpoint()
        self.assertTrue(validate_diffusion_pretraining_contract(source))
        for weighting,power in (('uniform',1.),('adaptive_residual',0.),('adaptive_residual',.5)):
            invalid = deepcopy(source)
            for section in ('contract','config'):
                invalid[section]['experiment']['loss_weighting'] = weighting
                invalid[section]['loss']['norm_p'] = power
            with self.subTest(weighting=weighting,power=power),self.assertRaisesRegex(ValueError,'match declared loss_weighting'):
                validate_diffusion_pretraining_contract(invalid)
        invalid = deepcopy(source)
        invalid['config']['experiment']['loss_weighting'] = 'uniform'
        with self.assertRaisesRegex(ValueError,'experiment must agree'):
            validate_diffusion_pretraining_contract(invalid)
        invalid = deepcopy(source)
        for section in ('contract','config'):
            invalid[section]['loss']['norm_eps'] = .01
        with self.assertRaisesRegex(ValueError,'norm_eps=1e-4'):
            validate_diffusion_pretraining_contract(invalid)

    def test_random_and_fixed_adaptive_checkpoints_start_fresh_posttraining(self):
        for name in ('b10_diffusion_no_time_adaptive_perceptual','b10_fixed_noise_no_time_adaptive_perceptual'):
            with self.subTest(recipe=name):
                config = self.load(name)
                source = self.checkpoint(config)
                model = self.tiny_model(config)
                optimizer,ema = initialize_posttraining(model,self.path,self.postconfig['optimizer'],config=self.postconfig)
                self.assertIsInstance(optimizer,torch.optim.AdamW)
                self.assertFalse(optimizer.state)
                self.assertEqual((ema.step_count,ema.label),(12510,'0.999'))
                for key,value in model.state_dict().items():
                    self.assertTrue(torch.equal(value,source['model'][key]),key)
                    self.assertTrue(torch.equal(ema.shadow[key],source['model_ema']['shadows']['0.9999'][key]),key)
                optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
                self.assertEqual(optimized,{id(p) for p in model.parameters()})
                plan = dict(python='python',output='/tmp/adaptive-pretrain',hf_home='/tmp/hf',torch_home='/tmp/torch',
                            posttraining=dict(output='/tmp/post',smoke_output='/tmp/post-smoke'))
                command = post_command(plan,'/tmp/post.yaml')
                self.assertIn('--init-checkpoint',command)
                self.assertNotIn('--resume',command)


if __name__ == '__main__':
    unittest.main()
