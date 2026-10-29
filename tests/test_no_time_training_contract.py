"""CPU checks for explicitly time-unconditioned diffusion pretraining/handoff."""
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

import pretrain
import posttrain
from models import priming
from scripts.run_diffusion_pretrain import resolved_post_configs
from utils.ema import EMA
from utils.experiments import recipe
from utils.posttrain_checkpoint import (initialize_posttraining, validate_timestep_contract,
                                        validate_diffusion_pretraining_contract)


class NoTimeTrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'no-time.pth'
        self.preconfig = pretrain.load_config('configs/pretrain/b10_diffusion_no_time_perceptual.yaml')[0]
        self.postconfig = recipe('supervised', self.directory.name, evaluation='drive')
        self.postconfig['model'].update(timestep_conditioning=False, generation_t=1.)

    def tiny_model(self):
        with patch.dict(priming.OST_models, OST_B=partial(priming.OST,depth=2,hidden_size=32,num_heads=4)):
            return pretrain.make_model(self.preconfig,device='cpu')

    def checkpoint(self):
        model = self.tiny_model()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter)*.1)
        ema = EMA(model,decay=.9999)
        for value in ema.shadow.values():
            value.add_(.02)
        ema.step_count = 12510
        config = deepcopy(self.preconfig)
        contract = {key:deepcopy(config[key]) for key in ('model','data','loss','train','optimizer','experiment')}
        state = dict(model=deepcopy(model.state_dict()),model_ema=ema.state_dict(),config=config,
                     contract=contract,step=12510,current_step=12511,optimizer={'do_not_load':True})
        torch.save(state,self.path)
        return state

    def test_recipe_uses_explicit_no_time_metadata_and_original_loss_components(self):
        config = self.preconfig
        self.assertEqual(config['experiment']['conditioning'],'none')
        self.assertIs(config['model']['timestep_conditioning'],False)
        self.assertTrue(config['model']['log_noise_metrics'])
        self.assertEqual((config['model']['t_min'],config['model']['t_max']),(0.,1.))
        self.assertEqual(config['model']['class_tokens'],8)
        self.assertEqual((config['loss']['norm_p'],config['loss']['norm_eps']),(0.,1e-4))
        self.assertEqual((config['loss']['perceptual_threshold'],config['loss']['aux']['lpips_weight'],
                          config['loss']['aux']['convnext_weight'],config['loss']['aux']['norm_p']), (1.,.4,.1,1.))
        train = config['train']
        self.assertEqual((train['epochs']*train['steps_per_epoch'],8*train['micro_batch_size']*train['gradient_accumulation_steps']),
                         (12510,1024))

    def test_existing_time_conditioned_profiles_cannot_silently_disable_time(self):
        legacy = pretrain.load_config('configs/pretrain/b10_diffusion.yaml')[0]
        self.assertNotIn('conditioning',legacy['experiment'])
        legacy['model']['timestep_conditioning'] = False
        with patch('utils.config.load_yaml',return_value=legacy),self.assertRaises(ValueError):
            pretrain.load_config('unused.yaml')
        for section,key,value in (('model','timestep_conditioning',True),('experiment','conditioning','unknown'),
                                  ('model','log_noise_metrics','true'),('model','conditioning_mode','adaln_zero'),
                                  ('model','conditioning_mode','prefix')):
            invalid = deepcopy(self.preconfig)
            invalid[section][key] = value
            with self.subTest(section=section,key=key,value=value),patch('utils.config.load_yaml',return_value=invalid),self.assertRaises(ValueError):
                pretrain.load_config('unused.yaml')
        for mode in ('adaln_zero','prefix'):
            invalid = deepcopy(self.postconfig)
            invalid['model']['conditioning_mode'] = mode
            with patch('utils.config.load_yaml',return_value=invalid),self.assertRaisesRegex(ValueError,'conditioning_mode is unsupported'):
                posttrain.load_config('unused.yaml')

    def test_meta_schema_is_exact_original_priming_in_both_training_phases(self):
        original = pretrain.load_config('configs/pretrain/b160_perceptual.yaml')[0]
        with torch.device('meta'):
            legacy = pretrain.make_model(original,device='meta')
            before = pretrain.make_model(self.preconfig,device='meta')
            after = posttrain.make_model(self.postconfig,device='meta')
        schema = lambda model:{key:value.shape for key,value in model.state_dict().items()}
        self.assertEqual(schema(legacy),schema(before))
        self.assertEqual(schema(before),schema(after))
        self.assertEqual(sum(p.numel() for p in before.parameters()),115063936)
        self.assertFalse(any('noise_t_embedder' in name or 'adaLN' in name for name in before.state_dict()))

    def test_actual_handoff_inherits_online_ema_and_starts_empty_adam(self):
        source = self.checkpoint()
        self.assertTrue(validate_diffusion_pretraining_contract(source))
        model = self.tiny_model()
        optimizer,ema = initialize_posttraining(model,self.path,self.postconfig['optimizer'],config=self.postconfig)
        self.assertIsInstance(optimizer,torch.optim.AdamW)
        self.assertFalse(optimizer.state)
        self.assertEqual((ema.step_count,ema.label),(12510,'0.999'))
        for name,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,source['model'][name]),name)
            self.assertTrue(torch.equal(ema.shadow[name],source['model_ema']['shadows']['0.9999'][name]),name)
        optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertEqual(optimized,{id(p) for p in model.parameters()})
        seen = []
        handle = model.net.register_forward_pre_hook(lambda module,args,kwargs:seen.append(kwargs),with_kwargs=True)
        try:
            output = model.sample_images_with_grad(torch.randn(1,3,256,256),torch.tensor([1]),self.postconfig['sampling'])
            output.square().mean().backward()
        finally:
            handle.remove()
        self.assertEqual(len(seen),1)
        self.assertNotIn('noise_t',seen[0])
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_no_time_checkpoint_cannot_be_relabelled_or_gain_conditioning_tensors(self):
        source = self.checkpoint()
        changed = deepcopy(source)
        del changed['contract']['experiment']['conditioning']
        del changed['config']['experiment']['conditioning']
        with self.assertRaisesRegex(ValueError,'declared timestep-conditioning'):
            validate_diffusion_pretraining_contract(changed)
        changed = deepcopy(source)
        changed['config']['experiment']['conditioning'] = 'timestep'
        with self.assertRaisesRegex(ValueError,'experiment must agree'):
            validate_diffusion_pretraining_contract(changed)
        for key in ('net.noise_t_embedder.mlp.0.weight','net.blocks.0.adaLN_modulation.1.weight'):
            for which in ('online','ema'):
                changed = deepcopy(source)
                tensors = changed['model'] if which=='online' else changed['model_ema']['shadows']['0.9999']
                tensors[key] = torch.zeros(2,2)
                with self.subTest(key=key,which=which),self.assertRaises(ValueError):
                    validate_timestep_contract(changed,self.postconfig)

    def test_time_conditioning_mismatch_rejected_by_initializer_and_launcher(self):
        self.checkpoint()
        configured = deepcopy(self.postconfig)
        configured['model']['timestep_conditioning'] = True
        with self.assertRaisesRegex(ValueError,'timestep_conditioning differs'):
            initialize_posttraining(self.tiny_model(),self.path,configured['optimizer'],config=configured)
        plan = dict(output='/tmp/new-no-time-run',posttraining=dict(config='/tmp/post.yaml'),
                    wandb_project='project',wandb_entity='entity')
        configured['initialization']['checkpoint'] = plan['output']+'/checkpoints/step_0012510.pth'
        configured['tracking'] = dict(project='project',entity='entity',name='post',id='post-id')
        with patch('posttrain.load_config',return_value=configured), \
                patch('scripts.run_diffusion_pretrain.resolved_configs',return_value=(self.preconfig,{})):
            with self.assertRaisesRegex(ValueError,'timestep-conditioning architecture'):
                resolved_post_configs(plan)
            configured['model']['timestep_conditioning'] = False
            production,smoke = resolved_post_configs(plan)
            self.assertFalse(production['model']['timestep_conditioning'])
            self.assertNotIn('tracking',smoke)


if __name__ == '__main__':
    unittest.main()
