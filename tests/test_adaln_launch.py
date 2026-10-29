"""Launch boundary checks that prevent mixed or mislabeled ablation runs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from prepare_adaln_ablation import validate_pair, pretrain_command, posttrain_command
from run_adaln_ablation import validate_payload, MetricsMirror, rewind_metrics


def config(arm):
    return dict(model=dict(model='OST_B_ADALN',t_dist='uniform',
            t_min=.8 if arm=='fixed' else 0.,t_max=.8 if arm=='fixed' else 1.),
        train=dict(epochs=10,steps_per_epoch=1251,seed=1,micro_batch_size=64,gradient_accumulation_steps=2),
        loss=dict(perceptual_threshold=.8,aux=dict(enabled=True,lpips_weight=.4,convnext_weight=.1)),
        experiment=dict(profile='adaln_timestep_pretraining',arm=arm))


class LaunchTests(unittest.TestCase):
    def test_only_timestep_may_differ(self):
        pair={arm:config(arm) for arm in ('fixed','uniform')}
        validate_pair(pair)
        pair['uniform']['loss']['aux']['lpips_weight']=0.
        with self.assertRaises(ValueError): validate_pair(pair)
        pair={arm:config(arm) for arm in ('fixed','uniform')}
        pair['uniform']['optimizer']={'lr':.002}
        with self.assertRaises(ValueError): validate_pair(pair)

    def test_checkpoint_cannot_be_misidentified_as_completed(self):
        c=config('fixed'); manifest={'runtime':'identity'}; step=12510
        state=dict(step=step,current_step=step+1,samples_seen=step*1024,config=c,manifest=manifest,
            world_size=8,rank_states=[dict(rank=r,next_data=dict(epoch=10,offset=0)) for r in range(8)],
            model_ema=dict(step_count=step,schedules=[('0.9999',.9999)],shadows={'0.9999':{}}))
        self.assertEqual(validate_payload(state,c,manifest,final=True),12510)
        for mutate in (
            lambda s:s.update(step=12509),
            lambda s:s['rank_states'][7]['next_data'].update(epoch=9),
            lambda s:s['model_ema'].update(schedules=[('0.999',.999)]),
            lambda s:s.update(config=config('uniform'))):
            broken=deepcopy(state); mutate(broken)
            with self.assertRaises(ValueError): validate_payload(broken,c,manifest,final=True)

    def test_resume_metrics_preserves_uncheckpointed_tail(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'metrics.jsonl'
            p.write_text(''.join(json.dumps(dict(step=i,loss=1.))+ '\n' for i in range(1,5))+'{"step":5')
            rewind_metrics(p,2)
            self.assertEqual([json.loads(x)['step'] for x in p.read_text().splitlines()],[1,2])
            self.assertEqual(len(list(Path(tmp).glob('metrics.jsonl.uncheckpointed-*'))),1)
            with self.assertRaises(ValueError): rewind_metrics(p,3)

    def test_mirror_resumes_wandb_steps_and_waits_for_complete_line(self):
        class Run:
            step=3
            def __init__(self): self.logs=[]
            def log(self,values,step): self.logs.append((step,values))
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'metrics.jsonl'
            p.write_text(''.join(json.dumps(dict(step=i,loss=.5))+'\n' for i in range(1,4))+'{"step":4')
            run=Run(); mirror=MetricsMirror(p,run); mirror.pump()
            self.assertEqual([x[0] for x in run.logs],[3])
            with p.open('a') as f:f.write(',"loss":0.4}\n')
            mirror.pump();mirror.pump()
            self.assertEqual([x[0] for x in run.logs],[3,4])
            self.assertEqual(run.logs[-1][1]['train/samples_seen'],4096)

    def test_commands_select_corresponding_initialization(self):
        plan=dict(python='/venv/main/bin/python',source_root='/repo',data_path='/data',
            hf_home='/hf',torch_home='/torch',lpips_weights='/lpips',arms={})
        for arm in ('fixed','uniform'):
            plan['arms'][arm]=dict(pretrain_recipe=f'/{arm}-pre.yaml',pretrain_output=f'/{arm}-pre',
                posttrain_recipe=f'/{arm}-post.yaml',posttrain_output=f'/{arm}-post',final_checkpoint=f'/{arm}.pth')
        pre=pretrain_command(plan,'uniform',resume='/own/checkpoint.pth',limit=3)
        self.assertEqual(pre[pre.index('--resume')+1],'/own/checkpoint.pth')
        self.assertEqual(pre[pre.index('--smoke-steps')+1],'3')
        post=posttrain_command(plan,'fixed')
        self.assertEqual(post[post.index('--init-checkpoint')+1],'/fixed.pth')
        self.assertNotIn('/uniform.pth',post)


if __name__=='__main__':
    unittest.main()
