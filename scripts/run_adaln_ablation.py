"""Explicit managed launch of fixed then uniform adaLN pretraining; never post-training."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_adaln_ablation import ARMS, pretrain_command, sha256, write_json, validate_pair


def status(plan, phase, **fields):
    record = dict(phase=phase, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), **fields)
    write_json(Path(plan['bundle'])/'status.json', record)
    print(json.dumps(record), flush=True)


def validate_payload(state, config, manifest, *, target=12510, final=False):
    """Checkpoint counters and every rank's data cursor must identify this exact run."""
    step = state.get('step')
    if type(step) is not int or not 0 <= step <= target or (final and step != target):
        raise ValueError('Checkpoint step is outside the requested horizon')
    train = config['train']
    local_batch = train['micro_batch_size'] * train['gradient_accumulation_steps']
    if (state.get('current_step') != step+1 or state.get('samples_seen') != step*local_batch*8
            or state.get('config') != config or state.get('manifest') != manifest
            or state.get('world_size') != 8 or len(state.get('rank_states',[])) != 8):
        raise ValueError('Checkpoint identity, counters or topology differ')
    ema = state.get('model_ema',{})
    if (ema.get('step_count') != step or ema.get('schedules') != [('0.9999',.9999)]
            or set(ema.get('shadows',{})) != {'0.9999'}):
        raise ValueError('Pretraining EMA identity differs')
    cursor = dict(epoch=step//train['steps_per_epoch'],offset=(step%train['steps_per_epoch'])*local_batch)
    for rank, row in enumerate(state['rank_states']):
        if row.get('rank') != rank or row.get('next_data') != cursor:
            raise ValueError('Rank cursor differs')
    return step


def checkpoint_step(path, config, manifest, *, final=False):
    import torch
    torch.set_num_threads(2)
    state = torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    step = validate_payload(state,config,manifest,final=final)
    if final:
        for tensors in (state['model'],state['model_ema']['shadows']['0.9999']):
            if not tensors or any(not torch.isfinite(value).all().item() for value in tensors.values()):
                raise ValueError('Empty or nonfinite final weights')
    return step


def preflight(plan, verify_wandb=True):
    root = Path(plan['source_root'])
    if root != ROOT or plan.get('version') != 1 or plan.get('posttraining_automatic') is not False:
        raise ValueError('Unexpected source root/plan version or automatic post-training')
    if plan.get('target_pretraining_updates') != 12510 or plan.get('arm_order') != list(ARMS):
        raise ValueError('Unexpected pretraining horizon/order')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() != plan['source_commit']:
        raise ValueError('Source commit changed since preparation')
    for name, expected in plan['source_sha256'].items():
        if sha256(root/name) != expected:
            raise ValueError(f'Runtime source changed since preparation: {name}')
    from pretrain import load_config
    from utils.assets import pretraining_manifest
    configs = {}
    for arm, item in plan['arms'].items():
        for prefix in ('pretrain','posttrain'):
            if sha256(item[f'{prefix}_recipe']) != item[f'{prefix}_recipe_sha256']:
                raise ValueError(f'{arm} {prefix} recipe changed')
        configs[arm], family = load_config(item['pretrain_recipe'])
        if family != 'b':
            raise ValueError('Expected a B-family experiment')
    validate_pair(configs)
    manifest = pretraining_manifest(configs['fixed'],root=root,family='b',hf_home=plan['hf_home'],
        torch_home=plan['torch_home'],lpips_weights=plan['lpips_weights'])
    if manifest != plan['pretraining_manifest']:
        raise ValueError('Runtime/data/perceptual asset identity differs from prepared plan')
    if verify_wandb:
        import wandb
        from wandb.sdk.lib.wbauth.authenticate import authenticate_session
        if authenticate_session(host=wandb.Settings().base_url,source='adaLN preflight',prompt=False,verify=True) is None:
            raise RuntimeError('W&B authentication unavailable')
    return configs, manifest


class MetricsMirror:
    def __init__(self,path,run,epoch_steps=1251):
        self.path,self.run,self.epoch_steps = Path(path),run,epoch_steps
        self.offset,self.observed_step = 0,0
        self.last_logged = max(0,int(run.step)-1)

    def pump(self):
        if not self.path.exists():
            return
        with self.path.open() as handle:
            handle.seek(self.offset)
            while True:
                before=handle.tell(); line=handle.readline()
                if not line or not line.endswith('\n'):
                    self.offset=before
                    return
                row=json.loads(line); step=row['step']
                self.observed_step=max(self.observed_step,step)
                if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in row.values()):
                    raise FloatingPointError(f'Invalid metric at update {step}')
                if step > self.last_logged:
                    values={f'train/{k}':v for k,v in row.items() if k!='step'}
                    values.update({'train/update':step,'train/epoch':step/self.epoch_steps,'train/samples_seen':step*1024})
                    self.run.log(values,step=step)
                    self.last_logged=step


def rewind_metrics(path, completed):
    """Preserve uncheckpointed tail before deterministic replay after a crash."""
    path=Path(path)
    if not path.exists():
        if completed:
            raise ValueError('Resumed checkpoint has no metric history')
        return
    original=path.read_text(); retained=[]
    for line in original.splitlines(keepends=True):
        if not line.endswith('\n'):
            break
        row=json.loads(line)
        if row['step'] <= completed:
            retained.append((row['step'],line))
    if [step for step,_ in retained] != list(range(1,completed+1)):
        raise ValueError('Metric history through the checkpoint is incomplete or duplicated')
    repaired=''.join(line for _,line in retained)
    if repaired != original:
        archive=path.with_name(f'{path.name}.uncheckpointed-{time.time_ns()}')
        archive.write_text(original)
        temporary=path.with_suffix('.tmp');temporary.write_text(repaired);temporary.replace(path)


def stopped(plan):
    return Path(plan['stop_file']).exists() or Path(plan['cancel_file']).exists()


def preserve_final(plan, arm, config, manifest):
    item=plan['arms'][arm]
    final=Path(item['pretrain_output'])/'checkpoints/step_0012510.pth'
    checkpoint_step(final,config,manifest,final=True)
    digest=sha256(final); canonical=Path(item['final_checkpoint'])
    canonical.parent.mkdir(parents=True,exist_ok=True)
    if canonical.exists():
        if sha256(canonical)!=digest:
            raise FileExistsError('Different checkpoint occupies the canonical path')
    else:
        os.link(final,canonical)
    record=dict(arm=arm,checkpoint=str(final),canonical_checkpoint=str(canonical),sha256=digest,
        completed_updates=12510,completed_epochs=10,initialized_from='random_from_scratch',
        backbone='OST_B_ADALN',ema_decay=.9999,finite_model_and_ema=True)
    write_json(item['completion'],record)
    return record


def launch_arm(plan, arm, config, manifest):
    import wandb
    item=plan['arms'][arm]; output=Path(item['pretrain_output'])
    resume=None; completed=0
    if output.exists():
        identity=output/'ablation-plan.json'; latest=output/'checkpoints/latest.pth'
        if not identity.exists() or json.loads(identity.read_text()) != plan or not latest.is_file():
            raise RuntimeError('Existing output does not belong to this plan/checkpoint')
        resume=latest.resolve()
        if resume.parent != output/'checkpoints':
            raise ValueError('Resume symlink points outside this run')
        completed=checkpoint_step(resume,config,manifest)
        rewind_metrics(output/'metrics.jsonl',completed)
    tracking=item['tracking']
    run=wandb.init(**tracking,resume='allow',dir=plan['bundle'],job_type='pretraining',
        tags=['B16','adaLN','pretrain10',arm,'perceptual'],
        config=dict(recipe=config,arm=arm,target_updates=12510,requested_epochs=10,
            initialization='random_from_scratch',source_commit=plan['source_commit'],
            source_sha256=plan['source_sha256'],global_batch_size=1024))
    run.define_metric('train/update'); run.define_metric('train/*',step_metric='train/update')
    child=None
    try:
        if completed==12510:
            MetricsMirror(output/'metrics.jsonl',run).pump()
            record=preserve_final(plan,arm,config,manifest)
            run.summary.update(record);run.summary['phase']='completed'
            run.finish();return True
        if stopped(plan):
            run.summary['phase']='cancelled_before_launch';run.finish();return False
        env=os.environ.copy()
        env.update(PYTHONPATH=plan['python_overlay'],HF_HOME=plan['hf_home'],TORCH_HOME=plan['torch_home'],
            PREEMPT_FLAG_FILE=plan['stop_file'],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
            TORCHINDUCTOR_COMPILE_THREADS='4',PYTORCH_ALLOC_CONF='expandable_segments:True',
            PYTHONUNBUFFERED='1',PRIMING_CACHE_RUN_ID=hashlib.sha256(
                f"adaln-{arm}-{plan['tag']}".encode()).hexdigest()[:8])
        env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        status(plan,'starting',arm=arm,resume=str(resume) if resume else None,wandb_url=run.url)
        with Path(item['log']).open('a',buffering=1) as log:
            child=subprocess.Popen(pretrain_command(plan,arm,resume),cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT)
            mirror=MetricsMirror(output/'metrics.jsonl',run); last_report=0
            while True:
                if output.is_dir() and not (output/'ablation-plan.json').exists():
                    write_json(output/'ablation-plan.json',plan)
                if Path(plan['cancel_file']).exists():
                    Path(plan['stop_file']).touch()
                mirror.pump()
                if time.monotonic()-last_report>=30:
                    status(plan,'training' if mirror.observed_step else 'initializing',arm=arm,
                        completed_updates=mirror.observed_step,target_updates=12510,
                        completed_epochs=mirror.observed_step/1251,pid=child.pid,wandb_url=run.url)
                    last_report=time.monotonic()
                code=child.poll()
                if code is not None:
                    mirror.pump();break
                time.sleep(2)
        if code==75 and stopped(plan):
            status(plan,'checkpointed_stop',arm=arm,completed_updates=mirror.observed_step)
            run.summary['phase']='checkpointed_stop';run.finish();return False
        if code!=0:
            raise RuntimeError(f'{arm} training exited {code}; see {item["log"]}')
        record=preserve_final(plan,arm,config,manifest)
        run.summary.update(record);run.summary['phase']='completed';run.finish()
        status(plan,'arm_completed',**record)
        return True
    except BaseException:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill();child.wait()
        run.finish(exit_code=1)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--arm',choices=(*ARMS,'all'),default='all')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--offline-check',action='store_true',help='Skip W&B auth during a check-only invocation')
    args=parser.parse_args(argv)
    if args.offline_check and not args.check_only:
        parser.error('--offline-check requires --check-only')
    plan=json.loads(Path(args.plan).read_text())
    with (Path(plan['bundle'])/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        configs,manifest=preflight(plan,verify_wandb=not args.offline_check)
        if args.check_only:
            print(json.dumps(dict(preflight='passed',gpus_checked=False,outputs_created=False,
                commands={arm:pretrain_command(plan,arm) for arm in ARMS}),indent=2));return 0
        if stopped(plan):
            status(plan,'cancelled_before_launch');return 0
        def request_stop(signum,frame):
            Path(plan['stop_file']).touch()
        signal.signal(signal.SIGTERM,request_stop)
        signal.signal(signal.SIGINT,request_stop)
        for arm in (ARMS if args.arm=='all' else (args.arm,)):
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if busy:
                raise RuntimeError('GPU compute processes are active; refusing an overlapping training run')
            if not launch_arm(plan,arm,configs[arm],manifest):
                return 0
            # Recheck all identities before starting the next arm.
            configs,manifest=preflight(plan,verify_wandb=False)
        status(plan,'completed',arms=list(ARMS) if args.arm=='all' else [args.arm],posttraining_started=False)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
