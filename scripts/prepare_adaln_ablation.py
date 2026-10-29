"""Prepare two matched adaLN pretraining runs without starting either run."""
import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ('fixed', 'uniform')
ENTITY = 'a01065522071-kaist-digital-humanities-and-social-science'
PROJECT = 'CVPR - Priming Model'


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def source_identities(root):
    root = Path(root)
    files = [root/'pretrain.py', root/'posttrain.py', root/'eval.py',
             root/'scripts/prepare_adaln_ablation.py', root/'scripts/run_adaln_ablation.py']
    for directory in ('models', 'losses', 'utils', 'data', 'metrics'):
        files.extend(sorted((root/directory).rglob('*.py')))
    return {str(path.relative_to(root)): sha256(path) for path in files}


def validate_pair(configs):
    """The arm label and timestep range are the only scientific differences."""
    from copy import deepcopy
    if set(configs) != set(ARMS):
        raise ValueError('Both fixed and uniform recipes are required')
    normalized = []
    for arm in ARMS:
        c = deepcopy(configs[arm])
        model, train, loss = c['model'], c['train'], c['loss']
        bounds = (.8, .8) if arm == 'fixed' else (0., 1.)
        if model['model'] != 'OST_B_ADALN' or model['t_dist'] != 'uniform':
            raise ValueError('Both arms require the same timestep-conditioned B/16')
        if (model['t_min'], model['t_max']) != bounds:
            raise ValueError(f'Wrong {arm} timestep distribution')
        if (train['epochs'], train['steps_per_epoch'], train['seed']) != (10, 1251, 1):
            raise ValueError('Both arms require ten epochs, 1251 steps/epoch, seed 1')
        if train['micro_batch_size'] * train['gradient_accumulation_steps'] * 8 != 1024:
            raise ValueError('Both arms require global batch 1024')
        if (loss['aux']['enabled'], loss['aux']['lpips_weight'], loss['aux']['convnext_weight'],
                loss['perceptual_threshold']) != (True, .4, .1, .8):
            raise ValueError('Original perceptual objective must be retained')
        c['model']['t_min'] = c['model']['t_max'] = None
        c.get('experiment', {}).pop('arm', None)
        normalized.append(c)
    if normalized[0] != normalized[1]:
        raise ValueError('Pretraining arms differ beyond the timestep distribution')


def pretrain_command(plan, arm, resume=None, limit=None):
    item = plan['arms'][arm]
    command = [plan['python'], '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
        str(Path(plan['source_root'])/'pretrain.py'), '--config', item['pretrain_recipe'],
        '--output', item['pretrain_output'], '--data-path', plan['data_path'],
        '--hf-home', plan['hf_home'], '--torch-home', plan['torch_home'],
        '--lpips-weights', plan['lpips_weights']]
    if resume:
        command.extend(['--resume', str(resume)])
    if limit is not None:
        command.extend(['--smoke-steps', str(limit)])
    return command


def posttrain_command(plan, arm):
    item = plan['arms'][arm]
    return [plan['python'], '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
        str(Path(plan['source_root'])/'posttrain.py'), '--config', item['posttrain_recipe'],
        '--output', item['posttrain_output'], '--init-checkpoint', item['final_checkpoint'],
        '--hf-home', plan['hf_home'], '--torch-home', plan['torch_home']]


def prepare(args):
    import yaml
    from utils.config import load_yaml
    from utils.assets import pretraining_manifest
    from utils.posttrain_assets import evaluation_assets
    storage, bundle = Path(args.storage).resolve(), Path(args.bundle).resolve()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', args.tag):
        raise ValueError('Tag must use lowercase letters, digits and hyphens')
    plan_path = bundle/'plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if any(Path(a[key]).exists() for a in previous['arms'].values()
               for key in ('pretrain_output', 'posttrain_output')):
            raise FileExistsError('A training output exists; do not regenerate its pinned plan')
        if not args.replace:
            raise FileExistsError('Plan exists; use --replace only before training starts')
    bundle.mkdir(parents=True, exist_ok=True)
    # Keep the venv entrypoint: resolving its symlink can select system Python.
    python = str(Path(args.python).expanduser().absolute())
    if not Path(python).is_file():
        raise FileNotFoundError(python)
    plan = dict(version=1, preparation_only=True, source_root=str(ROOT),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=source_identities(ROOT), python=python, tag=args.tag,
        requested_pretraining_epochs=10, target_pretraining_updates=12510,
        requested_posttraining_epochs=5, target_posttraining_updates=12515,
        posttraining_automatic=False, arm_order=list(ARMS),
        bundle=str(bundle), data_path=str(storage/'data/imagenet'),
        hf_home=str(storage/'assets/huggingface'), torch_home=str(storage/'assets/torch'),
        lpips_weights=str(storage/'assets/lpips/vgg.pth'),
        python_overlay=str(storage/'env/pretrain10-overrides'),
        stop_file=str(bundle/'STOP_REQUESTED'), cancel_file=str(bundle/'CANCEL_REQUESTED'), arms={})
    configs = {}
    for arm in ARMS:
        pre = load_yaml(ROOT/f'configs/pretrain/adaln_{arm}_10ep.yaml')
        pre['data']['data_path'] = plan['data_path']
        configs[arm] = pre
    validate_pair(configs)
    # The feature files are the same audited reference files as the completed run.
    baseline = json.loads(Path(args.baseline_run).read_text())['config']
    plan['baseline_run'] = str(Path(args.baseline_run).resolve())
    plan['baseline_run_sha256'] = sha256(args.baseline_run)
    pre_manifest = pretraining_manifest(configs['fixed'], root=ROOT, family='b',
        hf_home=plan['hf_home'], torch_home=plan['torch_home'], lpips_weights=plan['lpips_weights'])
    plan['pretraining_manifest'] = pre_manifest
    for arm in ARMS:
        run_id = f'adaln-{arm}-pre10-{args.tag}'
        pre_output = storage/'outputs/pretraining'/run_id
        post_output = storage/'outputs/posttrain'/f'adaln-{arm}-pre10-post5-{args.tag}'
        final = storage/'checkpoints/pretraining'/f'priming_b_adaln_{arm}_10_{args.tag}.pth'
        if pre_output.exists() or post_output.exists() or final.exists():
            raise FileExistsError(f'Output/checkpoint collision for {arm}; choose a new tag')
        pre_path, post_path = bundle/f'{arm}-pretrain.yaml', bundle/f'{arm}-posttrain.yaml'
        pre_path.write_text(yaml.safe_dump(configs[arm], sort_keys=False))
        post = load_yaml(ROOT/f'configs/posttrain/adaln_{arm}_post5.yaml')
        post['data'].update(data_path=plan['data_path'], encoder_cache=str(storage/'assets/ot/supervised'))
        post['initialization']['checkpoint'] = str(final)
        post['evaluation'].update({key: baseline['evaluation'][key] for key in
            ('protocol','fid_reference','prc_images','prc_reference','reference_sha256','real_num_images','stats_dir')})
        post['tracking'] = dict(entity=ENTITY, project=PROJECT,
            id=post_output.name, name=f'B16 adaLN {arm} pre10 → supervised RN50 post5')
        post_path.write_text(yaml.safe_dump(post, sort_keys=False))
        plan['arms'][arm] = dict(pretrain_recipe=str(pre_path), pretrain_recipe_sha256=sha256(pre_path),
            posttrain_recipe=str(post_path), posttrain_recipe_sha256=sha256(post_path),
            pretrain_output=str(pre_output), posttrain_output=str(post_output), final_checkpoint=str(final),
            tracking=dict(entity=ENTITY, project=PROJECT, id=run_id,
                name=f'B16 adaLN {arm} pretraining 10ep'),
            log=str(bundle/f'{arm}-pretrain.log'), completion=str(bundle/f'{arm}-completed.json'))
    evaluation_assets(post['evaluation'], hf_home=plan['hf_home'], torch_home=plan['torch_home'])
    # Validate the resolved files with the same parsers used by the actual launch.
    import pretrain, posttrain
    for arm in ARMS:
        pretrain.load_config(plan['arms'][arm]['pretrain_recipe'])
        posttrain.load_config(plan['arms'][arm]['posttrain_recipe'])
    service = f'priming-adaln-pretrain-{args.tag}'
    wrapper = bundle/f'{service}.sh'
    runner = ROOT/'scripts/run_adaln_ablation.py'
    wrapper.write_text('\n'.join(['#!/bin/bash', 'utils=/opt/supervisor-scripts/utils',
        '. "${utils}/logging.sh"', '. "${utils}/environment.sh"', 'set -euo pipefail',
        f'cd {shlex.quote(str(ROOT))}', 'unset PYTHONPATH PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF',
        'export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4',
        'export WANDB_MODE=online',
        f'exec {shlex.join([python,str(runner),"--plan",str(plan_path)])}', '']))
    wrapper.chmod(0o755)
    supervisor = bundle/f'{service}.conf'
    supervisor.write_text(f'''[program:{service}]
environment=PROC_NAME="%(program_name)s"
command={shlex.quote(str(wrapper))}
autostart=false
autorestart=false
exitcodes=0
startsecs=5
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=90
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
''')
    plan['supervisor'] = dict(name=service, wrapper=str(wrapper), config=str(supervisor), autostart=False)
    plan['commands'] = {arm:dict(pretrain=pretrain_command(plan,arm),posttrain=posttrain_command(plan,arm)) for arm in ARMS}
    write_json(plan_path, plan)
    write_json(bundle/'status.json', dict(phase='prepared_not_started', pretraining_automatic=False,
        posttraining_automatic=False, plan=str(plan_path)))
    print(json.dumps(dict(plan=str(plan_path), supervisor=str(supervisor),
        status='prepared_not_started', outputs_created=False, arms=plan['commands']),indent=2))
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', default='/workspace/priming-storage')
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--python', default='/venv/main/bin/python')
    parser.add_argument('--baseline-run', default='/workspace/priming-storage/outputs/posttrain/b10-supervised-rn50-32x3-post40-20261028/run.json')
    parser.add_argument('--replace', action='store_true', help='Rebuild a plan only while all training outputs remain absent')
    return prepare(parser.parse_args(argv))


if __name__ == '__main__':
    main()
