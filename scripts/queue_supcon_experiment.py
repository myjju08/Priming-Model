"""Wait for transferred assets, validate SupCon, then smoke-test and train.

Run under a process manager. A JSON plan keeps local paths and reference
identities outside Git. ``STORAGE/STOP_REQUESTED`` stops the waiting queue or
prevents its next stage; use ``OUTPUT/STOP_REQUESTED`` for a checkpointed stop
of active training. Failed commands are never automatically retried here.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.files import sha256, write_json

REFERENCE_NAMES = {'fid_reference': 'imagenet_256_fid_stats.npz',
                   'prc_images': 'imagenet_val_prc_arr0.npz'}


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    for key in ('storage', 'checkpoint', 'conversion_python', 'output',
                'smoke_output', 'wandb_project', 'wandb_entity', 'run_name'):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ValueError(f'Plan requires {key}')
    for key in ('storage', 'checkpoint', 'conversion_python', 'output', 'smoke_output'):
        plan[key] = str(Path(plan[key]).expanduser().resolve())
    if plan['output'] == plan['smoke_output']:
        raise ValueError('Smoke output must differ from production output')
    plan.setdefault('evaluation', 'drive')
    if plan['evaluation'] != 'drive':
        raise ValueError('This queue requires the matched Drive evaluation protocol')
    plan.setdefault('epochs', 40)
    plan.setdefault('eval_every_epochs', 5)
    for key in ('epochs', 'eval_every_epochs'):
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(f'Invalid {key}')
    if plan['eval_every_epochs'] > plan['epochs']:
        raise ValueError('Evaluation interval exceeds duration')
    refs = plan.get('reference_sha256', {})
    for key in REFERENCE_NAMES:
        digest = refs.get(key, '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'Plan requires a lowercase SHA256 for {key}')
    return plan


def plan_identity(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def readiness(plan, observed, now, *, stable_seconds=60):
    """Readiness only; identities and the RN50 state are verified before use."""
    storage = Path(plan['storage'])
    missing = []
    for group in ('dataset', 'support'):
        path = storage/'status'/f'{group}.json'
        if not path.is_file():
            missing.append(f'{group} download status')
            continue
        record = json.loads(path.read_text())
        if record.get('status') == 'failed':
            raise ValueError(f'{group} asset download failed; inspect {path}')
        if record.get('status') != 'complete':
            missing.append(f'{group} downloads')
        elif Path(record.get('storage', '')).resolve() != storage:
            raise ValueError(f'{group} status belongs to another storage root')
    env = storage/'status/environment.json'
    if not env.is_file():
        missing.append('environment validation')
    else:
        result = json.loads(env.read_text())
        if (result.get('status') != 'passed' or
                result.get('unit_tests', {}).get('status') != 'passed' or
                result.get('distributed_gpu', {}).get('status') != 'passed' or
                len(result.get('distributed_gpu', {}).get('gpus', [])) != 8):
            raise ValueError('Environment validation did not pass for eight GPUs')
    if not Path(plan['conversion_python']).is_file():
        missing.append('conversion Python environment')
    for name in REFERENCE_NAMES.values():
        if not (storage/'assets/drive_eval'/name).is_file():
            missing.append(name)
    checkpoint = Path(plan['checkpoint'])
    if not checkpoint.is_file():
        observed.clear()
        missing.append('SupCon checkpoint transfer')
    else:
        stat = checkpoint.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if observed.get('signature') != signature:
            observed.update(signature=signature, since=now)
        if not stat.st_size or now-observed['since'] < stable_seconds:
            missing.append(f'SupCon checkpoint stable for {stable_seconds:g}s')
    return missing


def verify_references(plan):
    directory = Path(plan['storage'])/'assets/drive_eval'
    for key, name in REFERENCE_NAMES.items():
        if sha256(directory/name) != plan['reference_sha256'][key]:
            raise ValueError(f'Drive reference SHA256 mismatch: {name}')


def resume_mode(output):
    output = Path(output)
    if not output.exists():
        return False
    if (output/'run.json').is_file() and (output/'checkpoints/latest.json').is_file():
        return True
    raise ValueError(f'Existing output has no resumable checkpoint: {output}')


def completed_updates(output):
    directory = Path(output)/'checkpoints'
    record = json.loads((directory/'latest.json').read_text())
    name = record.get('checkpoint', '')
    completed = record.get('completed')
    if (not name or Path(name).name != name or not (directory/name).is_file() or
            type(completed) is not int or completed < 0):
        raise ValueError('Invalid latest checkpoint record')
    return completed


def train_command(plan, *, smoke=False, resume=False):
    command = [sys.executable, str(ROOT/'scripts/experiment.py'), 'train',
               '--encoder', 'supcon', '--storage', plan['storage'],
               '--evaluation', 'drive', '--output', plan['smoke_output'] if smoke else plan['output'],
               '--epochs', str(plan['epochs']), '--eval-every-epochs', str(plan['eval_every_epochs'])]
    if smoke:
        command += ['--smoke-steps', '2']
    else:
        command += ['--wandb-project', plan['wandb_project'], '--wandb-entity',
                    plan['wandb_entity'], '--run-name', plan['run_name']]
    if resume:
        command.append('--resume')
    return command


def preserve_identity(path, identity):
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(f'Input or plan changed after preparation: {path}')
    write_json(path, identity)


def run_queue(plan, *, poll_seconds=30):
    storage = Path(plan['storage'])
    status = storage/'status/supcon.json'
    identity_path = storage/'status/supcon-input.json'
    smoke_marker = storage/'status/supcon-smoke.json'
    observed = {}
    stage = 'waiting'

    def record(state, **extra):
        write_json(status, dict(status=state, stage=stage, updated_at=time.time(),
                               output=plan['output'], checkpoint=plan['checkpoint'], **extra))
        print(f'SupCon queue: {state} / {stage}' +
              (': '+', '.join(extra['missing']) if extra.get('missing') else ''), flush=True)

    def stop_requested():
        if (storage/'STOP_REQUESTED').exists():
            record('stopped')
            return True
        return False

    def execute(command):
        record('running')
        subprocess.run(command, cwd=ROOT, check=True)

    try:
        while True:
            if stop_requested():
                return
            missing = readiness(plan, observed, time.monotonic())
            if not missing:
                break
            record('waiting', missing=missing)
            time.sleep(poll_seconds)
        stage = 'validating_inputs'
        record('running')
        verify_references(plan)
        from models.encoders.rn50 import load_backbone
        checkpoint = Path(plan['checkpoint'])
        before = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
        checkpoint_sha = sha256(checkpoint)
        backbone = load_backbone(checkpoint)
        del backbone
        after = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
        if before != after:
            raise ValueError('SupCon checkpoint changed during validation')
        identity = dict(checkpoint_sha256=checkpoint_sha, plan_sha256=plan_identity(plan))
        preserve_identity(identity_path, identity)
        if stop_requested():
            return
        stage = 'converting_convnext'
        converted = storage/'assets/ot/representations/convnext_encoder.pt'
        if not converted.is_file():
            execute([plan['conversion_python'], str(ROOT/'scripts/convert_ot_encoders.py'),
                     '--root', str(storage), '--family', 'convnext',
                     '--output-dir', str(converted.parent)])
        from utils.posttrain_assets import verify_file
        rows = json.loads((ROOT/'old/releases/posttrain_encoders.json').read_text())['files']
        verify_file(converted, next(row for row in rows if row['path'] == converted.name))
        if stop_requested():
            return
        stage = 'calibrating_supcon_and_references'
        execute([sys.executable, str(ROOT/'scripts/experiment.py'), 'prepare',
                 '--encoder', 'supcon', '--storage', str(storage), '--evaluation', 'drive',
                 '--encoder-checkpoint', str(checkpoint), '--encoder-sha256', checkpoint_sha])
        if stop_requested():
            return
        stage = 'smoke_test'
        if smoke_marker.exists():
            if json.loads(smoke_marker.read_text()) != dict(status='passed', **identity):
                raise ValueError('Smoke validation belongs to different inputs')
        else:
            # A checkpoint from an interrupted smoke check can finish that same check.
            execute(train_command(plan, smoke=True, resume=resume_mode(plan['smoke_output'])))
            if completed_updates(plan['smoke_output']) < 2:
                raise ValueError('Smoke test returned before two completed updates')
            if (Path(plan['smoke_output'])/'STOP_REQUESTED').exists():
                record('stopped')
                return
            write_json(smoke_marker, dict(status='passed', **identity))
        if stop_requested():
            return
        stage = 'training'
        execute(train_command(plan, resume=resume_mode(plan['output'])))
        stage = 'training_exited'
        completed = completed_updates(plan['output'])
        record('complete' if completed == 2503*plan['epochs'] else 'stopped',
               completed_updates=completed)
    except BaseException as error:
        record('failed', error_type=type(error).__name__, error=str(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    lock_path = Path(plan['storage'])/'status/supcon.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another SupCon queue already owns this storage root') from None
        run_queue(plan)


if __name__ == '__main__':
    main()
