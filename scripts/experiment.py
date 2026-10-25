"""Prepare assets or launch an encoder comparison without editing training code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.experiments import ENCODERS, positive, recipe


def download_assets(storage, encoder, protocol, conversion_python):
    from scripts.download_assets import download
    dataset = json.loads((ROOT/'old/releases/dataset.json').read_text())
    for row in dataset['files']:
        download(storage, dict(row, kind='hub_file', repo_id=dataset['repo_id'], repo_type='dataset',
            revision=dataset['revision'], filename=row['path']))
    rows = json.loads((ROOT/'old/releases/encoders.json').read_text())['files']
    for row in rows:
        if 'posttraining' in row['groups'] and (protocol == 'native' or 'inception' in row['path']):
            download(storage, row)
    # Small pinned class indices and calibration metadata are shipped in the dataset release.
    base = storage/'assets/ot/representations'
    for family in (['mae','convnext'] if encoder == 'mae640' else ['convnext']):
        if (base/f'{family}_encoder.pt').is_file():
            continue
        for row in rows:
            if 'conversion' in row['groups'] and f'/encoder_sources/{family}/' in row['path']:
                download(storage, row)
        subprocess.run([conversion_python, str(ROOT/'scripts/convert_ot_encoders.py'),
            '--root', str(storage), '--family', family, '--output-dir', str(base)], check=True)
    release = json.loads((ROOT/'old/checkpoints/pretraining_manifest.json').read_text())['priming_b_160']
    download(storage, dict(kind='hub_file', path=release['checkpoint'], bytes=release['checkpoint_bytes'],
        sha256=release['checkpoint_sha256'], repo_id='jaykim1102/priming',
        revision='9dc5fe94583b4479edb954327110850dc748ce44', filename=release['checkpoint']))


def prepare(args):
    from scripts.prepare_feature_encoder import prepare as prepare_encoder
    from utils.files import sha256
    storage = args.storage
    if args.download:
        download_assets(storage, args.encoder, args.evaluation, args.conversion_python)
    prepare_encoder(storage, args.encoder, args.encoder_checkpoint, args.encoder_sha256,
        device=args.device, recalibrate=args.recalibrate)
    config = recipe(args.encoder, storage, evaluation=args.evaluation)
    if args.evaluation == 'drive':
        import shutil
        from metrics.drive_evaluation import prepare_reference
        cache = storage/'assets/drive_eval'
        cache.mkdir(parents=True, exist_ok=True)
        for argument, field in ((args.fid_reference,'fid_reference'),(args.prc_images,'prc_images')):
            target = Path(config['evaluation'][field])
            if argument and argument.resolve() != target.resolve():
                if target.exists() and sha256(target) != sha256(argument):
                    raise ValueError(f'Reference already exists with different content: {target}')
                if not target.exists():
                    shutil.copyfile(argument, target)
            if not target.is_file():
                raise FileNotFoundError(f'Provide --{field.replace("_","-")} for {target}')
        weights = storage/'assets/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'
        prepare_reference(config['evaluation'], weights, device=args.device)
    print(f'Encoder ready: {args.encoder}. Use train with --evaluation {args.evaluation}.')


def configuration(args):
    if args.resume:
        record = json.loads((args.output/'run.json').read_text())
        config = record['config']
        if config['matching']['stack'] != ENCODERS[args.encoder]:
            raise ValueError('Resume encoder differs from saved experiment')
        # Preserve the entire scientific configuration and W&B run identity.
        return config
    config = recipe(args.encoder, args.storage, epochs=args.epochs,
                    eval_every=args.eval_every_epochs, evaluation=args.evaluation)
    if args.evaluation == 'drive' and not args.dry_run:
        from utils.files import sha256
        config['evaluation']['reference_sha256'] = {key:sha256(Path(config['evaluation'][key]))
            for key in ('fid_reference','prc_images','prc_reference')}
    if args.wandb_project:
        config['tracking'] = dict(project=args.wandb_project, name=args.run_name or f'B16-{args.encoder}-post{args.epochs}',
            id=uuid.uuid4().hex[:16])
        if args.wandb_entity:
            config['tracking']['entity'] = args.wandb_entity
    return config


def train(args):
    import yaml
    from posttrain import load_config
    config = configuration(args)
    recipe_path = args.storage/'recipes'/(hashlib.sha256(str(args.output).encode()).hexdigest()[:16]+'.yaml')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
        str(ROOT/'posttrain.py'), '--config', str(recipe_path), '--output', str(args.output),
        '--hf-home', str(args.storage/'assets/huggingface'), '--torch-home', str(args.storage/'assets/torch')]
    if args.resume:
        command.append('--resume')
    if args.smoke_steps:
        command.extend(['--smoke-steps', str(args.smoke_steps)])
    # Output must not exist until posttrain.prepare_run creates it.
    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        print(shlex.join(command))
        return
    if not args.resume and args.output.exists():
        raise FileExistsError('Output already exists; choose a new output or use --resume')
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(yaml.safe_dump(config, sort_keys=False))
    load_config(recipe_path)
    from utils.paths import configure_caches
    from utils.posttrain_assets import evaluation_assets
    homes = configure_caches(hf_home=args.storage/'assets/huggingface', torch_home=args.storage/'assets/torch')
    evaluation_assets(config['evaluation'], hf_home=homes[0], torch_home=homes[1])
    required = [Path(config['data']['data_path'])/'imagenet-latents-images/state.json',
                Path(config['initialization']['checkpoint']), Path(config['data']['encoder_cache'])/'calibration.json']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; run prepare first')
    os.execv(sys.executable, command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('prepare','train'):
        p = commands.add_parser(name)
        p.add_argument('--encoder', choices=ENCODERS, required=True)
        p.add_argument('--storage', type=lambda p:Path(p).expanduser().resolve(), required=True)
        p.add_argument('--evaluation', choices=['native','drive'], default='native')
        if name == 'prepare':
            p.add_argument('--download', action='store_true', help='Download pinned assets (large ImageNet dataset)')
            p.add_argument('--conversion-python', default=sys.executable, help='Python 3.11 / torch 2.6 CPU for byte-identical MAE/ConvNeXt conversion')
            p.add_argument('--encoder-checkpoint', type=Path)
            p.add_argument('--encoder-sha256')
            p.add_argument('--recalibrate', action='store_true')
            p.add_argument('--device', default='cuda', choices=['cuda'])
            p.add_argument('--fid-reference', type=Path)
            p.add_argument('--prc-images', type=Path)
        else:
            p.add_argument('--output', type=lambda p:Path(p).expanduser().resolve(), required=True)
            p.add_argument('--epochs', type=positive, default=40)
            p.add_argument('--eval-every-epochs', type=positive, default=5)
            p.add_argument('--wandb-project')
            p.add_argument('--wandb-entity')
            p.add_argument('--run-name')
            p.add_argument('--resume', action='store_true')
            p.add_argument('--smoke-steps', type=positive)
            p.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    (prepare if args.action == 'prepare' else train)(args)


if __name__ == '__main__':
    main()
