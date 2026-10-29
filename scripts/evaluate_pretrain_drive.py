#!/usr/bin/env python3
"""Evaluate only the final no-time pretraining EMA with the existing Drive protocol."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOG = logging.getLogger(__name__)
FINAL_STEP = 12510
EMA_LABEL = '0.9999'


def validate_checkpoint(state):
    """Reject partial, differently conditioned, or relabelled EMA checkpoints."""
    from utils.posttrain_checkpoint import validate_diffusion_pretraining_contract, validate_timestep_contract
    if not validate_diffusion_pretraining_contract(state):
        raise ValueError('An explicit final diffusion pretraining checkpoint is required')
    config = state['config']
    experiment = config['experiment']
    if (experiment.get('conditioning') != 'none' or experiment.get('variant') != 'variable_noise'
            or experiment.get('auxiliary') != 'perceptual'):
        raise ValueError('Expected the no-time variable-noise perceptual pretraining experiment')
    validate_timestep_contract(state, config)
    if (type(state.get('step')) is not int or state['step'] != FINAL_STEP or
            state.get('current_step') != FINAL_STEP+1 or
            state.get('samples_seen') != FINAL_STEP*1024 or state.get('world_size') != 8 or
            state['contract'].get('world_size') != 8 or state['contract'].get('global_batch') != 1024):
        raise ValueError('Expected complete ten-epoch checkpoint counters at step 12510')
    ema = state.get('model_ema', {})
    if (set(ema.get('shadows', {})) != {EMA_LABEL} or ema.get('step_count') != FINAL_STEP or
            ema.get('schedule') != 'const_schedule' or ema.get('batch_size') != 1024 or
            ema.get('schedules') != [(EMA_LABEL, .9999)] or
            config['train'].get('ema_rates') != [.9999] or config['train'].get('ema_type') != 'const'):
        raise ValueError('Expected the original constant EMA 0.9999 with 12510 updates')
    return config


def validate_evaluation(config):
    evaluation = config['evaluation']
    expected = dict(protocol='drive_fid_is_pr', num_images=50000, batch_size=32,
                    real_num_images=10000, prc_k=3, prc_batch_size=1000)
    if any(evaluation.get(key) != value for key, value in expected.items()):
        raise ValueError('Expected the 50k Drive protocol, batch32, all10k real, k3 and PR batch1000')
    if config['sampling'] != dict(cfg=1., t_min=.1, t_max=1., num_steps=1):
        raise ValueError('Expected the existing CFG1 one-NFE sampling configuration')
    if config['model'].get('timestep_conditioning', False) is not False:
        raise ValueError('The evaluation configuration must use the original no-time model')


def validate_tensors(state, config):
    import torch
    from pretrain import make_model
    with torch.device('meta'):
        template = make_model(config, device='meta')
    if sum(p.numel() for p in template.parameters()) != 115063936:
        raise ValueError('Expected the original 115,063,936-parameter Priming B')
    expected = template.state_dict()
    for label, values in [('online', state['model']), (EMA_LABEL, state['model_ema']['shadows'][EMA_LABEL])]:
        if set(values) != set(expected):
            raise ValueError(f'{label} tensor names differ from the saved pretraining architecture')
        for name, value in values.items():
            if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or
                    value.dtype != expected[name].dtype or not torch.isfinite(value).all()):
                raise ValueError(f'Invalid {label} tensor: {name}')


def load_online_and_ema(model, state):
    from utils.ema import EMA
    model.load_state_dict(state['model'], strict=True)
    # EMA must clone trainable parameters before freezing: otherwise its shadow
    # aliases the online tensor and loading the shadow overwrites online weights.
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Construct the evaluation EMA before freezing model parameters')
    ema = EMA(model, decay=.9999, batch_size=1024)
    ema.load_state_dict(state['model_ema'])
    model.requires_grad_(False)
    model.eval()
    return ema


def validate_results(rows):
    if (len(rows) != 1 or rows[0].get('ema_label') != EMA_LABEL or
            rows[0].get('num_samples') != 50000 or rows[0].get('real_num_samples') != 10000):
        raise ValueError('Expected exactly one complete 50k EMA 0.9999 evaluation with all10k real samples')
    metrics = rows[0]
    if not all(math.isfinite(metrics[key]) for key in ('fid', 'is', 'is_std', 'precision', 'recall')):
        raise ValueError('Evaluation metrics must be finite')
    if not 0. <= metrics['precision'] <= 1. or not 0. <= metrics['recall'] <= 1.:
        raise ValueError('Precision and recall must lie in [0, 1]')


def rank_zero_call(function, dist):
    message = [None]
    if dist.get_rank() == 0:
        try:
            message[0] = dict(value=function())
        except Exception as exc:
            message[0] = dict(error=repr(exc))
    dist.broadcast_object_list(message, src=0)
    if 'error' in message[0]:
        raise RuntimeError(message[0]['error'])
    return message[0]['value']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'config', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hf-home')
    parser.add_argument('--torch-home')
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    from utils.paths import configure_environment, resolve
    checkpoint, config_path, output = (resolve(getattr(args, key), canonical=True)
                                      for key in ('checkpoint', 'config', 'output'))
    hf_home, torch_home = configure_environment(output, phase='pretrain-ema-evaluation',
                                              hf_home=args.hf_home, torch_home=args.torch_home, threads=4)
    import torch
    import torch.distributed as dist
    from metrics.drive_evaluation import evaluate
    from pretrain import make_model
    from utils.config import load_yaml
    from utils.distributed import initialize
    from utils.files import prepare_output_directory, sha256, write_json
    from utils.posttrain_assets import evaluation_assets
    rank, world = initialize('Pretraining EMA evaluation')
    started = time.monotonic()
    try:
        logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                            format='%(asctime)s %(message)s')
        torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        prepare_output_directory(output, rank, distributed=dist)

        def prepare():
            config = load_yaml(config_path)
            validate_evaluation(config)
            for key in ('fid_reference', 'prc_images', 'prc_reference'):
                config['evaluation'][key] = str(resolve(config['evaluation'][key], canonical=True))
            LOG.info('Verifying final checkpoint and pinned Drive reference assets')
            state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
            saved_config = validate_checkpoint(state)
            validate_tensors(state, saved_config)
            weights = {name: str(path) for name, path in evaluation_assets(
                config['evaluation'], hf_home=hf_home, torch_home=torch_home).items()}
            source_files = [ROOT/'pretrain.py', ROOT/'eval.py', Path(__file__)]
            for folder in ('models', 'metrics', 'utils'):
                source_files.extend(sorted((ROOT/folder).rglob('*.py')))
            record = dict(checkpoint=dict(path=str(checkpoint), bytes=checkpoint.stat().st_size,
                              mtime_ns=checkpoint.stat().st_mtime_ns, sha256=sha256(checkpoint), step=FINAL_STEP),
                          ema_label=EMA_LABEL, saved_pretraining_config=saved_config,
                          saved_pretraining_config_sha256=hashlib.sha256(
                              json.dumps(saved_config, sort_keys=True).encode()).hexdigest(),
                          evaluation_config_path=str(config_path), evaluation_config_sha256=sha256(config_path),
                          evaluation=config['evaluation'], sampling=config['sampling'],
                          assets=dict(reference_sha256=config['evaluation']['reference_sha256'],
                                      inception_sha256=sha256(weights['FID(ADM)'])),
                          source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in source_files},
                          seed=args.seed, rank_seed_rule='seed + rank, reset after model and Inception construction',
                          label_rule='global sample index % 1000; contiguous rank shards; gather in rank order',
                          generated_samples=50000, real_pr_samples=10000, world_size=world,
                          nfe=1, conditioning='none', noise='independent standard Gaussian',
                          precision=dict(generator='BF16 autocast, TF32 enabled',
                                         inception='FP32, normalize=True, TF32 disabled',
                                         pixels='clamp((x+1)/2,0,1)*255; round; RGB uint8',
                                         precision_recall='FP32, TF32 disabled, k=3, batch=1000'),
                          is_splits=10, is_shuffle_seed=2020,
                          runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                       python=sys.version, compile=config['train']['compile'],
                                       gpu=torch.cuda.get_device_name(),
                                       cudnn_benchmark=True, cudnn_deterministic=False),
                          note='Independent EMA-only noise draw; not the same random samples as a scheduled online-then-EMA evaluation.')
            write_json(output/'manifest.json', record)
            return dict(config=config, weights=weights, record=record)

        prepared = rank_zero_call(prepare, dist)
        error = None
        try:
            state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
            saved_config = validate_checkpoint(state)
            identity = prepared['record']['checkpoint']
            if (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) != (identity['bytes'], identity['mtime_ns']):
                raise ValueError('Checkpoint changed after rank-zero validation')
            model = make_model(saved_config)
            ema = load_online_and_ema(model, state)
            del state
            if prepared['config']['train']['compile']:
                model.net.compile(fullgraph=True)
        except Exception as exc:
            error = repr(exc)
        errors = [None] * world
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError(f'Evaluation model setup failed across ranks: {errors}')
        LOG.info('Evaluating only pretraining EMA %s: 50000 images, 1 NFE, seed %d + rank', EMA_LABEL, args.seed)
        metrics = evaluate(model, ema, prepared['config']['evaluation'], prepared['config']['sampling'],
                           prepared['weights'], weight_labels=(ema.label,), progress_every=20, seed=args.seed)

        def finish():
            validate_results(metrics)
            result = dict(prepared['record'], status='complete', metrics=metrics[0],
                          duration_seconds=time.monotonic()-started,
                          completed_utc=datetime.now(timezone.utc).isoformat())
            write_json(output/'results.json', result)
            LOG.info('Evaluation complete: %s', json.dumps(metrics[0], sort_keys=True))
            return str(output/'results.json')

        rank_zero_call(finish, dist)
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
