"""Fresh post-training initialization and exact all-rank recovery."""
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from .checkpoint import load_weights
from .ema import EMA
from .optim import create_optimizer


def validate_timestep_contract(source, config):
    """Check explicit conditioning metadata and online/EMA tensor presence."""
    contract = source.get('contract') or {}
    if not isinstance(contract, dict) or not isinstance(contract.get('model', {}), dict):
        raise ValueError('Invalid pretraining model contract')
    source_model = contract.get('model', {})
    expected = config['model'].get('timestep_conditioning', False)
    actual = source_model.get('timestep_conditioning', False)
    if not isinstance(expected, bool) or not isinstance(actual, bool):
        raise ValueError('Checkpoint and config timestep_conditioning flags must be boolean')
    if actual != expected:
        raise ValueError('Pretraining timestep_conditioning differs: explicitly select the matching post-training architecture')
    source_t = source_model.get('generation_t')
    if actual and (isinstance(source_t, bool) or not isinstance(source_t, (int, float)) or source_t != 1.):
        raise ValueError('Time-conditioned pretraining must record generation_t=1 for pure-noise post-training')
    target_t = config['model'].get('generation_t', 1.)
    if isinstance(target_t, bool) or not isinstance(target_t, (int, float)) or target_t != 1.:
        raise ValueError('Post-training requires generation_t=1 for pure-noise inputs')
    selected = dict(online=source.get('model', {}),
                    ema=source.get('model_ema', {}).get('shadows', {}).get('0.9999', {}))
    for label, tensors in selected.items():
        if not isinstance(tensors, dict):
            raise ValueError(f'Invalid pretraining {label} tensor dictionary')
        has_timestep_weights = any(name.startswith('net.noise_t_embedder.') for name in tensors)
        if has_timestep_weights != actual:
            raise ValueError(f'Pretraining {label} timestep tensors disagree with its architecture contract')


def validate_diffusion_pretraining_contract(source):
    """Recognize only the explicit x0/linear/uniform 10-epoch diffusion profile.

    This is the sole exception to the legacy perceptual-pretraining requirement.
    The scientific recipe is duplicated in config and contract intentionally;
    both must agree before nonperceptual initialization is permitted.
    """
    contract = source.get('contract') or {}
    saved_config = source.get('config') or {}
    experiment = contract.get('experiment')
    configured_experiment = saved_config.get('experiment')
    if experiment is None and configured_experiment is None:
        if contract.get('model', {}).get('timestep_conditioning', False):
            raise ValueError('Time-conditioned initialization requires an explicit diffusion_pretraining contract')
        return False
    if (not isinstance(experiment, dict) or experiment != configured_experiment
            or experiment.get('profile') != 'diffusion_pretraining'):
        raise ValueError('Diffusion experiment must agree in checkpoint config and contract')
    required = dict(prediction_target='x0', noise_schedule='linear', loss_weighting='uniform')
    if any(experiment.get(key) != value for key, value in required.items()):
        raise ValueError('Diffusion initialization requires linear-path x0 prediction with uniform loss weighting')
    for section in ('model', 'data', 'loss', 'train', 'optimizer'):
        if not isinstance(contract.get(section), dict) or contract[section] != saved_config.get(section):
            raise ValueError(f'Diffusion checkpoint config and contract differ: {section}')
    model, data, loss, train = (contract[key] for key in ('model', 'data', 'loss', 'train'))
    fixed_model = dict(model='OST_B', patch_size=16, token_channels=3, tokenizer_patch_size=1,
                       class_tokens=8, guidance_tokens=0, interval_tokens=0, label_drop_prob=0.,
                       corruption='denoising', t_dist='uniform', noise_scale=1., rope_2d=True,
                       learned_pe=True, timestep_conditioning=True, generation_t=1.)
    if (any(model.get(key) != value for key, value in fixed_model.items())
            or model.get('timestep_conditioning') is not True
            or isinstance(model.get('generation_t'), bool)):
        raise ValueError('Diffusion initialization requires the time-conditioned B/16 pixel architecture')
    intervals = {'variable_noise': (0., 1.), 'fixed_noise_control': (.8, .8)}
    if (experiment.get('variant') not in intervals
            or (model.get('t_min'), model.get('t_max')) != intervals[experiment['variant']]):
        raise ValueError('Diffusion noise interval differs from its declared variant')
    if (data.get('img_size'), data.get('num_classes')) != (256, 1000) or model.get('input_space') not in (None, 'pixel'):
        raise ValueError('Diffusion initialization requires ImageNet RGB256')
    if (type(train.get('epochs')) is not int or type(train.get('steps_per_epoch')) is not int
            or (train['epochs'], train['steps_per_epoch']) != (10, 1251)):
        raise ValueError('Diffusion initialization requires the 10-epoch, 1251-update recipe')
    if type(source.get('step')) is not int or not 0 <= source['step'] <= train['epochs'] * train['steps_per_epoch']:
        raise ValueError('Diffusion checkpoint step is outside its training horizon')
    if loss.get('norm_p') != 0 or loss.get('norm_eps') != 1e-4:
        raise ValueError('Diffusion initialization requires plain x0 MSE, norm_p=0')
    auxiliary = loss.get('aux', {})
    enabled = auxiliary.get('enabled')
    if not isinstance(enabled, bool) or experiment.get('auxiliary') != ('perceptual' if enabled else 'none'):
        raise ValueError('Diffusion auxiliary loss differs from its declared experiment')
    if enabled:
        expected_aux = dict(lpips_weight=.4, convnext_weight=.1, norm_eps=.01,
                            norm_p=1., random_crop=True, crop_scale=[.08, 1.])
        if loss.get('perceptual_threshold') != 1. or any(auxiliary.get(key) != value for key, value in expected_aux.items()):
            raise ValueError('Optional diffusion perceptual supervision must preserve its declared weights and cover all noise levels')
    return True


def initialize_posttraining(model, path, optimizer_config, *, config):
    """Inherit online weights and EMA; deliberately start AdamW with no history."""
    source = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    validate_timestep_contract(source, config)
    state = source.get('model_ema', {})
    step = source.get('step')
    if not isinstance(step, int) or step < 0 or source.get('current_step') != step + 1:
        raise ValueError('Invalid pretraining counters')
    if (state.get('schedule'), state.get('schedules'), state.get('step_count')) != (
            'const_schedule', [('0.9999', .9999)], step):
        raise ValueError('Expected the original pretraining EMA and counter')
    if set(state.get('shadows', {})) != {'0.9999'}:
        raise ValueError('Expected one pretraining EMA shadow')
    diffusion = validate_diffusion_pretraining_contract(source)
    contract = source.get('contract')
    if config['model']['model'] == 'OST_B' and not isinstance(contract, dict):
        raise ValueError('B post-training requires a pretraining contract')
    if contract is not None:
        for section, keys in (('model', ('model','patch_size','class_tokens','guidance_tokens',
                                       'interval_tokens','label_drop_prob','rope_2d','learned_pe')),
                              ('data', ('num_classes',))):
            for key in keys:
                if contract.get(section, {}).get(key) != config[section][key]:
                    raise ValueError(f'Pretraining contract differs: {section}.{key}')
        if not diffusion and contract.get('loss', {}).get('aux', {}).get('enabled') is not True:
            raise ValueError('Post-training requires perceptual pretraining')
    if diffusion:
        if optimizer_config.get('optimizer') != 'adamw':
            raise ValueError('Diffusion post-training starts a fresh AdamW optimizer')
        if (getattr(model, 'timestep_conditioning', False) is not True
                or getattr(model, 'generation_t', None) != 1.
                or getattr(getattr(model, 'net', None), 'timestep_conditioning', False) is not True
                or getattr(model.net, 'generation_t', None) != 1.
                or any(not parameter.requires_grad for parameter in model.parameters())):
            raise ValueError('Diffusion post-training retains all trainable parameters and generation_t=1')
    expected = model.state_dict()
    shadow = state['shadows']['0.9999']
    if set(shadow) != set(expected):
        raise ValueError('Pretraining EMA tensor names differ')
    for name, value in shadow.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'Invalid pretraining EMA tensor: {name}')
    load_weights(model, path, stage='pretraining', weights='online')
    ema = EMA(model, decay=.999)
    ema.load_state_dict(state)
    return create_optimizer(model, optimizer_config), ema


def snapshot_rng():
    # Keep the historical keys and all-device CUDA states for checkpoint compatibility.
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state().clone(),
                cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['py'])
    np.random.set_state(state['np'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _rank_states(local, rank, world):
    if world == 1:
        return [local]
    states = [None] * world if rank == 0 else None
    dist.gather_object(local, states, dst=0)
    return states


def save_posttraining(path, model, optimizer, ema, matcher, stream, completed, *, config):
    rank, world = stream.rank, stream.world_size
    rngs = _rank_states(snapshot_rng(), rank, world)
    streams = _rank_states(stream.state_dict(), rank, world)
    if rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), model_ema=ema.state_dict(),
                       step=completed-1, current_step=completed, samples_seen=completed*stream.global_generated_count,
                       posttrain_config=config, posttrain_ema_origin=ema.step_count-completed, mmd_repulsion_queue_state=matcher.running_state_dict(),
                       mmd_rng_state=dict(version=1, world_size=world, rank_states=rngs),
                       sinkhorn_ot_stream_state=dict(version=2, world_size=world, rank_states=streams))
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, path)
    if world > 1:
        dist.barrier()


def resume_posttraining(path, model, matcher, stream, optimizer_config, *, config):
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    completed = state.get('current_step')
    if not isinstance(completed, int) or completed < 0 or state.get('step') != completed-1:
        raise ValueError('Invalid post-training counters')
    if state.get('samples_seen') != completed*stream.global_generated_count or state.get('posttrain_config') != config:
        raise ValueError('Post-training recipe or sample counter differs')
    matcher.load_running_state_dict(state['mmd_repulsion_queue_state'])
    payloads = [state['mmd_rng_state'], state['sinkhorn_ot_stream_state']]
    for payload, version in zip(payloads, (1, 2)):
        if (payload.get('version'), payload.get('world_size')) != (version, stream.world_size):
            raise ValueError('Post-training RNG version or world size differs')
        if len(payload.get('rank_states', [])) != stream.world_size:
            raise ValueError('Missing rank RNG states')
    states = payloads[1]['rank_states']
    first = states[0]['class_generator_state']
    if any(not torch.equal(item['class_generator_state'], first) for item in states):
        raise ValueError('Global class RNG differs across ranks')
    if any(item.get('version') != 2 or not isinstance(item.get('negative_noise_generator_state'), torch.Tensor) for item in states):
        raise ValueError('Missing private generated-reference RNG state')
    ema_state = state['model_ema']
    if ema_state.get('schedules') != [('0.999', .999)] or set(ema_state.get('shadows', {})) != {'0.999'}:
        raise ValueError('Expected post-training EMA 0.999')
    if (ema_state.get('schedule') != 'const_schedule' or
            ema_state.get('step_count') != state.get('posttrain_ema_origin', -1)+completed):
        raise ValueError('Post-training EMA counter differs')
    shadow = ema_state['shadows']['0.999']
    expected = model.state_dict()
    if set(shadow) != set(expected):
        raise ValueError('Post-training EMA tensor names differ')
    for name, value in shadow.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'Invalid post-training EMA tensor: {name}')
    # Load before creating the optimizer, since assignment replaces parameters.
    load_weights(model, path, weights='online')
    optimizer = create_optimizer(model, optimizer_config)
    optimizer.load_state_dict(state['optimizer'])
    ema = EMA(model, decay=.999)
    ema.load_state_dict(ema_state)
    stream.load_state_dict(states[stream.rank])
    restore_rng(payloads[0]['rank_states'][stream.rank])
    return optimizer, ema, completed
