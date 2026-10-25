"""Portable B/16 encoder-comparison recipes; no GPU imports."""
import math
from pathlib import Path
from utils.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = {'mae640': 'convnext_full_mae', 'supcon': 'convnext_supcon_tuned',
            'spark': 'convnext_spark_pretrained'}


def positive(value):
    value = int(value)
    if value < 1:
        raise ValueError('Epoch counts must be positive integers')
    return value


def validate_schedule(config):
    exp, train, matching = config['experiment'], config['train'], config['matching']
    steps = math.ceil(1281167 / 512)
    epochs, interval = positive(exp['epochs']), positive(exp['eval_every_epochs'])
    if interval > epochs:
        raise ValueError('Evaluation interval exceeds training duration')
    if (exp['dataset_size'], exp['steps_per_epoch']) != (1281167, steps):
        raise ValueError('Unexpected dataset size / epoch definition')
    if (train['steps'], train['save_every'], config['evaluation']['every']) != (epochs*steps, interval*steps, interval*steps):
        raise ValueError('Epoch, checkpoint and evaluation schedules differ')
    if matching['stack'] not in ENCODERS.values():
        raise ValueError('Unknown feature encoder')


def recipe(encoder, storage, *, epochs=40, eval_every=5, evaluation='native'):
    if encoder not in ENCODERS or evaluation not in ('native', 'drive'):
        raise ValueError('Unknown encoder or evaluation protocol')
    storage = Path(storage).expanduser().resolve()
    c = load_yaml(ROOT/'configs/posttrain/_base.yaml')
    c['experiment'] = dict(profile='encoder_comparison', encoder=encoder,
        dataset_size=1281167, steps_per_epoch=2503, epochs=positive(epochs),
        eval_every_epochs=positive(eval_every), epoch_definition='ceil(dataset_size/global_batch_size)')
    c['train'].update(steps=2503*epochs, global_batch_size=512, save_every=2503*eval_every)
    c['matching'].update(stack=ENCODERS[encoder], samples_per_class=32,
        real_references_per_class=32, compile_encoders=True, sinkhorn_cuda_graph=True,
        mae_bf16_channels_last=encoder=='mae640')
    c['data'].update(data_path=str(storage/'data/imagenet'),
        encoder_cache=str(storage/'assets/ot'/('representations' if encoder=='mae640' else encoder)))
    c['initialization']['checkpoint'] = str(storage/'checkpoints/pretraining/priming_b_160.pth')
    c['evaluation'].update(every=2503*eval_every, batch_size=32, prc_batch_size=1000,
        stats_dir=str(storage/'assets/fid_stats'),
        prc_reference=str(storage/'data/prc/VIRTUAL_imagenet256_labeled_inception.pt'))
    if evaluation == 'drive':
        cache = storage/'assets/drive_eval'
        c['evaluation'].update(protocol='drive_fid_is_pr', real_num_images=10000,
            fid_reference=str(cache/'imagenet_256_fid_stats.npz'),
            prc_images=str(cache/'imagenet_val_prc_arr0.npz'),
            prc_reference=str(cache/'real_inception_10000.pt'))
    validate_schedule(c)
    return c
