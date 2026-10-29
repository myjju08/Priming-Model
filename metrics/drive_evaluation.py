"""FP32 ADM Inception metrics using the user's Drive reference files.

Generated and real images are rounded RGB uint8 before feature extraction.
FID uses every supplied reference moment; P/R uses all 10,000 real images.
"""
from contextlib import nullcontext
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from metrics.fid import feature_statistics, fid_distance
from metrics.inception import compute_isc, compute_precision_recall
from models.encoders.adm import load_inception
from utils.files import sha256, write_json

LOG = logging.getLogger(__name__)


def select_weight_labels(ema_label, weight_labels=None):
    labels = ('online', ema_label) if weight_labels is None else weight_labels
    if (not isinstance(labels, (tuple, list)) or not labels or
            any(label not in ('online', ema_label) for label in labels) or
            len(set(labels)) != len(labels)):
        raise ValueError('weight_labels must select unique online/EMA labels')
    return tuple(labels)


def _check_rank_errors(error, context, stage):
    errors = [error]
    if context.world_size > 1:
        errors = [None] * context.world_size
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f'Drive evaluation {stage} failed across ranks: {errors}')


def prepare_reference(config, weights_path, device='cuda'):
    raw, cache = Path(config['prc_images']), Path(config['prc_reference'])
    identity = dict(images_sha256=sha256(raw), inception_sha256=sha256(weights_path),
                    count=10000, preprocessing='RGB uint8; ADM normalize=True; FP32; TF32=False')
    manifest = cache.with_suffix('.json')
    if cache.is_file() and manifest.is_file():
        previous = __import__('json').loads(manifest.read_text())
        if previous == dict(identity, features_sha256=sha256(cache)):
            return
        raise ValueError('Cached Drive reference has a different identity')
    with np.load(raw, allow_pickle=False) as data:
        images = data['arr_0']
    if images.shape != (10000, 256, 256, 3) or images.dtype != np.uint8:
        raise ValueError(f'Expected all 10,000 uint8 RGB reference images, got {images.shape} {images.dtype}')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    encoder = load_inception(weights_path, device=device)
    encoder.normalize = True
    parts = []
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        for start in range(0, len(images), 32):
            batch = torch.from_numpy(images[start:start+32]).permute(0, 3, 1, 2).to(device)
            features, _ = encoder(batch)
            parts.append(features.cpu())
    features = torch.cat(parts)
    if features.shape != (10000, 2048) or not torch.isfinite(features).all():
        raise ValueError('Incomplete/nonfinite reference features')
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix('.partial')
    torch.save(features, temporary)
    temporary.replace(cache)
    write_json(manifest, dict(identity, features_sha256=sha256(cache)))


@torch.inference_mode()
def evaluate(model, ema, config, sampling, weights, *, weight_labels=None, progress_every=0, seed=None):
    from eval import partition, gather_rows
    from utils.collectives import DistributedContext
    device = next(model.parameters()).device
    context = DistributedContext('auto')
    selected = select_weight_labels(ema.label, weight_labels)
    if type(progress_every) is not int or progress_every < 0:
        raise ValueError('progress_every must be a nonnegative integer')
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    error = None
    try:
        encoder = load_inception(weights['FID(ADM)'], device=device)
        encoder.normalize = True
        with np.load(config['fid_reference'], allow_pickle=False) as archive:
            reference = dict(mu=archive['ref_mu'], sigma=archive['ref_sigma'])
            fid_real_count = int(archive['num_samples'])
        real = torch.load(config['prc_reference'], map_location='cpu', weights_only=True) if context.rank == 0 else None
        if real is not None and real.shape != (10000, 2048):
            raise ValueError('Precision/recall requires exactly 10,000 Drive reference features')
    except Exception as exc:
        error = repr(exc)
    _check_rank_errors(error, context, 'setup')
    if seed is not None:
        from utils.random import seed_all
        seed_all(seed + context.rank)
    total = config['num_images']
    start, end = partition(total, context.world_size, context.rank)
    labels = torch.arange(start, end).remainder(1000)
    args = SimpleNamespace(interval_min=sampling['t_min'], interval_max=sampling['t_max'], same_noise=False)
    rows = []
    for label in selected:
        parts, logits = [], []
        model.eval()
        error = None
        try:
            with nullcontext() if label == 'online' else ema.swap(model):
                batches = labels.split(config['batch_size'])
                for index, classes in enumerate(batches, 1):
                    classes = classes.to(device)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        images = model.generate(len(classes), classes, cfg=sampling['cfg'], args=args, verbose=False)
                    pixels = ((images.float()+1)/2).clamp(0, 1).mul(255).round().to(torch.uint8)
                    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                    with torch.autocast('cuda', enabled=False):
                        features, scores = encoder(pixels)
                    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
                    parts.append(features.cpu())
                    logits.append(scores.cpu())
                    if context.rank == 0 and progress_every and (index % progress_every == 0 or index == len(batches)):
                        LOG.info('Drive %s generation: rank 0 batch %d/%d, local images %d/%d',
                                 label, index, len(batches), min(index*config['batch_size'], end-start), end-start)
        except Exception as exc:
            error = repr(exc)
        _check_rank_errors(error, context, 'generation')
        features = gather_rows(torch.cat(parts).to(device), context)
        scores = gather_rows(torch.cat(logits).to(device), context)
        message = [None]
        if context.rank == 0:
            try:
                if progress_every:
                    LOG.info('Drive %s: gathered %d images; computing FID, IS and precision/recall', label, total)
                if len(features) != total or len(scores) != total:
                    raise ValueError('Missing generated evaluation images')
                score, score_std = compute_isc(scores.cpu())
                fid = fid_distance(feature_statistics(features.cpu()), reference)
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                try:
                    precision, recall = compute_precision_recall(real.to(device), features.float(),
                        k=config['prc_k'], batch_size=config['prc_batch_size'])
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
                message[0] = dict(ema_label=label, cfg=sampling['cfg'], num_samples=total,
                    real_num_samples=len(real), fid_real_num_samples=fid_real_count,
                    fid=fid, **{'is':score}, is_std=score_std, precision=precision, recall=recall,
                    protocol='Drive references / FP32 ADM Inception / rounded uint8')
            except Exception as exc:
                message[0] = dict(error=repr(exc))
        if context.world_size > 1:
            dist.broadcast_object_list(message, src=0)
        if 'error' in message[0]:
            raise RuntimeError(message[0]['error'])
        rows.append(message[0])
    return rows
