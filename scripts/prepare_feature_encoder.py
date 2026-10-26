"""Calibrate an RN50 variant on the same 2,000 ImageNet rows as MAE640."""
import json
import shutil
import tempfile
from pathlib import Path


def resolve_checkpoint(cache, encoder, checkpoint=None, expected_sha=None):
    """Download and verify the input weights before conversion or calibration."""
    from models.encoders.rn50 import (SPARK_URL, SPARK_SHA256,
                                       SUPERVISED_URL, SUPERVISED_SHA256)
    from utils.files import sha256
    cache = Path(cache)
    if encoder == 'supervised':
        expected_sha = expected_sha or SUPERVISED_SHA256
    if checkpoint is None:
        if encoder == 'supcon':
            raise ValueError('SupCon requires --encoder-checkpoint /path/to/backbone.pth')
        if encoder == 'supervised':
            from torch.hub import download_url_to_file
            checkpoint = cache/'resnet50-0676ba61.pth'
            if not checkpoint.exists():
                download_url_to_file(SUPERVISED_URL, str(checkpoint),
                                     hash_prefix=SUPERVISED_SHA256, progress=True)
        elif encoder == 'spark':
            import gdown
            checkpoint = cache/'spark_official.pth'
            if not checkpoint.exists():
                gdown.download(SPARK_URL, str(checkpoint), quiet=False)
            expected_sha = expected_sha or SPARK_SHA256
        else:
            raise ValueError(f'Unknown RN50 encoder: {encoder}')
    checkpoint = Path(checkpoint).expanduser().resolve()
    identity = sha256(checkpoint)
    if expected_sha and identity != expected_sha:
        raise ValueError('Input checkpoint SHA256 mismatch')
    return checkpoint, identity


def median_distance(features, device, block=128):
    import numpy as np
    import torch
    n = len(features)
    if n < 2:
        raise ValueError('At least two calibration images required')
    distances = []
    # Each unordered pair exactly once; FP32 Euclidean distances, no TF32.
    norms = np.einsum('ij,ij->i', features, features, dtype=np.float32)
    for i in range(0, n, block):
        x = torch.from_numpy(np.array(features[i:i+block])).to(device)
        for j in range(i, n, block):
            y = torch.from_numpy(np.array(features[j:j+block])).to(device)
            squared = torch.from_numpy(norms[i:i+len(x)]).to(device)[:, None] + torch.from_numpy(norms[j:j+len(y)]).to(device)[None, :] - 2*x@y.T
            distance = squared.clamp_min_(0).sqrt_()
            if i == j:
                indices = torch.triu_indices(len(x), len(y), 1, device=device)
                distance = distance[indices[0], indices[1]]
            distances.append(distance.flatten().cpu())
    value = torch.quantile(torch.cat(distances), .5).item()
    if not 0 < value < float('inf'):
        raise ValueError('Degenerate/nonfinite feature scale')
    return value


def prepare(storage, encoder, checkpoint=None, expected_sha=None, *, device='cuda', recalibrate=False):
    import numpy as np
    import torch
    from datasets import load_from_disk
    from models.encoders.rn50 import RN50ConvNext, load_backbone, source_identity, SUPERVISED_SHA256
    from utils.experiments import ENCODERS
    from utils.files import sha256, write_json
    from utils.posttrain_assets import verify_references, verify_file
    storage = Path(storage)
    base = storage/'assets/ot/representations'
    if encoder not in ENCODERS:
        raise ValueError(f'Unknown encoder: {encoder}')
    if encoder == 'mae640':
        verify_references(base, full=True)
        return
    if encoder == 'supervised':
        expected_sha = expected_sha or SUPERVISED_SHA256
    cache = storage/'assets/ot'/encoder
    if device != 'cuda':
        raise ValueError('Calibration must use CUDA BF16 to match training precision')
    manifest_path = cache/'encoder_manifest.json'
    if manifest_path.exists() and not recalibrate:
        manifest = json.loads(manifest_path.read_text())
        expected = dict(family=encoder, source_sha256=source_identity(),
            calibration_sha256=sha256(cache/'calibration.json'),
            encoder_sha256=sha256(cache/f'{encoder}_encoder.pt'),
            convnext_sha256=sha256(cache/'convnext_encoder.pt'))
        if any(manifest.get(k) != v for k,v in expected.items()):
            raise ValueError('Stale encoder cache: use --recalibrate')
        if checkpoint and sha256(Path(checkpoint)) != manifest['input_sha256']:
            raise ValueError('Different checkpoint requested: use --recalibrate')
        if expected_sha and manifest['input_sha256'] != expected_sha:
            raise ValueError('Requested checkpoint SHA256 differs')
        print(f'Verified existing {encoder} calibration: {cache}')
        return
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint, identity = resolve_checkpoint(cache, encoder, checkpoint, expected_sha)
    verify_references(base, full=False)
    records = json.loads((Path(__file__).resolve().parents[1]/'old/releases/posttrain_encoders.json').read_text())['files']
    for name in ('calibration.json', 'convnext_encoder.pt'):
        verify_file(base/name, next(row for row in records if row['path'] == name))
    calibration = json.loads((base/'calibration.json').read_text())
    for name in ('class_order.npy', 'dataset_labels.npy', 'class_counts.npy', 'convnext_encoder.pt'):
        shutil.copyfile(base/name, cache/name)
    # Strict loading allows only the classification head to be discarded.
    torch.save(load_backbone(checkpoint).state_dict(), cache/f'{encoder}_encoder.pt')
    model = RN50ConvNext(cache, family=encoder, device=device)
    dataset = load_from_disk(str(storage/'data/imagenet/imagenet-latents-images'))
    selected = calibration['dataset_indices']
    if len(selected) != 2000 or len(set(selected)) != 2000:
        raise ValueError('Expected the released 2,000 distinct calibration rows')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    arrays = {}
    with tempfile.TemporaryDirectory(prefix='priming-calibration-', dir=cache) as scratch:
        with torch.inference_mode():
            for start in range(0, len(selected), 8):
                rows = [dataset[int(i)]['image'].convert('RGB') for i in selected[start:start+8]]
                if any(image.size != (256,256) for image in rows):
                    raise ValueError('Calibration requires released RGB256 images')
                images = torch.from_numpy(np.stack([np.asarray(im).copy() for im in rows])).permute(0,3,1,2).to(device).float()/255
                features = model.encode_family(images, encoder)
                for key, value in features.items():
                    if not torch.isfinite(value).all():
                        raise ValueError(f'Nonfinite calibration feature: {key}')
                    value = value.float().cpu().numpy()
                    if key not in arrays:
                        arrays[key] = np.memmap(Path(scratch)/(key+'.bin'), dtype='float32', mode='w+', shape=(2000,value.shape[1]))
                    arrays[key][start:start+len(rows)] = value
                if start % 200 == 0:
                    print(f'Calibration {encoder}: {start+len(rows)}/2000', flush=True)
        for field in ('bandwidths', 'weights', 'dimensions'):
            calibration[field] = {k:v for k,v in calibration[field].items() if k.startswith('convnext__')}
        for key, array in arrays.items():
            calibration['bandwidths'][key] = median_distance(array, device)
            calibration['weights'][key] = .5/12
            calibration['dimensions'][key] = array.shape[1]
            print(f'{key}: scale={calibration["bandwidths"][key]:.6g}', flush=True)
        arrays.clear()
    calibration.update(encoder_source_sha256=source_identity(), rn50_compute='bf16' if str(device).startswith('cuda') else 'fp32',
        calibration_algorithm='median unordered Euclidean pair distance; FP32; no TF32')
    write_json(cache/'calibration.json', calibration)
    write_json(manifest_path, dict(family=encoder, source_sha256=source_identity(), input_sha256=identity,
        calibration_sha256=sha256(cache/'calibration.json'), encoder_sha256=sha256(cache/f'{encoder}_encoder.pt'),
        convnext_sha256=sha256(cache/'convnext_encoder.pt')))
    print(f'Prepared {encoder}: {cache}', flush=True)
