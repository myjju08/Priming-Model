"""Frozen RN50 feature variants with per-checkpoint calibration and provenance."""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet50

from models.encoders.full.encoders import PixelDriftingEncoders, make_convnext
from models.encoders.load import configure_full_execution
from utils.files import sha256
from utils.posttrain_assets import verify_file, verify_references

ROOT = Path(__file__).resolve().parents[2]
STACKS = {'convnext_supcon_tuned': 'supcon', 'convnext_spark_pretrained': 'spark'}
SPARK_URL = 'https://drive.google.com/uc?id=1H8605HbxGvrsu4x4rIoNr-Wkd7JkxFPQ'
SPARK_SHA256 = 'ada57a3bd57b72a0d75fd5cbb02469bc688de3ed02b9ecd77dc781ed274fbc3d'


def source_identity():
    files = [Path(__file__), ROOT/'models/encoders/full/encoders.py',
             ROOT/'models/encoders/full/vendor/convnext.py', ROOT/'scripts/prepare_feature_encoder.py']
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()


def load_backbone(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError('Expected an RN50 backbone state dictionary')
    for key in ('state_dict', 'module'):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if state and all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    state = {k: v for k, v in state.items() if k not in ('fc.weight', 'fc.bias')}
    model = resnet50(weights=None)
    model.fc = nn.Identity()
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite RN50 checkpoint tensor')
    return model


class RN50ConvNext(nn.Module):
    def __init__(self, cache, *, family, device='cuda', microbatch=8):
        super().__init__()
        if family not in STACKS.values() or microbatch < 1:
            raise ValueError('Expected supcon/spark and a positive microbatch')
        self.family = family
        self.family_names = (family, 'convnext')
        cache = Path(cache)
        self.backbone = load_backbone(cache/f'{family}_encoder.pt')
        self.convnext = make_convnext()
        self.convnext.load_state_dict(torch.load(cache/'convnext_encoder.pt', map_location='cpu',
            mmap=True, weights_only=True), strict=True, assign=True)
        self.microbatch = int(microbatch)
        self.to(device).eval().requires_grad_(False)

    def encode_family(self, images, family):
        if family == 'convnext':
            return PixelDriftingEncoders.encode_family(self, images, family)
        if family != self.family:
            raise ValueError(family)
        with torch.autocast(images.device.type, enabled=False):
            x = images.float()
            mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
            std = x.new_tensor([.229, .224, .225])[None, :, None, None]
            x = (x-mean)/std
            rows = {}
            # Same RGB256 input, no crop; identical stage features for both RN50 arms.
            with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=images.is_cuda):
                x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
                for i, stage in enumerate((self.backbone.layer1, self.backbone.layer2,
                                          self.backbone.layer3, self.backbone.layer4), 1):
                    x = stage(x)
                    y = x.float()
                    rows[f'{family}__stage{i}_spatial'] = y.flatten(1)
                    rows[f'{family}__stage{i}_mean'] = y.mean((2, 3))
                    rows[f'{family}__stage{i}_std'] = (y.var((2, 3), unbiased=False)+1e-6).sqrt()
        return rows

    def forward(self, images):
        rows = {}
        for family in self.family_names:
            pieces = [self.encode_family(p, family) for p in images.split(self.microbatch)]
            rows.update({k: torch.cat([p[k] for p in pieces]) for k in pieces[0]})
        return rows


class FrozenRN50Encoders(nn.Module):
    def __init__(self, cache, *, stack, device='cuda', microbatch=8, compile_encoders=False,
                 mae_bf16_channels_last=False, **unused):
        super().__init__()
        if stack not in STACKS or mae_bf16_channels_last:
            raise ValueError('RN50 stacks do not accept MAE execution flags')
        family, cache = STACKS[stack], Path(cache)
        verify_references(cache, full=False)
        records = json.loads((ROOT/'old/releases/posttrain_encoders.json').read_text())['files']
        verify_file(cache/'convnext_encoder.pt', next(r for r in records if r['path']=='convnext_encoder.pt'))
        manifest_path = cache/'encoder_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        self.calibration = json.loads((cache/'calibration.json').read_text())
        expected = dict(family=family, source_sha256=source_identity(),
            calibration_sha256=sha256(cache/'calibration.json'),
            encoder_sha256=sha256(cache/f'{family}_encoder.pt'),
            convnext_sha256=sha256(cache/'convnext_encoder.pt'))
        if any(manifest.get(k)!=v for k, v in expected.items()):
            raise ValueError('Encoder/calibration identity differs; run experiment.py prepare again')
        self.bandwidths, self.weights = self.calibration['bandwidths'], self.calibration['weights']
        self.names = tuple(sorted(self.bandwidths))
        if len(self.names)!=25 or set(self.names)!=set(self.weights):
            raise ValueError('Expected 12 RN50 and 13 unchanged ConvNeXt features')
        if any(not math.isfinite(v) or v<=0 for v in self.bandwidths.values()):
            raise ValueError('Invalid calibrated bandwidth')
        for f in (family, 'convnext'):
            if not math.isclose(sum(v for k,v in self.weights.items() if k.startswith(f+'__')), .5):
                raise ValueError('Each encoder family must receive weight 0.5')
        self.encoders = RN50ConvNext(cache, family=family, device=device, microbatch=microbatch)
        configure_full_execution(self.encoders, compile_encoders=compile_encoders)
        self.manifest_digest = sha256(manifest_path)
        self.stack = stack
        self.execution = dict(microbatch=microbatch, compile_encoders=compile_encoders,
                              rn50_precision='bf16', convnext_precision='fp32')
        self.eval().requires_grad_(False)

    @property
    def family_names(self): return self.encoders.family_names
    @property
    def microbatch(self): return self.encoders.microbatch
    def train(self, mode=True): super().train(False); return self
    def encode_family(self, images, family): return self.encoders.encode_family(images, family)
    def forward(self, images):
        rows = self.encoders(images)
        if set(rows)!=set(self.names): raise ValueError('Calibrated feature names differ')
        return rows
    def checkpoint_metadata(self):
        return dict(feature_stack=self.stack, manifest_sha256=self.manifest_digest,
                    encoder_execution=self.execution)
