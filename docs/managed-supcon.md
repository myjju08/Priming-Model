# Prepare a managed SupCon comparison

Use the existing `supcon` encoder recipe to replace the ResNet-50 weights while
preserving the SparK/supervised comparison settings. The ConvNeXt branch remains
unchanged. The recipe uses 32 real, 32 independent generated and 32 trainable
generated images per class, 40 epochs (100,120 updates), and evaluation every
5 epochs. Choose the same evaluation protocol and reference files as the
baseline runs; the native protocol and the custom Drive protocol are different.

## Download before the encoder arrives

Keep the GPU-compatible PyTorch installation and install
`requirements-experiments.txt`. Authenticate to Hugging Face using its normal
login command. The pinned dataset requires access to `jaykim1102/priming`.

```bash
python scripts/prepare_experiment_assets.py --group dataset \
  --storage /workspace/priming-storage \
  --status-file /workspace/priming-storage/status/dataset.json
python scripts/prepare_experiment_assets.py --group support \
  --storage /workspace/priming-storage \
  --status-file /workspace/priming-storage/status/support.json
```

Run these two commands as separate jobs under your process manager. Each uses
four download workers and verifies release sizes and SHA256 hashes. Transient
network failures are retried; access and checksum failures stop the job. Restarting
a job reuses verified files and resumes Hub transfers. `--verify-only` checks
existing files, and `--dry-run` lists the selected assets without downloading.

The dataset is about 258.3 GB and support files about 8.3 GB. Reserve additional
space for the two custom evaluation references, converted weights, calibration
scratch arrays and training checkpoints. Support includes the online B/16
160-epoch initialization, evaluation assets and ConvNeXt conversion inputs. It
does not download MAE conversion sources or a replacement SupCon checkpoint.

Convert ConvNeXt in a separate Python 3.11 / PyTorch 2.6 CPU environment; keep
the GPU environment unchanged:

```bash
/path/to/conversion/bin/python scripts/convert_ot_encoders.py \
  --root /workspace/priming-storage --family convnext
```

## Incoming checkpoint and evaluation references

Transfer the raw torchvision-compatible ResNet-50 backbone state dictionary to
`/workspace/priming-storage/incoming/supcon_resnet50.pth`. The existing loader
also accepts `state_dict` or `module` wrappers and an optional classification
head. A SupContrast archive containing `model`, `encoder.*` and `head.*` needs
an explicit backbone export before use.

Use an atomic final rename when transferring a file so readers cannot mistake
an interrupted transfer for a complete checkpoint:

```bash
mv supcon_resnet50.pth.partial supcon_resnet50.pth
```

For the existing Drive comparison, place the original files in
`STORAGE/assets/drive_eval/imagenet_256_fid_stats.npz` and
`STORAGE/assets/drive_eval/imagenet_val_prc_arr0.npz`. Compare their SHA256 hashes
with the baseline W&B run's `evaluation.reference_sha256` values. Preparation
computes the 10,000 real Inception features and records the source identities.
Do not substitute the native reference statistics.

## Launch and monitor

Follow the [main experiment commands](../README.md#run) with `--encoder supcon`,
`--evaluation drive`, an explicit W&B entity, and the existing project name.
Use a separate output directory for a two-update smoke run before the full run.
The smoke run should not carry the production W&B tracking identity.

`scripts/queue_supcon_experiment.py --plan /path/to/plan.json` provides this
sequence for a managed server: wait for verified downloads and the transferred
checkpoint, verify reference hashes, convert ConvNeXt if necessary, calibrate
SupCon, run the smoke check, then launch training. It records its current stage
in `STORAGE/status/supcon.json`. Keep the plan and all downloaded assets outside
Git. See the script's plan validation for required fields. A failed preparation
or smoke check stops the job; inspect its log before restarting it.

On Vast, run download and queue commands through supervisor with portal logging,
`stopasgroup=true` and `killasgroup=true`. These are background compute jobs and
need no public port or Caddy entry. Do not configure automatic restart on a
failed GPU smoke check. An existing production output is resumed only when it
has a valid saved run and checkpoint pointer; an incomplete output is retained
for diagnosis.

The instance filesystem may not survive recycle or destroy. Check the live
storage manifest and preserve checkpoints off-instance when needed.
