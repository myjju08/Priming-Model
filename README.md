# Priming Model: selectable feature encoders

B/16 post-training with **MAE640**, **SupCon-tuned ResNet-50**, **SparK ResNet-50**, or **supervised ImageNet-1K ResNet-50**. Choose an encoder using the CLI; training code does not need editing. Each variant retains the original ConvNeXt branch.

Based on [JaeyeonKim01/priming](https://github.com/JaeyeonKim01/priming), commit `9e887928427d68abd0a1222f10574b3f93924cd0`. Original license and vendored notices are retained. See [upstream documentation](UPSTREAM_README.md) for the original paper and pretraining workflows.

## Setup

Use Python 3.11+ and a working CUDA-enabled PyTorch/torchvision installation compatible with your GPU. RTX 50-series requires CUDA 12.8 or newer wheels. Keep the installed GPU stack; `requirements-experiments.txt` deliberately does not replace PyTorch.

```bash
pip install -r requirements-experiments.txt
hf auth login
# Optional, when using --wandb-project:
wandb login
```

Training uses **8 GPUs on one node**. The default recipe is RGB256 B/16, initialized from the online B/16 160-epoch pretraining checkpoint. It is pixel-space post-training: latent caches are not used here. Frozen encoders still propagate gradients to generated images.

## Prepare assets

Choose a storage root containing `data/`, `assets/`, and `checkpoints/`. An already prepared upstream storage root can be reused; new RN50 caches live separately and do not overwrite the MAE cache.

```bash
# Existing prepared assets:
python scripts/experiment.py prepare --encoder mae640 --storage /path/to/storage

# Official SparK weights are downloaded automatically and SHA256 verified:
python scripts/experiment.py prepare --encoder spark --storage /path/to/storage

# Official torchvision IMAGENET1K_V1 supervised RN50, also SHA256 pinned:
python scripts/experiment.py prepare --encoder supervised --storage /path/to/storage

# Your tuned SupCon checkpoint must be supplied locally:
python scripts/experiment.py prepare --encoder supcon --storage /path/to/storage \
  --encoder-checkpoint /path/to/backbone_final.pth
```

Add `--encoder-sha256 EXPECTED_HASH` to pin a custom checkpoint. Raw torchvision RN50 states and `state_dict`/`module` wrappers are supported; only the classification head can be discarded. Incompatible backbones fail strict loading.

For a fresh storage root, add `--download`. This downloads the pinned ImageNet release, initial checkpoint, and evaluation assets. ImageNet is large; ensure access rights and sufficient disk space. MAE/ConvNeXt source conversion requires a **separate Python 3.11 + torch 2.6 CPU environment** to reproduce release archive hashes; pass `--conversion-python /path/to/conversion-env/bin/python`. This environment needs torchvision, timm, einops, numpy, msgpack, and huggingface_hub. Do not downgrade the GPU training environment. Conversion is skipped when converted encoders already exist. See [asset details](docs/assets.md).

If the SupCon checkpoint is arriving from another server, [managed SupCon preparation](docs/managed-supcon.md) downloads and verifies the dataset and support assets independently, then queues calibration, a smoke check and training once the checkpoint and matching evaluation references are ready.

RN50 preparation recalibrates all 12 features on the same 2,000 training images as the released MAE calibration, using the median distance over all 1,999,000 unordered pairs. It requires a GPU and temporary disk space for FP32 feature arrays. ConvNeXt features, scales and weights are preserved. Cached checkpoint/calibration/source hashes are verified; use `--recalibrate` explicitly after changing the checkpoint or extractor.

## Run

```bash
python scripts/experiment.py train --encoder spark --storage /path/to/storage \
  --output /path/to/runs/spark-40ep --epochs 40 --eval-every-epochs 5 \
  --wandb-project 'CVPR - Priming Model'
```

Replace `spark` with `mae640`, `supcon`, or `supervised`. W&B is enabled only when a project is provided; add `--wandb-entity YOUR_TEAM` if needed. Run IDs are generated per fresh run and preserved on resume. Run under your scheduler or process manager for long jobs.

The `supervised` control uses torchvision `ResNet50_Weights.IMAGENET1K_V1` (`resnet50-0676ba61.pth`, SHA256 `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`). Only the RN50 weights, encoder identity, and corresponding calibration change from SupCon/SparK: the 12 stage features, RGB256 preprocessing without a crop, ConvNeXt branch, populations, optimizer, schedule, and selected evaluation protocol are identical. It does not use torchvision's `DEFAULT` alias. Its independent cache is `assets/ot/supervised`.

Use `--dry-run` to print the full recipe and launch command without accessing GPUs or creating a run directory. Use `--smoke-steps 2` for a short training check with actual assets; use a separate output for smoke checks.

| Setting | Value |
|---|---|
| Classes per global step | 16 |
| Positive real / independent generated / trainable generated per class | 32 / 32 / 32 |
| Global generated batch | 512 |
| Epoch definition | ceil(1,281,167 / 512) = 2,503 updates |
| Default post-training | 40 epochs = 100,120 updates |
| Checkpoint and evaluation | Every 5 epochs = 12,515 updates |
| Optimizer | Fresh AdamW, LR 1e-5, 125-update warmup |
| EMA | 0.999 |
| Feature family weight | Replacement encoder 50%, ConvNeXt 50% |
| PatchDrift separate control | Off |

MAE640 exposes 84 MAE features; RN50 variants expose 12 features (stage 1–4 spatial, mean, std). ConvNeXt contributes 13 features in all variants. RN50 uses RGB256 ImageNet normalization, BF16 activations, FP32 statistics, and no inference crop. The CUDA-graph Sinkhorn optimization is enabled. These are the requested 32/32/32 comparison settings, not the unchanged upstream production 64/64/128 population recipe.

## Evaluation protocols

The default `--evaluation native` uses the upstream six feature-distance metrics, FDr-6, IS, precision and recall with release references. Both online and EMA 0.999 generate 50,000 images at every milestone. It does **not** use a custom Drive reference.

To use the custom FID/IS/precision/recall protocol with 50,000 generated images per model and all 10,000 real PR images:

```bash
python scripts/experiment.py prepare --encoder spark --storage /path/to/storage \
  --evaluation drive --fid-reference /path/to/imagenet_256_fid_stats.npz \
  --prc-images /path/to/imagenet_val_prc_arr0.npz
python scripts/experiment.py train --encoder spark --storage /path/to/storage \
  --output /path/to/runs/spark-drive --evaluation drive \
  --wandb-project 'CVPR - Priming Model'
```

Supply downloaded reference files locally. FID stats require `ref_mu` (2048) and `ref_sigma` (2048×2048); PR images require `arr_0` with shape `(10000,256,256,3)` and dtype uint8. Preparation computes the 10,000 real Inception features and records hashes. Drive evaluation uses rounded uint8 pixels and FP32 ADM Inception with TF32 disabled. Compare runs using the **same protocol and reference hashes**. Results are saved as `evaluation/step_*.json` and logged to W&B.

## Resume

```bash
python scripts/experiment.py train --encoder spark --storage /path/to/storage \
  --output /path/to/runs/spark-drive --resume
```

Resume reads the saved configuration, protocol, schedule and W&B ID; fresh-run arguments do not change them. Online/EMA weights, optimizer, counters, rank RNG states and sampling streams are restored. There is no feature memory bank. Source or asset changes are rejected. Resume is intended for runs created by this checkout; older experiment checkpoints with different source provenance are not silently migrated. To stop at a completed update, create `STOP_REQUESTED` inside the run output. An abrupt kill resumes from the last completed checkpoint.

## Validation

```bash
PYTHONPATH=tests OMP_NUM_THREADS=2 python -m unittest \
  test_encoder_experiments test_custom_posttrain test_posttrain_runner test_sinkhorn_graph
```

Weights, datasets, credentials and run outputs are intentionally excluded from Git. Adding another encoder requires implementing the frozen extractor contract, registering its stack, and providing its own validated calibration; renaming checkpoint files alone is insufficient.
