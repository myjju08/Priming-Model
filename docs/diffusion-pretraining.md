# Diffusion denoising pretraining followed by one-step distribution matching

This experiment trains a time-conditioned Priming B/16 from scratch for ten
epochs, then reuses the same architecture and checkpoint for the existing OT
post-training objective. It is an explicit linear-noise, clean-image-prediction
experiment, not a claim to reproduce an external diffusion implementation.

For each training image, independently sample `t ~ Uniform(0, 1)` and Gaussian
noise `epsilon`. Construct `x_t = (1-t) * x0 + t * epsilon`, predict
`D(x_t, t, class)`, and minimize the mean squared error against `x0`:

```text
alpha(t) = 1-t, sigma(t) = t
prediction_target = x0
w(t) = 1
loss = mean((D(x_t, t, class) - x0)**2)
```

One network forward predicts the denoising target at a sampled noise level;
there is no iterative sampling trajectory during this objective. This is not a
variance-preserving DDPM schedule: `alpha(t)**2 + sigma(t)**2` need not equal one.
The implementation keeps Priming's existing linear corruption convention.

The original pretraining recipes retain fixed `t=0.8`, residual-adaptive pixel
MSE, and their perceptual settings. The new default sets `loss.norm_p=0` so the
pixel objective is ordinary MSE, rather than a residual-dependent adaptive
gradient. It disables auxiliary perceptual losses for the diffusion-only run.

## Matched comparisons

| Recipe | Noise fraction | Auxiliary output losses |
| --- | --- | --- |
| `configs/pretrain/b10_diffusion.yaml` | Uniform `[0,1)` per image | None; selected default |
| `configs/pretrain/b10_fixed_noise_control.yaml` | Fixed `0.8` | None |
| `configs/pretrain/b10_diffusion_perceptual.yaml` | Uniform `[0,1)` per image | LPIPS `0.4`, ConvNeXt `0.1` |
| `configs/pretrain/b10_fixed_noise_control_perceptual.yaml` | Fixed `0.8` | LPIPS `0.4`, ConvNeXt `0.1` |

Every recipe uses the same timestep-conditioned architecture, clean-image MSE,
seed, batch, optimizer, and number of updates. Within each auxiliary-loss pair,
only the noise interval and its experiment label change. The fixed-noise control
is intentionally different from the untouched original Priming recipe: it also
has the timestep module and ordinary MSE. Thus a direct comparison to an old
Priming run changes more than the noise schedule.

The perceptual variants retain the original auxiliary preprocessing, adaptive
normalization, random crops, and coefficients. They set
`perceptual_threshold=1.0`: original fixed `t=0.8` applies perceptual supervision
to every image, whereas leaving the threshold at `0.8` under uniform noise would
silently exclude approximately 20% of images. Perceptual features are computed
from the predicted clean **RGB image**, not intermediate transformer features.
This repository's B/16 experiment is pixel-space, not latent-space.

These recipes do not implement the separate proposed `Diffusion + MAE`
pretraining ablation. MAE as a post-training feature encoder already exists, but
that is a different use from adding MAE feature matching during pretraining.
Neither auxiliary perceptual setting is labelled as MAE or REPA.

## Architecture and post-training boundary

`timestep_conditioning=True` adds a sinusoidal embedding/MLP of `1000*t` to the
existing class-token conditioning. It adds no sequence tokens. The new B/16 has
115,851,904 parameters. With the flag absent, baseline parameter names, shapes,
initialization RNG, and forward arithmetic are unchanged.

Post-training uses independent `z ~ Normal(0, I)` and
`G(z, c) = D(z, noise_t=1, c)`. The `generation_t=1.0` configuration supplies this
value when a generation call omits `noise_t`. There is no clean-image mixture,
iterative denoising, or clean-output conversion in this path; the network already
predicts `x0`. No division by `alpha(1)=0` occurs.
The older `sampling.t_min`/`sampling.t_max` fields describe CFG intervals; they
do not override this explicit noise-conditioning endpoint.

The timestep **input** is fixed. All network parameters, including timestep
embedding weights, remain trainable. The existing post-training code computes
features from generated images with gradients, and uses frozen real/reference
features plus an independent generated reference batch without gradients. The
existing featurewise OT attraction-minus-repulsion loss and optimizer schedule
are retained. One generated image needs one network call (1 NFE); the separate
reference batch is another batch of one-call generations.

Pretraining checkpoint loading validates the declared conditioning architecture,
objective, online weights, and EMA tensors. Diffusion-only checkpoints are
accepted through a dedicated diffusion contract; the legacy perceptual contract
is not silently weakened. Online and EMA weights, including the timestep MLP,
are inherited, and post-training starts a fresh AdamW optimizer. A baseline
timestep-free checkpoint cannot be loaded by dropping or inventing parameters.

To adapt an existing resolved post-training YAML, set:

```yaml
model:
  # Retain all other fields of the chosen post-training recipe.
  timestep_conditioning: true
  generation_t: 1.0
initialization:
  checkpoint: /path/to/diffusion-run/checkpoints/step_0012510.pth
```

Pass the same path explicitly with `posttrain.py --init-checkpoint` so it is
treated as a custom initialization. Use the existing post-training `--config`
workflow, not a released-model inference shortcut that builds a timestep-free
architecture. Keep encoder, populations, optimizer, evaluation references,
post-training updates, and 1-NFE generation identical between comparisons.

## Ten-epoch training preparation

All recipes use RGB256 ImageNet, 1,251 updates/epoch, 10 epochs = **12,510 updates**,
8 GPUs, global batch 1,024, BF16, Muon LR 0.001, constant schedule, seed 1, and
EMA 0.9999. MSE-only recipes use microbatch 64/GPU and accumulation 2. The standard
perceptual recipes use microbatch 32/GPU and accumulation 4, which also fits the
initial **eager FLOPs diagnostic** on RTX 5090. That diagnostic's memory failure
at batch 64 does not establish the memory requirement of compiled training.

The `b10_diffusion_perceptual_compiled64.yaml` and matched
`b10_fixed_noise_control_perceptual_compiled64.yaml` recipes use microbatch 64,
accumulation 2, and `measure_flops=false`. They retain compilation and every
scientific loss setting. Use `PYTORCH_ALLOC_CONF=expandable_segments:True` and
validate full compiled forward/backward plus optimizer updates on the target
machine before selecting them. Both profiling and DDP warmup honor the configured
batch size. The effective batch and mean-loss objective remain the same; changing
microbatch boundaries changes the stateless noise/crop random realizations.
Both comparisons initialize from
scratch; the transferred `priming_b_10.pth` is not their initialization.

Validate schema and parameter count without GPUs:

```bash
CUDA_VISIBLE_DEVICES='' python pretrain.py \
  --config configs/pretrain/b10_diffusion.yaml --check-config
```

`scripts/run_diffusion_pretrain.py --plan /path/to/plan.json --check` checks assets,
the preceding job, and GPU process status without starting a run. Add `--wait`
instead to queue behind a completed predecessor and eight idle GPUs. An explicit
`predecessor_stop_receipt` also permits a deliberately interrupted predecessor:
its checkpoint, counter, and checksum must match the receipt. A stopped job is
never relabelled as completed. Without `--wait`, capacity blockers fail promptly.

The configured experiment chain runs these stages in order:

1. Two-update diffusion GPU smoke test, W&B disabled.
2. Fresh 12,510-update diffusion pretraining, with W&B.
3. Prepare/calibrate official supervised ResNet-50 ImageNet1K V1 features while
   retaining the existing ConvNeXt branch and evaluation references.
4. Two-update fixed-`t=1` OT smoke test from the final pretraining checkpoint,
   W&B disabled.
5. Fresh supervised ResNet-50 OT post-training for 40 epochs = 100,120 updates,
   with W&B and online/EMA evaluation every five epochs.

Smoke weights are never reused by production. Both post-training stages load
the same complete ten-epoch checkpoint with a pinned digest. The launcher
records stage receipts and checks source, configuration, checkpoint, and encoder
identities at the boundaries. Failed or partial runs require explicit recovery;
they are not automatically repeated. Use supervisor for the long-running chain.
The storage or runtime `STOP_REQUESTED` marker cancels waiting/prevents the next
stage; the storage marker also requests a checkpointed stop during training.

A plan can explicitly adopt a completed 25-update benchmark as its pretraining
smoke test using `prevalidated_pretrain_smoke`. The launcher checks checkpoint
SHA256, exact smoke configuration, source/data/asset manifest, counters, and finite
metrics before skipping that one smoke launch. Production still initializes from
scratch, and the post-training smoke test is retained. This avoids recompiling an
already validated setup solely to repeat two updates.

The production plan logs to W&B project `CVPR - Priming Model` with a separate
run ID. Tracking includes pixel MSE, optional auxiliary losses, mean sampled
noise level, gradient norm, and update time. Each epoch saves checkpoints and
fixed validation panels at `t=0.2, 0.5, 0.8, 1.0` for online and EMA weights.
Validation randomness and module modes are restored afterward. Paired-target MSE
at `t=1` is not an image-quality metric; noise sensitivity is a small diagnostic,
not a substitute for post-training FID/precision/recall.

CPU tests and meta-model checks do not establish full-model GPU memory use,
throughput, or 10-epoch convergence. The deferred GPU smoke test checks execution
before the production run. No quality guarantee follows from the denoising
conditional-mean optimum alone.
