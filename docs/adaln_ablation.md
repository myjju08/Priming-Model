# Fixed versus uniform timestep pretraining

This experiment compares two newly initialized **Priming B/16 backbones with
DiT-style timestep conditioning and adaLN-Zero**. It retains Priming's pixel
patch embedding, 16 blocks, width 768, 12 attention heads, RoPE, and class
tokens. It is not the unmodified latent-space DiT-B architecture.

| Setting | Fixed arm | Uniform arm |
| --- | --- | --- |
| Timestep | `t = 0.8` | Per-image `t ~ Uniform(0, 1)` |
| Corruption | `(1-t) * x + t * noise` | Same |
| Target | Clean image `x` | Same |
| Objective | Adaptive pixel reconstruction + 0.4 LPIPS + 0.1 ConvNeXt | Same |
| Perceptual mask | `t <= 0.8` | Same rule; high-noise samples above 0.8 omit perceptual terms |
| Paired perceptual crops | Original paired random crop | Same |
| Pretraining | 10 epochs, 12,510 optimizer updates | Same |
| Global batch / seed | 1,024 / 1 | Same |
| Optimizer / LR | Original Muon / constant 0.001 | Same |
| EMA | Constant 0.9999 | Same |

Both arms consume a timestep random draw even when the fixed arm returns 0.8.
This keeps subsequent noise draws aligned between arms. The same initialization
seed, data order, and stateless perceptual crop stream are used. Frozen LPIPS
and ConvNeXt loss networks use the existing downloaded weights.

This is a comparison of fixed versus variable noise-level reconstruction. It
does **not** introduce DDPM's variance schedule, epsilon-prediction loss, learned
variance, or iterative inference. Sampling and the later distributional
post-training use the model's pure-noise endpoint with explicit `t = 1`.

The new recipes disable the startup-only FLOPs diagnostic, which exceeded GPU
memory in the earlier experiment. This does not remove any optimizer update.
The original released B/L recipes and their validators remain separate.

## Prepare without starting training

Run from this worktree, after source changes and validation are finished:

```bash
/venv/main/bin/python scripts/prepare_adaln_ablation.py \
  --storage /workspace/priming-storage \
  --bundle /workspace/priming-ops/adaln-timestep-20261029 \
  --tag 20261029
```

This verifies local data/perceptual/reference asset identities and writes four
resolved YAML recipes, `plan.json`, a supervisor wrapper/config, and explicit
pretraining and post-training commands. It does not create training outputs,
start a service, create a W&B run, or arm subsequent post-training.

If source files change during preparation, regenerate with `--replace` before
either output exists. Once training begins, the pinned source, configuration,
data metadata, and asset hashes are immutable for resumption.

The prepared service has `autostart=false` and `autorestart=false`. Installation
and an explicit `supervisorctl start` are separate operations. Once explicitly
started, it trains the fixed arm followed by the uniform arm, each on eight
GPUs. It refuses to overlap other GPU compute processes. Both runs log to
`CVPR - Priming Model` with separate IDs and names.

Check the resolved plan without launching:

```bash
/venv/main/bin/python scripts/run_adaln_ablation.py \
  --plan /workspace/priming-ops/adaln-timestep-20261029/plan.json \
  --check-only
```

`--offline-check` may additionally skip W&B authentication for a local-only
check. A launch always checks authentication. Neither check performs GPU work.

## Short validation runs

Use a separate output directory for each validation run. The pretraining CLI
supports `--smoke-steps N` for an absolute update limit in these ablation
profiles, including when resuming. It still saves complete checkpoint state.
Use the exact recipes and asset paths shown in `plan.json`, with a distinct
validation output; do not use a production output for tests and then edit its
scientific configuration.

The launcher mirrors every completed update into W&B using `train/update` as
the plotted step. On resume it validates every rank's checkpoint cursor and
preserves an uncheckpointed metric tail before deterministic replay. Final
online and EMA tensors must be finite before the final checkpoint is linked
to its canonical path. Each arm gets a separate completion report.

For a graceful stop, create the prepared bundle's `STOP_REQUESTED` file.
Training saves at an update boundary and the second arm stays unstarted.
`CANCEL_REQUESTED` also prevents launching another arm. Before resuming,
remove the marker deliberately and start the same prepared service. A hard
supervisor stop may require replay from the latest completed checkpoint.

## Later five-epoch post-training

The prepared post-training recipes use the corresponding arm's ten-epoch
checkpoint, fresh AdamW, and the same supervised ResNet-50 V1 plus ConvNeXt
feature stack as the completed baseline. Each recipe runs **5 epochs = 12,515
updates** at batch 512, with one-step generation at `t = 1` and CFG 1.

Evaluation and checkpointing occur each epoch (2,503 updates), yielding five
comparison points. FID, IS, precision, and recall use the same Drive protocol:
50,000 generated images and 10,000 real precision/recall reference images,
reported separately for online and post-training EMA 0.999.

These commands are recorded in `plan.json` but are **not scheduled or started**
by the pretraining service. They require a later explicit launch. Use a
post-training process without the pretraining Python dependency overlay.
