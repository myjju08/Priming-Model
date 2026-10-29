# Original Priming with variable input noise and no timestep conditioning

This experiment keeps the original 115,063,936-parameter Priming B/16 network:
16 transformer blocks, width 768, eight class prefix tokens, RMSNorm, RoPE,
SwiGLU and the existing residual scales. It creates no timestep embedding and
no AdaLN modules. The image sequence still contains 256 image patches plus
eight class tokens. Existing original Priming tensor names, shapes,
initialization and generator arithmetic are preserved.

The selected recipe is
`configs/pretrain/b10_diffusion_no_time_perceptual.yaml`. Its explicit
`experiment.conditioning: none` and `model.timestep_conditioning: false`
distinguish this run from the earlier time-conditioned and AdaLN experiments.
Those experiments and their checkpoints are separate; this run starts from
scratch.

## Pretraining objective

For every training image, independently sample a Gaussian noise image and
`t ~ Uniform[0,1)`, then form

```text
x_t = (1-t) * x0 + t * epsilon
prediction = G(x_t, class)
pixel_loss = mean((prediction - x0)**2)
```

The sampled scalar is used only to corrupt the input. It is never forwarded to
the network, appended as a token, or embedded into a hidden feature. The model
directly predicts the clean RGB image. Each prediction uses one network call;
there is no iterative reverse diffusion or epsilon-to-x0 conversion.

Pixel loss uses `norm_p=0`, so it is ordinary MSE with uniform weighting over
the sampled noise levels. The inherited output perceptual losses retain their
coefficients, frozen encoders, paired crop/preprocessing and detached residual
normalization:

```text
per-image loss = pixel_MSE
    + 0.4 * LPIPS / (stop_gradient(LPIPS) + 0.01)
    + 0.1 * ConvNeXt_distance / (stop_gradient(ConvNeXt_distance) + 0.01)
```

The target of all three terms is the same clean training image. Gradients flow
from perceptual features into the predicted image while encoder parameters
stay frozen. `perceptual_threshold=1.0` applies both auxiliary terms to every
sampled noise level, including any explicit t=1 diagnostic case. The original
fixed-t=0.8 recipe also applied its auxiliary losses to every training example.

This changes the original fixed-noise training distribution and its adaptive
pixel weighting. It does not change the generator architecture. With plain
MSE alone, the unrestricted optimum is `E[x0 | x_t, class]`, with the unknown
noise level marginalized out; it is not a separately conditioned optimum for
each t. Adding the perceptual terms further changes the optimum. Training this
mixture does not guarantee a class-mean output, good samples at pure noise, or
improved post-training quality.

## Training and handoff

Pretraining uses ImageNet RGB256, 10 epochs at 1,251 optimizer updates per epoch
(12,510 updates), global batch 1,024, BF16, seed 1, constant Muon learning rate
0.001 and EMA 0.9999. The selected configuration uses eight GPUs, microbatch
64 per GPU and two accumulation steps. Compilation remains enabled; the eager
FLOPs pass is disabled. GPU smoke/benchmark results determine whether this
specific hardware configuration is executable.

The complete ten-epoch online and EMA checkpoint initializes the existing
supervised ResNet-50 OT post-training recipe for 40 epochs (100,120 updates).
Post-training starts a fresh AdamW optimizer and keeps all generator weights
trainable. The same generator receives independent pure noise:

```text
z ~ Normal(0, I)
generated_image = G(z, class)     # 1 NFE, no timestep argument
```

`generation_t=1.0` is retained as handoff metadata but does not supply a hidden
time input when timestep conditioning is disabled. The endpoint is defined by
the pure-noise input itself. Feature encoder, feature calibration, OT loss,
evaluation references and post-training schedule remain those selected by the
managed experiment plan. W&B uses project `CVPR - Priming Model` and a new run.

## Diagnostics and checks

`model.log_noise_metrics=true` reports `noise_t_mean` and
`perceptual_active_fraction` independently of network conditioning. This flag
adds no parameters, randomness or loss operations. Its default is false, so
original recipes retain their existing metric dictionaries.

Validation panels use fixed noisy images at corruption fractions
0.2, 0.5, 0.8 and 1.0 for online and EMA weights. For this recipe, the network
receives no t argument; panel rows mark the corruption level as **unobserved**,
and the JSON report states `conditioning: none`. Diagnostic randomness,
module modes and online/EMA weights are restored afterward. Paired-target MSE
at pure noise and small-sample noise sensitivity do not measure generation
quality; use the existing post-training evaluation.

CPU/meta validation:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest \
  tests.test_no_time_conditioning tests.test_noise_conditioning \
  tests.test_diffusion_pretrain -v
```

These checks cover original schema/RNG/forward equivalence, exact parameter
count, per-image corruption without time forwarding, ordinary pixel MSE plus
the actual perceptual loss wrapper with frozen encoder stubs, all-t masks,
one-call pure-noise generation, diagnostic state restoration and full-graph
CPU compilation. They do not establish GPU memory use, throughput, learned
image quality or convergence.
