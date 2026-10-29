# Original adaptive reconstruction with variable input noise

This fresh ten-epoch experiment restores the original Priming pixel-loss
normalization. The previous no-time experiment used ordinary pixel MSE
(`norm_p=0`); consequently, comparing it with original fixed-t=0.8 pretraining
changed both the noise distribution and the pixel-loss gradient. Its checkpoint
is a separate experiment and is not resumed for this correction.

The generator remains the original Priming B/16: **115,063,936 parameters**,
eight class prefix tokens, RMSNorm, RoPE and SwiGLU. There is no timestep
embedding or AdaLN. For each image, the variable-noise recipe samples
`t ~ Uniform[0,1)` and independent Gaussian noise, constructs
`x_t = (1-t)*x0 + t*epsilon`, and predicts `G(x_t, class)`. The scalar t controls
only input corruption and is never passed to the network.

## Exact reconstruction objective

For image i, let `M_i = mean((prediction_i - x0_i)**2)`. Pixel loss is

```text
L_pixel = mean_i[M_i / stop_gradient(M_i + 0.0001)]
```

Its gradient is the MSE gradient divided by the detached per-image residual
magnitude. Differentiating through the denominator would give a different
gradient and is not the original implementation. At first order this pixel
gradient equals the gradient of `mean_i[log(M_i + 0.0001)]`, but the logged loss
value is the normalized ratio, not the logarithm. There is no ordinary-MSE
conditional-mean guarantee for this adaptive objective.

The complete loss retains the existing output perceptual terms:

```text
mean_i[
    M_i / stop_gradient(M_i + 0.0001)
    + 0.4 * LPIPS_i / (stop_gradient(LPIPS_i) + 0.01)
    + 0.1 * ConvNeXt_i / (stop_gradient(ConvNeXt_i) + 0.01)
]
```

All terms target the same clean training image. The production auxiliary
wrapper maps predicted/target RGB from [-1,1] to clamped [0,1], then retains the
original encoder normalization and paired random crop. LPIPS and ConvNeXt
remain frozen while gradients flow into predictions. The auxiliary weights,
`norm_p=1`, `norm_eps=0.01` and crop settings are unchanged.

`perceptual_threshold=1.0` ensures auxiliary supervision for every sampled t.
Original fixed t=0.8 with threshold 0.8 also supervised every image. Leaving the
threshold at 0.8 under a uniform noise distribution would remove perceptual
supervision from about 20% of examples and create another comparison difference.

## Matched recipes

| Recipe | Input noise fraction |
| --- | --- |
| `configs/pretrain/b10_diffusion_no_time_adaptive_perceptual.yaml` | Per-image uniform [0,1) |
| `configs/pretrain/b10_fixed_noise_no_time_adaptive_perceptual.yaml` | Fixed 0.8 |

The resolved pair differs only in the noise interval and its experiment variant
label. Both explicitly record `experiment.loss_weighting: adaptive_residual`,
`experiment.conditioning: none`, pixel `norm_p=1`, and pixel `norm_eps=0.0001`.
Older `loss_weighting: uniform` recipes continue to mean plain MSE and are not
silently reinterpreted. The fixed-noise recipe provides the matched control;
creating it does not by itself schedule an additional training run.

Both recipes use ten epochs at 1,251 optimizer updates per epoch, for 12,510
updates total, global batch 1,024, BF16, seed 1, the same Muon optimizer and EMA
0.9999. Initialization is from scratch. The complete checkpoint is intended to
initialize the same supervised ResNet-50 OT post-training for 40 epochs, with
all generator weights trainable and one-call pure-noise generation
`G(z, class)`, `z ~ Normal(0,I)`.

This pair isolates the configured noise-distribution change. It does not promise
identical random draws across differently sampled t distributions or bitwise
agreement across hardware, compilers, batches or historical runs. CPU checks
with the same t, weights and random stream do establish exact original loss,
parameter-gradient and RNG equivalence for the tested execution path.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_no_time_adaptive_loss -v
```

Tests check the analytical detached-denominator gradient, distinguish it from
the incorrect differentiable denominator, verify the matched recipe pair and
original parameter schema, and compare actual Priming training plus the
production auxiliary wrapper at controlled t=0.8 against the original recipe.
Pretrained perceptual encoders are replaced by small differentiable frozen CPU
stubs; crop, clamp, normalization, loss weighting and masks remain the real
implementations. The all-t test includes t=0 and t=1. These checks do not measure
GPU throughput or establish downstream sample quality.
