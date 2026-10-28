"""Fixed validation denoising panels; no iterative sampler or training RNG use."""
from contextlib import nullcontext
import json
from pathlib import Path

import torch

from .random import snapshot_rng, restore_rng


@torch.no_grad()
def diffusion_panels(output, step, model, ema, rows, *, device="cuda"):
    from torchvision.utils import save_image
    saved = snapshot_rng()
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        images = torch.stack([x for x, _ in rows]).to(device)
        labels = torch.tensor([y for _, y in rows], device=device)
        generator = torch.Generator(device=device).manual_seed(20261028)
        noise = torch.randn(images.shape, device=device, generator=generator)
        second_noise = torch.randn(images.shape, device=device, generator=generator)
        panels = [images]
        report = dict(step=step, source="8 fixed ImageNet validation images",
                      nfe_per_prediction=1, rows=["real"], metrics={})
        for label in ("online", ema.label):
            swap = ema.swap(model) if label != "online" else nullcontext()
            with swap, torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                for t in (.2, .5, .8, 1.):
                    times = torch.full((len(labels),), t, device=device)
                    prediction = model.net((1-t)*images + t*noise, labels, noise_t=times).float()
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError("Nonfinite diffusion diagnostic prediction")
                    report["metrics"][f"{label}/mse_t_{t:.1f}"] = (prediction-images).square().mean().item()
                    panels.append(prediction)
                    report["rows"].append(f"{label} t={t:.1f}")
                generated_second = model.net(second_noise, labels, noise_t=times).float()
                if not torch.isfinite(generated_second).all():
                    raise FloatingPointError("Nonfinite second-noise diagnostic prediction")
                report["metrics"][f"{label}/paired_noise_output_mse"] = (prediction-generated_second).square().mean().item()
        report["interpretation"] = ("Denoising MSE uses paired validation targets. At t=1 the input is independent "
            "noise: paired-target MSE is not a generation-quality metric. Paired-noise output MSE is only a "
            "small fixed-sample sensitivity diagnostic; FID/recall are measured in post-training.")
        directory = Path(output)/"visualization"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/f"step_{step:07d}.png"
        save_image((torch.cat(panels)*.5+.5).clamp(0, 1), path, nrow=len(images))
        (directory/f"step_{step:07d}.json").write_text(json.dumps(report, indent=2)+"\n")
        return report, path
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(saved)
