"""Managed diffusion pretraining followed by supervised-RN50 one-step OT.

``--wait`` queues behind the preceding experiment; ``--check`` is read-only.
Completed stages are recorded and never automatically retried. A partial run
requires explicit recovery rather than silently starting a new experiment.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.files import sha256, write_json


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    paths = ("storage", "python", "config", "output", "smoke_output", "data_path",
             "hf_home", "torch_home", "lpips_weights", "predecessor_status",
             "predecessor_output")
    for key in paths:
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ValueError(f"Plan requires {key}")
        path = Path(plan[key]).expanduser()
        # Resolving a venv's Python symlink discards its site-packages isolation.
        plan[key] = str(path.absolute() if key == "python" else path.resolve())
    if Path(plan["output"]) == Path(plan["smoke_output"]):
        raise ValueError("Smoke and production outputs must differ")
    for key in ("name", "wandb_entity", "wandb_project", "wandb_id"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ValueError(f"Plan requires {key}")
    if Path(plan["name"]).name != plan["name"] or plan["name"] in (".", ".."):
        raise ValueError("Plan name must be a simple directory name")
    if type(plan.get("predecessor_updates")) is not int or plan["predecessor_updates"] < 1:
        raise ValueError("Plan requires a positive predecessor update count")
    if plan.get("predecessor_stop_receipt"):
        plan["predecessor_stop_receipt"] = str(Path(plan["predecessor_stop_receipt"]).resolve())
    if "posttraining" in plan:
        post = plan["posttraining"]
        for key in ("config", "output", "smoke_output", "encoder_checkpoint"):
            if not isinstance(post.get(key), str) or not post[key].strip():
                raise ValueError(f"posttraining requires {key}")
            post[key] = str(Path(post[key]).expanduser().resolve())
        outputs = [plan["output"], plan["smoke_output"], post["output"], post["smoke_output"]]
        if len(set(outputs)) != 4:
            raise ValueError("All smoke and production outputs must differ")
        digest = post.get("encoder_sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("posttraining requires the supervised encoder SHA256")
    return plan


def predecessor_blockers(plan):
    if plan.get("predecessor_stop_receipt"):
        receipt_path = Path(plan["predecessor_stop_receipt"])
        if not receipt_path.is_file():
            return ["Waiting for the explicitly authorized SupCon stop receipt"]
        receipt = json.loads(receipt_path.read_text())
        output = Path(plan["predecessor_output"])
        checkpoint = Path(receipt.get("checkpoint", "")).resolve()
        if (receipt.get("status") != "stopped_for_diffusion" or receipt.get("gpu_processes_gone") is not True
                or Path(receipt.get("output", "")).resolve() != output
                or checkpoint.parent != output / "checkpoints" or not checkpoint.is_file()
                or type(receipt.get("completed_updates")) is not int or receipt["completed_updates"] < 1):
            raise ValueError("Invalid explicitly authorized SupCon stop receipt")
        latest = json.loads((output / "checkpoints/latest.json").read_text())
        if latest.get("completed") != receipt["completed_updates"] or latest.get("checkpoint") != checkpoint.name:
            raise ValueError("SupCon stop receipt differs from its latest checkpoint")
        return []
    status_path = Path(plan["predecessor_status"])
    if not status_path.is_file():
        return ["Preceding SupCon run has no completion status"]
    record = json.loads(status_path.read_text())
    if record.get("status") in {"failed", "stopped"}:
        raise RuntimeError("Preceding SupCon stopped without an authorized replacement receipt")
    expected, output = plan["predecessor_updates"], Path(plan["predecessor_output"])
    if (record.get("status") != "complete" or record.get("completed_updates") != expected
            or Path(record.get("output", "")).resolve() != output):
        return [f"Preceding SupCon run is not complete at {expected} updates"]
    latest_path = output / "checkpoints/latest.json"
    if not latest_path.is_file():
        return ["Preceding SupCon run has no final checkpoint record"]
    latest = json.loads(latest_path.read_text())
    name = latest.get("checkpoint", "")
    if (latest.get("completed") != expected or Path(name).name != name or not name
            or not (latest_path.parent / name).is_file()):
        return ["Preceding SupCon final checkpoint is missing or incomplete"]
    evaluation = output / "evaluation" / f"step_{expected:06d}.json"
    if not evaluation.is_file():
        return ["Preceding SupCon final evaluation is missing"]
    result = json.loads(evaluation.read_text())
    if result.get("identity", {}).get("step") != expected or not result.get("rows"):
        return ["Preceding SupCon final evaluation is incomplete"]
    return []


def gpu_blockers(run=subprocess.run):
    """nvidia-smi is read-only and never creates a CUDA context."""
    cards = run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                check=True, capture_output=True, text=True)
    if len(cards.stdout.strip().splitlines()) != 8:
        return ["This launcher requires exactly eight visible physical GPUs"]
    processes = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True)
    pids = sorted(set(line.strip() for line in processes.stdout.splitlines() if line.strip()))
    return ["GPU compute processes are still active: " + ", ".join(pids)] if pids else []


def resolved_configs(plan):
    from pretrain import load_config
    config, family = load_config(plan["config"])
    if family != "b" or config["train"]["epochs"] != 10:
        raise ValueError("This launcher requires a ten-epoch Priming-B recipe")
    config["data"]["data_path"] = plan["data_path"]
    config["tracking"] = dict(enabled=True, entity=plan["wandb_entity"],
                              project=plan["wandb_project"], name=plan["name"], id=plan["wandb_id"])
    smoke = copy.deepcopy(config)
    smoke.pop("tracking", None)
    return config, smoke


def initialization_checkpoint(plan):
    return Path(plan["output"]) / "checkpoints/step_0012510.pth"


def resolved_post_configs(plan):
    from posttrain import load_config
    config = load_config(plan["posttraining"]["config"])
    if (config["train"]["steps"] != 100120 or config["matching"]["stack"] != "convnext_supervised_rn50"
            or config["model"].get("timestep_conditioning") is not True
            or config["model"].get("generation_t") != 1.
            or config["sampling"]["num_steps"] != 1
            or config["initialization"]["checkpoint"] != str(initialization_checkpoint(plan))):
        raise ValueError("Post-training must use the complete diffusion checkpoint, supervised RN50 and fixed-t=1 one-step generation")
    tracking = config.get("tracking", {})
    if (tracking.get("project"), tracking.get("entity")) != (plan["wandb_project"], plan["wandb_entity"]):
        raise ValueError("Post-training W&B project/entity differs from the authorized project")
    smoke = copy.deepcopy(config)
    smoke.pop("tracking", None)
    return config, smoke


def source_manifest(plan, config):
    from utils.assets import pretraining_manifest
    manifest = pretraining_manifest(config, root=ROOT, family="b", hf_home=plan["hf_home"],
                                    torch_home=plan["torch_home"], lpips_weights=plan["lpips_weights"])
    if "posttraining" in plan:
        files = [ROOT / "posttrain.py", ROOT / "eval.py", ROOT / "scripts/prepare_feature_encoder.py",
                 ROOT / "scripts/experiment.py", Path(__file__)]
        files.extend((ROOT / "metrics").rglob("*.py"))
        manifest["runtime"].update({str(path.relative_to(ROOT)): sha256(path) for path in files})
        manifest["posttraining_config_sha256"] = sha256(plan["posttraining"]["config"])
    return manifest


def check(plan):
    config, _ = resolved_configs(plan)
    if not Path(plan["python"]).is_file():
        raise FileNotFoundError(plan["python"])
    manifest = source_manifest(plan, config)
    blockers = predecessor_blockers(plan) + gpu_blockers()
    fatal = []
    outputs = [plan["output"], plan["smoke_output"]]
    if "posttraining" in plan:
        post, _ = resolved_post_configs(plan)
        from utils.posttrain_assets import evaluation_assets
        evaluation_assets(post["evaluation"], hf_home=plan["hf_home"], torch_home=plan["torch_home"])
        if sha256(plan["posttraining"]["encoder_checkpoint"]) != plan["posttraining"]["encoder_sha256"]:
            raise ValueError("Supervised RN50 input weights differ")
        outputs += [plan["posttraining"]["output"], plan["posttraining"]["smoke_output"]]
    for output in outputs:
        if Path(output).exists():
            fatal.append(f"Fresh-run output already exists; explicit recovery is required: {output}")
    if shutil.disk_usage(plan["storage"]).free < plan.get("min_free_gib", 60) * (1 << 30):
        fatal.append("Insufficient free disk space for both training phases")
    if stop_requested(plan):
        fatal.append("STOP_REQUESTED is present")
    blockers += fatal
    return dict(ready=not blockers, blockers=blockers, config_valid=True,
                fatal_blockers=fatal,
                asset_manifest=manifest, production_updates=config["train"]["epochs"]
                * config["train"]["steps_per_epoch"], posttraining_updates=100120 if "posttraining" in plan else None,
                smoke_updates=2, launched=False)


def command(plan, config_path, *, smoke=False):
    result = [plan["python"], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
              str(ROOT / "pretrain.py"), "--config", str(config_path), "--output",
              plan["smoke_output"] if smoke else plan["output"], "--data-path", plan["data_path"],
              "--hf-home", plan["hf_home"], "--torch-home", plan["torch_home"],
              "--lpips-weights", plan["lpips_weights"]]
    if smoke:
        result += ["--smoke-steps", "2"]
    return result


def verify_completed(output, expected, expected_config):
    import torch
    output = Path(output)
    latest = output / "checkpoints/latest.pth"
    if not latest.is_file() or latest.resolve().parent != (output / "checkpoints").resolve():
        raise ValueError("Pretraining did not produce a local checkpoint")
    state = torch.load(latest, map_location="cpu", mmap=True, weights_only=False)
    if (state.get("step") != expected or state.get("current_step") != expected + 1
            or state.get("samples_seen") != expected * 1024 or state.get("world_size") != 8
            or state.get("config") != expected_config or len(state.get("rank_states", [])) != 8):
        raise ValueError("Pretraining checkpoint counters, configuration, or rank topology differ")
    last = json.loads((output / "metrics.jsonl").read_text().splitlines()[-1])
    if last.get("step") != expected or not all(math.isfinite(last[k]) for k in ("loss", "grad_norm")):
        raise ValueError("Pretraining did not finish with finite loss and gradients")


def post_command(plan, config_path, *, smoke=False):
    post = plan["posttraining"]
    result = [plan["python"], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
              str(ROOT / "posttrain.py"), "--config", str(config_path), "--output",
              post["smoke_output"] if smoke else post["output"],
              "--init-checkpoint", str(initialization_checkpoint(plan)),
              "--hf-home", plan["hf_home"], "--torch-home", plan["torch_home"]]
    if smoke:
        result += ["--smoke-steps", "2"]
    return result


def prepare_command(plan):
    post = plan["posttraining"]
    return [plan["python"], str(ROOT / "scripts/experiment.py"), "prepare", "--encoder", "supervised",
            "--storage", plan["storage"], "--evaluation", "drive", "--encoder-checkpoint",
            post["encoder_checkpoint"], "--encoder-sha256", post["encoder_sha256"], "--device", "cuda"]


def verify_post_completed(output, expected, config, initialization):
    output = Path(output)
    record = json.loads((output / "checkpoints/latest.json").read_text())
    name = record.get("checkpoint", "")
    checkpoint = output / "checkpoints" / name
    if (not name or Path(name).name != name or not checkpoint.is_file() or record.get("completed") != expected):
        raise ValueError("Post-training did not produce its expected completed checkpoint")
    run = json.loads((output / "run.json").read_text())
    if run.get("config") != config or run.get("source") != initialization:
        raise ValueError("Post-training configuration or pretraining checkpoint identity differs")
    last = json.loads((output / "training.jsonl").read_text().splitlines()[-1])
    if not all(math.isfinite(last[k]) for k in ("loss", "grad_norm")):
        raise ValueError("Post-training loss or gradient is nonfinite")
    if expected == 100120:
        evaluation = json.loads((output / "evaluation/step_100120.json").read_text())
        if (evaluation.get("identity", {}).get("step") != expected
                or evaluation["identity"].get("checkpoint_sha256") != sha256(checkpoint)
                or {row.get("ema_label") for row in evaluation.get("rows", [])} != {"online", "0.999"}):
            raise ValueError("Final post-training evaluation is incomplete or belongs to another checkpoint")
        for row in evaluation["rows"]:
            if not all(math.isfinite(row[key]) for key in ("fid", "is", "precision", "recall")):
                raise ValueError("Final post-training metrics are nonfinite")
    return checkpoint


def runtime_path(plan):
    return Path(plan["storage"]) / "plans" / (plan["name"] + "-runtime")


def stop_requested(plan):
    return any(path.exists() for path in (Path(plan["storage"]) / "STOP_REQUESTED",
                                          runtime_path(plan) / "STOP_REQUESTED"))


class QueueStopped(RuntimeError):
    pass


def wait_for_capacity(plan, *, wait, record, sleep=time.sleep, poll_seconds=30):
    if not 0 < poll_seconds <= 30:
        raise ValueError("Queue polling must be between zero and thirty seconds")
    while True:
        if stop_requested(plan):
            raise QueueStopped("STOP_REQUESTED prevents the next stage")
        blockers = predecessor_blockers(plan) + gpu_blockers()
        if not blockers:
            return
        if not wait:
            raise RuntimeError("; ".join(blockers))
        record("waiting", blockers=blockers)
        sleep(poll_seconds)


def preserve(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Queue inputs changed after preparation: {path}")
    write_json(path, value)


def run_plan(plan, *, wait=False):
    import yaml
    storage = Path(plan["storage"])
    runtime = runtime_path(plan)
    status = storage / "status" / (plan["name"] + ".json")
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status.is_file() and json.loads(status.read_text()).get("status") == "complete":
            raise RuntimeError("This experiment chain already completed; refusing to repeat it")
        report = check(plan)
        if report.get("fatal_blockers") or (not wait and not report["ready"]):
            raise RuntimeError("; ".join(report["blockers"]))
        production, smoke = resolved_configs(plan)
        paths = {"smoke": runtime / "smoke.yaml", "production": runtime / "production.yaml"}
        for key, config in (("smoke", smoke), ("production", production)):
            paths[key].write_text(yaml.safe_dump(config, sort_keys=False))
        preserve(runtime / "queue-plan.json", plan)
        preserve(runtime / "input-manifest.json", report["asset_manifest"])
        post_production = post_smoke = None
        if "posttraining" in plan:
            post_production, post_smoke = resolved_post_configs(plan)
            for key, config in (("posttrain_smoke", post_smoke), ("posttraining", post_production)):
                paths[key] = runtime / (key + ".yaml")
                paths[key].write_text(yaml.safe_dump(config, sort_keys=False))
        environment = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
                           OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                           TORCHINDUCTOR_COMPILE_THREADS="2", HF_HUB_OFFLINE="1",
                           HF_HOME=plan["hf_home"], TORCH_HOME=plan["torch_home"],
                           PREEMPT_FLAG_FILE=str(storage / "STOP_REQUESTED"))
        stage = "smoke"

        def record(state, **extra):
            write_json(status, dict(status=state, stage=stage, output=plan["output"],
                                    posttraining_output=plan.get("posttraining", {}).get("output"),
                                    updated_at=time.time(), **extra))
            print(f"Diffusion chain: {state} / {stage}", flush=True)

        def ready():
            wait_for_capacity(plan, wait=wait, record=record)
            if source_manifest(plan, production) != report["asset_manifest"]:
                raise ValueError("Source, dataset, assets, or post-training configuration changed after preflight")
            if plan.get("predecessor_stop_receipt"):
                receipt = json.loads(Path(plan["predecessor_stop_receipt"]).read_text())
                if sha256(receipt["checkpoint"]) != receipt.get("checkpoint_sha256"):
                    raise ValueError("Authorized SupCon stop checkpoint checksum differs")

        def receipt(path, updates):
            preserve(runtime / (stage + "-complete.json"), dict(stage=stage, completed_updates=updates,
                     checkpoint=str(path), checkpoint_sha256=sha256(path)))

        try:
            for stage, config, expected in (("smoke", smoke, 2),
                                           ("production", production, report["production_updates"])):
                ready()
                record("running")
                stage_env = dict(environment, WANDB_MODE="disabled" if stage == "smoke" else "online")
                subprocess.run(command(plan, paths[stage], smoke=stage == "smoke"),
                               cwd=ROOT, env=stage_env, check=True)
                verify_completed(plan["smoke_output"] if stage == "smoke" else plan["output"],
                                 expected, config)
                finished = Path(plan["smoke_output"] if stage == "smoke" else plan["output"])
                receipt((finished / "checkpoints/latest.pth").resolve(), expected)
            if post_production is not None:
                checkpoint = initialization_checkpoint(plan)
                initialization = dict(path=str(checkpoint), bytes=checkpoint.stat().st_size, sha256=sha256(checkpoint))
                preserve(runtime / "posttraining-initialization.json", initialization)
                stage = "supervised_assets"
                ready()
                record("running")
                subprocess.run(prepare_command(plan), cwd=ROOT,
                               env=dict(environment, CUDA_VISIBLE_DEVICES="0", WANDB_MODE="disabled"), check=True)
                cache = Path(post_production["data"]["encoder_cache"])
                assets = {path.name: sha256(path) for path in cache.iterdir() if path.is_file()}
                if not {"calibration.json", "encoder_manifest.json", "supervised_encoder.pt"} <= set(assets):
                    raise ValueError("Supervised encoder preparation did not produce the required assets")
                preserve(runtime / "supervised-assets.json", assets)
                for stage, config, expected in (("posttrain_smoke", post_smoke, 2),
                                               ("posttraining", post_production, 100120)):
                    ready()
                    if sha256(checkpoint) != initialization["sha256"]:
                        raise ValueError("Pretraining checkpoint changed before post-training")
                    if assets != {path.name: sha256(path) for path in cache.iterdir() if path.is_file()}:
                        raise ValueError("Supervised encoder assets changed after calibration")
                    record("running")
                    smoke_stage = stage == "posttrain_smoke"
                    subprocess.run(post_command(plan, paths[stage], smoke=smoke_stage), cwd=ROOT,
                                   env=dict(environment, WANDB_MODE="disabled" if smoke_stage else "online"), check=True)
                    output = plan["posttraining"]["smoke_output" if smoke_stage else "output"]
                    finished = verify_post_completed(output, expected, config, initialization)
                    receipt(finished, expected)
            record("complete", completed_updates=expected)
        except QueueStopped as exc:
            record("stopped", reason=str(exc))
            return
        except BaseException as exc:
            record("failed", error=repr(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Wait for predecessor completion/authorized stop and idle GPUs")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.check:
        report = check(plan)
        report.pop("asset_manifest")
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 2
    run_plan(plan, wait=args.wait)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
