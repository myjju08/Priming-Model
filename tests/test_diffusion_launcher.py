import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.run_diffusion_pretrain import (command, gpu_blockers, load_plan,
                                           predecessor_blockers, run_plan, post_command,
                                           wait_for_capacity, QueueStopped, initialization_checkpoint,
                                           verify_prevalidated_smoke)


class DiffusionLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = dict(storage=str(self.root), python=str(self.root / "venv/bin/python"),
                         config=str(self.root / "recipe.yaml"), output=str(self.root / "production"),
                         smoke_output=str(self.root / "smoke"), data_path=str(self.root / "data"),
                         hf_home=str(self.root / "hf"), torch_home=str(self.root / "torch"),
                         lpips_weights=str(self.root / "vgg.pth"),
                         predecessor_status=str(self.root / "predecessor.json"),
                         predecessor_output=str(self.root / "previous"),
                         predecessor_updates=100120, name="B10-noise-conditioned",
                         wandb_project="CVPR - Priming Model", wandb_entity="team", wandb_id="run-id")

    def write(self, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def complete_predecessor(self):
        self.write(self.plan["predecessor_status"], dict(status="complete", completed_updates=100120,
                   output=self.plan["predecessor_output"]))
        base = Path(self.plan["predecessor_output"])
        self.write(base / "checkpoints/latest.json", dict(checkpoint="step_100120.pth", completed=100120))
        (base / "checkpoints/step_100120.pth").write_bytes(b"checkpoint")
        self.write(base / "evaluation/step_100120.json", dict(identity=dict(step=100120), rows=[{}]))

    def test_finished_status_requires_checkpoint_and_final_evaluation(self):
        self.assertTrue(predecessor_blockers(self.plan))
        self.complete_predecessor()
        self.assertEqual(predecessor_blockers(self.plan), [])
        (Path(self.plan["predecessor_output"]) / "evaluation/step_100120.json").unlink()
        self.assertIn("evaluation", predecessor_blockers(self.plan)[0])

    def test_other_output_or_incomplete_status_cannot_unlock_training(self):
        self.complete_predecessor()
        for record in (dict(status="running", completed_updates=100120, output=self.plan["predecessor_output"]),
                       dict(status="complete", completed_updates=99999, output=self.plan["predecessor_output"]),
                       dict(status="complete", completed_updates=100120, output="/wrong/output")):
            self.write(self.plan["predecessor_status"], record)
            self.assertTrue(predecessor_blockers(self.plan))

    def test_gpu_guard_detects_active_processes_without_cuda_context(self):
        def result(text):
            return subprocess.CompletedProcess([], 0, stdout=text, stderr="")
        run = mock.Mock(side_effect=[result("\n".join(map(str, range(8)))), result("123\n123\n456\n")])
        self.assertIn("123, 456", gpu_blockers(run)[0])
        run = mock.Mock(side_effect=[result("\n".join(map(str, range(8)))), result("")])
        self.assertEqual(gpu_blockers(run), [])
        self.assertEqual(run.call_count, 2)

    def test_smoke_and_production_are_fresh_separate_torchrun_jobs(self):
        smoke = command(self.plan, "/tmp/smoke.yaml", smoke=True)
        production = command(self.plan, "/tmp/production.yaml")
        self.assertIn("--nproc_per_node=8", smoke)
        self.assertIn("--smoke-steps", smoke)
        self.assertNotIn("--smoke-steps", production)
        self.assertNotIn("--resume", smoke + production)
        self.assertIn(self.plan["smoke_output"], smoke)
        self.assertIn(self.plan["output"], production)

    def test_venv_python_path_is_not_resolved_outside_venv(self):
        interpreter = Path(self.plan["python"])
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to("/usr/bin/python3")
        path = self.root / "plan.json"
        self.write(path, self.plan)
        self.assertEqual(load_plan(path)["python"], str(interpreter))
        invalid = copy.deepcopy(self.plan)
        invalid["smoke_output"] = invalid["output"]
        self.write(path, invalid)
        with self.assertRaisesRegex(ValueError, "must differ"):
            load_plan(path)

    def test_blocked_preflight_never_launches_a_subprocess(self):
        with mock.patch("scripts.run_diffusion_pretrain.check", return_value=dict(ready=False, blockers=["busy"])), \
                mock.patch("scripts.run_diffusion_pretrain.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                run_plan(self.plan)
            run.assert_not_called()

    def test_failed_smoke_never_launches_production(self):
        report = dict(ready=True, asset_manifest={}, production_updates=12510)
        with mock.patch("scripts.run_diffusion_pretrain.check", return_value=report), \
                mock.patch("scripts.run_diffusion_pretrain.resolved_configs", return_value=({}, {})), \
                mock.patch("scripts.run_diffusion_pretrain.predecessor_blockers", return_value=[]), \
                mock.patch("scripts.run_diffusion_pretrain.gpu_blockers", return_value=[]), \
                mock.patch("utils.assets.pretraining_manifest", return_value={}), \
                mock.patch("scripts.run_diffusion_pretrain.subprocess.run",
                           side_effect=subprocess.CalledProcessError(1, "smoke")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                run_plan(self.plan)
            self.assertEqual(run.call_count, 1)
        status = json.loads((self.root / "status" / (self.plan["name"] + ".json")).read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["stage"], "smoke")

    def test_wait_polls_then_starts_only_after_capacity_is_available(self):
        sleep, record = mock.Mock(), mock.Mock()
        with mock.patch("scripts.run_diffusion_pretrain.predecessor_blockers", side_effect=[["running"], []]), \
                mock.patch("scripts.run_diffusion_pretrain.gpu_blockers", return_value=[]):
            wait_for_capacity(self.plan, wait=True, record=record, sleep=sleep)
        sleep.assert_called_once_with(30)
        record.assert_called_once_with("waiting", blockers=["running"])

    def test_stop_cancels_wait_without_launching_or_sleeping(self):
        (self.root / "STOP_REQUESTED").touch()
        sleep = mock.Mock()
        with self.assertRaises(QueueStopped):
            wait_for_capacity(self.plan, wait=True, record=mock.Mock(), sleep=sleep)
        sleep.assert_not_called()

    def test_authorized_predecessor_stop_requires_bound_checkpoint_receipt(self):
        self.complete_predecessor()
        self.plan["predecessor_stop_receipt"] = str(self.root / "stop-receipt.json")
        self.assertTrue(predecessor_blockers(self.plan))
        from utils.files import sha256
        checkpoint = Path(self.plan["predecessor_output"]) / "checkpoints/step_100120.pth"
        record = dict(status="stopped_for_diffusion", output=self.plan["predecessor_output"],
                      checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                      completed_updates=100120, gpu_processes_gone=True)
        self.write(self.plan["predecessor_stop_receipt"], record)
        self.assertEqual(predecessor_blockers(self.plan), [])
        record["output"] = "/another/run"
        self.write(self.plan["predecessor_stop_receipt"], record)
        with self.assertRaisesRegex(ValueError, "stop receipt"):
            predecessor_blockers(self.plan)

    def test_posttraining_uses_final_pretraining_checkpoint_and_fresh_optimizer(self):
        self.plan["posttraining"] = dict(output=str(self.root / "post"), smoke_output=str(self.root / "post-smoke"))
        smoke = post_command(self.plan, "/tmp/post-smoke.yaml", smoke=True)
        production = post_command(self.plan, "/tmp/post.yaml")
        self.assertIn(str(initialization_checkpoint(self.plan)), production)
        self.assertIn("--init-checkpoint", production)
        self.assertNotIn("--resume", smoke + production)
        self.assertIn("--smoke-steps", smoke)
        self.assertNotIn("--smoke-steps", production)

    def test_full_chain_order_and_smoke_tracking_are_isolated(self):
        self.assert_full_chain(adopt=False)

    def test_adopted_benchmark_skips_only_pretraining_smoke_and_starts_fresh_production(self):
        self.assert_full_chain(adopt=True)

    def assert_full_chain(self, *, adopt):
        adopted = dict(adopted=True, completed_updates=25, checkpoint_sha256="0"*64)
        if adopt:
            self.plan["prevalidated_pretrain_smoke"] = dict(output=str(self.root / "benchmark"),
                                                           updates=25, checkpoint_sha256="0"*64)
        cache = self.root / "encoder-cache"
        cache.mkdir()
        for name in ("calibration.json", "encoder_manifest.json", "supervised_encoder.pt"):
            (cache / name).write_bytes(b"asset")
        self.plan["posttraining"] = dict(config=str(self.root / "post.yaml"), output=str(self.root / "post"),
                smoke_output=str(self.root / "post-smoke"), encoder_checkpoint=str(self.root / "rn50.pth"),
                encoder_sha256="0"*64)
        for directory, step in ((self.plan["smoke_output"], 2), (self.plan["output"], 12510)):
            directory = Path(directory) / "checkpoints"
            directory.mkdir(parents=True)
            checkpoint = directory / f"step_{step:07d}.pth"
            checkpoint.write_bytes(b"pretrain checkpoint")
            (directory / "latest.pth").symlink_to(checkpoint.name)
        for key in ("smoke_output", "output"):
            output = Path(self.plan["posttraining"][key])
            output.mkdir()
            (output / "checkpoint.pth").write_bytes(b"post checkpoint")
        post_config = dict(data=dict(encoder_cache=str(cache)))
        with mock.patch("scripts.run_diffusion_pretrain.check", return_value=dict(ready=True, asset_manifest={}, production_updates=12510)), \
                mock.patch("scripts.run_diffusion_pretrain.resolved_configs", return_value=({}, {})), \
                mock.patch("scripts.run_diffusion_pretrain.resolved_post_configs", return_value=(post_config, post_config)), \
                mock.patch("scripts.run_diffusion_pretrain.source_manifest", return_value={}), \
                mock.patch("scripts.run_diffusion_pretrain.wait_for_capacity"), \
                mock.patch("scripts.run_diffusion_pretrain.verify_completed"), \
                mock.patch("scripts.run_diffusion_pretrain.verify_prevalidated_smoke", return_value=adopted), \
                mock.patch("scripts.run_diffusion_pretrain.verify_post_completed", side_effect=lambda output,*args:Path(output)/"checkpoint.pth"), \
                mock.patch("scripts.run_diffusion_pretrain.subprocess.run") as execute:
            run_plan(self.plan, wait=True)
        calls = execute.call_args_list
        self.assertEqual(len(calls), 4 if adopt else 5)
        if not adopt:
            self.assertIn("--smoke-steps", calls[0].args[0])
        tail = calls[-4:]
        self.assertNotIn("--smoke-steps", tail[0].args[0])
        self.assertNotIn("--resume", tail[0].args[0])
        self.assertIn("prepare", tail[1].args[0])
        self.assertIn("--smoke-steps", tail[2].args[0])
        self.assertNotIn("--smoke-steps", tail[3].args[0])
        self.assertEqual([call.kwargs["env"]["WANDB_MODE"] for call in calls],
                         (["disabled"] if not adopt else []) + ["online", "disabled", "disabled", "online"])
        status = json.loads((self.root / "status" / (self.plan["name"] + ".json")).read_text())
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["completed_updates"], 100120)

    def make_prevalidated_smoke(self):
        import torch
        from utils.files import sha256
        output = self.root / "verified-benchmark"
        directory = output / "checkpoints"
        directory.mkdir(parents=True)
        checkpoint = directory / "step_0000025.pth"
        config = dict(model=dict(timestep_conditioning=True), loss=dict(aux=dict(enabled=True)))
        manifest = dict(runtime={"pretrain.py":"source-hash"}, assets={"vgg16":"asset-hash"})
        state = dict(step=25, current_step=26, samples_seen=25*1024, world_size=8,
                     rank_states=[dict(rank=rank) for rank in range(8)], config=config, manifest=manifest)
        torch.save(state, checkpoint)
        (directory / "latest.pth").symlink_to(checkpoint.name)
        self.write(output / "manifest.json", manifest)
        self.write(output / "config.json", config)
        rows = [dict(step=step, loss=.1, grad_norm=.01, update_seconds=1.2) for step in range(1,26)]
        (output / "metrics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        self.plan["prevalidated_pretrain_smoke"] = dict(output=str(output), updates=25,
                                                       checkpoint_sha256=sha256(checkpoint))
        return output, config, manifest

    def test_prevalidated_smoke_requires_exact_config_manifest_and_checksum(self):
        output, config, manifest = self.make_prevalidated_smoke()
        with mock.patch("utils.assets.pretraining_manifest", return_value=manifest):
            report = verify_prevalidated_smoke(self.plan, config)
            self.assertTrue(report["adopted"])
            self.assertEqual(report["completed_updates"], 25)
            changed = copy.deepcopy(config)
            changed["loss"]["aux"]["enabled"] = False
            with self.assertRaisesRegex(ValueError, "configuration"):
                verify_prevalidated_smoke(self.plan, changed)
        with mock.patch("utils.assets.pretraining_manifest", return_value={"runtime":"changed"}):
            with self.assertRaisesRegex(ValueError, "source, dataset, or asset"):
                verify_prevalidated_smoke(self.plan, config)
        self.plan["prevalidated_pretrain_smoke"]["checkpoint_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_prevalidated_smoke(self.plan, config)

    def test_prevalidated_smoke_rejects_missing_updates_and_nonfinite_metrics(self):
        output, config, manifest = self.make_prevalidated_smoke()
        path = output / "metrics.jsonl"
        rows = path.read_text().splitlines()
        path.write_text("\n".join(rows[1:])+"\n")
        with mock.patch("utils.assets.pretraining_manifest", return_value=manifest):
            with self.assertRaisesRegex(ValueError, "incomplete or nonfinite"):
                verify_prevalidated_smoke(self.plan, config)
        bad = json.loads(rows[0])
        bad["update_seconds"] = float("nan")
        path.write_text("\n".join([json.dumps(bad), *rows[1:]])+"\n")
        with mock.patch("utils.assets.pretraining_manifest", return_value=manifest):
            with self.assertRaisesRegex(ValueError, "incomplete or nonfinite"):
                verify_prevalidated_smoke(self.plan, config)

    def test_prevalidated_smoke_plan_rejects_unbound_digest_and_reused_output(self):
        path = self.root / "plan.json"
        self.plan["prevalidated_pretrain_smoke"] = dict(output=str(self.root / "benchmark"), updates=25)
        self.write(path, self.plan)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_plan(path)
        self.plan["prevalidated_pretrain_smoke"].update(checkpoint_sha256="0"*64, output=self.plan["output"])
        self.write(path, self.plan)
        with self.assertRaisesRegex(ValueError, "separate"):
            load_plan(path)


if __name__ == "__main__":
    unittest.main()
