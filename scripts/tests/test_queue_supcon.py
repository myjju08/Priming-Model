import json
from pathlib import Path
import tempfile
import unittest

from scripts.queue_supcon_experiment import (completed_updates, load_plan,
    preserve_identity, readiness, resume_mode, train_command, verify_references)
from utils.files import sha256, write_json


class QueueSupConTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = dict(storage=str(self.root), checkpoint=str(self.root/'backbone.pth'),
            conversion_python=str(self.root/'python'), output=str(self.root/'run'),
            smoke_output=str(self.root/'smoke'), wandb_project='project',
            wandb_entity='entity', run_name='supcon', epochs=40, eval_every_epochs=5,
            reference_sha256={})
        for group in ('dataset', 'support'):
            write_json(self.root/'status'/f'{group}.json',
                       dict(status='complete', storage=str(self.root)))
        write_json(self.root/'status/environment.json', dict(status='passed',
            unit_tests=dict(status='passed'), distributed_gpu=dict(status='passed',gpus=[{}]*8)))
        (self.root/'python').touch()
        (self.root/'backbone.pth').write_bytes(b'weights')
        for key,name in [('fid_reference','imagenet_256_fid_stats.npz'),
                         ('prc_images','imagenet_val_prc_arr0.npz')]:
            path=self.root/'assets/drive_eval'/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(key.encode())
            self.plan['reference_sha256'][key]=sha256(path)

    def test_checkpoint_must_remain_stable_for_sixty_seconds(self):
        observed={}
        self.assertTrue(readiness(self.plan,observed,0))
        self.assertTrue(readiness(self.plan,observed,59))
        self.assertEqual(readiness(self.plan,observed,60),[])
        (self.root/'backbone.pth').write_bytes(b'new weights')
        self.assertTrue(readiness(self.plan,observed,61))
        self.assertEqual(readiness(self.plan,observed,121),[])

    def test_failed_download_is_not_waited_on_forever(self):
        write_json(self.root/'status/dataset.json',dict(status='failed'))
        with self.assertRaisesRegex(ValueError,'download failed'):
            readiness(self.plan,{},0)

    def test_incomplete_download_and_missing_reference_wait(self):
        write_json(self.root/'status/dataset.json',dict(status='running'))
        (self.root/'assets/drive_eval/imagenet_256_fid_stats.npz').unlink()
        missing=readiness(self.plan,{},0)
        self.assertIn('dataset downloads',missing)
        self.assertIn('imagenet_256_fid_stats.npz',missing)

    def test_reference_hash_mismatch_fails(self):
        verify_references(self.plan)
        (self.root/'assets/drive_eval/imagenet_256_fid_stats.npz').write_bytes(b'wrong reference')
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
            verify_references(self.plan)

    def test_resume_requires_run_and_checkpoint_records(self):
        output=self.root/'run'
        self.assertFalse(resume_mode(output))
        output.mkdir()
        write_json(output/'run.json',{})
        with self.assertRaisesRegex(ValueError,'no resumable checkpoint'):
            resume_mode(output)
        write_json(output/'checkpoints/latest.json',{})
        self.assertTrue(resume_mode(output))

    def test_changed_input_identity_cannot_reuse_smoke(self):
        path=self.root/'identity.json'
        identity={'checkpoint_sha256':'a','plan_sha256':'b'}
        preserve_identity(path,identity)
        preserve_identity(path,identity)
        with self.assertRaisesRegex(ValueError,'changed'):
            preserve_identity(path,dict(identity,checkpoint_sha256='c'))

    def test_completion_requires_saved_payload_and_update_count(self):
        output=self.root/'run'
        write_json(output/'checkpoints/latest.json',dict(checkpoint='step_000002.pth',completed=2))
        with self.assertRaisesRegex(ValueError,'Invalid latest checkpoint'):
            completed_updates(output)
        (output/'checkpoints/step_000002.pth').touch()
        self.assertEqual(completed_updates(output),2)
        write_json(output/'checkpoints/latest.json',dict(checkpoint='../outside.pth',completed=2))
        with self.assertRaisesRegex(ValueError,'Invalid latest checkpoint'):
            completed_updates(output)

    def test_smoke_does_not_create_wandb_run_and_production_resumes(self):
        smoke=train_command(self.plan,smoke=True)
        self.assertIn('--smoke-steps',smoke)
        self.assertNotIn('--wandb-project',smoke)
        production=train_command(self.plan,resume=True)
        self.assertIn('--wandb-project',production)
        self.assertIn('--resume',production)

    def test_plan_rejects_shared_output_or_changed_protocol(self):
        path=self.root/'plan.json'
        write_json(path,self.plan)
        self.assertEqual(load_plan(path)['evaluation'],'drive')
        write_json(path,dict(self.plan,evaluation='native'))
        with self.assertRaisesRegex(ValueError,'Drive'):
            load_plan(path)
        write_json(path,dict(self.plan,smoke_output=self.plan['output']))
        with self.assertRaisesRegex(ValueError,'must differ'):
            load_plan(path)


if __name__ == '__main__':
    unittest.main()
