"""Pinned downloads can resume transport errors but never replace corrupt assets."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import prepare_experiment_assets as assets


class ExperimentAssetTests(unittest.TestCase):
    def test_dataset_keeps_exact_release_order_and_identities(self):
        release = json.loads((ROOT/'old/releases/dataset.json').read_text())
        rows = assets.asset_rows('dataset')
        self.assertEqual(len(rows), len(release['files']))
        for row, expected in zip(rows, release['files']):
            self.assertEqual(row['path'], expected['path'])
            self.assertEqual(row['sha256'], expected['sha256'])
            self.assertEqual(row['bytes'], expected['bytes'])
            self.assertEqual(row['revision'], release['revision'])
            self.assertEqual(row['repo_type'], 'dataset')

    def test_support_includes_exact_native_assets_convnext_and_b160(self):
        rows = assets.asset_rows('support')
        paths = {row['path'] for row in rows}
        release = json.loads((ROOT/'old/releases/encoders.json').read_text())
        expected = {row['path'] for row in release['files'] if
                    'posttraining' in row['groups'] or
                    ('conversion' in row['groups'] and
                     '/encoder_sources/convnext/' in row['path'])}
        expected.add('checkpoints/pretraining/priming_b_160.pth')
        self.assertEqual(paths, expected)
        self.assertEqual(len(rows), len(paths))
        self.assertFalse(any('/encoder_sources/mae/' in path for path in paths))
        self.assertFalse(any('/drive_eval/' in path for path in paths))

    def test_timeout_retries_and_retains_full_checksum_record(self):
        row = {'path': 'sample', 'sha256': 'expected', 'bytes': 8}
        with patch.object(assets, 'download', side_effect=[TimeoutError(), None]) as download, \
             patch.object(assets.time, 'sleep') as sleep:
            assets.download_with_retry(Path('/tmp/unused'), row)
        self.assertEqual(download.call_count, 2)
        self.assertEqual(download.call_args.args[1], row)
        sleep.assert_called_once_with(5)

    def test_authorization_and_integrity_failures_do_not_retry(self):
        for error in (ValueError('checksum mismatch'), FileNotFoundError('missing'),
                      urllib.error.HTTPError('https://example.invalid', 403, 'denied', {}, None)):
            with self.subTest(error=type(error).__name__), \
                 patch.object(assets, 'download', side_effect=error) as download, \
                 patch.object(assets.time, 'sleep') as sleep:
                with self.assertRaises(type(error)):
                    assets.download_with_retry(Path('/tmp/unused'), {'path': 'sample'})
                download.assert_called_once()
                sleep.assert_not_called()

    def test_corrupt_existing_file_is_not_changed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root/'sample'
            target.write_bytes(b'bad')
            row = dict(path='sample', bytes=3, sha256=hashlib.sha256(b'yes').hexdigest(),
                       kind='url', url='https://example.invalid')
            with patch.object(assets.time, 'sleep') as sleep:
                with self.assertRaises(ValueError):
                    assets.download_with_retry(root, row)
            self.assertEqual(target.read_bytes(), b'bad')
            sleep.assert_not_called()

    def test_completion_record_requires_every_verified_file(self):
        rows = [dict(path='a', bytes=3), dict(path='b', bytes=5)]
        with TemporaryDirectory() as temporary:
            status = Path(temporary)/'status.json'
            with patch.object(assets, 'asset_rows', return_value=rows), \
                 patch.object(assets, 'download_with_retry'):
                assets.prepare(temporary, 'support', status_file=status)
            record = json.loads(status.read_text())
            self.assertEqual(record['status'], 'complete')
            self.assertEqual(record['completed_files'], 2)
            self.assertEqual(record['completed_bytes'], 8)
            with patch.object(assets, 'asset_rows', return_value=rows), \
                 patch.object(assets, 'download_with_retry', side_effect=ValueError('corrupt')):
                with self.assertRaises(ValueError):
                    assets.prepare(temporary, 'support', status_file=status)
            record = json.loads(status.read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertEqual(record['error_type'], 'ValueError')
            self.assertNotEqual(record['completed_files'], 2)


if __name__ == '__main__':
    unittest.main()
