"""Download pinned RN50 experiment assets before the replacement encoder arrives.

Dataset and support groups can run as separate managed jobs. This command only
downloads and verifies files; ConvNeXt conversion, SupCon calibration and
training remain explicit subsequent steps. Native evaluation assets are a
superset for preparation, not a choice of the eventual evaluation protocol.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import errno
import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import time
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_assets import download
from utils.files import write_json


def asset_rows(group, root=ROOT):
    """Select release identities without downloading unrelated MAE weights."""
    root = Path(root)
    if group == 'dataset':
        release = json.loads((root/'old/releases/dataset.json').read_text())
        rows = [dict(row, kind='hub_file', repo_id=release['repo_id'],
                     repo_type='dataset', revision=release['revision'],
                     filename=row['path']) for row in release['files']]
    elif group == 'support':
        release = json.loads((root/'old/releases/encoders.json').read_text())
        rows = [row for row in release['files']
                if 'posttraining' in row['groups'] or
                ('conversion' in row['groups'] and
                 '/encoder_sources/convnext/' in row['path'])]
        checkpoint = json.loads((root/'old/checkpoints/pretraining_manifest.json').read_text())['priming_b_160']
        rows.append(dict(kind='hub_file', path=checkpoint['checkpoint'],
                         bytes=checkpoint['checkpoint_bytes'],
                         sha256=checkpoint['checkpoint_sha256'],
                         repo_id='jaykim1102/priming', repo_type='model',
                         revision='9dc5fe94583b4479edb954327110850dc748ce44',
                         filename=checkpoint['checkpoint']))
    else:
        raise ValueError(f'Unknown asset group: {group}')
    unique = {}
    for row in rows:
        if row['path'] in unique and unique[row['path']] != row:
            raise ValueError(f'Conflicting asset identities: {row["path"]}')
        unique[row['path']] = row
    return list(unique.values())


def transient(error):
    """Retry transport failures only; access and integrity failures are final."""
    response = getattr(error, 'response', None)
    status = getattr(response, 'status_code', getattr(error, 'code', None))
    if isinstance(status, int):
        return status in (408, 425, 429) or 500 <= status < 600
    if isinstance(error, ValueError):
        return False
    if isinstance(error, (TimeoutError, ConnectionError, http.client.IncompleteRead,
                          http.client.RemoteDisconnected)):
        return True
    if isinstance(error, OSError) and error.errno in {
            errno.ECONNRESET, errno.ECONNABORTED, errno.ETIMEDOUT,
            errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ECONNREFUSED}:
        return True
    # Hub clients use either requests or httpx, depending on installed version.
    if type(error).__module__.split('.')[0] in ('requests', 'httpx', 'httpx2', 'urllib3'):
        if type(error).__name__ in {'ConnectError', 'ReadError', 'WriteError',
                                   'RemoteProtocolError', 'ConnectTimeout',
                                   'ReadTimeout', 'WriteTimeout', 'PoolTimeout',
                                   'Timeout', 'ConnectionError'}:
            return True
    if isinstance(error, urllib.error.URLError):
        return isinstance(error.reason, Exception) and transient(error.reason)
    cause = error.__cause__
    return cause is not None and cause is not error and transient(cause)


def download_with_retry(storage, row, *, attempts=5, retry_delay=5,
                        verify_only=False):
    for attempt in range(1, attempts+1):
        try:
            download(storage, row, verify_only=verify_only)
            return
        except Exception as error:
            if verify_only or attempt == attempts or not transient(error):
                raise
            delay = min(retry_delay * 2**(attempt-1), 120)
            print(f'Retrying {row["path"]} after {type(error).__name__}; '
                  f'attempt {attempt+1}/{attempts} in {delay:g}s', flush=True)
            time.sleep(delay)


def prepare(storage, group, *, workers=4, attempts=5, retry_delay=5,
            status_file=None, verify_only=False):
    storage = Path(storage).expanduser().resolve()
    rows = asset_rows(group)
    os.environ.setdefault('HF_XET_CACHE', str(storage/'assets/.xet-cache'))
    record = dict(group=group, storage=str(storage), status='running',
                  total_files=len(rows), total_bytes=sum(row['bytes'] for row in rows),
                  completed_files=0, completed_bytes=0, started_at=time.time(),
                  manifest_sha256=hashlib.sha256(
                      json.dumps(rows, sort_keys=True).encode()).hexdigest())

    def save():
        if status_file:
            write_json(status_file, record)

    save()
    print(f'{group}: {len(rows)} files, {record["total_bytes"]/1e9:.3f} GB; '
          f'{workers} workers; destination {storage}', flush=True)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(download_with_retry, storage, row,
                                   attempts=attempts, retry_delay=retry_delay,
                                   verify_only=verify_only): row for row in rows}
            try:
                for future in as_completed(pending):
                    row = pending[future]
                    future.result()
                    record['completed_files'] += 1
                    record['completed_bytes'] += row['bytes']
                    record['last_verified'] = row['path']
                    save()
                    print(f'{group}: verified {record["completed_files"]}/{len(rows)} '
                          f'({record["completed_bytes"]/1e9:.3f} GB)', flush=True)
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__,
                      failed_path=row['path'] if 'row' in locals() else None,
                      finished_at=time.time())
        save()
        raise
    record.update(status='complete', finished_at=time.time())
    save()
    print(f'{group}: all {len(rows)} files checksum-verified.', flush=True)
    return record


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('dataset', 'support'), required=True)
    parser.add_argument('--storage', type=Path, required=True)
    parser.add_argument('--workers', type=positive, default=4)
    parser.add_argument('--attempts', type=positive, default=5)
    parser.add_argument('--retry-delay', type=positive, default=5)
    parser.add_argument('--status-file', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.dry_run:
        rows = asset_rows(args.group)
        print(json.dumps(dict(group=args.group, files=len(rows),
                              bytes=sum(row['bytes'] for row in rows),
                              paths=[row['path'] for row in rows]), indent=2))
        return
    try:
        prepare(args.storage, args.group, workers=args.workers,
                attempts=args.attempts, retry_delay=args.retry_delay,
                status_file=args.status_file, verify_only=args.verify_only)
    except Exception as error:
        # Do not dump signed download URLs or credentials from client exceptions.
        raise SystemExit(f'Asset preparation failed: {type(error).__name__}. '
                         'See the last file/status record; access and checksum '
                         'failures are never bypassed.') from None


if __name__ == '__main__':
    main()
