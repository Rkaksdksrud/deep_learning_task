"""Bounded sequential GPU comparisons with verified-run skipping.

This runner never resumes optimizer/scheduler/RNG state. An interrupted training
run is preserved; use --only and a new --run-suffix to restart it from epoch 1.
Example: python code/run_comparison.py --phase initial --only geometry --run-suffix retry1
Read research_20261111/EXPERIMENT_PLAN.md before launching GPU training.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from common import TASK_DIR, load_config, task_path, train, write_json
from data import load_records, make_splits
from validate import read_manifest, validate_checkpoint


CRITICAL_SOURCES = ('code/common.py', 'code/data.py', 'code/resnet.py', 'code/validate.py')


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_suffix(value: str) -> str:
    """A suffix is one short portable name, without paths or shell syntax."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise argparse.ArgumentTypeError(
            '--run-suffix must be 1-64 letters/digits/underscores/hyphens and start with a letter or digit')
    return value


def verified_run_warnings(run_dir: Path, expected_config: dict) -> list[str]:
    """Reject stale verification before skipping; never alter an existing run.

    This checks current CSV metadata, the exact requested patient split, and the
    critical source hashes plus their archived copies. Noncritical code changes
    are disclosed as warnings, because changes to this runner do not change an
    already trained checkpoint. Existing runs without these provenance files
    need manual review and cannot be silently skipped.
    """
    required = ['config.json', 'source_manifest.json', 'split_manifest.csv',
                'training_summary.json', 'best_metrics.json', 'resnet.pt',
                'validation/metrics.json']
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f'missing completed-run provenance: {missing}')
    saved = json.loads((run_dir / 'config.json').read_text())
    mismatches = [key for key, value in expected_config.items()
                  if saved.get(key, False if key == 'geometric_augmentation' else None) != value]
    if mismatches:
        raise ValueError(f'configuration differs: {mismatches}')
    if saved.get('smoke_test') is not False or saved.get('model_name') != 'resnet':
        raise ValueError('expected a full ResNet run, not smoke weights')
    validation = json.loads((run_dir / 'validation/metrics.json').read_text())
    if validation.get('all_checks_passed') is not True:
        raise ValueError('independent validation did not pass')
    if Path(validation.get('weights', '')).resolve() != (run_dir / 'resnet.pt').resolve():
        raise ValueError('validation refers to a different checkpoint path')
    best = json.loads((run_dir / 'best_metrics.json').read_text())
    summary = json.loads((run_dir / 'training_summary.json').read_text())
    if validation.get('selected_epoch') != best.get('epoch'):
        raise ValueError('validation epoch differs from the selected checkpoint')
    if summary.get('epochs_completed', 0) < best.get('epoch', 1):
        raise ValueError('training completion record does not cover the selected epoch')

    current_records = load_records(task_path(expected_config['data_root']))
    manifest = read_manifest(run_dir / 'split_manifest.csv', current_records)
    expected_split = make_splits(current_records, expected_config['n_splits'], expected_config['val_fold'])
    columns = list(expected_split.columns)
    try:
        pd.testing.assert_frame_equal(
            manifest.sort_values('sample_id')[columns].reset_index(drop=True),
            expected_split.sort_values('sample_id')[columns].reset_index(drop=True),
            check_dtype=False)
    except AssertionError as exc:
        raise ValueError('saved patient split differs from the requested split') from exc

    source_manifest = json.loads((run_dir / 'source_manifest.json').read_text())
    for relative in CRITICAL_SOURCES:
        expected_sha = source_manifest.get(relative)
        current = TASK_DIR / relative
        archived = run_dir / 'source' / Path(relative).name
        if not isinstance(expected_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_sha):
            raise ValueError(f'missing or invalid critical SHA256: {relative}')
        if not current.is_file() or sha256_file(current) != expected_sha:
            raise ValueError(f'training-critical source changed: {relative}')
        if not archived.is_file() or sha256_file(archived) != expected_sha:
            raise ValueError(f'archived training-critical source is missing or changed: {relative}')
    labels_path = task_path(expected_config['data_root']) / 'labels.csv'
    if source_manifest.get('labels.csv') != sha256_file(labels_path):
        raise ValueError('labels.csv SHA256 differs from the completed run')

    warnings = []
    current_code = {str(path.relative_to(TASK_DIR)): path for path in (TASK_DIR / 'code').glob('*.py')}
    noncritical = (set(current_code) | {key for key in source_manifest if key.startswith('code/')}) - set(CRITICAL_SOURCES)
    for relative in sorted(noncritical):
        path = current_code.get(relative)
        if path is None or source_manifest.get(relative) != sha256_file(path):
            warnings.append(f'noncritical source differs: {relative}; trained checkpoint is preserved')
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['initial', 'confirm'], required=True)
    parser.add_argument('--candidate', choices=['plain_ce', 'geometry'])
    parser.add_argument('--only', choices=['control', 'plain_ce', 'geometry'],
                        help='run only this kind; initial control is the baseline repeat')
    parser.add_argument('--run-suffix', type=safe_suffix,
                        help='new name suffix for a restart from epoch 1; existing runs are preserved')
    args = parser.parse_args()
    if args.phase == 'confirm' and args.candidate is None:
        parser.error('--candidate is required for confirmation')
    root = TASK_DIR / 'research_20261111'
    root.mkdir(exist_ok=True)
    plan = [('plain_ce', 0, 42), ('geometry', 0, 42)] if args.phase == 'initial' else [
        (kind, fold, seed) for fold, seed in [(0, 43), (1, 42)]
        for kind in ['control', args.candidate]]
    if args.only:
        if args.phase == 'initial' and args.only == 'control':
            plan = [('control', 0, 42)]
        else:
            plan = [entry for entry in plan if entry[0] == args.only]
        if not plan:
            parser.error('--only does not belong to this phase/candidate plan')
    suite_name = f'{args.phase}_{args.candidate or "candidates"}'
    if args.only:
        suite_name += f'_only_{args.only}'
    if args.run_suffix:
        suite_name += f'_{args.run_suffix}'
    suite_path = root / f'{suite_name}_status.json'
    status = {'phase': args.phase, 'candidate': args.candidate, 'only': args.only,
              'run_suffix': args.run_suffix, 'pid': os.getpid(),
              'started_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
              'restart_semantics': 'new suffix starts at epoch 1; no optimizer/scheduler/RNG resume',
              'gpu_jobs_at_once': 1, 'runs': []}
    for kind, fold, seed in plan:
        name = f'resnet18_20261111_{kind}_f{fold}_s{seed}'
        if args.run_suffix:
            name += f'_{args.run_suffix}'
        config = load_config()
        config.update(device='cuda', seed=seed, val_fold=fold,
                      class_weighted_loss=kind != 'plain_ce', geometric_augmentation=kind == 'geometry')
        status['runs'].append({'name': name, 'config': config, 'status': '계획'})
    # Preserve the previous status history as well as checkpoint directories.
    if suite_path.exists():
        previous = suite_path.with_name(f'{suite_path.stem}_{datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")}.json')
        previous.write_bytes(suite_path.read_bytes())
        status['previous_status_copy'] = str(previous)
    write_json(suite_path, status)
    for item in status['runs']:
        run_dir = task_path(item['config']['output_root']) / item['name']
        try:
            if run_dir.exists():
                try:
                    warnings = verified_run_warnings(run_dir, item['config'])
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    raise FileExistsError(
                        f'Unverified or different existing run; preserve and inspect: {run_dir}. '
                        f'Use --only and a new --run-suffix to restart from epoch 1. Reason: {exc}') from exc
                item['status'] = '검증 완료'
                item['resumed_by_skipping_verified_run'] = True
                item['provenance_warnings'] = warnings
                item['best_metrics'] = json.loads((run_dir / 'best_metrics.json').read_text())
                write_json(suite_path, status)
                for warning in warnings:
                    print(f'WARNING {item["name"]}: {warning}', flush=True)
                print(f'SKIP verified {item["name"]}', flush=True)
                continue
            print(f'START {item["name"]}', flush=True)
            item['status'] = '실행 중'
            write_json(suite_path, status)
            with (root / f'{item["name"]}_console.log').open('x') as stream, contextlib.redirect_stdout(stream):
                train('resnet', item['config'], run_name=item['name'])
                item['status'] = '실행 완료'
                write_json(suite_path, status)
                validate_checkpoint(run_dir / 'resnet.pt', device='cuda')
            item['status'] = '검증 완료'
            item['best_metrics'] = json.loads((run_dir / 'best_metrics.json').read_text())
            write_json(suite_path, status)
            print(f'VERIFIED {item["name"]} BA={item["best_metrics"]["balanced_accuracy"]:.6f}', flush=True)
        except BaseException as exc:
            item['status'] = '중단/실패'
            item['error'] = repr(exc)
            write_json(suite_path, status)
            traceback.print_exc()
            raise
    status['finished_kst'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    write_json(suite_path, status)
    print(f'FINISHED {suite_path}', flush=True)


if __name__ == '__main__':
    main()
