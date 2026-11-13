"""Build seven model-only release archives from completed local Task2 runs.

The source training data, per-image records, PDFs and notebooks are never copied.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models/task2-resnet18-20261111'
REPO = 'Rkaksdksrud/deep_learning_task'
TAG = 'task2-resnet18-20261111-v1'
IDS = ['control_f0_s42', 'plain_ce_f0_s42', 'geometry_f0_s42', 'control_f0_s43',
       'geometry_f0_s43', 'control_f1_s42', 'geometry_f1_s42']

def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()

def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-task2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new directory outside the Git repository')
    args = parser.parse_args()
    source, output = args.source_task2.resolve(), args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('release archives must be outside the Git repository')
    if output.exists():
        parser.error('preserve existing archives; choose a new output directory')
    output.mkdir(parents=True)
    expected_hashes = json.loads((source / 'research_20261111/checkpoint_sha256.json').read_text())
    rows = []
    for run_id in IDS:
        run_name = 'resnet18_20261111_' + run_id
        run_dir = source / 'runs' / run_name
        weights = run_dir / 'resnet.pt'
        weight_hash = sha256(weights)
        if weight_hash != expected_hashes[f'runs/{run_name}/resnet.pt']:
            raise ValueError(f'Original checkpoint changed: {run_id}')
        best = json.loads((run_dir / 'best_metrics.json').read_text())
        config = json.loads((run_dir / 'config.json').read_text())
        checked = json.loads((run_dir / 'validation/metrics.json').read_text())
        if checked['all_checks_passed'] is not True or config['smoke_test']:
            raise ValueError(f'Unverified or smoke checkpoint: {run_id}')
        aggregate_keys = ['balanced_accuracy', 'sensitivity', 'specificity', 'auroc', 'accuracy', 'confusion_matrix', 'samples', 'threshold']
        metadata = {'run_id': run_id, 'architecture': 'resnet18', 'initialization': 'random; weights=None',
                    'class_weighted_loss': config['class_weighted_loss'],
                    'geometric_augmentation': config.get('geometric_augmentation', False),
                    'seed': config['seed'], 'validation_fold': config['val_fold'],
                    'selected_epoch': best['epoch'], 'training_data_scope': 'provided training partition only; patient-disjoint validation',
                    'all_data_final_model': False, 'checkpoint_sha256': weight_hash,
                    'original_independent_validation_passed': True,
                    'validation_metrics': {k: best[k] for k in aggregate_keys},
                    'training_state_resume_supported': False,
                    'release_tag': TAG}
        readme = f'''# Task2 ResNet18 · {run_id}

이 ZIP은 독립 실행 가능한 단일 모델 추론 패키지입니다. 입력 영상은 사용자가 별도로 준비합니다.

```bash
python -m pip install -r requirements.txt
python predict.py --images input_images --output predictions.csv --device cpu
```

CUDA가 준비된 환경에서는 `--device cuda`를 사용할 수 있습니다. 출력은 sample_id/path/probability/prediction입니다.
Present=1, Absent=0, 임계값0.5이며 학습 증강을 추론에 적용하지 않습니다.

- 환자 검증 BA: {best['balanced_accuracy']:.6%}, Present recall {best['sensitivity']:.6%}, Absent recall {best['specificity']:.6%}.
- fold {config['val_fold']}, seed {config['seed']}, 선택 epoch {best['epoch']}.
- 이 모델은 제공 데이터의 Train partition으로만 학습한 holdout 실험 모델입니다. 전체 데이터 최종 모델이나 private Test 성능이 아닙니다.
- 모델은 무작위 초기화했으며 외부 가중치·앙상블·추가 신경망을 사용하지 않았습니다.
- checkpoint는 모델 상태와 metadata를 포함하지만 optimizer/scheduler/RNG 학습 재개 상태는 없습니다.
- SHA256SUMS.json으로 패키지 각 파일을 확인할 수 있습니다. 원본 checkpoint SHA256은 {weight_hash} 입니다.
- 원본 영상·라벨·개별 예측·환자 분할표는 포함하지 않습니다.

전체 코드·실험 비교: https://github.com/{REPO}/tree/ktg_tried
데이터를 별도로 보유한 팀원은 저장소의 task2/code/evaluate_released.py를 사용해 검증 partition을 재생성하고 aggregate 지표를 비교할 수 있습니다.
원래 저장한 표본별 확률의 bit parity를 다시 확인하는 기능과는 구분합니다.
'''
        contents = {'predict.py': (MODEL_DIR / 'predict.py').read_bytes(),
                    'requirements.txt': (MODEL_DIR / 'requirements.txt').read_bytes(),
                    'metadata.json': json_bytes(metadata), 'README.md': readme.encode()}
        hashes = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
        hashes['resnet.pt'] = weight_hash
        contents['SHA256SUMS.json'] = json_bytes(hashes)
        archive = output / f'{run_id}.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name in sorted([*contents, 'resnet.pt']):
                info = zipfile.ZipInfo(name, date_time=(2026,11,11,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                if name == 'resnet.pt':
                    with weights.open('rb') as src, z.open(info, 'w', force_zip64=True) as dst:
                        import shutil
                        shutil.copyfileobj(src, dst, 1024*1024)
                else:
                    z.writestr(info, contents[name])
        row = {'id': run_id, 'recommended': run_id == 'geometry_f0_s42', 'architecture': 'resnet18',
               'validation_fold': config['val_fold'], 'seed': config['seed'], 'selected_epoch': best['epoch'],
               **{k:best[k] for k in ['balanced_accuracy','sensitivity','specificity']},
               'asset_name': archive.name, 'archive_size_bytes': archive.stat().st_size,
               'archive_sha256': sha256(archive), 'checkpoint_sha256': weight_hash,
               'download_url': f'https://github.com/{REPO}/releases/download/{TAG}/{archive.name}'}
        rows.append(row)
        print(f'{run_id}: {archive.stat().st_size:,} bytes SHA256={row["archive_sha256"]}', flush=True)
    manifest = {'format_version': 1, 'repository': REPO, 'release_tag': TAG,
                'recommended_run': 'geometry_f0_s42', 'total_models': len(rows),
                'contains_training_data': False, 'contains_ensembles': False,
                'notes': ['all seven are completed patient-holdout experiments, not all-data final submission models',
                          'download URLs require the named release to be published'], 'models': rows}
    (MODEL_DIR / 'manifest.json').write_bytes(json_bytes(manifest))
    (output / 'manifest.json').write_bytes(json_bytes(manifest))
    print(f'Built {len(rows)} archives; total {sum(r["archive_size_bytes"] for r in rows):,} bytes')

if __name__ == '__main__':
    main()
