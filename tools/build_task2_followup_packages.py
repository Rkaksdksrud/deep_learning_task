"""Package completed, independently validated Task2 follow-up holdout models.

Copies only the original inference checkpoint and an explicit five-file
allowlist. Never copies dataset images, labels, split tables, predictions,
notebooks, PDFs or resumable optimizer states.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/task2-followups-20261118"
REPOSITORY = "Rkaksdksrud/deep_learning_task"
TAG = "task2-followups-20261118-v1"
SPECS = [
    ("googlenet_20261118_geometry_f0_s42", "googlenet", "googlenet", 224, 0),
    ("vgg11_20261118_geometry_f0_s42", "vgg", "vgg11", 224, 0),
    ("resnet18_20261118_320_f0_s42", "resnet", "resnet18", 320, 0),
    ("resnet18_20261118_320_f1_s42", "resnet", "resnet18", 320, 1),
    ("vgg11_20261118_lr1e4_f0_s42", "vgg", "vgg11", 224, 0),
    ("vgg11_20261118_lr1e4_f1_s42", "vgg", "vgg11", 224, 1),
]
METRIC_KEYS = ("balanced_accuracy", "sensitivity", "specificity", "auroc", "accuracy",
               "confusion_matrix", "samples", "threshold")
CONFIG_KEYS = ("seed", "n_splits", "val_fold", "image_size", "batch_size", "epochs", "patience",
               "learning_rate", "weight_decay", "class_weighted_loss", "aux_loss_weight",
               "threshold", "geometric_augmentation")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(source):
    ready, omitted = [], []
    execution_path=source / "research_20261118/execution_status.json"
    execution=read_json(execution_path) if execution_path.exists() else {}
    for run_id, model_name, architecture, image_size, fold in SPECS:
        run_dir = source / "runs" / run_id
        if not run_dir.is_dir():
            status="not_run_predeclared_gate_not_met" if execution.get("runs", {}).get(run_id, {}).get("status")=="not_run" else "not_run_or_not_created"
            omitted.append({"id": run_id, "status": status})
            continue
        if not (run_dir / "training_summary.json").is_file():
            omitted.append({"id": run_id, "status": "training_not_completed"})
            continue
        if not (run_dir / "validation/metrics.json").is_file():
            omitted.append({"id": run_id, "status": "awaiting_independent_validation"})
            continue
        summary = read_json(run_dir / "training_summary.json")
        best = read_json(run_dir / "best_metrics.json")
        config = read_json(run_dir / "config.json")
        validation = read_json(run_dir / "validation/metrics.json")
        if validation.get("all_checks_passed") is not True:
            raise ValueError(f"Independent validation failed: {run_id}")
        expected = {"model_name": model_name, "architecture": architecture, "smoke_test": False,
                    "image_size": image_size, "val_fold": fold, "seed": 42, "threshold": 0.5,
                    "geometric_augmentation": True, "class_weighted_loss": True,
                    "learning_rate": 0.0001 if "_lr1e4_" in run_id else 0.0003,
                    "batch_size": 16, "epochs": 30, "patience": 7, "weight_decay": 0.0001,
                    "n_splits": 5, "num_workers": 0, "aux_loss_weight": 0.3}
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"Unexpected {key} for {run_id}")
        if summary.get("smoke_test") is not False or summary.get("model_name") != model_name or best.get("smoke_test") is not False:
            raise ValueError(f"Smoke or inconsistent summary for {run_id}")
        if validation.get("selected_epoch") != best["epoch"] or not 1 <= best["epoch"] <= summary["epochs_completed"]:
            raise ValueError(f"Selected epoch mismatch: {run_id}")
        if validation.get("architecture") != architecture or validation.get("smoke_test") is not False:
            raise ValueError(f"Validated architecture mismatch: {run_id}")
        for key in ("balanced_accuracy", "sensitivity", "specificity"):
            if not math.isclose(best[key], validation["metrics"][key], abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"Validated {key} differs: {run_id}")
        if best["confusion_matrix"] != validation["metrics"]["confusion_matrix"] or best["samples"] != validation["metrics"]["samples"]:
            raise ValueError(f"Validated confusion matrix/sample count differs: {run_id}")
        # Preserve inference checkpoint bytes; reject machine-specific path
        # configs rather than silently rewriting binary metadata.
        for key in ("data_root", "output_root"):
            if Path(config[key]).is_absolute() or "\\" in config[key]:
                raise ValueError(f"Machine-specific checkpoint configuration requires review: {run_id}/{key}")
        weights = run_dir / f"{model_name}.pt"
        if not weights.is_file():
            raise ValueError(f"Completed checkpoint missing: {run_id}")
        release_validation = read_json(run_dir / "release_validation/metrics.json")
        if release_validation.get("all_checks_passed") is not True or release_validation.get("checkpoint_bytes_unchanged_during_validation") is not True or release_validation.get("checkpoint_sha256") != sha256(weights):
            raise ValueError(f"Release-time validation is not bound to these checkpoint bytes: {run_id}")
        environment = read_json(run_dir / "environment.json")
        source_hashes = read_json(run_dir / "source_manifest.json")
        code_hashes = {name: digest for name, digest in source_hashes.items()
                       if name.startswith("code/") and Path(name).name == name[len("code/"):] and name.endswith(".py")}
        peak = []
        with (run_dir / "history.csv").open(newline="", encoding="utf-8") as stream:
            history_rows = list(csv.DictReader(stream))
        if len(history_rows) != summary["epochs_completed"]:
            raise ValueError(f"Completed history length mismatch: {run_id}")
        for row in history_rows:
            if row.get("peak_gpu_memory_mib"):
                peak.append(float(row["peak_gpu_memory_mib"]))
        metadata = {
            "run_id": run_id, "model_name": model_name, "architecture": architecture,
            "checkpoint_filename": "model.pt", "checkpoint_sha256": sha256(weights),
            "initialization": "random; weights=None", "configuration": {key: config[key] for key in CONFIG_KEYS},
            "selected_epoch": best["epoch"], "epochs_completed": summary["epochs_completed"],
            "training_elapsed_seconds": summary["elapsed_seconds"],
            "peak_gpu_memory_mib": max(peak) if peak else None,
            "validation_metrics": {key: best[key] for key in METRIC_KEYS},
            "original_independent_validation_passed": True,
            "release_validation_bound_to_checkpoint_sha256": True,
            "all_data_final_model": False, "private_test_evaluated": False,
            "training_data_scope": "provided training partition only; patient-disjoint holdout",
            "checkpoint_is_inference_only": True, "training_resume_state_included": False,
            "environment_versions": {key: value for key, value in environment.items() if key != "runtime"},
            "training_code_sha256": code_hashes, "release_tag": TAG,
        }
        ready.append((run_id, weights, metadata))
    return ready, omitted


def build(source, output):
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Release archives must be outside the Git repository")
    if output.exists():
        raise FileExistsError("Preserve existing packages; choose a new output directory")
    ready, omitted = collect(source)
    if not ready:
        raise ValueError("No completed, independently validated follow-up models are available")
    output.mkdir(parents=True)
    rows = []
    for run_id, weights, metadata in ready:
        metrics = metadata["validation_metrics"]
        readme = f"""# Task2 {metadata['architecture']} · {run_id}

이 패키지는 제공 데이터의 환자별 Train partition으로 학습한 단일 holdout 모델입니다.
전체 데이터 최종 제출 모델이나 private Test 성능이 아닙니다.

```bash
python -m pip install -r requirements.txt
python predict.py --images input_images --output predictions.csv --device cpu
```

가중치는 원본 checkpoint 바이트를 그대로 `model.pt`에 담았습니다. CUDA 환경에서는 `--device cuda`를 쓸 수 있습니다.
RGB bilinear letterbox, 고정 normalization0.5/0.5, Present=1/Absent=0, threshold0.5입니다.
학습 증강은 추론에 적용하지 않습니다. 각 ZIP은 독립 모델이며 여러 모델을 합쳐 쓰지 않습니다.

- 검증 BA {metrics['balanced_accuracy']:.6%}; Present recall {metrics['sensitivity']:.6%}; Absent recall {metrics['specificity']:.6%}.
- 입력 {metadata['configuration']['image_size']}px; fold {metadata['configuration']['val_fold']}; seed {metadata['configuration']['seed']}; 선택 epoch {metadata['selected_epoch']}.
- 원본 가중치 SHA256: `{metadata['checkpoint_sha256']}`.
- inference checkpoint만 포함하며 optimizer/scheduler/RNG 재개 상태는 없습니다.
- 원본 영상·라벨·환자 ID·분할표·개별 예측·과제 PDF를 포함하지 않습니다.
- 원 실험의 독립 재검증은 통과했습니다. 다른 backend의 확률 bit parity까지 보장하지는 않습니다.

코드와 비교 결과: https://github.com/{REPOSITORY}/tree/ktg_tried
"""
        contents = {"predict.py": (MODEL_DIR / "predict.py").read_bytes(),
                    "requirements.txt": (MODEL_DIR / "requirements.txt").read_bytes(),
                    "metadata.json": json_bytes(metadata), "README.md": readme.encode("utf-8")}
        checks = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
        checks["model.pt"] = metadata["checkpoint_sha256"]
        contents["SHA256SUMS.json"] = json_bytes(checks)
        archive = output / f"{run_id}.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
            for name in sorted([*contents, "model.pt"]):
                info = zipfile.ZipInfo(name, date_time=(2026, 11, 18, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                if name == "model.pt":
                    with weights.open("rb") as source_stream, stream.open(info, "w", force_zip64=True) as target:
                        shutil.copyfileobj(source_stream, target, 1024 * 1024)
                else:
                    stream.writestr(info, contents[name])
        if sha256(weights) != metadata["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed while packaging: {run_id}")
        row = {"id": run_id, "architecture": metadata["architecture"],
               "image_size": metadata["configuration"]["image_size"],
               "validation_fold": metadata["configuration"]["val_fold"], "seed": metadata["configuration"]["seed"],
               "selected_epoch": metadata["selected_epoch"],
               **{key: metrics[key] for key in ("balanced_accuracy", "sensitivity", "specificity")},
               "checkpoint_filename": "model.pt", "checkpoint_sha256": metadata["checkpoint_sha256"],
               "asset_name": archive.name, "archive_size_bytes": archive.stat().st_size,
               "archive_sha256": sha256(archive),
               "download_url": f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{archive.name}"}
        rows.append(row)
        print(f"{run_id}: {archive.stat().st_size:,} bytes SHA256={row['archive_sha256']}", flush=True)
    manifest = {"format_version": 1, "repository": REPOSITORY, "release_tag": TAG,
                "checkpoint_filename": "model.pt", "total_models": len(rows), "models": rows,
                "omitted_runs": omitted, "contains_training_data": False, "contains_ensembles": False,
                "notes": ["Completed patient-disjoint holdout experiments; not all-data final submission models",
                          "Downloads become available only after the named release is published",
                          "Conditional fold1/320 experiment may be omitted when its preregistered gate is not met"]}
    (MODEL_DIR / "manifest.json").write_bytes(json_bytes(manifest))
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    for record in omitted:
        print(f"Omitted {record['id']}: {record['status']}", flush=True)
    print(f"Built {len(rows)} model packages", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-task2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory outside the Git repository")
    args = parser.parse_args()
    build(args.source_task2.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
