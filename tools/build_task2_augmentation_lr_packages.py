"""Export sanitized Task2 2026-11-25 records and eight portable inference ZIPs.

Original local checkpoints are immutable. Published copies change only the two
machine-specific config paths; tensor values and prediction metadata stay exact.
No dataset, sample/patient records or resumable optimizer/RNG state is exported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
from statistics import mean, stdev
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/task2-augmentation-lr-20261125"
EXPERIMENT = ROOT / "experiments/task2-augmentation-lr-20261125"
REPOSITORY = "Rkaksdksrud/deep_learning_task"
TAG = "task2-augmentation-lr-20261125-v1"
PORTABLE_CHECKPOINT_PATHS = {"data_root": "../data/task2", "output_root": "../runs/task2"}
PORTABLE_TRAINING_PATHS = {"data_root": "../../../data/task2", "output_root": "../../../runs/task2/20261125"}
CHECKPOINT_KEYS = {"format_version", "model_name", "architecture", "state_dict", "config", "epoch", "valid_metrics",
                   "threshold", "class_weights", "validation_class_weights", "smoke_test", "initialization", "label_mapping", "preprocessing"}
METRICS = {"balanced_accuracy", "accuracy", "sensitivity", "specificity", "precision", "f1", "auroc", "confusion_matrix",
           "samples", "threshold", "loss", "elapsed_seconds", "images_per_second"}
PAIR_KEYS = ("procedure", "axis", "role", "fold", "seed", "source_scope", "rule_status", "control_id", "candidate_id",
             "config_differences", "control_ba", "candidate_ba", "delta_ba", "positive_recall_delta", "negative_recall_delta",
             "training_time_ratio", "control_training_seconds", "candidate_training_seconds", "time_scope", "improved", "tied", "worsened")
AXIS_KEYS = ("axis", "name", "change", "matched_conditions", "mean_control_ba", "mean_candidate_ba", "mean_delta_ba",
             "delta_sample_sd", "positive_conditions", "numeric_adoption_gate_passed", "submission_eligible", "conclusion", "variability_scope")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def safe(value):
    if isinstance(value, str):
        require(not re.search(r"/home/|/Users/|/tmp/|/opt/|(?<![A-Za-z])[A-Za-z]:[\\/]|\bG\d{2}\b|github_pat_|gh[pousr]_[A-Za-z0-9]{20,}", value),
                "Private-looking identifier or machine path in exported content")
    elif isinstance(value, dict):
        for key, item in value.items():
            safe(key)
            safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            safe(item)
    else:
        require(value is None or type(value) in (int, float, bool), "Unexpected export metadata type")
        if type(value) is float:
            require(math.isfinite(value), "Nonfinite exported metadata")


def write_json(path, value):
    safe(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(json_bytes(value))


def metric_check(best, measured):
    require(set(best) == METRICS | {"epoch", "smoke_test"}, "Unexpected best metric schema")
    for key in ("balanced_accuracy", "sensitivity", "specificity", "accuracy"):
        require(math.isclose(best[key], measured[key], abs_tol=1e-12, rel_tol=0), "Validation aggregate metric differs")
    require(best["confusion_matrix"] == measured["confusion_matrix"] and best["samples"] == measured["samples"] and
            best["threshold"] == measured["threshold"] == .5 and
            math.isclose(best["loss"], measured["loss"], abs_tol=1e-5, rel_tol=1e-5), "Validation support/loss differs")


def state_equal(a, b, torch):
    require(a.keys() == b.keys() and getattr(a, "__dict__", {}) == getattr(b, "__dict__", {}), "State dictionary structure differs")
    require(all(isinstance(value, torch.Tensor) and isinstance(b[key], torch.Tensor) and value.dtype == b[key].dtype and
                value.shape == b[key].shape and torch.equal(value, b[key]) for key, value in a.items()),
            "A model tensor changed during portable repackaging")


def portable_checkpoint(original, destination, config, best, torch):
    original_sha = sha(original)
    checkpoint = torch.load(original, map_location="cpu", weights_only=True)
    expected_config = {key: value for key, value in config.items() if key not in {"model_name", "architecture", "smoke_test"}}
    require(set(checkpoint) == CHECKPOINT_KEYS and checkpoint["config"] == expected_config and checkpoint["format_version"] == 1 and
            checkpoint["model_name"] == "resnet" and checkpoint["architecture"] == "resnet18" and checkpoint["smoke_test"] is False and
            checkpoint["initialization"] == "random; weights=None" and checkpoint["label_mapping"] == {0: "absent", 1: "present"} and
            checkpoint["threshold"] == .5 and checkpoint["epoch"] == best["epoch"], "Unexpected original inference checkpoint schema")
    require(checkpoint["valid_metrics"] == {key: value for key, value in best.items() if key not in {"epoch", "smoke_test"}},
            "Original checkpoint metrics differ")
    expected_preprocessing = {"version": 1, "color_mode": "RGB", "resize": "aspect_preserving_letterbox", "interpolation": "bilinear",
                              "padding_color": [0, 0, 0], "pixel_scale": 255.0, "mean": [.5] * 3, "std": [.5] * 3, "image_size": 224}
    require(checkpoint["preprocessing"] == expected_preprocessing and len(checkpoint["state_dict"]) == 122,
            "Unexpected ResNet18 preprocessing/state dictionary")
    require(tuple(checkpoint["state_dict"]["fc.weight"].shape) == (2, 512), "Classifier is not binary ResNet18")
    original_config = dict(checkpoint["config"])
    published = {**checkpoint, "config": {**original_config, **PORTABLE_CHECKPOINT_PATHS}}
    safe({key: value for key, value in published.items() if key != "state_dict"})
    safe(getattr(published["state_dict"], "__dict__", {}))
    require(all(isinstance(value, torch.Tensor) for value in published["state_dict"].values()), "Unexpected parameter payload")
    torch.save(published, destination)
    reloaded = torch.load(destination, map_location="cpu", weights_only=True)
    state_equal(checkpoint["state_dict"], reloaded["state_dict"], torch)
    require(set(reloaded) == set(checkpoint) and {key for key in original_config if original_config[key] != reloaded["config"][key]} ==
            set(PORTABLE_CHECKPOINT_PATHS) and set(reloaded["config"]) == set(original_config), "Metadata repack changed an unexpected config field")
    require(all(reloaded[key] == checkpoint[key] for key in checkpoint if key not in {"state_dict", "config"}),
            "Repack changed prediction/selection metadata")
    require(sha(original) == original_sha, "Original checkpoint bytes changed")
    result = {"original_checkpoint_sha256": original_sha, "published_checkpoint_sha256": sha(destination),
              "changed_metadata_fields": ["config.data_root", "config.output_root"], "all_122_model_tensors_exact": True,
              "other_metadata_exact": True, "original_local_checkpoint_preserved": True,
              "published_checkpoint_bytes_equal_original": False,
              "training_class_weights": checkpoint["class_weights"], "validation_class_weights": checkpoint["validation_class_weights"]}
    del checkpoint, published, reloaded
    return result


def load_suite(source):
    here = source / "research_20261125"
    spec, status = read(here / "experiment_specs.json"), read(here / "execution_status.json")
    integrity = read(here / "FINAL_INTEGRITY.json")
    require(status["controller_status"] == "completed" and integrity["all_checks_passed"] is True and
            status["verified_full_runs"] == integrity["verified_new_states_checked"] == 8, "Incomplete or unaudited experiment suite")
    entries = spec["primary_runs"] + spec["optional_runs"]
    verified = {name for name, record in status["runs"].items() if record["status"] == "verified"}
    require(verified == {record["id"] for record in integrity["new_state_checks"]} and len(verified) == 8,
            "Actual completed model IDs differ from the final audit")
    original, frozen, inputs = (read(here / name) for name in ("original_source.json", "frozen_source.json", "suite_inputs.json"))
    require(len(original) == len(frozen) == 13 and len(inputs) == 18 and
            all(sha(source / name) == digest for name, digest in original.items()) and
            all(sha(here / "workspace" / name) == digest for name, digest in frozen.items()) and
            all(sha(here / name) == digest for name, digest in inputs.items()), "Frozen source/input preservation differs")
    ready, omitted = [], []
    for item in entries:
        run_id = item["run_id"]
        if run_id not in verified:
            require(status["runs"][run_id]["status"] == "not_run", "Unexpected unfinished optional run")
            omitted.append({"id": run_id, "axis": item["axis"], "status": "not_run_predeclared_gate", "role": item["role"]})
            continue
        folder = source / "runs" / run_id
        config, best, summary = (read(folder / name) for name in ("config.json", "best_metrics.json", "training_summary.json"))
        expected = {**spec["base_config"], **item["changed"], "val_fold": item["fold"], "seed": item["seed"]}
        require(read(here / item["config_file"]) == expected and config == {**expected, "resolved_device": "cuda", "model_name": "resnet",
                "architecture": "resnet18", "smoke_test": False}, "Run differs from the frozen one-factor specification")
        validation = read(folder / "validation/metrics.json")
        original_sha = sha(folder / "resnet.pt")
        require(validation["all_checks_passed"] is True and validation["checkpoint_bytes_unchanged_during_validation"] is True and
                validation["checkpoint_sha256"] == status["runs"][run_id]["checkpoint_sha256"] == original_sha,
                "Independent validation is not bound to the original checkpoint bytes")
        require(status["runs"][run_id]["best_metrics"] == best and status["runs"][run_id]["training_summary"] == summary,
                "Controller/run completion records differ")
        metric_check(best, validation["metrics"])
        hashes = read(folder / "source_manifest.json")
        require(all(hashes[name] == digest and sha(folder / "source" / Path(name).name) == digest for name, digest in frozen.items()),
                "Archived training source differs")
        with (folder / "history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        require([int(row["epoch"]) for row in history] == list(range(1, summary["epochs_completed"] + 1)), "History is incomplete")
        selected = min(history, key=lambda row: (-float(row["valid_balanced_accuracy"]), float(row["valid_loss"]), int(row["epoch"])))
        require(int(selected["epoch"]) == best["epoch"], "Selected epoch does not match the fixed BA/loss rule")
        ready.append((item, folder, config, best, summary, validation, history))
    return here, spec, status, ready, omitted, frozen, integrity


def export_records(here, ready, frozen, integrity, manifest_rows, omitted):
    raw = read(here / "RESULT_METRICS.json")
    require(raw["controller_status"] == "completed" and raw["verified_full_runs"] == 8, "Final comparison report is incomplete")
    result_by_id = {row["run_id"]: row for row in raw["results"]}
    published = {row["id"]: row for row in manifest_rows}
    safe_rows = []
    for item, folder, cfg, best, summary, validation, history in ready:
        run_id = item["run_id"]
        target = EXPERIMENT / "run_records" / run_id
        target.mkdir(parents=True, exist_ok=False)
        recorded = {key: value for key, value in cfg.items() if key not in {"data_root", "output_root"}}
        portable = {key: value for key, value in cfg.items() if key not in {"resolved_device", "model_name", "architecture", "smoke_test"}}
        portable.update(PORTABLE_TRAINING_PATHS)
        write_json(target / "config_recorded.json", recorded)
        write_json(target / "config.json", portable)
        write_json(target / "best_metrics.json", best)
        text = (folder / "history.csv").read_text(encoding="utf-8")
        safe(text)
        (target / "history.csv").write_text(text, encoding="utf-8")
        write_json(target / "source_hashes.json", frozen)
        write_json(target / "training_summary.json", {key: value for key, value in summary.items()
                   if key not in {"weights_path", "resumable_state_path"}})
        environment = read(folder / "environment.json")
        write_json(target / "environment.json", {key: environment[key] for key in ("torch", "torchvision", "numpy", "pandas", "Pillow", "scikit-learn")})
        write_json(target / "validation_checks.json", {"all_checks_passed": True, "checks": validation["checks"],
                   "original_checkpoint_sha256": validation["checkpoint_sha256"], "checkpoint_bytes_unchanged_during_validation": True,
                   "selected_epoch": validation["selected_epoch"], "metrics": validation["metrics"]})
        write_json(target / "checkpoint_repack.json", {key: published[run_id][key] for key in
                   ("original_checkpoint_sha256", "checkpoint_sha256", "published_checkpoint_sha256", "changed_metadata_fields", "all_122_model_tensors_exact")})
        record = result_by_id[run_id]
        safe_rows.append({"run_id": run_id, "axis": item["axis"], "role": item["role"], "rule_status": item["rule_status"],
                          "fold": item["fold"], "seed": item["seed"], "control_id": item["control_id"], "status": "independently_verified",
                          "metrics": {key: best[key] for key in METRICS if key not in {"elapsed_seconds", "images_per_second"}},
                          "selected_epoch": best["epoch"], "epochs_completed": summary["epochs_completed"],
                          "training_seconds": summary["elapsed_seconds"], "peak_gpu_memory_mib": record["peak_gpu_memory_mib"],
                          "max_train_balanced_accuracy": record["train_ba_max"],
                          "worst_defined_patient_ba": record["worst_patient"]["balanced_accuracy"] if record["worst_patient"] else None,
                          "original_checkpoint_sha256": published[run_id]["original_checkpoint_sha256"],
                          "published_checkpoint_sha256": published[run_id]["published_checkpoint_sha256"]})
    pairs = [{key: row[key] for key in PAIR_KEYS} for row in raw["comparisons"]]
    axis_summaries = [{key: row[key] for key in AXIS_KEYS} for row in raw["axis_summaries"]]
    summary = {"format_version": 1, "experiment_date": "2026-11-25", "experiment_id": "rotation_photometric_threshold_20261125",
               "publication_status": "packaged locally; remote publication must be verified separately", "release_tag": TAG,
               "completed_full_runs": 8, "independently_verified_runs": 8, "new_training_seconds": raw["new_training_seconds"],
               "runs": safe_rows, "comparisons": pairs, "axis_summaries": axis_summaries, "omitted_runs": omitted,
               "checkpoint_repackaging": "Only config.data_root/output_root changed in published copies; all model tensors and prediction metadata are exact; original local files are immutable.",
               "config_recorded_note": "Original machine data/output paths are omitted; other recorded values are preserved. config.json is a separate portable recipe for task2/experiments/20261125/code.",
               "threshold_status": "exploratory conditional LOPO diagnosis only; deployment threshold remains .5 and no threshold map is distributed",
               "limitations": raw["limitations"][:-1] + ["Local original source and all twenty older checkpoints were preserved; remote publication is a separate operation."]}
    write_json(EXPERIMENT / "metrics_summary.json", summary)
    write_json(EXPERIMENT / "source_hashes.json", {"original_canonical_source": read(here / "original_source.json"), "isolated_training_source": frozen})
    write_json(EXPERIMENT / "integrity_summary.json", {key: integrity[key] for key in ("all_checks_passed", "historical_checkpoint_count",
               "canonical_source_files_preserved", "workspace_source_files_preserved", "frozen_suite_inputs_preserved", "verified_new_states_checked")})
    comparison = (here / "COMPARISONS.csv").read_text(encoding="utf-8")
    safe(comparison)
    (EXPERIMENT / "comparison.csv").write_text(comparison, encoding="utf-8")
    thresholds = read(here / "threshold_results.json")
    safe_thresholds = {key: thresholds[key] for key in ("status", "method", "threshold_grid", "selection_objective", "tie_break",
                      "candidate_2_rule_status", "unique_validation_patients", "unique_validation_images", "analysis_seconds",
                      "original_inputs_unchanged", "deployment_threshold", "limitations")}
    safe_thresholds["results"] = [{key: row[key] for key in ("run_id", "role", "fixed_threshold", "fixed_05", "crossfitted", "delta_ba",
                                       "deployment_threshold", "protocol_scope")} for row in thresholds["results"]]
    safe_thresholds["mean_delta_ba"] = mean(row["delta_ba"] for row in thresholds["results"])
    safe_thresholds["interpretation"] = "Conditional diagnosis on checkpoints already selected by validation; no deployment threshold/held-patient map and no independent Test claim"
    write_json(EXPERIMENT / "threshold_summary.json", safe_thresholds)
    write_json(EXPERIMENT / "photometric_diagnostics.json", read(here / "photometric_diagnostics.json"))
    write_json(EXPERIMENT / "implementation_checks.json", {"cpu_augmentation_checks": read(here / "AUGMENTATION_WORKSPACE_TESTS.json"),
               "threshold_checks": read(here / "THRESHOLD_TESTS.json")})
    return summary


def build(source, output):
    require(output != ROOT and ROOT not in output.parents and output != source and source not in output.parents,
            "Model ZIP output must be outside the repository and original project")
    require(not output.exists(), "Existing release artifacts are preserved; use a fresh output directory")
    require(not (MODEL / "manifest.json").exists(), "An existing model manifest is preserved; prepare a new version")
    here, spec, status, ready, omitted, frozen, integrity = load_suite(source)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    output.mkdir(parents=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix=".task2-portable-model-", dir=output.parent) as temporary:
        portable_path = Path(temporary) / "model.pt"
        for item, folder, config, best, summary, validation, history in ready:
            run_id = item["run_id"]
            proof = portable_checkpoint(folder / "resnet.pt", portable_path, config, best, torch)
            require(proof["original_checkpoint_sha256"] == validation["checkpoint_sha256"], "Original checkpoint changed after independent validation")
            public_config = {key: value for key, value in config.items() if key not in {"data_root", "output_root", "resolved_device", "model_name", "architecture", "smoke_test"}}
            metadata = {"run_id": run_id, "model_name": "resnet", "architecture": "resnet18", "axis": item["axis"], "role": item["role"],
                        "rule_status": item["rule_status"], "submission_eligible": item["rule_status"] == "within_confirmed_scope",
                        "checkpoint_filename": "model.pt", "checkpoint_sha256": proof["published_checkpoint_sha256"], **proof,
                        "configuration": public_config, "selected_epoch": best["epoch"], "epochs_completed": summary["epochs_completed"],
                        "training_seconds": summary["elapsed_seconds"], "initialization": "random; weights=None",
                        "validation_metrics": {key: best[key] for key in METRICS if key not in {"elapsed_seconds", "images_per_second"}},
                        "original_independent_validation_passed": True, "all_data_final_model": False,
                        "private_test_evaluated": False, "checkpoint_is_inference_only": True, "training_resume_state_included": False,
                        "training_code_sha256": frozen, "release_tag": TAG}
            safe(metadata)
            readme = f"""# Task2 ResNet18 · {run_id}

독립 단일 모델의 환자 holdout 실험 패키지입니다. 전체 데이터 최종 모델이나 private Test 결과가 아닙니다.

```bash
python -m pip install -r requirements.txt
python predict.py --images input_images --output predictions.csv --device cpu
```

RGB/bilinear224 letterbox, normalization0.5/0.5, Present=1/Absent=0, threshold0.5입니다.
학습 증강은 추론에 적용하지 않습니다. CUDA 환경에서는 `--device cuda`를 쓸 수 있습니다.

- 검증 BA {best['balanced_accuracy']:.6%}; Present recall {best['sensitivity']:.6%}; Absent recall {best['specificity']:.6%}.
- fold {item['fold']}, seed {item['seed']}, 선택epoch {best['epoch']}; 규칙 상태 `{item['rule_status']}`.
- 밝기/대비는 수업 허용 미확정인 탐색 실험입니다. 점수가 높아도 제출 레시피에 자동 채택하지 않습니다.
- 원본 checkpoint는 로컬에서 보존했습니다. 배포 복사본은 config.data_root/output_root 두 경로만 portable 값으로 변경해 재직렬화했습니다.
- 원본 파일 SHA256 `{proof['original_checkpoint_sha256']}`; 배포 파일 SHA256 `{proof['published_checkpoint_sha256']}`.
- 모든122개 model tensor와 두 경로 외 metadata는 원본과 정확히 같습니다. 원본 파일과 배포 파일의 바이트가 같다는 뜻은 아닙니다.
- 원본 영상·라벨·환자ID·개별 예측·분할표·optimizer/RNG 재개 state는 포함하지 않습니다.
- 독립 검증은 원본 가중치 SHA에 연결되어 있습니다. 배포 가중치의 synthetic CPU/CLI parity는 별도 package_validation 기록으로 확인합니다.

전체 비교: https://github.com/{REPOSITORY}/tree/ktg_tried/experiments/task2-augmentation-lr-20261125
"""
            contents = {"predict.py": (MODEL / "predict.py").read_bytes(), "requirements.txt": (MODEL / "requirements.txt").read_bytes(),
                        "metadata.json": json_bytes(metadata), "README.md": readme.encode("utf-8")}
            for value in contents.values():
                safe(value.decode("utf-8"))
            checks = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
            checks["model.pt"] = proof["published_checkpoint_sha256"]
            contents["SHA256SUMS.json"] = json_bytes(checks)
            archive = output / f"{run_id}.zip"
            with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
                for name in sorted([*contents, "model.pt"]):
                    info = zipfile.ZipInfo(name, date_time=(2026, 11, 25, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    if name == "model.pt":
                        with portable_path.open("rb") as src, stream.open(info, "w", force_zip64=True) as dst:
                            shutil.copyfileobj(src, dst, 1024 * 1024)
                    else:
                        stream.writestr(info, contents[name])
            require(sha(folder / "resnet.pt") == proof["original_checkpoint_sha256"], "Original checkpoint changed during packaging")
            row = {"id": run_id, "architecture": "resnet18", "axis": item["axis"], "rule_status": item["rule_status"],
                   "validation_fold": item["fold"], "seed": item["seed"], "selected_epoch": best["epoch"],
                   **{key: best[key] for key in ("balanced_accuracy", "sensitivity", "specificity")},
                   "checkpoint_filename": "model.pt", "checkpoint_sha256": proof["published_checkpoint_sha256"],
                   **{key: proof[key] for key in ("original_checkpoint_sha256", "published_checkpoint_sha256", "changed_metadata_fields", "all_122_model_tensors_exact")},
                   "asset_name": archive.name, "archive_size_bytes": archive.stat().st_size, "archive_sha256": sha(archive),
                   "download_url": f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{archive.name}"}
            rows.append(row)
            print("Packaged:", run_id, flush=True)
    export_records(here, ready, frozen, integrity, rows, omitted)
    manifest = {"format_version": 1, "repository": REPOSITORY, "release_tag": TAG, "build_status": "packaged",
                "checkpoint_filename": "model.pt", "total_models": len(rows), "models": rows, "omitted_runs": omitted,
                "contains_training_data": False, "contains_ensembles": False, "contains_training_resume_state": False,
                "final_selection": {"status": "not_provided", "run_id": None},
                "checkpoint_repackaging": "Only config.data_root and config.output_root changed. All122 model tensors and all other metadata are exact. Original/published SHA256 are recorded separately.",
                "notes": ["URLs require this release to be published; packaging itself does not upload.",
                          "Brightness/contrast remain exploratory with unconfirmed assignment permission; threshold remains0.5 in all released models."]}
    write_json(MODEL / "manifest.json", manifest)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    table = "\n".join(f"| {row['id']} | {row['balanced_accuracy']:.4%} | {row['rule_status']} | [ZIP]({row['download_url']}) |" for row in rows)
    (MODEL / "README.md").write_text(f"""# Task2 회전·학습률·밝기/대비 모델 — 2026-11-25

완료·독립 검증한8개 holdout 모델의 패키지입니다. [manifest.json](manifest.json)에 원본/배포 가중치 및 ZIP 해시를 기록했습니다.
Release 다운로드는 `task2-augmentation-lr-20261125-v1` 공개 후 동작합니다. 기존20개 모델은 보존합니다.

| Run | 검증 BA | 규칙 상태 | 다운로드 |
|---|---:|---|---|
{table}

```bash
python models/task2-augmentation-lr-20261125/download.py --list
python models/task2-augmentation-lr-20261125/download.py --run resnet18_20261125_rotation_f0_s42 --output downloads/task2-augmentation-lr/rotation
python -m pip install -r downloads/task2-augmentation-lr/rotation/requirements.txt
python downloads/task2-augmentation-lr/rotation/predict.py --images input_images --output predictions.csv --device cpu
```

각 ZIP은 `model.pt,predict.py,requirements.txt,metadata.json,README.md,SHA256SUMS.json`의 여섯 flat 파일입니다.
다운로더는 ZIP과 내부 전체 파일의 SHA를 확인합니다. 여러 모델을 결합해 예측하지 않습니다.

원본 checkpoint의 머신 경로 유출을 막기 위해 배포 복사본의 config.data_root/output_root 두 값만 portable 경로로 변경해 재직렬화했습니다.
모든 model tensor와 다른 metadata·임계값0.5는 정확히 같고 로컬 원본 bytes는 보존했습니다. `original_checkpoint_sha256`와
`published_checkpoint_sha256`는 서로 다른 파일을 식별합니다. 실제 CPU/CLI 비교 범위는 `package_validation.json`을 확인하세요.

밝기·대비·threshold 허용 범위는 미확정입니다. 탐색 결과를 제출 설정에 자동 채택하지 않습니다.
이8개는 환자 holdout 모델이며 전체 데이터 최종 제출 모델이나 private Test 성능이 아닙니다.
원본 데이터·영상/라벨·환자목록·개별예측·분할표·과제PDF·학습 재개 state를 포함하지 않습니다.
비교/해석은 [실험 기록](../../experiments/task2-augmentation-lr-20261125/)을 참고하세요.
""", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-task2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_task2.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
