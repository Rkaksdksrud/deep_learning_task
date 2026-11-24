"""Package only a completed, SHA-bound Task2 training-settings comparison suite.

Preserve original inference checkpoint bytes and use the established six-file
ZIP format. Dataset records and resumable training states are never copied.
This command performs local CPU inspection and packaging; it does not publish.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/task2-training-settings-20261124"
REPOSITORY = "Rkaksdksrud/deep_learning_task"
TAG = "task2-training-settings-20261124-v1"
SUITE_DIR = "research_20261124"
EXPERIMENT_ID = "training_settings_20261124"
FILES = {"model.pt", "predict.py", "requirements.txt", "metadata.json", "README.md", "SHA256SUMS.json"}
CHANGES = {"reference": {}, "class_weighted_loss": {"class_weighted_loss": False},
           "learning_rate": {"learning_rate": 0.0001}, "weight_decay": {"weight_decay": 0.0},
           "lr_scheduler": {"lr_scheduler": "constant"}}
KINDS = {"reference": "reference", "class_weighted_loss": "unweighted",
         "learning_rate": "lr1e4", "weight_decay": "wd0", "lr_scheduler": "constant"}
FIXED = {"seed": 42, "n_splits": 5, "val_fold": 0, "image_size": 224, "batch_size": 16,
         "num_workers": 0, "cpu_threads": 4, "device": "cuda", "epochs": 30, "patience": 7,
         "learning_rate": 0.0003, "weight_decay": 0.0001, "class_weighted_loss": True,
         "aux_loss_weight": 0.3, "threshold": 0.5, "geometric_augmentation": False,
         "lr_scheduler": "cosine", "validation_class_weighted_loss": True}
CONFIG_KEYS = tuple(FIXED)
METRIC_KEYS = ("balanced_accuracy", "sensitivity", "specificity", "auroc", "accuracy",
               "confusion_matrix", "samples", "threshold", "loss")
ALL_METRIC_KEYS = set(METRIC_KEYS) | {"precision", "f1", "elapsed_seconds", "images_per_second"}
CHECKPOINT_KEYS = {"format_version", "model_name", "architecture", "state_dict", "config", "epoch",
                   "valid_metrics", "threshold", "class_weights", "validation_class_weights",
                   "smoke_test", "initialization", "label_mapping", "preprocessing"}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def contained(base, relative):
    path = Path(relative)
    require(not path.is_absolute() and "\\" not in str(relative) and ".." not in path.parts,
            "Expected a portable relative path")
    resolved = (base / path).resolve()
    require(base.resolve() in resolved.parents, "Input escapes its expected directory")
    return resolved


def verify_metrics(best, other, run_id):
    require(set(METRIC_KEYS) <= set(best) and set(best) <= ALL_METRIC_KEYS | {"epoch", "smoke_test"},
            f"Unexpected aggregate metric fields: {run_id}")
    require(best.get("auroc") is None or (finite(best["auroc"]) and 0 <= best["auroc"] <= 1),
            f"Invalid aggregate AUROC: {run_id}")
    for key in ("balanced_accuracy", "sensitivity", "specificity", "accuracy", "threshold"):
        require(finite(best.get(key)) and finite(other.get(key)) and
                math.isclose(best[key], other[key], abs_tol=1e-12, rel_tol=0),
                f"Validated {key} differs: {run_id}")
    require(finite(best.get("loss")) and finite(other.get("loss")) and
            math.isclose(best["loss"], other["loss"], abs_tol=1e-5, rel_tol=1e-5),
            f"Weighted validation loss differs: {run_id}")
    require(best.get("confusion_matrix") == other.get("confusion_matrix") and
            best.get("samples") == other.get("samples"), f"Confusion matrix/support differs: {run_id}")
    cm = best["confusion_matrix"]
    require(isinstance(cm, list) and len(cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cm)
            and all(type(value) is int and value >= 0 for row in cm for value in row),
            f"Invalid confusion matrix: {run_id}")
    tn, fp = cm[0]
    fn, tp = cm[1]
    require(tn + fp > 0 and tp + fn > 0 and sum(map(sum, cm)) == best["samples"],
            f"Invalid class support: {run_id}")
    expected = {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2}
    require(all(math.isclose(best[key], value, abs_tol=1e-12, rel_tol=0) for key, value in expected.items()),
            f"Metrics disagree with the confusion matrix: {run_id}")


def inspect_checkpoint(weights, config, best):
    """CPU-only safe load: reject unexpected/private payloads instead of rewriting."""
    import torch

    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    require(isinstance(checkpoint, dict) and set(checkpoint) == CHECKPOINT_KEYS,
            "Unexpected checkpoint fields; only the inference schema may be shared")
    expected_config = {key: value for key, value in config.items()
                       if key not in {"model_name", "architecture", "smoke_test"}}
    require(checkpoint["config"] == expected_config and checkpoint["format_version"] == 1 and
            checkpoint["model_name"] == "resnet" and checkpoint["architecture"] == "resnet18" and
            checkpoint["smoke_test"] is False and checkpoint["initialization"] == "random; weights=None" and
            checkpoint["threshold"] == 0.5 and checkpoint["epoch"] == best["epoch"] and
            checkpoint["label_mapping"] == {0: "absent", 1: "present"},
            "Checkpoint configuration/identity differs from the completed run")
    expected_prep = {"version": 1, "color_mode": "RGB", "resize": "aspect_preserving_letterbox",
                     "interpolation": "bilinear", "padding_color": [0, 0, 0], "pixel_scale": 255.0,
                     "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "image_size": 224}
    require(checkpoint["preprocessing"] == expected_prep, "Unsupported inference preprocessing")
    verify_metrics(best, checkpoint["valid_metrics"], weights.parent.name)
    require(checkpoint["valid_metrics"] == {key: value for key, value in best.items()
                                           if key not in {"epoch", "smoke_test"}},
            "Checkpoint aggregate metrics contain unexpected fields or differ from best_metrics")
    for name in ("class_weights", "validation_class_weights"):
        values = checkpoint[name]
        require(isinstance(values, list) and len(values) == 2 and all(finite(x) and x > 0 for x in values),
                "Invalid train/evaluation class weights")
    if config["class_weighted_loss"]:
        require(checkpoint["class_weights"] == checkpoint["validation_class_weights"],
                "Weighted training/evaluation objectives differ")
    else:
        require(checkpoint["class_weights"] == [1.0, 1.0], "Unweighted training has non-unit weights")
    state = checkpoint["state_dict"]
    key_pattern = re.compile(r"(?:conv1|bn1|layer[1-4]\.[01]\.(?:conv[12]|bn[12]|downsample\.[01])|fc)\.(?:weight|bias|running_mean|running_var|num_batches_tracked)")
    require(isinstance(state, dict) and len(state) == 122 and
            all(isinstance(key, str) and key_pattern.fullmatch(key) and isinstance(value, torch.Tensor)
                for key, value in state.items()), "Unexpected ResNet18 state dictionary payload")
    require(tuple(state["fc.weight"].shape) == (2, 512) and tuple(state["fc.bias"].shape) == (2,),
            "Invalid binary ResNet18 classifier")
    result = {"training_class_weights": checkpoint["class_weights"],
              "validation_class_weights": checkpoint["validation_class_weights"]}
    del checkpoint
    return result


def load_suite(source):
    here = source / SUITE_DIR
    spec, status = read_json(here / "experiment_specs.json"), read_json(here / "execution_status.json")
    require(status.get("controller_status") == "completed", "Controller has not completed; no partial release is allowed")
    require(spec.get("experiment_id") == EXPERIMENT_ID and spec.get("max_new_full_runs") == 8 and
            spec.get("fixed_validation_class_weighted_loss") is True, "Unexpected suite identity/budget")
    base = spec.get("base_config", {})
    require(set(base) == set(FIXED) | {"data_root", "output_root"} and
            all(type(base[key]) is type(value) and base[key] == value for key, value in FIXED.items()),
            "Reference recipe differs from the approved no-augmentation comparison")
    for key in ("data_root", "output_root"):
        contained(source, base[key])
    primary, optional = spec["primary_runs"], spec["optional_runs"]
    require(len(primary) == 5 and len(optional) == 8 and {s["axis"] for s in primary} == set(CHANGES),
            "Unexpected primary/optional experiment templates")
    require(primary[0]["axis"] == "reference", "Reference must be the first primary experiment")
    entries = primary + optional
    ids = [s["run_id"] for s in entries]
    require(len(set(ids)) == len(ids), "Duplicate experiment IDs")
    frozen_inputs = read_json(here / "suite_inputs.json")
    required_inputs = {"run_oat.py", "experiment_specs.json", "frozen_source.json"} | {s["config_file"] for s in entries}
    require(required_inputs <= set(frozen_inputs), "Suite freeze does not bind controller/spec/configs")
    for relative, expected in frozen_inputs.items():
        require(sha256(contained(here, relative)) == expected, f"Frozen suite input changed: {relative}")
    for s in entries:
        axis, fold, seed = s["axis"], s["fold"], s["seed"]
        require(axis in CHANGES and s["changed"] == CHANGES[axis] and s["model"] == "resnet" and
                s["run_id"] == f"resnet18_20261124_{KINDS[axis]}_f{fold}_s{seed}", "Unexpected experiment specification")
        require((s in primary and (fold, seed) == (0, 42)) or
                (s in optional and axis != "reference" and (fold, seed) in {(1, 42), (0, 43)}),
                "Unexpected confirmation condition")
        expected = {**base, **s["changed"], "val_fold": fold, "seed": seed}
        require(read_json(contained(here, s["config_file"])) == expected, "Config differs from the frozen one-factor specification")
    recorded = status.get("runs", {})
    require(set(recorded) == set(ids), "Controller run IDs differ from the frozen templates")
    require(all(recorded[s["run_id"]].get("status") == "verified" for s in primary),
            "All five primary experiments must be verified")
    fold_axes, seed_axis = status.get("fold_confirmation_selection"), status.get("seed_confirmation_selection")
    require(isinstance(fold_axes, list) and len(set(fold_axes)) == len(fold_axes) <= 2 and
            all(axis in CHANGES and axis != "reference" for axis in fold_axes), "Invalid fold-confirmation selection")
    require(seed_axis is None or seed_axis in fold_axes, "Seed confirmation is not a selected fold-confirmed candidate")
    expected_verified = {s["run_id"] for s in primary} | {
        s["run_id"] for s in optional if (s["fold"] == 1 and s["axis"] in fold_axes) or
        (s["seed"] == 43 and s["axis"] == seed_axis)}
    verified = {run_id for run_id, entry in recorded.items() if entry.get("status") == "verified"}
    require(verified == expected_verified and 5 <= len(verified) <= 8 and
            status.get("verified_full_runs") == len(verified), "Verified runs do not match the finite-budget selection")
    require(all(entry.get("status") == "not_run" for run_id, entry in recorded.items() if run_id not in verified),
            "A completed controller still has unfinished/unverified experiments")
    parity = read_json(here / "baseline_parity.json")
    require(parity.get("all_required_checks_passed") is True and parity.get("checks") and
            all(value is True for value in parity["checks"].values()) and
            parity.get("new_reference") == primary[0]["run_id"] and
            parity.get("old_reference") == spec["historical_reference_f0"], "Reference parity was not verified")
    return here, spec, status, entries, verified


def collect(source):
    here, spec, status, entries, verified = load_suite(source)
    frozen_code = read_json(here / "frozen_source.json")
    require({"code/common.py", "code/data.py", "code/resnet.py", "code/validate.py", "code/training_state.py"} <= set(frozen_code),
            "Training source freeze omits a core implementation")
    ready, omitted = [], []
    for s in entries:
        run_id = s["run_id"]
        if run_id not in verified:
            if s["role"] == "conditional_fold_confirmation":
                reason = "not_selected_by_primary_effect_gate_or_two_candidate_fold_budget"
            elif s["axis"] not in status["fold_confirmation_selection"]:
                reason = "axis_not_selected_for_fold_confirmation"
            elif status["seed_confirmation_selection"] is None:
                reason = "no_axis_passed_the_two_condition_seed_confirmation_gate"
            else:
                reason = "another_axis_selected_for_the_single_seed_confirmation_budget"
            omitted.append({"id": run_id, "axis": s["axis"], "role": s["role"],
                            "status": "not_run_predeclared_finite_budget_selection", "reason": reason})
            continue
        folder = source / "runs" / run_id
        config = read_json(folder / "config.json")
        expected = {**spec["base_config"], **s["changed"], "val_fold": s["fold"], "seed": s["seed"],
                    "resolved_device": "cuda", "model_name": "resnet", "architecture": "resnet18", "smoke_test": False}
        require(config == expected, f"Completed run configuration differs: {run_id}")
        source_hashes = read_json(folder / "source_manifest.json")
        for relative, expected_hash in frozen_code.items():
            require(relative.startswith("code/") and Path(relative).name == relative[len("code/"):] and relative.endswith(".py"),
                    "Unexpected source-manifest field")
            require(source_hashes.get(relative) == expected_hash and
                    sha256(folder / "source" / Path(relative).name) == expected_hash,
                    f"Archived training source differs: {run_id}")
        summary, best = read_json(folder / "training_summary.json"), read_json(folder / "best_metrics.json")
        validation = read_json(folder / "validation/metrics.json")
        entry = status["runs"][run_id]
        require(summary.get("model_name") == "resnet" and summary.get("smoke_test") is False and
                best.get("smoke_test") is False and validation.get("smoke_test") is False and
                validation.get("model_name") == "resnet" and validation.get("architecture") == "resnet18" and
                validation.get("all_checks_passed") is True, f"Invalid independent validation: {run_id}")
        require(type(best.get("epoch")) is int and type(summary.get("epochs_completed")) is int and
                1 <= best["epoch"] <= summary["epochs_completed"] <= 30 and
                validation.get("selected_epoch") == best["epoch"], f"Selected epoch differs: {run_id}")
        require(entry.get("best_metrics") == best and entry.get("training_summary") == summary,
                f"Controller completion records disagree with run files: {run_id}")
        verify_metrics(best, validation["metrics"], run_id)
        require(math.isclose(summary["best_valid_balanced_accuracy"], best["balanced_accuracy"], abs_tol=1e-12, rel_tol=0),
                f"Training summary BA differs: {run_id}")
        weights = folder / "resnet.pt"
        checkpoint_hash = sha256(weights)
        require(validation.get("checkpoint_bytes_unchanged_during_validation") is True and
                validation.get("checkpoint_sha256") == checkpoint_hash == entry.get("checkpoint_sha256") and
                Path(validation.get("weights", "")).resolve() == weights.resolve(),
                f"Independent validation is not bound to these original checkpoint bytes: {run_id}")
        objectives = inspect_checkpoint(weights, config, best)
        require(sha256(weights) == checkpoint_hash, f"Checkpoint changed during CPU inspection: {run_id}")
        with (folder / "history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        require([int(row["epoch"]) for row in history] == list(range(1, summary["epochs_completed"] + 1)),
                f"Non-contiguous completed history: {run_id}")
        selected_row = history[best["epoch"] - 1]
        require(math.isclose(float(selected_row["valid_balanced_accuracy"]), best["balanced_accuracy"], abs_tol=1e-12, rel_tol=0)
                and math.isclose(float(selected_row["valid_loss"]), best["loss"], abs_tol=1e-12, rel_tol=0),
                f"Selected history metrics differ: {run_id}")
        elapsed = summary.get("elapsed_seconds")
        require(finite(elapsed) and elapsed >= 0, f"Invalid elapsed time: {run_id}")
        peak = [float(row["peak_gpu_memory_mib"]) for row in history if row.get("peak_gpu_memory_mib")]
        environment = read_json(folder / "environment.json")
        versions = {key: environment[key] for key in
                    ("torch", "torchvision", "numpy", "pandas", "Pillow", "scikit-learn")}
        require(all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.+-]+", value) for value in versions.values()),
                f"Unexpected or machine-specific library version: {run_id}")
        metadata = {"run_id": run_id, "model_name": "resnet", "architecture": "resnet18", "axis": s["axis"],
                    "comparison_role": s["role"], "control_run_id": s["control_id"],
                    "checkpoint_filename": "model.pt", "checkpoint_sha256": checkpoint_hash,
                    "initialization": "random; weights=None", "configuration": {key: config[key] for key in CONFIG_KEYS},
                    **objectives, "validation_loss_objective": "weighted CE estimated from the training partition; fixed across all candidates",
                    "selected_epoch": best["epoch"], "epochs_completed": summary["epochs_completed"],
                    "training_elapsed_seconds": elapsed, "peak_gpu_memory_mib": max(peak) if peak else None,
                    "validation_metrics": {key: best[key] for key in METRIC_KEYS},
                    "original_independent_validation_passed": True, "validation_bound_to_checkpoint_sha256": True,
                    "all_data_final_model": False, "private_test_evaluated": False,
                    "training_data_scope": "provided training partition only; patient-disjoint holdout",
                    "checkpoint_is_inference_only": True, "training_resume_state_included": False,
                    "environment_versions": versions,
                    "training_code_sha256": {key: source_hashes[key] for key in frozen_code}, "release_tag": TAG}
        ready.append((run_id, weights, metadata))
    return ready, omitted, status


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(output != ROOT and ROOT not in output.parents and output != source and source not in output.parents,
            "Release archives must be outside both the Git repository and canonical Task2 project")
    if output.exists():
        raise FileExistsError("Preserve existing packages; choose a new output directory")
    manifest_path = MODEL_DIR / "manifest.json"
    if manifest_path.exists():
        require(read_json(manifest_path).get("build_status") == "prepared_pending_completed_controller",
                "An existing packaged manifest is immutable; prepare a new release version")
    ready, omitted, status = collect(source)
    output.mkdir(parents=True)
    rows = []
    for run_id, weights, metadata in ready:
        metrics = metadata["validation_metrics"]
        readme = f"""# Task2 ResNet18 training settings · {run_id}

제공 데이터의 환자별 Train partition으로 학습한 단일 holdout 모델입니다.
전체 데이터 최종 제출 모델이나 private Test 성능을 나타내지 않습니다.

```bash
python -m pip install -r requirements.txt
python predict.py --images input_images --output predictions.csv --device cpu
```

원본 추론 checkpoint 바이트를 그대로 `model.pt`에 담았습니다. RGB/bilinear 224 letterbox,
고정 normalization0.5/0.5, Present=1/Absent=0, threshold0.5입니다. 학습·추론 모두 기하 증강을 끕니다.
각 ZIP은 독립 모델입니다. CUDA 사용은 `--device cuda`로 지정합니다.

- 비교 축: {metadata['axis']}; fold {metadata['configuration']['val_fold']}; seed {metadata['configuration']['seed']}.
- 검증 BA {metrics['balanced_accuracy']:.6%}; Present recall {metrics['sensitivity']:.6%}; Absent recall {metrics['specificity']:.6%}.
- 선택 epoch {metadata['selected_epoch']}; 완료 epoch {metadata['epochs_completed']}.
- weighted Valid loss와 checkpoint tie-break는 모든 비교에서 동일합니다. 학습 loss 가중치는 metadata에서 별도로 표시합니다.
- 원본 checkpoint SHA256: `{metadata['checkpoint_sha256']}`.
- 원본 영상·라벨·환자 ID·분할표·개별 예측·과제 PDF·학습 재개 상태는 포함하지 않습니다.
- SHA에 연결된 기존 독립 재검증을 통과했습니다. 다른 backend의 확률 bit parity까지 보장하지 않습니다.

코드와 비교 결과: https://github.com/{REPOSITORY}/tree/ktg_tried
"""
        contents = {"predict.py": (MODEL_DIR / "predict.py").read_bytes(),
                    "requirements.txt": (MODEL_DIR / "requirements.txt").read_bytes(),
                    "metadata.json": json_bytes(metadata), "README.md": readme.encode("utf-8")}
        checks = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
        checks["model.pt"] = metadata["checkpoint_sha256"]
        contents["SHA256SUMS.json"] = json_bytes(checks)
        require(set(contents) | {"model.pt"} == FILES, "Unexpected package contents")
        archive = output / f"{run_id}.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
            for name in sorted(FILES):
                info = zipfile.ZipInfo(name, date_time=(2026, 11, 24, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                if name == "model.pt":
                    with weights.open("rb") as original, stream.open(info, "w", force_zip64=True) as target:
                        shutil.copyfileobj(original, target, 1024 * 1024)
                else:
                    stream.writestr(info, contents[name])
        require(sha256(weights) == metadata["checkpoint_sha256"], f"Checkpoint changed while packaging: {run_id}")
        row = {"id": run_id, "architecture": "resnet18", "axis": metadata["axis"],
               "comparison_role": metadata["comparison_role"], "image_size": 224,
               "validation_fold": metadata["configuration"]["val_fold"], "seed": metadata["configuration"]["seed"],
               "selected_epoch": metadata["selected_epoch"],
               **{key: metrics[key] for key in ("balanced_accuracy", "sensitivity", "specificity")},
               "checkpoint_filename": "model.pt", "checkpoint_sha256": metadata["checkpoint_sha256"],
               "asset_name": archive.name, "archive_size_bytes": archive.stat().st_size, "archive_sha256": sha256(archive),
               "download_url": f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{archive.name}"}
        rows.append(row)
        print(f"{run_id}: {archive.stat().st_size:,} bytes SHA256={row['archive_sha256']}", flush=True)
    manifest = {"format_version": 1, "repository": REPOSITORY, "release_tag": TAG, "build_status": "packaged",
                "checkpoint_filename": "model.pt", "experiment_id": EXPERIMENT_ID, "total_models": len(rows), "models": rows,
                "omitted_runs": omitted, "final_selection": {
                    "status": "not_provided", "run_id": None,
                    "note": "All packages are comparison checkpoints. Settings recommendations belong in the experiment report; the existing global reference is unchanged."},
                "conditional_confirmation_selection": {"fold_axes": status["fold_confirmation_selection"],
                                                       "seed_axis": status["seed_confirmation_selection"]},
                "contains_training_data": False, "contains_ensembles": False, "contains_training_resume_state": False,
                "notes": ["All primary and selected optional runs completed and passed SHA-bound independent validation.",
                          "Optional templates not selected by the finite-budget gate are explicitly omitted, without scores.",
                          "One-factor comparisons around the no-augmentation reference; not additive contributions or a full optimum search.",
                          "Holdout checkpoints only; neither all-data final models nor independent private Test results.",
                          "Download URLs become available only after this new release is published."]}
    manifest_path.write_bytes(json_bytes(manifest))
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    print(f"Built {len(rows)} models; omitted {len(omitted)} conditional templates; no upload performed", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-task2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory outside the repository and canonical Task2 project")
    args = parser.parse_args()
    build(args.source_task2.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
