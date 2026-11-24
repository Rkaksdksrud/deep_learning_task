"""CPU-only conditional LOPO threshold diagnosis from audited cached predictions.

No model is loaded, trained, changed, or published. The threshold grid is fixed
before reading any probabilities. A held-out patient's labels AND probabilities
are excluded from selecting that patient's threshold. Because checkpoint epoch
selection already used these validation patients, this is not independent Test.
LOPO yields diagnostic held-out decisions, not patient-specific deployment rules.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

RUNS = [
    ("resnet18_20261111_geometry_f0_s42", "primary"),
    ("resnet18_20261111_geometry_f0_s43", "seed_confirmation_same_validation_patients"),
    ("resnet18_20261111_geometry_f1_s42", "fold_confirmation"),
]
THRESHOLD_PERCENT_GRID = tuple(range(5, 96))  # Includes exactly 50; no data-dependent candidate construction.
FIELDS = ("sample_id", "path", "label", "patient_group")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def close(actual, expected, name, tolerance=1e-12):
    if not math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0):
        raise ValueError(f"Inconsistent {name}: {actual} != {expected}")


def validate_rows(rows):
    if not rows:
        raise ValueError("Empty prediction/calibration collection")
    for row in rows:
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError("Expected binary integer labels")
        if not isinstance(row["patient_group"], str) or not row["patient_group"]:
            raise ValueError("Every sample requires an official nonempty patient group")
        probability = row["probability"]
        if not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probability must be finite and in [0,1]")


def metrics_from_decisions(rows, predictions):
    if len(rows) != len(predictions) or not rows:
        raise ValueError("Prediction count differs or evaluation is empty")
    if any(type(value) is not int or value not in (0, 1) for value in predictions):
        raise ValueError("Expected binary integer decisions")
    tn = sum(row["label"] == 0 and prediction == 0 for row, prediction in zip(rows, predictions))
    fp = sum(row["label"] == 0 and prediction == 1 for row, prediction in zip(rows, predictions))
    fn = sum(row["label"] == 1 and prediction == 0 for row, prediction in zip(rows, predictions))
    tp = sum(row["label"] == 1 and prediction == 1 for row, prediction in zip(rows, predictions))
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    return {"balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
            "sensitivity": tpr, "specificity": tnr, "confusion_matrix": [[tn, fp], [fn, tp]],
            "accuracy": (tn + tp) / len(rows), "samples": len(rows), "positive_samples": tp + fn,
            "negative_samples": tn + fp,
            "ba_interpretation": "image-level two-class BA" if tpr is not None and tnr is not None
            else "undefined: only one true class; report available-class recall"}


def at_threshold(rows, threshold):
    validate_rows(rows)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and within [0,1]")
    return metrics_from_decisions(rows, [int(row["probability"] >= threshold) for row in rows])


def select_threshold(calibration_rows):
    """Select only from calibration patients; exact BA handles plateaus consistently."""
    validate_rows(calibration_rows)
    if {row["label"] for row in calibration_rows} != {0, 1}:
        raise ValueError("Calibration requires both true classes; no held-out-label fallback")
    candidates = []
    for percent in THRESHOLD_PERCENT_GRID:
        threshold = percent / 100
        metrics = at_threshold(calibration_rows, threshold)
        tn, fp = metrics["confusion_matrix"][0]
        fn, tp = metrics["confusion_matrix"][1]
        exact_ba = (Fraction(tn, tn + fp) + Fraction(tp, tp + fn)) / 2
        candidates.append(((-exact_ba, abs(percent - 50), percent), threshold, metrics))
    _, threshold, metrics = min(candidates, key=lambda item: item[0])
    return threshold, metrics


def crossfit_lopo(rows):
    """Each fold deploys a single global threshold fit on the other groups only."""
    validate_rows(rows)
    groups = sorted({row["patient_group"] for row in rows})
    if len(groups) < 2:
        raise ValueError("LOPO needs at least two patient groups")
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Combined evaluation requires both classes")
    folds = []
    combined_rows, combined_decisions = [], []
    for held_out in groups:
        calibration = [row for row in rows if row["patient_group"] != held_out]
        held = [row for row in rows if row["patient_group"] == held_out]
        threshold, calibration_metrics = select_threshold(calibration)
        decisions = [int(row["probability"] >= threshold) for row in held]
        fixed = at_threshold(held, .5)
        changed = metrics_from_decisions(held, decisions)
        delta = changed["balanced_accuracy"] - fixed["balanced_accuracy"] if fixed["balanced_accuracy"] is not None else None
        folds.append({
            "held_out_patient": held_out,
            "calibration_patients": [patient for patient in groups if patient != held_out],
            "calibration_samples": len(calibration), "selected_threshold": threshold,
            "calibration_selected_metrics": calibration_metrics,
            "calibration_metric_scope": "in-sample threshold selection objective; not held-out performance",
            "held_out_fixed_05": fixed, "held_out_crossfitted": changed, "held_out_delta_ba": delta,
        })
        combined_rows.extend(held)
        combined_decisions.extend(decisions)
    fixed = at_threshold(rows, .5)
    combined = metrics_from_decisions(combined_rows, combined_decisions)
    return {"fixed_threshold": .5, "fixed_05": fixed, "crossfitted": combined,
            "delta_ba": combined["balanced_accuracy"] - fixed["balanced_accuracy"], "folds": folds,
            "deployment_threshold": None,
            "protocol_scope": "conditional LOPO diagnosis with fixed checkpoint; no deployable per-patient threshold map"}


def checked_index(rows, name):
    if not rows or any(any(row.get(field, "") == "" for field in FIELDS) for row in rows):
        raise ValueError(f"Incomplete {name} metadata")
    index = {row["sample_id"]: row for row in rows}
    if len(index) != len(rows) or len({row["path"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs or image paths in {name}")
    if any(row["label"] not in {"0", "1"} for row in rows):
        raise ValueError(f"Invalid labels in {name}")
    return index


def load_bound_predictions(task, run_id):
    folder = task / "runs" / run_id
    audit_folder = task / "research_20261125/historical_validation" / run_id
    audit_path, prediction_path = audit_folder / "metrics.json", audit_folder / "predictions.csv"
    if not audit_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"Wait for the new SHA-bound historical audit: {run_id}")
    audit = read_json(audit_path)
    if audit.get("all_checks_passed") is not True or audit.get("smoke_test") is not False:
        raise ValueError(f"Independent audit is unsuccessful/smoke-only: {run_id}")
    if audit.get("model_name") != "resnet" or audit.get("architecture") != "resnet18":
        raise ValueError("Unexpected model family")
    if not audit.get("checkpoint_sha256"):
        raise ValueError("Cached audit must be bound to an explicit checkpoint SHA256")
    weights = folder / "resnet.pt"
    actual_weights_sha = sha256(weights)
    if actual_weights_sha != audit["checkpoint_sha256"]:
        raise ValueError("Actual checkpoint differs from the audited weights")
    config = read_json(folder / "config.json")
    if config.get("geometric_augmentation") is not True or config["threshold"] != .5:
        raise ValueError("Only the selected geometry policy with threshold .5 is this diagnostic's input")
    data_root = Path(config["data_root"])
    data_root = data_root if data_root.is_absolute() else task / data_root
    labels_path = data_root / "labels.csv"
    labels = checked_index(read_csv(labels_path), "official supplied labels")
    manifest_path = folder / "split_manifest.csv"
    manifest_rows = read_csv(manifest_path)
    manifest = checked_index(manifest_rows, "saved patient split")
    if labels.keys() != manifest.keys() or any(any(labels[key][field] != manifest[key][field] for field in FIELDS) for key in labels):
        raise ValueError("Current official labels/groups/paths differ from the saved split")
    if {row["split"] for row in manifest_rows} != {"train", "valid"}:
        raise ValueError("Expected exactly train and valid partitions")
    train_groups = {row["patient_group"] for row in manifest_rows if row["split"] == "train"}
    valid_groups = {row["patient_group"] for row in manifest_rows if row["split"] == "valid"}
    if train_groups & valid_groups:
        raise ValueError("Training and validation patients overlap")
    patient_folds = {}
    for row in manifest_rows:
        patient_folds.setdefault(row["patient_group"], set()).add(row["fold"])
    if any(len(folds) != 1 for folds in patient_folds.values()):
        raise ValueError("One patient appears in multiple saved folds")
    valid = {key: row for key, row in manifest.items() if row["split"] == "valid"}
    if {row["fold"] for row in valid.values()} != {str(config["val_fold"])} or len(valid_groups) != 6:
        raise ValueError("Expected the original six-patient validation fold for the declared protocol")
    saved_rows = read_csv(prediction_path)
    predictions = checked_index(saved_rows, "SHA-bound cached predictions")
    if predictions.keys() != valid.keys():
        raise ValueError("Cached predictions are not exactly the locked validation split")
    rows = []
    for key, saved in predictions.items():
        if any(saved[field] != valid[key][field] for field in FIELDS):
            raise ValueError("Cached labels/groups/paths disagree with official labels or the saved split")
        if saved.get("split") != "valid" or saved.get("fold") != valid[key]["fold"]:
            raise ValueError("Cached patient split/fold metadata differs")
        row = {"sample_id": key, "patient_group": saved["patient_group"],
               "label": int(saved["label"]), "probability": float(saved["probability"])}
        validate_rows([row])
        if int(saved["prediction"]) != int(row["probability"] >= .5):
            raise ValueError("Stored fixed-.5 decisions do not match stored probabilities")
        rows.append(row)
    fixed = at_threshold(rows, .5)
    if fixed["confusion_matrix"] != audit["metrics"]["confusion_matrix"]:
        raise ValueError("Recomputed confusion matrix differs from the audited checkpoint")
    for metric in ("balanced_accuracy", "sensitivity", "specificity", "accuracy", "samples"):
        close(fixed[metric], audit["metrics"][metric], f"fixed-.5 {metric}")
    if sorted(valid_groups) != audit["validation_groups"]:
        raise ValueError("Audit validation patients differ from the saved manifest")
    before = sha256(prediction_path)
    if audit.get("predictions_sha256") is not None and before != audit["predictions_sha256"]:
        raise ValueError("Cached predictions differ from their recorded audit digest")
    inputs = {"checkpoint": str(weights.relative_to(task)), "checkpoint_sha256": actual_weights_sha,
              "audit": str(audit_path.relative_to(task)), "audit_sha256": sha256(audit_path),
              "cached_predictions": str(prediction_path.relative_to(task)), "cached_predictions_sha256": before,
              "official_labels_sha256": sha256(labels_path), "saved_split_sha256": sha256(manifest_path),
              "all_input_checks_passed": True, "selected_epoch": audit["selected_epoch"],
              "fold": config["val_fold"], "seed": config["seed"],
              "validation_patients": sorted(valid_groups), "new_inference_performed_by_this_script": False}
    return rows, inputs


def csv_text(rows, keys):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue()


def analyze(task, include_oracle=False):
    started = time.perf_counter()
    results = []
    input_files = {}
    for run_id, role in RUNS:
        rows, inputs = load_bound_predictions(task, run_id)
        paths = [task / inputs["checkpoint"], task / inputs["audit"], task / inputs["cached_predictions"],
                 task / "runs" / run_id / "split_manifest.csv"]
        input_files.update({str(path): sha256(path) for path in paths})
        result = crossfit_lopo(rows)
        result.update(run_id=run_id, role=role, inputs=inputs)
        result["oracle_same_valid_diagnostic"] = None
        if include_oracle:
            threshold, metrics = select_threshold(rows)
            result["oracle_same_valid_diagnostic"] = {
                "threshold": threshold, "metrics": metrics,
                "scope": "optimistic same-valid threshold selection; diagnostic only, not the primary estimate or a deployment threshold"}
        results.append(result)
    for path, expected in input_files.items():
        if sha256(Path(path)) != expected:
            raise ValueError("An original checkpoint/audit/cache changed during CPU analysis")
    unique_patients = set().union(*(set(result["inputs"]["validation_patients"]) for result in results))
    unique_sample_ids = set()
    for run_id, _ in RUNS:
        rows, _ = load_bound_predictions(task, run_id)
        unique_sample_ids.update(row["sample_id"] for row in rows)
    return {
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "exploratory conditional cached-CPU analysis completed",
        "method": "fixed-checkpoint leave-one-validation-patient-out global threshold selection",
        "threshold_grid": [percent / 100 for percent in THRESHOLD_PERCENT_GRID],
        "selection_objective": "pooled image-level BA on other patients only",
        "tie_break": "maximum exact rational BA, then closest to .5, then smaller threshold",
        "candidate_2_rule_status": "unconfirmed for assignment deployment; this is exploratory diagnosis only",
        "results": results, "unique_validation_patients": len(unique_patients),
        "unique_validation_images": len(unique_sample_ids), "analysis_seconds": time.perf_counter() - started,
        "original_inputs_unchanged": True, "deployment_threshold": None,
        "limitations": [
            "Checkpoint epoch selection already used all of these validation patients. LOPO threshold separation does not remove this earlier selection dependence.",
            "This is not independent Test, an unbiased generalization estimate, or nested patient validation.",
            "Each held-out patient's diagnostic threshold is fit on the other patients; this is not a patient-specific deployment threshold policy.",
            "No single deployment threshold is selected. Independently evaluating a calibration rule would require new nested patient training, which was not performed.",
            "Do not select one fold0 threshold and call its use on an existing fold1 model independent: calibration patients can be in that other model's training set.",
            "Fold0 seed42/43 share validation patients. Three model conditions are not three independent patient cohorts.",
            "This evaluates only the predeclared [.05,.95] grid with step .01; probability tails or finer thresholds are not explored, so no universal threshold-utility conclusion follows.",
            "Photometric model fixed-.5 results and conditional LOPO diagnostics are different procedures and cannot be ranked as directly matched treatment effects.",
        ],
    }


def write_results(output, data):
    summary, patients = [], []
    for result in data["results"]:
        fixed, changed = result["fixed_05"], result["crossfitted"]
        summary.append({"run_id": result["run_id"], "role": result["role"],
                        "fixed_ba": fixed["balanced_accuracy"], "lopo_ba": changed["balanced_accuracy"],
                        "delta_ba_pp": 100*result["delta_ba"], "fixed_positive_recall": fixed["sensitivity"],
                        "lopo_positive_recall": changed["sensitivity"], "fixed_negative_recall": fixed["specificity"],
                        "lopo_negative_recall": changed["specificity"], "samples": fixed["samples"]})
        for fold in result["folds"]:
            old, new = fold["held_out_fixed_05"], fold["held_out_crossfitted"]
            patients.append({"run_id": result["run_id"], "held_out_patient": fold["held_out_patient"],
                             "selected_from_other_patients_threshold": fold["selected_threshold"],
                             "samples": old["samples"], "fixed_ba": old["balanced_accuracy"], "lopo_ba": new["balanced_accuracy"],
                             "delta_ba_pp": 100*fold["held_out_delta_ba"] if fold["held_out_delta_ba"] is not None else None,
                             "fixed_positive_recall": old["sensitivity"], "lopo_positive_recall": new["sensitivity"],
                             "fixed_negative_recall": old["specificity"], "lopo_negative_recall": new["specificity"]})
    contents = {"threshold_results.json": json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                "threshold_run_summary.csv": csv_text(summary, list(summary[0])),
                "threshold_lopo.csv": csv_text(patients, list(patients[0]))}
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in contents):
        raise FileExistsError("Existing threshold analysis is preserved; choose a new --output-dir")
    for name, content in contents.items():
        (output / name).write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--include-oracle", action="store_true", help="separate optimistic same-valid diagnostic; never a primary estimate")
    parser.add_argument("--dry-run", action="store_true", help="input verification and cached CPU analysis only; no writes")
    args = parser.parse_args()
    data = analyze(args.task_root.expanduser().resolve(), args.include_oracle)
    if not args.dry_run:
        write_results(args.output_dir.expanduser().resolve(), data)
    print(json.dumps({"dry_run": args.dry_run, "status": data["status"], "deployment_threshold": None,
                      "results": [{"run_id": result["run_id"], "fixed_ba": result["fixed_05"]["balanced_accuracy"],
                                   "lopo_ba": result["crossfitted"]["balanced_accuracy"], "delta_ba_pp": 100*result["delta_ba"]}
                                  for result in data["results"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
