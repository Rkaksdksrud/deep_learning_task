"""Independently validate one saved model on its locked patient holdout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common import classification_metrics, load_trained_model, task_path, write_json
from data import ImageDataset, REQUIRED_COLUMNS, load_records, split_summary


def read_manifest(path: Path, current_records: pd.DataFrame) -> pd.DataFrame:
    """Reject changed IDs/labels/groups instead of silently rebuilding a split."""
    manifest = pd.read_csv(path, dtype={column: str for column in REQUIRED_COLUMNS})
    required = set(REQUIRED_COLUMNS) | {"fold", "split"}
    if not required.issubset(manifest.columns) or manifest.empty:
        raise ValueError("The saved split manifest is incomplete")
    if manifest[list(required)].isna().any().any():
        raise ValueError("The saved split manifest contains missing values")
    if manifest["sample_id"].duplicated().any() or manifest["path"].duplicated().any():
        raise ValueError("The saved split manifest contains duplicate IDs/paths")
    if not set(manifest["label"]).issubset({"0", "1"}):
        raise ValueError("The saved split manifest has invalid labels")
    manifest["label"] = manifest["label"].astype(int)
    saved = manifest.sort_values("sample_id").reset_index(drop=True)
    current = current_records.sort_values("sample_id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(saved[list(REQUIRED_COLUMNS)], current[list(REQUIRED_COLUMNS)])
    except AssertionError as exc:
        raise ValueError("Current data IDs, paths, labels or patient groups differ from the saved manifest") from exc
    if set(manifest["split"]) != {"train", "valid"}:
        raise ValueError("Expected exactly train and valid partitions")
    train_groups = set(manifest.loc[manifest["split"] == "train", "patient_group"])
    valid_groups = set(manifest.loc[manifest["split"] == "valid", "patient_group"])
    if train_groups & valid_groups:
        raise ValueError("The saved training and validation patient groups overlap")
    if manifest.groupby("patient_group")["fold"].nunique().max() != 1:
        raise ValueError("A patient is assigned to multiple folds")
    for split in ("train", "valid"):
        if set(manifest.loc[manifest["split"] == split, "label"]) != {0, 1}:
            raise ValueError(f"The saved {split} partition does not contain both classes")
    return manifest


def per_patient_metrics(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Patient diagnostics supplement, and never replace, image-level BA."""
    rows = []
    for patient, group in predictions.groupby("patient_group", sort=True):
        labels = group["label"].to_numpy()
        predicted = group["prediction"].to_numpy()
        positive = labels == 1
        negative = labels == 0
        tp = int((positive & (predicted == 1)).sum())
        tn = int((negative & (predicted == 0)).sum())
        fp, fn = int(negative.sum()) - tn, int(positive.sum()) - tp
        sensitivity = tp / int(positive.sum()) if positive.any() else None
        specificity = tn / int(negative.sum()) if negative.any() else None
        both_classes = positive.any() and negative.any()
        rows.append({
            "patient_group": patient, "samples": len(group),
            "negative_samples": int(negative.sum()), "positive_samples": int(positive.sum()),
            "balanced_accuracy": (sensitivity + specificity) / 2 if both_classes else None,
            "sensitivity": sensitivity, "specificity": specificity,
            "accuracy": float((predicted == labels).mean()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "auroc": classification_metrics(labels, group["probability"].to_numpy(), threshold)["auroc"]
            if both_classes else None,
        })
    return pd.DataFrame(rows)


def validate_checkpoint(weights: str | Path, device: str = "auto",
                        output_dir: str | Path | None = None,
                        allow_smoke_weights: bool = False) -> dict:
    weights = Path(weights).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else weights.parent / "validation"
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    model, checkpoint, resolved_device = load_trained_model(weights, device)
    smoke_test = bool(checkpoint["smoke_test"])
    if smoke_test and not allow_smoke_weights:
        raise ValueError("Smoke checkpoints are not baseline results; use --allow-smoke-weights only for a functional check")
    config = checkpoint["config"]
    data_root = task_path(config["data_root"])
    manifest = read_manifest(weights.parent / "split_manifest.csv", load_records(data_root))
    valid = manifest.loc[manifest["split"] == "valid"].reset_index(drop=True)
    if smoke_test:
        subset = pd.read_csv(weights.parent / "smoke_subset.csv", dtype={"sample_id": str, "path": str, "patient_group": str})
        subset = subset.loc[subset["split"] == "valid"].reset_index(drop=True)
        if subset["sample_id"].duplicated().any() or not set(subset["sample_id"]).issubset(set(valid["sample_id"])):
            raise ValueError("The smoke validation subset is not part of the saved validation partition")
        selected = valid.set_index("sample_id").loc[subset["sample_id"]].reset_index()
        try:
            pd.testing.assert_frame_equal(selected[list(REQUIRED_COLUMNS)], subset[list(REQUIRED_COLUMNS)], check_dtype=False)
        except AssertionError as exc:
            raise ValueError("Smoke subset metadata differs from the saved manifest") from exc
        valid = selected
    if set(valid["label"]) != {0, 1}:
        raise ValueError("Validation must contain both classes")
    class_weights = torch.tensor(checkpoint.get("validation_class_weights", checkpoint["class_weights"]),
                                 dtype=torch.float32, device=resolved_device)
    if class_weights.shape != (2,) or not bool(torch.isfinite(class_weights).all()) or not bool((class_weights > 0).all()):
        raise ValueError("Checkpoint class weights must be two finite positive values")
    loader = DataLoader(ImageDataset(valid, data_root, config["image_size"]),
                        batch_size=config["batch_size"], shuffle=False, num_workers=0)
    probabilities = []
    numerator_sum, denominator_sum = 0.0, 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(resolved_device), labels.to(resolved_device)
            logits = model(images)
            numerator_sum += F.cross_entropy(logits, labels, weight=class_weights, reduction="sum").item()
            denominator_sum += class_weights[labels].sum().item()
            probabilities.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
    probabilities = np.concatenate(probabilities)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise RuntimeError("Model returned invalid probabilities")
    threshold = float(checkpoint["threshold"])
    metrics = classification_metrics(valid["label"].to_numpy(), probabilities, threshold)
    metrics["loss"] = float(numerator_sum / denominator_sum)
    predictions = valid.copy()
    predictions["probability"] = probabilities
    predictions["prediction"] = (probabilities >= threshold).astype(int)
    saved_predictions = pd.read_csv(weights.parent / "valid_predictions.csv", dtype={"sample_id": str})
    if saved_predictions["sample_id"].duplicated().any() or set(saved_predictions["sample_id"]) != set(valid["sample_id"]):
        raise ValueError("Saved prediction IDs differ from the validation partition")
    aligned = saved_predictions.set_index("sample_id").loc[valid["sample_id"]]
    if not np.array_equal(aligned["label"].to_numpy(), valid["label"].to_numpy()):
        raise ValueError("Saved prediction labels differ from the validation partition")
    saved_probabilities = aligned["probability"].to_numpy()
    probability_parity = bool(np.allclose(probabilities, saved_probabilities, atol=1e-5, rtol=1e-5))
    prediction_mismatches = int((predictions["prediction"].to_numpy() != aligned["prediction"].to_numpy()).sum())
    saved_metrics = checkpoint["valid_metrics"]
    metric_parity = bool(np.isclose(metrics["balanced_accuracy"], saved_metrics["balanced_accuracy"], atol=1e-12, rtol=0))
    checks = {
        "manifest_matches_current_data": True,
        "patient_groups_disjoint": True,
        "sample_count_matches_checkpoint": len(valid) == saved_metrics["samples"],
        "probabilities_finite_and_in_range": True,
        "saved_probabilities_match": probability_parity,
        "max_absolute_probability_difference": float(np.max(np.abs(probabilities - saved_probabilities))),
        "saved_prediction_mismatches": prediction_mismatches,
        "saved_balanced_accuracy_matches": metric_parity,
    }
    passed = checks["sample_count_matches_checkpoint"] and probability_parity and prediction_mismatches == 0 and metric_parity
    result = {
        "weights": str(weights), "model_name": checkpoint["model_name"],
        "architecture": checkpoint["architecture"], "selected_epoch": checkpoint["epoch"],
        "device": str(resolved_device), "smoke_test": smoke_test,
        "metric_unit": "image", "validation_groups": sorted(valid["patient_group"].unique().tolist()),
        "metrics": metrics, "checks": checks, "all_checks_passed": bool(passed),
        "full_split_summary": split_summary(manifest),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    per_patient_metrics(predictions, threshold).to_csv(output_dir / "per_patient_metrics.csv", index=False)
    write_json(output_dir / "metrics.json", result)
    print(f"Validation BA={metrics['balanced_accuracy']:.6f}; checks passed={passed}; output={output_dir}", flush=True)
    if not passed:
        raise RuntimeError(f"Independent validation differs from the saved checkpoint; inspect {output_dir / 'metrics.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-smoke-weights", action="store_true")
    args = parser.parse_args()
    validate_checkpoint(args.weights, args.device, args.output_dir, args.allow_smoke_weights)


if __name__ == "__main__":
    main()
