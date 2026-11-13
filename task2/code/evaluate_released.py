"""Evaluate a released holdout checkpoint without sharing per-sample metadata.

Requires the original authorized local dataset. This regenerates the patient
split and compares aggregate BA/confusion matrix. It cannot verify the original
saved probabilities, because those sample-level records are not distributed.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import classification_metrics, load_trained_model, task_path, write_json
from data import ImageDataset, load_records, make_splits


# Original course dataset fingerprints; no images, labels, IDs, or paths are embedded.
ORIGINAL_LABELS_SHA256 = "74df1038da880f2a670f448b14ae0a99624b018972a6ecc0312406fd2c3ed4ad"
ORIGINAL_IMAGES_SHA256 = "e0306251f0c05ebbf477728328a653b7f38cb618b885a2895223fccec953ae79"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_collection_sha256(records, data_root: Path) -> str:
    """SHA256 of sorted UTF-8 path + NUL + hex file SHA256 + newline records."""
    digest = hashlib.sha256()
    for relative in sorted(records["path"].tolist()):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256_file(data_root / relative).encode("ascii") + b"\n")
    return digest.hexdigest()


def evaluate_released(weights: str | Path, data_root: str | Path,
                      output_dir: str | Path, device: str = "cpu") -> dict:
    weights = Path(weights).expanduser().resolve()
    data_root = task_path(data_root)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    labels_sha = sha256_file(data_root / "labels.csv")
    if labels_sha != ORIGINAL_LABELS_SHA256:
        raise ValueError("labels.csv differs from the original dataset SHA256; use the unmodified authorized course CSV")
    records = load_records(data_root)
    images_sha = image_collection_sha256(records, data_root)
    if images_sha != ORIGINAL_IMAGES_SHA256:
        raise ValueError("Image collection differs from the original byte-level dataset fingerprint")

    model, checkpoint, resolved_device = load_trained_model(weights, device)
    if checkpoint["smoke_test"]:
        raise ValueError("This evaluator accepts full holdout checkpoints only, not smoke-test weights")
    config = checkpoint["config"]
    if float(checkpoint["threshold"]) != 0.5:
        raise ValueError("Released Task2 experiments use the fixed threshold 0.5")
    manifest = make_splits(records, config["n_splits"], config["val_fold"])
    valid = manifest.loc[manifest["split"] == "valid"].reset_index(drop=True)
    loader = DataLoader(ImageDataset(valid, data_root, config["image_size"]),
                        batch_size=config["batch_size"], shuffle=False, num_workers=0)
    probabilities = []
    with torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(resolved_device))
            probabilities.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
    probabilities = np.concatenate(probabilities)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise RuntimeError("Model returned invalid probabilities")
    measured = classification_metrics(valid["label"].to_numpy(), probabilities, checkpoint["threshold"])
    expected = checkpoint["valid_metrics"]
    checks = {
        "original_labels_sha256_matches": True,
        "original_images_sha256_matches": True,
        "sample_count_matches": measured["samples"] == expected["samples"],
        "balanced_accuracy_matches": bool(np.isclose(measured["balanced_accuracy"], expected["balanced_accuracy"], atol=1e-12, rtol=0)),
        "confusion_matrix_matches": measured["confusion_matrix"] == expected["confusion_matrix"],
        "original_saved_probability_parity_checked": False,
    }
    matched = checks["sample_count_matches"] and checks["balanced_accuracy_matches"] and checks["confusion_matrix_matches"]
    result = {
        "evaluation_type": "regenerated patient holdout; aggregate comparison only",
        "weights": str(weights), "architecture": checkpoint["architecture"],
        "selected_epoch": checkpoint["epoch"], "seed": config["seed"],
        "n_splits": config["n_splits"], "val_fold": config["val_fold"],
        "device": str(resolved_device), "data_root": str(data_root),
        "labels_sha256": labels_sha, "images_sha256": images_sha,
        "metrics": measured, "checkpoint_reference_metrics": expected,
        "checks": checks, "aggregate_checks_passed": bool(matched),
        "limitations": "Original sample-level split/prediction files are not distributed. Their saved-probability parity is not checked. Use the pinned environment; differing backends may change predictions near the threshold.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "metrics.json", result)
    print(f"Holdout BA={measured['balanced_accuracy']:.6f}; aggregate checks passed={matched}; output={output_dir}", flush=True)
    if not matched:
        raise RuntimeError(f"Aggregate metrics differ from the released checkpoint; inspect {output_dir / 'metrics.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Original local labels.csv/images root; relative paths resolve from task2/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate_released(args.weights, args.data_root, args.output_dir, args.device)


if __name__ == "__main__":
    main()
