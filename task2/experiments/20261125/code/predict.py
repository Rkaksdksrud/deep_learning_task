"""Predict binary sheath visibility for an image directory without labels.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import load_trained_model
from data import load_image_tensor


class PredictionDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int):
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return load_image_tensor(self.paths[index], self.image_size)


def predict_directory(weights: str | Path, images: str | Path, output: str | Path,
                      device: str = "auto", batch_size: int = 16,
                      allow_smoke_weights: bool = False) -> pd.DataFrame:
    image_root = Path(images).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)
    if output.exists():
        raise FileExistsError(f"Choose a new output path: {output}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    paths = sorted(path for path in image_root.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not paths:
        raise ValueError(f"No image files found under {image_root}")
    model, checkpoint, resolved_device = load_trained_model(weights, device)
    if checkpoint["smoke_test"] and not allow_smoke_weights:
        raise ValueError("Smoke-test weights are not trained baselines. Use --allow-smoke-weights only to check inference.")
    loader = DataLoader(PredictionDataset(paths, checkpoint["config"]["image_size"]),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch.to(resolved_device))
            probabilities.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
    probabilities = np.concatenate(probabilities)
    predictions = pd.DataFrame({
        "sample_id": [path.stem for path in paths],
        "path": [path.relative_to(image_root).as_posix() for path in paths],
        "probability": probabilities,
        "prediction": (probabilities >= checkpoint["threshold"]).astype(int),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Saved {len(predictions)} predictions to {output}", flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-smoke-weights", action="store_true")
    args = parser.parse_args()
    predict_directory(args.weights, args.images, args.output, args.device, args.batch_size, args.allow_smoke_weights)


if __name__ == "__main__":
    main()
