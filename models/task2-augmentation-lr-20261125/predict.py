"""Standalone inference for one Task2 ResNet18, VGG11 or GoogLeNet checkpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torchvision

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ARCHITECTURES = {"resnet": "resnet18", "vgg": "vgg11", "googlenet": "googlenet"}


def load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.pad(image.convert("RGB"), (image_size, image_size),
                             method=Image.Resampling.BILINEAR, color=(0, 0, 0))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((pixels - 0.5) / 0.5).permute(2, 0, 1).contiguous()


def load_model(weights: str | Path, device: str = "cpu"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model_name = checkpoint.get("model_name")
    if checkpoint.get("format_version") != 1 or model_name not in ARCHITECTURES:
        raise ValueError("Expected an inference format-v1 Task2 CNN checkpoint")
    if checkpoint.get("architecture") != ARCHITECTURES[model_name]:
        raise ValueError("Unsupported or inconsistent checkpoint architecture")
    if checkpoint.get("smoke_test") is not False:
        raise ValueError("Smoke checkpoints are not supported by this model package")
    expected = {"version": 1, "color_mode": "RGB", "resize": "aspect_preserving_letterbox",
                "interpolation": "bilinear", "padding_color": [0, 0, 0], "pixel_scale": 255.0,
                "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    prep, config = checkpoint.get("preprocessing", {}), checkpoint.get("config", {})
    if any(prep.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported checkpoint preprocessing; use its original inference code")
    if checkpoint.get("label_mapping") != {0: "absent", 1: "present"} or checkpoint.get("threshold") != 0.5:
        raise ValueError("Unsupported label mapping or threshold")
    size = prep.get("image_size")
    if type(size) is not int or size < 64 or size != config.get("image_size") or config.get("threshold") != 0.5:
        raise ValueError("Inconsistent image-size or threshold metadata")
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; select --device cpu")
    if model_name == "resnet":
        model = torchvision.models.resnet18(weights=None, num_classes=2)
    elif model_name == "vgg":
        model = torchvision.models.vgg11(weights=None, num_classes=2)
    else:
        model = torchvision.models.googlenet(weights=None, num_classes=2, aux_logits=True,
                                             transform_input=False, init_weights=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(resolved).eval(), checkpoint, resolved


def predict_directory(images, output, weights=None, device="cpu", batch_size=16):
    weights = Path(weights) if weights else Path(__file__).with_name("model.pt")
    image_root, output = Path(images).resolve(), Path(output).resolve()
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)
    if output.exists():
        raise FileExistsError(f"Use a new output file: {output}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    paths = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError("No supported image files found")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Image stems must be unique because they form sample_id")
    model, checkpoint, resolved = load_model(weights, device)
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([load_image_tensor(p, checkpoint["config"]["image_size"])
                                 for p in paths[start:start + batch_size]])
            probabilities.extend(model(batch.to(resolved)).softmax(1)[:, 1].cpu().numpy().tolist())
    values = np.asarray(probabilities)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Invalid probabilities")
    frame = pd.DataFrame({"sample_id": [p.stem for p in paths],
                          "path": [p.relative_to(image_root).as_posix() for p in paths],
                          "probability": values, "prediction": (values >= checkpoint["threshold"]).astype(int)})
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} predictions to {output}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path(__file__).with_name("model.pt"))
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    predict_directory(args.images, args.output, args.weights, args.device, args.batch_size)


if __name__ == "__main__":
    main()
