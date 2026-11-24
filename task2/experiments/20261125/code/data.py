"""Shared image loading and patient-group validation split for Task 2."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset


REQUIRED_COLUMNS = ("sample_id", "path", "label", "patient_group")


def validate_augmentation_options(geometric_horizontal_flip=True, brightness_jitter=0.0,
                                  contrast_jitter=0.0, training=None) -> None:
    if type(geometric_horizontal_flip) is not bool:
        raise ValueError("geometric_horizontal_flip must be a boolean")
    if training is not None and type(training) is not bool:
        raise ValueError("training must be a boolean when specified")
    for name, value in (("brightness_jitter", brightness_jitter), ("contrast_jitter", contrast_jitter)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number in [0, 1]")


def load_image_tensor(image_path: Path, image_size: int = 224, augment: bool = False,
                      geometric_horizontal_flip: bool = True, brightness_jitter: float = 0.0,
                      contrast_jitter: float = 0.0, training: bool | None = None) -> torch.Tensor:
    """Apply identical preprocessing during training and label-free inference."""
    validate_augmentation_options(geometric_horizontal_flip, brightness_jitter, contrast_jitter, training)
    # Existing callers using augment=True retain the original training policy.
    # An explicit training=False always disables every augmentation.
    training = bool(augment) if training is None else training
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if training and augment:
            # Resize first for inexpensive, single-image training augmentation.
            # expand=True preserves all image corners before the final letterbox.
            image.thumbnail((image_size, image_size), Image.Resampling.BILINEAR)
            # Consume the legacy flip decision even in rotation-only mode so
            # each paired sample receives the same subsequent rotation angle.
            flip_decision = torch.rand(()).item() < 0.5
            angle = torch.empty(()).uniform_(-7.0, 7.0).item()
        # Adjust only the real image, before expand/letterbox create black
        # padding. In particular, contrast must not turn padded corners gray.
        if training and brightness_jitter > 0:
            factor = torch.empty(()).uniform_(1 - brightness_jitter, 1 + brightness_jitter).item()
            image = TF.adjust_brightness(image, factor)
        if training and contrast_jitter > 0:
            factor = torch.empty(()).uniform_(1 - contrast_jitter, 1 + contrast_jitter).item()
            image = TF.adjust_contrast(image, factor)
        if training and augment:
            if flip_decision and geometric_horizontal_flip:
                image = ImageOps.mirror(image)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR,
                              expand=True, fill=0)
        image = ImageOps.pad(
            image, (image_size, image_size),
            method=Image.Resampling.BILINEAR, color=(0, 0, 0),
        )
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((pixels - 0.5) / 0.5).permute(2, 0, 1).contiguous()


def load_records(data_root: Path) -> pd.DataFrame:
    """Read labels.csv and validate IDs, binary labels, and image locations."""
    data_root = Path(data_root).resolve()
    records = pd.read_csv(data_root / "labels.csv", dtype=str)
    missing = set(REQUIRED_COLUMNS) - set(records.columns)
    if missing:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing)}")
    if records.empty:
        raise ValueError("labels.csv contains no samples")
    for column in REQUIRED_COLUMNS:
        if records[column].isna().any() or records[column].str.strip().eq("").any():
            raise ValueError(f"Column {column!r} contains missing or empty values")
    for column in ("sample_id", "path"):
        if records[column].duplicated().any():
            raise ValueError(f"Column {column!r} must contain unique values")
    if not set(records["label"]).issubset({"0", "1"}):
        raise ValueError("Labels must be exactly 0 or 1")
    records["label"] = records["label"].astype(int)
    if set(records["label"]) != {0, 1}:
        raise ValueError("Both labels 0 and 1 must be present")

    resolved_paths = []
    for relative_path in records["path"]:
        image_path = (data_root / relative_path).resolve()
        if Path(relative_path).is_absolute() or not image_path.is_relative_to(data_root):
            raise ValueError(f"Image path must stay inside data_root: {relative_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        resolved_paths.append(image_path)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Multiple records refer to the same image file")
    return records.sort_values("sample_id", kind="stable").reset_index(drop=True)


def make_splits(
    records: pd.DataFrame, n_splits: int = 5, val_fold: int = 0
) -> pd.DataFrame:
    """Keep one group-stratified fold for validation, all others for training.

    The private TA test set is external and is never simulated by holding out
    another part of the supplied training data. Sorting and shuffle=False make
    the split deterministic without choosing a seed to optimize validation.
    """
    if n_splits < 2 or not 0 <= val_fold < n_splits:
        raise ValueError("Require n_splits >= 2 and 0 <= val_fold < n_splits")
    records = records.sort_values("sample_id", kind="stable").reset_index(drop=True).copy()
    if records["patient_group"].nunique() < n_splits:
        raise ValueError("Not enough patient groups for the requested number of folds")
    if set(records["label"]) != {0, 1}:
        raise ValueError("Both labels 0 and 1 must be present")
    if records["label"].value_counts().min() < n_splits:
        raise ValueError("Each class needs at least n_splits samples")

    folds = np.full(len(records), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    for fold, (_, valid_indices) in enumerate(
        splitter.split(records, records["label"], groups=records["patient_group"])
    ):
        folds[valid_indices] = fold
    if (folds < 0).any():
        raise RuntimeError("Some samples were not assigned a fold")
    records["fold"] = folds
    records["split"] = np.where(folds == val_fold, "valid", "train")

    train_groups = set(records.loc[records["split"] == "train", "patient_group"])
    valid_groups = set(records.loc[records["split"] == "valid", "patient_group"])
    if train_groups & valid_groups:
        raise RuntimeError("Patient groups overlap between training and validation")
    for split in ("train", "valid"):
        if set(records.loc[records["split"] == split, "label"]) != {0, 1}:
            raise ValueError(f"Split {split!r} must contain both classes")
    return records


def split_summary(records_with_splits: pd.DataFrame) -> dict:
    """Return JSON-serializable class counts and patient groups per partition."""
    summary = {}
    for split in ("train", "valid"):
        records = records_with_splits.loc[records_with_splits["split"] == split]
        counts = records["label"].value_counts()
        summary[split] = {
            "samples": int(len(records)),
            "groups": int(records["patient_group"].nunique()),
            "patient_groups": sorted(records["patient_group"].unique().tolist()),
            "label_counts": {str(label): int(counts.get(label, 0)) for label in (0, 1)},
        }
    return summary


class ImageDataset(Dataset):
    """Load RGB images with a deterministic, aspect-preserving square letterbox."""

    def __init__(
        self, records: pd.DataFrame, data_root: Path, image_size: int = 224,
        augment: bool = False, geometric_horizontal_flip: bool = True,
        brightness_jitter: float = 0.0, contrast_jitter: float = 0.0,
        training: bool | None = None,
    ):
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.augment = bool(augment)
        validate_augmentation_options(geometric_horizontal_flip, brightness_jitter, contrast_jitter, training)
        self.geometric_horizontal_flip = geometric_horizontal_flip
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.training = self.augment if training is None else training
        self.records = records.reset_index(drop=True).copy()
        self.data_root = Path(data_root).resolve()
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[index]
        tensor = load_image_tensor(self.data_root / row["path"], self.image_size, augment=self.augment,
                                   geometric_horizontal_flip=self.geometric_horizontal_flip,
                                   brightness_jitter=self.brightness_jitter, contrast_jitter=self.contrast_jitter,
                                   training=self.training)
        return tensor, torch.tensor(int(row["label"]), dtype=torch.long)
