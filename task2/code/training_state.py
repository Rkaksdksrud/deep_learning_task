"""Atomic epoch-boundary state; inference checkpoints keep their old format."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch


STATE_NAME = "last_training_state.pt"


def capture_rng(loaders: dict) -> dict:
    algorithm, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": algorithm, "keys": torch.tensor(keys.astype(np.int64)),
            "position": position, "has_gauss": has_gauss, "cached_gauss": cached_gauss,
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": {name: loader.generator.get_state() for name, loader in loaders.items()},
    }


def restore_rng(state: dict, loaders: dict) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["algorithm"], value["keys"].numpy().astype(np.uint32),
                         value["position"], value["has_gauss"], value["cached_gauss"]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved training RNG state")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if set(state["loaders"]) != set(loaders):
        raise ValueError("DataLoader names differ from the saved training state")
    for name, loader in loaders.items():
        loader.generator.set_state(state["loaders"][name])


def atomic_save(path: Path, value: dict) -> float:
    started = time.perf_counter()
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
    return time.perf_counter() - started


def read_state(path: str | Path) -> tuple[Path, dict]:
    path = Path(path).expanduser().resolve()
    path = path / STATE_NAME if path.is_dir() else path
    if path.name != STATE_NAME or not path.is_file():
        raise ValueError(f"--resume requires a run directory containing {STATE_NAME}, or that file")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("training_state_version") != 1 or state.get("purpose") != "epoch_boundary_resume":
        raise ValueError("This is not a resumable training state; best-model weights alone cannot resume")
    return path.parent, state


def check_metadata(saved: dict, expected: dict) -> None:
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Refusing resume: {key} differs from saved training state")


def check_progress(state: dict) -> None:
    epoch = state["completed_epoch"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError("Invalid completed epoch in training state")
    if [row["epoch"] for row in state["history"]] != list(range(1, epoch + 1)):
        raise ValueError("Training-state history is not contiguous through completed_epoch")
    if not 1 <= state["best_result"]["epoch"] <= epoch:
        raise ValueError("Invalid selected epoch in training state")
    if not 0 <= state["stale_epochs"] <= epoch:
        raise ValueError("Invalid early-stopping counter in training state")
