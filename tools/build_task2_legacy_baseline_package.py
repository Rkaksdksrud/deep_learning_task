"""Package the separately preserved original baseline without changing it.

The old training source snapshot is unavailable. Tensor/prediction parity with
the already audited control is checked instead of inventing source provenance.
The output directory must be new. No network or training is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from build_task2_augmentation_lr_packages import json_bytes, read, require, safe, sha, state_equal, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = "resnet18_baseline_20261101"
CONTROL = "resnet18_20261111_control_f0_s42"
TAG = "task2-augmentation-lr-20261125-v1"
MODEL = ROOT / "models/task2-legacy-baseline-20261101"


def build(source: Path, output: Path) -> None:
    import torch
    require(not output.exists(), "Use a new artifact directory")
    require(not (MODEL / "manifest.json").exists(), "Existing manifests are preserved")
    run, control = source / "runs" / RUN, source / "runs" / CONTROL
    original_sha, control_sha = sha(run / "resnet.pt"), sha(control / "resnet.pt")
    original = torch.load(run / "resnet.pt", map_location="cpu", weights_only=True)
    reference = torch.load(control / "resnet.pt", map_location="cpu", weights_only=True)
    state_equal(original["state_dict"], reference["state_dict"], torch)
    require(original["model_name"] == "resnet" and original["architecture"] == "resnet18"
            and original["smoke_test"] is False and original["threshold"] == .5,
            "Unexpected legacy model format")
    require(original["epoch"] == reference["epoch"] == 2, "Selection epoch differs")
    require({k: v for k, v in original["valid_metrics"].items() if k not in {"elapsed_seconds", "images_per_second"}} ==
            {k: v for k, v in reference["valid_metrics"].items() if k not in {"elapsed_seconds", "images_per_second"}},
            "Legacy/control prediction metrics differ")
    for name in ("valid_predictions.csv", "split_manifest.csv"):
        require((run / name).read_bytes() == (control / name).read_bytes(), "Legacy/control records differ")
    audited = read(control / "validation/metrics.json")
    known = read(ROOT / "models/task2-resnet18-20261111/manifest.json")
    known_control = next(row for row in known["models"] if row["id"] == "control_f0_s42")
    require(audited["all_checks_passed"] is True and known_control["checkpoint_sha256"] == control_sha,
            "The parity control must retain its audited records and published checkpoint digest")
    summary, best = read(run / "training_summary.json"), read(run / "best_metrics.json")
    portable = {**original, "config": {**original["config"], "data_root": "../data/task2", "output_root": "../runs/task2"}}
    safe({key: value for key, value in portable.items() if key != "state_dict"})
    MODEL.mkdir(parents=True, exist_ok=True)
    for name in ("predict.py", "download.py", "requirements.txt"):
        shutil.copyfile(ROOT / "models/task2-augmentation-lr-20261125" / name, MODEL / name)
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="task2_legacy_package_") as directory:
        weights = Path(directory) / "model.pt"
        torch.save(portable, weights)
        check = torch.load(weights, map_location="cpu", weights_only=True)
        state_equal(original["state_dict"], check["state_dict"], torch)
        require(all(check[k] == original[k] for k in original if k not in {"state_dict", "config"}), "Other metadata changed")
        require(set(check["config"]) == set(original["config"]) and
                {k for k in original["config"] if original["config"][k] != check["config"][k]} == {"data_root", "output_root"},
                "Only two portable config paths may change")
        public_sha = sha(weights)
        metrics = {k: v for k, v in best.items() if k not in {"epoch", "smoke_test"}}
        proof = {"original_checkpoint_sha256": original_sha, "published_checkpoint_sha256": public_sha,
                 "changed_metadata_fields": ["config.data_root", "config.output_root"], "all_122_model_tensors_exact": True,
                 "other_metadata_exact": True, "original_local_checkpoint_preserved": True,
                 "parity_control_checkpoint_sha256": control_sha, "control_all_tensors_exact": True,
                 "saved_predictions_and_split_bytes_equal_control": True,
                 "direct_sha_bound_legacy_reinference_this_export": False,
                 "historical_control_audit_included_checkpoint_digest": False,
                 "control_checkpoint_matches_preserved_published_manifest": True,
                 "original_training_source_snapshot_available": False}
        metadata = {"format_version": 1, "run_id": RUN, "architecture": "resnet18", "model_name": "resnet",
                    "checkpoint_filename": "model.pt", "checkpoint_sha256": public_sha, **proof,
                    "config": portable["config"], "best_epoch": original["epoch"], "metrics": metrics,
                    "training_seconds": summary["elapsed_seconds"], "epochs_completed": summary["epochs_completed"],
                    "scope": "Preserved legacy holdout baseline, not an extra candidate or final all-data model"}
        readme = """# Original ResNet18 baseline, 2026-11-01

Preserved legacy run; BA47.5536%, best epoch2/end9. This is development validation, not final Test.
No original training source snapshot is available. All model tensors and saved
predictions match the independently audited 11/11 control. This is a preservation
and parity check, not a new direct legacy reinference in this export.

Only config.data_root/output_root in the public checkpoint have portable values;
all tensors and other metadata are exact. Original/public SHA256 are distinct.
No source images, labels, patient/sample lists, split or resume state is included.

Install requirements.txt, then run:
python predict.py --images input_images --output predictions.csv --device cpu

One randomly initialized ResNet18, threshold0.5; no inference augmentation or ensemble.
"""
        contents = {"model.pt": weights.read_bytes(), "predict.py": (MODEL / "predict.py").read_bytes(),
                    "requirements.txt": (MODEL / "requirements.txt").read_bytes(),
                    "metadata.json": json_bytes(metadata), "README.md": readme.encode()}
        import hashlib
        checks = {name: hashlib.sha256(payload).hexdigest() for name, payload in contents.items()}
        contents["SHA256SUMS.json"] = json_bytes(checks)
        archive = output / (RUN + ".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
            for name, payload in contents.items():
                zipped.writestr(name, payload)
        row = {"id": RUN, "architecture": "resnet18", "image_size": 224, "validation_fold": 0, "seed": 42,
               "selected_epoch": 2, "balanced_accuracy": metrics["balanced_accuracy"],
               "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"],
               "checkpoint_filename": "model.pt", "checkpoint_sha256": public_sha, **proof,
               "asset_name": archive.name, "archive_size_bytes": archive.stat().st_size, "archive_sha256": sha(archive),
               "download_url": f"https://github.com/Rkaksdksrud/deep_learning_task/releases/download/{TAG}/{archive.name}"}
    manifest = {"format_version": 1, "repository": "Rkaksdksrud/deep_learning_task", "release_tag": TAG,
                "checkpoint_filename": "model.pt", "total_models": 1, "models": [row], "omitted_runs": [],
                "contains_training_data": False, "contains_ensembles": False, "contains_training_resume_state": False,
                "notes": ["Separately preserved initial baseline; original training source snapshot unavailable.",
                          "Not a new performance-improvement trial; same tensors/predictions as the audited control."]}
    write_json(MODEL / "manifest.json", manifest)
    (output / "legacy_baseline_manifest.json").write_bytes(json_bytes(manifest))
    write_json(MODEL / "preservation_checks.json", proof)
    require(sha(run / "resnet.pt") == original_sha and sha(control / "resnet.pt") == control_sha,
            "An original checkpoint changed")
    print("Packaged one preserved legacy baseline; original source and tensors remain unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-task2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.source_task2.resolve(), args.output.resolve())
