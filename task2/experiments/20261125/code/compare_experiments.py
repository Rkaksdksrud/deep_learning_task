"""Read saved Task2 runs and compare matched, independently verified experiments.

No torch import, checkpoint loading, inference, training, or input-file writes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


TASK_DIR = Path(__file__).resolve().parents[1]
FINISHED = {"실행 완료", "검증 완료"}
METRICS = ("balanced_accuracy", "sensitivity", "specificity", "tn", "fp", "fn", "tp")
OPERATIONAL_KEYS = {
    "output_root", "device", "resolved_device", "num_workers", "cpu_threads",
    "experiment_role", "experiment_name", "description", "notes", "run_name",
}
DEFAULT_CHANGE_KEYS = {"class_weighted_loss", "geometric_augmentation"}
RUN_COLUMNS = [
    "run", "run_dir", "role", "status", "execution_state", "model", "seed", "fold",
    "smoke_test", "epochs_completed", "best_epoch", "elapsed_seconds", "balanced_accuracy",
    "sensitivity", "specificity", "tn", "fp", "fn", "tp", "samples", "split_fingerprint",
    "independent_validation", "aggregate_checks_passed", "validation_dir", "notes",
]
PATIENT_COLUMNS = ["run", "role", "seed", "fold", "patient_group", "samples", "negative_samples",
                   "positive_samples", *METRICS, "ba_interpretation"]
PAIR_COLUMNS = ["baseline", "candidate", "seed", "fold", "comparison", "comparison_family",
                "changed_config", "baseline_best_epoch", "candidate_best_epoch",
                "baseline_elapsed_seconds", "candidate_elapsed_seconds", "delta_elapsed_seconds",
                "time_ratio", "patients_worse_ba", "patients_better_ba", "max_patient_ba_drop"]
for _metric in METRICS:
    PAIR_COLUMNS += [f"baseline_{_metric}", f"candidate_{_metric}", f"delta_{_metric}"]
PATIENT_PAIR_COLUMNS = ["baseline", "candidate", "seed", "fold", "comparison", "patient_group",
                        "samples", "negative_samples", "positive_samples", "ba_interpretation"]
for _metric in METRICS:
    PATIENT_PAIR_COLUMNS += [f"baseline_{_metric}", f"candidate_{_metric}", f"delta_{_metric}"]
AGGREGATE_COLUMNS = ["comparison", "comparison_family", "matched_pairs", "matched_seed_folds"]
for _metric in ("balanced_accuracy", "sensitivity", "specificity", "elapsed_seconds"):
    AGGREGATE_COLUMNS += [f"mean_baseline_{_metric}", f"mean_candidate_{_metric}",
                          f"mean_delta_{_metric}", f"stdev_delta_{_metric}"]


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def read_json(path: Path, notes: list[str]) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (OSError, ValueError) as exc:
        notes.append(f"{path.name} 읽기 실패(기록 중일 수 있음): {exc}")
        return {}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def number(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def metrics(rows: list[dict]) -> dict:
    counts = {name: 0 for name in ("tn", "fp", "fn", "tp")}
    for row in rows:
        label, predicted = int(row["label"]), int(row["prediction"])
        if label not in (0, 1) or predicted not in (0, 1):
            raise ValueError("label/prediction must be binary")
        counts[{(0, 0): "tn", (0, 1): "fp", (1, 0): "fn", (1, 1): "tp"}[label, predicted]] += 1
    negative, positive = counts["tn"] + counts["fp"], counts["tp"] + counts["fn"]
    tpr = counts["tp"] / positive if positive else None
    tnr = counts["tn"] / negative if negative else None
    return {**counts, "samples": len(rows), "negative_samples": negative, "positive_samples": positive,
            "sensitivity": tpr, "specificity": tnr,
            "balanced_accuracy": (tpr + tnr) / 2 if positive and negative else None,
            "ba_interpretation": "both_classes" if positive and negative else "single_class_BA_undefined"}


def metrics_match(calculated: dict, saved: dict) -> bool:
    for key in ("balanced_accuracy", "sensitivity", "specificity"):
        left, right = number(calculated.get(key)), number(saved.get(key))
        if left is None or right is None or not math.isclose(left, right, abs_tol=1e-10, rel_tol=0):
            return False
    cm = saved.get("confusion_matrix")
    return cm == [[calculated["tn"], calculated["fp"]], [calculated["fn"], calculated["tp"]]]


def validate_saved_predictions(run_dir: Path, config: dict, best: dict) -> tuple[dict, list[dict], str]:
    manifest = read_csv(run_dir / "split_manifest.csv")
    metadata = ("sample_id", "path", "label", "patient_group", "split")
    if any(not all(row.get(key) for key in metadata) for row in manifest):
        raise ValueError("manifest has missing required metadata")
    if len({row["sample_id"] for row in manifest}) != len(manifest):
        raise ValueError("duplicate manifest sample_id")
    if len({row["path"] for row in manifest}) != len(manifest):
        raise ValueError("duplicate manifest path")
    if {row["split"] for row in manifest} != {"train", "valid"}:
        raise ValueError("manifest must contain train/valid only")
    groups = {split: {row["patient_group"] for row in manifest if row["split"] == split}
              for split in ("train", "valid")}
    if groups["train"] & groups["valid"]:
        raise ValueError("patient leakage between train and valid")
    valid = {row["sample_id"]: row for row in manifest if row["split"] == "valid"}
    rows = read_csv(run_dir / "valid_predictions.csv")
    if len({row["sample_id"] for row in rows}) != len(rows) or set(valid) != {r["sample_id"] for r in rows}:
        raise ValueError("prediction IDs differ from validation manifest")
    threshold = number(config.get("threshold"))
    if threshold is None or not 0 < threshold < 1:
        raise ValueError("missing/invalid threshold")
    if number(best.get("threshold")) != threshold:
        raise ValueError("best_metrics threshold differs from config")
    for row in rows:
        source = valid[row["sample_id"]]
        if any(row.get(key) != source[key] for key in metadata[:-1]):
            raise ValueError("prediction metadata differs from manifest")
        probability = number(row.get("probability"))
        if probability is None or not 0 <= probability <= 1:
            raise ValueError("probability outside [0, 1]")
        if int(row["prediction"]) != int(probability >= threshold):
            raise ValueError("saved prediction differs from fixed threshold")
    calculated = metrics(rows)
    if not metrics_match(calculated, best) or int(best.get("samples", -1)) != len(rows):
        raise ValueError("recomputed predictions disagree with best_metrics")
    patient_rows = [dict(patient_group=patient, **metrics([r for r in rows if r["patient_group"] == patient]))
                    for patient in sorted(groups["valid"])]
    fingerprint = digest(sorted(tuple(row[key] for key in metadata) for row in manifest))
    return calculated, patient_rows, fingerprint


def role_for(path: Path, config: dict, baselines: set[Path], legacy: set[Path]) -> str:
    if path in legacy:
        return "legacy"
    if path in baselines:
        return "baseline"
    if baselines:
        return "candidate"
    declared = config.get("experiment_role")
    if declared in {"legacy", "baseline", "candidate"}:
        return declared
    return "baseline" if any(word in path.name.lower() for word in ("baseline", "control")) else "candidate"


def inspect_run(path: Path, baselines: set[Path], legacy: set[Path], validation_dirs: dict[Path, Path]) -> dict:
    notes: list[str] = []
    config = read_json(path / "config.json", notes)
    summary = read_json(path / "training_summary.json", notes)
    best = read_json(path / "best_metrics.json", notes)
    status_file = read_json(path / "run_status.json", notes) or read_json(path / "status.json", notes)
    validation_dir = validation_dirs.get(path, path / "validation")
    validation = read_json(validation_dir / "metrics.json", notes)
    state = str(status_file.get("status", status_file.get("state", ""))).lower()
    terminal_failure = state in {"failed", "error", "interrupted", "cancelled", "실행 실패", "중단"}
    has_execution = (path / "history.csv").exists() or (path / "training.log").exists()
    stage = "실행 완료" if summary else "실행 중" if has_execution else "구현 완료" if config else "계획"
    if not summary and state in {"planned", "계획"}:
        stage = "계획"
    elif not summary and state in {"implemented", "구현 완료", "구현완료"}:
        stage = "구현 완료"
    elif not summary and state in {"running", "실행 중", "실행중"}:
        stage = "실행 중"
    if terminal_failure and not summary:
        stage = "구현 완료"
        notes.append("실행 실패/중단: 완료된 성능으로 집계하지 않음")
    elif stage == "실행 중" and not state:
        notes.append("종료 메타데이터 없음: 실제 실행/중단 여부는 이 읽기 전용 집계로 확인하지 않음")
    row = {key: None for key in RUN_COLUMNS}
    row.update(run=path.name, run_dir=str(path), role=role_for(path, config, baselines, legacy),
               status=stage, execution_state=state or ("completed" if summary else "unconfirmed"),
               model=config.get("architecture", config.get("model_name")), seed=config.get("seed"),
               fold=config.get("val_fold"), smoke_test=bool(config.get("smoke_test", False)),
               independent_validation=False, aggregate_checks_passed=False,
               validation_dir=str(validation_dir))
    patients: list[dict] = []
    if stage in FINISHED and not row["smoke_test"]:
        try:
            history = read_csv(path / "history.csv")
            epochs = int(summary["epochs_completed"])
            best_epoch = int(best["epoch"])
            if len(history) != epochs or not 1 <= best_epoch <= epochs:
                raise ValueError("history/summary/best epoch disagree")
            selected = next((r for r in history if int(r["epoch"]) == best_epoch), None)
            if selected is None or not math.isclose(float(selected["valid_balanced_accuracy"]),
                                                  float(best["balanced_accuracy"]), abs_tol=1e-10):
                raise ValueError("history does not match selected best BA")
            if not math.isclose(float(summary["best_valid_balanced_accuracy"]),
                                float(best["balanced_accuracy"]), abs_tol=1e-10):
                raise ValueError("summary disagrees with best BA")
            calculated, patients, fingerprint = validate_saved_predictions(path, config, best)
            elapsed = number(summary.get("elapsed_seconds"))
            if elapsed is None or elapsed < 0:
                raise ValueError("invalid training elapsed_seconds")
            row.update({key: calculated[key] for key in (*METRICS, "samples")})
            row.update(epochs_completed=epochs, best_epoch=best_epoch, elapsed_seconds=elapsed,
                       split_fingerprint=fingerprint, aggregate_checks_passed=True)
            weights = validation.get("weights")
            same_run = bool(weights) and resolve(weights).parent == path and resolve(weights).is_file()
            independent = (validation.get("all_checks_passed") is True and same_run
                           and metrics_match(calculated, validation.get("metrics", {}))
                           and number(validation.get("metrics", {}).get("threshold")) == number(config.get("threshold"))
                           and validation.get("selected_epoch") == best_epoch)
            row["independent_validation"] = independent
            if independent:
                row["status"] = "검증 완료"
            else:
                notes.append("독립 체크포인트 검증 통과 근거 없음: paired 집계 제외")
        except (OSError, ValueError, KeyError, StopIteration, TypeError) as exc:
            notes.append(f"완료 산출물 집계 검증 실패: {exc}")
    elif row["smoke_test"]:
        notes.append("smoke run: 성능·paired 집계 제외")
    row["notes"] = "; ".join(notes)
    return {"row": row, "config": config, "patients": patients}


def difference(left: dict, right: dict) -> dict:
    changes = {}
    for key in sorted(set(left) | set(right)):
        if key in OPERATIONAL_KEYS:
            continue
        lvalue, rvalue = left.get(key), right.get(key)
        if key == "geometric_augmentation":
            lvalue, rvalue = bool(lvalue), bool(rvalue)
        if lvalue != rvalue:
            changes[key] = {"baseline": lvalue, "candidate": rvalue}
    return changes


def delta(right: object, left: object) -> float | None:
    return float(right) - float(left) if right is not None and left is not None else None


def compare(runs: list[dict], change_keys: set[str]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    pairs, patient_pairs, skipped = [], [], []
    baselines = [r for r in runs if r["row"]["role"] == "baseline"]
    for candidate in (r for r in runs if r["row"]["role"] == "candidate"):
        c = candidate["row"]
        if c["status"] != "검증 완료" or not c["aggregate_checks_passed"]:
            skipped.append({"candidate": c["run"], "reason": "candidate not independently verified"})
            continue
        matching = [b for b in baselines if b["row"]["status"] == "검증 완료"
                    and b["row"]["aggregate_checks_passed"]
                    and (b["row"]["model"], b["row"]["seed"], b["row"]["fold"], b["row"]["split_fingerprint"])
                    == (c["model"], c["seed"], c["fold"], c["split_fingerprint"])]
        if len(matching) != 1:
            skipped.append({"candidate": c["run"], "reason": f"expected one matching verified baseline; found {len(matching)}"})
            continue
        baseline = matching[0]
        b = baseline["row"]
        changes = difference(baseline["config"], candidate["config"])
        if set(changes) - change_keys:
            skipped.append({"candidate": c["run"], "reason": f"uncontrolled config differences: {sorted(set(changes) - change_keys)}"})
            continue
        label = "; ".join(f"{k}={v['baseline']}→{v['candidate']}" for k, v in changes.items()) or "identical_config_repeat"
        family_config = {key: value for key, value in baseline["config"].items()
                         if key not in OPERATIONAL_KEYS | {"seed", "val_fold"}}
        family_config.setdefault("geometric_augmentation", False)
        family = digest({"baseline_config": family_config, "changes": changes})[:16]
        pair = dict(baseline=b["run"], candidate=c["run"], seed=c["seed"], fold=c["fold"],
                    comparison=label, comparison_family=family,
                    changed_config=json.dumps(changes, sort_keys=True, ensure_ascii=False),
                    baseline_best_epoch=b["best_epoch"], candidate_best_epoch=c["best_epoch"],
                    baseline_elapsed_seconds=b["elapsed_seconds"], candidate_elapsed_seconds=c["elapsed_seconds"],
                    delta_elapsed_seconds=delta(c["elapsed_seconds"], b["elapsed_seconds"]),
                    time_ratio=c["elapsed_seconds"] / b["elapsed_seconds"] if b["elapsed_seconds"] else None)
        for metric in METRICS:
            pair.update({f"baseline_{metric}": b[metric], f"candidate_{metric}": c[metric],
                         f"delta_{metric}": delta(c[metric], b[metric])})
        base_patients = {p["patient_group"]: p for p in baseline["patients"]}
        changes_ba = []
        for cp in candidate["patients"]:
            bp = base_patients[cp["patient_group"]]
            pp = {key: pair[key] for key in ("baseline", "candidate", "seed", "fold", "comparison")}
            pp.update({key: cp[key] for key in ("patient_group", "samples", "negative_samples", "positive_samples", "ba_interpretation")})
            for metric in METRICS:
                pp.update({f"baseline_{metric}": bp[metric], f"candidate_{metric}": cp[metric],
                           f"delta_{metric}": delta(cp[metric], bp[metric])})
            if pp["delta_balanced_accuracy"] is not None:
                changes_ba.append(pp["delta_balanced_accuracy"])
            patient_pairs.append(pp)
        pair.update(patients_worse_ba=sum(d < 0 for d in changes_ba),
                    patients_better_ba=sum(d > 0 for d in changes_ba),
                    max_patient_ba_drop=min([0.0, *changes_ba]) if changes_ba else None)
        pairs.append(pair)
    aggregates = []
    for family in sorted({p["comparison_family"] for p in pairs}):
        group = [p for p in pairs if p["comparison_family"] == family]
        aggregate = dict(comparison=group[0]["comparison"], comparison_family=family,
                         matched_pairs=len(group), matched_seed_folds=";".join(f"s{p['seed']}/f{p['fold']}" for p in group))
        for metric in ("balanced_accuracy", "sensitivity", "specificity", "elapsed_seconds"):
            for prefix in ("baseline", "candidate", "delta"):
                values = [p[f"{prefix}_{metric}"] for p in group]
                aggregate[f"mean_{prefix}_{metric}"] = statistics.mean(values)
            aggregate[f"stdev_delta_{metric}"] = statistics.stdev([p[f"delta_{metric}"] for p in group]) if len(group) > 1 else None
        aggregates.append(aggregate)
    return pairs, patient_pairs, aggregates, skipped


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "해당 없음.\n"
    def cell(value: object) -> str:
        return (f"{value:.6f}" if isinstance(value, float) else "" if value is None else str(value)).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |",
                       *("| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows)]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", action="extend", required=True)
    parser.add_argument("--baseline-runs", type=Path, nargs="+", action="extend", default=[])
    parser.add_argument("--legacy-runs", type=Path, nargs="+", action="extend", default=[])
    parser.add_argument("--validation-dir", action="append", default=[], metavar="RUN=DIR",
                        help="override independent validation directory for one run (repeatable)")
    parser.add_argument("--change-key", action="append", default=[], help="additional permitted config difference")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; existing paths rejected")
    args = parser.parse_args()
    paths = list(dict.fromkeys(resolve(p) for p in args.runs))
    baselines, legacy = {resolve(p) for p in args.baseline_runs}, {resolve(p) for p in args.legacy_runs}
    if baselines & legacy:
        parser.error("one run cannot be both baseline and legacy")
    if (baselines | legacy) - set(paths):
        parser.error("--baseline-runs and --legacy-runs must also appear in --runs")
    overrides = {}
    for value in args.validation_dir:
        if "=" not in value:
            parser.error("--validation-dir requires RUN=DIR")
        run_value, directory = value.split("=", 1)
        run_path = resolve(run_value)
        if run_path not in paths:
            parser.error("validation override run must also appear in --runs")
        overrides[run_path] = resolve(directory)
    output = resolve(args.output_dir)
    if output.exists():
        parser.error(f"choose a new --output-dir; already exists: {output}")
    if any(output == path or path in output.parents for path in paths):
        parser.error("output-dir must be outside input run directories to preserve them")
    runs = [inspect_run(path, baselines, legacy, overrides) for path in paths]
    pairs, patient_pairs, aggregates, skipped = compare(runs, DEFAULT_CHANGE_KEYS | set(args.change_key))
    rows = [r["row"] for r in runs]
    patients = [{**{k: r["row"][k] for k in ("run", "role", "seed", "fold")}, **p}
                for r in runs for p in r["patients"]]
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "runs.csv", rows, RUN_COLUMNS)
    write_csv(output / "legacy_runs.csv", [r for r in rows if r["role"] == "legacy"], RUN_COLUMNS)
    write_csv(output / "per_patient.csv", patients, PATIENT_COLUMNS)
    write_csv(output / "paired_comparisons.csv", pairs, PAIR_COLUMNS)
    write_csv(output / "paired_patients.csv", patient_pairs, PATIENT_PAIR_COLUMNS)
    write_csv(output / "paired_summary.csv", aggregates, AGGREGATE_COLUMNS)
    write_csv(output / "skipped_pairs.csv", skipped, ["candidate", "reason"])
    metadata = dict(generated_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                    command=[sys.executable, *sys.argv], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    run_inputs=[str(p) for p in paths], change_keys=sorted(DEFAULT_CHANGE_KEYS | set(args.change_key)),
                    counts={"runs": len(rows), "verified_pairs": len(pairs), "legacy": sum(r["role"] == "legacy" for r in rows)})
    (output / "comparison_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sections = ["# Task2 실험 비교\n", f"생성: {metadata['generated_at']}\n",
                "저장 산출물만 읽어 집계했습니다. 모델 추론·재학습은 실행하지 않았습니다. 실행 중·미완료 run은 상태만 표시하며 성능 집계에 포함하지 않습니다.\n",
                "## 상태와 결과\n", table(rows, ["run", "role", "status", "seed", "fold", "best_epoch", "elapsed_seconds", "balanced_accuracy", "sensitivity", "specificity", "notes"]),
                "## 검증된 동일 seed/fold 비교\n",
                "같은 환자 분할 전체 manifest와 seed/fold를 가진 baseline/candidate만 연결했습니다. 허용된 핵심 변경 외 config가 다르면 제외했습니다. legacy는 연결하지 않습니다.\n",
                table(pairs, ["baseline", "candidate", "seed", "fold", "comparison", "delta_balanced_accuracy", "delta_sensitivity", "delta_specificity", "time_ratio", "patients_worse_ba", "max_patient_ba_drop"]),
                "## paired 평균과 변동\n",
                "평균은 이미 연결된 동일 seed/fold 쌍의 차이만 집계합니다. 여러 seed/fold 결과의 표준편차는 기술 통계이며 독립 환자 표본의 신뢰구간이 아닙니다. 쌍이 하나면 표준편차를 비워 두고 안정성을 판단하지 않습니다.\n",
                table(aggregates, ["comparison", "matched_pairs", "matched_seed_folds", "mean_baseline_balanced_accuracy", "mean_candidate_balanced_accuracy", "mean_delta_balanced_accuracy", "stdev_delta_balanced_accuracy", "mean_delta_sensitivity", "mean_delta_specificity", "mean_delta_elapsed_seconds"]),
                "## Legacy 참고 결과\n", table([r for r in rows if r["role"] == "legacy"], ["run", "status", "balanced_accuracy", "sensitivity", "specificity", "best_epoch", "elapsed_seconds"]),
                "## 제외된 비교\n", table(skipped, ["candidate", "reason"]),
                "환자별 변화와 혼동행렬은 paired_patients.csv, 전체 환자별 결과는 per_patient.csv, run/paired 혼동행렬은 runs.csv와 paired_comparisons.csv에 있습니다. 한 클래스만 존재하는 환자는 BA를 비워 두고 single_class_BA_undefined로 표시합니다. 단일 검증 상승만으로 채택을 결론내리지 않습니다.\n"]
    (output / "COMPARISON.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"runs={len(rows)} verified_pairs={len(pairs)} output={output}")


if __name__ == "__main__":
    main()
