"""Run the shared LOPO algorithm on a locally held prediction CSV.

This adapter is separate from the byte-identical original analysis. It does not
verify checkpoint/split provenance; the caller must supply audited predictions.
It never trains a model or selects a deployment threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from threshold_analysis import crossfit_lopo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path,
                        help="private CSV with patient_group,label,probability")
    parser.add_argument("--output", required=True, type=Path,
                        help="new local JSON; contains patient-level diagnostics")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    with args.predictions.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"patient_group", "label", "probability"}
        if not required.issubset(reader.fieldnames or []):
            parser.error("Input CSV requires patient_group,label,probability")
        rows = [{"patient_group": row["patient_group"], "label": int(row["label"]),
                 "probability": float(row["probability"])} for row in reader]
    result = crossfit_lopo(rows)
    result["limitations"] = [
        "The adapter does not validate model/split/checkpoint provenance.",
        "If validation selected checkpoint epochs, this diagnosis remains conditional on that selection.",
        "No deployment threshold is selected; official permission remains unconfirmed.",
        "The output contains patient-level records and should stay local.",
    ]
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print("Conditional LOPO diagnosis saved locally; deployment threshold remains unset.")


if __name__ == "__main__":
    main()
