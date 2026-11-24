"""Synthetic CPU-only tests. No assignment images, checkpoints, training or GPU."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import threshold_analysis as analysis


def row(group, label, probability):
    return {"patient_group": group, "label": label, "probability": probability}


class ThresholdProtocolTests(unittest.TestCase):
    def test_fixed_grid_is_independent_and_contains_05(self):
        self.assertEqual(analysis.THRESHOLD_PERCENT_GRID, tuple(range(5, 96)))
        self.assertIn(50, analysis.THRESHOLD_PERCENT_GRID)

    def test_flat_plateau_prefers_05(self):
        threshold, metrics = analysis.select_threshold([row("A", 0, .2), row("A", 1, .2)])
        self.assertEqual(threshold, .5)
        self.assertEqual(metrics["balanced_accuracy"], .5)

    def test_equal_distance_tie_chooses_lower_threshold(self):
        records = [row("A", 0, .485), row("A", 1, .495), row("B", 0, .505), row("B", 1, .515)]
        threshold, metrics = analysis.select_threshold(records)
        self.assertEqual(threshold, .49)
        self.assertEqual(metrics["balanced_accuracy"], .75)

    def test_heldout_labels_do_not_affect_its_threshold(self):
        records = [row("A", 0, .2), row("A", 1, .9), row("B", 0, .485), row("B", 1, .495),
                   row("C", 0, .505), row("C", 1, .515)]
        changed = copy.deepcopy(records)
        for record in changed:
            if record["patient_group"] == "A":
                record["label"] = 1 - record["label"]
        before = next(fold for fold in analysis.crossfit_lopo(records)["folds"] if fold["held_out_patient"] == "A")
        after = next(fold for fold in analysis.crossfit_lopo(changed)["folds"] if fold["held_out_patient"] == "A")
        self.assertEqual(before["selected_threshold"], after["selected_threshold"])
        self.assertEqual(before["calibration_selected_metrics"], after["calibration_selected_metrics"])

    def test_heldout_probabilities_do_not_affect_its_threshold(self):
        records = [row("A", 0, .2), row("A", 1, .9), row("B", 0, .485), row("B", 1, .495),
                   row("C", 0, .505), row("C", 1, .515)]
        changed = copy.deepcopy(records)
        changed[0]["probability"], changed[1]["probability"] = 1.0, 0.0
        before = next(fold for fold in analysis.crossfit_lopo(records)["folds"] if fold["held_out_patient"] == "A")
        after = next(fold for fold in analysis.crossfit_lopo(changed)["folds"] if fold["held_out_patient"] == "A")
        self.assertEqual(before["selected_threshold"], after["selected_threshold"])
        self.assertEqual(before["calibration_selected_metrics"], after["calibration_selected_metrics"])

    def test_combined_metric_is_image_pooled_not_patient_mean(self):
        records = [row("A", 0, .1)] * 9 + [row("A", 1, .9)]
        records += [row("B", 0, .9), row("B", 1, .9)]
        fixed = analysis.at_threshold(records, .5)
        self.assertEqual(fixed["balanced_accuracy"], .95)
        a = analysis.at_threshold([r for r in records if r["patient_group"] == "A"], .5)
        b = analysis.at_threshold([r for r in records if r["patient_group"] == "B"], .5)
        self.assertNotEqual(fixed["balanced_accuracy"], (a["balanced_accuracy"] + b["balanced_accuracy"]) / 2)

    def test_nonfinite_and_out_of_range_are_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf"), -.01, 1.01):
            with self.subTest(probability=bad), self.assertRaises(ValueError):
                analysis.select_threshold([row("A", 0, bad), row("B", 1, .7)])

    def test_invalid_and_empty_rows_are_rejected(self):
        for records in ([], [row("", 0, .2), row("B", 1, .8)], [row("A", 2, .1), row("B", 1, .8)]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                analysis.select_threshold(records)

    def test_degenerate_calibration_does_not_use_heldout_fallback(self):
        with self.assertRaises(ValueError):
            analysis.select_threshold([row("A", 0, .1)])
        with self.assertRaises(ValueError):
            analysis.crossfit_lopo([row("A", 0, .1), row("A", 1, .8)])
        with self.assertRaises(ValueError):
            analysis.crossfit_lopo([row("A", 0, .1), row("B", 1, .8)])

    def test_single_class_heldout_ba_is_undefined(self):
        records = [row("A", 0, .1), row("B", 0, .1), row("B", 1, .8), row("C", 0, .1), row("C", 1, .8)]
        result = analysis.crossfit_lopo(records)
        held = next(fold for fold in result["folds"] if fold["held_out_patient"] == "A")
        self.assertIsNone(held["held_out_crossfitted"]["balanced_accuracy"])
        self.assertEqual(held["held_out_crossfitted"]["specificity"], 1.0)
        self.assertEqual(result["crossfitted"]["balanced_accuracy"], 1.0)
        self.assertIsNone(result["deployment_threshold"])

    def test_checkpoint_digest_mismatch_is_rejected_without_loading(self):
        with tempfile.TemporaryDirectory(prefix="threshold_synthetic_") as temporary:
            task = Path(temporary)
            run_id = "synthetic_fixture"
            weights = task / "runs" / run_id / "resnet.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"synthetic hash-only fixture; not a neural checkpoint")
            audit = task / "research_20261125/historical_validation" / run_id
            audit.mkdir(parents=True)
            (audit / "predictions.csv").write_text("synthetic fixture\n")
            (audit / "metrics.json").write_text(json.dumps({
                "all_checks_passed": True, "smoke_test": False, "model_name": "resnet", "architecture": "resnet18",
                "checkpoint_sha256": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                analysis.load_bound_predictions(task, run_id)


def main():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ThresholdProtocolTests)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), end="")
    record = {
        "status": "검증 완료" if result.wasSuccessful() else "실패",
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "scope": "synthetic CPU-only rows and a hash-only dummy file; no assignment data, model loading, GPU, or training",
        "checks": {name.removeprefix("test_"): result.wasSuccessful()
                   for name in unittest.defaultTestLoader.getTestCaseNames(ThresholdProtocolTests)},
    }
    (Path(__file__).resolve().parent / "THRESHOLD_TESTS.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
