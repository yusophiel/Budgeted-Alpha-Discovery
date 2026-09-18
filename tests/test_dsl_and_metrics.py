from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from adaptive_alpha.data_handling import make_synthetic_panel
from adaptive_alpha.factor_generation import call, evaluate, field, validate_expression, validate_signal
from adaptive_alpha.testing_evaluation import bh_fdr, build_shadow_snapshot, factor_metrics, rank_ic_series


class DslAndMetricsTests(unittest.TestCase):
    def test_expression_hash_and_lineage_are_stable(self) -> None:
        expr = call("rank", call("delta", field("close"), window=5))
        same = call("rank", call("delta", field("close"), window=5))
        nearby = call("rank", call("delta", field("close"), window=7))

        self.assertEqual(expr.hash(), same.hash())
        self.assertNotEqual(expr.hash(), nearby.hash())
        self.assertEqual(expr.lineage_hash(), nearby.lineage_hash())
        self.assertTrue(validate_expression(expr, max_window=10).ok)

    def test_safe_divide_produces_finite_or_nan_signal(self) -> None:
        panel = make_synthetic_panel(n_assets=12, n_days=80, seed=3)
        expr = call("safe_divide", call("delta", field("close"), window=5), call("ts_std", field("return"), window=10))
        signal = evaluate(expr, panel, {})

        values = signal.to_numpy(dtype=float)
        self.assertFalse(np.isinf(values).any())
        self.assertTrue(validate_signal(signal, min_coverage=0.2).ok)

    def test_rank_ic_detects_aligned_signal(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=30)
        cols = list("ABCDE")
        base = pd.DataFrame(np.tile(np.arange(5), (30, 1)), index=dates, columns=cols)
        forward = base + np.random.default_rng(1).normal(0, 0.01, size=base.shape)

        ic = rank_ic_series(base, forward, min_assets=5)
        metrics = factor_metrics(base, forward, min_assets=5)

        self.assertGreater(float(ic.mean()), 0.95)
        self.assertGreater(metrics.mean_rank_ic, 0.95)
        self.assertGreater(metrics.quantile_spread, 0.0)

    def test_bh_fdr_orders_q_values(self) -> None:
        adjusted = bh_fdr({"a": 0.001, "b": 0.02, "c": 0.5}, alpha=0.05)
        self.assertTrue(bool(adjusted.loc["a", "reject"]))
        self.assertLessEqual(float(adjusted.loc["a", "q_value"]), float(adjusted.loc["b", "q_value"]))

    def test_shadow_snapshot_has_balanced_targets(self) -> None:
        panel = make_synthetic_panel(n_assets=20, n_days=40, seed=9)
        signal = panel.field("return").rolling(5).mean()
        snapshot = build_shadow_snapshot(signal, panel.field("close"))

        self.assertIn("target_weight", snapshot.columns)
        self.assertAlmostEqual(float(snapshot["target_weight"].sum()), 0.0, places=8)
        self.assertLessEqual(float(snapshot["target_weight"].abs().sum()), 1.0 + 1e-8)


if __name__ == "__main__":
    unittest.main()
