from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from adaptive_alpha.experiment_logic import run_research_pipeline


class PipelineTests(unittest.TestCase):
    def test_tiny_pipeline_runs_end_to_end(self) -> None:
        config = {
            "seed": 5,
            "data": {"mode": "synthetic", "n_assets": 28, "n_days": 230, "start": "2019-01-01"},
            "factors": {"n_candidates": 5, "max_depth": 4, "max_window": 20},
            "labels": {
                "horizon": 5,
                "execution_delay": 1,
                "future_min_rank_ic": 0.0,
                "future_min_t_stat": -10.0,
            },
            "episodes": {
                "search_days": 60,
                "falsification_days": 40,
                "future_days": 30,
                "step_days": 80,
                "sealed_episodes": 1,
            },
            "falsification": {"placebo_repetitions": 2, "synthetic_repetitions": 2},
            "budget": {"total": 80.0, "reject_threshold": 0.72, "survive_threshold": 0.28, "minimum_tests": 2},
            "policy": {"gamma": 0.8, "eta": 0.15},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config["report"] = {"path": str(Path(tmp) / "tiny_report.md")}
            result = run_research_pipeline(config, output=Path(tmp) / "tiny")
            benchmark = pd.read_csv(result["benchmark_path"])
            summary = pd.read_csv(result["summary_path"])

        self.assertFalse(benchmark.empty)
        self.assertFalse(summary.empty)
        self.assertIn("adaptive_policy", set(summary["policy"]))
        self.assertIn("oracle", set(summary["policy"]))


if __name__ == "__main__":
    unittest.main()
