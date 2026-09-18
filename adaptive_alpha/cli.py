"""Command-line interface for the research framework."""

from __future__ import annotations

import argparse

import pandas as pd

from adaptive_alpha.experiment_logic import (
    AdaptiveFailurePolicy,
    CheapestFirstPolicy,
    FixedChecklistPolicy,
    OraclePolicy,
    RandomOrderPolicy,
    StoppingRule,
    run_research_pipeline,
    simulate_budget,
    train_failure_models,
)
from adaptive_alpha.results_config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Budgeted adaptive falsification for equity alpha discovery.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-benchmark", help="Build benchmark and run policies.")
    build.add_argument("--config", default="configs/demo.json")
    build.add_argument("--prices", default=None, help="Optional local price CSV; otherwise use synthetic data.")
    build.add_argument("--output", default="artifacts/demo")

    sim = sub.add_parser("simulate", help="Simulate a few policies on an existing benchmark CSV.")
    sim.add_argument("--benchmark", required=True)
    sim.add_argument("--budget", type=float, default=1000.0)

    args = parser.parse_args(argv)
    if args.command == "build-benchmark":
        config = load_config(args.config)
        result = run_research_pipeline(config, output=args.output, prices=args.prices)
        print(f"benchmark={result['benchmark_path']}")
        print(f"policy_results={result['summary_path']}")
        print(f"report={result['report_path']}")
        return 0

    if args.command == "simulate":
        benchmark = pd.read_csv(args.benchmark)
        order = benchmark.groupby("test_name")[["tier", "cost"]].first().sort_values(["tier", "cost"]).index.astype(str).tolist()
        train = benchmark[benchmark["episode"] != sorted(benchmark["episode"].unique())[-1]]
        sealed = benchmark[benchmark["episode"] == sorted(benchmark["episode"].unique())[-1]]
        correct, false = train_failure_models(train)
        candidate_records = sealed.groupby("candidate_name").first().to_dict(orient="index")
        policies = [
            RandomOrderPolicy(seed=3),
            FixedChecklistPolicy(order=order),
            CheapestFirstPolicy(),
            AdaptiveFailurePolicy(correct, false, candidate_records={str(k): v for k, v in candidate_records.items()}),
            OraclePolicy(),
        ]
        stopping = StoppingRule()
        rows = []
        for policy in policies:
            rows.append(simulate_budget(sealed, policy, args.budget, stopping).summary)
        print(pd.DataFrame(rows).to_string(index=False))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
