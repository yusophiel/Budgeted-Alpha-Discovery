"""Consolidated module for the adaptive alpha project."""

from __future__ import annotations

from adaptive_alpha.data_handling import PanelData, load_price_volume_csv, make_synthetic_panel
from adaptive_alpha.factor_generation import CandidateFactory, FactorCandidate, call, evaluate
from adaptive_alpha.testing_evaluation import EvaluationCache, FalsificationContext, TestResult, default_test_registry, factor_metrics, tier0_checks
from adaptive_alpha.results_config import ensure_dir, write_json, write_markdown_report

from dataclasses import dataclass, field



@dataclass
class CandidateState:
    candidate_name: str
    family: str
    future_valid: bool
    initial_score: float
    posterior_failure: float = 0.5
    status: str = "active"
    tests_run: set[str] = field(default_factory=set)
    history: list[TestResult] = field(default_factory=list)
    cost_spent: float = 0.0

    @property
    def active(self) -> bool:
        return self.status == "active"

    def record(self, result: TestResult) -> None:
        self.tests_run.add(result.test_name)
        self.history.append(result)
        self.cost_spent += result.cost

    def history_features(self) -> dict[str, float | int]:
        n = len(self.history)
        rejects = sum(r.rejected for r in self.history)
        warns = sum(r.status == "warn" for r in self.history)
        evidence = sum(r.evidence_score for r in self.history)
        return {
            "history_n_tests": n,
            "history_n_rejects": rejects,
            "history_n_warns": warns,
            "history_evidence_sum": evidence,
            "history_cost_spent": self.cost_spent,
            "posterior_failure": self.posterior_failure,
        }

import math



class StoppingRule:
    def __init__(
        self,
        reject_threshold: float = 0.72,
        survive_threshold: float = 0.28,
        minimum_tests: int = 2,
    ) -> None:
        self.reject_threshold = reject_threshold
        self.survive_threshold = survive_threshold
        self.minimum_tests = minimum_tests

    def update(self, state: CandidateState) -> None:
        state.posterior_failure = self.posterior_failure(state)
        if not state.active:
            return
        if len(state.history) < self.minimum_tests:
            return
        if state.posterior_failure >= self.reject_threshold:
            state.status = "rejected"
        elif state.posterior_failure <= self.survive_threshold:
            state.status = "survived"

    def posterior_failure(self, state: CandidateState) -> float:
        initial_strength = abs(state.initial_score)
        log_odds = -0.35 * min(3.0, initial_strength * 100.0)
        for result in state.history:
            if result.rejected:
                log_odds += 1.7 * result.evidence_score + 0.25 * result.tier
            elif result.status == "warn":
                log_odds += 0.65 * result.evidence_score
            else:
                log_odds -= 0.38 + 0.12 * min(result.tier, 4)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, log_odds))))

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np



TestSpec = dict[str, float | int | str]
ResultLookup = dict[tuple[str, str], TestResult]


class SelectionPolicy(Protocol):
    name: str

    def choose(
        self,
        states: dict[str, CandidateState],
        tests: dict[str, TestSpec],
        results: ResultLookup,
        remaining_budget: float,
    ) -> tuple[str, str] | None:
        ...


def available_actions(
    states: dict[str, CandidateState],
    tests: dict[str, TestSpec],
    remaining_budget: float,
) -> list[tuple[str, str]]:
    actions: list[tuple[str, str]] = []
    for candidate_name, state in states.items():
        if not state.active:
            continue
        for test_name, spec in tests.items():
            if test_name in state.tests_run:
                continue
            if float(spec["cost"]) <= remaining_budget:
                actions.append((candidate_name, test_name))
    return actions


@dataclass
class NoFalsificationPolicy:
    name: str = "no_falsification"

    def choose(self, states: dict[str, CandidateState], tests: dict[str, TestSpec], results: ResultLookup, remaining_budget: float) -> None:
        return None


@dataclass
class FixedChecklistPolicy:
    order: list[str]
    name: str = "fixed_checklist"

    def choose(
        self,
        states: dict[str, CandidateState],
        tests: dict[str, TestSpec],
        results: ResultLookup,
        remaining_budget: float,
    ) -> tuple[str, str] | None:
        for candidate_name, state in states.items():
            if not state.active:
                continue
            for test_name in self.order:
                if test_name not in tests or test_name in state.tests_run:
                    continue
                if float(tests[test_name]["cost"]) <= remaining_budget:
                    return candidate_name, test_name
        return None


@dataclass
class FullBatteryPolicy(FixedChecklistPolicy):
    name: str = "full_battery"


@dataclass
class CheapestFirstPolicy:
    name: str = "cheapest_first"

    def choose(
        self,
        states: dict[str, CandidateState],
        tests: dict[str, TestSpec],
        results: ResultLookup,
        remaining_budget: float,
    ) -> tuple[str, str] | None:
        actions = available_actions(states, tests, remaining_budget)
        if not actions:
            return None
        return min(actions, key=lambda action: (float(tests[action[1]]["cost"]), action[0], action[1]))


@dataclass
class RandomOrderPolicy:
    seed: int = 1
    name: str = "random_order"
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def choose(
        self,
        states: dict[str, CandidateState],
        tests: dict[str, TestSpec],
        results: ResultLookup,
        remaining_budget: float,
    ) -> tuple[str, str] | None:
        actions = available_actions(states, tests, remaining_budget)
        if not actions:
            return None
        return actions[int(self.rng.integers(0, len(actions)))]


@dataclass
class GlobalRejectionPerCostPolicy:
    rejection_rate_by_test: dict[str, float]
    name: str = "global_rejection_per_cost"

    def choose(
        self,
        states: dict[str, CandidateState],
        tests: dict[str, TestSpec],
        results: ResultLookup,
        remaining_budget: float,
    ) -> tuple[str, str] | None:
        actions = available_actions(states, tests, remaining_budget)
        if not actions:
            return None
        return max(
            actions,
            key=lambda action: (
                self.rejection_rate_by_test.get(action[1], 0.0) / max(float(tests[action[1]]["cost"]), 1e-9),
                -states[action[0]].cost_spent,
            ),
        )


@dataclass
class RuleBasedPolicy:
    family_orders: dict[str, list[str]]
    default_order: list[str]
    name: str = "rule_based"
    stopping_rule: Any | None = None

    def choose(
        self,
        states: dict[str, CandidateState],
        tests: dict[str, TestSpec],
        results: ResultLookup,
        remaining_budget: float,
    ) -> tuple[str, str] | None:
        for candidate_name, state in states.items():
            if not state.active:
                continue
            order = self.family_orders.get(state.family, self.default_order)
            for test_name in order:
                if test_name in tests and test_name not in state.tests_run and float(tests[test_name]["cost"]) <= remaining_budget:
                    return candidate_name, test_name
        return None


@dataclass
class OraclePolicy:
    name: str = "oracle"

    def choose(
        self,
        states: dict[str, CandidateState],
        tests: dict[str, TestSpec],
        results: ResultLookup,
        remaining_budget: float,
    ) -> tuple[str, str] | None:
        actions = available_actions(states, tests, remaining_budget)
        if not actions:
            return None
        useful = []
        for candidate_name, test_name in actions:
            state = states[candidate_name]
            result = results[(candidate_name, test_name)]
            if (not state.future_valid) and result.rejected:
                useful.append((candidate_name, test_name))
        if not useful:
            return None
        return max(useful, key=lambda action: (results[action].evidence_score / max(float(tests[action[1]]["cost"]), 1e-9), -float(tests[action[1]]["cost"])))

import numpy as np


class LinUCB:
    def __init__(self, n_features: int, alpha: float = 1.0) -> None:
        self.n_features = n_features
        self.alpha = alpha
        self.a: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}

    def add_action(self, action: str) -> None:
        if action not in self.a:
            self.a[action] = np.eye(self.n_features)
            self.b[action] = np.zeros(self.n_features)

    def score(self, action: str, context: np.ndarray) -> float:
        self.add_action(action)
        inv = np.linalg.inv(self.a[action])
        theta = inv @ self.b[action]
        exploit = float(theta @ context)
        explore = float(self.alpha * np.sqrt(context @ inv @ context))
        return exploit + explore

    def choose(self, actions: list[str], context: np.ndarray) -> str | None:
        if not actions:
            return None
        return max(actions, key=lambda action: self.score(action, context))

    def update(self, action: str, context: np.ndarray, reward: float) -> None:
        self.add_action(action)
        self.a[action] += np.outer(context, context)
        self.b[action] += reward * context

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd



class FeatureEncoder:
    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.feature_index: dict[str, int] = {}

    def fit_transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        names = sorted({name for row in rows for name in _expand_features(row)})
        self.feature_names = names
        self.feature_index = {name: i for i, name in enumerate(names)}
        return self.transform(rows)

    def transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        expanded = [_expand_features(row) for row in rows]
        matrix = np.zeros((len(rows), len(self.feature_names)), dtype=float)
        for i, row in enumerate(expanded):
            for name, value in row.items():
                j = self.feature_index.get(name)
                if j is not None:
                    matrix[i, j] = value
        return matrix


class NumpyLogisticRegression:
    def __init__(self, lr: float = 0.08, l2: float = 1e-3, max_iter: int = 800) -> None:
        self.lr = lr
        self.l2 = l2
        self.max_iter = max_iter
        self.weights: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.constant: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        y = y.astype(float)
        if len(y) == 0:
            self.constant = 0.0
            return self
        if np.all(y == y[0]):
            self.constant = float(y[0])
            return self
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0
        z = (x - self.mean) / self.std
        z = np.column_stack([np.ones(len(z)), z])
        weights = np.zeros(z.shape[1], dtype=float)
        for _ in range(self.max_iter):
            pred = 1.0 / (1.0 + np.exp(-np.clip(z @ weights, -30, 30)))
            grad = z.T @ (pred - y) / len(y)
            grad[1:] += self.l2 * weights[1:]
            weights -= self.lr * grad
        self.weights = weights
        self.constant = None
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(x.shape[0], self.constant, dtype=float)
        if self.weights is None or self.mean is None or self.std is None:
            return np.full(x.shape[0], 0.5, dtype=float)
        z = (x - self.mean) / self.std
        z = np.column_stack([np.ones(len(z)), z])
        return 1.0 / (1.0 + np.exp(-np.clip(z @ self.weights, -30, 30)))


@dataclass
class FailureModel:
    encoder: FeatureEncoder
    model: NumpyLogisticRegression

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return self.model.predict_proba(self.encoder.transform(rows))


@dataclass
class AdaptiveFailurePolicy:
    correct_rejection_model: FailureModel
    false_elimination_model: FailureModel
    candidate_records: dict[str, dict[str, Any]]
    gamma: float = 0.8
    eta: float = 0.15
    max_false_risk: float | None = None
    min_correct_advantage: float | None = None
    name: str = "adaptive_policy"
    _prediction_cache: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict, init=False, repr=False)

    def choose(
        self,
        states: dict[str, CandidateState],
        tests: dict[str, TestSpec],
        results: ResultLookup,
        remaining_budget: float,
    ) -> tuple[str, str] | None:
        actions = available_actions(states, tests, remaining_budget)
        if not actions:
            return None
        p_correct_values: list[float] = []
        p_false_values: list[float] = []
        for candidate_name, test_name in actions:
            p_correct, p_false = self._predict_action(candidate_name, test_name, tests)
            p_correct_values.append(p_correct)
            p_false_values.append(p_false)
        p_correct = np.array(p_correct_values, dtype=float)
        p_false = np.array(p_false_values, dtype=float)
        uncertainty = np.array([states[c].posterior_failure * (1.0 - states[c].posterior_failure) for c, _ in actions])
        costs = np.array([max(float(tests[t]["cost"]), 1e-9) for _, t in actions])
        scores = (p_correct - self.gamma * p_false + self.eta * uncertainty) / costs
        if self.max_false_risk is not None:
            risk_mask = p_false <= self.max_false_risk
            if risk_mask.any():
                scores = np.where(risk_mask, scores, -np.inf)
        if self.min_correct_advantage is not None:
            advantage_mask = (p_correct - p_false) >= self.min_correct_advantage
            if advantage_mask.any():
                scores = np.where(advantage_mask, scores, -np.inf)
        best = int(np.argmax(scores))
        return actions[best]

    def _predict_action(
        self,
        candidate_name: str,
        test_name: str,
        tests: dict[str, TestSpec],
    ) -> tuple[float, float]:
        key = (candidate_name, test_name)
        cached = self._prediction_cache.get(key)
        if cached is not None:
            return cached
        row = build_feature_row(
            self.candidate_records[candidate_name],
            test_name=test_name,
            test_cost=float(tests[test_name]["cost"]),
            test_tier=int(tests[test_name]["tier"]),
            state=None,
        )
        p_correct = float(self.correct_rejection_model.predict([row])[0])
        p_false = float(self.false_elimination_model.predict([row])[0])
        self._prediction_cache[key] = (p_correct, p_false)
        return p_correct, p_false


def train_failure_models(training_rows: pd.DataFrame) -> tuple[FailureModel, FailureModel]:
    if training_rows.empty:
        empty_encoder = FeatureEncoder()
        empty_encoder.feature_names = []
        empty_model = NumpyLogisticRegression()
        empty_model.constant = 0.0
        return FailureModel(empty_encoder, empty_model), FailureModel(empty_encoder, empty_model)

    feature_rows = [row.to_dict() for _, row in training_rows.iterrows()]
    correct_target = training_rows["target_correct_rejection"].to_numpy(dtype=float)
    false_target = training_rows["target_false_elimination"].to_numpy(dtype=float)

    encoder = FeatureEncoder()
    x = encoder.fit_transform(feature_rows)
    correct_model = NumpyLogisticRegression().fit(x, correct_target)
    false_model = NumpyLogisticRegression().fit(x, false_target)
    return FailureModel(encoder, correct_model), FailureModel(encoder, false_model)


def build_feature_row(
    candidate_record: dict[str, Any],
    test_name: str,
    test_cost: float,
    test_tier: int | None = None,
    state: CandidateState | None = None,
) -> dict[str, Any]:
    row = dict(candidate_record)
    row["test_name"] = test_name
    row["test_cost"] = test_cost
    row["cost"] = test_cost
    if test_tier is not None:
        row["tier"] = test_tier
    if state is not None:
        row.update(state.history_features())
    else:
        row.update(
            {
                "history_n_tests": 0,
                "history_n_rejects": 0,
                "history_n_warns": 0,
                "history_evidence_sum": 0.0,
                "history_cost_spent": 0.0,
                "posterior_failure": 0.5,
            }
        )
    return row


def _expand_features(row: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    skip = {
        "candidate_name",
        "name",
        "factor_hash",
        "lineage_hash",
        "expression",
        "notes",
        "status",
        "rejected",
        "evidence_score",
        "future_valid",
        "future_failed",
        "search_start",
        "search_end",
        "falsification_start",
        "falsification_end",
        "future_start",
        "future_end",
    }
    allowed_exact = {
        "family",
        "generated_by",
        "test_name",
        "test_cost",
        "tier",
        "cost",
        "ast_depth",
        "node_count",
        "max_window",
        "n_ts_ops",
        "has_division",
        "has_corr",
        "has_neutralize",
        "has_rank",
        "oriented",
        "posterior_failure",
    }
    allowed_prefixes = ("initial_", "history_")
    for key, value in row.items():
        if key in skip or key.startswith("target_") or key.startswith("future_"):
            continue
        if key not in allowed_exact and not key.startswith(allowed_prefixes):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
            features[key] = float(value)
        elif isinstance(value, str):
            features[f"{key}={value}"] = 1.0
        elif isinstance(value, bool):
            features[key] = float(value)
    return features

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd



@dataclass(frozen=True)
class BudgetSimulationResult:
    policy_name: str
    summary: dict[str, float | int | str]
    decisions: pd.DataFrame
    history: pd.DataFrame


def simulate_budget(
    benchmark: pd.DataFrame,
    policy: SelectionPolicy,
    total_budget: float,
    stopping_rule: StoppingRule,
) -> BudgetSimulationResult:
    if benchmark.empty:
        empty = pd.DataFrame()
        return BudgetSimulationResult(str(policy.name), {"policy": policy.name, "spent_budget": 0.0}, empty, empty)

    candidate_records = benchmark.sort_values(["candidate_name", "test_name"]).groupby("candidate_name").first()
    states: dict[str, CandidateState] = {}
    for candidate_name, row in candidate_records.iterrows():
        states[str(candidate_name)] = CandidateState(
            candidate_name=str(candidate_name),
            family=str(row["family"]),
            future_valid=bool(row["future_valid"]),
            initial_score=float(row.get("initial_mean_rank_ic", 0.0)),
        )

    tests: dict[str, TestSpec] = {}
    for test_name, row in benchmark.groupby("test_name").first().iterrows():
        tests[str(test_name)] = {"test_name": str(test_name), "cost": float(row["cost"]), "tier": int(row["tier"])}

    results: dict[tuple[str, str], TestResult] = {}
    for _, row in benchmark.iterrows():
        metrics = {
            key: _safe_value(row[key])
            for key in benchmark.columns
            if key
            not in {
                "episode",
                "candidate_name",
                "test_name",
                "tier",
                "cost",
                "status",
                "rejected",
                "evidence_score",
                "notes",
            }
        }
        result = TestResult(
            candidate_name=str(row["candidate_name"]),
            test_name=str(row["test_name"]),
            tier=int(row["tier"]),
            cost=float(row["cost"]),
            status=str(row["status"]),
            rejected=bool(row["rejected"]),
            evidence_score=float(row["evidence_score"]),
            metrics=metrics,
            notes=str(row.get("notes", "")),
        )
        results[(result.candidate_name, result.test_name)] = result

    spent = 0.0
    history_rows: list[dict[str, Any]] = []
    min_cost = min(float(spec["cost"]) for spec in tests.values()) if tests else float("inf")
    effective_stopping_rule = getattr(policy, "stopping_rule", None) or stopping_rule
    while spent + min_cost <= total_budget:
        action = policy.choose(states, tests, results, total_budget - spent)
        if action is None:
            break
        candidate_name, test_name = action
        result = results.get((candidate_name, test_name))
        if result is None or result.cost > total_budget - spent:
            break
        state = states[candidate_name]
        if test_name in state.tests_run or not state.active:
            break
        state.record(result)
        spent += result.cost
        effective_stopping_rule.update(state)
        history_rows.append(
            {
                "step": len(history_rows) + 1,
                "candidate_name": candidate_name,
                "test_name": test_name,
                "cost": result.cost,
                "spent_budget": spent,
                "status_after": state.status,
                "posterior_failure": state.posterior_failure,
                "test_rejected": int(result.rejected),
                "future_valid": int(state.future_valid),
            }
        )

    for state in states.values():
        if state.status == "active":
            state.status = "undetermined"

    decision_rows = []
    for state in states.values():
        decision_rows.append(
            {
                "policy": policy.name,
                "candidate_name": state.candidate_name,
                "family": state.family,
                "future_valid": int(state.future_valid),
                "status": state.status,
                "posterior_failure": state.posterior_failure,
                "tests_run": len(state.history),
                "cost_spent": state.cost_spent,
            }
        )
    decisions = pd.DataFrame(decision_rows)
    history = pd.DataFrame(history_rows)
    summary = summarize_decisions(decisions, spent, policy.name, total_budget)
    return BudgetSimulationResult(policy.name, summary, decisions, history)


def summarize_decisions(decisions: pd.DataFrame, spent_budget: float, policy_name: str, total_budget: float) -> dict[str, float | int | str]:
    retained = decisions["status"].isin(["survived", "undetermined"])
    rejected = decisions["status"].eq("rejected")
    valid = decisions["future_valid"].astype(bool)
    invalid = ~valid
    retained_count = int(retained.sum())
    valid_count = int(valid.sum())
    correct_rejections = int((rejected & invalid).sum())
    false_eliminations = int((rejected & valid).sum())
    summary = {
        "policy": policy_name,
        "spent_budget": float(spent_budget),
        "budget_utilization": float(spent_budget / total_budget) if total_budget else 0.0,
        "n_candidates": int(len(decisions)),
        "n_retained": retained_count,
        "n_rejected": int(rejected.sum()),
        "n_valid": valid_count,
        "survivor_precision": float((retained & valid).sum() / retained_count) if retained_count else 0.0,
        "survivor_recall": float((retained & valid).sum() / valid_count) if valid_count else 0.0,
        "correct_rejections": correct_rejections,
        "cost_to_correct_rejection": float(spent_budget / correct_rejections) if correct_rejections else float("inf"),
        "false_elimination_rate": float(false_eliminations / valid_count) if valid_count else 0.0,
    }
    return summary


def _safe_value(value: object) -> object:
    if isinstance(value, (np.floating, float)) and not np.isfinite(float(value)):
        return 0.0
    return value

from dataclasses import dataclass

import pandas as pd



@dataclass(frozen=True)
class Episode:
    name: str
    search_start: pd.Timestamp
    search_end: pd.Timestamp
    falsification_start: pd.Timestamp
    falsification_end: pd.Timestamp
    future_start: pd.Timestamp
    future_end: pd.Timestamp

    def to_record(self) -> dict[str, str]:
        return {
            "episode": self.name,
            "search_start": str(self.search_start.date()),
            "search_end": str(self.search_end.date()),
            "falsification_start": str(self.falsification_start.date()),
            "falsification_end": str(self.falsification_end.date()),
            "future_start": str(self.future_start.date()),
            "future_end": str(self.future_end.date()),
        }


def make_episodes(
    panel: PanelData,
    search_days: int,
    falsification_days: int,
    future_days: int,
    step_days: int,
) -> list[Episode]:
    dates = panel.dates
    total = search_days + falsification_days + future_days
    if total > len(dates):
        raise ValueError("Not enough dates to build one full episode.")
    episodes: list[Episode] = []
    start = 0
    while start + total <= len(dates):
        search = dates[start : start + search_days]
        falsification = dates[start + search_days : start + search_days + falsification_days]
        future = dates[start + search_days + falsification_days : start + total]
        episodes.append(
            Episode(
                name=f"episode_{len(episodes):03d}",
                search_start=search[0],
                search_end=search[-1],
                falsification_start=falsification[0],
                falsification_end=falsification[-1],
                future_start=future[0],
                future_end=future[-1],
            )
        )
        start += step_days
    return episodes


def slice_episode(panel: PanelData, episode: Episode) -> tuple[PanelData, PanelData, PanelData]:
    return (
        panel.slice_dates(episode.search_start, episode.search_end),
        panel.slice_dates(episode.falsification_start, episode.falsification_end),
        panel.slice_dates(episode.future_start, episode.future_end),
    )

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd



def build_factor_test_benchmark(
    panel: PanelData,
    episodes: list[Episode],
    config: dict[str, Any],
) -> pd.DataFrame:
    seed = int(config.get("seed", 1))
    factor_cfg = config.get("factors", {})
    label_cfg = config.get("labels", {})
    n_candidates = int(factor_cfg.get("n_candidates", 100))
    max_window = int(factor_cfg.get("max_window", 60))
    horizon = int(label_cfg.get("horizon", 5))
    execution_delay = int(label_cfg.get("execution_delay", 1))
    future_min_rank_ic = float(label_cfg.get("future_min_rank_ic", 0.005))
    future_min_t_stat = float(label_cfg.get("future_min_t_stat", 0.0))
    falsification_cfg = config.get("falsification", {})
    tests = default_test_registry(
        placebo_repetitions=int(falsification_cfg.get("placebo_repetitions", 20)),
        synthetic_repetitions=int(falsification_cfg.get("synthetic_repetitions", 20)),
    )
    rows: list[dict[str, Any]] = []

    verbose = bool(config.get("verbose", True))
    for episode_idx, episode in enumerate(episodes):
        episode_seed_idx = _episode_index(episode, fallback=episode_idx)
        if verbose:
            print(f"building {episode.name} ({episode_idx + 1}/{len(episodes)})", flush=True)
        search_panel, falsification_panel, future_panel = slice_episode(panel, episode)
        factory = CandidateFactory(n_candidates=n_candidates, max_window=max_window, seed=seed + episode_seed_idx * 101)
        candidates = factory.generate()

        for candidate_idx, raw_candidate in enumerate(candidates, start=1):
            if verbose and (candidate_idx == 1 or candidate_idx % 10 == 0 or candidate_idx == len(candidates)):
                print(f"  candidate {candidate_idx}/{len(candidates)}", flush=True)
            check = tier0_checks(raw_candidate, search_panel, max_window=max_window)
            if not check.ok:
                continue

            initial_returns = search_panel.future_returns(horizon=horizon, execution_delay=execution_delay)
            initial_signal = evaluate(raw_candidate.expr, search_panel, {})
            initial = factor_metrics(initial_signal, initial_returns)
            candidate = _orient_candidate(raw_candidate, initial.mean_rank_ic)
            if candidate is not raw_candidate:
                initial_signal = evaluate(candidate.expr, search_panel, {})
                initial = factor_metrics(initial_signal, initial_returns)

            future_signal = evaluate(candidate.expr, future_panel, {})
            future = factor_metrics(
                future_signal,
                future_panel.future_returns(horizon=horizon, execution_delay=execution_delay),
            )
            future_valid = bool(
                future.mean_rank_ic >= future_min_rank_ic
                and future.t_stat >= future_min_t_stat
                and future.coverage >= 0.5
            )

            context = FalsificationContext(
                panel=falsification_panel,
                horizon=horizon,
                execution_delay=execution_delay,
                rng=np.random.default_rng(seed + episode_seed_idx * 1009 + int(candidate.factor_hash[:6], 16)),
            )
            cache = EvaluationCache()
            candidate_record = {
                **episode.to_record(),
                **candidate.to_record(),
                **initial.to_dict(prefix="initial_"),
                **future.to_dict(prefix="future_"),
                "candidate_name": candidate.name,
                "future_valid": int(future_valid),
                "future_failed": int(not future_valid),
            }
            for test in tests:
                result = test.run(candidate, context, cache)
                record = {
                    **candidate_record,
                    **result.to_record(),
                    "target_correct_rejection": int((not future_valid) and result.rejected),
                    "target_false_elimination": int(future_valid and result.rejected),
                }
                rows.append(record)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def split_train_sealed(
    benchmark: pd.DataFrame,
    sealed_episodes: int = 1,
    lineage_isolation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if benchmark.empty:
        return benchmark.copy(), benchmark.copy()
    episodes = sorted(benchmark["episode"].unique())
    sealed_names = set(episodes[-sealed_episodes:])
    sealed = benchmark[benchmark["episode"].isin(sealed_names)].copy()
    train = benchmark[~benchmark["episode"].isin(sealed_names)].copy()
    if lineage_isolation and not train.empty and not sealed.empty:
        sealed_lineages = set(sealed["lineage_hash"].unique())
        isolated = train[~train["lineage_hash"].isin(sealed_lineages)].copy()
        if len(isolated["candidate_name"].unique()) >= 10:
            train = isolated
    return train.reset_index(drop=True), sealed.reset_index(drop=True)


def _orient_candidate(candidate: FactorCandidate, initial_mean_ic: float) -> FactorCandidate:
    if initial_mean_ic >= 0:
        return candidate
    return replace(
        candidate,
        expr=call("neg", candidate.expr),
        parent_hash=candidate.lineage_hash,
        metadata={**candidate.metadata, "oriented": 1},
    )


def _episode_index(episode: Episode, fallback: int) -> int:
    try:
        return int(episode.name.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return fallback

from typing import Any

import pandas as pd



def make_policy_suite(train: pd.DataFrame, sealed_episode: pd.DataFrame, config: dict[str, Any]) -> list[object]:
    test_order = (
        sealed_episode.groupby("test_name")[["tier", "cost"]]
        .first()
        .sort_values(["tier", "cost"])
        .index.astype(str)
        .tolist()
    )
    rejection_rate = train.groupby("test_name")["rejected"].mean().to_dict() if not train.empty else {}
    learned_family_orders, learned_default_order = _learned_family_orders(
        train,
        false_penalty=float(config.get("policy", {}).get("learned_false_penalty", 8.0)),
        shrink=float(config.get("policy", {}).get("learned_shrink", 20.0)),
    )
    family_orders = {
        "reversal": ["execution_delay", "parameter_perturbation", "time_slice_drop", "label_shift", "placebo_shuffle"],
        "momentum": ["parameter_perturbation", "regime_slice", "time_slice_drop", "universe_stress", "cost_capacity"],
        "liquidity": ["universe_stress", "cost_capacity", "neutralization", "regime_slice", "placebo_shuffle"],
        "volatility": ["regime_slice", "neutralization", "parameter_perturbation", "time_slice_drop", "synthetic_null"],
        "volume_price": ["operator_substitution", "universe_stress", "neutralization", "label_shift", "synthetic_null"],
        "mean_reversion": ["execution_delay", "parameter_perturbation", "operator_substitution", "cost_capacity", "placebo_shuffle"],
    }
    policy_cfg = config.get("policy", {})
    correct_model, false_model = train_failure_models(train)
    candidate_records = (
        sealed_episode.sort_values(["candidate_name", "test_name"])
        .groupby("candidate_name")
        .first()
        .reset_index()
        .set_index("candidate_name")
        .to_dict(orient="index")
    )
    return [
        NoFalsificationPolicy(),
        RandomOrderPolicy(seed=int(config.get("seed", 1))),
        FixedChecklistPolicy(order=test_order),
        CheapestFirstPolicy(),
        GlobalRejectionPerCostPolicy(rejection_rate_by_test={str(k): float(v) for k, v in rejection_rate.items()}),
        RuleBasedPolicy(family_orders=family_orders, default_order=test_order),
        RuleBasedPolicy(
            family_orders=learned_family_orders,
            default_order=learned_default_order or test_order,
            name="learned_guardrail_policy",
            stopping_rule=StoppingRule(
                reject_threshold=float(policy_cfg.get("learned_reject_threshold", 0.90)),
                survive_threshold=float(policy_cfg.get("learned_survive_threshold", 0.24)),
                minimum_tests=int(policy_cfg.get("learned_minimum_tests", 8)),
            ),
        ),
        AdaptiveFailurePolicy(
            correct_rejection_model=correct_model,
            false_elimination_model=false_model,
            candidate_records={str(k): v for k, v in candidate_records.items()},
            gamma=float(policy_cfg.get("gamma", 0.8)),
            eta=float(policy_cfg.get("eta", 0.15)),
        ),
        AdaptiveFailurePolicy(
            correct_rejection_model=correct_model,
            false_elimination_model=false_model,
            candidate_records={str(k): v for k, v in candidate_records.items()},
            gamma=float(policy_cfg.get("risk_controlled_gamma", 2.0)),
            eta=float(policy_cfg.get("risk_controlled_eta", 0.1)),
            max_false_risk=float(policy_cfg.get("max_false_risk", 0.12)),
            min_correct_advantage=float(policy_cfg.get("min_correct_advantage", 0.0)),
            name="risk_controlled_adaptive",
        ),
        FullBatteryPolicy(order=test_order),
        OraclePolicy(),
    ]


def _learned_family_orders(
    train: pd.DataFrame,
    false_penalty: float = 8.0,
    shrink: float = 20.0,
) -> tuple[dict[str, list[str]], list[str]]:
    if train.empty:
        return {}, []
    global_stats = train.groupby("test_name").agg(
        correct=("target_correct_rejection", "mean"),
        false=("target_false_elimination", "mean"),
        cost=("cost", "first"),
        n=("target_correct_rejection", "size"),
    )
    global_stats["utility"] = (global_stats["correct"] - false_penalty * global_stats["false"]) / global_stats["cost"]
    default_order = global_stats.sort_values("utility", ascending=False).index.astype(str).tolist()
    family_orders: dict[str, list[str]] = {}
    for family, rows in train.groupby("family"):
        stats = rows.groupby("test_name").agg(
            correct=("target_correct_rejection", "mean"),
            false=("target_false_elimination", "mean"),
            cost=("cost", "first"),
            n=("target_correct_rejection", "size"),
        )
        stats = stats.join(
            global_stats[["correct", "false"]].rename(columns={"correct": "global_correct", "false": "global_false"})
        )
        weight = stats["n"] / (stats["n"] + shrink)
        stats["utility"] = (
            weight * stats["correct"]
            + (1.0 - weight) * stats["global_correct"]
            - false_penalty * (weight * stats["false"] + (1.0 - weight) * stats["global_false"])
        ) / stats["cost"]
        family_orders[str(family)] = stats.sort_values("utility", ascending=False).index.astype(str).tolist()
    return family_orders, default_order

from pathlib import Path
from typing import Any

import pandas as pd



def run_research_pipeline(
    config: dict[str, Any],
    output: str | Path,
    prices: str | Path | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(output)
    panel = _load_panel(config, prices=prices)
    episode_cfg = config.get("episodes", {})
    episodes = make_episodes(
        panel,
        search_days=int(episode_cfg.get("search_days", 252)),
        falsification_days=int(episode_cfg.get("falsification_days", 126)),
        future_days=int(episode_cfg.get("future_days", 126)),
        step_days=int(episode_cfg.get("step_days", 84)),
    )
    benchmark = _build_benchmark_with_optional_checkpoints(panel, episodes, config, out_dir)
    benchmark_path = out_dir / "benchmark.csv"
    benchmark.to_csv(benchmark_path, index=False)

    sealed_episodes = int(episode_cfg.get("sealed_episodes", 1))
    train, sealed = split_train_sealed(benchmark, sealed_episodes=sealed_episodes, lineage_isolation=True)
    train.to_csv(out_dir / "train_benchmark.csv", index=False)
    sealed.to_csv(out_dir / "sealed_benchmark.csv", index=False)

    budget_cfg = config.get("budget", {})
    stopping_rule = StoppingRule(
        reject_threshold=float(budget_cfg.get("reject_threshold", 0.72)),
        survive_threshold=float(budget_cfg.get("survive_threshold", 0.28)),
        minimum_tests=int(budget_cfg.get("minimum_tests", 2)),
    )
    total_budget = float(budget_cfg.get("total", 1000.0))

    results: list[BudgetSimulationResult] = []
    for episode_name, sealed_episode in sealed.groupby("episode", sort=True):
        for policy in make_policy_suite(train, sealed_episode, config):
            result = simulate_budget(sealed_episode.reset_index(drop=True), policy, total_budget, stopping_rule)
            result.summary["episode"] = episode_name
            results.append(result)

    summary = pd.DataFrame([r.summary for r in results])
    decisions = pd.concat([r.decisions.assign(episode=r.summary.get("episode", "")) for r in results], ignore_index=True) if results else pd.DataFrame()
    history = pd.concat([r.history.assign(policy=r.policy_name, episode=r.summary.get("episode", "")) for r in results if not r.history.empty], ignore_index=True) if results else pd.DataFrame()
    summary.to_csv(out_dir / "policy_results.csv", index=False)
    decisions.to_csv(out_dir / "policy_decisions.csv", index=False)
    history.to_csv(out_dir / "policy_history.csv", index=False)

    report_cfg = config.get("report", {})
    report_path = Path(report_cfg.get("path", out_dir / "report.md"))
    write_markdown_report(report_path, config, panel.summary(), episodes, benchmark, summary)
    write_json(
        out_dir / "run_manifest.json",
        {
            "panel": panel.summary(),
            "n_episodes": len(episodes),
            "benchmark_rows": int(len(benchmark)),
            "benchmark_path": str(benchmark_path),
            "report_path": str(report_path),
        },
    )
    return {
        "panel": panel.summary(),
        "episodes": len(episodes),
        "benchmark_path": benchmark_path,
        "summary_path": out_dir / "policy_results.csv",
        "report_path": report_path,
        "summary": summary,
    }


def _load_panel(config: dict[str, Any], prices: str | Path | None = None):
    data_cfg = config.get("data", {})
    if prices is not None:
        panel = load_price_volume_csv(prices)
        return panel.slice_dates(data_cfg.get("start"), data_cfg.get("end"))
    if data_cfg.get("mode", "synthetic") == "synthetic":
        panel = make_synthetic_panel(
            n_assets=int(data_cfg.get("n_assets", 120)),
            n_days=int(data_cfg.get("n_days", 720)),
            seed=int(config.get("seed", 7)),
            start=str(data_cfg.get("start", "2018-01-01")),
        )
        return panel.slice_dates(None, data_cfg.get("end"))
    if "path" in data_cfg:
        panel = load_price_volume_csv(data_cfg["path"])
        return panel.slice_dates(data_cfg.get("start"), data_cfg.get("end"))
    raise ValueError("No data source configured. Use synthetic mode or pass a price CSV.")


def _build_benchmark_with_optional_checkpoints(
    panel,
    episodes,
    config: dict[str, Any],
    out_dir: Path,
) -> pd.DataFrame:
    checkpoint_cfg = config.get("checkpoint", {})
    if not checkpoint_cfg.get("enabled", False):
        return build_factor_test_benchmark(panel, episodes, config)

    checkpoint_dir = out_dir / str(checkpoint_cfg.get("dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(checkpoint_cfg.get("resume", True))
    frames: list[pd.DataFrame] = []
    for episode in episodes:
        path = checkpoint_dir / f"{episode.name}.csv"
        if resume and path.exists() and path.stat().st_size > 0:
            print(f"loading checkpoint {path}", flush=True)
            frames.append(pd.read_csv(path))
            continue
        frame = build_factor_test_benchmark(panel, [episode], config)
        frame.to_csv(path, index=False)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
