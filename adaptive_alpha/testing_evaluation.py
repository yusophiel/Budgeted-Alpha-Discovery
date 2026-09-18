"""Consolidated module for the adaptive alpha project."""

from __future__ import annotations

from adaptive_alpha.data_handling import PanelData
from adaptive_alpha.factor_generation import Expr, FactorCandidate, call, evaluate, validate_expression, validate_signal, ValidationResult

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorMetrics:
    mean_rank_ic: float
    median_rank_ic: float
    ic_std: float
    icir: float
    t_stat: float
    positive_ic_ratio: float
    coverage: float
    quantile_spread: float
    monotonicity: float
    turnover: float
    cost_drag: float
    net_quantile_spread: float
    n_dates: int

    def to_dict(self, prefix: str = "") -> dict[str, float | int]:
        return {f"{prefix}{k}": v for k, v in self.__dict__.items()}


def rank_ic_series(signal: pd.DataFrame, future_returns: pd.DataFrame, min_assets: int = 10) -> pd.Series:
    x, y = _align(signal, future_returns)
    x_rank = x.rank(axis=1)
    y_rank = y.rank(axis=1)
    ic = x_rank.corrwith(y_rank, axis=1)
    valid_counts = (x.notna() & y.notna()).sum(axis=1)
    ic = ic.where(valid_counts >= min_assets)
    return ic.dropna()


def factor_metrics(
    signal: pd.DataFrame,
    future_returns: pd.DataFrame,
    quantile: float = 0.2,
    cost_bps: float = 5.0,
    min_assets: int = 10,
) -> FactorMetrics:
    x, y = _align(signal, future_returns)
    ic = rank_ic_series(x, y, min_assets=min_assets)
    long_short = quantile_spread_series(x, y, quantile=quantile, min_assets=min_assets)
    turnover = rank_turnover(x)
    cost_drag = turnover * cost_bps / 10000.0
    spread_mean = float(long_short.mean()) if len(long_short) else 0.0
    coverage = float((x.notna() & y.notna()).mean().mean()) if x.size else 0.0

    mean_ic = float(ic.mean()) if len(ic) else 0.0
    std_ic = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    return FactorMetrics(
        mean_rank_ic=mean_ic,
        median_rank_ic=float(ic.median()) if len(ic) else 0.0,
        ic_std=std_ic,
        icir=float(mean_ic / std_ic * np.sqrt(252.0)) if std_ic > 0 else 0.0,
        t_stat=hac_t_stat(ic),
        positive_ic_ratio=float((ic > 0).mean()) if len(ic) else 0.0,
        coverage=coverage,
        quantile_spread=spread_mean,
        monotonicity=quantile_monotonicity(x, y, min_assets=min_assets),
        turnover=float(turnover),
        cost_drag=float(cost_drag),
        net_quantile_spread=float(spread_mean - cost_drag),
        n_dates=int(len(ic)),
    )


def quantile_spread_series(
    signal: pd.DataFrame,
    future_returns: pd.DataFrame,
    quantile: float = 0.2,
    min_assets: int = 10,
) -> pd.Series:
    x, y = _align(signal, future_returns)
    valid = x.notna() & y.notna()
    ranks = x.where(valid).rank(axis=1, pct=True)
    long_mask = ranks >= 1.0 - quantile
    short_mask = ranks <= quantile
    valid_counts = valid.sum(axis=1)
    long_returns = y.where(long_mask).mean(axis=1)
    short_returns = y.where(short_mask).mean(axis=1)
    spread = (long_returns - short_returns).where(valid_counts >= min_assets).dropna()
    spread.name = "long_short"
    return spread


def quantile_monotonicity(signal: pd.DataFrame, future_returns: pd.DataFrame, buckets: int = 5, min_assets: int = 10) -> float:
    x, y = _align(signal, future_returns)
    valid = x.notna() & y.notna()
    ranks = x.where(valid).rank(axis=1, pct=True)
    valid_counts = valid.sum(axis=1)
    means = []
    for bucket in range(buckets):
        lo = bucket / buckets
        hi = (bucket + 1) / buckets
        selected = y.where((ranks > lo) & (ranks <= hi) & (valid_counts >= max(min_assets, buckets * 2)))
        means.append(float(selected.mean(axis=1).mean(skipna=True)))
    means = np.array(means, dtype=float)
    if np.isfinite(means).sum() < 3:
        return 0.0
    valid = np.isfinite(means)
    x_rank = pd.Series(np.arange(buckets)[valid]).rank()
    y_rank = pd.Series(means[valid]).rank()
    corr = x_rank.corr(y_rank)
    return float(corr) if pd.notna(corr) else 0.0


def rank_turnover(signal: pd.DataFrame) -> float:
    ranks = signal.rank(axis=1, pct=True).sub(0.5)
    delta = ranks.diff().abs()
    return float(delta.mean(axis=1).mean(skipna=True)) if delta.size else 0.0


def hac_t_stat(series: pd.Series, max_lag: int | None = None) -> float:
    values = series.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 3:
        return 0.0
    centered = values - values.mean()
    if max_lag is None:
        max_lag = int(min(10, max(1, np.floor(4 * (n / 100.0) ** (2 / 9)))))
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        variance += 2.0 * weight * cov
    if variance <= 0:
        return 0.0
    se = np.sqrt(variance / n)
    return float(values.mean() / se) if se > 0 else 0.0


def _align(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = left.index.intersection(right.index)
    columns = left.columns.intersection(right.columns)
    return left.loc[index, columns], right.loc[index, columns]

import numpy as np
import pandas as pd


def bh_fdr(pvalues: pd.Series | dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    series = pd.Series(pvalues, dtype=float).dropna().clip(0.0, 1.0)
    if series.empty:
        return pd.DataFrame(columns=["p_value", "q_value", "reject"])
    ordered = series.sort_values()
    n = len(ordered)
    raw_q = ordered.to_numpy() * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(raw_q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = pd.DataFrame({"p_value": ordered, "q_value": q}, index=ordered.index)
    out["reject"] = out["q_value"] <= alpha
    return out.reindex(series.index)

import numpy as np
import pandas as pd



def combine_signals(signals: dict[str, pd.DataFrame], weights: dict[str, float] | None = None) -> pd.DataFrame:
    if not signals:
        raise ValueError("At least one signal is required.")
    if weights is None:
        weights = {name: 1.0 / len(signals) for name in signals}
    index = None
    columns = None
    for frame in signals.values():
        index = frame.index if index is None else index.intersection(frame.index)
        columns = frame.columns if columns is None else columns.intersection(frame.columns)
    assert index is not None and columns is not None
    combined = pd.DataFrame(0.0, index=index, columns=columns)
    total_weight = 0.0
    for name, frame in signals.items():
        weight = float(weights.get(name, 0.0))
        if weight == 0:
            continue
        standardized = frame.loc[index, columns].rank(axis=1, pct=True).sub(0.5)
        combined += weight * standardized
        total_weight += abs(weight)
    return combined / total_weight if total_weight > 0 else combined


def target_long_short_weights(signal: pd.DataFrame, quantile: float = 0.2, gross: float = 1.0) -> pd.DataFrame:
    ranks = signal.rank(axis=1, pct=True)
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    long_mask = ranks >= 1.0 - quantile
    short_mask = ranks <= quantile
    long_counts = long_mask.sum(axis=1).replace(0, np.nan)
    short_counts = short_mask.sum(axis=1).replace(0, np.nan)
    weights = weights.mask(long_mask, gross / 2.0)
    weights = weights.div(long_counts, axis=0).where(long_mask, weights)
    short_weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns).mask(short_mask, -gross / 2.0)
    short_weights = short_weights.div(short_counts, axis=0).where(short_mask, short_weights)
    return (weights.fillna(0.0) + short_weights.fillna(0.0)).where(signal.notna(), 0.0)


def portfolio_metrics(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    quantile: float = 0.2,
    cost_bps: float = 5.0,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    returns = quantile_spread_series(signal, forward_returns, quantile=quantile)
    turnover = rank_turnover(signal)
    cost = turnover * cost_bps / 10000.0
    net = returns - cost
    ann = float(net.mean() * periods_per_year) if len(net) else 0.0
    vol = float(net.std(ddof=1) * np.sqrt(periods_per_year)) if len(net) > 1 else 0.0
    sharpe = float(ann / vol) if vol > 0 else 0.0
    equity = (1.0 + net.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)
    return {
        "gross_annualized_return": float(returns.mean() * periods_per_year) if len(returns) else 0.0,
        "net_annualized_return": ann,
        "net_sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "turnover": float(turnover),
        "cost_drag": float(cost),
        "n_periods": int(len(net)),
    }

from pathlib import Path

import pandas as pd



def build_shadow_snapshot(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
    previous_weights: pd.Series | None = None,
    quantile: float = 0.2,
    gross: float = 1.0,
) -> pd.DataFrame:
    """Build an immutable paper-trading target snapshot from observable signals."""

    if as_of_date is None:
        date = signal.dropna(how="all").index.max()
    else:
        date = pd.Timestamp(as_of_date)
    if date not in signal.index:
        raise KeyError(f"Signal date not available: {date}")
    weights = target_long_short_weights(signal.loc[[date]], quantile=quantile, gross=gross).loc[date]
    prices = close.reindex(index=[date], columns=signal.columns).loc[date]
    previous = previous_weights.reindex(signal.columns).fillna(0.0) if previous_weights is not None else pd.Series(0.0, index=signal.columns)
    out = pd.DataFrame(
        {
            "as_of_date": date.date().isoformat(),
            "asset": signal.columns,
            "signal": signal.loc[date].to_numpy(dtype=float),
            "close": prices.to_numpy(dtype=float),
            "previous_weight": previous.to_numpy(dtype=float),
            "target_weight": weights.to_numpy(dtype=float),
        }
    )
    out["trade_weight"] = out["target_weight"] - out["previous_weight"]
    out["side"] = "hold"
    out.loc[out["trade_weight"] > 0, "side"] = "buy"
    out.loc[out["trade_weight"] < 0, "side"] = "sell"
    return out.sort_values(["side", "asset"]).reset_index(drop=True)


def append_shadow_snapshot(path: str | Path, snapshot: pd.DataFrame) -> None:
    """Append a snapshot to CSV without modifying prior rows."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    snapshot.to_csv(out, mode="a", header=write_header, index=False)

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd



@dataclass
class FalsificationContext:
    panel: PanelData
    horizon: int = 5
    execution_delay: int = 1
    cost_bps: float = 5.0
    min_assets: int = 10
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1))

    @property
    def future_returns(self) -> pd.DataFrame:
        return self.panel.future_returns(horizon=self.horizon, execution_delay=self.execution_delay)


@dataclass(frozen=True)
class TestResult:
    candidate_name: str
    test_name: str
    tier: int
    cost: float
    status: str
    rejected: bool
    evidence_score: float
    metrics: dict[str, float | int | str]
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "test_name": self.test_name,
            "tier": self.tier,
            "cost": self.cost,
            "status": self.status,
            "rejected": int(self.rejected),
            "evidence_score": self.evidence_score,
            "notes": self.notes,
            **self.metrics,
        }


class FalsificationTest(Protocol):
    name: str
    tier: int
    cost: float

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: "EvaluationCache") -> TestResult:
        ...


class EvaluationCache:
    def __init__(self) -> None:
        self._signals: dict[str, pd.DataFrame] = {}
        self._metrics: dict[tuple[str, int, int], FactorMetrics] = {}

    def signal(self, expr: Expr, panel: PanelData) -> pd.DataFrame:
        key = expr.hash()
        if key not in self._signals:
            self._signals[key] = evaluate(expr, panel, self._signals)
        return self._signals[key]

    def metrics(self, expr: Expr, context: FalsificationContext) -> FactorMetrics:
        key = (expr.hash(), context.horizon, context.execution_delay)
        if key not in self._metrics:
            self._metrics[key] = factor_metrics(
                self.signal(expr, context.panel),
                context.future_returns,
                cost_bps=context.cost_bps,
                min_assets=context.min_assets,
            )
        return self._metrics[key]


def classify_from_score(score: float, reject_cutoff: float = 0.6, warn_cutoff: float = 0.35) -> tuple[str, bool]:
    if score >= reject_cutoff:
        return "reject", True
    if score >= warn_cutoff:
        return "warn", False
    return "pass", False


def bounded_score(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))

from dataclasses import dataclass

import numpy as np
import pandas as pd



def _direction(value: float) -> float:
    return 1.0 if value >= 0 else -1.0


def _signed_ic(value: float, direction: float) -> float:
    return float(value * direction)


def _base_result(
    candidate: FactorCandidate,
    name: str,
    tier: int,
    cost: float,
    score: float,
    metrics: dict[str, float | int | str],
    notes: str = "",
    reject_cutoff: float = 0.6,
) -> TestResult:
    status, rejected = classify_from_score(score, reject_cutoff=reject_cutoff)
    return TestResult(
        candidate_name=candidate.name,
        test_name=name,
        tier=tier,
        cost=cost,
        status=status,
        rejected=rejected,
        evidence_score=bounded_score(score),
        metrics=metrics,
        notes=notes,
    )


@dataclass(frozen=True)
class ParameterPerturbationTest:
    name: str = "parameter_perturbation"
    tier: int = 1
    cost: float = 1.0

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: EvaluationCache) -> TestResult:
        base = cache.metrics(candidate.expr, context)
        direction = _direction(base.mean_rank_ic)
        variants = []
        for multiplier in (0.7, 0.85, 1.15, 1.3):
            expr = candidate.expr.map_windows(lambda w, m=multiplier: max(2, round(w * m)))
            variants.append(cache.metrics(expr, context).mean_rank_ic)
        signed = np.array([_signed_ic(v, direction) for v in variants], dtype=float)
        base_signed = _signed_ic(base.mean_rank_ic, direction)
        worst = float(np.nanmin(signed)) if len(signed) else 0.0
        drop = max(0.0, base_signed - worst)
        flip_rate = float(np.mean(signed < 0.0)) if len(signed) else 0.0
        score = 0.55 * min(1.0, drop / max(0.01, abs(base.mean_rank_ic) + 1e-6)) + 0.45 * flip_rate
        return _base_result(
            candidate,
            self.name,
            self.tier,
            self.cost,
            score,
            {
                "base_mean_rank_ic": base.mean_rank_ic,
                "worst_perturbed_signed_ic": worst,
                "ic_drop": drop,
                "sign_flip_rate": flip_rate,
            },
        )


@dataclass(frozen=True)
class ExecutionDelayTest:
    name: str = "execution_delay"
    tier: int = 1
    cost: float = 1.0
    delay_days: int = 1

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: EvaluationCache) -> TestResult:
        base = cache.metrics(candidate.expr, context)
        delayed_signal = cache.signal(candidate.expr, context.panel).shift(self.delay_days)
        delayed = factor_metrics(delayed_signal, context.future_returns, cost_bps=context.cost_bps, min_assets=context.min_assets)
        direction = _direction(base.mean_rank_ic)
        base_signed = _signed_ic(base.mean_rank_ic, direction)
        delayed_signed = _signed_ic(delayed.mean_rank_ic, direction)
        score = 0.7 * min(1.0, max(0.0, base_signed - delayed_signed) / max(0.01, abs(base.mean_rank_ic) + 1e-6))
        score += 0.3 * float(delayed_signed < 0.0)
        return _base_result(
            candidate,
            self.name,
            self.tier,
            self.cost,
            score,
            {
                "base_mean_rank_ic": base.mean_rank_ic,
                "delay_mean_rank_ic": delayed.mean_rank_ic,
                "delay_icir": delayed.icir,
                "delay_turnover": delayed.turnover,
            },
        )


@dataclass(frozen=True)
class NodeDeletionTest:
    name: str = "node_deletion"
    tier: int = 1
    cost: float = 1.2

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: EvaluationCache) -> TestResult:
        base = cache.metrics(candidate.expr, context)
        simpler_expr = candidate.expr.without_top()
        simpler = cache.metrics(simpler_expr, context)
        direction = _direction(base.mean_rank_ic)
        base_signed = _signed_ic(base.mean_rank_ic, direction)
        simpler_signed = _signed_ic(simpler.mean_rank_ic, direction)
        complexity_penalty = min(1.0, max(0.0, candidate.expr.node_count() - simpler_expr.node_count()) / 8.0)
        score = 0.65 * float(simpler_signed >= base_signed * 0.9) + 0.35 * complexity_penalty
        return _base_result(
            candidate,
            self.name,
            self.tier,
            self.cost,
            score,
            {
                "base_mean_rank_ic": base.mean_rank_ic,
                "simpler_mean_rank_ic": simpler.mean_rank_ic,
                "base_nodes": candidate.expr.node_count(),
                "simpler_nodes": simpler_expr.node_count(),
            },
            notes="Reject means the extra top node added fragility or unnecessary complexity.",
        )


@dataclass(frozen=True)
class OperatorSubstitutionTest:
    name: str = "operator_substitution"
    tier: int = 1
    cost: float = 1.3

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: EvaluationCache) -> TestResult:
        base = cache.metrics(candidate.expr, context)
        replacements = []
        if "rank" in candidate.expr.operators():
            replacements.append({"rank": "zscore"})
        if "zscore" in candidate.expr.operators():
            replacements.append({"zscore": "rank"})
        if "ts_mean" in candidate.expr.operators():
            replacements.append({"ts_mean": "decay_linear"})
        if not replacements:
            replacements.append({})
        variant_ics = [cache.metrics(candidate.expr.map_ops(rep), context).mean_rank_ic for rep in replacements]
        direction = _direction(base.mean_rank_ic)
        signed = np.array([_signed_ic(v, direction) for v in variant_ics], dtype=float)
        base_signed = _signed_ic(base.mean_rank_ic, direction)
        worst = float(np.nanmin(signed)) if len(signed) else base_signed
        score = min(1.0, max(0.0, base_signed - worst) / max(0.01, abs(base.mean_rank_ic) + 1e-6))
        return _base_result(
            candidate,
            self.name,
            self.tier,
            self.cost,
            score,
            {
                "base_mean_rank_ic": base.mean_rank_ic,
                "worst_substituted_signed_ic": worst,
                "n_substitutions": len(replacements),
            },
        )


@dataclass(frozen=True)
class UniverseStressTest:
    name: str = "universe_stress"
    tier: int = 2
    cost: float = 2.0

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: EvaluationCache) -> TestResult:
        base = cache.metrics(candidate.expr, context)
        signal = cache.signal(candidate.expr, context.panel)
        returns = context.future_returns
        cap = context.panel.market_cap().reindex_like(signal)
        dollar_volume = context.panel.field("dollar_volume").reindex_like(signal)
        masks = {
            "large_cap": cap.ge(cap.median(axis=1), axis=0),
            "small_cap": cap.lt(cap.median(axis=1), axis=0),
            "liquid": dollar_volume.ge(dollar_volume.median(axis=1), axis=0),
            "illiquid": dollar_volume.lt(dollar_volume.median(axis=1), axis=0),
        }
        direction = _direction(base.mean_rank_ic)
        subset_ics = {}
        for name, mask in masks.items():
            metrics = factor_metrics(signal.where(mask), returns, cost_bps=context.cost_bps, min_assets=max(5, context.min_assets // 2))
            subset_ics[name] = metrics.mean_rank_ic
        signed = np.array([_signed_ic(v, direction) for v in subset_ics.values()], dtype=float)
        base_signed = max(_signed_ic(base.mean_rank_ic, direction), 1e-6)
        weak_fraction = float(np.mean(signed < base_signed * 0.25))
        flip_fraction = float(np.mean(signed < 0.0))
        score = 0.55 * weak_fraction + 0.45 * flip_fraction
        return _base_result(
            candidate,
            self.name,
            self.tier,
            self.cost,
            score,
            {"base_mean_rank_ic": base.mean_rank_ic, **{f"{k}_mean_rank_ic": v for k, v in subset_ics.items()}},
        )


@dataclass(frozen=True)
class NeutralizationTest:
    name: str = "neutralization"
    tier: int = 2
    cost: float = 2.4

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: EvaluationCache) -> TestResult:
        base = cache.metrics(candidate.expr, context)
        industry_expr = call("industry_neutralize", candidate.expr)
        size_expr = call("size_neutralize", candidate.expr)
        industry = cache.metrics(industry_expr, context)
        size = cache.metrics(size_expr, context)
        direction = _direction(base.mean_rank_ic)
        base_signed = max(_signed_ic(base.mean_rank_ic, direction), 1e-6)
        scores = []
        for value in [industry.mean_rank_ic, size.mean_rank_ic]:
            signed = _signed_ic(value, direction)
            scores.append(0.7 * min(1.0, max(0.0, base_signed - signed) / max(0.01, abs(base.mean_rank_ic) + 1e-6)) + 0.3 * float(signed < 0.0))
        score = float(np.mean(scores))
        return _base_result(
            candidate,
            self.name,
            self.tier,
            self.cost,
            score,
            {
                "base_mean_rank_ic": base.mean_rank_ic,
                "industry_neutral_mean_rank_ic": industry.mean_rank_ic,
                "size_neutral_mean_rank_ic": size.mean_rank_ic,
            },
        )


@dataclass(frozen=True)
class TimeSliceDropTest:
    name: str = "time_slice_drop"
    tier: int = 3
    cost: float = 2.8
    n_slices: int = 4

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: EvaluationCache) -> TestResult:
        base = cache.metrics(candidate.expr, context)
        signal = cache.signal(candidate.expr, context.panel)
        returns = context.future_returns
        chunks = np.array_split(np.arange(len(signal.index)), self.n_slices)
        ics: list[float] = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            idx = signal.index[chunk]
            metrics = factor_metrics(signal.loc[idx], returns.loc[idx], cost_bps=context.cost_bps, min_assets=context.min_assets)
            ics.append(metrics.mean_rank_ic)
        direction = _direction(base.mean_rank_ic)
        signed = np.array([_signed_ic(v, direction) for v in ics], dtype=float)
        flip_fraction = float(np.mean(signed < 0.0)) if len(signed) else 0.0
        concentration = 0.0
        if len(signed) and np.nansum(np.abs(signed)) > 0:
            concentration = float(np.nanmax(np.abs(signed)) / np.nansum(np.abs(signed)))
        score = 0.6 * flip_fraction + 0.4 * max(0.0, concentration - 0.45) / 0.55
        return _base_result(
            candidate,
            self.name,
            self.tier,
            self.cost,
            bounded_score(score),
            {
                "base_mean_rank_ic": base.mean_rank_ic,
                "slice_flip_fraction": flip_fraction,
                "slice_concentration": concentration,
                **{f"slice_{i}_mean_rank_ic": value for i, value in enumerate(ics)},
            },
        )


@dataclass(frozen=True)
class RegimeSliceTest:
    name: str = "regime_slice"
    tier: int = 3
    cost: float = 3.0

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: EvaluationCache) -> TestResult:
        base = cache.metrics(candidate.expr, context)
        signal = cache.signal(candidate.expr, context.panel)
        returns = context.future_returns
        market_return = context.panel.field("return").mean(axis=1)
        realized_vol = market_return.rolling(20, min_periods=5).std()
        regimes = {
            "up_market": market_return >= market_return.median(),
            "down_market": market_return < market_return.median(),
            "high_vol": realized_vol >= realized_vol.median(),
            "low_vol": realized_vol < realized_vol.median(),
        }
        direction = _direction(base.mean_rank_ic)
        regime_ics = {}
        for name, dates in regimes.items():
            idx = signal.index.intersection(dates[dates].index)
            metrics = factor_metrics(signal.loc[idx], returns.loc[idx], cost_bps=context.cost_bps, min_assets=context.min_assets)
            regime_ics[name] = metrics.mean_rank_ic
        signed = np.array([_signed_ic(v, direction) for v in regime_ics.values()], dtype=float)
        base_signed = max(_signed_ic(base.mean_rank_ic, direction), 1e-6)
        flip_fraction = float(np.mean(signed < 0.0))
        weak_fraction = float(np.mean(signed < base_signed * 0.2))
        score = 0.5 * flip_fraction + 0.5 * weak_fraction
        return _base_result(
            candidate,
            self.name,
            self.tier,
            self.cost,
            score,
            {"base_mean_rank_ic": base.mean_rank_ic, **{f"{k}_mean_rank_ic": v for k, v in regime_ics.items()}},
        )


@dataclass(frozen=True)
class CostCapacityTest:
    name: str = "cost_capacity"
    tier: int = 4
    cost: float = 3.8

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: EvaluationCache) -> TestResult:
        base = cache.metrics(candidate.expr, context)
        dollar_volume = context.panel.field("dollar_volume")
        capacity_proxy = float(dollar_volume.median(axis=1).median() * 0.01)
        spread_abs = abs(base.quantile_spread)
        net_bad = float(base.net_quantile_spread <= 0.0 and spread_abs > 1e-9)
        turnover_bad = min(1.0, max(0.0, base.turnover - 0.25) / 0.35)
        capacity_bad = float(capacity_proxy < 100_000.0)
        score = 0.55 * net_bad + 0.35 * turnover_bad + 0.10 * capacity_bad
        return _base_result(
            candidate,
            self.name,
            self.tier,
            self.cost,
            score,
            {
                "mean_rank_ic": base.mean_rank_ic,
                "turnover": base.turnover,
                "quantile_spread": base.quantile_spread,
                "net_quantile_spread": base.net_quantile_spread,
                "capacity_proxy": capacity_proxy,
            },
        )


@dataclass(frozen=True)
class LabelShiftTest:
    name: str = "label_shift"
    tier: int = 4
    cost: float = 3.6
    shift_days: int = 5

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: EvaluationCache) -> TestResult:
        base = cache.metrics(candidate.expr, context)
        signal = cache.signal(candidate.expr, context.panel)
        shifted_forward = context.future_returns.shift(self.shift_days)
        shifted_backward = context.future_returns.shift(-self.shift_days)
        fwd = factor_metrics(signal, shifted_forward, cost_bps=context.cost_bps, min_assets=context.min_assets)
        bwd = factor_metrics(signal, shifted_backward, cost_bps=context.cost_bps, min_assets=context.min_assets)
        observed = abs(base.mean_rank_ic)
        best_shift = max(abs(fwd.mean_rank_ic), abs(bwd.mean_rank_ic))
        score = float(best_shift >= max(0.005, observed * 0.9))
        return _base_result(
            candidate,
            self.name,
            self.tier,
            self.cost,
            score,
            {
                "base_mean_rank_ic": base.mean_rank_ic,
                "forward_shift_mean_rank_ic": fwd.mean_rank_ic,
                "backward_shift_mean_rank_ic": bwd.mean_rank_ic,
                "best_shift_abs_ic": best_shift,
            },
            reject_cutoff=0.8,
        )


@dataclass(frozen=True)
class PlaceboShuffleTest:
    name: str = "placebo_shuffle"
    tier: int = 4
    cost: float = 4.2
    repetitions: int = 20

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: EvaluationCache) -> TestResult:
        base = cache.metrics(candidate.expr, context)
        signal = cache.signal(candidate.expr, context.panel)
        returns = context.future_returns.copy()
        observed = base.mean_rank_ic
        null_ics = []
        values = returns.to_numpy(dtype=float)
        for _ in range(self.repetitions):
            shuffled = values.copy()
            for row in shuffled:
                context.rng.shuffle(row)
            shuffled_returns = pd.DataFrame(shuffled, index=returns.index, columns=returns.columns)
            null_ics.append(float(rank_ic_series(signal, shuffled_returns, min_assets=context.min_assets).mean()))
        null = np.array(null_ics, dtype=float)
        if observed >= 0:
            p_value = (1.0 + np.sum(null >= observed)) / (len(null) + 1.0)
        else:
            p_value = (1.0 + np.sum(null <= observed)) / (len(null) + 1.0)
        score = min(1.0, p_value / 0.2)
        return _base_result(
            candidate,
            self.name,
            self.tier,
            self.cost,
            score,
            {
                "base_mean_rank_ic": observed,
                "placebo_mean_ic": float(np.nanmean(null)) if len(null) else 0.0,
                "placebo_p_value": float(p_value),
            },
            notes="Reject means the observed IC is not unusual versus shuffled labels.",
        )


@dataclass(frozen=True)
class SyntheticNullTest:
    name: str = "synthetic_null"
    tier: int = 4
    cost: float = 4.0
    repetitions: int = 20

    def run(self, candidate: FactorCandidate, context: FalsificationContext, cache: EvaluationCache) -> TestResult:
        base = cache.metrics(candidate.expr, context)
        signal = cache.signal(candidate.expr, context.panel)
        returns = context.future_returns
        observed = abs(base.mean_rank_ic)
        null_ics = []
        values = signal.to_numpy(dtype=float)
        for _ in range(self.repetitions):
            shuffled = values.copy()
            for row in shuffled:
                context.rng.shuffle(row)
            null_signal = pd.DataFrame(shuffled, index=signal.index, columns=signal.columns)
            null_ics.append(abs(float(rank_ic_series(null_signal, returns, min_assets=context.min_assets).mean())))
        null = np.array(null_ics, dtype=float)
        p_value = (1.0 + np.sum(null >= observed)) / (len(null) + 1.0)
        score = min(1.0, p_value / 0.2)
        return _base_result(
            candidate,
            self.name,
            self.tier,
            self.cost,
            score,
            {
                "base_abs_rank_ic": observed,
                "synthetic_null_mean_abs_ic": float(np.nanmean(null)) if len(null) else 0.0,
                "synthetic_null_p_value": float(p_value),
            },
            notes="Null preserves cross-sectional signal distribution but breaks stock identity.",
        )

def default_test_registry(placebo_repetitions: int = 20, synthetic_repetitions: int = 20) -> list[object]:
    return [
        ParameterPerturbationTest(),
        ExecutionDelayTest(),
        NodeDeletionTest(),
        OperatorSubstitutionTest(),
        UniverseStressTest(),
        NeutralizationTest(),
        TimeSliceDropTest(),
        RegimeSliceTest(),
        CostCapacityTest(),
        LabelShiftTest(),
        PlaceboShuffleTest(repetitions=placebo_repetitions),
        SyntheticNullTest(repetitions=synthetic_repetitions),
    ]


def tier0_checks(candidate: FactorCandidate, panel: PanelData, max_window: int = 252) -> ValidationResult:
    expr_check = validate_expression(candidate.expr, max_window=max_window)
    if not expr_check.ok:
        return expr_check
    signal = evaluate(candidate.expr, panel, {})
    signal_check = validate_signal(signal)
    errors = tuple(expr_check.errors) + tuple(signal_check.errors)
    warnings = tuple(expr_check.warnings) + tuple(signal_check.warnings)
    metrics = signal_check.metrics or {}
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, metrics=metrics)
