"""Consolidated module for the adaptive alpha project."""

from __future__ import annotations

from adaptive_alpha.data_handling import PanelData

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Iterable


JsonScalar = str | int | float | bool | None
Arg = "Expr | JsonScalar"


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple[Arg, ...] = ()
    params: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "params", tuple(sorted(self.params)))

    @property
    def param_dict(self) -> dict[str, JsonScalar]:
        return dict(self.params)

    def serialize(self, lineage: bool = False) -> dict[str, Any]:
        params = {}
        for key, value in self.params:
            if lineage and key in {"window", "lag", "horizon", "power", "lower", "upper"}:
                params[key] = "<param>"
            else:
                params[key] = value
        args: list[Any] = []
        for arg in self.args:
            if isinstance(arg, Expr):
                args.append(arg.serialize(lineage=lineage))
            elif lineage and isinstance(arg, (int, float)):
                args.append("<number>")
            else:
                args.append(arg)
        return {"op": self.op, "args": args, "params": params}

    def to_json(self, lineage: bool = False) -> str:
        return json.dumps(self.serialize(lineage=lineage), sort_keys=True, separators=(",", ":"))

    def hash(self, length: int = 16) -> str:
        digest = hashlib.blake2b(self.to_json().encode("utf-8"), digest_size=16).hexdigest()
        return digest[:length]

    def lineage_hash(self, length: int = 16) -> str:
        digest = hashlib.blake2b(self.to_json(lineage=True).encode("utf-8"), digest_size=16).hexdigest()
        return digest[:length]

    def depth(self) -> int:
        child_depths = [arg.depth() for arg in self.args if isinstance(arg, Expr)]
        return 1 + (max(child_depths) if child_depths else 0)

    def node_count(self) -> int:
        return 1 + sum(arg.node_count() for arg in self.args if isinstance(arg, Expr))

    def fields(self) -> set[str]:
        if self.op == "field" and self.args:
            return {str(self.args[0])}
        out: set[str] = set()
        for arg in self.args:
            if isinstance(arg, Expr):
                out.update(arg.fields())
        return out

    def operators(self) -> list[str]:
        out = [self.op]
        for arg in self.args:
            if isinstance(arg, Expr):
                out.extend(arg.operators())
        return out

    def max_window(self) -> int:
        window = int(self.param_dict.get("window") or self.param_dict.get("lag") or 0)
        child_windows = [arg.max_window() for arg in self.args if isinstance(arg, Expr)]
        return max([window, *child_windows])

    def map_windows(self, fn: Callable[[int], int]) -> "Expr":
        params = []
        for key, value in self.params:
            if key in {"window", "lag"} and isinstance(value, int):
                params.append((key, max(1, int(fn(value)))))
            else:
                params.append((key, value))
        args = tuple(arg.map_windows(fn) if isinstance(arg, Expr) else arg for arg in self.args)
        return Expr(self.op, args, tuple(params))

    def map_ops(self, replacements: dict[str, str]) -> "Expr":
        args = tuple(arg.map_ops(replacements) if isinstance(arg, Expr) else arg for arg in self.args)
        return Expr(replacements.get(self.op, self.op), args, self.params)

    def child_exprs(self) -> Iterable["Expr"]:
        for arg in self.args:
            if isinstance(arg, Expr):
                yield arg

    def without_top(self) -> "Expr":
        for child in self.child_exprs():
            return child
        return self

    def pretty(self) -> str:
        if self.op == "field":
            return str(self.args[0])
        if self.op == "const":
            return str(self.args[0])
        args = ", ".join(arg.pretty() if isinstance(arg, Expr) else repr(arg) for arg in self.args)
        params = ", ".join(f"{k}={v!r}" for k, v in self.params)
        joined = ", ".join(part for part in [args, params] if part)
        return f"{self.op}({joined})"


def field(name: str) -> Expr:
    return Expr("field", (name,), ())


def constant(value: float | int) -> Expr:
    return Expr("const", (float(value),), ())


def call(op: str, *args: Arg, **params: JsonScalar) -> Expr:
    return Expr(op, tuple(args), tuple(sorted(params.items())))

from dataclasses import dataclass, field as dataclass_field
from typing import Any



@dataclass
class FactorCandidate:
    name: str
    expr: Expr
    family: str
    generated_by: str = "template"
    parent_hash: str | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def factor_hash(self) -> str:
        return self.expr.hash()

    @property
    def lineage_hash(self) -> str:
        return self.parent_hash or self.expr.lineage_hash()

    def structural_features(self) -> dict[str, float | int | str]:
        operators = self.expr.operators()
        return {
            "name": self.name,
            "factor_hash": self.factor_hash,
            "lineage_hash": self.lineage_hash,
            "family": self.family,
            "generated_by": self.generated_by,
            "ast_depth": self.expr.depth(),
            "node_count": self.expr.node_count(),
            "max_window": self.expr.max_window(),
            "n_ts_ops": sum(op.startswith("ts_") or op in {"delay", "delta", "decay_linear"} for op in operators),
            "has_division": int("safe_divide" in operators),
            "has_corr": int("ts_corr" in operators),
            "has_neutralize": int(any(op.endswith("neutralize") for op in operators)),
            "has_rank": int("rank" in operators or "ts_rank" in operators),
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.structural_features(),
            **self.metadata,
            "expression": self.expr.pretty(),
        }

from collections.abc import Mapping

import numpy as np
import pandas as pd



ALLOWED_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "return",
    "volume",
    "dollar_volume",
    "shares_outstanding",
}


def evaluate(expr: Expr, panel: PanelData, cache: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    if cache is None:
        cache = {}
    key = expr.hash()
    if key in cache:
        return cache[key]

    op = expr.op
    params = expr.param_dict

    if op == "field":
        name = str(expr.args[0])
        if name not in ALLOWED_FIELDS:
            raise ValueError(f"Illegal field in factor expression: {name}")
        out = panel.field(name)
    elif op == "const":
        value = float(expr.args[0])
        base = panel.field("close")
        out = pd.DataFrame(value, index=base.index, columns=base.columns)
    else:
        args = [evaluate(arg, panel, cache) if isinstance(arg, Expr) else arg for arg in expr.args]
        out = _apply_operator(op, args, params, panel)

    out = out.replace([np.inf, -np.inf], np.nan)
    cache[key] = out
    return out


def _apply_operator(op: str, args: list[object], params: Mapping[str, object], panel: PanelData) -> pd.DataFrame:
    if op == "add":
        return _df(args[0]) + _df(args[1])
    if op == "sub":
        return _df(args[0]) - _df(args[1])
    if op == "mul":
        return _df(args[0]) * _df(args[1])
    if op == "neg":
        return -_df(args[0])
    if op == "abs":
        return _df(args[0]).abs()
    if op == "clip":
        return _df(args[0]).clip(float(params.get("lower", -10.0)), float(params.get("upper", 10.0)))
    if op == "log1p_abs":
        return np.log1p(_df(args[0]).abs())
    if op == "signed_power":
        power = float(params.get("power", 2.0))
        x = _df(args[0])
        return np.sign(x) * np.power(np.abs(x), power)
    if op == "safe_divide":
        numerator = _df(args[0])
        denominator = _df(args[1])
        eps = float(params.get("eps", 1e-8))
        return numerator.where(denominator.abs() > eps) / denominator.where(denominator.abs() > eps)
    if op == "delay":
        return _df(args[0]).shift(int(params["window"]))
    if op == "delta":
        return _df(args[0]).diff(int(params["window"]))
    if op == "ts_mean":
        return _df(args[0]).rolling(int(params["window"]), min_periods=_min_periods(params)).mean()
    if op == "ts_sum":
        return _df(args[0]).rolling(int(params["window"]), min_periods=_min_periods(params)).sum()
    if op == "ts_std":
        return _df(args[0]).rolling(int(params["window"]), min_periods=_min_periods(params)).std()
    if op == "ts_rank":
        window = int(params["window"])
        return _df(args[0]).rolling(window, min_periods=_min_periods(params)).apply(_rank_last, raw=True)
    if op == "ts_corr":
        return _df(args[0]).rolling(int(params["window"]), min_periods=_min_periods(params)).corr(_df(args[1]))
    if op == "decay_linear":
        window = int(params["window"])
        weights = np.arange(1, window + 1, dtype=float)
        weights = weights / weights.sum()
        return _df(args[0]).rolling(window, min_periods=_min_periods(params)).apply(
            lambda values: _weighted_last(values, weights), raw=True
        )
    if op == "rank":
        return cross_rank(_df(args[0]))
    if op == "zscore":
        return cross_zscore(_df(args[0]))
    if op == "winsorize":
        return winsorize(_df(args[0]), float(params.get("lower", 0.01)), float(params.get("upper", 0.99)))
    if op == "industry_neutralize":
        return industry_neutralize(_df(args[0]), panel)
    if op == "size_neutralize":
        return size_neutralize(_df(args[0]), panel)
    raise ValueError(f"Unsupported factor operator: {op}")


def _df(value: object) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("Operator expected a pandas DataFrame argument.")
    return value


def _min_periods(params: Mapping[str, object]) -> int:
    window = int(params["window"])
    return int(params.get("min_periods", max(2, int(window * 0.6))))


def _rank_last(values: np.ndarray) -> float:
    last = values[-1]
    if not np.isfinite(last):
        return np.nan
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return np.nan
    less = np.sum(finite < last)
    equal = np.sum(finite == last)
    return float((less + 0.5 * equal) / len(finite))


def _weighted_last(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    local_weights = weights[-len(values) :].copy()
    local_weights = local_weights * mask
    denom = local_weights.sum()
    if denom <= 0:
        return np.nan
    return float(np.nansum(values * local_weights) / denom)


def cross_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True) - 0.5


def cross_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def winsorize(frame: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    lo = frame.quantile(lower, axis=1)
    hi = frame.quantile(upper, axis=1)
    return frame.clip(lower=lo, upper=hi, axis=0)


def industry_neutralize(frame: pd.DataFrame, panel: PanelData) -> pd.DataFrame:
    if "industry" not in panel.metadata.columns:
        return cross_zscore(frame)
    out = frame.copy()
    industries = panel.metadata["industry"].reindex(frame.columns)
    for industry in industries.dropna().unique():
        cols = industries[industries == industry].index
        out.loc[:, cols] = frame.loc[:, cols].sub(frame.loc[:, cols].mean(axis=1), axis=0)
    return out


def size_neutralize(frame: pd.DataFrame, panel: PanelData) -> pd.DataFrame:
    market_cap = np.log1p(panel.market_cap().reindex_like(frame))
    out = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for date in frame.index:
        y = frame.loc[date].to_numpy(dtype=float)
        x = market_cap.loc[date].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 3:
            continue
        design = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
        residual = y.copy()
        residual[mask] = y[mask] - design @ beta
        out.loc[date] = residual
    return out

from dataclasses import dataclass

import numpy as np
import pandas as pd



@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: dict[str, float] | None = None


def validate_expression(expr: Expr, max_window: int = 252) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    fields = expr.fields()
    illegal = fields.difference(ALLOWED_FIELDS)
    if illegal:
        errors.append(f"Illegal fields: {sorted(illegal)}")
    if expr.max_window() > max_window:
        errors.append(f"Window {expr.max_window()} exceeds max_window={max_window}")
    if expr.depth() > 10:
        warnings.append(f"Deep expression tree: depth={expr.depth()}")
    return ValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def validate_signal(signal: pd.DataFrame, min_coverage: float = 0.6, max_abs_value: float = 1e6) -> ValidationResult:
    finite = np.isfinite(signal.to_numpy(dtype=float))
    coverage = float(finite.mean()) if finite.size else 0.0
    errors: list[str] = []
    warnings: list[str] = []
    if coverage < min_coverage:
        errors.append(f"Coverage {coverage:.3f} below minimum {min_coverage:.3f}")
    max_abs = float(np.nanmax(np.abs(signal.to_numpy(dtype=float)))) if finite.any() else float("nan")
    if np.isfinite(max_abs) and max_abs > max_abs_value:
        warnings.append(f"Large absolute signal value: {max_abs:.3g}")
    constant_rows = signal.nunique(axis=1, dropna=True).le(1).mean()
    if constant_rows > 0.5:
        warnings.append(f"More than half of dates have constant cross-sections: {constant_rows:.3f}")
    return ValidationResult(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics={"coverage": coverage, "max_abs": max_abs, "constant_row_fraction": float(constant_rows)},
    )

from dataclasses import dataclass

import numpy as np



@dataclass
class CandidateFactory:
    n_candidates: int = 100
    max_window: int = 60
    seed: int = 1

    def generate(self) -> list[FactorCandidate]:
        rng = np.random.default_rng(self.seed)
        candidates: list[FactorCandidate] = []
        seen: set[str] = set()
        attempts = 0
        while len(candidates) < self.n_candidates and attempts < self.n_candidates * 20:
            attempts += 1
            family = str(rng.choice(FAMILIES))
            expr = _template_expr(family, rng, self.max_window)
            factor_hash = expr.hash()
            if factor_hash in seen:
                continue
            seen.add(factor_hash)
            idx = len(candidates)
            candidates.append(
                FactorCandidate(
                    name=f"F{idx:04d}_{factor_hash[:8]}",
                    expr=expr,
                    family=family,
                    generated_by="template_rng",
                )
            )
        return candidates


FAMILIES = (
    "reversal",
    "momentum",
    "liquidity",
    "volatility",
    "volume_price",
    "mean_reversion",
)


def _window(rng: np.random.Generator, max_window: int, low: int = 2) -> int:
    values = np.array([2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 120])
    values = values[(values >= low) & (values <= max_window)]
    if len(values) == 0:
        return max(low, min(max_window, 5))
    return int(rng.choice(values))


def _template_expr(family: str, rng: np.random.Generator, max_window: int) -> Expr:
    close = field("close")
    ret = field("return")
    volume = field("volume")
    dollar_volume = field("dollar_volume")
    high = field("high")
    low = field("low")

    w1 = _window(rng, max_window)
    w2 = _window(rng, max_window)

    if family == "reversal":
        base = call("neg", call("delta", close, window=w1))
        return _maybe_standardize(base, rng)

    if family == "momentum":
        mom = call("safe_divide", call("delta", close, window=w1), call("delay", close, window=w1))
        smooth = call("ts_mean", mom, window=max(2, min(w2, max_window)))
        return _maybe_standardize(smooth, rng)

    if family == "liquidity":
        illiq = call("safe_divide", call("abs", ret), call("log1p_abs", dollar_volume))
        base = call("neg", call("ts_mean", illiq, window=w1))
        return _maybe_standardize(base, rng)

    if family == "volatility":
        vol = call("ts_std", ret, window=w1)
        base = call("neg", vol) if rng.random() < 0.7 else vol
        return _maybe_standardize(base, rng)

    if family == "volume_price":
        price_rank = call("ts_rank", close, window=w1)
        volume_rank = call("ts_rank", volume, window=w2)
        corr = call("ts_corr", price_rank, volume_rank, window=max(3, min(max_window, int((w1 + w2) / 2))))
        base = call("neg", corr) if rng.random() < 0.5 else corr
        return _maybe_standardize(base, rng)

    if family == "mean_reversion":
        spread = call("safe_divide", call("sub", close, call("ts_mean", close, window=w1)), call("ts_std", close, window=w1))
        range_adj = call("safe_divide", call("sub", close, low), call("sub", high, low))
        base = call("sub", call("neg", spread), range_adj)
        return _maybe_standardize(base, rng)

    return call("rank", ret)


def _maybe_standardize(expr: Expr, rng: np.random.Generator) -> Expr:
    choice = rng.choice(["rank", "zscore", "winsor_rank", "identity"], p=[0.45, 0.25, 0.2, 0.1])
    if choice == "rank":
        return call("rank", expr)
    if choice == "zscore":
        return call("zscore", expr)
    if choice == "winsor_rank":
        return call("rank", call("winsorize", expr, lower=0.02, upper=0.98))
    return expr
