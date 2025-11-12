# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import deque
import os


# Core Data Structures
@dataclass
class FactorConfig:
    name: str
    lookback_period: int
    rebalance_freq: str
    ic_threshold: float = 0.01
    turnover_limit: float = 0.5


# RPN Parser (Reverse Polish Notation)
class ReversePolishNotationParser:

    def __init__(self):
        self.operators = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: np.divide(a, b, where=b != 0, out=np.zeros_like(a, dtype=float)),
            'max': lambda a, b: np.maximum(a, b),
            'min': lambda a, b: np.minimum(a, b),
            '^': lambda a, b: np.power(a, b),
        }

    def parse(self, rpn_expr: str, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        stack = deque()
        tokens = rpn_expr.split()

        for token in tokens:
            if token in self.operators:
                if len(stack) < 2:
                    raise ValueError(f"Operator '{token}' requires at least 2 operands")
                b = stack.pop()
                a = stack.pop()
                result = self.operators[token](a, b)
                stack.append(result)
            elif token in data_dict:
                stack.append(data_dict[token])
            else:
                try:
                    stack.append(float(token))
                except ValueError:
                    raise ValueError(f"Unknown token: {token}")

        if len(stack) != 1:
            raise ValueError("Invalid RPN expression")
        return stack.pop()


# Factor Pool Manager
class FactorPoolManager:

    def __init__(self, config: FactorConfig):
        self.config = config
        self.factors = {}
        self.factor_scores = {}
        self.rpn_parser = ReversePolishNotationParser()

    def register_factor(self, name: str, rpn_expr: str):
        self.factors[name] = rpn_expr
        self.factor_scores[name] = {'ic': 0, 'icir': 0, 'turnover': 0}

    def compute_factor(self, rpn_expr: str, price_data: pd.DataFrame) -> pd.Series:
        data_dict = {
            'close': price_data['close'].values,
            'open': price_data['open'].values,
            'high': price_data['high'].values,
            'low': price_data['low'].values,
            'volume': price_data['volume'].values,
            'sma_20': self._sma(price_data['close'], 20).values,
            'sma_50': self._sma(price_data['close'], 50).values,
            'std_20': self._std(price_data['close'], 20).values,
        }
        factor_values = self.rpn_parser.parse(rpn_expr, data_dict)
        return pd.Series(factor_values, index=price_data.index)

    @staticmethod
    def _sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window).mean()

    @staticmethod
    def _std(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window).std()


# Analytics Helpers
class FactorBacktestAnalyzer:

    @staticmethod
    def _safe_series(x) -> pd.Series:
        if x is None:
            return pd.Series(dtype=float)
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        return x

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free: float = 0.0) -> float:
        r = self._safe_series(returns)
        if len(r) == 0:
            return 0.0
        ex = r - risk_free / 252.0
        mu = ex.mean()
        sd = ex.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return 0.0
        sharpe = (mu / sd) * np.sqrt(252.0)
        if np.isnan(sharpe) or np.isinf(sharpe):
            return 0.0
        return float(sharpe)

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        r = self._safe_series(returns)
        if len(r) == 0:
            return 0.0
        equity = (1.0 + r).cumprod()
        peak = equity.cummax()
        dd = (equity / peak) - 1.0
        mdd = dd.min()
        if np.isnan(mdd):
            return 0.0
        return float(mdd)


class IntelligentFactorEnhancer:

    def detect_market_regime(self, returns: pd.Series, win: int = 60) -> str:
        if returns is None or len(returns) == 0:
            return "unknown"
        vol = returns.rolling(win).std().bfill()
        q1, q2, q3 = vol.quantile([0.25, 0.5, 0.75])
        v = vol.iloc[-1]
        if v <= q1: return "low_vol"
        if v <= q2: return "normal"
        if v <= q3: return "high_vol"
        return "crisis"

    def detect_factor_decay(self, factor: pd.Series, lookback: int = 120, thr: float = 0.1) -> bool:
        if factor is None or len(factor) < lookback:
            return False
        tail = factor.dropna().tail(lookback)
        if len(tail) < lookback:
            return False
        mean_abs = tail.abs().mean()
        return float(mean_abs) < thr


class LLMEnhancer:

    def __init__(self, model_name: str = "mistral", simulate_available: bool = True, verbose: bool = True):
        self.model_name = model_name
        self.simulate_available = simulate_available
        self.verbose = verbose

    def is_available(self) -> bool:
        return self.simulate_available

    def enhance_weights(self, factors_dict: Dict[str, pd.Series], weights: Dict[str, float],
                        market_regime: str = "normal", context: Dict = None) -> Dict[str, float]:
        if self.verbose and self.is_available():
            print(f"Using Ollama service (model: {self.model_name})")

        if not self.is_available():
            return dict(weights)

        # Regime-aware adjustment of weights
        w = dict(weights)
        keys = sorted(w.keys())
        if len(keys) == 0:
            return w

        if market_regime in ("high_vol", "crisis"):
            if "Volatility" in w:
                w["Volatility"] = min(1.0, w["Volatility"] + 0.05)
        else:
            if "Momentum" in w:
                w["Momentum"] = min(1.0, w["Momentum"] + 0.03)
            if "Volume" in w:
                w["Volume"] = min(1.0, w["Volume"] + 0.02)

        # Normalize weights
        for k in list(w.keys()):
            w[k] = max(0.0, float(w[k]))
        s = sum(w.values()) or 1.0
        w = {k: v / s for k, v in w.items()}

        if self.verbose:
            print(f"Weight adjustment completed: {w}")
        return w


# Backtest Engine
class BacktestEngine:

    def backtest_factor(self, factor: pd.Series, returns: pd.Series, n_groups: int = 5) -> Dict[str, float]:
        idx = factor.index.intersection(returns.index)
        if len(idx) == 0:
            return {"long_short_return": 0.0, "long_return": 0.0, "short_return": 0.0}

        f = factor.loc[idx].replace([np.inf, -np.inf], np.nan).dropna()
        r = returns.loc[f.index].replace([np.inf, -np.inf], np.nan).dropna()
        idx = f.index.intersection(r.index)
        f, r = f.loc[idx], r.loc[idx]

        if len(idx) < 30:
            return {"long_short_return": 0.0, "long_return": 0.0, "short_return": 0.0}

        try:
            groups = pd.qcut(f.rank(method="first"), q=n_groups, labels=False, duplicates="drop")
        except Exception:
            groups = (f > f.median()).astype(int) * (n_groups - 1)

        top = (groups == groups.max())
        bot = (groups == groups.min())

        long_ret = float(r[top].mean()) if top.any() else 0.0
        short_ret = float(r[bot].mean()) if bot.any() else 0.0
        ls = long_ret - short_ret

        return {"long_short_return": ls, "long_return": long_ret, "short_return": short_ret}


# Dynamic Factor Enhancer
class DynamicFactorEnhancer:

    def __init__(self, llm_enhancer: Optional[LLMEnhancer] = None, verbose: bool = True):
        self.llm_enhancer = llm_enhancer or LLMEnhancer(simulate_available=True, verbose=verbose)
        self.verbose = verbose
        self.intel = IntelligentFactorEnhancer()

    def optimize_weights(self, scores: Dict[str, float], base_weights: Dict[str, float],
                         returns: pd.Series) -> Dict[str, float]:
        # Stage 1: heuristic weighting based on scores
        w = {}
        for k, v in scores.items():
            w[k] = max(0.0, float(v))
        s = sum(w.values())
        if s == 0.0:
            n = max(1, len(scores))
            w = {k: 1.0 / n for k in scores.keys()}
        else:
            w = {k: v / s for k, v in w.items()}

        # Stage 2: LLM-based adjustment (with fallback)
        regime = self.intel.detect_market_regime(returns)
        ctx = {"regime": regime, "scores": dict(scores)}
        w2 = self.llm_enhancer.enhance_weights({}, w, market_regime=regime)
        return w2


# Main AlphaFactorSystem
class AlphaFactorSystem:

    def __init__(self, config: FactorConfig = None, llm_enhancer: Optional[LLMEnhancer] = None):
        self.config = config or FactorConfig("Default", 20, 'D')
        self.factor_manager = FactorPoolManager(self.config)
        self.backtest_engine = BacktestEngine()
        self.analyzer = FactorBacktestAnalyzer()
        self.intel = IntelligentFactorEnhancer()
        self.enhancer = DynamicFactorEnhancer(llm_enhancer=llm_enhancer)
        self.factor_funcs: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}

    def register_factor(self, name: str, func: Callable[[pd.DataFrame], pd.Series]):
        self.factor_funcs[name] = func

    def compute_factors(self, price_df: pd.DataFrame) -> Dict[str, pd.Series]:
        out = {}
        for name, fn in self.factor_funcs.items():
            try:
                out[name] = fn(price_df).rename(name)
                print(f"{name}: Computation successful")
            except Exception as e:
                print(f"{name}: Computation failed: {e}")
        return out

    def _calc_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        idx = factor.index.intersection(returns.index)
        if len(idx) < 10:
            return 0.0
        x = factor.loc[idx]
        y = returns.loc[idx]
        if x.std(ddof=1) == 0 or y.std(ddof=1) == 0:
            return 0.0
        return float(pd.Series(x).corr(pd.Series(y)))

    def _calc_icir(self, factor: pd.Series, returns: pd.Series, win: int = 60) -> float:
        idx = factor.index.intersection(returns.index)
        if len(idx) < win:
            return 0.0
        rolling = []
        f = factor.loc[idx]
        r = returns.loc[idx]
        for i in range(win, len(idx)):
            xx = f.iloc[i - win:i]
            yy = r.iloc[i - win:i]
            if xx.std(ddof=1) == 0 or yy.std(ddof=1) == 0:
                continue
            ic = float(xx.corr(yy))
            rolling.append(ic)
        if len(rolling) == 0:
            return 0.0
        return float(np.mean(rolling) / (np.std(rolling, ddof=1) + 1e-12))

    def _calc_turnover(self, factor: pd.Series, win: int = 20) -> float:
        x = factor.replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < win + 1:
            return 0.0
        d = (x - x.shift(1)).abs().dropna()
        return float(d.tail(win).mean())

    def evaluate_factors(self, factors: Dict[str, pd.Series], returns: pd.Series) -> pd.DataFrame:
        rows = []
        for name, f in factors.items():
            ic = self._calc_ic(f, returns)
            icir = self._calc_icir(f, returns)
            to = self._calc_turnover(f)
            rows.append({"factor": name, "ic": ic, "icir": icir, "turnover": to})
        df = pd.DataFrame(rows).set_index("factor").sort_index()

        print("\nFactor Evaluation (IC / ICIR / Turnover)\n")
        print(f"{'Factor':<15}{'IC':>12}{'ICIR':>14}{'Turnover':>14}")
        print("-" * 51)
        for i, row in df.iterrows():
            ic = row["ic"]
            icir = row["icir"]
            to = row["turnover"]
            print(f"{i:<15}{ic:>+12.4f}{icir:>+14.4f}{to:>14.4f}")
        print("")
        return df

    def select_factors(self, eval_df: pd.DataFrame, top_k: int = 3) -> List[str]:
        if eval_df is None or len(eval_df) == 0:
            return []
        df = eval_df.copy()
        df = df.sort_values(by=["icir", "ic"], ascending=[False, False])
        selected = list(df.index[:top_k])
        print(f"Selected factors: {selected}")
        return selected

    def optimize_weights(self, selected: List[str], factor_scores: Dict[str, float],
                         returns: pd.Series) -> Dict[str, float]:
        base = {k: max(0.0, factor_scores.get(k, 0.0)) for k in selected}
        s = sum(base.values())
        if s == 0.0 and len(selected) > 0:
            base = {k: 1.0 / len(selected) for k in selected}
        weights = self.enhancer.optimize_weights(base, base, returns)

        for k in selected:
            print(f"{k}: {weights.get(k, 0.0):.4f}")
        return weights

    def combine_factors(self, factors: Dict[str, pd.Series],
                        weights: Dict[str, float]) -> Optional[pd.Series]:
        if len(factors) == 0 or len(weights) == 0:
            return None
        series_list = []
        for name, w in weights.items():
            if name not in factors:
                continue
            s = factors[name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            series_list.append(s * float(w))
        if not series_list:
            return None
        out = sum(series_list)
        return out.rename("Composite_Factor")

    def backtest_by_regime(self, composite_factor: pd.Series, returns: pd.Series,
                           win: int = 60) -> pd.DataFrame:
        if composite_factor is None or len(composite_factor) == 0:
            return pd.DataFrame(columns=["regime", "n_days", "ls", "sharpe", "mdd"])

        vol = returns.rolling(win).std().bfill()
        q1, q2, q3 = vol.quantile([0.25, 0.5, 0.75])

        def tag(v):
            if v <= q1: return "low_vol"
            if v <= q2: return "normal"
            if v <= q3: return "high_vol"
            return "crisis"

        regimes = vol.apply(tag)
        rows = []

        for rg in ["low_vol", "normal", "high_vol", "crisis"]:
            idx = regimes[regimes == rg].index
            if len(idx) < 30:
                continue
            sub_factor = composite_factor.loc[idx]
            sub_returns = returns.loc[idx]
            res = self.backtest_engine.backtest_factor(sub_factor, sub_returns, n_groups=5)
            ls = res.get("long_short_return", 0.0)
            sharpe = self.analyzer.calculate_sharpe_ratio(sub_returns)
            mdd = abs(self.analyzer.calculate_max_drawdown(sub_returns))
            rows.append({
                "regime": rg,
                "n_days": int(len(idx)),
                "ls": float(ls),
                "sharpe": float(sharpe),
                "mdd": float(mdd)
            })

        return pd.DataFrame(rows)

    def run_pipeline(self, price_df: pd.DataFrame, returns_data: pd.Series,
                     factor_definitions: Dict[str, str], top_k: int = 3) -> Dict:
        # Compute RPN-based factors
        print("Computing factors...")
        factors = {}
        for name, rpn_expr in factor_definitions.items():
            try:
                factor = self.factor_manager.compute_factor(rpn_expr, price_df)
                factors[name] = factor
                print(f"{name}: Computation successful")
            except Exception as e:
                print(f"{name}: Computation failed for {e}")
        # Evaluate factors
        print("\nEvaluating factors...")
        eval_df = self.evaluate_factors(factors, returns_data)

        # Select factors
        print("\nSelecting factors...")
        selected = self.select_factors(eval_df, top_k=top_k)

        # Optimize weights
        print("\nOptimizing weights...")
        scores = {k: eval_df.loc[k, 'ic'] for k in selected if k in eval_df.index}
        weights = self.optimize_weights(selected, scores, returns_data)

        # Combine factors
        print("\nCombining factors...")
        composite = self.combine_factors(factors, weights)
        if composite is not None:
            print("Composite factor created successfully")

        # Backtest composite
        print("\nBacktesting...")
        if composite is not None:
            res = self.backtest_engine.backtest_factor(composite, returns_data)
            print(f"Long-short return: {res['long_short_return']:.4f}")
            print(f"Long return: {res['long_return']:.4f}")
            print(f"Short return: {res['short_return']:.4f}")

        return {
            "factors": factors,
            "evaluation": eval_df,
            "selected": selected,
            "weights": weights,
            "composite_factor": composite,
            "evaluation_results": {name: {"ic": eval_df.loc[name, "ic"] if name in eval_df.index else 0,
                                          "icir": eval_df.loc[name, "icir"] if name in eval_df.index else 0}
                                   for name in selected},
        }