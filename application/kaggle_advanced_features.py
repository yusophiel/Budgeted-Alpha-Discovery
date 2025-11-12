# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Callable
import datetime

import numpy as np
import pandas as pd


# Unified Logging System
class Logger:

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.log_file = f"./outputs/{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        os.makedirs('./outputs', exist_ok=True)

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"[{timestamp}] [{self.name}] [{level}]"
        full_msg = f"{prefix} {message}"

        if self.verbose:
            print(full_msg)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(full_msg + "\n")

    def info(self, message: str):
        self.log(message, "INFO")

    def warn(self, message: str):
        self.log(message, "WARN")

    def error(self, message: str):
        self.log(message, "ERROR")

    def debug(self, message: str):
        self.log(message, "DEBUG")


# Financial Analytics
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
        sharpe = (mu / sd) * math.sqrt(252.0)
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


# Market Regime Detection
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


# Robust JSON Extractor
class RobustJSONExtractor:
    
    def __init__(self, logger: Logger = None):
        self.logger = logger or Logger("JSONExtractor")
    
    @staticmethod
    def _extract_json_candidates(text: str) -> List[str]:
        candidates = []

        # balanced braces
        depth = 0
        start_idx = -1
        
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start_idx >= 0:
                    candidates.append(text[start_idx:i+1])
                    start_idx = -1

        # regex-based fallback
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        regex_matches = re.findall(pattern, text)
        candidates.extend(regex_matches)

        return list(set(candidates))
    
    def extract_dict_from_text(self, text: str, target_keys: List[str] = None) -> Optional[Dict]:

        if not text:
            return None
        
        candidates = self._extract_json_candidates(text)
        
        self.logger.debug(f"Found {len(candidates)} JSON candidates")

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)

                if not isinstance(parsed, dict):
                    continue

                if target_keys:
                    keys = set(parsed.keys())
                    target_set = set(target_keys)
                    if len(keys & target_set) < len(target_set) * 0.5:
                        self.logger.debug(
                            f"Skipping candidate: only {len(keys & target_set)}/{len(target_set)} keys matched"
                        )
                        continue
                
                self.logger.debug(f"Successfully extracted JSON: {list(parsed.keys())}")
                return parsed
                
            except json.JSONDecodeError as e:
                self.logger.debug(f"Failed to parse candidate: {str(e)[:50]}")
                continue
            except Exception as e:
                self.logger.debug(f"Unexpected error parsing JSON: {e}")
                continue
        
        self.logger.warn(f"Failed to extract valid JSON from {len(candidates)} candidates")
        return None
    
    def extract_numeric_dict(self, text: str, target_keys: List[str] = None,
                            allow_nan: bool = False) -> Optional[Dict[str, float]]:

        parsed = self.extract_dict_from_text(text, target_keys)
        
        if parsed is None:
            return None
        
        result = {}
        try:
            for k, v in parsed.items():
                fv = float(v)

                if np.isnan(fv) or np.isinf(fv):
                    if not allow_nan:
                        self.logger.warn(f"Skipping invalid value for key '{k}': {v}")
                        continue
                    fv = 0.0
                
                result[k] = fv
            
            if result:
                self.logger.debug(f"Extracted numeric dict: {result}")
                return result
            
        except (ValueError, TypeError) as e:
            self.logger.warn(f"Failed to convert values to numeric: {e}")
        
        return None


# Checkpointing
class DynamicCheckpointing:

    def __init__(self, checkpoint_dir: str = "./outputs", filename: str = "best_checkpoint.json"):
        self.outdir = checkpoint_dir
        self.filename = filename
        self.logger = Logger("Checkpointing")
        os.makedirs(self.outdir, exist_ok=True)

    def save_checkpoint(self, payload: Dict):
        path = os.path.join(self.outdir, self.filename)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        self.logger.info(f"Checkpoint saved to {path}")


class RewardCheckpointCallback:
    def __init__(self, ckpt: DynamicCheckpointing, window: int = 5):
        self.ckpt = ckpt
        self.window = window
        self.hist: List[float] = []
        self.best: Optional[float] = None
        self.logger = Logger("RewardCallback")

    def on_episode_end(self, episode_idx: int, reward: float, agent_snapshot: Dict):
        self.hist.append(float(reward))
        recent = self.hist[-self.window:]
        avg = float(np.mean(recent)) if len(recent) > 0 else float(reward)

        self.logger.info(f"Episode {episode_idx}: reward={reward:.4f}, avg_recent={avg:.4f}")

        if self.best is None or avg > self.best:
            self.best = avg
            payload = dict(agent_snapshot)
            payload["episode"] = episode_idx
            payload["recent_avg_reward"] = avg
            self.ckpt.save_checkpoint(payload)


# RL State Structure
@dataclass
class RLState:
    factor_scores: Dict[str, float]
    market_regime: str
    portfolio_state: Dict
    timestamp: int
    prev_weights: Optional[Dict] = None
    prev_position: int = 0
    eval_snapshot: Optional[Dict[str, Dict[str, float]]] = None


# Improved LLM Enhancer (Ollama + Logging + JSON Extractor)
class ImprovedLLMEnhancer:

    def __init__(self, use_local: bool = True, ollama_endpoint: str = None, model_name: str = "mistral"):
        self.use_local = use_local
        self.ollama_endpoint = ollama_endpoint or "http://localhost:11434"
        self.model_name = model_name
        self.intel = IntelligentFactorEnhancer()
        self.logger = Logger("LLMEnhancer")
        self.call_count = 0
        self.success_count = 0
        self.fallback_count = 0
        self.json_extractor = RobustJSONExtractor(self.logger)

    def _try_ollama_call(self, factors_dict: Dict[str, pd.Series],
                         weights: Dict[str, float], market_regime: str) -> Optional[Dict[str, float]]:
        try:
            import requests

            self.logger.debug(f"Trying to connect to Ollama: {self.ollama_endpoint}")

            # English prompt
            prompt = f"""
            Given the current factor weights and market regime, optimize the portfolio weights.

            Current weights: {json.dumps({k: f"{v:.4f}" for k, v in weights.items()})}
            Market regime: {market_regime}
            Available factors: {list(factors_dict.keys())}

            Requirements:
            1. Adjust weights based on the market regime.
            2. All weights must be non-negative.
            3. The sum of all weights must be 1.
            4. Return optimized weights in JSON format.

            Response format: {{"factor_name": weight, ...}}
            """

            response = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                },
                timeout=60
            )

            if response.status_code == 200:
                self.logger.info("Ollama LLM call succeeded")
                self.success_count += 1

                result = response.json()
                response_text = result.get("response", "")
                self.logger.debug(f"LLM response: {response_text[:200]}...")

                enhanced = self.json_extractor.extract_numeric_dict(
                    response_text,
                    target_keys=list(factors_dict.keys()),
                    allow_nan=False
                )
                
                if enhanced:
                    total = sum(enhanced.values())
                    if total > 0:
                        enhanced = {k: v / total for k, v in enhanced.items()}
                        self.logger.info(f"Optimized weights from LLM: {enhanced}")
                        return enhanced
                else:
                    self.logger.warn("Failed to extract numeric values from LLM response")
                    
            else:
                self.logger.warn(f"Ollama API returned error: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Ollama call exception: {type(e).__name__}: {e}")
            self.fallback_count += 1

        return None

    def enhance_weights(self, factors_dict: Dict[str, pd.Series], 
                       weights: Dict[str, float],
                       market_regime: str = "normal") -> Optional[Dict[str, float]]:
        if self.use_local:
            return None
        
        return self._try_ollama_call(factors_dict, weights, market_regime)


# Q-Learning Factor Optimization Agent — With Dynamic Early Stop
class ReinforcementLearningFactorAgent:

    def __init__(self, learning_rate: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 0.2, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.90):
        self.alpha = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.logger = Logger("RLAgent")

    @staticmethod
    def _bin(x: float, edges=(-0.05, 0.0, 0.05)) -> int:
        if x is None or np.isnan(x):
            return 1
        if x < edges[0]:
            return 0
        if x > edges[2]:
            return 2
        return 1

    @staticmethod
    def _compute_weight_concentration(weights: Dict[str, float]) -> int:
        if len(weights) == 0:
            return 1

        w_array = np.array([max(0.0, float(v)) for v in weights.values()])
        w_array = w_array[w_array > 1e-10]

        if len(w_array) == 0:
            return 1

        probs = w_array / w_array.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(weights)) if len(weights) > 0 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        if normalized_entropy > 0.7:
            return 0
        elif normalized_entropy > 0.3:
            return 1
        else:
            return 2

    def state_to_hash(self, state: RLState) -> str:
        regime = state.market_regime
        pos = int(state.portfolio_state.get("position", 0))

        ic_values = []
        if state.eval_snapshot:
            for fname in sorted(state.eval_snapshot.keys()):
                ic = state.eval_snapshot[fname].get("ic", 0.0)
                ic_values.append(float(ic))
        elif state.factor_scores:
            for fname in sorted(state.factor_scores.keys()):
                ic_values.append(float(state.factor_scores[fname]))

        ic_bucket = self._bin(np.mean(ic_values) if ic_values else 0.0,
                              edges=(-0.05, 0.0, 0.05))

        icir_values = []
        if state.eval_snapshot:
            for fname in sorted(state.eval_snapshot.keys()):
                icir = state.eval_snapshot[fname].get("icir", 0.0)
                icir_values.append(float(icir))

        icir_bucket = self._bin(np.mean(icir_values) if icir_values else 0.0,
                                edges=(-0.5, 0.0, 0.5))

        w = state.portfolio_state.get("weights", {}) or {}
        concentration = self._compute_weight_concentration(w)

        state_hash = f"R:{regime}|P:{pos}|IC:{ic_bucket}|ICIR:{icir_bucket}|C:{concentration}"
        return state_hash

    def select_action(self, state_key: str, actions: List[str]) -> str:
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        qvals = self.q_table.get(state_key, {})
        if not qvals:
            return np.random.choice(actions)
        return max(qvals.items(), key=lambda kv: kv[1])[0]

    def update_q_value(self, s: str, a: str, r: float, s_next: str, done: bool):
        if s not in self.q_table:
            self.q_table[s] = {}
        if a not in self.q_table[s]:
            self.q_table[s][a] = 0.0
        old = self.q_table[s][a]
        if done:
            target = r
        else:
            next_qs = self.q_table.get(s_next, {})
            target = r + self.gamma * (max(next_qs.values()) if next_qs else 0.0)
        self.q_table[s][a] = old + self.alpha * (target - old)
        self.logger.debug(
            f"Q-Update: state={s[:30]}..., action={a}, reward={r:.4f}, new_Q={self.q_table[s][a]:.4f}"
        )

    def decay_epsilon(self):
        old_eps = self.epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.logger.debug(
            f"Epsilon decayed: {old_eps:.4f} to {self.epsilon:.4f}"
        )

    def snapshot(self) -> Dict:
        return {
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "q_table": self.q_table
        }

    def train_episode(self, init_state: RLState,
                      env_step: Callable[[RLState, str], Tuple[RLState, float, bool, str]],
                      actions: List[str] = None, max_steps: int = 256,
                      callbacks: List[RewardCheckpointCallback] = None,
                      verbose_first_steps: int = 0, early_stop_patience: int = 5,
                      improve_eps: float = 1e-5,
                      early_stop_strategy: str = "adaptive",
                      no_improve_burnin: int = 10) -> float:
        if actions is None:
            actions = ['buy', 'sell', 'hold', 'rebalance']
        if callbacks is None:
            callbacks = []

        state = init_state
        total_reward = 0.0
        no_improve_count = 0
        best_reward = -float('inf')
        step_rewards = []

        for step in range(max_steps):
            state_key = self.state_to_hash(state)
            action = self.select_action(state_key, actions)
            next_state, reward, done, log_str = env_step(state, action)

            self.update_q_value(state_key, action, reward,
                                self.state_to_hash(next_state), done)
            total_reward += float(reward)
            step_rewards.append(float(reward))
            state = next_state

            if verbose_first_steps and step < verbose_first_steps:
                self.logger.info(log_str)

            if step < no_improve_burnin:
                self.logger.debug(f"Step {step}: Burnin phase, skipping early stop check")
                if done:
                    break
                continue

            # Compute adaptive tolerance
            if early_stop_strategy == "adaptive":
                progress = step / max_steps
                adaptive_eps = improve_eps * (1.0 + progress * 5.0)
                
                self.logger.debug(
                    f"Step {step}: adaptive improve_eps={adaptive_eps:.6f} (progress={progress:.2f})"
                )
                
                if total_reward > best_reward + adaptive_eps:
                    best_reward = total_reward
                    no_improve_count = 0
                    self.logger.debug(f"Step {step}: New best reward! {total_reward:.4f}")
                else:
                    no_improve_count += 1
                    
            elif early_stop_strategy == "aggressive":
                if total_reward > best_reward + improve_eps * 0.5:
                    best_reward = total_reward
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
            else:  # patience
                if total_reward > best_reward + improve_eps:
                    best_reward = total_reward
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            if done or no_improve_count >= early_stop_patience:
                self.logger.info(
                    f"Early stop triggered: step={step}, "
                    f"no_improve_count={no_improve_count}, "
                    f"strategy={early_stop_strategy}"
                )
                break

        self.decay_epsilon()

        for cb in callbacks:
            cb.on_episode_end(episode_idx=-1, reward=total_reward, agent_snapshot=self.snapshot())

        avg_step_reward = np.mean(step_rewards) if step_rewards else 0.0
        self.logger.info(
            f"Episode completed: total_reward={total_reward:.4f}, "
            f"avg_step_reward={avg_step_reward:.4f}, "
            f"steps={step + 1}, "
            f"early_stop_strategy={early_stop_strategy}"
        )
        return float(total_reward)


# Helper Functions
def apply_action_to_weights(current_weights: Dict[str, float], action: str,
                            target_factor: str = None, eta: float = 0.05) -> Dict[str, float]:
    w = dict(current_weights)
    for k in w:
        w[k] = max(0.0, float(w[k]))

    if target_factor and action == "buy":
        w[target_factor] = w.get(target_factor, 0.0) + eta
    elif target_factor and action == "sell":
        w[target_factor] = max(0.0, w.get(target_factor, 0.0) - eta)
    elif action == "rebalance":
        n = max(1, len(w))
        eq = 1.0 / n
        for k in w:
            w[k] += 0.25 * (eq - w[k])

    s = sum(w.values()) or 1.0
    w = {k: v / s for k, v in w.items()}
    return w


def compute_financial_reward(window_returns: pd.Series, reward_mode: str = "sharpe",
                             mdd_penalty: float = 0.7, normalize: float = 5.0) -> float:
    analyzer = FactorBacktestAnalyzer()
    r = window_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return 0.0

    if reward_mode == "sharpe":
        core = analyzer.calculate_sharpe_ratio(r)
    else:
        core = float(r.mean()) * 252.0

    mdd = abs(analyzer.calculate_max_drawdown(r))
    rew = core - mdd_penalty * mdd
    return float(rew) / float(normalize)


def compute_trading_cost(prev_weights: Dict[str, float], curr_weights: Dict[str, float],
                         prev_position: int, curr_position: int,
                         fee_rate: float = 0.002, spread: float = 0.0003,
                         pos_cost: float = 0.001) -> float:
    if not prev_weights or not curr_weights:
        return 0.0

    weight_cost = 0.0
    for k in set(prev_weights.keys()) | set(curr_weights.keys()):
        prev_w = prev_weights.get(k, 0.0)
        curr_w = curr_weights.get(k, 0.0)
        weight_cost += abs(curr_w - prev_w) * fee_rate

    position_cost = 0.0
    if prev_position != curr_position:
        position_cost = abs(curr_position - prev_position) * (spread + pos_cost)

    return float(weight_cost + position_cost)