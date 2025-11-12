# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from alpha_factor_system import (
    AlphaFactorSystem, FactorConfig, ReversePolishNotationParser
)
from advanced_features import (
    ReinforcementLearningFactorAgent, RLState, FactorBacktestAnalyzer,
    IntelligentFactorEnhancer, compute_financial_reward, apply_action_to_weights,
    compute_trading_cost, RewardCheckpointCallback, DynamicCheckpointing,
    ImprovedLLMEnhancer
)

os.makedirs('./outputs', exist_ok=True)

EPISODE_REWARDS = []
EPISODE_WEIGHTS = []


# Global Configuration Constants
FACTOR_DEFINITIONS = {
    # Improved Momentum Factor: (20-day MA - 50-day MA) / current price
    # RPN expression: sma_20 sma_50 - close /
    'Momentum': 'sma_20 sma_50 - close /',

    # Improved Volatility Factor: Relative volatility = std_20 / sma_20
    # RPN expression: std_20 sma_20 /
    'Volatility': 'std_20 sma_20 /',

    # Volume Factor
    'Volume': 'volume',
}

# Reward Configuration
REWARD_CONFIG = {
    'reward_mode': 'sharpe',  # Sharpe-based reward mode
    'mdd_penalty': 0.3,
    'normalize': 1.0,
}

# Trading Cost Configuration
COST_CONFIG = {
    'fee_rate': 0.0005,
    'spread': 0.0001,
    'pos_cost': 0.0001,
    'lambda_turnover': 0.01,  # Turnover penalty coefficient
}

# RL Training Configuration
RL_CONFIG = {
    'num_episodes': 30,
    'max_steps': 80,
    'early_stop_patience': 10,
    'no_improve_burnin': 15,  # Added 15 burnin steps
    'learning_rate': 0.15,
    'epsilon': 0.3,
}

# Environment Configuration
ENV_CONFIG = {
    'W': 60,
    'H': 60,
    'threshold': 0.10,  # Signal threshold
    'selector_mode': 'softmax',  # Factor selection mode
}


# Utility Functions
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def save_q_table(path: str, q_table: dict):

    def to_float(o):
        if isinstance(o, dict):
            return {k: to_float(v) for k, v in o.items()}
        try:
            return float(o)
        except Exception:
            return o

    with open(path, "w") as f:
        json.dump({k: to_float(v) for k, v in q_table.items()}, f, indent=2)


# Data Generation
def generate_synthetic_data_improved(n_days: int = 1000, n_assets: int = 1) -> tuple:
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    np.random.seed(42)

    S0 = 100

    mu_base = 0.0015  # Changed to 0.15% daily return
    sigma = 0.018  # Changed to 1.8% daily volatility (moderate)

    returns_list = []
    prev_ret = 0.0
    vol_regime = 0.015  # Initial volatility regime

    for i in range(n_days):
        # Momentum effect (20% chance of continued movement)
        momentum_effect = 0.2 * prev_ret if np.random.rand() < 0.3 else 0.0

        # Mean reversion (15% chance of reversal)
        mean_revert_effect = -0.15 * prev_ret if np.random.rand() < 0.2 else 0.0

        # Volatility clustering (non-IID)
        if np.random.rand() < 0.05:  # 5% chance to switch volatility regime
            vol_regime = np.random.uniform(0.010, 0.025)

        # Random shock
        shock = np.random.normal(mu_base, vol_regime)

        # Combine all features
        ret = momentum_effect + mean_revert_effect + shock
        returns_list.append(ret)
        prev_ret = ret

    returns = np.array(returns_list)
    price = S0 * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    price_data = pd.DataFrame({
        'date': dates,
        'open': price * (1 + np.random.normal(0, 0.002, n_days)),
        'high': price * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
        'close': price,
        'volume': np.random.randint(5000000, 15000000, n_days),
    }).set_index('date')

    returns_data = price_data['close'].pct_change().fillna(0.0)

    # Print data statistics
    print("\nData Generation Statistics:")
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"  Sample count: {n_days} days")
    print(f"  Average daily return: {returns.mean() * 100:.4f}%")
    print(f"  Annualized return: {returns.mean() * 252 * 100:.2f}%")
    print(f"  Daily volatility: {returns.std() * 100:.4f}%")
    print(f"  Annualized volatility: {returns.std() * np.sqrt(252) * 100:.2f}%")
    sharpe_annual = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    print(f"  Annualized Sharpe ratio: {sharpe_annual:.4f}\n")

    return price_data, returns_data


# Complete Pipeline
def run_complete_pipeline():
    print("=" * 80)
    print("Generating improved synthetic data...")
    print("=" * 80)

    price_data, returns_data = generate_synthetic_data_improved(n_days=1000, n_assets=1)

    config = FactorConfig(
        name="Demo_Alpha_System_Improved",
        lookback_period=20,
        rebalance_freq='D',
        ic_threshold=0.005,
        turnover_limit=0.5
    )

    llm_enhancer = ImprovedLLMEnhancer(use_local=True)
    system = AlphaFactorSystem(config, llm_enhancer=llm_enhancer)

    results = system.run_pipeline(price_data, returns_data, FACTOR_DEFINITIONS)
    return results, system, price_data, returns_data


# Visualization
def plot_results(episode_rewards, episode_weights_history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Alpha Factor Generation with RL',
                 fontsize=14, fontweight='bold')

    episodes = list(range(1, len(episode_rewards) + 1))

    # Plot 1: Training curve
    axes[0].plot(episodes, episode_rewards, 'o-', linewidth=2, markersize=8,
                 color='#2E86AB', label='Reward Trajectory')
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    axes[0].set_xlabel('Episode', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Total Reward', fontsize=11, fontweight='bold')
    axes[0].set_title('RL Training Curve (Improved)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Mark best point
    best_idx = np.argmax(episode_rewards)
    best_reward = episode_rewards[best_idx]
    axes[0].plot(best_idx + 1, best_reward, 'r*', markersize=20,
                 label=f'Best: {best_reward:.4f}')
    axes[0].legend(fontsize=9)

    # Plot 2: Weight evolution
    if episode_weights_history:
        df_weights = pd.DataFrame(episode_weights_history)
        df_weights.plot(kind='area', stacked=True, ax=axes[1],
                        color=['#A23B72', '#F18F01', '#C73E1D'])
        axes[1].set_xlabel('Training Step', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Weight Allocation', fontsize=11, fontweight='bold')
        axes[1].set_title('Factor Weight Evolution', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper left', fontsize=9)
        axes[1].grid(True, alpha=0.3)

    # Plot 3: Improvement percentage
    improvement = [(episode_rewards[i] - episode_rewards[0]) / abs(episode_rewards[0] + 1e-8) * 100
                   for i in range(len(episode_rewards))]
    colors = ['green' if x >= 0 else 'red' for x in improvement]
    axes[2].bar(episodes, improvement, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_xlabel('Episode', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    axes[2].set_title('Improvement vs Baseline', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('./outputs/rl_training_results_final.png', dpi=150, bbox_inches='tight')
    print("Figure saved to: ./outputs/rl_training_results_final.png")
    plt.close()


# RL Training
def demo_rl_closed_loop_rolling():
    global EPISODE_REWARDS, EPISODE_WEIGHTS

    (results, system, price_data, returns_data) = run_complete_pipeline()

    factors_dict = {name: system.factor_manager.compute_factor(rpn, price_data)
                    for name, rpn in FACTOR_DEFINITIONS.items()}

    evaluations = results['evaluation_results']
    weights0 = dict(results['weights'])
    selected = list(weights0.keys()) if weights0 else list(FACTOR_DEFINITIONS.keys())

    analyzer = FactorBacktestAnalyzer()
    intel = IntelligentFactorEnhancer()
    regime = intel.detect_market_regime(returns_data)

    # Use improved RL parameters
    agent = ReinforcementLearningFactorAgent(
        learning_rate=RL_CONFIG['learning_rate'],
        gamma=0.95,
        epsilon=RL_CONFIG['epsilon']
    )
    actions = ['buy', 'sell', 'hold', 'rebalance']

    # Use improved environment parameters
    W = ENV_CONFIG['W']
    H = ENV_CONFIG['H']
    T = len(returns_data)

    reward_mode = REWARD_CONFIG['reward_mode']
    mdd_penalty = REWARD_CONFIG['mdd_penalty']
    normalize = REWARD_CONFIG['normalize']
    thr = ENV_CONFIG['threshold']
    selector_mode = ENV_CONFIG['selector_mode']
    softmax_tau = 0.5

    # Use improved trading cost parameters
    fee_rate = COST_CONFIG['fee_rate']
    spread = COST_CONFIG['spread']
    pos_cost = COST_CONFIG['pos_cost']
    lambda_turnover = COST_CONFIG['lambda_turnover']

    rr_cursor = 0

    def select_target_factor():
        nonlocal rr_cursor
        if not selected:
            return None
        if selector_mode == "top":
            return max(selected, key=lambda f: evaluations[f]['icir'])
        elif selector_mode == "round_robin":
            f = selected[rr_cursor % len(selected)]
            rr_cursor += 1
            return f
        else:
            icir = np.array([evaluations[f]['icir'] for f in selected], dtype=float)
            icir = (icir - icir.mean()) / (icir.std() + 1e-8)
            probs = softmax(icir / max(softmax_tau, 1e-6))
            return np.random.choice(selected, p=probs)

    def combine_at(index: int, new_w: dict) -> float:
        vals = []
        for fname, w in new_w.items():
            series = factors_dict[fname]
            start = max(0, index - W + 1)
            sub = series.iloc[start:index + 1]
            if len(sub) == 0:
                continue
            mu = sub.mean()
            sd = sub.std(ddof=0)
            z_last = 0.0 if sd == 0 or np.isnan(sd) else float((sub.iloc[-1] - mu) / (sd + 1e-12))
            vals.append(w * z_last)
        return float(np.sum(vals)) if vals else 0.0

    def env_step(state: RLState, action: str):
        t = state.timestamp
        if t + H >= T - 1:
            done = True
            return state, 0.0, done, ""

        tgt = select_target_factor()
        prev_w = state.portfolio_state.get('weights', {}).copy()
        prev_pos = int(state.portfolio_state.get('position', 0))

        new_w = apply_action_to_weights(prev_w, action, target_factor=tgt, eta=0.05)

        llm_enhancer = ImprovedLLMEnhancer(use_local=True)
        market_regime = intel.detect_market_regime(returns_data.iloc[:t])
        enhanced_w = llm_enhancer.enhance_weights(
            factors_dict, new_w, market_regime=market_regime
        )
        # Ensure all factors exist in evaluations
        enhanced_w = {k: v for k, v in enhanced_w.items() if k in evaluations}

        # Normalize weights again if needed
        if sum(enhanced_w.values()) > 0:
            total = sum(enhanced_w.values())
            enhanced_w = {k: v / total for k, v in enhanced_w.items()}
        else:
            # fallback to equal weights if all were removed
            n = len(evaluations)
            enhanced_w = {k: 1.0 / n for k in evaluations.keys()}
        new_w = enhanced_w if enhanced_w else new_w

        EPISODE_WEIGHTS.append(dict(new_w))

        sig = combine_at(t, new_w)

        if sig > thr:
            curr_pos = +1
        elif sig < -thr:
            curr_pos = -1
        else:
            curr_pos = 0

        future_slice = returns_data.iloc[t + 1:t + H + 1]
        pnl_series = curr_pos * future_slice

        cost = compute_trading_cost(
            prev_weights=prev_w, curr_weights=new_w,
            prev_position=prev_pos, curr_position=curr_pos,
            fee_rate=fee_rate, spread=spread, pos_cost=pos_cost
        )
        if lambda_turnover > 0.0 and prev_w:
            l1 = sum(abs(new_w.get(k, 0.0) - prev_w.get(k, 0.0))
                     for k in set(new_w) | set(prev_w))
            cost += lambda_turnover * l1

        step_reward_gross = compute_financial_reward(
            pnl_series, reward_mode=reward_mode,
            mdd_penalty=mdd_penalty, normalize=normalize
        )
        reward = step_reward_gross - cost

        next_t = t + H
        next_scores = {k: evaluations[k]['ic'] for k in selected}
        next_state = RLState(
            factor_scores=next_scores,
            market_regime=regime,
            portfolio_state={'weights': new_w, 'position': curr_pos},
            timestamp=next_t,
            prev_weights=prev_w,
            prev_position=prev_pos,
            eval_snapshot=evaluations
        )
        done = (next_t + H >= T - 1)

        return next_state, float(reward), done, ""

    # Use improved training parameters
    num_episodes = RL_CONFIG['num_episodes']
    max_steps = RL_CONFIG['max_steps']
    early_stop_patience = RL_CONFIG['early_stop_patience']
    no_improve_burnin = RL_CONFIG['no_improve_burnin']

    num_starts = max(1, (T - (W + 2 * H)) // H)

    ckpt = DynamicCheckpointing(checkpoint_dir="./outputs")
    cb = RewardCheckpointCallback(ckpt)

    print("\n" + "=" * 80)
    print(f"Starting RL training: {num_episodes} episodes, {max_steps} steps/episode")
    print("=" * 80 + "\n")

    for ep in range(num_episodes):
        k = ep % max(1, num_starts)
        t0 = W + k * H

        init_scores = {k2: evaluations[k2]['ic'] for k2 in selected if k2 in evaluations}
        init_state = RLState(
            factor_scores=init_scores,
            market_regime=regime,
            portfolio_state={'weights': weights0, 'position': 0},
            timestamp=t0,
            prev_weights=None,
            prev_position=0,
            eval_snapshot=evaluations
        )

        ep_ret = agent.train_episode(
            init_state, env_step, max_steps=max_steps,
            early_stop_patience=early_stop_patience,
            improve_eps=1e-5,
            early_stop_strategy="adaptive",
            no_improve_burnin=no_improve_burnin,
            callbacks=[cb]
        )
        EPISODE_REWARDS.append(ep_ret)
        agent.epsilon = max(0.05, agent.epsilon * 0.90)

        print(f"Episode {ep + 1:2d}/{num_episodes}: Reward = {ep_ret:+.4f}, Epsilon = {agent.epsilon:.4f}")

    save_q_table("./outputs/q_table_final.json", agent.q_table)

    return agent, results, evaluations


# Results Reporting
def print_summary(agent, results, evaluations):
    print("\n" + "=" * 80)
    print("Factor Evaluation Results (IC / ICIR / Turnover)")
    print("=" * 80)
    print(f"{'Factor':<15} {'IC':<12} {'ICIR':<12} {'Turnover':<12}")
    print("-" * 51)
    for factor, eval_res in results['evaluation_results'].items():
        ic = eval_res.get('ic', 0)
        icir = eval_res.get('icir', 0)
        turnover = eval_res.get('turnover', 0) if 'turnover' in str(eval_res) else 0.0
        print(f"{factor:<15} {ic:>+10.4f}  {icir:>+10.4f}  {turnover:>10.4f}")

    print("\n" + "=" * 80)
    print(f"RL Training Results ({len(EPISODE_REWARDS)} Episodes)")
    print("=" * 80)

    best_idx = np.argmax(EPISODE_REWARDS)
    worst_idx = np.argmin(EPISODE_REWARDS)
    avg_reward = np.mean(EPISODE_REWARDS)

    print(f"{'Episode':<12} {'Reward':<15} {'Status':<20}")
    print("-" * 47)
    for i, reward in enumerate(EPISODE_REWARDS):
        status = ""
        if i == best_idx:
            status = "BEST"
        elif i == worst_idx:
            status = "WORST"
        elif reward > 0:
            status = "POSITIVE"
        else:
            status = "NEGATIVE"
        print(f"{i + 1:<12} {reward:>+13.4f}  {status:<20}")

    print("\n" + "=" * 80)
    print("Statistics Summary")
    print("=" * 80)
    print(f"Average reward: {avg_reward:+.4f}")
    print(f"Best reward: {EPISODE_REWARDS[best_idx]:+.4f} (Episode {best_idx + 1})")
    print(f"Worst reward: {EPISODE_REWARDS[worst_idx]:+.4f} (Episode {worst_idx + 1})")
    print(f"Positive rewards: {sum(1 for r in EPISODE_REWARDS if r > 0)}/{len(EPISODE_REWARDS)}")

    if EPISODE_REWARDS[worst_idx] != 0:
        improvement = (EPISODE_REWARDS[best_idx] - EPISODE_REWARDS[worst_idx]) / \
                      abs(EPISODE_REWARDS[worst_idx]) * 100
        print(f"Max improvement: {improvement:>6.1f}% "
              f"(from {EPISODE_REWARDS[worst_idx]:.4f} to {EPISODE_REWARDS[best_idx]:.4f})")

    print("\n" + "=" * 80)
    print("Key Parameter Configuration")
    print("=" * 80)
    print(f"normalize: {REWARD_CONFIG['normalize']}")
    print(f"mdd_penalty: {REWARD_CONFIG['mdd_penalty']}")
    print(f"fee_rate: {COST_CONFIG['fee_rate']}")
    print(f"num_episodes: {RL_CONFIG['num_episodes']}")
    print(f"H (observation window): {ENV_CONFIG['H']}")


# Main
def main():
    print("\n" + "=" * 80)
    print("Alpha Factor Generation System - Improved Version (RL + Enhanced Synthetic Data)")
    print("=" * 80 + "\n")

    agent, results, evaluations = demo_rl_closed_loop_rolling()

    print_summary(agent, results, evaluations)

    print("\nGenerating visualization chart...\n")
    weights_history = EPISODE_WEIGHTS if EPISODE_WEIGHTS else [{}]
    plot_results(EPISODE_REWARDS, weights_history)

    print("\n" + "=" * 80)
    print("Completed Successfully!")
    print("=" * 80)
    print("\nOutput files:")
    print("  • ./outputs/rl_training_results_final.png (training visualization)")
    print("  • ./outputs/q_table_final.json (Q-learning decision table)")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()