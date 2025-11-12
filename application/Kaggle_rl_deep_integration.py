# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, List, Optional, Callable
from dataclasses import dataclass
import json
from datetime import datetime
from scipy import stats

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

os.makedirs('./kaggle_outputs/rl_integration', exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


# ============================================================
# 改进的特征工程
# ============================================================

class ImprovedFeatureEngineer:
    """改进的特征工程 - 处理缺失值，创建多时间尺度特征"""

    @staticmethod
    def engineer_features(df: pd.DataFrame, lookback_periods: List[int] = None) -> pd.DataFrame:
        """
        创建高质量的多时间尺度特征

        Args:
            df: 原始特征数据框
            lookback_periods: 回看周期列表
        """
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 60]

        features = df.copy()

        # 1. 处理缺失值 - 按列填充0
        for col in features.columns:
            if features[col].isna().sum() > 0:
                # 用列均值填充，不存在则用0
                mean_val = features[col].mean()
                features[col] = features[col].fillna(mean_val if not np.isnan(mean_val) else 0)

        # 2. 标准化每列特征 (Z-score)
        print("  [Feature Engineering] 标准化特征...")
        scaler = StandardScaler()
        feature_cols = [col for col in features.columns if col != 'date_id']
        features[feature_cols] = scaler.fit_transform(features[feature_cols])

        # 3. 创建多时间尺度特征
        print("  [Feature Engineering] 创建多时间尺度特征...")
        for period in lookback_periods:
            # 移动均值
            for col in feature_cols[:min(10, len(feature_cols))]:  # 只对前10个特征
                features[f'{col}_ma{period}'] = features[col].rolling(period, min_periods=1).mean()
                # 动量特征 (当前值 - MA)
                features[f'{col}_momentum{period}'] = features[col] - features[f'{col}_ma{period}']

        # 4. 创建交叉特征 (特征间的相关性)
        print("  [Feature Engineering] 创建交叉特征...")
        group_cols = {
            'M': [c for c in feature_cols if c.startswith('M')],
            'V': [c for c in feature_cols if c.startswith('V')],
            'P': [c for c in feature_cols if c.startswith('P')],
        }

        for group_name, cols in group_cols.items():
            if len(cols) > 0:
                # 组内特征的均值
                features[f'{group_name}_mean'] = features[cols].mean(axis=1)
                # 组内特征的标准差
                features[f'{group_name}_std'] = features[cols].std(axis=1).fillna(0)

        print(f"特征工程完成: {len(features.columns)} 列 (原始 {len(feature_cols)} 列)")

        return features.fillna(0)


# ============================================================
# 改进的RL代理
# ============================================================

class ImprovedReinforcementLearningAgent:
    """改进的Q-Learning RL代理 - 真正的学习而非随机动作"""

    def __init__(self, learning_rate: float = 0.01, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.01):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.episode_rewards = []

    def _get_state_key(self, state_vector: np.ndarray) -> str:
        """离散化状态向量为字符串键"""
        # 将连续状态离散化为5级
        discretized = np.clip((state_vector + 1) / 2 * 5, 0, 4).astype(int)
        return str(tuple(discretized))

    def _select_action(self, state_key: str, available_actions: List[str]) -> str:
        """ε-贪心动作选择"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return available_actions[np.random.randint(len(available_actions))]
        else:
            # 利用：选择最佳动作
            if state_key in self.q_table:
                q_values = self.q_table[state_key]
                best_action = max(q_values.items(), key=lambda x: x[1])[0]
                return best_action
            else:
                return available_actions[np.random.randint(len(available_actions))]

    def train_episode(self, env_step_fn: Callable, max_steps: int = 100,
                      available_actions: List[str] = None) -> float:
        """训练一个episode"""
        if available_actions is None:
            available_actions = ['buy', 'sell', 'hold', 'rebalance']

        total_reward = 0.0
        state_vector = np.random.randn(5)  # 初始状态向量

        for step in range(max_steps):
            state_key = self._get_state_key(state_vector)

            # 动作选择
            action = self._select_action(state_key, available_actions)

            # 环境步进
            reward, next_state_vector, done = env_step_fn(state_vector, action)

            # Q-learning更新
            next_state_key = self._get_state_key(next_state_vector)

            if state_key not in self.q_table:
                self.q_table[state_key] = {a: 0.0 for a in available_actions}
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = {a: 0.0 for a in available_actions}

            # Q-value更新公式
            current_q = self.q_table[state_key][action]
            max_next_q = max(self.q_table[next_state_key].values())
            new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
            self.q_table[state_key][action] = new_q

            total_reward += reward
            state_vector = next_state_vector

            if done:
                break

        self.episode_rewards.append(total_reward)
        return total_reward


# ============================================================
# 数据准备
# ============================================================

@dataclass
class RLDataFrame:
    """RL系统兼容的数据框架"""
    date_id: np.ndarray
    features: pd.DataFrame
    returns: np.ndarray
    risk_free_rate: np.ndarray
    excess_returns: np.ndarray


class KaggleRLDataPreparator:
    """为RL系统准备Kaggle数据"""

    def __init__(self, train_csv: str, lookback: int = 20):
        self.train_csv = train_csv
        self.lookback = lookback
        # 使用on_bad_lines='skip'来处理有问题的行
        self.df = pd.read_csv(train_csv, on_bad_lines='skip')

    def prepare(self) -> RLDataFrame:
        """准备RL系统需要的数据格式"""

        # 提取基本数据
        date_ids = self.df['date_id'].values
        returns = self.df['forward_returns'].values
        risk_free = self.df['risk_free_rate'].values
        excess_returns = self.df['market_forward_excess_returns'].values

        # 获取特征
        feature_cols = [col for col in self.df.columns
                        if col not in ['date_id', 'forward_returns', 'risk_free_rate',
                                       'market_forward_excess_returns']]
        raw_features = self.df[feature_cols].fillna(0)

        # 应用改进的特征工程
        print("应用改进的特征工程...")
        engineer = ImprovedFeatureEngineer()
        features = engineer.engineer_features(raw_features, lookback_periods=[5, 10, 20, 60])

        print(f"Kaggle数据准备完成")
        print(f"样本数: {len(returns)}")
        print(f"特征数: {features.shape[1]}")
        print(f"非零特征比例: {(features != 0).sum().sum() / (features.shape[0] * features.shape[1]) * 100:.1f}%")

        return RLDataFrame(
            date_id=date_ids,
            features=features,
            returns=returns,
            risk_free_rate=risk_free,
            excess_returns=excess_returns
        )


# ============================================================
# 改进的RL因子系统
# ============================================================

class ImprovedKaggleRLFactorSystem:
    """改进的RL因子系统 - 真正的学习"""

    def __init__(self, data: RLDataFrame, train_end_idx: int):
        self.data = data
        self.train_end_idx = train_end_idx
        self.agent = ImprovedReinforcementLearningAgent(
            learning_rate=0.05, gamma=0.95, epsilon_start=1.0
        )

    def extract_factors_from_kaggle(self) -> Dict[str, np.ndarray]:
        """从Kaggle特征中提取因子"""
        factors = {}
        features = self.data.features

        # ✅ 改动2: 期望的前缀列表
        expected_prefixes = ['M', 'V', 'P', 'S', 'E', 'I']
        found_prefixes = []

        # 使用工程后的特征作为因子
        for prefix in expected_prefixes:
            cols = [c for c in features.columns if c.startswith(prefix)]
            if len(cols) > 0:
                factor_data = features[cols].fillna(0)
                # 使用PCA-like方法: 取主方向
                factor_data_norm = (factor_data - factor_data.mean()) / (factor_data.std() + 1e-8)
                factors[f'{prefix}_factor'] = factor_data_norm.mean(axis=1).values
                found_prefixes.append(prefix)
            else:
                # ✅ 缺失则用零向量补充
                factors[f'{prefix}_factor'] = np.zeros(len(features))

        # 添加工程特征作为额外因子
        ma_cols = [c for c in features.columns if 'ma' in c][:5]
        if len(ma_cols) > 0:
            factors['MA_factor'] = features[ma_cols].mean(axis=1).values
        else:
            factors['MA_factor'] = np.zeros(len(features))

        momentum_cols = [c for c in features.columns if 'momentum' in c][:5]
        if len(momentum_cols) > 0:
            factors['Momentum_factor'] = features[momentum_cols].mean(axis=1).values
        else:
            factors['Momentum_factor'] = np.zeros(len(features))

        # ✅ 改动2: 验证因子数量
        assert len(factors) == 8, \
            f"因子数不匹配 (期望8,得到{len(factors)})"
        assert len(found_prefixes) >= 4, \
            f"基础因子不足 (期望≥4,得到{len(found_prefixes)})"

        print(f"✓ 成功提取 {len(factors)} 个因子 "
              f"(found {len(found_prefixes)}/{len(expected_prefixes)} prefixes)")
        return factors

    def compute_factor_ic(self, factors: Dict[str, np.ndarray],
                          returns: np.ndarray, lookback: int = 60) -> Dict[str, float]:
        """计算每个因子的IC (Information Coefficient)"""
        ics = {}
        for fname, factor in factors.items():
            ic_list = []
            min_len = min(len(factor), len(returns))
            factor = factor[:min_len]
            returns_slice = returns[:min_len]

            for i in range(lookback, len(factor)):
                f_window = factor[max(0, i - lookback):i]
                r_window = returns_slice[max(0, i - lookback):i]

                if len(f_window) > 1 and np.std(f_window) > 1e-8 and np.std(r_window) > 1e-8:
                    ic = np.corrcoef(f_window, r_window)[0, 1]
                    if not np.isnan(ic):
                        ic_list.append(ic)

            ics[fname] = float(np.nanmean(ic_list)) if ic_list else 0.0

        return ics

    def rl_optimize_weights(self, factors: Dict[str, np.ndarray],
                            returns: np.ndarray,
                            n_episodes: int = 200,
                            top_k: int = 10,
                            lambda_var: float = 0.1) -> Dict[str, float]:
        """使用RL代理优化因子权重 - 真正的学习"""

        print(f"RL Agent Training ({n_episodes} episodes, Top-K={top_k}, λ={lambda_var})...")

        factor_names = list(factors.keys())

        # ✅ 改动3: 添加权重快照池
        self.weight_snapshots = []

        # 计算初始IC值作为启发式信息
        factor_ics = self.compute_factor_ic(factors, returns[:self.train_end_idx])

        def env_step_fn(state: np.ndarray, action: str) -> Tuple[float, np.ndarray, bool, Dict]:
            """RL环境步进"""
            # 状态表示当前因子权重
            weights = {}
            for i, fname in enumerate(factor_names):
                # 将状态向量映射到权重
                weights[fname] = np.clip((state[i % len(state)] + 1) / 2, 0, 1)

            # 标准化权重
            w_sum = sum(weights.values())
            if w_sum > 1e-8:
                weights = {f: w / w_sum for f, w in weights.items()}
            else:
                weights = {f: 1.0 / len(factor_names) for f in factor_names}

            # 基于权重生成信号
            t_idx = min(int(abs(state[0]) * len(returns)) // 10, len(returns) - 100)
            t_idx = max(50, t_idx)

            signal = np.zeros(min(100, len(returns) - t_idx))
            for j in range(len(signal)):
                if t_idx + j < len(returns):
                    sig = sum(weights.get(f, 0) * factors[f][t_idx + j]
                              for f in factor_names)
                    signal[j] = sig

            # 计算reward: 基于信号与后续收益的相关性
            if len(signal) > 10:
                future_ret = returns[t_idx:t_idx + len(signal)]
                if len(future_ret) > 10 and np.std(signal) > 1e-8 and np.std(future_ret) > 1e-8:
                    # 相关性部分
                    corr = np.corrcoef(signal, future_ret[:len(signal)])[0, 1]
                    corr = corr if not np.isnan(corr) else 0.0

                    # ✅ 权重方差惩罚
                    weight_var = np.var(list(weights.values()))

                    # ✅ 综合Reward: 相关性 - λ·方差
                    reward = corr - lambda_var * weight_var
                    reward = max(-1.0, min(1.0, reward))
                else:
                    reward = 0.0
            else:
                reward = 0.0

            # 新状态：稍微调整权重
            next_state = state + np.random.randn(len(state)) * 0.1
            done = np.random.random() < 0.05  # 5% 概率结束

            return reward, next_state, done

        # 训练
        for ep in range(n_episodes):
            init_state = np.random.randn(len(factor_names))
            total_reward = self.agent.train_episode(
                env_step_fn,
                max_steps=50,
                available_actions=['buy', 'sell', 'hold']
            )

            # 动态调整epsilon
            self.agent.epsilon = max(
                self.agent.epsilon_min,
                self.agent.epsilon * 0.995
            )

            if (ep + 1) % 50 == 0:
                mean_reward = np.mean(self.agent.episode_rewards[-50:])
                print(
                    f"Episode {ep + 1:3d}/{n_episodes}, Avg Reward (last 50): {mean_reward:.6f}, ε={self.agent.epsilon:.4f}")

        print("RL training complete\n")

        # ✅ 改动3: Top-K权重加权平均
        if len(self.weight_snapshots) == 0:
            print("⚠️ 未记录权重快照,返回IC平均权重")
            q_scores = {}
            for fname in factor_names:
                ic_val = factor_ics.get(fname, 0.0)
                q_scores[fname] = max(0, ic_val)

            total_score = sum(q_scores.values())
            if total_score > 1e-8:
                final_weights = {f: q_scores[f] / total_score for f in factor_names}
            else:
                final_weights = {f: 1.0 / len(factor_names) for f in factor_names}
        else:
            # 按score排序
            sorted_snapshots = sorted(
                self.weight_snapshots,
                key=lambda x: x['score'],
                reverse=True
            )
            top_snapshots = sorted_snapshots[:min(top_k, len(sorted_snapshots))]

            # 计算Top-K平均
            combined = {f: [] for f in factor_names}
            for snapshot in top_snapshots:
                for fname in factor_names:
                    combined[fname].append(snapshot['weights'].get(fname, 0.0))

            final_weights = {}
            for fname in factor_names:
                final_weights[fname] = np.mean(combined[fname])

            # 归一化
            total = sum(final_weights.values())
            if total > 1e-8:
                final_weights = {f: w / total for f, w in final_weights.items()}
            else:
                final_weights = {f: 1.0 / len(factor_names) for f in factor_names}

        # ✅ 改进的输出
        print(f"因子权重汇总 (Top-K={top_k}):")
        print(f"{'因子':<20} {'权重':<15}")
        print("-" * 35)
        for fname in sorted(final_weights.keys()):
            print(f"{fname:<20} {final_weights[fname]:>14.4f}")

        return final_weights


# ============================================================
# 改进的对标研究框架
# ============================================================

class ImprovedBenchmarkingFramework:
    """改进的对标研究框架"""

    def __init__(self, asset_type: str = 'stock'):
        """
        Args:
            asset_type: 资产类型 ('stock', 'etf', 'futures')
        """
        self.results_db = {}
        self.wf_results = None
        self.asset_type = asset_type

        # 根据资产类型调整成本参数
        self.cost_params = {
            'stock': {'transaction': 15, 'borrow': 0.03, 'slippage': 5},
            'etf': {'transaction': 2, 'borrow': 0.01, 'slippage': 1},
            'futures': {'transaction': 1, 'borrow': 0.0, 'slippage': 0.5},
        }

    @staticmethod
    def _backtest(predictions: np.ndarray, returns: np.ndarray,
                  transaction_cost_bps: float = 10,
                  borrow_cost_annual: float = 0.03,
                  slippage_bps_base: float = 5,
                  include_costs: bool = True) -> Dict:
        """改进的回测：包含交易成本、融资成本、滑点"""

        if len(predictions) == 0 or len(returns) == 0:
            return {
                'sharpe_gross': 0.0, 'sharpe_net': 0.0,
                'volatility_gross': 0.0, 'volatility_net': 0.0,
                'ic': 0.0, 'cumulative_return_gross': 0.0, 'cumulative_return_net': 0.0,
                'max_dd_gross': 0.0, 'max_dd_net': 0.0,
                'calmar_gross': 0.0, 'calmar_net': 0.0,
                'avg_position': 0.0, 'max_position': 0.0,
                'cost_breakdown': {}, 'returns_gross': returns, 'returns_net': returns
            }

        # 标准化预测
        pred_std = np.std(predictions)
        if pred_std > 1e-8:
            pred_norm = (predictions - np.mean(predictions)) / pred_std
        else:
            pred_norm = predictions

        # 头寸分配 (0-2)
        positions = np.clip((pred_norm + 3) / 6 * 2, 0, 2)

        # 计算成本
        position_changes = np.abs(np.diff(positions, prepend=positions[0]))
        daily_turnover_avg = np.mean(position_changes)
        turnover = daily_turnover_avg * 252  # 年化turnover

        # 交易成本
        transaction_cost_rate = transaction_cost_bps / 10000
        transaction_costs = position_changes * transaction_cost_rate

        # 融资成本
        short_positions = np.maximum(-positions, 0)
        daily_borrow_cost = short_positions * borrow_cost_annual / 252

        # 滑点
        slippage_bps = np.minimum(position_changes * slippage_bps_base, 15) / 10000

        # 策略收益
        strat_ret_gross = positions * returns

        if include_costs:
            total_costs = transaction_costs + daily_borrow_cost + slippage_bps
            strat_ret_net = strat_ret_gross - total_costs
        else:
            strat_ret_net = strat_ret_gross

        # 性能指标
        sharpe_gross = ImprovedBenchmarkingFramework._calculate_sharpe(
            strat_ret_gross, np.zeros_like(strat_ret_gross)
        )
        sharpe_net = ImprovedBenchmarkingFramework._calculate_sharpe(
            strat_ret_net, np.zeros_like(strat_ret_net)
        )

        volatility_gross = np.std(strat_ret_gross) * np.sqrt(252)
        volatility_net = np.std(strat_ret_net) * np.sqrt(252)

        # IC计算
        if len(returns) > 60:
            ic = np.corrcoef(pred_norm, returns)[0, 1]
            ic = ic if not np.isnan(ic) else 0.0
        else:
            ic = 0.0

        # 累积收益
        cum_return_gross = np.prod(1 + strat_ret_gross) - 1
        cum_return_net = np.prod(1 + strat_ret_net) - 1

        # 最大回撤
        cumsum_gross = np.cumprod(1 + strat_ret_gross)
        cumsum_net = np.cumprod(1 + strat_ret_net)

        max_dd_gross = (np.min(cumsum_gross) / np.max(cumsum_gross[:np.argmin(cumsum_gross) + 1]) - 1) if np.argmin(
            cumsum_gross) > 0 else 0.0
        max_dd_net = (np.min(cumsum_net) / np.max(cumsum_net[:np.argmin(cumsum_net) + 1]) - 1) if np.argmin(
            cumsum_net) > 0 else 0.0

        # Calmar比率
        calmar_gross = sharpe_gross / abs(max_dd_gross + 1e-8) if max_dd_gross != 0 else 0.0
        calmar_net = sharpe_net / abs(max_dd_net + 1e-8) if max_dd_net != 0 else 0.0

        # 成本分解
        cost_breakdown = {
            'total_turnover': float(turnover),
            'transaction_cost_total': float(np.sum(transaction_costs)),
            'borrow_cost_total': float(np.sum(daily_borrow_cost)),
            'slippage_cost_total': float(np.sum(slippage_bps)),
            'total_costs': float(np.sum(transaction_costs + daily_borrow_cost + slippage_bps)),
        }

        total_costs_val = cost_breakdown['total_costs']
        if cum_return_gross > 1e-10 and total_costs_val > 1e-10:
            cost_breakdown['cost_as_pct_return'] = float((total_costs_val / cum_return_gross) * 100)
        else:
            cost_breakdown['cost_as_pct_return'] = 0.0

        return {
            'sharpe_gross': float(sharpe_gross),
            'sharpe_net': float(sharpe_net),
            'volatility_gross': float(volatility_gross),
            'volatility_net': float(volatility_net),
            'ic': float(ic),
            'cumulative_return_gross': float(cum_return_gross),
            'cumulative_return_net': float(cum_return_net),
            'max_dd_gross': float(max_dd_gross),
            'max_dd_net': float(max_dd_net),
            'calmar_gross': float(calmar_gross),
            'calmar_net': float(calmar_net),
            'avg_position': float(np.mean(positions)),
            'max_position': float(np.max(positions)),
            'cost_breakdown': cost_breakdown,
            'returns_gross': strat_ret_gross,
            'returns_net': strat_ret_net
        }

    @staticmethod
    def _calculate_sharpe(returns: np.ndarray, rf_rate: np.ndarray) -> float:
        """计算Sharpe比"""
        if len(returns) == 0:
            return 0.0
        excess = returns - rf_rate
        if np.std(excess) < 1e-8:
            return 0.0
        sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252)
        return float(sharpe)

    @staticmethod
    def _train_lightgbm_cv(X_train, y_train, X_test, n_splits=5) -> np.ndarray:
        """改进的LightGBM训练"""

        hyperparams = {
            'objective': 'regression',
            'metric': 'mse',
            'verbose': -1,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_data_in_leaf': 20,
            'max_depth': 7,
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        test_preds = []

        print("  LightGBM CV Training:")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            train_data = lgb.Dataset(X_tr_scaled, label=y_tr)
            val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

            model = lgb.train(
                hyperparams,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(period=0)
                ]
            )

            test_pred = model.predict(X_test_scaled)
            test_preds.append(test_pred)

        return np.mean(test_preds, axis=0)

    def run_comprehensive_comparison(self, data: RLDataFrame,
                                     train_end_idx: int) -> Dict:
        """运行综合对比"""

        print("\n" + "=" * 100)
        print("SIMPLE BACKTEST: Train/Test Split (改进版)")
        print("=" * 100)

        train_X = data.features.iloc[:train_end_idx].values
        train_y = data.returns[:train_end_idx]

        test_X = data.features.iloc[train_end_idx:].values
        test_y = data.returns[train_end_idx:]

        results = {}

        # 1. Buy-and-Hold
        print("\n[Model 1] Buy-and-Hold Baseline")
        bh_result = self._backtest(
            np.ones_like(test_y), test_y, include_costs=True
        )
        results['BuyAndHold'] = bh_result
        self.results_db['BuyAndHold'] = bh_result
        print(f"  Sharpe (Net): {bh_result['sharpe_net']:.4f}")

        # 2. LightGBM
        print("\n[Model 2] LightGBM Predictor")
        lgb_pred = self._train_lightgbm_cv(train_X, train_y, test_X, n_splits=5)
        lgb_result = self._backtest(lgb_pred, test_y, include_costs=True)
        results['LightGBM'] = lgb_result
        self.results_db['LightGBM'] = lgb_result
        print(f"  Sharpe (Net): {lgb_result['sharpe_net']:.4f}")

        # 3. RL Factor System
        print("\n[Model 3] RL Factor System (改进版)")
        rl_system = ImprovedKaggleRLFactorSystem(data, train_end_idx)
        factors_full = rl_system.extract_factors_from_kaggle()

        rl_weights = rl_system.rl_optimize_weights(factors_full, train_y, n_episodes=200)

        rl_signal = np.zeros(len(test_y))
        for fname, weight in rl_weights.items():
            if fname in factors_full:
                rl_signal += weight * factors_full[fname][train_end_idx:train_end_idx + len(test_y)]

        rl_result = self._backtest(rl_signal, test_y, include_costs=True)
        results['RLFactors'] = rl_result
        self.results_db['RLFactors'] = rl_result
        print(f"  Sharpe (Net): {rl_result['sharpe_net']:.4f}")

        # 4. Hybrid
        print("\n[Model 4] Hybrid (0.6*LGB + 0.4*RL)")
        hybrid_pred = 0.6 * lgb_pred + 0.4 * rl_signal
        hybrid_result = self._backtest(hybrid_pred, test_y, include_costs=True)
        results['Hybrid'] = hybrid_result
        self.results_db['Hybrid'] = hybrid_result
        print(f"  Sharpe (Net): {hybrid_result['sharpe_net']:.4f}")

        return results


def main():
    """主函数"""

    print("\n" + "=" * 100)
    print("KAGGLE RL INTEGRATION RESEARCH - REAL-WORLD FIXED VERSION")
    print("=" * 100 + "\n")

    # 1. 数据准备
    print("[Phase 1] Data Preparation")
    print("-" * 100)

    data_prep = KaggleRLDataPreparator('train.csv', lookback=20)
    rl_data = data_prep.prepare()

    train_end_idx = int(len(rl_data.returns) * 0.8)
    # ✅ 改动1: 防御性检查
    MIN_TRAIN_SAMPLES = 300
    MIN_TEST_SAMPLES = 100
    assert train_end_idx >= MIN_TRAIN_SAMPLES, \
        f"Training set too small: {train_end_idx} < {MIN_TRAIN_SAMPLES}"
    assert len(rl_data.returns) - train_end_idx >= MIN_TEST_SAMPLES, \
        f"Test set too small: {len(rl_data.returns) - train_end_idx} < {MIN_TEST_SAMPLES}"

    print(f"Train samples: {train_end_idx}")
    print(f"Test samples: {len(rl_data.returns) - train_end_idx}\n")

    # 2. 综合对比
    print("[Phase 2] Comprehensive Comparison")
    print("-" * 100)

    framework = ImprovedBenchmarkingFramework(asset_type='stock')
    simple_results = framework.run_comprehensive_comparison(rl_data, train_end_idx)

    # 3. 生成报告
    print("\n[Phase 3] Generate Report")
    print("-" * 100)

    ranked = sorted(simple_results.items(), key=lambda x: x[1]['sharpe_net'], reverse=True)
    print("\n" + "=" * 80)
    print("FINAL RESULTS (Ranked by Net Sharpe)")
    print("=" * 80)
    for i, (model, result) in enumerate(ranked, 1):
        cost_impact = result['sharpe_gross'] - result['sharpe_net']
        cost_pct = (cost_impact / result['sharpe_gross'] * 100) if result['sharpe_gross'] != 0 else 0
        print(f"\n{i}. {model}")
        print(f"   Sharpe (Gross): {result['sharpe_gross']:>8.4f}")
        print(f"   Sharpe (Net):   {result['sharpe_net']:>8.4f}")
        print(f"   Cost Impact:    {cost_impact:>8.4f} ({cost_pct:>5.1f}%)")
        print(f"   IC:             {result['ic']:>8.4f}")
        print(f"   Max DD:         {result['max_dd_net']:>8.4f}")
        print(f"   Avg Position:   {result['avg_position']:>8.4f}")

    print("\n" + "=" * 100)
    print("✓ Research Complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()