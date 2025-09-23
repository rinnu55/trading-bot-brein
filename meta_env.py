# meta_env.py (Versie met Correlatie Matrix & Causale Features)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

class MetaEnv(gym.Env):
    def __init__(self, dataframes: dict, performance_data: dict, causality_data: dict, initial_capital=1000):
        super(MetaEnv, self).__init__()

        self.dataframes = dataframes
        self.performance_data = performance_data
        # --- NIEUW: Causale data wordt nu meegegeven ---
        self.causality_data = causality_data
        
        self.product_ids = list(dataframes.keys())
        self.initial_capital = initial_capital
        
        num_assets = len(self.product_ids)
        
        # Actieruimte blijft hetzelfde
        low_bounds = [-1.0] + [0.0] * num_assets
        high_bounds = [1.0] + [1.0] * num_assets
        self.action_space = spaces.Box(low=np.array(low_bounds), high=np.array(high_bounds), dtype=np.float32)

        # --- AANPASSING: Observatieruimte wordt NOG groter ---
        # [markt regime (1)] + [sharpe per asset (N)] + [correlatie (N*N)] + [causale score per asset (N)]
        correlation_matrix_size = num_assets * num_assets
        num_causal_features = num_assets
        num_obs_features = 1 + num_assets + correlation_matrix_size + num_causal_features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs_features,), dtype=np.float32)

        self.df_full_length = len(self.dataframes[self.product_ids[0]])
        self.current_step = 0
        self.max_steps = self.df_full_length - 1

    def _get_obs(self):
        if self.current_step >= self.max_steps:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        current_data = {pid: df.iloc[self.current_step] for pid, df in self.dataframes.items()}
        
        adx_values = [data['ADX_14'] for data in current_data.values()]
        sharpe_ratios = [self.performance_data[pid]['TREND']['sharpe_ratio'] for pid in self.product_ids]
        avg_adx = np.mean(adx_values)
        regime = 1 if avg_adx > 25 else 0

        start_idx = max(0, self.current_step - 24)
        returns_df = pd.DataFrame({
            pid: df['close'].iloc[start_idx:self.current_step + 1].pct_change()
            for pid, df in self.dataframes.items()
        })
        correlation_matrix = returns_df.corr().fillna(0)
        flat_correlation = correlation_matrix.values.flatten()

        # --- NIEUW: Bereken de causale scores ---
        causal_scores = []
        for pid in self.product_ids:
            score = self.causality_data[pid].iloc[self.current_step] if pid in self.causality_data else 0
            causal_scores.append(score)

        # Combineer alle observaties
        obs = np.concatenate(([regime], sharpe_ratios, flat_correlation, causal_scores)).astype(np.float32)
        return np.nan_to_num(obs)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action_vector):
        action_value = action_vector[0]
        if action_value < -0.33: strategy_name = "ZIJWAARTS"
        elif action_value < 0.33: strategy_name = "TREND"
        else: strategy_name = "CROSSOVER"
        
        risk_allocations_raw = action_vector[1:]
        risk_sum = np.sum(risk_allocations_raw)
        risk_allocations = risk_allocations_raw / risk_sum if risk_sum > 0 else np.full_like(risk_allocations_raw, 1.0 / len(self.product_ids))

        total_reward = 0
        for i, product_id in enumerate(self.product_ids):
            risk_for_asset = risk_allocations[i]
            pnl_for_this_step = self.performance_data[product_id][strategy_name]['pnl_series'].iloc[self.current_step]
            total_reward += pnl_for_this_step * risk_for_asset

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        return self._get_obs(), total_reward, terminated, False, {}