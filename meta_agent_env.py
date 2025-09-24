# meta_agent_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import random

import config
# <<< BELANGRIJK: We importeren de correcte, robuuste backtest-functie >>>
from backtester import run_walk_forward_analysis 

class MetaAgentEnv(gym.Env):
    def __init__(self, all_historical_data, market_regime_model, causality_map, product_ids):
        super(MetaAgentEnv, self).__init__()

        self.all_historical_data = all_historical_data
        self.market_regime_model = market_regime_model
        self.causality_map = causality_map
        self.product_ids = product_ids
        self.num_assets = len(product_ids)

        # Actieruimte:
        # Actie 0: Kies strategie (0=ZIJWAARTS, 1=TREND)
        # Actie 1 tot N: Risico-allocatie per asset (hier gesimplificeerd)
        self.action_space = spaces.Discrete(2) # 0 for ZIJWAARTS, 1 for TREND

        # Observatieruimte:
        # 1 (HMM Regime) + N (Sharpe Ratios) + N*N (Correlatie) + N (Causale Scores)
        obs_shape = 1 + self.num_assets + (self.num_assets ** 2) + self.num_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        # Start op een willekeurig punt in de data (minimaal 180 dagen van het einde)
        self.start_tick = 0
        self.end_tick = min(len(df) for df in self.all_historical_data.values()) - 180 * 24 # 180 dagen aan uur-data
        self.current_step = self.start_tick

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = random.randint(self.start_tick, self.end_tick)
        return self._get_observation(), {}

    def step(self, action):
        strategy_choice = action
        strategy_map = {0: "ZIJWAARTS", 1: "TREND"}
        chosen_strategy = strategy_map.get(strategy_choice, "TREND")

        logging.info(f"Stap {self.current_step}: Grootmeester kiest strategie '{chosen_strategy}'...")

        # Voer een backtest uit voor de gekozen strategie over de volgende periode
        # We simuleren een "maand" (30 dagen)
        period_df_dict = {}
        start_index = self.current_step
        end_index = start_index + (30 * 24) # 30 dagen aan uur-data

        for symbol, df in self.all_historical_data.items():
            if end_index < len(df):
                period_df_dict[symbol] = df.iloc[start_index:end_index]

        # De beloning is de gemiddelde Sharpe ratio over alle assets voor de gekozen strategie
        total_sharpe = 0
        num_tested = 0
        for symbol, df_period in period_df_dict.items():
            # Gebruik de default config, we testen alleen de strategie-keuze
            results = run_walk_forward_analysis(
                full_df=df_period,
                strategy_name=chosen_strategy,
                strategy_config=config.STRATEGY_CONFIG,
                training_days=20, testing_days=10, step_days=10, # Kortere WFA binnen de maand
                sl_atr_multiplier=2.0, tp_atr_multiplier=4.0 # Standaard waarden
            )
            total_sharpe += results.get('avg_sharpe', -1)
            num_tested += 1
        
        reward = total_sharpe / num_tested if num_tested > 0 else -1

        logging.info(f"Strategie '{chosen_strategy}' resulteerde in een gemiddelde Sharpe van {reward:.2f}. Beloning = {reward:.2f}")

        # Ga naar de volgende periode
        self.current_step += (30 * 24)
        done = self.current_step >= self.end_tick
        
        obs = self._get_observation()
        
        return obs, reward, done, False, {}

    def _get_observation(self):
        # Bouw de observatie op voor de huidige tijdstap
        obs_df_dict = {s: df.iloc[:self.current_step] for s, df in self.all_historical_data.items()}

        # 1. HMM Regime
        main_asset_df = next(iter(obs_df_dict.values()))
        regime = self.market_regime_model.predict(main_asset_df) if not main_asset_df.empty else 0

        # 2. Sharpe Ratios (gesimuleerd, in een live versie zou dit de TradeMonitor zijn)
        sharpe_ratios = np.random.randn(self.num_assets)

        # 3. Correlatie Matrix
        try:
            returns_df = pd.DataFrame({pid: df['close'].pct_change() for pid, df in obs_df_dict.items()}).iloc[-96:]
            correlation_matrix = returns_df.corr().fillna(0)
            flat_correlation = correlation_matrix.values.flatten()
        except Exception:
            flat_correlation = np.zeros(self.num_assets ** 2)
            
        # 4. Causale Scores (gesimuleerd)
        causal_scores = np.random.randn(self.num_assets)
        
        # Combineer alles
        obs_list = np.concatenate(([regime], sharpe_ratios, flat_correlation, causal_scores)).astype(np.float32)
        return np.nan_to_num(obs_list)

    def render(self, mode='human'):
        pass