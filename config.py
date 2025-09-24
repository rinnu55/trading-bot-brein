# config.py (Definitieve Versie 3.0)

import os
import json
import logging

# --- KERN CONFIGURATIE ---
LIVE_TRADING_ENABLED = True
INITIAL_CAPITAL = 400.0  # Startkapitaal in EUR, gebaseerd op uw laatste dashboard status

# --- BELANGRIJK: URL VOOR HET 'BREIN' VAN DE BOT ---
# Vul hier de URL in die u krijgt na het deployen van grootmeester_api.py
GROOTMEESTER_API_URL = "https://grootmeester-api-uniek.onrender.com" # VOORBEELD URL

# --- API CONFIGURATIE ---
API_CONFIG = {
    "api_key_name": os.getenv("COINBASE_API_KEY_NAME", "organizations/d437dfd4-2844-4d20-b9a0-00ae186716ae/apiKeys/86bcb20e-e5bb-4e15-92cb-f5abb7a58503"),
    "api_secret_pem": os.getenv("COINBASE_API_SECRET_PEM", """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIIlRDq8lQsRvk1KgkyF7EKGEEYOjiSxImJRN1o95PtGzoAoGCCqGSM49
AwEHoUQDQgAEf0rN2RbkElMEkJCWD2b5AzfhuTAS2PxYngNZ0ynVm8lfvydf2Yzg
+VAEwFpNBGac3D0HhK6d6lPJ2qzpsjpf2w==
-----END EC PRIVATE KEY-----"""),
}

# --- NOTIFICATIE CONFIGURATIE ---
TELEGRAM_CONFIG = {
    "token": os.getenv("TELEGRAM_TOKEN", "7838264563:AAHdfmB3bO_BIma_MTagdJvUYdD6G18kPIw"),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID", "1251059883")
}

# --- DATA CONFIGURATIE ---
DATA_CONFIG = {
    "product_ids": ["BTC-EUR", "ETH-EUR", "SOL-EUR", "ADA-EUR"],
    "granularity": "FIFTEEN_MINUTE",
    "training_data_dir": "training_data",
}

# --- MODEL CONFIGURATIE ---
MODEL_CONFIG = {
    "meta_agent_path": "meta_agent_ppo.zip",
    "hmm_model_path": "hmm_regime_model.pkl",
}

# --- DRL FEATURE CONFIGURATIE (MET ORAKEL FEATURES) ---
DRL_FEATURE_NAMES = [
    # Technische Indicatoren & Basis Features
    'close_norm', 'RSI_14', 'ADX_14', 'MACDh_12_26_9', 'BBP_20_2.0',
    'ATRr_14', 'hour_norm', 'long_distance_pct', 'short_distance_pct',
    'large_swap_signal', 'predicted_change_pct', 'in_position_feature',
    
    # HMM Marktregime (One-Hot Encoded)
    'regime_0', 'regime_1', 'regime_2', 'regime_3',
    
    # Orakel Features
    'net_exchange_flow',
    'meme_sentiment_score',
    'meme_concept_score'
]

# --- STRATEGIE & RISICO CONFIGURATIE ---
# Deze parameters worden geladen, maar de Grootmeester AI zal ze overschrijven.
STRATEGY_PARAMETERS = {
    "DEFAULT": {
        "TREND": { "adx_period": 14, "atr_period_sltp": 14, "sl_atr_multiplier": 2.0, "tp_atr_multiplier": 5.0 },
        "ZIJWAARTS": { "rsi_period": 14, "bb_period": 20, "bb_std_dev": 2.0, "sl_atr_multiplier": 1.5, "tp_atr_multiplier": 3.0 },
        "CROSSOVER": { "ema_short_period": 12, "ema_long_period": 26, "sl_atr_multiplier": 2.0, "tp_atr_multiplier": 4.0 }
    }
}

RISK_CONFIG = {
    "max_open_positions": 3,
    "max_total_risk_pct": 8.0,
    "kelly_fraction": 0.75,
}

PORTFOLIO_CONFIG = {
    "dust_threshold_eur": 1.0,
    "commission_pct": 0.4
}

def get_strategy_params():
    params = STRATEGY_PARAMETERS.copy()
    opt_path = 'optimized_parameters.json'
    if os.path.exists(opt_path):
        try:
            with open(opt_path, 'r') as f:
                optimized = json.load(f)
            for asset, strategies in optimized.items():
                if asset not in params: params[asset] = {}
                for strategy, values in strategies.items():
                    if strategy not in params[asset]: params[asset][strategy] = params["DEFAULT"][strategy].copy()
                    params[asset][strategy].update(values)
            logging.info("Geoptimaliseerde parameters succesvol geladen.")
        except Exception as e:
            logging.error(f"Fout bij laden 'optimized_parameters.json': {e}")
    else:
        logging.warning("'optimized_parameters.json' niet gevonden, gebruikt standaard parameters.")
    return params