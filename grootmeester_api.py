# grootmeester_api.py

from flask import Flask, request, jsonify
import logging
import pandas as pd
import numpy as np
import os
import sys

# Voeg de projectmap toe aan het Python-pad
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from market_regime_model import MarketRegimeModel
from onchain_analyzer import OnChainAnalyzer
from cultural_compass import CulturalCompass
from meta_agent_env import MetaAgentEnv  # We hergebruiken de logica

# --- Initialisatie ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - API - %(message)s")
app = Flask(__name__)

# Laad de modellen bij het opstarten van de API
try:
    META_AGENT_MODEL = PPO.load("meta_agent_ppo.zip")
    HMM_MODEL = MarketRegimeModel()
    HMM_MODEL.load_model("hmm_regime_model.pkl")
    ONCHAIN_ANALYZER = OnChainAnalyzer()
    CULTURAL_COMPASS = CulturalCompass()
    logging.info("✅ Alle modellen en analyzers succesvol geladen in de API.")
except Exception as e:
    logging.critical(f"❌ Fout bij het laden van de modellen: {e}")
    META_AGENT_MODEL = None

@app.route('/decide', methods=['POST'])
def decide():
    if not META_AGENT_MODEL:
        return jsonify({"error": "Modellen zijn niet geladen"}), 500

    try:
        # Haal de data uit het verzoek van de lokale bot
        data = request.json
        market_data_json = data['market_data']
        
        # Converteer de JSON data terug naar een DataFrame
        df = pd.read_json(market_data_json, orient='split')
        
        # --- Simuleer de observatie, net als in de training ---
        current_regime = HMM_MODEL.predict(df)
        onchain_features = ONCHAIN_ANALYZER.get_onchain_features('BTC')
        cultural_features = CULTURAL_COMPASS.analyze_market_meme_sentiment()
        
        regime_one_hot = np.zeros(4, dtype=np.float32)
        if current_regime != -1:
            regime_one_hot[current_regime] = 1.0

        obs = np.concatenate([
            regime_one_hot,
            [onchain_features.get('net_exchange_flow', 0.0)],
            [cultural_features.get('sentiment_score', 0.0)],
            [cultural_features.get('concept_score', 0.0)]
        ]).astype(np.float32)
        obs = np.nan_to_num(obs)

        # --- Laat het AI-model de beslissing nemen ---
        action, _ = META_AGENT_MODEL.predict(obs, deterministic=True)
        strategy_choice_val, risk_factor = action

        # Map de actie naar een duidelijke strategie
        if strategy_choice_val < -0.33: strategy_name = "ZIJWAARTS"
        elif strategy_choice_val < 0.33: strategy_name = "TREND"
        else: strategy_name = "CROSSOVER"

        # Stuur de beslissing terug naar de lokale bot
        return jsonify({
            "strategy": strategy_name,
            "risk_factor": float(risk_factor)
        })

    except Exception as e:
        logging.error(f"Fout in /decide endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Gebruik 'gunicorn' of een andere productieserver om dit te draaien
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))