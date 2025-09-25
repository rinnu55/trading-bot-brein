

from flask import Flask, request, jsonify
import logging
import pandas as pd
import numpy as np
import os


try:
    from stable_baselines3 import PPO
    from market_regime_model import MarketRegimeModel
    from onchain_analyzer import OnChainAnalyzer
    from cultural_compass import CulturalCompass
except ImportError as e:
    raise ImportError(f"Kon een bibliotheek niet importeren. Zorg dat requirements_api.txt correct is. Fout: {e}")

# --- Initialisatie ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - API - %(message)s")
app = Flask(__name__)

META_AGENT_MODEL = None
HMM_MODEL = None
ONCHAIN_ANALYZER = None
CULTURAL_COMPASS = None

try:
    if os.path.exists("meta_agent_ppo.zip"):
        META_AGENT_MODEL = PPO.load("meta_agent_ppo.zip", device='cpu')
        logging.info("✅ Grootmeester AI model geladen.")
    else:
        logging.error("❌ 'meta_agent_ppo.zip' niet gevonden!")

    if os.path.exists("hmm_regime_model.pkl"):
        HMM_MODEL = MarketRegimeModel()
        HMM_MODEL.load_model("hmm_regime_model.pkl")
        logging.info("✅ Marktregime model geladen.")
    else:
        logging.error("❌ 'hmm_regime_model.pkl' niet gevonden!")
        
    ONCHAIN_ANALYZER = OnChainAnalyzer()
    CULTURAL_COMPASS = CulturalCompass()
    logging.info("✅ Analyzers succesvol geladen in de API.")
except Exception as e:
    logging.critical(f"❌ Fout bij het laden van de modellen tijdens opstarten: {e}")

@app.route('/decide', methods=['POST'])
def decide():
    if not all([META_AGENT_MODEL, HMM_MODEL]):
        return jsonify({"error": "Een of meerdere modellen zijn niet correct geladen"}), 500

    try:
        data = request.json
        market_data_json = data['market_data']
        df = pd.read_json(market_data_json, orient='split')
        df.index = pd.to_datetime(df.index, unit='ms')

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

        action, _ = META_AGENT_MODEL.predict(obs, deterministic=True)
        strategy_choice_val, risk_factor = action

        if strategy_choice_val < -0.33: strategy_name = "ZIJWAARTS"
        elif strategy_choice_val < 0.33: strategy_name = "TREND"
        else: strategy_name = "CROSSOVER"

        return jsonify({
            "strategy": strategy_name,
            "risk_factor": float(risk_factor)
        })

    except Exception as e:
        logging.error(f"Fout in /decide endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))