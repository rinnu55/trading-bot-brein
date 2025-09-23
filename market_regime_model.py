# market_regime_model.py (Geüpgraded met HMM)

import logging
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import joblib
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(module)s] - %(message)s")

class MarketRegimeModel:
    """
    Identificeert marktregimes met een Hidden Markov Model (HMM).
    """
    def __init__(self, n_regimes=4, model_path="hmm_regime_model.pkl"):
        self.n_regimes = n_regimes
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self._load_model()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereidt features voor die het HMM zal gebruiken."""
        features = pd.DataFrame(index=df.index)
        features['returns'] = df['close'].pct_change().fillna(0)
        # Volatiliteit als de range van de candle
        features['volatility'] = (df['high'] - df['low']) / df['close']
        return features.dropna()

    def train(self, historical_data: pd.DataFrame):
        """Traint het HMM model en slaat het op."""
        logging.info(f"Starten van HMM training met {self.n_regimes} regimes...")
        features = self._prepare_features(historical_data)
        
        if len(features) < self.n_regimes:
            logging.error("Niet genoeg data om het HMM-model te trainen.")
            return

        self.model = GaussianHMM(n_components=self.n_regimes, covariance_type="full", n_iter=100, random_state=42)
        try:
            self.model.fit(features)
            self.is_trained = True
            joblib.dump(self.model, self.model_path)
            logging.info(f"HMM-model succesvol getraind en opgeslagen in {self.model_path}")
            self._describe_regimes(features)
        except Exception as e:
            logging.error(f"Fout tijdens trainen HMM-model: {e}")
            self.is_trained = False

    def _load_model(self):
        """Laadt een getraind HMM-model."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                logging.info(f"HMM-model succesvol geladen van {self.model_path}")
            except Exception as e:
                logging.error(f"Fout bij laden HMM-model: {e}")
        else:
            logging.warning(f"Geen HMM-model gevonden op {self.model_path}. Model moet getraind worden.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Voorspelt het regime voor een volledige dataset."""
        if not self.is_trained:
            logging.warning("HMM-model is niet getraind. Kan regimes niet voorspellen.")
            return np.zeros(len(df), dtype=int)
            
        features = self._prepare_features(df)
        return self.model.predict(features)

    def get_current_regime(self, df_latest_slice: pd.DataFrame) -> int:
        """Voorspelt het regime voor de meest recente data."""
        if not self.is_trained:
            return 0 # Fallback naar een neutraal regime
            
        features = self._prepare_features(df_latest_slice)
        if features.empty:
            return 0 # Fallback
            
        return self.model.predict(features)[-1]

    def _describe_regimes(self, features: pd.DataFrame):
        """Geeft een beschrijving van de gevonden regimes na training."""
        if not self.is_trained: return
        
        regimes = self.model.predict(features)
        features['regime'] = regimes
        
        logging.info("--- Analyse van Gevonden Marktregimes ---")
        for i in range(self.n_regimes):
            regime_data = features[features['regime'] == i]
            mean_return = regime_data['returns'].mean() * 100 
            mean_volatility = regime_data['volatility'].mean() * 100
            
            description = ""
            if mean_volatility < 0.5: description += "Lage Volatiliteit, "
            else: description += "Hoge Volatiliteit, "
            
            if abs(mean_return) < 0.05: description += "Stabiel"
            elif mean_return > 0: description += "Positief (Rally)"
            else: description += "Negatief (Crash)"

            logging.info(f"Regime {i}: {description} (Gem. Rendement: {mean_return:.4f}%, Gem. Volatiliteit: {mean_volatility:.4f}%)")
        logging.info("------------------------------------------")