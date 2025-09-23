# cultural_compass.py

import logging
import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class CulturalCompass:
    """
    Analyseert de 'memetische energie' en het visuele sentiment van de markt
    met behulp van OpenAI's CLIP model.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = None
        self.processor = None
        try:
            logging.info(f"Laden van CLIP model: {model_name}...")
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logging.info(f"CulturalCompass geïnitialiseerd op device: {self.device}")
        except Exception as e:
            logging.error(f"Kon CLIP model niet laden. Zorg dat 'transformers', 'torch' en 'Pillow' geïnstalleerd zijn. Fout: {e}")

    def _scrape_latest_meme_image_url(self, subreddit="wallstreetbets"):
        """Simuleert het scrapen van de URL van de nieuwste afbeelding van een subreddit."""
        try:
            # We gebruiken de .json endpoint van Reddit om te scrapen zonder complexe bibliotheek
            headers = {'User-Agent': 'Mozilla/5.0'}
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=5"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Zoek de eerste post met een afbeelding
            for post in data['data']['children']:
                post_data = post['data']
                if post_data.get('post_hint') == 'image':
                    return post_data['url']
            return None
        except Exception as e:
            logging.error(f"Kon afbeelding niet scrapen van r/{subreddit}: {e}")
            return None

    def analyze_market_meme_sentiment(self) -> dict:
        """
        Scrapet de nieuwste meme en analyseert deze op financieel sentiment en concepten.
        """
        if not self.model:
            return {"sentiment_score": 0.0, "concept_score": 0.0}

        image_url = self._scrape_latest_meme_image_url()
        if not image_url:
            return {"sentiment_score": 0.0, "concept_score": 0.0}

        try:
            image = Image.open(requests.get(image_url, stream=True).raw)
            
            # Definieer de teksten waartegen we de afbeelding testen
            sentiment_prompts = ["a bullish financial meme", "a bearish financial meme", "a neutral market chart"]
            concept_prompts = ["diamond hands, to the moon, stonks", "paper hands, stock market crash, fear"]
            
            # Verwerk de inputs
            inputs = self.processor(text=sentiment_prompts + concept_prompts, images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Bereken de waarschijnlijkheden
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
            # Interpreteer de resultaten
            bullish_prob, bearish_prob, _ = probs[:3]
            diamond_hands_prob, paper_hands_prob = probs[3:]

            # Bereken een gewogen score
            sentiment_score = (bullish_prob - bearish_prob)
            concept_score = (diamond_hands_prob - paper_hands_prob)

            logging.info(f"Meme Analyse: Sentiment Score = {sentiment_score:.2f}, Concept Score = {concept_score:.2f}")
            
            return {
                "sentiment_score": sentiment_score,
                "concept_score": concept_score
            }

        except Exception as e:
            logging.error(f"Fout tijdens analyseren van afbeelding van URL {image_url}: {e}")
            return {"sentiment_score": 0.0, "concept_score": 0.0}