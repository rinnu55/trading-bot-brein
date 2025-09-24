# gunicorn_config.py
import sys
import os

# Voeg de projectdirectory toe aan het Python-pad.
# Dit zorgt ervoor dat alle modules (config, onchain_analyzer, etc.) gevonden kunnen worden.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))