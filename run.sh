#!/bin/bash
# Dit script stelt de omgeving correct in en start de API.

# Stel de PYTHONPATH expliciet in zodat alle .py bestanden gevonden worden.
export PYTHONPATH=/opt/render/project/src

# Start de Gunicorn webserver op de door Render opgegeven poort.
gunicorn grootmeester_api:app --bind 0.0.0.0:${PORT}