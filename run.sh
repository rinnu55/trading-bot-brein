#!/bin/bash
# Start de Gunicorn webserver met een expliciet configuratiebestand.
gunicorn -c gunicorn_config.py grootmeester_api:app --bind 0.0.0.0:${PORT}