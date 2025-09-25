#!/usr/bin/env bash
set -e

: "${PORT:=7860}"
: "${HOST:=0.0.0.0}"

# Your Flask app exposes a module-level `app` in app.py → "app:app"
exec gunicorn app:app \
  --config gunicorn.conf.py \
  --bind "${HOST}:${PORT}"
