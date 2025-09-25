# syntax=docker/dockerfile:1

FROM python:3.10-slim

# Python logging and no .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m appuser

# --- OS deps ---
# libgomp1 is required by xgboost; curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy and install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo (app.py, artifacts/, start.sh, etc.)
COPY . /app
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# Make entrypoint executable
RUN chmod +x /app/start.sh

# Non-root
USER appuser

# Spaces provide $PORT; keep a default for local runs
ENV PORT=7860 HOST=0.0.0.0 ARTIFACTS_DIR=artifacts

# Optional: document the port
EXPOSE 7860

# Healthcheck hits your /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

# Launch via gunicorn
ENTRYPOINT ["/app/start.sh"]
