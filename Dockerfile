FROM python:3.11-slim

# System deps — minimal; we no longer need build-essential for torch compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create persistent data directories
RUN mkdir -p data/chroma_guidelines data/guideline_cache

# Seed demo database on build so first visitors see real data
# init_db() must run first to create tables; reset_demo_data.py fills them.
RUN python -c "from api.main import init_db; init_db()" && \
    python scripts/reset_demo_data.py

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# IMPORTANT: --workers 1 on Railway free/trial tier (1 GB RAM cap).
# Each worker duplicates the entire Python memory footprint including
# the full CrewAI+LangGraph+ChromaDB stack. 2 workers → ~2.5 GB → OOM kill.
# 1 worker + FastAPI's async threadpool is plenty for ~100 concurrent users.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1