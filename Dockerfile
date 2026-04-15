FROM python:3.11-slim

# System deps — minimal set only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install CPU-only PyTorch FIRST (saves ~1.5GB vs default CUDA build) ──────
# Default torch pulls CUDA build (~2.5GB). CPU-only is ~800MB.
# sentence-transformers only needs CPU inference — no GPU required.
RUN pip install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining dependencies ───────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── Create persistent data directories ───────────────────────────────────────
RUN mkdir -p data/chroma_guidelines data/guideline_cache

# ── Seed demo database ────────────────────────────────────────────────────────
RUN python -c "from api.main import init_db; init_db()" && \
    python scripts/reset_demo_data.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]