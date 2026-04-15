FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
# Install CPU-only PyTorch first — prevents pip from pulling the 2.5 GB CUDA build
# when sentence-transformers is installed below. Saves ~2.3 GB in the final image.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create persistent data directories
RUN mkdir -p data/chroma_guidelines data/guideline_cache

# Seed demo database on build so first visitors see real data
RUN python -c "from api.main import init_db; init_db()" && \
    python scripts/reset_demo_data.py

# Expose FastAPI port
EXPOSE 8000

# Health check — Railway and Render use this to know when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]