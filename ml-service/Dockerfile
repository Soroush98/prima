# Prima ML ensemble — FastAPI + PyTorch (Donut-VAE) + Chronos-Bolt forecaster.
# CPU-only torch keeps the image to a sane size (the GPU wheels are ~2GB larger).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Cache HuggingFace/Chronos weights on the writable layer.
    HF_HOME=/app/.hfcache

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install the CPU build of torch explicitly, then the rest (torch>=2.4 is then satisfied).
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" \
 && pip install -r requirements.txt

COPY . .

EXPOSE 8000
# Bind 0.0.0.0 (run.sh uses 127.0.0.1, which is unreachable from the web app on Fly).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
