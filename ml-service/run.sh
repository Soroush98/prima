#!/usr/bin/env bash
# Start the OmniAnomaly anomaly-detection service.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating Python 3.12 venv with uv..."
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi

exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port "${ML_PORT:-8000}"
