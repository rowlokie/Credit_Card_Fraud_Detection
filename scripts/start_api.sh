#!/bin/bash
# Start FastAPI with Uvicorn
exec uvicorn api.main:app \
    --host ${API_HOST:-0.0.0.0} \
    --port ${API_PORT:-8000} \
    --workers ${API_WORKERS:-1} \
    --log-level info
