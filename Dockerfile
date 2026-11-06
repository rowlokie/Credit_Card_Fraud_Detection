# =============================================================================
# Multi-stage Dockerfile for Credit Card Fraud Detection
# =============================================================================
# Build: docker build -t fraud-detection .
# Run API: docker run -p 8000:8000 fraud-detection uvicorn api.main:app --host 0.0.0.0
# Run Streamlit: docker run -p 8501:8501 fraud-detection streamlit run app.py

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 1: Install dependencies
# ---------------------------------------------------------------------------
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Production image
# ---------------------------------------------------------------------------
FROM dependencies AS production

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY config.yaml .
COPY app.py .
COPY train.py .
COPY predict.py .

# Copy model artifacts (if present at build time)
COPY models/ ./models/

# Create required directories
RUN mkdir -p logs outputs/figures data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Default command: run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
