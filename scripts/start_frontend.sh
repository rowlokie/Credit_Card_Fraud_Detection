#!/bin/bash
# Start Streamlit frontend
exec streamlit run app.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0
