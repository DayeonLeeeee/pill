FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 BUNDLE_ROOT=/app/PillModels4 DATA_DIR=/app/data DEVICE=cpu

COPY server/requirements.txt server/
RUN pip install --no-cache-dir -r server/requirements.txt

COPY PillModels4 PillModels4
COPY server server

WORKDIR /app/server
CMD python fetch_weights.py && uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
