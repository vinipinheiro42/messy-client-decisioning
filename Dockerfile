# Local scoring image for the synthetic Credara models.
# Build after `make train` so ml/artifacts/models.joblib is present.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ml/
COPY api/ api/
COPY docs/model_metrics.json docs/model_metrics.json
# Trained binaries are gitignored; copy them from a machine that has run make train.
COPY ml/artifacts/models.joblib ml/artifacts/models.joblib

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
