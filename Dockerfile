FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements-service.txt .
RUN pip install -r requirements-service.txt

COPY wsignals ./wsignals
COPY api ./api
COPY ui ./ui
COPY models ./models
COPY data/labels ./data/labels
COPY data/dataset ./data/dataset
COPY reports ./reports

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
