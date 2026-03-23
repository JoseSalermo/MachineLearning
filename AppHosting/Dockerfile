FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV MODEL_DIR=/app/model_artifacts

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY model_artifacts ./model_artifacts

EXPOSE 8000

CMD ["uvicorn", "salary_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
