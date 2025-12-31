# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ENVIRONMENT=production

# Install system dependencies
RUN apt-get update && apt-get install -y apt-utils && \
    apt-get install -y \
    curl unzip gcc g++ git wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache utilization
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-eval.txt .
RUN pip install --no-cache-dir -r requirements-eval.txt

# Copy the rest of the application
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY eval/ ./eval/

# Create non-root user and switch to it
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Set environment variables for workers/threads
ENV WORKERS=${WORKERS:-4} \
    THREADS=${THREADS:-4} \
    TIMEOUT=${TIMEOUT:-300}

# Run the application with Gunicorn
CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-connections", "7", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300", "--graceful-timeout", "30"]

