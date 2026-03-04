FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/app/.cache/huggingface

# Create application directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install the application and dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Copy web app requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY app.py vllm_start_config_from_estimate.py ./
COPY templates/ ./templates/

# OpenShift compatibility: Grant directory permissions to root group
RUN mkdir -p /app/.cache/huggingface && \
    chgrp -R 0 /app && \
    chmod -R g=u /app

# Ensure non-root user for security (OpenShift requires arbitrary UIDs to have access)
USER 1001

EXPOSE 8080

# Run the web service using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "600", "app:app"]
