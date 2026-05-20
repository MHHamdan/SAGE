# Dockerfile for SAGE (Stabilize, Assess, Govern, Enforce)
# Reproducible environment for paper experiments

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Create directories for results
RUN mkdir -p /app/results /app/logs

# Set default environment variables
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    SEED=42 \
    LOG_LEVEL=INFO

# Default command: run all experiments
CMD ["./run_all.sh"]

# Alternative entry points:
# docker run sage-framework python -m sage.core.experiment
# docker run sage-framework pytest tests/
