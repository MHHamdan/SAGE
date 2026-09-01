# SAGE — Stabilize, Assess, Govern, Enforce
# Reproducible environment for the IEEE TAI paper's experiments.
#
#   docker build -t sage-framework .
#   docker run --rm sage-framework                                   # test suite
#   docker run --rm -v "$PWD/results:/app/results" sage-framework \
#       python experiments/e4_closed_loop.py --seed 42
#
# No API credentials are needed for the test suite or the simulated
# experiments. To run against hosted APIs, pass credentials at run time
# (`--env-file .env`); never bake them into the image.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Pinned so hash-dependent RNG paths are stable across processes, and the
# default seed used by every experiment script.
ENV PYTHONHASHSEED=0 \
    SEED=42 \
    LOG_LEVEL=INFO

# Local open-weight backend (`--backend ollama`) reaches the host's server.
ENV OLLAMA_HOST=http://host.docker.internal:11434

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency metadata first, so the dependency layer caches across code edits.
COPY pyproject.toml requirements.txt README.md ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Editable install exposes the `sage` package and the experiment entry points.
RUN pip install --no-cache-dir -e ".[experiments]"

RUN mkdir -p /app/results /app/logs

# Default: run the test suite, which needs no credentials and no network.
CMD ["pytest", "tests/", "--tb=short", "-q"]
