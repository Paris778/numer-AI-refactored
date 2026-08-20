# Dataless CI-reproduction image (Linux, pinned deps).
#
# Catches the failure class that a local run structurally cannot: tests that
# pass ONLY because data/v5.3 exists on the dev box. `git archive HEAD` feeds
# the repo (no gitignored data), so the container sees exactly what CI sees.
#
# Build once (reusable, ~13.5 GB):
#   docker build -f scripts/ci_repro.Dockerfile -t ci-repro .
# Run the exact CI test step:
#   git archive HEAD | docker run -i --rm ci-repro bash -c \
#     "mkdir -p /work && tar x -C /work && cd /work && \
#      python -m pytest -q -rs --cov=nmr --cov=dashboard_ui --cov-branch \
#      --cov-report=term-missing --cov-report=json"
#
# Requires Docker Desktop (Windows). This is the pre-sign-off dataless gate
# documented in CONTRIBUTING.md; do not treat local green as sufficient for
# changes touching data loading, campaign orchestration, or test fixtures.

FROM python:3.12-slim

# LightGBM's native lib needs the GNU OpenMP runtime; slim images lack it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt /req/
RUN pip install -q -r /req/requirements.txt -r /req/requirements-dev.txt
