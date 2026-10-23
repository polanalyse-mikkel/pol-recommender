# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

WORKDIR /app

COPY ./pyproject.toml ./uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY recommender/ recommender/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "recommender"]
