# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.9.7 AS uv
FROM python:3.12.12-slim-bookworm

COPY --from=uv /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

COPY src ./src
COPY tests ./tests
COPY data/sample ./data/sample
COPY web ./web
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "music_store_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--reload", "--reload-dir", "src"]
