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
    uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY web ./web
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

EXPOSE 8501
CMD ["uv", "run", "--no-sync", "streamlit", "run", "web/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.runOnSave=true", "--server.fileWatcherType=poll", "--browser.gatherUsageStats=false"]
