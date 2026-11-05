# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.9.7 AS uv
FROM python:3.12.12-slim-bookworm AS dependencies

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

FROM python:3.12.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    HOME=/tmp/music-store \
    PATH=/opt/venv/bin:$PATH
RUN groupadd --gid 10001 music-store \
    && useradd --uid 10001 --gid music-store --no-create-home music-store
WORKDIR /app
COPY --from=dependencies /opt/venv /opt/venv
# Retain the source-relative public data path; no editable build or test tree.
COPY src ./src
COPY web ./web
COPY data/sample ./data/sample
COPY .streamlit/config.toml ./.streamlit/config.toml
USER 10001:10001

FROM runtime AS api
EXPOSE 8000
CMD ["uvicorn", "music_store_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--no-proxy-headers", "--timeout-graceful-shutdown", "180"]

FROM runtime AS web
EXPOSE 8501
CMD ["streamlit", "run", "web/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.runOnSave=false", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
