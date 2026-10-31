"""Minimal FastAPI application with an independent liveness endpoint."""

from fastapi import FastAPI

app = FastAPI(title="Music Store AI Agent")


@app.get("/health")
def health() -> dict[str, str]:
    """Report application liveness without connecting to external services.

    Returns:
        The application status.
    """
    return {"status": "ok"}
