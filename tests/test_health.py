"""Tests for the API liveness endpoint without external services."""

from fastapi.testclient import TestClient

from music_store_agent.api.main import app


def test_health_endpoint() -> None:
    """Serve a successful JSON health response without a database connection."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"status": "ok"}
