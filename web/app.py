"""Streamlit entry point for the customer support interface."""

import httpx
import streamlit as st
from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings


class FrontendSettings(BaseSettings):
    """Read backend connectivity settings from environment variables.

    Attributes:
        backend_url: Internal address of the FastAPI service.
        backend_timeout_seconds: Maximum duration of a backend health request.
    """

    backend_url: AnyHttpUrl = AnyHttpUrl("http://api:8000")
    backend_timeout_seconds: float = Field(default=5.0, gt=0)


def main() -> None:
    """Render the development notice and current backend connectivity status."""
    st.set_page_config(page_title="Empório da Música", page_icon="🎵")
    st.title("Empório da Música")
    st.subheader("AI Customer Support Assistant")
    st.info("The customer support interface is under development.")

    settings = FrontendSettings()
    health_url = f"{str(settings.backend_url).rstrip('/')}/health"
    try:
        response = httpx.get(health_url, timeout=settings.backend_timeout_seconds)
        response.raise_for_status()
        healthy = response.json() == {"status": "ok"}
    except (httpx.HTTPError, ValueError):
        st.warning("Backend unavailable. Please try again shortly.")
    else:
        if healthy:
            st.success("Backend connected and healthy.")
        else:
            st.warning("Backend returned an unexpected health response.")

    if st.button("Refresh backend status"):
        st.rerun()


if __name__ == "__main__":
    main()
