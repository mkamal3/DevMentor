"""FastAPI scaffold for DevMentor backend."""

from fastapi import FastAPI

from config.settings import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Basic health endpoint."""
    return {"status": "ok", "service": settings.app_name}
