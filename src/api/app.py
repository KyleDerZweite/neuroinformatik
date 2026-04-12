"""FastAPI application factory for the Neuroinformatik API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.health import router as health_router
from src.api.routes.meta import router as meta_router
from src.api.routes.session import router as session_router
from src.api.services.session import SessionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down shared application services."""
    service = SessionService()
    await service.initialize()
    app.state.session_service = service
    try:
        yield
    finally:
        await service.shutdown()


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Neuroinformatik API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(meta_router)
    app.include_router(session_router)
    return app
