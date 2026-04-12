"""Dependency helpers for FastAPI routes."""

from __future__ import annotations

from fastapi import Request

from src.api.services.session import SessionService


def get_session_service(request: Request) -> SessionService:
    """Return the shared session service from the FastAPI app state."""
    return request.app.state.session_service
