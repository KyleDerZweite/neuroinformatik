"""Metadata endpoints for presets and capabilities."""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_session_service
from src.api.schemas import PresetSummary
from src.api.services.session import SessionService

router = APIRouter(prefix="/api/v1/meta", tags=["meta"])


@router.get("/presets", response_model=list[PresetSummary])
async def list_presets(
    service: SessionService = Depends(get_session_service),
) -> list[PresetSummary]:
    """Return supported presets for the frontend."""
    return await service.list_presets()
