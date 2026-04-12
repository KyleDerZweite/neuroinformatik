"""Session control and streaming endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from src.api.dependencies import get_session_service
from src.api.schemas import ConfigureRequest, RunRequest, SessionState, StepEpochRequest, StepForwardRequest
from src.api.services.session import SessionService

router = APIRouter(prefix="/api/v1/session", tags=["session"])


@router.get("/state", response_model=SessionState)
async def get_state(
    service: SessionService = Depends(get_session_service),
) -> SessionState:
    """Return the current full training state."""
    return await service.get_state()


@router.post("/configure", response_model=SessionState)
async def configure_session(
    request: ConfigureRequest,
    service: SessionService = Depends(get_session_service),
) -> SessionState:
    """Configure a new training session."""
    return await service.configure(request)


@router.post("/reset", response_model=SessionState)
async def reset_session(
    service: SessionService = Depends(get_session_service),
) -> SessionState:
    """Reset the current session."""
    return await service.reset()


@router.post("/step/epoch", response_model=SessionState)
async def step_epoch(
    request: StepEpochRequest,
    service: SessionService = Depends(get_session_service),
) -> SessionState:
    """Run one or more complete epochs."""
    return await service.step_epoch(request.count)


@router.post("/step/forward", response_model=SessionState)
async def step_forward(
    request: StepForwardRequest,
    service: SessionService = Depends(get_session_service),
) -> SessionState:
    """Run a single forward pass."""
    return await service.step_forward(request.sample_index)


@router.post("/step/backward", response_model=SessionState)
async def step_backward(
    service: SessionService = Depends(get_session_service),
) -> SessionState:
    """Run a single backward pass."""
    return await service.step_backward()


@router.post("/step/layer/forward", response_model=SessionState)
async def step_layer_forward(
    service: SessionService = Depends(get_session_service),
) -> SessionState:
    """Run a single layer forward."""
    return await service.step_layer_forward()


@router.post("/step/layer/backward", response_model=SessionState)
async def step_layer_backward(
    service: SessionService = Depends(get_session_service),
) -> SessionState:
    """Run a single layer backward."""
    return await service.step_layer_backward()


@router.post("/run", response_model=SessionState)
async def start_run(
    request: RunRequest,
    service: SessionService = Depends(get_session_service),
) -> SessionState:
    """Start the continuous training loop."""
    return await service.start_run(request.speed_ms)


@router.post("/pause", response_model=SessionState)
async def pause_run(
    service: SessionService = Depends(get_session_service),
) -> SessionState:
    """Pause the continuous training loop."""
    return await service.pause()


@router.websocket("/ws")
async def session_stream(websocket: WebSocket) -> None:
    """Stream state and summary events to the frontend."""
    service: SessionService = websocket.app.state.session_service
    await websocket.accept()
    queue = await service.subscribe()
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
    finally:
        await service.unsubscribe(queue)
