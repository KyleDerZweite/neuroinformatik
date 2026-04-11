"""Single-port WebSocket server serving static files and live training.

Entry point: python3 src/web/server.py
Dashboard: http://localhost:8765

Uses websockets v15 process_request hook to serve static files on the
same port as the WebSocket connection.
"""

import asyncio
import json
import mimetypes
import os
import sys
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.http11 import Response
from websockets.datastructures import Headers

# Add src/ to path so controller can import neuralnet/simulation
_src_dir = os.path.join(os.path.dirname(__file__), "..")
if _src_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_src_dir))

from web.controller import NetworkController

STATIC_DIR = Path(__file__).parent / "static"

HOST = "localhost"
PORT = 8765

# --- Static file cache (reloaded on mtime change) ---

_static_cache: dict[str, tuple[bytes, str, float]] = {}


def _serve_static(path: str) -> Response:
    """Serve a static file with simple in-memory caching."""
    if path in ("/", ""):
        path = "/index.html"

    file_path = (STATIC_DIR / path.lstrip("/")).resolve()

    # Security: prevent directory traversal
    if not str(file_path).startswith(str(STATIC_DIR.resolve())):
        return Response(403, "Forbidden", Headers(), b"Forbidden")

    if not file_path.is_file():
        return Response(404, "Not Found", Headers(), b"Not Found")

    # Check cache — re-read only if file changed
    mtime = file_path.stat().st_mtime
    cached = _static_cache.get(path)
    if cached and cached[2] == mtime:
        body, content_type, _ = cached
    else:
        body = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        _static_cache[path] = (body, content_type, mtime)

    return Response(
        200, "OK",
        Headers({"Content-Type": content_type, "Content-Length": str(len(body))}),
        body,
    )


def process_request(connection, request):
    """Route: WebSocket upgrade -> None (proceed), else serve static file."""
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None
    return _serve_static(request.path)


# --- Command dispatch table ---

async def _cmd_configure(msg, controller, run_task, **_kw):
    if run_task and not run_task.done():
        controller.pause()
        run_task.cancel()
    state = controller.configure(
        msg.get("structure", [2, 2, 1]),
        msg.get("learning_rate", 1.0),
        msg.get("preset", "xor"),
    )
    return {"type": "state", "data": state}, run_task


async def _cmd_reset(_msg, controller, run_task, **_kw):
    if run_task and not run_task.done():
        controller.pause()
        run_task.cancel()
    state = controller.reset()
    return {"type": "state", "data": state}, run_task


async def _cmd_step_epoch(msg, controller, **_kw):
    state = controller.step_epoch(count=msg.get("count", 1))
    return {"type": "state", "data": state}, _kw.get("run_task")


async def _cmd_step_forward(msg, controller, **_kw):
    state = controller.step_forward(sample_index=msg.get("sample_index"))
    return {"type": "state", "data": state}, _kw.get("run_task")


async def _cmd_step_backward(_msg, controller, **_kw):
    state = controller.step_backward()
    return {"type": "state", "data": state}, _kw.get("run_task")


async def _cmd_step_layer_forward(_msg, controller, **_kw):
    state = controller.step_layer_forward()
    return {"type": "state", "data": state}, _kw.get("run_task")


async def _cmd_step_layer_backward(_msg, controller, **_kw):
    state = controller.step_layer_backward()
    return {"type": "state", "data": state}, _kw.get("run_task")


async def _cmd_run(msg, controller, websocket, run_task, **_kw):
    if run_task and not run_task.done():
        controller.pause()
        run_task.cancel()
        await asyncio.sleep(0.05)

    async def send_callback(data):
        await websocket.send(json.dumps(data))

    task = asyncio.create_task(
        controller.run_continuous(send_callback, msg.get("speed_ms", 50))
    )
    return None, task


async def _cmd_pause(_msg, controller, run_task, **_kw):
    controller.pause()
    if run_task and not run_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(run_task), timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
    return None, run_task


async def _cmd_get_state(_msg, controller, **_kw):
    state = controller.get_state()
    return {"type": "state", "data": state}, _kw.get("run_task")


COMMANDS = {
    "configure": _cmd_configure,
    "reset": _cmd_reset,
    "step_epoch": _cmd_step_epoch,
    "step_forward": _cmd_step_forward,
    "step_backward": _cmd_step_backward,
    "step_layer_forward": _cmd_step_layer_forward,
    "step_layer_backward": _cmd_step_layer_backward,
    "run": _cmd_run,
    "pause": _cmd_pause,
    "get_state": _cmd_get_state,
}


# --- WebSocket handler ---

async def handler(websocket):
    """Handle a single WebSocket connection."""
    controller = NetworkController()
    run_task = None

    # Auto-configure with XOR so the dashboard shows something on connect
    try:
        state = controller.configure([2, 2, 1], 1.0, "xor")
        await websocket.send(json.dumps({"type": "state", "data": state}))
    except Exception as exc:
        await websocket.send(json.dumps({"type": "error", "message": str(exc)}))

    async for raw_message in websocket:
        try:
            msg = json.loads(raw_message)
        except (json.JSONDecodeError, TypeError):
            await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
            continue

        msg_type = msg.get("type")
        handler_fn = COMMANDS.get(msg_type)

        if not handler_fn:
            await websocket.send(json.dumps({
                "type": "error", "message": f"Unknown command: {msg_type}"
            }))
            continue

        try:
            response, run_task = await handler_fn(
                msg, controller=controller, websocket=websocket, run_task=run_task
            )
            if response is not None:
                await websocket.send(json.dumps(response))
        except Exception as exc:
            await websocket.send(json.dumps({"type": "error", "message": str(exc)}))

    if run_task and not run_task.done():
        run_task.cancel()


# --- Entry point ---

async def main():
    async with serve(handler, HOST, PORT, process_request=process_request):
        print(f"Dashboard: http://{HOST}:{PORT}")
        print("Press Ctrl+C to stop")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")
