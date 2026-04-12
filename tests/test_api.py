from __future__ import annotations

import queue
import threading
import time

from fastapi.testclient import TestClient

from src.api.main import app


def receive_json_with_timeout(websocket, timeout: float = 1.0) -> dict:
    result: queue.Queue[tuple[str, dict | BaseException]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result.put(("ok", websocket.receive_json()))
        except BaseException as exc:  # pragma: no cover - test helper
            result.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError("Timed out waiting for websocket event.")

    status, payload = result.get_nowait()
    if status == "error":
        raise payload
    return payload


def test_health_and_preset_metadata() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        presets = client.get("/api/v1/meta/presets")
        assert presets.status_code == 200

        payload = presets.json()
        assert [item["key"] for item in payload] == ["xor", "sine"]
        assert payload[0]["default_structure"] == [2, 2, 1]
        assert payload[1]["sample_count"] == 701


def test_session_configure_and_step_flow() -> None:
    with TestClient(app) as client:
        configure = client.post(
            "/api/v1/session/configure",
            json={
                "preset": "xor",
                "structure": [2, 3, 1],
                "learning_rate": 0.123,
            },
        )
        assert configure.status_code == 200
        assert configure.json()["topology"]["structure"] == [2, 3, 1]

        state = client.get("/api/v1/session/state")
        assert state.status_code == 200
        assert state.json()["topology"]["learning_rate"] == 0.123

        forward = client.post("/api/v1/session/step/forward", json={})
        assert forward.status_code == 200
        assert forward.json()["training"]["phase"] == "forward_done"

        backward = client.post("/api/v1/session/step/backward")
        assert backward.status_code == 200
        assert backward.json()["training"]["phase"] == "backward_done"
        assert backward.json()["training"]["current_sample_index"] == 1

        epoch = client.post("/api/v1/session/step/epoch", json={"count": 2})
        assert epoch.status_code == 200
        assert epoch.json()["training"]["current_epoch"] == 2
        assert len(epoch.json()["training"]["epoch_losses"]) == 2


def test_run_pause_and_websocket_stream() -> None:
    with TestClient(app) as client:
        events: list[dict] = []

        with client.websocket_connect("/api/v1/session/ws") as websocket:
            initial_event = websocket.receive_json()
            events.append(initial_event)
            assert initial_event["type"] == "state"

            run_response = client.post("/api/v1/session/run", json={"speed_ms": 10})
            assert run_response.status_code == 200
            assert run_response.json()["training"]["mode"] == "run"

            deadline = time.time() + 0.5
            while time.time() < deadline and len(events) < 4:
                try:
                    events.append(receive_json_with_timeout(websocket, timeout=0.1))
                except TimeoutError:
                    pass

            pause_response = client.post("/api/v1/session/pause")
            assert pause_response.status_code == 200

            deadline = time.time() + 0.5
            while time.time() < deadline:
                try:
                    event = receive_json_with_timeout(websocket, timeout=0.1)
                    events.append(event)
                    if event["type"] == "state" and event["data"]["training"]["mode"] == "inspect":
                        break
                except TimeoutError:
                    pass

        event_types = {event["type"] for event in events}
        assert "summary" in event_types
        assert events[-1]["type"] == "state"
        assert events[-1]["data"]["training"]["mode"] == "inspect"
