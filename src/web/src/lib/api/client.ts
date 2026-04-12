import type {
  ConfigureRequest,
  PresetSummary,
  RunRequest,
  SessionState,
  StepEpochRequest,
  StepForwardRequest,
} from "./types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    let message = `API request failed: ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) {
        message =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
      }
    } catch {
      /* ignore parse failure */
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

function postJson<T>(path: string, body?: unknown): Promise<T> {
  return requestJson<T>(path, {
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export const apiClient = {
  getPresets: () => requestJson<PresetSummary[]>("/api/v1/meta/presets"),
  getState: () => requestJson<SessionState>("/api/v1/session/state"),
  configure: (body: ConfigureRequest) =>
    postJson<SessionState>("/api/v1/session/configure", body),
  reset: () => postJson<SessionState>("/api/v1/session/reset"),
  stepEpoch: (body: StepEpochRequest) =>
    postJson<SessionState>("/api/v1/session/step/epoch", body),
  stepForward: (body: StepForwardRequest) =>
    postJson<SessionState>("/api/v1/session/step/forward", body),
  stepBackward: () => postJson<SessionState>("/api/v1/session/step/backward"),
  stepLayerForward: () =>
    postJson<SessionState>("/api/v1/session/step/layer/forward"),
  stepLayerBackward: () =>
    postJson<SessionState>("/api/v1/session/step/layer/backward"),
  run: (body: RunRequest) => postJson<SessionState>("/api/v1/session/run", body),
  pause: () => postJson<SessionState>("/api/v1/session/pause"),
  health: () => requestJson<{ status: "ok" }>("/health"),
};

export function createSessionEventSocket(): WebSocket {
  const url = new URL("/api/v1/session/ws", API_BASE_URL);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return new WebSocket(url);
}
