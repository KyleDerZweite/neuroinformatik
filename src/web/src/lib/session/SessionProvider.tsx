import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";

import { apiClient, createSessionEventSocket } from "../api/client";
import type {
  ConfigureRequest,
  DashboardEvent,
  PresetSummary,
  SessionState,
  SummaryState,
} from "../api/types";

interface SessionActions {
  configure: (body: ConfigureRequest) => Promise<void>;
  reset: () => Promise<void>;
  run: (speedMs: number) => Promise<void>;
  pause: () => Promise<void>;
  stepEpoch: (count: number) => Promise<void>;
  stepForward: (sampleIndex?: number | null) => Promise<void>;
  stepBackward: () => Promise<void>;
  stepLayerForward: () => Promise<void>;
  stepLayerBackward: () => Promise<void>;
}

interface SessionContextValue {
  connected: boolean;
  state: SessionState | null;
  presets: PresetSummary[];
  lastError: string | null;
  clearError: () => void;
  actions: SessionActions;
}

const SessionContext = createContext<SessionContextValue | null>(null);

const MAX_RECONNECT_DELAY_MS = 5000;
const INITIAL_RECONNECT_DELAY_MS = 500;

function mergeSummary(
  prev: SessionState,
  summary: SummaryState,
): SessionState {
  return {
    ...prev,
    training: summary.training,
    network_output: summary.network_output,
  };
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<SessionState | null>(null);
  const [presets, setPresets] = useState<PresetSummary[]>([]);
  const [lastError, setLastError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelayRef = useRef<number>(INITIAL_RECONNECT_DELAY_MS);
  const reconnectTimerRef = useRef<number | null>(null);
  const cancelledRef = useRef<boolean>(false);

  useEffect(() => {
    apiClient
      .getPresets()
      .then(setPresets)
      .catch((err: Error) => setLastError(err.message));
  }, []);

  useEffect(() => {
    cancelledRef.current = false;

    const connect = () => {
      if (cancelledRef.current) return;
      const ws = createSessionEventSocket();
      wsRef.current = ws;

      ws.addEventListener("open", () => {
        reconnectDelayRef.current = INITIAL_RECONNECT_DELAY_MS;
        setConnected(true);
      });

      ws.addEventListener("close", () => {
        wsRef.current = null;
        setConnected(false);
        if (cancelledRef.current) return;
        reconnectTimerRef.current = window.setTimeout(
          connect,
          reconnectDelayRef.current,
        );
        reconnectDelayRef.current = Math.min(
          Math.round(reconnectDelayRef.current * 1.6),
          MAX_RECONNECT_DELAY_MS,
        );
      });

      ws.addEventListener("error", () => {
        /* close event will drive reconnect */
      });

      ws.addEventListener("message", (event) => {
        let msg: DashboardEvent;
        try {
          msg = JSON.parse(event.data) as DashboardEvent;
        } catch {
          setLastError("Invalid WebSocket payload");
          return;
        }
        if (msg.type === "state") {
          setState(msg.data);
        } else if (msg.type === "summary") {
          setState((prev) => (prev ? mergeSummary(prev, msg.data) : prev));
        } else if (msg.type === "error") {
          setLastError(msg.message);
        }
      });
    };

    connect();

    return () => {
      cancelledRef.current = true;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  const invoke = useCallback(async (fn: () => Promise<SessionState>) => {
    try {
      const result = await fn();
      setState(result);
    } catch (err) {
      setLastError((err as Error).message);
    }
  }, []);

  const actions = useMemo<SessionActions>(
    () => ({
      configure: (body) => invoke(() => apiClient.configure(body)),
      reset: () => invoke(() => apiClient.reset()),
      run: (speedMs) => invoke(() => apiClient.run({ speed_ms: speedMs })),
      pause: () => invoke(() => apiClient.pause()),
      stepEpoch: (count) => invoke(() => apiClient.stepEpoch({ count })),
      stepForward: (sampleIndex) =>
        invoke(() =>
          apiClient.stepForward({ sample_index: sampleIndex ?? null }),
        ),
      stepBackward: () => invoke(() => apiClient.stepBackward()),
      stepLayerForward: () => invoke(() => apiClient.stepLayerForward()),
      stepLayerBackward: () => invoke(() => apiClient.stepLayerBackward()),
    }),
    [invoke],
  );

  const value = useMemo<SessionContextValue>(
    () => ({
      connected,
      state,
      presets,
      lastError,
      clearError: () => setLastError(null),
      actions,
    }),
    [connected, state, presets, lastError, actions],
  );

  return (
    <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
  );
}

export function useSession(): SessionContextValue {
  const ctx = useContext(SessionContext);
  if (!ctx) {
    throw new Error("useSession must be used within SessionProvider");
  }
  return ctx;
}
