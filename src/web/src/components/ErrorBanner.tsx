import { AlertTriangle, X } from "lucide-react";
import { useEffect } from "react";

import { useSession } from "../lib/session/SessionProvider";

export function ErrorBanner() {
  const { lastError, clearError } = useSession();

  useEffect(() => {
    if (!lastError) return;
    const timer = window.setTimeout(clearError, 6000);
    return () => window.clearTimeout(timer);
  }, [lastError, clearError]);

  if (!lastError) return null;

  return (
    <div className="error-overlay" role="alert">
      <AlertTriangle size={12} strokeWidth={1.6} />
      <span className="error-label">fault</span>
      <span className="error-separator">·</span>
      <span className="error-message">{lastError}</span>
      <button
        type="button"
        aria-label="Dismiss"
        className="error-dismiss"
        onClick={clearError}
      >
        <X size={12} strokeWidth={1.6} />
      </button>
    </div>
  );
}
