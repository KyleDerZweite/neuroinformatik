/**
 * Core application: WebSocket connection + state distribution.
 */

let ws = null;
let lastFullState = null;

const MAX_RECONNECT_DELAY = 5000;
let reconnectDelay = 500;

function getLastFullState() {
    return lastFullState;
}

function updateConnectionStatus(connected) {
    const el = document.getElementById("connection-status");
    if (!el) return;
    el.textContent = connected ? "Online" : "Offline";
    el.className = "conn-badge " + (connected ? "conn-online" : "conn-offline");
}

function showError(message) {
    const banner = document.getElementById("error-banner");
    if (!banner) return;
    banner.textContent = message;
    banner.classList.remove("hidden");
    clearTimeout(showError._timer);
    showError._timer = setTimeout(() => banner.classList.add("hidden"), 4000);
}

function connect() {
    updateConnectionStatus(false);
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${protocol}://${location.host}`);

    ws.addEventListener("open", () => {
        reconnectDelay = 500;
        updateConnectionStatus(true);
    });

    ws.addEventListener("close", () => {
        ws = null;
        updateConnectionStatus(false);
        setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(Math.round(reconnectDelay * 1.6), MAX_RECONNECT_DELAY);
    });

    ws.addEventListener("message", (event) => {
        let message;
        try {
            message = JSON.parse(event.data);
        } catch {
            showError("Received invalid JSON from the server.");
            return;
        }

        if (message.type === "state") {
            lastFullState = message.data;
            onFullState(message.data);
        } else if (message.type === "summary") {
            onSummary(message.data);
        } else if (message.type === "error") {
            showError(message.message || "Unknown server error.");
        }
    });

    ws.addEventListener("error", () => {
        showError("Could not reach the training server.");
    });
}

function sendCommand(message) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        showError("Not connected yet.");
        return false;
    }
    ws.send(JSON.stringify(message));
    return true;
}

function onFullState(state) {
    if (typeof updateControls === "function") updateControls(state);
    if (typeof updateNetworkGraph === "function") updateNetworkGraph(state);
    if (typeof updateLossChart === "function") updateLossChart(state);
    if (typeof updateInspector === "function") updateInspector(state);
}

function onSummary(summary) {
    if (typeof updateStatusFromSummary === "function") updateStatusFromSummary(summary);
    if (typeof updateLossChartSummary === "function") updateLossChartSummary(summary);
}

window.sendCommand = sendCommand;
window.showFrontendError = showError;
window.getLastFullState = getLastFullState;

document.addEventListener("DOMContentLoaded", connect);
