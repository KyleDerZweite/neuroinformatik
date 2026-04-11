/**
 * Loss chart: canvas rendering with safe min/max.
 */

let lossCanvas, lossCtx;
let lossHistory = [];

const THEME = {
    grid: "rgba(240, 236, 228, 0.06)",
    line: "#d4613e",
    fill: "rgba(212, 97, 62, 0.12)",
    point: "#3a9a9c",
    text: "#6b6358",
    muted: "#8a8070",
};

document.addEventListener("DOMContentLoaded", () => {
    lossCanvas = document.getElementById("loss-chart");
    if (!lossCanvas) return;
    lossCtx = lossCanvas.getContext("2d");
    resizeLossCanvas();
    window.addEventListener("resize", resizeLossCanvas);
});

function resizeLossCanvas() {
    if (!lossCanvas || !lossCtx) return;
    const rect = lossCanvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    lossCanvas.width = Math.max(1, Math.floor(rect.width * scale));
    lossCanvas.height = Math.max(1, Math.floor(rect.height * scale));
    lossCtx.setTransform(scale, 0, 0, scale, 0, 0);
    drawLossChart();
}

/* Safe min/max — no stack overflow on large arrays */
function safeMin(values) {
    let result = Infinity;
    for (let i = 0; i < values.length; i++) {
        if (values[i] < result) result = values[i];
    }
    return result === Infinity ? null : result;
}

function safeMax(values) {
    let result = -Infinity;
    for (let i = 0; i < values.length; i++) {
        if (values[i] > result) result = values[i];
    }
    return result === -Infinity ? null : result;
}

function updateLossMeta() {
    const lengthEl = document.getElementById("loss-length");
    const bestEl = document.getElementById("loss-best");
    const captionEl = document.getElementById("loss-caption");

    const latest = lossHistory.length ? lossHistory[lossHistory.length - 1] : null;
    const best = safeMin(lossHistory);

    if (lengthEl) lengthEl.textContent = `${lossHistory.length} epoch${lossHistory.length === 1 ? "" : "s"}`;
    if (bestEl) bestEl.textContent = best === null ? "best —" : `best ${best.toFixed(6)}`;
    if (captionEl) captionEl.textContent = latest === null
        ? "Run training to generate a curve."
        : `Latest ${latest.toFixed(6)}. Lower is better.`;
}

function drawLossChart() {
    if (!lossCtx || !lossCanvas) return;

    const width = lossCanvas.width / (window.devicePixelRatio || 1);
    const height = lossCanvas.height / (window.devicePixelRatio || 1);
    const pad = { top: 24, right: 16, bottom: 30, left: 50 };

    lossCtx.clearRect(0, 0, width, height);

    if (lossHistory.length === 0) {
        lossCtx.fillStyle = THEME.text;
        lossCtx.font = '600 18px "Iowan Old Style", Georgia, serif';
        lossCtx.textAlign = "center";
        lossCtx.fillText("No training history yet.", width / 2, height / 2 - 6);
        lossCtx.fillStyle = THEME.muted;
        lossCtx.font = '12px "IBM Plex Mono", Consolas, monospace';
        lossCtx.fillText("Loss will be plotted after epoch updates.", width / 2, height / 2 + 18);
        updateLossMeta();
        return;
    }

    const chartW = width - pad.left - pad.right;
    const chartH = height - pad.top - pad.bottom;
    const minLoss = safeMin(lossHistory);
    const maxLoss = safeMax(lossHistory);
    const range = Math.max(maxLoss - minLoss, 0.000001);

    // Grid lines
    lossCtx.strokeStyle = THEME.grid;
    lossCtx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (chartH * i) / 4;
        lossCtx.beginPath();
        lossCtx.moveTo(pad.left, y);
        lossCtx.lineTo(width - pad.right, y);
        lossCtx.stroke();
    }

    // Y-axis labels
    lossCtx.fillStyle = THEME.muted;
    lossCtx.font = '11px "IBM Plex Mono", Consolas, monospace';
    lossCtx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
        const value = maxLoss - (range * i) / 4;
        const y = pad.top + (chartH * i) / 4 + 4;
        lossCtx.fillText(value.toFixed(4), pad.left - 8, y);
    }

    // Compute points
    const points = lossHistory.map((loss, i) => ({
        x: pad.left + (chartW * i) / Math.max(lossHistory.length - 1, 1),
        y: pad.top + ((maxLoss - loss) / range) * chartH,
    }));

    // Y-axis line
    lossCtx.strokeStyle = "rgba(240, 236, 228, 0.12)";
    lossCtx.beginPath();
    lossCtx.moveTo(pad.left, pad.top);
    lossCtx.lineTo(pad.left, pad.top + chartH);
    lossCtx.stroke();

    // Fill area
    lossCtx.beginPath();
    lossCtx.moveTo(points[0].x, pad.top + chartH);
    points.forEach(p => lossCtx.lineTo(p.x, p.y));
    lossCtx.lineTo(points[points.length - 1].x, pad.top + chartH);
    lossCtx.closePath();
    lossCtx.fillStyle = THEME.fill;
    lossCtx.fill();

    // Line
    lossCtx.beginPath();
    points.forEach((p, i) => i === 0 ? lossCtx.moveTo(p.x, p.y) : lossCtx.lineTo(p.x, p.y));
    lossCtx.strokeStyle = THEME.line;
    lossCtx.lineWidth = 2.5;
    lossCtx.stroke();

    // End point
    const last = points[points.length - 1];
    lossCtx.beginPath();
    lossCtx.arc(last.x, last.y, 4, 0, Math.PI * 2);
    lossCtx.fillStyle = THEME.point;
    lossCtx.fill();

    // X-axis labels
    lossCtx.fillStyle = THEME.muted;
    lossCtx.font = '11px "IBM Plex Mono", Consolas, monospace';
    lossCtx.textAlign = "left";
    lossCtx.fillText("1", pad.left, height - 8);
    lossCtx.textAlign = "right";
    lossCtx.fillText(String(lossHistory.length), width - pad.right, height - 8);

    updateLossMeta();
}

function updateLossChart(state) {
    lossHistory = Array.isArray(state?.training?.epoch_losses)
        ? [...state.training.epoch_losses]
        : [];
    drawLossChart();
}

function updateLossChartSummary(summary) {
    const epoch = summary?.training?.current_epoch;
    const latestLoss = summary?.training?.latest_loss;

    if (!Number.isInteger(epoch) || epoch <= 0 || typeof latestLoss !== "number") return;

    if (lossHistory.length < epoch) {
        lossHistory.push(latestLoss);
    } else {
        lossHistory[epoch - 1] = latestLoss;
        lossHistory.length = epoch;
    }

    drawLossChart();
}
