/**
 * Network graph: canvas rendering with background caching.
 */

let networkCanvas, networkCtx;
let bgCache = null, bgCacheW = 0, bgCacheH = 0;

const THEME = {
    grid: "rgba(240, 236, 228, 0.06)",
    text: "#e8e2d6",
    muted: "rgba(232, 226, 214, 0.55)",
    positive: [58, 154, 156],
    negative: [212, 97, 62],
    active: [201, 162, 77],
    neuronBase: [232, 226, 214],
    neuronStroke: "rgba(232, 226, 214, 0.15)",
};

document.addEventListener("DOMContentLoaded", () => {
    networkCanvas = document.getElementById("network-canvas");
    if (!networkCanvas) return;
    networkCtx = networkCanvas.getContext("2d");
    resizeNetworkCanvas();
    window.addEventListener("resize", resizeNetworkCanvas);
});

function resizeNetworkCanvas() {
    if (!networkCanvas || !networkCtx) return;
    const rect = networkCanvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    networkCanvas.width = Math.max(1, Math.floor(rect.width * scale));
    networkCanvas.height = Math.max(1, Math.floor(rect.height * scale));
    networkCtx.setTransform(scale, 0, 0, scale, 0, 0);
    bgCache = null; // invalidate
    const state = window.getLastFullState ? window.getLastFullState() : null;
    if (state) updateNetworkGraph(state);
    else drawPlaceholder();
}

function drawPlaceholder() {
    if (!networkCtx || !networkCanvas) return;
    const width = networkCanvas.width / (window.devicePixelRatio || 1);
    const height = networkCanvas.height / (window.devicePixelRatio || 1);
    drawBackground(width, height);
    networkCtx.fillStyle = THEME.text;
    networkCtx.font = '600 20px "Iowan Old Style", Georgia, serif';
    networkCtx.textAlign = "center";
    networkCtx.fillText("Configure a network to begin.", width / 2, height / 2 - 6);
    networkCtx.fillStyle = THEME.muted;
    networkCtx.font = '13px "IBM Plex Mono", Consolas, monospace';
    networkCtx.fillText("Weights, activations, and targets will appear here.", width / 2, height / 2 + 22);
}

function drawBackground(width, height) {
    if (bgCache && bgCacheW === width && bgCacheH === height) {
        networkCtx.drawImage(bgCache, 0, 0);
        return;
    }

    const scale = window.devicePixelRatio || 1;
    const off = document.createElement("canvas");
    off.width = Math.max(1, Math.floor(width * scale));
    off.height = Math.max(1, Math.floor(height * scale));
    const ctx = off.getContext("2d");
    ctx.setTransform(scale, 0, 0, scale, 0, 0);

    ctx.strokeStyle = THEME.grid;
    ctx.lineWidth = 1;
    for (let x = 24; x < width; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }
    for (let y = 24; y < height; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }

    bgCache = off;
    bgCacheW = width;
    bgCacheH = height;
    networkCtx.drawImage(off, 0, 0);
}

function drawLabel(x, y, text, color, font) {
    networkCtx.fillStyle = color;
    networkCtx.font = font;
    networkCtx.textAlign = "center";
    networkCtx.textBaseline = "middle";
    networkCtx.fillText(text, x, y);
}

function drawConnection(from, to, weight) {
    const colorParts = weight >= 0 ? THEME.positive : THEME.negative;
    const mag = Math.abs(weight);
    const opacity = Math.min(0.15 + mag * 0.35, 0.9);
    const thickness = Math.min(1 + mag * 2.5, 5.5);

    networkCtx.beginPath();
    networkCtx.moveTo(from.x, from.y);
    networkCtx.bezierCurveTo(from.x + 36, from.y, to.x - 36, to.y, to.x, to.y);
    networkCtx.strokeStyle = `rgba(${colorParts[0]},${colorParts[1]},${colorParts[2]},${opacity})`;
    networkCtx.lineWidth = thickness;
    networkCtx.stroke();
}

function drawNeuron(node, activation, label) {
    const radius = node.radius;
    const clamped = typeof activation === "number" ? Math.max(0, Math.min(1, activation)) : 0;

    const fill = [
        Math.round(THEME.neuronBase[0] * (1 - clamped) + THEME.active[0] * clamped),
        Math.round(THEME.neuronBase[1] * (1 - clamped) + THEME.active[1] * clamped),
        Math.round(THEME.neuronBase[2] * (1 - clamped) + THEME.active[2] * clamped),
    ];

    networkCtx.beginPath();
    networkCtx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    networkCtx.fillStyle = `rgb(${fill[0]},${fill[1]},${fill[2]})`;
    networkCtx.shadowColor = "rgba(201, 162, 77, 0.2)";
    networkCtx.shadowBlur = 14;
    networkCtx.fill();
    networkCtx.shadowBlur = 0;
    networkCtx.strokeStyle = THEME.neuronStroke;
    networkCtx.lineWidth = 1.5;
    networkCtx.stroke();

    if (label) {
        drawLabel(node.x, node.y + radius + 16, label, THEME.text, '11px "IBM Plex Mono", Consolas, monospace');
    }
}

function computePositions(structure, width, height) {
    const left = 64, right = width - 64, top = 80, bottom = height - 48;
    const cols = structure.length;
    const xGap = cols > 1 ? (right - left) / (cols - 1) : 0;

    return structure.map((count, col) => {
        const total = bottom - top;
        const yGap = count > 1 ? total / (count - 1) : 0;
        const yStart = count > 1 ? top : (top + bottom) / 2;
        const radius = Math.max(12, Math.min(18, 28 - count));

        return Array.from({ length: count }, (_, i) => ({
            x: left + col * xGap,
            y: count > 1 ? yStart + i * yGap : yStart,
            radius,
        }));
    });
}

function fmtNet(value) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "—";
}

function updateNetworkGraph(state) {
    if (!networkCtx || !networkCanvas) return;

    const structure = state?.topology?.structure || [];
    const width = networkCanvas.width / (window.devicePixelRatio || 1);
    const height = networkCanvas.height / (window.devicePixelRatio || 1);

    drawBackground(width, height);

    if (structure.length < 2 || !state.layers || state.layers.length === 0) {
        drawPlaceholder();
        return;
    }

    drawLabel(width / 2, 30, "Input → Output", THEME.muted, '12px "IBM Plex Mono", Consolas, monospace');

    const positions = computePositions(structure, width, height);

    // Draw connections
    for (let li = 0; li < state.layers.length; li++) {
        const layer = state.layers[li];
        const from = positions[li];
        const to = positions[li + 1];
        for (let fi = 0; fi < from.length; fi++) {
            for (let ti = 0; ti < to.length; ti++) {
                drawConnection(from[fi], to[ti], layer.weights[fi][ti]);
            }
        }
    }

    // Draw input neurons
    const inputValues = state.current_sample?.input || [];
    for (let i = 0; i < positions[0].length; i++) {
        const node = positions[0][i];
        const val = inputValues[i];
        drawNeuron(node, val, typeof val === "number" ? val.toFixed(3) : null);
        drawLabel(node.x, node.y - node.radius - 16, `I${i + 1}`, THEME.muted, '11px "IBM Plex Mono", Consolas, monospace');
    }

    // Draw hidden + output neurons
    for (let li = 0; li < state.layers.length; li++) {
        const layer = state.layers[li];
        const nodes = positions[li + 1];
        for (let ni = 0; ni < nodes.length; ni++) {
            const node = nodes[ni];
            const activation = layer.last_output?.[ni];
            drawNeuron(node, activation, typeof activation === "number" ? activation.toFixed(3) : null);
            drawLabel(node.x, node.y - node.radius - 16, `b ${fmtNet(layer.biases?.[ni])}`, THEME.muted, '10px "IBM Plex Mono", Consolas, monospace');
        }
    }

    // Draw target labels
    const targets = state.current_sample?.target || [];
    const outputNodes = positions[positions.length - 1];
    for (let ni = 0; ni < outputNodes.length; ni++) {
        const target = targets[ni];
        if (typeof target === "number") {
            drawLabel(outputNodes[ni].x, outputNodes[ni].y + outputNodes[ni].radius + 32, `t ${target.toFixed(3)}`, THEME.muted, '10px "IBM Plex Mono", Consolas, monospace');
        }
    }

    // Layer labels
    positions.forEach((col, ci) => {
        const label = ci === 0 ? "Input" : ci === positions.length - 1 ? "Output" : `Hidden ${ci}`;
        drawLabel(col[0].x, 52, label, THEME.text, '600 16px "Iowan Old Style", Georgia, serif');
    });
}
