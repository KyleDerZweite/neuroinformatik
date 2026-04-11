/**
 * Controls: sidebar inputs, buttons, sample list, inspector.
 */

const PRESET_DEFAULTS = {
    xor: { structure: "2, 2, 1", lr: 1.0 },
    sine: { structure: "1, 2, 1", lr: 0.275 },
};

const PRESET_PREVIEWS = {
    xor: [
        { index: 0, label: "Quiet pair", input: [0.0, 0.0], target: [0.0] },
        { index: 1, label: "Right spike", input: [0.0, 1.0], target: [1.0] },
        { index: 2, label: "Left spike", input: [1.0, 0.0], target: [1.0] },
        { index: 3, label: "Dual spike", input: [1.0, 1.0], target: [0.0] },
    ],
    sine: [
        { index: 0, label: "x = 0.00", input: [0.0], target: [0.5] },
        { index: 100, label: "x = 1.00", input: [1 / 7], target: [(Math.sin(1) + 1) / 2] },
        { index: 200, label: "x = 2.00", input: [2 / 7], target: [(Math.sin(2) + 1) / 2] },
        { index: 300, label: "x = 3.00", input: [3 / 7], target: [(Math.sin(3) + 1) / 2] },
        { index: 400, label: "x = 4.00", input: [4 / 7], target: [(Math.sin(4) + 1) / 2] },
        { index: 500, label: "x = 5.00", input: [5 / 7], target: [(Math.sin(5) + 1) / 2] },
        { index: 600, label: "x = 6.00", input: [6 / 7], target: [(Math.sin(6) + 1) / 2] },
        { index: 700, label: "x = 7.00", input: [1.0], target: [(Math.sin(7) + 1) / 2] },
    ],
};

let isRunning = false;
let lastRenderedPreset = null;
const inspectorOpenState = {};

/* ── Helpers ── */

function byId(id) {
    return document.getElementById(id);
}

function fmt(value, digits = 4) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function fmtArr(values, digits = 4) {
    if (!Array.isArray(values)) return "—";
    return "[" + values.map(v => fmt(v, digits)).join(", ") + "]";
}

function activityHint(phase, mode) {
    if (mode === "run") return "Continuous training active.";
    if (phase === "forward_done") return "Forward done — backward controls armed.";
    if (phase === "backward_done") return "Backward applied — advance to next sample.";
    return "Ready for manual stepping.";
}

/* ── Sample list ── */

function renderSampleList(activeIndex) {
    const preset = byId("preset-select")?.value || "xor";
    const list = byId("sample-list");
    const summary = byId("dataset-summary");
    if (!list || !summary) return;

    if (lastRenderedPreset === preset) {
        list.querySelectorAll(".sample-card").forEach(card => {
            const idx = parseInt(card.dataset.sampleIndex, 10);
            card.classList.toggle("is-active", idx === activeIndex);
        });
        return;
    }

    lastRenderedPreset = preset;
    const preview = PRESET_PREVIEWS[preset] || [];
    summary.textContent = preset === "xor" ? "4 binary samples" : "Sine anchors x = 0..7";

    list.innerHTML = preview.map(s => `
        <button class="sample-card${s.index === activeIndex ? " is-active" : ""}" data-sample-index="${s.index}">
            <strong>${s.label}</strong>
            <span class="sample-meta">sample ${s.index + 1}</span>
            <span class="sample-data">in ${fmtArr(s.input, 3)} → ${fmtArr(s.target, 3)}</span>
        </button>
    `).join("");
}

/* ── Running state ── */

function setRunningState(running) {
    isRunning = running;
    byId("btn-run").disabled = running;
    byId("btn-pause").disabled = !running;
    byId("btn-step-epoch").disabled = running;
    byId("btn-step-forward").disabled = running;
    byId("btn-step-backward").disabled = running || true; // depends on phase
    byId("btn-layer-forward").disabled = running;
    byId("btn-layer-backward").disabled = running || true;
    byId("epoch-count").disabled = running;
}

function updateStepButtons(phase) {
    if (isRunning) return;
    byId("btn-step-forward").disabled = false;
    byId("btn-step-epoch").disabled = false;
    byId("btn-layer-forward").disabled = false;
    byId("btn-step-backward").disabled = phase !== "forward_done";
    byId("btn-layer-backward").disabled = phase !== "forward_done";
}

/* ── Inspector ── */

function buildValueRow(label, value) {
    return `<div class="value-row"><span class="value-label">${label}</span><span>${value}</span></div>`;
}

function buildMatrix(weights) {
    if (!Array.isArray(weights) || weights.length === 0) {
        return '<p class="empty-state">No weights yet.</p>';
    }
    return `<div class="matrix-grid">${weights.map(row => `
        <div class="matrix-row" style="grid-template-columns:repeat(${row.length},1fr)">
            ${row.map(v => `<span class="matrix-cell ${v >= 0 ? "positive" : "negative"}">${fmt(v, 3)}</span>`).join("")}
        </div>`).join("")}</div>`;
}

function saveInspectorState(container) {
    container.querySelectorAll("details.inspector-card").forEach(d => {
        inspectorOpenState[d.dataset.key] = d.open;
    });
}

function restoreInspectorState(container) {
    container.querySelectorAll("details.inspector-card").forEach(d => {
        const saved = inspectorOpenState[d.dataset.key];
        if (saved !== undefined) d.open = saved;
    });
}

function updateInspector(state) {
    const container = byId("inspector-content");
    if (!container) return;

    saveInspectorState(container);

    if (!state.layers || state.layers.length === 0) {
        container.innerHTML = '<p class="empty-state">Configure a network to inspect values.</p>';
        return;
    }

    const sample = state.current_sample;
    const currentLayer = state.training.current_layer_index;

    const sampleCard = sample ? `
        <details class="inspector-card" data-key="sample" open>
            <summary class="inspector-toggle">
                <div><h4>Current sample</h4><span class="layer-meta">sample ${sample.index + 1}</span></div>
                <span class="chevron">›</span>
            </summary>
            <div class="inspector-body">
                <div class="value-grid">
                    ${buildValueRow("Input", fmtArr(sample.input))}
                    ${buildValueRow("Target", fmtArr(sample.target))}
                    ${buildValueRow("Prediction", fmtArr(sample.prediction))}
                    ${buildValueRow("Loss", sample.loss === null ? "—" : fmt(sample.loss, 6))}
                </div>
            </div>
        </details>` : "";

    const layerCards = state.layers.map(layer => {
        const isActive = currentLayer === layer.index || currentLayer === layer.index + 1;
        return `
        <details class="inspector-card" data-key="layer-${layer.index}" ${isActive ? "open" : ""}>
            <summary class="inspector-toggle">
                <div><h4>Layer ${layer.index + 1}</h4><span class="layer-meta">${layer.input_size} → ${layer.output_size}</span></div>
                <span class="chevron">›</span>
            </summary>
            <div class="inspector-body">
                <p class="inspector-subhead">Weights</p>
                ${buildMatrix(layer.weights)}
                <div class="value-grid">
                    ${buildValueRow("Biases", fmtArr(layer.biases))}
                    ${buildValueRow("Input", fmtArr(layer.last_input))}
                    ${buildValueRow("Pre-act", fmtArr(layer.last_pre_activation))}
                    ${buildValueRow("Output", fmtArr(layer.last_output))}
                    ${buildValueRow("Gradient", fmtArr(layer.last_gradient))}
                </div>
            </div>
        </details>`;
    }).join("");

    container.innerHTML = sampleCard + layerCards;
    restoreInspectorState(container);
}

/* ── Main update functions ── */

function updateControls(state) {
    const training = state.training;
    const sampleIdx = state.current_sample ? state.current_sample.index : training.current_sample_index;
    const losses = Array.isArray(training.epoch_losses) ? training.epoch_losses : [];
    const latestLoss = losses.length ? losses[losses.length - 1] : null;

    byId("status-epoch").textContent = String(training.current_epoch);
    byId("status-sample").textContent = `${Math.min(sampleIdx + 1, training.total_samples)}/${training.total_samples}`;
    byId("status-phase").textContent = training.phase;
    byId("status-loss").textContent = latestLoss === null ? "—" : fmt(latestLoss, 6);
    byId("activity-hint").textContent = activityHint(training.phase, training.mode);
    byId("topology-chip").textContent = state.topology.structure.length ? state.topology.structure.join(" → ") : "—";
    byId("learning-rate-chip").textContent = state.topology.learning_rate === null ? "lr —" : `lr ${fmt(state.topology.learning_rate, 3)}`;

    setRunningState(training.mode === "run");
    updateStepButtons(training.phase);
    renderSampleList(sampleIdx);
}

function updateStatusFromSummary(summary) {
    const training = summary.training;
    byId("status-epoch").textContent = String(training.current_epoch);
    byId("status-phase").textContent = training.phase;
    byId("status-loss").textContent = training.latest_loss === null ? "—" : fmt(training.latest_loss, 6);
    byId("activity-hint").textContent = activityHint(training.phase, training.mode);
    setRunningState(training.mode === "run");
}

/* ── Event listeners ── */

document.addEventListener("DOMContentLoaded", () => {
    const presetSelect = byId("preset-select");
    const structureInput = byId("structure-input");
    const lrInput = byId("lr-input");
    const speedSlider = byId("speed-slider");
    const speedValue = byId("speed-value");

    renderSampleList(0);

    presetSelect.addEventListener("change", () => {
        const defaults = PRESET_DEFAULTS[presetSelect.value];
        if (defaults) {
            structureInput.value = defaults.structure;
            lrInput.value = String(defaults.lr);
        }
        lastRenderedPreset = null;
        renderSampleList(0);
    });

    speedSlider.addEventListener("input", () => {
        speedValue.textContent = `${speedSlider.value} ms`;
    });

    byId("btn-configure").addEventListener("click", () => {
        const structure = structureInput.value
            .split(",")
            .map(s => parseInt(s.trim(), 10))
            .filter(n => Number.isInteger(n) && n > 0);

        if (structure.length < 2) {
            window.showFrontendError("Structure needs at least input and output layers.");
            return;
        }

        setRunningState(false);
        window.sendCommand({
            type: "configure",
            structure,
            learning_rate: parseFloat(lrInput.value) || 1.0,
            preset: presetSelect.value,
        });
    });

    byId("btn-reset").addEventListener("click", () => {
        setRunningState(false);
        window.sendCommand({ type: "reset" });
    });

    byId("btn-run").addEventListener("click", () => {
        const sent = window.sendCommand({
            type: "run",
            speed_ms: parseInt(speedSlider.value, 10) || 50,
        });
        if (sent) setRunningState(true);
    });

    byId("btn-pause").addEventListener("click", () => {
        window.sendCommand({ type: "pause" });
    });

    byId("btn-step-epoch").addEventListener("click", () => {
        window.sendCommand({
            type: "step_epoch",
            count: parseInt(byId("epoch-count").value, 10) || 1,
        });
    });

    byId("btn-step-forward").addEventListener("click", () => {
        window.sendCommand({ type: "step_forward" });
    });

    byId("btn-step-backward").addEventListener("click", () => {
        window.sendCommand({ type: "step_backward" });
    });

    byId("btn-layer-forward").addEventListener("click", () => {
        window.sendCommand({ type: "step_layer_forward" });
    });

    byId("btn-layer-backward").addEventListener("click", () => {
        window.sendCommand({ type: "step_layer_backward" });
    });

    byId("sample-list").addEventListener("click", (event) => {
        const button = event.target.closest("[data-sample-index]");
        if (!button || isRunning) return;
        const sampleIndex = parseInt(button.dataset.sampleIndex, 10);
        if (!Number.isInteger(sampleIndex)) return;
        window.sendCommand({ type: "step_forward", sample_index: sampleIndex });
    });
});
