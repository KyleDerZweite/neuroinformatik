import { useEffect, useRef } from "react";

import type { SessionState } from "../lib/api/types";

const THEME = {
  gridMinor: "rgba(232, 226, 214, 0.035)",
  gridMajor: "rgba(232, 226, 214, 0.07)",
  ink: "#e8e2d6",
  inkSoft: "#b8b0a0",
  inkDim: "#6f685c",
  inkFaint: "rgba(232, 226, 214, 0.2)",
  positive: [85, 166, 168] as const,
  negative: [217, 122, 86] as const,
  rust: "#c06544",
  teal: "#3a8a8c",
  tealBright: "#55a6a8",
};

const FONT_LABEL = '10px "JetBrains Mono", Consolas, monospace';
const FONT_NUMBER = '11px "JetBrains Mono", Consolas, monospace';
const FONT_LEADER = '9px "JetBrains Mono", Consolas, monospace';
const FONT_CAP = '12px "JetBrains Mono", Consolas, monospace';

interface Node {
  x: number;
  y: number;
  radius: number;
}

export function NetworkCanvas({ state }: { state: SessionState | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      drawNetwork(canvas, state);
    };

    draw();

    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [state]);

  return (
    <div className="canvas-frame">
      <span className="canvas-corner canvas-corner-tl" aria-hidden />
      <span className="canvas-corner canvas-corner-tr" aria-hidden />
      <span className="canvas-corner canvas-corner-bl" aria-hidden />
      <span className="canvas-corner canvas-corner-br" aria-hidden />
      <canvas
        ref={canvasRef}
        className="network-canvas"
        aria-label="Network visualization"
      />
    </div>
  );
}

function drawNetwork(canvas: HTMLCanvasElement, state: SessionState | null) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
  ctx.setTransform(scale, 0, 0, scale, 0, 0);

  const width = rect.width;
  const height = rect.height;

  ctx.clearRect(0, 0, width, height);
  drawBackground(ctx, width, height);

  const structure = state?.topology?.structure ?? [];
  if (structure.length < 2 || !state?.layers?.length) {
    drawPlaceholder(ctx, width, height);
    return;
  }

  const positions = computePositions(structure, width, height);

  // column labels + ticks
  positions.forEach((col, ci) => {
    const node = col[0];
    if (!node) return;
    const label =
      ci === 0 ? "INPUT" : ci === positions.length - 1 ? "OUTPUT" : `HIDDEN·${ci}`;
    ctx.strokeStyle = THEME.rust;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(node.x, 28);
    ctx.lineTo(node.x, 46);
    ctx.stroke();
    ctx.globalAlpha = 1;
    drawText(ctx, node.x, 20, label, THEME.rust, FONT_LABEL, "center", "middle");
    drawText(
      ctx,
      node.x,
      58,
      `n=${structure[ci] ?? 0}`,
      THEME.inkDim,
      FONT_LEADER,
      "center",
      "middle",
    );
  });

  // horizontal ref line
  const firstCol = positions[0];
  const lastCol = positions[positions.length - 1];
  if (firstCol && lastCol && firstCol[0] && lastCol[0]) {
    ctx.strokeStyle = "rgba(232, 226, 214, 0.1)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(firstCol[0].x, 72);
    ctx.lineTo(lastCol[0].x, 72);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // connections first (under neurons)
  for (let li = 0; li < state.layers.length; li++) {
    const layer = state.layers[li];
    const from = positions[li];
    const to = positions[li + 1];
    if (!layer?.weights || !from || !to) continue;
    for (let fi = 0; fi < from.length; fi++) {
      const src = from[fi];
      if (!src) continue;
      const weights = layer.weights[fi];
      if (!weights) continue;
      for (let ti = 0; ti < to.length; ti++) {
        const dst = to[ti];
        const w = weights[ti];
        if (!dst || typeof w !== "number") continue;
        drawConnection(ctx, src, dst, w);
      }
    }
  }

  // input neurons
  const inputValues = state.current_sample?.input ?? [];
  const inputCol = positions[0];
  if (inputCol) {
    for (let i = 0; i < inputCol.length; i++) {
      const node = inputCol[i];
      if (!node) continue;
      const val = inputValues[i];
      drawNeuron(
        ctx,
        node,
        val,
        typeof val === "number" ? val.toFixed(3) : null,
      );
      ctx.strokeStyle = "rgba(232, 226, 214, 0.25)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(node.x - node.radius - 22, node.y);
      ctx.lineTo(node.x - node.radius - 2, node.y);
      ctx.stroke();
      drawText(
        ctx,
        node.x - node.radius - 26,
        node.y,
        `x${i + 1}`,
        THEME.inkDim,
        FONT_LABEL,
        "right",
        "middle",
      );
    }
  }

  // hidden + output neurons
  for (let li = 0; li < state.layers.length; li++) {
    const layer = state.layers[li];
    const nodes = positions[li + 1];
    if (!layer || !nodes) continue;
    for (let ni = 0; ni < nodes.length; ni++) {
      const node = nodes[ni];
      if (!node) continue;
      const activation = layer.last_output?.[ni];
      drawNeuron(
        ctx,
        node,
        activation,
        typeof activation === "number" ? activation.toFixed(3) : null,
      );
      const bias = layer.biases?.[ni];
      drawText(
        ctx,
        node.x,
        node.y - node.radius - 12,
        `b ${typeof bias === "number" ? bias.toFixed(2) : "—"}`,
        THEME.inkDim,
        FONT_LEADER,
        "center",
        "middle",
      );
    }
  }

  // target diamonds at output
  const outputNodes = positions[positions.length - 1];
  const targets = state.current_sample?.target ?? [];
  if (outputNodes) {
    for (let ni = 0; ni < outputNodes.length; ni++) {
      const node = outputNodes[ni];
      const target = targets[ni];
      if (!node || typeof target !== "number") continue;
      const tx = node.x + node.radius + 52;

      ctx.strokeStyle = THEME.teal;
      ctx.globalAlpha = 0.55;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(node.x + node.radius + 22, node.y);
      ctx.lineTo(tx, node.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      ctx.fillStyle = THEME.teal;
      ctx.beginPath();
      ctx.moveTo(tx, node.y - 4);
      ctx.lineTo(tx + 4, node.y);
      ctx.lineTo(tx, node.y + 4);
      ctx.lineTo(tx - 4, node.y);
      ctx.closePath();
      ctx.fill();

      drawText(
        ctx,
        tx + 8,
        node.y,
        `t ${target.toFixed(3)}`,
        THEME.tealBright,
        FONT_LABEL,
        "left",
        "middle",
      );
    }
  }

  drawText(
    ctx,
    width / 2,
    height - 18,
    "—— forward pass →   ·   ← backward gradient ——",
    THEME.inkDim,
    FONT_LEADER,
    "center",
    "middle",
  );
}

function drawBackground(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
) {
  ctx.strokeStyle = THEME.gridMinor;
  ctx.lineWidth = 1;
  for (let x = 0; x <= width; x += 32) {
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, height);
    ctx.stroke();
  }
  for (let y = 0; y <= height; y += 32) {
    ctx.beginPath();
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(width, y + 0.5);
    ctx.stroke();
  }
  ctx.strokeStyle = THEME.gridMajor;
  for (let x = 0; x <= width; x += 128) {
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, height);
    ctx.stroke();
  }
  for (let y = 0; y <= height; y += 128) {
    ctx.beginPath();
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(width, y + 0.5);
    ctx.stroke();
  }

  // origin crosshair
  ctx.strokeStyle = THEME.rust;
  ctx.globalAlpha = 0.5;
  ctx.beginPath();
  ctx.moveTo(10, 0);
  ctx.lineTo(10, 14);
  ctx.moveTo(0, 10);
  ctx.lineTo(14, 10);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawPlaceholder(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
) {
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = THEME.ink;
  ctx.font = 'italic 500 26px "Cormorant Garamond", Georgia, serif';
  ctx.fillText("awaiting configuration", width / 2, height / 2 - 12);
  ctx.fillStyle = THEME.inkDim;
  ctx.font = FONT_CAP;
  ctx.fillText(
    "// select a preset and press configure",
    width / 2,
    height / 2 + 18,
  );
  ctx.strokeStyle = THEME.inkFaint;
  ctx.lineWidth = 1;
  for (const r of [24, 52, 88]) {
    ctx.beginPath();
    ctx.arc(width / 2, height / 2, r, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function computePositions(
  structure: number[],
  width: number,
  height: number,
): Node[][] {
  const left = 90;
  const right = width - 130;
  const top = 90;
  const bottom = height - 50;
  const cols = structure.length;
  const xGap = cols > 1 ? (right - left) / (cols - 1) : 0;

  return structure.map((count, col) => {
    const total = bottom - top;
    const yGap = count > 1 ? total / (count - 1) : 0;
    const yStart = count > 1 ? top : (top + bottom) / 2;
    const radius = Math.max(14, Math.min(22, 30 - count));
    return Array.from({ length: count }, (_, i) => ({
      x: left + col * xGap,
      y: count > 1 ? yStart + i * yGap : yStart,
      radius,
    }));
  });
}

function drawText(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
  color: string,
  font: string,
  align: CanvasTextAlign = "center",
  baseline: CanvasTextBaseline = "middle",
) {
  ctx.fillStyle = color;
  ctx.font = font;
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  ctx.fillText(text, x, y);
}

function drawConnection(
  ctx: CanvasRenderingContext2D,
  from: Node,
  to: Node,
  weight: number,
) {
  const color = weight >= 0 ? THEME.positive : THEME.negative;
  const mag = Math.abs(weight);
  const opacity = Math.min(0.18 + mag * 0.55, 0.95);
  const thickness = Math.min(0.8 + mag * 2.2, 4.5);

  const ang = Math.atan2(to.y - from.y, to.x - from.x);
  const fx = from.x + Math.cos(ang) * from.radius;
  const fy = from.y + Math.sin(ang) * from.radius;
  const tx = to.x - Math.cos(ang) * to.radius;
  const ty = to.y - Math.sin(ang) * to.radius;

  ctx.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},${opacity})`;
  ctx.lineWidth = thickness;
  ctx.beginPath();
  ctx.moveTo(fx, fy);
  ctx.lineTo(tx, ty);
  ctx.stroke();

  ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${Math.min(opacity + 0.1, 1)})`;
  ctx.beginPath();
  ctx.arc(tx, ty, thickness * 0.6, 0, Math.PI * 2);
  ctx.fill();
}

function drawNeuron(
  ctx: CanvasRenderingContext2D,
  node: Node,
  activation: number | null | undefined,
  label: string | null,
) {
  const radius = node.radius;
  const clamped =
    typeof activation === "number" ? Math.max(0, Math.min(1, activation)) : 0;

  ctx.fillStyle = "#0b0f14";
  ctx.strokeStyle = THEME.ink;
  ctx.globalAlpha = 0.85;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.globalAlpha = 1;

  if (clamped > 0.02) {
    const innerR = radius * 0.62 * clamped + radius * 0.08;
    ctx.fillStyle = `rgba(232, 226, 214, ${0.35 + clamped * 0.55})`;
    ctx.beginPath();
    ctx.arc(node.x, node.y, innerR, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.strokeStyle = "rgba(232, 226, 214, 0.15)";
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  ctx.moveTo(node.x - radius * 0.45, node.y);
  ctx.lineTo(node.x + radius * 0.45, node.y);
  ctx.moveTo(node.x, node.y - radius * 0.45);
  ctx.lineTo(node.x, node.y + radius * 0.45);
  ctx.stroke();

  if (label) {
    const lx = node.x + radius + 10;
    ctx.strokeStyle = "rgba(232, 226, 214, 0.25)";
    ctx.setLineDash([2, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(node.x + radius + 2, node.y);
    ctx.lineTo(lx, node.y);
    ctx.stroke();
    ctx.setLineDash([]);
    drawText(ctx, lx + 2, node.y, label, THEME.inkSoft, FONT_NUMBER, "left", "middle");
  }
}
