export function fmt(value: number | null | undefined, digits = 4): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

export function fmtArr(
  values: number[] | null | undefined,
  digits = 4,
): string {
  if (!Array.isArray(values)) return "—";
  return "[" + values.map((v) => fmt(v, digits)).join(", ") + "]";
}

export function safeMin(values: number[]): number | null {
  let r = Infinity;
  for (const v of values) if (v < r) r = v;
  return r === Infinity ? null : r;
}

export function safeMax(values: number[]): number | null {
  let r = -Infinity;
  for (const v of values) if (v > r) r = v;
  return r === -Infinity ? null : r;
}
