// Pure helpers, kept apart from components so vitest can check them without a DOM.

export type MetricRow = {
  step: number;
  loss: number;
  lr: number;
  grad_norm: number;
  tokens_per_s: number;
  tflops: number;
  time: number;
  val_loss?: number;
  mfu?: number;
};

/** Every k-th row so a chart draws at most `max` points; the last row is always kept. */
export function downsample<T>(rows: T[], max: number): T[] {
  if (rows.length <= max) return rows;
  const k = Math.ceil(rows.length / max);
  const out = rows.filter((_, i) => i % k === 0);
  if (out[out.length - 1] !== rows[rows.length - 1]) out.push(rows[rows.length - 1]);
  return out;
}

/** "x=2, w=-0.5, v=[1,2,3]" -> {x: 2, w: -0.5, v: [1, 2, 3]}. Throws on anything else. */
export function parseVariables(text: string): Record<string, number | number[]> {
  const out: Record<string, number | number[]> = {};
  const re = /([A-Za-z_]\w*)\s*=\s*(\[[^\]]*\]|[-+]?[\d.eE+-]+)/g;
  const consumed = text.replace(re, (_, name: string, value: string) => {
    const parsed = JSON.parse(value);
    const ok = typeof parsed === "number" || (Array.isArray(parsed) && parsed.every((v) => typeof v === "number"));
    if (!ok) throw new Error(`${name} must be a number or a list of numbers`);
    out[name] = parsed;
    return "";
  });
  if (consumed.replace(/[\s,]/g, "")) throw new Error(`could not read "${consumed.trim()}"`);
  return out;
}

export function formatNumber(v: number | undefined, digits = 4): string {
  if (v === undefined || Number.isNaN(v)) return "-";
  if (Math.abs(v) >= 1e4) return Math.round(v).toLocaleString("en-US");
  if (v !== 0 && Math.abs(v) < 1e-3) return v.toExponential(2);
  return Number(v.toPrecision(digits)).toString();
}

/** Vertical offsets for end-of-line labels: series whose last values are `close` fan out 15px apart instead of overlapping. */
export function labelOffsets(lasts: Record<string, number>, close: (a: number, b: number) => boolean): Record<string, number> {
  const order = Object.entries(lasts).sort((a, b) => b[1] - a[1]);
  const dy: Record<string, number> = {};
  let run: string[] = [];
  const flush = () => {
    run.forEach((k, i) => (dy[k] = 4 + (i - (run.length - 1) / 2) * 15));
    run = [];
  };
  order.forEach(([k, v], i) => {
    if (i > 0 && !close(order[i - 1][1], v)) flush();
    run.push(k);
  });
  flush();
  return dy;
}
