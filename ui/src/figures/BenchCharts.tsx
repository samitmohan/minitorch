import { useState } from "react";
import {
  CartesianGrid,
  Label,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { DEVICE_COLOR, DEVICE_NAME, DEVICE_SHORT, endLabel, INK, RULE } from "../chart";
import { formatNumber, labelOffsets } from "../lib";

export type ParityRow = { step: number; minitorch: number; torch: number; abs_diff: number };
export type Parity = {
  params: number;
  steps: number;
  dtype?: string;
  max_abs_loss_diff: number;
  final_abs_loss_diff: number;
  median_ms_per_step: Record<string, number>;
  rows: ParityRow[];
};
export type StepRow = { impl: string; device: string; precision?: string; batch: number; ms: number; tokens_per_s: number; tflops: number };
export type OpRow = { device: string; path: string; n: number; us_per_op: number };
export type MatmulRow = { device: string; dtype: string; n: number; tflops: number };
export type Overhead = { machine: string; gpu?: string; elementwise: OpRow[]; matmul: MatmulRow[]; train_step: StepRow[] };
export type AttnRow = { shape: string; dtype: string; impl: string; ms: number; tflops: number; gb_s: number };

export function ParityChart({ p }: { p: Parity }) {
  return (
    <section aria-label="Loss parity">
      <div className="hero-chart" style={{ height: 300 }}>
        <ResponsiveContainer>
          <LineChart data={p.rows} margin={{ top: 16, right: 24, bottom: 28, left: 16 }}>
            <CartesianGrid stroke="#c9d1ca" strokeDasharray="2 4" />
            <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} tick={{ fontSize: 12 }}>
              <Label value="step" position="insideBottom" offset={-16} fontSize={12} />
            </XAxis>
            <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} width={48} />
            <Tooltip formatter={(v) => formatNumber(Number(v), 6)} labelFormatter={(s) => `step ${s}`} />
            <Line dataKey="torch" name="PyTorch" stroke="#a3266f" strokeWidth={5} strokeOpacity={0.35} dot={false} isAnimationActive={false} />
            <Line dataKey="minitorch" name="minitorch" stroke="#2456b8" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function stepLabel(r: StepRow) {
  const precision = r.precision && r.precision !== "float32" ? `, ${r.precision}` : "";
  return `${r.impl === "minitorch" ? "minitorch" : r.impl.replace(/ \(.*\)/, "")} on ${DEVICE_NAME[r.device]}${precision}`;
}

export function StepBars({ rows }: { rows: StepRow[] }) {
  const data = [...rows]
    .sort((a, b) => b.tokens_per_s - a.tokens_per_s)
    .map((r) => ({ ...r, label: stepLabel(r), note: `${r.tokens_per_s.toLocaleString("en-US")} tok/s, ${formatNumber(r.ms)} ms` }));
  return (
    <section aria-label="Training step per backend">
      <ul className="bars" aria-label="Tokens per second, higher is faster">
        {data.map((r) => (
          <li key={r.label}>
            <span>{r.label}</span>
            <span className="track">
              <span style={{ width: `${(100 * r.tokens_per_s) / data[0].tokens_per_s}%`, background: DEVICE_COLOR[r.device] }} />
            </span>
            <span className="num">{r.note}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function AttentionTable({ rows }: { rows: AttnRow[] }) {
  const impls = [...new Set(rows.map((r) => r.impl))];
  const groups = [...new Set(rows.map((r) => `${r.shape}|${r.dtype}`))].map((key) => {
    const [shape, dtype] = key.split("|");
    const cells = rows.filter((r) => r.shape === shape && r.dtype === dtype);
    return { shape, dtype, cells, fastest: Math.min(...cells.map((c) => c.ms)) };
  });
  return (
    <section aria-label="Attention kernels">
      <div>
        <table className="results">
          <thead>
            <tr>
              <th>Shape</th>
              <th>Format</th>
              {impls.map((i) => <th key={i} className="num">{i}</th>)}
            </tr>
          </thead>
          <tbody>
            {groups.map((g) => (
              <tr key={`${g.shape}${g.dtype}`}>
                <th>{g.shape}</th>
                <td>{g.dtype}</td>
                {impls.map((i) => {
                  const c = g.cells.find((x) => x.impl === i);
                  const ms = c ? formatNumber(c.ms, 3) : "-";
                  return (
                    <td key={i} className="num" title={c ? `${formatNumber(c.tflops)} TFLOP/s, ${formatNumber(c.gb_s)} GB/s` : undefined}>
                      {c?.ms === g.fastest ? <b>{ms}</b> : ms}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// phones: no room for end-of-line labels beside the plot, so the legend alone names the lines
const narrow = () => window.matchMedia("(max-width: 720px)").matches;

const PATHS = ["raw", "sync every op", "minitorch"] as const;
const PATH_NOTE: Record<(typeof PATHS)[number], string> = {
  raw: "64 chained multiplies on the backend's own arrays, one wait at the end.",
  "sync every op": "The same chain, waiting for the device after every multiply: launch plus round-trip latency.",
  minitorch: "The same chain through minitorch Tensors, so it adds the framework's Python cost per op.",
};

/** Powers of ten spanning the data, for log axes. */
function decades(values: number[]) {
  const lo = Math.floor(Math.log10(Math.min(...values)));
  const hi = Math.ceil(Math.log10(Math.max(...values)));
  return Array.from({ length: hi - lo + 1 }, (_, i) => 10 ** (lo + i));
}

export function OpCost({ rows }: { rows: OpRow[] }) {
  const [path, setPath] = useState<(typeof PATHS)[number]>("minitorch");
  const devices = Object.keys(DEVICE_COLOR).filter((d) => rows.some((r) => r.device === d));
  const shown = rows.filter((r) => r.path === path);
  const yTicks = decades(rows.map((r) => r.us_per_op));
  const dy = labelOffsets(
    Object.fromEntries(devices.map((d) => [d, shown.filter((r) => r.device === d).at(-1)?.us_per_op ?? 0])),
    (a, b) => a / b < 2,
  );
  return (
    <section aria-label="Cost per op">
      <div className="page-head">
        <div className="runs" role="group" aria-label="Path">
          {PATHS.map((p) => (
            <button key={p} aria-pressed={p === path} onClick={() => setPath(p)}>{p}</button>
          ))}
        </div>
      </div>
      <p className="muted">
        {PATH_NOTE[path]}
      </p>
      <div className="hero-chart" style={{ height: narrow() ? 400 : 320 }}>
        <ResponsiveContainer>
          <LineChart margin={{ top: 16, right: narrow() ? 16 : 150, bottom: 28, left: 0 }}>
            <CartesianGrid stroke={RULE} strokeDasharray="2 4" />
            <XAxis dataKey="n" type="number" scale="log" domain={[1e3, 1e7]} ticks={[1e3, 1e4, 1e5, 1e6, 1e7]} allowDuplicatedCategory={false}
              stroke={INK} tick={{ fontSize: 12 }} tickFormatter={(v) => Number(v).toExponential(0).replace("+", "")}>
              <Label value="floats per array" position="insideBottom" offset={-16} fill={INK} fontSize={12} />
            </XAxis>
            <YAxis type="number" scale="log" domain={[yTicks[0], yTicks[yTicks.length - 1]]} ticks={yTicks} stroke={INK} width={76}
              tick={{ fontSize: 12 }} tickFormatter={(v) => `${formatNumber(Number(v))} µs`} />
            <Tooltip formatter={(v) => `${formatNumber(Number(v))} µs per op`} labelFormatter={(n) => `${Number(n).toLocaleString("en-US")} floats`} />
            <Legend verticalAlign="top" formatter={(v) => <span style={{ color: INK }}>{v}</span>} />
            {devices.map((d) => {
              const data = shown.filter((r) => r.device === d);
              return (
                <Line key={d} data={data} dataKey="us_per_op" name={DEVICE_NAME[d]} stroke={DEVICE_COLOR[d]} strokeWidth={2}
                  dot={{ r: 4, fill: DEVICE_COLOR[d], stroke: "#fafbfc", strokeWidth: 2 }} isAnimationActive={false}
                  label={narrow() ? undefined : endLabel(data.length, DEVICE_SHORT[d], INK, dy[d])} />
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <details>
        <summary>Every measurement as a table</summary>
        <table>
          <thead><tr><th>device</th><th>path</th><th className="num">floats</th><th className="num">µs per op</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.device}${r.path}${r.n}`}>
                <td>{DEVICE_NAME[r.device]}</td><td>{r.path}</td>
                <td className="num">{r.n.toLocaleString("en-US")}</td><td className="num">{formatNumber(r.us_per_op)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}

export function Matmul({ rows }: { rows: MatmulRow[] }) {
  const series = [...new Set(rows.map((r) => `${r.device}|${r.dtype}`))].map((key) => {
    const [device, dtype] = key.split("|");
    return { key, device, dtype, data: rows.filter((r) => r.device === device && r.dtype === dtype) };
  });
  const dy = labelOffsets(Object.fromEntries(series.map((s) => [s.key, s.data.at(-1)?.tflops ?? 0])), (a, b) => a / b < 2);
  const yTicks = [0.1, 0.3, 1, 3, 10, 30].filter((t) => t <= 3 * Math.max(...rows.map((r) => r.tflops)));
  return (
    <section aria-label="Matmul throughput">
      <div className="hero-chart" style={{ height: narrow() ? 440 : 320 }}>
        <ResponsiveContainer>
          <LineChart margin={{ top: 16, right: narrow() ? 16 : 170, bottom: 28, left: 0 }}>
            <CartesianGrid stroke={RULE} strokeDasharray="2 4" />
            <XAxis dataKey="n" type="number" scale="log" domain={[256, 4096]} ticks={[256, 512, 1024, 2048, 4096]} allowDuplicatedCategory={false}
              stroke={INK} tick={{ fontSize: 12 }}>
              <Label value="n" position="insideBottom" offset={-16} fill={INK} fontSize={12} />
            </XAxis>
            <YAxis type="number" scale="log" domain={[yTicks[0], yTicks[yTicks.length - 1]]} ticks={yTicks} allowDataOverflow
              stroke={INK} width={56} tick={{ fontSize: 12 }} tickFormatter={(v) => formatNumber(Number(v))}>
              <Label value="TFLOP/s" angle={-90} position="insideLeft" fill={INK} fontSize={12} />
            </YAxis>
            <Tooltip formatter={(v) => `${formatNumber(Number(v))} TFLOP/s`} labelFormatter={(n) => `n = ${n}`} />
            <Legend verticalAlign="top" formatter={(v) => <span style={{ color: INK }}>{v}</span>} />
            {series.map((s) => {
              const name = `${DEVICE_NAME[s.device]}, ${s.dtype}`;
              return (
                <Line key={name} data={s.data} dataKey="tflops" name={name} stroke={DEVICE_COLOR[s.device]} strokeWidth={2}
                  strokeDasharray={s.dtype === "float32" ? undefined : "6 4"} dot={{ r: 4, fill: DEVICE_COLOR[s.device], stroke: "#fafbfc", strokeWidth: 2 }}
                  isAnimationActive={false} label={narrow() ? undefined : endLabel(s.data.length, `${DEVICE_SHORT[s.device]} ${s.dtype}`, INK, dy[s.key])} />
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <details>
        <summary>Every measurement as a table</summary>
        <table>
          <thead><tr><th>device</th><th>dtype</th><th className="num">n</th><th className="num">TFLOP/s</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.device}${r.dtype}${r.n}`}>
                <td>{DEVICE_NAME[r.device]}</td><td>{r.dtype}</td><td className="num">{r.n}</td><td className="num">{formatNumber(r.tflops)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}
