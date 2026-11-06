import { useMemo } from "react";
import { CartesianGrid, Label, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { INK, RULE } from "../chart";
import { downsample, formatNumber } from "../lib";

const FWD = "#2456b8";
const GRAD = "#a3266f";

type Row = { step: number; val_loss?: number } & Record<string, number | undefined>;

/** The mean of `field` over each step's last `n` steps. */
function runningMean(rows: Row[], field: string, n: number) {
  return rows.map((r, i) => {
    const w = rows.slice(Math.max(0, i - n + 1), i + 1);
    return { step: r.step, mean: w.reduce((a, x) => a + (x[field] as number), 0) / w.length };
  });
}

/** One field of a run's metrics against the step: the train loss by default, with the validation loss when the run has one.
 * `smooth` adds a running mean over that many steps; `compare` adds another run's running mean beside it. */
export function RunCurve({ rows, field = "loss", label = "train", log = true, smooth, compare }: {
  rows: Row[]; field?: string; label?: string; log?: boolean; smooth?: number; compare?: { rows: Row[]; label: string };
}) {
  const value = (r: Row) => r[field] as number;
  const curve = useMemo(() => downsample(rows, 700), [rows]);
  const mean = useMemo(() => (smooth ? runningMean(rows, field, smooth) : []), [rows, smooth, field]);
  const other = useMemo(() => (smooth && compare ? runningMean(compare.rows, field, smooth) : []), [compare, smooth, field]);
  const evals = useMemo(() => (field === "loss" ? rows.filter((r) => r.val_loss !== undefined) : []), [rows, field]);
  const values = curve.map(value).concat(evals.map((r) => r.val_loss!));
  if (!values.length) return <p className="muted">Loading the run…</p>;
  const domain = log ? [Math.min(...values) * 0.95, Math.max(...values) * 1.05] : [0, 1];
  return (
    <div className="hero-chart" style={{ height: 320 }} aria-label={`${field} curve`}>
      <ResponsiveContainer>
        <LineChart margin={{ top: 16, right: 24, bottom: 28, left: 8 }}>
          <CartesianGrid stroke={RULE} strokeDasharray="2 4" />
          <XAxis dataKey="step" type="number" domain={[0, "dataMax"]} allowDuplicatedCategory={false} stroke={INK} tick={{ fontSize: 12 }}
            tickFormatter={(v) => Number(v).toLocaleString("en-US")}>
            <Label value="step" position="insideBottom" offset={-16} fill={INK} fontSize={12} />
          </XAxis>
          <YAxis type="number" scale={log ? "log" : "linear"} domain={domain} allowDataOverflow stroke={INK}
            width={48} tick={{ fontSize: 12 }} tickFormatter={(v) => formatNumber(Number(v), 2)} />
          <Tooltip formatter={(v) => formatNumber(Number(v))} labelFormatter={(s) => `step ${Number(s).toLocaleString("en-US")}`} />
          {(evals.length > 0 || smooth) && <Legend verticalAlign="top" height={24} formatter={(name) => <span style={{ color: INK }}>{name}</span>} />}
          <Line data={curve} dataKey={field} name={smooth ? `${label}, each step` : label} stroke={FWD} strokeWidth={1.25}
            strokeOpacity={smooth ? 0.3 : 1} dot={false} isAnimationActive={false} />
          {smooth && <Line data={mean} dataKey="mean" name={`${label}, mean of the last ${smooth} steps`} stroke={FWD} strokeWidth={2.5} dot={false} isAnimationActive={false} />}
          {compare && <Line data={other} dataKey="mean" name={`${compare.label}, mean of the last ${smooth} steps`} stroke={GRAD} strokeWidth={2.5} dot={false} isAnimationActive={false} />}
          {evals.length > 0 && (
            <Line data={evals} dataKey="val_loss" name="validation" stroke={GRAD} strokeWidth={2} dot={evals.length < 30} isAnimationActive={false} />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
