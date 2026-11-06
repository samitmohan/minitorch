import { CartesianGrid, Label, Legend, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { endLabel, INK, RULE } from "../chart";
import { formatNumber } from "../lib";

// one hue, light to dark as the compute budget grows; each step is 3:1 or more against the figure background
const RAMP = ["#6b8fd6", "#3d68c4", "#1d3f8f"];

export type ScalingRun = { budget: number; d_model: number; params: number; val_loss: number };
export type Optimum = { budget: number; params: number; loss: number };

const flops = (b: number) => `${b.toExponential(0).replace("e+", "e")} FLOPs`;
const millions = (p: number) => `${formatNumber(p / 1e6, 2)}M`;

/** Final validation loss against model size, one curve per compute budget; a ring marks each curve's fitted bottom. */
export function IsoFlop({ runs, optima }: { runs: ScalingRun[]; optima: Optimum[] }) {
  const budgets = [...new Set(runs.map((r) => r.budget))].sort((a, b) => a - b);
  return (
    <>
      <div className="hero-chart" style={{ height: 340 }} aria-label="IsoFLOP curves">
        <ResponsiveContainer>
          <LineChart margin={{ top: 16, right: 104, bottom: 28, left: 8 }}>
            <CartesianGrid stroke={RULE} strokeDasharray="2 4" />
            <XAxis dataKey="params" type="number" scale="log" domain={["dataMin", "dataMax"]} allowDuplicatedCategory={false}
              stroke={INK} tick={{ fontSize: 12 }} tickFormatter={millions}>
              <Label value="parameters in the matrix products" position="insideBottom" offset={-16} fill={INK} fontSize={12} />
            </XAxis>
            <YAxis type="number" domain={["auto", "auto"]} stroke={INK} width={48} tick={{ fontSize: 12 }}
              tickFormatter={(v) => formatNumber(Number(v), 3)} />
            <Tooltip formatter={(v) => formatNumber(Number(v), 4)} labelFormatter={(p) => `${millions(Number(p))} parameters`} />
            <Legend verticalAlign="top" height={24} formatter={(name) => <span style={{ color: INK }}>{name}</span>}
              itemSorter={(item) => budgets.findIndex((b) => flops(b) === item.value)} />
            {budgets.map((b, i) => {
              const pts = runs.filter((r) => r.budget === b).sort((x, y) => x.params - y.params);
              return (
                <Line key={b} data={pts} dataKey="val_loss" name={flops(b)} stroke={RAMP[i]} strokeWidth={2}
                  dot={{ r: 4, fill: RAMP[i], stroke: RAMP[i] }} label={endLabel(pts.length, flops(b), RAMP[i])} isAnimationActive={false} />
              );
            })}
            {optima.map((o) => (
              <ReferenceDot key={o.budget} x={o.params} y={o.loss} r={7} fill="none"
                stroke={RAMP[budgets.indexOf(o.budget)]} strokeWidth={2} ifOverflow="extendDomain" />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <details>
        <summary>Every run as a table</summary>
        <table>
          <thead><tr><th>budget</th><th className="num">d_model</th><th className="num">parameters</th><th className="num">validation loss</th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={`${r.budget}${r.d_model}`}>
                <td>{flops(r.budget)}</td><td className="num">{r.d_model}</td>
                <td className="num">{millions(r.params)}</td><td className="num">{formatNumber(r.val_loss, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </>
  );
}
