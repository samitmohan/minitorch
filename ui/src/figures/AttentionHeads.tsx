import { useEffect, useRef, useState } from "react";
import { api, type AttentionPatterns } from "../api";

// one hue, light to dark: weight 0 is the sheet, weight 1 is cobalt; a square root lifts the faint patterns
const shade = (w: number) => {
  const t = Math.sqrt(w);
  const mix = (a: number, b: number) => Math.round(a + (b - a) * t);
  return `rgb(${mix(250, 36)}, ${mix(251, 86)}, ${mix(252, 184)})`;
};

function Thumb({ m, selected, onPick, label }: { m: number[][]; selected: boolean; onPick: () => void; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    const n = m.length;
    const cell = 36 / n;
    m.forEach((row, i) => row.forEach((w, j) => {
      ctx.fillStyle = shade(w);
      ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
    }));
  }, [m]);
  return (
    <button className={`thumb${selected ? " picked" : ""}`} onClick={onPick} aria-label={label} aria-pressed={selected}>
      <canvas ref={ref} width={36} height={36} />
    </button>
  );
}

/** Every attention head of the story model on a sentence you type: rows are layers, columns are heads. */
export function AttentionHeads() {
  const [text, setText] = useState("Once upon a time, there was a little girl named Lily. Lily loved to play with her red ball.");
  const [data, setData] = useState<AttentionPatterns | null>(null);
  const [pick, setPick] = useState<[number, number]>([2, 10]);  // layer 3, head 11: the post describes it
  const [error, setError] = useState<string | null>(null);
  const run = () => api.attention(text).then(
    (d) => {
      setData(d);
      setError(null);
    },
    (e) => setError(e.message),
  );
  useEffect(() => void run(), []);

  const m = data?.layers[pick[0]]?.[pick[1]];
  const n = data?.tokens.length ?? 0;
  const cell = Math.min(18, Math.floor(520 / Math.max(n, 1)));
  const label = (t: string) => t.replace(/\n/g, "↵").trim() || "␣";
  return (
    <div className="attention-heads" aria-label="Attention heads">
      <form className="row" onSubmit={(e) => {
        e.preventDefault();
        run();
      }}>
        <label style={{ flex: 1 }}>
          Text
          <input value={text} onChange={(e) => setText(e.target.value)} />
        </label>
        <button type="submit">Show attention</button>
      </form>
      {error && <p className="error">{error}</p>}
      {data && (
        <>
          <div className="head-grid" style={{ gridTemplateColumns: `auto repeat(${data.layers[0].length}, 38px)` }}>
            {data.layers.map((heads, l) => [
              <span key={`l${l}`} className="layer-label">layer {l + 1}</span>,
              ...heads.map((h, hi) => (
                <Thumb key={`${l}-${hi}`} m={h} selected={pick[0] === l && pick[1] === hi} onPick={() => setPick([l, hi])}
                  label={`layer ${l + 1}, head ${hi + 1}`} />
              )),
            ])}
          </div>
          {m && (
            <div className="head-detail">
              <b>Layer {pick[0] + 1}, head {pick[1] + 1}.</b> Each row is a token looking back; each column is a token being looked
              at. Darker means more weight.
              <svg width={cell * n + 110} height={cell * n + 110} role="img" aria-label="Selected head's attention weights">
                {data.tokens.map((t, j) => (
                  <text key={`c${j}`} transform={`translate(${100 + j * cell + cell * 0.7}, 96) rotate(-60)`} fontSize={11} fill="#1b2330">{label(t)}</text>
                ))}
                {data.tokens.map((t, i) => (
                  <text key={`r${i}`} x={96} y={100 + i * cell + cell * 0.75} fontSize={11} textAnchor="end" fill="#1b2330">{label(t)}</text>
                ))}
                {m.map((row, i) => row.map((w, j) => j <= i && (
                  <rect key={`${i}-${j}`} x={100 + j * cell} y={100 + i * cell} width={cell - 1} height={cell - 1} fill={shade(w)}>
                    <title>{`${label(data.tokens[i])} → ${label(data.tokens[j])}: ${w.toFixed(3)}`}</title>
                  </rect>
                )))}
              </svg>
            </div>
          )}
        </>
      )}
    </div>
  );
}
