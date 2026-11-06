import { useEffect, useState } from "react";
import { api, type Run } from "../api";

export function Generate() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [run, setRun] = useState("");
  const [prompt, setPrompt] = useState("Once upon a time, there was a little");
  const [temperature, setTemperature] = useState(0.8);
  const [topP, setTopP] = useState(0.95);
  const [maxTokens, setMaxTokens] = useState(120);
  const [out, setOut] = useState<{ prompt: string; text: string; seconds: number; tokens: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.runs().then((rs) => {
      const ready = rs.filter((r) => r.has_checkpoint);
      setRuns(ready);
      if (ready.length) setRun(ready[ready.length - 1].name);
    }, (e) => setError(e.message));
  }, []);

  const generate = async () => {
    setBusy(true);
    setError(null);
    const t0 = performance.now();
    try {
      const r = await api.generate({ run, prompt, max_tokens: maxTokens, temperature, top_p: topP });
      setOut({ prompt, text: r.text, seconds: (performance.now() - t0) / 1000, tokens: r.new_tokens });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section aria-label="Generate">
      {!runs.length ? (
        <p className="muted">No run has saved a checkpoint yet. Training writes one at every evaluation.</p>
      ) : (
        <form onSubmit={(e) => {
          e.preventDefault();
          generate();
        }} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <label>
            Prompt
            <textarea rows={2} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          </label>
          <div className="row">
            <label>
              Run
              <select value={run} onChange={(e) => setRun(e.target.value)}>
                {runs.map((r) => <option key={r.name}>{r.name}</option>)}
              </select>
            </label>
            <label>
              Temperature {temperature.toFixed(2)}
              <input type="range" min={0} max={1.5} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} />
            </label>
            <label>
              Top-p {topP.toFixed(2)}
              <input type="range" min={0.05} max={1} step={0.05} value={topP} onChange={(e) => setTopP(+e.target.value)} />
            </label>
            <label>
              New tokens
              <input type="number" min={1} max={512} value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)} style={{ width: 90 }} />
            </label>
            <button type="submit" disabled={busy}>{busy ? "Generating…" : "Generate"}</button>
          </div>
        </form>
      )}
      {error && <p className="error">{error}</p>}
      {out && (
        <>
          <div className="output">
            <span className="prompt">{out.prompt}</span>
            {out.text}
          </div>
          <p className="muted">
            {out.tokens} tokens in {out.seconds.toFixed(2)} s ({(out.tokens / out.seconds).toFixed(1)} tok/s, NumPy on CPU, KV cache on).
          </p>
        </>
      )}
    </section>
  );
}

