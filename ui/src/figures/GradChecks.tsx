import { useState } from "react";
import { api, type GradCheck } from "../api";

export function GradChecks() {
  const [results, setResults] = useState<GradCheck[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = () => {
    setBusy(true);
    api.gradcheck().then(setResults, (e) => setError(e.message)).finally(() => setBusy(false));
  };


  const passed = results?.filter((r) => r.ok).length ?? 0;
  return (
    <section aria-label="Gradient checks">
      <div className="row">
        <button onClick={run} disabled={busy}>{busy ? "Checking…" : "Run gradient checks"}</button>
        {results && <span>{passed} of {results.length} match</span>}
      </div>
      {error && <p className="error">{error}</p>}
      {results && (
        <ul className="checks">
          {results.map((r) => (
            <li key={r.name} className={r.ok ? "ok" : "fail"} title={r.error}>
              {r.ok ? "✓" : "✗"} {r.name}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
