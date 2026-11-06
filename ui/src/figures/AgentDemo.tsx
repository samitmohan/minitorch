import { useState } from "react";
import { api, type AgentResult } from "../api";

const PROBLEM = "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?";

/** The calculator's argument, from the JSON the model wrote; the raw text when it is not valid JSON. */
function expression(call: string) {
  try {
    return String(JSON.parse(call).arguments.expression);
  } catch {
    return call;
  }
}

/** The agent, live: each model reply, each calculator call and its result, in order. */
export function AgentDemo() {
  const [problem, setProblem] = useState(PROBLEM);
  const [result, setResult] = useState<AgentResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = () => {
    setBusy(true);
    setError(null);
    api.agent(problem).then(setResult, (e) => setError(e.message)).finally(() => setBusy(false));
  };

  return (
    <section aria-label="Agent demo" className="demo">
      <form onSubmit={(e) => {
        e.preventDefault();
        run();
      }} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <label>
          Problem
          <textarea rows={3} value={problem} maxLength={1000} onChange={(e) => setProblem(e.target.value)} />
        </label>
        <div className="row">
          <button type="submit" disabled={busy || !problem.trim()}>{busy ? "Solving…" : "Solve with the calculator"}</button>
        </div>
      </form>
      {error && <p className="error">{error}</p>}
      {result && (
        <ol className="trace">
          {result.trace.map((step, i) => (
            <li key={i}>
              {step.kind === "model" ? (
                <>
                  <small>model, {step.reply_tokens} token{step.reply_tokens === 1 ? "" : "s"} in {step.seconds.toFixed(2)} s</small>
                  <pre>{step.text}</pre>
                </>
              ) : (
                <><small>calculator</small> <code>{expression(step.expression)}</code> = <code>{step.result}</code></>
              )}
            </li>
          ))}
          <li><b>Final answer: {result.answer ?? "none"}</b></li>
        </ol>
      )}
    </section>
  );
}
