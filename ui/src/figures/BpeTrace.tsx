import { useEffect, useState } from "react";
import { api, type Bpe } from "../api";

export function BpeTrace() {
  const [text, setText] = useState("Once upon a time, a curious little robot found a shiny gear.");
  const [bpe, setBpe] = useState<Bpe | null>(null);
  const [picked, setPicked] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const run = () => api.bpe(text).then(
    (b) => {
      setBpe(b);
      setPicked(Math.min(3, b.trace.length - 1));
      setError(null);
    },
    (e) => setError(e.message),
  );
  useEffect(() => void run(), []);

  const steps = bpe?.trace[picked]?.steps ?? [];
  return (
    <section aria-label="BPE tokenizer">
      <form className="row" onSubmit={(e) => {
        e.preventDefault();
        run();
      }}>
        <label style={{ flex: 1 }}>
          Text
          <textarea rows={2} value={text} onChange={(e) => setText(e.target.value)} />
        </label>
        <button type="submit">Tokenize</button>
      </form>
      {error && <p className="error">{error}</p>}
      {bpe && (
        <>
          <p className="muted">
            {bpe.ids.length} tokens from {new TextEncoder().encode(text).length} bytes with the {bpe.tokenizer} vocabulary.
          </p>
          <div className="tokens code">
            {bpe.tokens.map((t, i) => (
              <span className="tok" key={i}>
                {t.replace(/ /g, "␣").replace(/\n/g, "↵")}
                <small>{bpe.ids[i]}</small>
              </span>
            ))}
          </div>
          <h3>Merges for one pre-token</h3>
          <div className="pretokens code" role="group" aria-label="Pre-tokens">
            {bpe.trace.map((t, i) => (
              <button key={i} aria-pressed={i === picked} onClick={() => setPicked(i)}>
                {t.pretoken.replace(/ /g, "␣").replace(/\n/g, "↵")}
              </button>
            ))}
          </div>
          <div className="ladder">
            <ol className="code">
              {steps.map((parts, i) => (
                <li key={i}>
                  <span className="tokens">
                    {parts.map((p, j) => (
                      <span className="tok" key={j}>{p.replace(/ /g, "␣").replace(/\n/g, "↵")}</span>
                    ))}
                  </span>
                </li>
              ))}
            </ol>
            <p className="muted">
              {steps.length - 1} merge{steps.length === 2 ? "" : "s"}, from bytes to final tokens.
            </p>
          </div>
        </>
      )}
    </section>
  );
}
