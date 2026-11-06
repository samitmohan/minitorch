import { useState } from "react";
import { api, type RagResult } from "../api";

// questions from the SQuAD development set the index covers
const EXAMPLES = [
  "What language did Tesla study while in school?",
  "Where was the Charles Porter steam engine indicator shown?",
  "What is the chloroplast double membrane sometimes compared to?",
];

/** Retrieval, live: the question is embedded, the three nearest paragraphs come back, and the model answers from them. */
export function RagDemo() {
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [result, setResult] = useState<RagResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = (q: string) => {
    setBusy(true);
    setError(null);
    api.rag(q).then(setResult, (e) => setError(e.message)).finally(() => setBusy(false));
  };

  return (
    <section aria-label="Retrieval demo" className="demo">
      <form onSubmit={(e) => {
        e.preventDefault();
        run(question);
      }} className="row">
        <label className="grow">
          Question
          <input value={question} maxLength={1000} onChange={(e) => setQuestion(e.target.value)} />
        </label>
        <button type="submit" disabled={busy || !question.trim()}>{busy ? "Searching…" : "Search and answer"}</button>
      </form>
      <p className="muted">
        Or try:{" "}
        {EXAMPLES.slice(1).map((q) => (
          <button key={q} type="button" className="link" disabled={busy} onClick={() => {
            setQuestion(q);
            run(q);
          }}>{q}</button>
        ))}
      </p>
      {error && <p className="error">{error}</p>}
      {result && (
        <>
          <p className="answer"><b>Answer:</b> {result.answer}</p>
          {result.passages.map((p, i) => (
            <details key={p.title + i} open={i === 0} className="passage">
              <summary>[{i + 1}] {p.title} <small>similarity {p.score.toFixed(3)}</small></summary>
              <p>{p.text}</p>
            </details>
          ))}
          <p className="muted">
            Embedding the question took {result.embed_ms} ms, the search over every paragraph {result.search_ms} ms, and the
            answer {result.answer_s} s.
          </p>
        </>
      )}
    </section>
  );
}
