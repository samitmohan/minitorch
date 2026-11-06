import { useState } from "react";

type Turn = { role: "user" | "assistant"; content: string; stats?: string };

/** Talk to Qwen3-0.6B through minitorch's serving engine, over its OpenAI-compatible endpoint, streaming. */
export function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("What is 17 times 23? Show the steps briefly.");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = async () => {
    const history: Turn[] = [...turns, { role: "user", content: input.trim() }];
    setTurns([...history, { role: "assistant", content: "" }]);
    setInput("");
    setBusy(true);
    setError(null);
    const t0 = performance.now();
    let first = 0;
    let tokens = 0;
    let text = "";
    try {
      const res = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: history.map(({ role, content }) => ({ role, content })), stream: true, max_tokens: 400, temperature: 0.7 }),
      });
      if (!res.ok || !res.body) throw new Error((await res.json()).detail ?? res.statusText);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        for (const event of events) {
          const data = event.replace(/^data: /, "");
          if (data === "[DONE]") continue;
          const chunk = JSON.parse(data);
          if (chunk.error) throw new Error(chunk.error.message);
          const delta = chunk.choices[0].delta.content;
          if (delta) {
            first ||= performance.now();
            text += delta;
            setTurns([...history, { role: "assistant", content: text }]);
          }
          if (chunk.usage) tokens = chunk.usage.completion_tokens;
        }
      }
      const end = performance.now();
      const stats = `first token after ${((first - t0) / 1000).toFixed(2)} s, then ${(tokens / ((end - first) / 1000)).toFixed(1)} tokens/s`;
      setTurns([...history, { role: "assistant", content: text, stats }]);
    } catch (e) {
      setError((e as Error).message);
      setTurns(history);
    } finally {
      setBusy(false);
    }
  };

  const submit = () => {
    if (input.trim() && !busy) send();
  };

  return (
    <div className="chat" aria-label="Chat with Qwen3">
      <div className="turns" aria-live="polite">
        {turns.length === 0 && <p className="muted">The first message loads the model into the GPU; later ones start at once.</p>}
        {turns.map((t, i) => (
          <div key={i} className={`turn ${t.role}`}>
            <p>{t.content || (busy && i === turns.length - 1 ? "…" : "")}</p>
            {t.stats && <small>{t.stats}</small>}
          </div>
        ))}
      </div>
      <form className="row" onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}>
        <label style={{ flex: 1 }}>
          Message
          <textarea rows={2} value={input} onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }} />
        </label>
        <button type="submit" disabled={busy || !input.trim()}>{busy ? "Answering…" : "Send"}</button>
        {turns.length > 0 && <button type="button" className="quiet" disabled={busy} onClick={() => setTurns([])}>Clear</button>}
      </form>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
