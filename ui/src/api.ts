import type { MetricRow } from "./lib";

export type Run = {
  name: string;
  config: Record<string, number | string | null>;
  steps: number;
  last: MetricRow | null;
  has_checkpoint: boolean;
};
export type Sample = { step: number; text: string };
export type GraphNode = {
  id: number;
  label: string;
  op: string;
  shape: number[];
  value: number | number[] | null;
  grad: number | number[] | null;
  leaf: boolean;
  requires_grad: boolean;
};
export type Graph = { nodes: GraphNode[]; edges: { source: number; target: number }[]; root: number };
export type Bpe = { tokenizer: string; ids: number[]; tokens: string[]; trace: { pretoken: string; steps: string[][] }[] };
export type GradCheck = { name: string; ok: boolean; error?: string };
export type AttentionPatterns = { tokens: string[]; layers: number[][][][] };  // layers[layer][head][query][key]

async function call<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, body === undefined ? undefined : {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data as T;
}

export const api = {
  runs: () => call<Run[]>("/api/runs"),
  samples: (run: string) => call<Sample[]>(`/api/runs/${encodeURIComponent(run)}/samples`),
  graph: (expr: string, variables: Record<string, number | number[]>) => call<Graph>("/api/graph", { expr, variables }),
  bpe: (text: string) => call<Bpe>("/api/bpe", { text }),
  gradcheck: () => call<GradCheck[]>("/api/gradcheck", {}),
  generate: (req: { run: string; prompt: string; max_tokens: number; temperature: number; top_p: number }) =>
    call<{ text: string; prompt_tokens: number; new_tokens: number }>("/api/generate", req),
  bench: () => call<Record<string, Record<string, unknown>>>("/api/bench"),
  attention: (text: string) => call<AttentionPatterns>("/api/attention", { text }),
  rag: (question: string) => call<RagResult>("/api/rag", { question }),
  agent: (question: string) => call<AgentResult>("/api/agent", { question }),
};

export type RagResult = {
  passages: { title: string; text: string; score: number }[];
  answer: string;
  embed_ms: number;
  search_ms: number;
  answer_s: number;
};
export type AgentStep =
  | { kind: "model"; text: string; prompt_tokens: number; reply_tokens: number; seconds: number }
  | { kind: "tool"; expression: string; result: string; seconds: number };
export type AgentResult = { reply: string; answer: number | null; trace: AgentStep[] };

/** Every metrics row of a run, the whole history first, then each new one as training appends it. */
export function streamMetrics(run: string, onRows: (rows: MetricRow[]) => void): () => void {
  const source = new EventSource(`/api/runs/${encodeURIComponent(run)}/stream`);
  let pending: MetricRow[] = [];
  let lastStep = -1;
  source.onmessage = (e) => {
    const row: MetricRow = JSON.parse(e.data);
    // EventSource reconnects on its own after a drop, and the server replays the file from the start
    if (row.step > lastStep) {
      lastStep = row.step;
      pending.push(row);
    }
  };
  // the server ends a finished run's stream; without this, EventSource would reconnect and replay it
  source.addEventListener("end", () => source.close());
  // batch updates: a 40k-step history arrives as 40k events, which must not mean 40k renders
  const timer = setInterval(() => {
    if (pending.length) {
      onRows(pending);
      pending = [];
    }
  }, 250);
  return () => {
    clearInterval(timer);
    source.close();
  };
}
