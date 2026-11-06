import { Line, LineChart, ResponsiveContainer, YAxis } from "recharts";
import { RULE } from "../chart";
import { formatNumber } from "../lib";
import { Fig, int, num, pct, pval, sci, useResults } from "./kit";

const FWD = "#2456b8";
const GRAD = "#a3266f";

export function Head() {
  const r = useResults();
  return (
    <>
      <header className="paper-title">
        <div className="col">
          <p className="kicker">minitorch:</p>
          <h1>Deep learning framework from scratch</h1>
        </div>
      </header>
      <div className="byline-band">
        <div className="byline col">
          <div>
            <h3>Author</h3>
            <p><a href="https://samitmohan.in">Samit Mohan</a></p>
          </div>
          <div>
            <h3>Hardware</h3>
            <p>{r.ovM?.machine ?? "…"} laptop</p>
            <p>{r.ovC?.gpu ?? "…"}</p>
          </div>
          <div>
            <h3>Code</h3>
            <p><a href="https://github.com/samitmohan/minitorch">github.com/samitmohan/minitorch</a></p>
          </div>
        </div>
      </div>
    </>
  );
}

export function Tldr() {
  const r = useResults();
  return (
    <div className="tldr">
      <b>Abstract</b>
      <p>
        minitorch is a deep learning framework written from scratch in {int(r.lines?.package)} lines of Python, built in the
        order a study plan would take: reverse-mode automatic differentiation, a transformer trained from nothing, GPU backends
        and attention kernels, a real pretrained model, a server, fine-tuning, and on top of it all the two applications
        most language-model systems exist for, retrieval-augmented generation and a tool-using agent. Each layer is checked
        against a reference. The gradients match PyTorch's to {sci(r.p64?.rows[0].abs_diff)}; a{" "}
        {r.tsParams ? `${(r.tsParams / 1e6).toFixed(1)}M` : "…"}-parameter model trains on TinyStories to a validation loss of{" "}
        {num(r.finalVal)} in {num(r.minutes, 2)} minutes on one {r.ovC?.gpu?.replace("NVIDIA GeForce ", "") ?? "…"}; and Qwen3-0.6B's
        logits match Hugging Face's to {sci(r.qwenCpu?.max_gap)}. Through minitorch's own server, retrieval puts the paragraph
        a SQuAD question was written from among its top three {r.rag ? pct(r.rag.recall_at_3, 1) : "…"} of the time, and{" "}
        {r.rag ? pct(r.rag.rag_contains, 0) : "…"} of answers then contain the reference.
        {r.gsmBase && r.gsmLora && r.gsmGrpo && r.agent && r.vsLora && r.vsGrpo && r.vsAgent && (
          <> Three attempts to make the model better at grade-school math failed: LoRA fine-tuning lost{" "}
            {r.gsmBase.correct - r.gsmLora.correct} of {r.gsmBase.n} problems ({pval(r.vsLora.p_value)}), GRPO changed nothing
            measurable ({pval(r.vsGrpo.p_value)}), and a calculator tool lost {r.gsmBase.correct - r.agent.correct} ({pval(r.vsAgent.p_value)}).</>
        )}
      </p>
    </div>
  );
}

export function Teaser() {
  const r = useResults();
  const bars = (r.attn ?? []).filter((x) => x.shape === "CS336 training" && x.dtype === "float32");
  const slowest = Math.max(1e-9, ...bars.map((x) => x.ms));
  // Apple's kernel in pink, my fastest in blue, the earlier attempts in gray
  const barColor = (impl: string) => {
    if (impl.startsWith("mx.")) return GRAD;
    if (impl.startsWith("v3")) return FWD;
    return RULE;
  };
  return (
    <Fig n={1} caption={<>Three results, each read live from the project's result files. Left: minitorch (thin blue) and PyTorch (wide pink) train the same model and their losses overlap step for step. Middle: the story model's validation loss over {int(r.cfg?.steps)} training steps. Right: time to compute attention at the training shape, shorter is faster: my kernel in blue, Apple's in pink, earlier attempts in gray.</>}>
      <div className="teaser">
        <div>
          <div className="spark">
            {r.p64 && (
              <ResponsiveContainer>
                <LineChart data={r.p64.rows} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
                  <Line dataKey="torch" stroke={GRAD} strokeWidth={5} strokeOpacity={0.35} dot={false} isAnimationActive={false} />
                  <Line dataKey="minitorch" stroke={FWD} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
          <b>{sci(r.p64?.rows[0].abs_diff)}</b>
          <span>gap to PyTorch's loss at step 1</span>
        </div>
        <div>
          <div className="spark">
            {r.evals.length > 0 && (
              <ResponsiveContainer>
                <LineChart data={r.evals} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
                  <YAxis hide scale="log" domain={["auto", "auto"]} />
                  <Line dataKey="val_loss" stroke={GRAD} strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
          <b>{num(r.finalVal)}</b>
          <span>validation loss on TinyStories</span>
        </div>
        <div>
          <div className="spark bars-mini" role="img" aria-label={`Attention time at the training shape: ${bars.map((x) => `${x.impl} ${formatNumber(x.ms, 3)} ms`).join(", ")}`}>
            {bars.map((x) => (
              <div key={x.impl} title={`${x.impl}: ${formatNumber(x.ms, 3)} ms`}>
                <span style={{ width: `${(100 * x.ms) / slowest}%`, background: barColor(x.impl) }} />
              </div>
            ))}
          </div>
          <b>{num(r.kernelWin, 2)}x</b>
          <span>faster than Apple's attention kernel</span>
        </div>
      </div>
    </Fig>
  );
}
