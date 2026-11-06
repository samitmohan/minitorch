import { useState } from "react";
import { IsoFlop } from "../figures/IsoFlop";
import { DecodingAppendix } from "./RealModel";
import { Cite, Fig, REFS, Section, TeX, int, num, pct, sci, useResults } from "./kit";

const GLOSSARY: [string, string][] = [
  ["Activation", "Any intermediate value a network computes on the way from input to output."],
  ["Advantage", "How much better a sampled reply scored than the other replies to the same question; GRPO makes replies with a positive advantage likelier (§12)."],
  ["Agent", "A program that lets a language model call tools, here a calculator, and feeds the results back into the conversation (§13)."],
  ["Attention", "The step where each position gathers information from earlier positions, weighted by how well its query matches their keys (§4)."],
  ["Autograd", "Automatic differentiation: a program that records a computation and then computes the gradient of its result with respect to every input (§2)."],
  ["Backend", "The array library that does the arithmetic: NumPy on the CPU, MLX on Apple's GPU, CuPy on NVIDIA's (§6)."],
  ["Bandwidth", "How many bytes per second a chip can move between its memory and its cores."],
  ["bfloat16", "A 16-bit number format with float32's range and two to three significant digits (§8)."],
  ["Convolution", "Sliding a small filter over a grid of features and taking a weighted sum under each placement (appendix, Other models)."],
  ["Cross-entropy", "The usual loss for predicting the next token: minus the log of the probability given to the right one (§5)."],
  ["Diffusion model", "A model that learns to remove noise, and draws by starting from pure noise and removing it step by step (appendix, Other models)."],
  ["Embedding", "The learned vector each token id is replaced by before the first layer."],
  ["FLOP", "One floating-point operation, such as a multiply or an add. TFLOP/s is trillions of them per second."],
  ["Gradient", "For each weight, how much the loss changes when that weight moves a little. Training steps against it (§2)."],
  ["GRPO", "Group relative policy optimization: sample several replies per question, score each, and train toward the ones that beat their group (§12)."],
  ["KV cache", "Stored keys and values of earlier tokens, so generating each new token does not recompute them (§9)."],
  ["Kernel", "A small program that a GPU runs on many pieces of an array at once."],
  ["Logit", "The model's raw score for a vocabulary entry; softmax turns logits into probabilities."],
  ["LoRA", "Fine-tuning that freezes the original weights and trains a small low-rank correction beside each matrix (§11)."],
  ["Loss", "One number saying how wrong the model is; training lowers it."],
  ["MFU", "Model FLOPs utilization: the arithmetic a step must do divided by what the chip could do at its rated peak (§6)."],
  ["Parameter (weight)", "One adjustable number inside the network."],
  ["Perplexity", "The exponential of the loss. A perplexity of 4 means the model is as unsure as a uniform choice among 4 tokens."],
  ["Policy gradient", "The gradient of a model's expected reward, estimated from replies it sampled itself (§12)."],
  ["Quantization", "Storing weights as small integers plus a scale per group, to cut memory and bandwidth (§10)."],
  ["Retrieval-augmented generation (RAG)", "Answering from documents: embed the question, find the nearest passages, and put them in the prompt (§13)."],
  ["Speculative decoding", "A small model guesses the next few tokens and the large one checks them all in one pass, keeping the ones it agrees with (§10)."],
  ["Softmax", "Turns a list of scores into positive weights that add up to 1, giving larger scores larger weights."],
  ["Tensor", "An array of numbers with any number of dimensions: a vector, a matrix, or larger."],
  ["Token", "A piece of text, a word or part of one, that the model reads as a single id (§3)."],
];

const BIBTEX = `@misc{mohan2026minitorch,
  author = {Samit Mohan},
  title  = {Deep learning framework from scratch},
  year   = {2026},
  url    = {https://github.com/samitmohan/minitorch}
}`;

export function Appendix() {
  const [copied, setCopied] = useState(false);
  const r = useResults();
  const mp = (m: typeof r.mpC, precision: string) => m?.rows.find((x) => x.precision === precision);
  const c32 = mp(r.mpC, "float32"), c16 = mp(r.mpC, "float16"), m32 = mp(r.mpM, "float32"), m16 = mp(r.mpM, "bfloat16");
  const mmMs = r.profM?.rows.filter((x) => x.op === "Matmul" || x.op === "Matmul backward").reduce((a, x) => a + x.ms, 0);
  const mmShare = mmMs !== undefined && r.profM ? mmMs / r.profM.profiled_ms : undefined;  // matrix products' share of a profiled Metal step
  const f32Peak = r.peakMatmul("metal"), bf16Peak = r.peakMatmul("metal", "bfloat16");
  const ck = (c: typeof r.ckC, on: boolean, micro: number) => c?.rows.find((x) => x.checkpointing === on && x.micro === micro);
  const fullOff = ck(r.ckC, false, 8), fullOn = ck(r.ckC, true, 8), bestOff = r.ckC?.rows.filter((x) => !x.checkpointing && x.peak_gb).at(-1);
  const ro = r.rollout, many = ro?.rows.at(-1);
  const decodeS = many ? (320 * many.decode_ms_per_step) / 1000 : undefined;
  const sc = r.scaling, opt = sc ? [...sc.optima].sort((a, b) => a.budget - b.budget) : [];
  const tex10 = (v?: number) => {  // 3.5e16 -> 3.5 \times 10^{16}
    if (v === undefined) return "…";
    const e = Math.floor(Math.log10(v));
    return String.raw`${Number((v / 10 ** e).toPrecision(2))} \times 10^{${e}}`;
  };
  const usedN = sc && r.tsTokens ? sc.cs336_budget / (6 * r.tsTokens) : undefined;  // the run's matrix-product parameters
  const exp2 = opt.length >= 3 ? Math.log(opt[2].params / opt[1].params) / Math.log(opt[2].budget / opt[1].budget) : undefined;
  const n2 = sc && exp2 !== undefined ? opt[2].params * (sc.cs336_budget / opt[2].budget) ** exp2 : undefined;
  const bkM = r.bkM, bkC = r.bkC, s = r.serve;
  return (
    <Section id="appendix" title="Appendix">
      <h3 id="glossary">Glossary</h3>
      <dl className="glossary">
        {GLOSSARY.map(([term, meaning]) => (
          <div key={term}>
            <dt>{term}</dt>
            <dd>{meaning}</dd>
          </div>
        ))}
      </dl>

      <h3 id="tests">Test results</h3>
      {bkM && (
        <Fig n="A1" table caption={<>Each GPU backend against NumPy, from <code>tests/backends.py</code> (§6). A gap is the largest absolute difference between the two backends' losses or outputs. The last row checks that a bfloat16 model (§8) never falls back to 32-bit numbers.</>}>
          <table className="results">
            <thead><tr><th>Check</th><th className="num">Apple GPU (MLX)</th><th className="num">NVIDIA GPU (CuPy)</th></tr></thead>
            <tbody>
              <tr><th>Gradient checks that agree</th><td className="num">{bkM.ops.agree} of {bkM.ops.total}</td><td className="num">{bkC ? `${bkC.ops.agree} of ${bkC.ops.total}` : "…"}</td></tr>
              <tr><th>Story model loss, {bkM.training.steps} steps</th><td className="num">{sci(bkM.training.max_gap)}</td><td className="num">{sci(bkC?.training.max_gap)}</td></tr>
              <tr><th>Digit classifier loss, {bkM.mnist?.steps ?? "…"} steps</th><td className="num">{sci(bkM.mnist?.max_gap)}</td><td className="num">{sci(bkC?.mnist?.max_gap)}</td></tr>
              <tr><th>Outputs while generating</th><td className="num">{sci(bkM.decode.max_logit_gap)}</td><td className="num">{sci(bkC?.decode.max_logit_gap)}</td></tr>
              <tr><th>bfloat16 values promoted to 32 bits</th><td className="num">{bkM.bfloat16 ? `${bkM.bfloat16.promoted} of ${bkM.bfloat16.nodes}` : "…"}</td><td className="num">not applicable</td></tr>
            </tbody>
          </table>
        </Fig>
      )}
      {s && (
        <Fig n="A2" table caption={<>The scheduler's correctness test, from <code>tests/serve.py</code> (§9). Each request must receive the same tokens it gets when it runs alone. A block holds {s.block_size} tokens.</>}>
          <table className="results">
            <thead><tr><th>Load</th><th className="num">Requests</th><th className="num">Cache blocks</th><th className="num">Batch limit</th><th className="num">Steps</th><th className="num">Preemptions</th><th>Same tokens</th></tr></thead>
            <tbody>
              {s.scenarios.map((x) => (
                <tr key={x.scenario}>
                  <th>{x.scenario}</th><td className="num">{int(x.requests)}</td><td className="num">{int(x.num_blocks)}</td><td className="num">{int(x.max_batch)}</td>
                  <td className="num">{int(x.steps)}</td><td className="num">{int(x.preemptions)}</td><td>{x.identical ? "yes" : "no"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Fig>
      )}

      {r.profM && (
        <Fig n="A3" table caption={<>One forward and backward pass of the story model, {r.profM.batch} sequences of {r.profM.context} tokens, timed one primitive at a time by <code>bench/profile_step.py</code> (§7). A backward row is that primitive's gradient, and "between ops" is the time outside every primitive. Shares are of the profiled total.</>}>
          <table className="results">
            <thead><tr><th>Primitive</th><th className="num">Calls</th><th className="num">Apple GPU, ms</th><th className="num">Share</th><th className="num">NVIDIA GPU, ms</th><th className="num">Share</th></tr></thead>
            <tbody>
              {r.profM.rows.map((x) => {
                const c = r.profC?.rows.find((y) => y.op === x.op);
                return (
                  <tr key={x.op}>
                    <th>{x.op}</th><td className="num">{int(x.calls)}</td>
                    <td className="num">{num(x.ms, 3)}</td><td className="num">{pct(x.ms / r.profM!.profiled_ms)}</td>
                    <td className="num">{num(c?.ms, 3)}</td><td className="num">{c && r.profC ? pct(c.ms / r.profC.profiled_ms) : "…"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Fig>
      )}

      {r.fa2 && (
        <Fig n="A4" table caption={<>Attention on the {r.fa2.device}, {r.fa2.dtype}, from <code>bench/flash_attention_cuda.py</code> (§7): the version composed from minitorch's primitives against the Triton kernels, in milliseconds.</>}>
          <table className="results">
            <thead><tr><th>Shape</th><th>Version</th><th className="num">Forward</th><th className="num">Forward and backward</th></tr></thead>
            <tbody>
              {r.fa2.rows.map((x) => (
                <tr key={x.shape + x.impl}><th>{x.shape}</th><td>{x.impl}</td><td className="num">{num(x.forward_ms, 3)}</td><td className="num">{num(x.forward_backward_ms, 3)}</td></tr>
              ))}
            </tbody>
          </table>
        </Fig>
      )}


      {r.abl && (
        <Fig n="A5" table caption={<>The ablations of §5, from <code>scripts/ablations.py</code>: the story model trained for {int(r.abl[0]?.steps)} steps on the NVIDIA GPU, changing one thing at a time. A diverged run's loss is no longer a number.</>}>
          <table className="results">
            <thead><tr><th>Change</th><th className="num">Learning rate</th><th className="num">Final validation loss</th><th className="num">Tokens / s</th></tr></thead>
            <tbody>
              {r.abl.map((x) => (
                <tr key={x.run}><th>{x.change}</th><td className="num">{x.lr}</td>
                  <td className="num">{x.diverged_at_step !== null ? `diverged at step ${int(x.diverged_at_step)}` : num(x.final_val_loss ?? undefined)}</td>
                  <td className="num">{int(x.tokens_per_s)}</td></tr>
              ))}
            </tbody>
          </table>
        </Fig>
      )}

      {sc && opt.length >= 3 && usedN && r.tsTokens && (
        <>
          <h3 id="scaling">Scaling laws (CS336 assignment 3)</h3>
          <p>
            Was the story model of §5 the right size for its compute? Training costs about six floating-point operations per parameter per token: a multiply and an add in the forward
            pass, and twice that in the backward. The full run spent <TeX>{tex10(sc.cs336_budget)}</TeX> of them. The same
            compute could buy a larger model trained on fewer tokens, or a smaller one on more. Hoffmann et al.{" "}
            <Cite k="chinchilla" /> find the best split with IsoFLOP curves: fix a budget <TeX>C</TeX>, train models of several
            sizes <TeX>N</TeX>, each on <TeX>C/6N</TeX> tokens, and see which ends lowest. I ran three budgets, from{" "}
            <TeX>{tex10(opt[0].budget)}</TeX> to <TeX>{tex10(opt[2].budget)}</TeX> operations, with five sizes each (Figure A1).
          </p>
          <Fig n="A1" caption={<>IsoFLOP curves from <code>scripts/scaling.py</code>: the final validation loss of TinyStories models of five sizes, each trained on exactly its budget's tokens. Darker lines spend more compute. A ring marks the bottom of the parabola fitted to each curve.</>}>
            <IsoFlop runs={sc.runs} optima={sc.optima} />
          </Fig>
          <p>
            Each curve has a bottom. A model too small cannot hold what the data teaches, and one too large sees too few tokens
            to fit its weights. A parabola through each curve, in the logarithm of the size, puts the best size at{" "}
            {opt.map((o) => `${num(o.params / 1e6, 2)}M`).join(", ").replace(/, ([^,]*)$/, ", and $1")} parameters, which grows
            as <TeX>{String.raw`C^{${sc.exponent.toFixed(2)}}`}</TeX>. That exponent is loose. The smallest budget's losses only
            rise from its smallest model on, so its bottom sits at the edge of the grid and its value leans on the parabola; the
            two larger budgets alone give <TeX>{String.raw`C^{${exp2!.toFixed(2)}}`}</TeX>.
          </p>
          <p>
            Carried {int(sc.cs336_budget / opt[2].budget)} times past the largest budget, to the full run's compute, the fit asks
            for {num(sc.cs336_params_predicted / 1e6, 2)}M parameters on {num(sc.cs336_tokens_predicted / 1e6, 3)}M tokens, or{" "}
            {num(n2! / 1e6, 2)}M with the two-budget exponent. The run used {num(usedN / 1e6, 3)}M on {num(r.tsTokens / 1e6, 3)}M:
            a model {num(n2! / usedN, 2)} to {num(sc.cs336_params_predicted / usedN, 2)} times smaller than the fit's choice,
            and so, at the same compute, trained on that many times more tokens.
          </p>
        </>
      )}

      <h3 id="mixed-precision">Mixed precision (CS336 assignment 2)</h3>
      {r.mpC && c32 && c16 && (
        <>
          <p>
            A 16-bit number takes half the memory of a 32-bit one, and both GPUs multiply 16-bit matrices faster (the dashed
            lines in Figure 10). <b>Mixed precision</b> <Cite k="mixed" /> spends that speed without giving up much accuracy: the
            weights, their gradients, and the optimizer stay in float32, and only the inputs of each matrix product drop to 16
            bits. In minitorch, <code>with autocast("float16"):</code> makes every <code>Linear</code> layer cast its input and
            weight down, multiply, and cast the product back up.
          </p>
          <p>
            The two 16-bit formats fail differently. bfloat16 keeps float32's range and gives up digits. float16 keeps more
            digits but nothing smaller than about <TeX>{String.raw`6 \times 10^{-8}`}</TeX>, and many gradients are smaller
            than that, so they would become zero. A <b>loss scaler</b> multiplies the loss by <TeX>{String.raw`2^{15}`}</TeX>{" "}
            before backward, which multiplies every gradient too, and divides them back before the optimizer reads them. If a
            gradient overflows instead, the scaler skips the step and halves the scale.
          </p>
          <p>
            Trained for {int(r.mpC.steps)} steps from the same weights on the same batches, the story model on the NVIDIA GPU takes{" "}
            {num(c16.step_ms, 3)} ms per step in float16 against {num(c32.step_ms, 3)} ms in float32 (with TF32 products), and
            its mean loss over the last 50 steps differs by {num(r.mpC.last_50_loss_gap, 2)}; the scaler skipped{" "}
            {int(c16.skipped_steps)} step{c16.skipped_steps === 1 ? "" : "s"}.
            {r.mpM && m32 && m16 && <> On the Apple GPU, bfloat16 takes {num(m16.step_ms, 3)} ms per step against {num(m32.step_ms, 3)} ms in float32, with a loss gap of {num(r.mpM.last_50_loss_gap, 2)}. There the products gain less from 16 bits, {num(r.peakMatmul("metal", "bfloat16"), 3)} against {num(r.peakMatmul("metal"), 3)} TFLOP/s at their best in Figure 10, and they take {pct(mmShare, 0)} of a step (Table A3), so even with every product at the bfloat16 peak a step could shrink by only {pct(mmShare !== undefined && bf16Peak && f32Peak ? mmShare * (1 - f32Peak / bf16Peak) : undefined, 0)}.</>}
          </p>
        </>
      )}

      <h3 id="checkpointing">Activation checkpointing (CS336 assignment 2)</h3>
      {r.ckC && fullOn && bestOff && (
        <>
          <p>
            Training keeps every intermediate result of the forward pass alive until backward has used it, so memory grows
            with layers times tokens. A batch of {int(r.ckC.examples)} training examples, {int(r.ckC.tokens)} tokens in all,
            {fullOff?.out_of_memory ? " does not fit in one pass on the 3090's 24 GB" : " fits in one pass"}; without help, the most that
            fits is {bestOff.micro} per pass, at {num(bestOff.peak_gb, 3)} GB. <b>Activation checkpointing</b>{" "}
            <Cite k="checkpoint" /> keeps only each transformer block's input and runs the block again when backward reaches it,
            trading one more forward pass for the memory. <code>minitorch/checkpoint.py</code> wraps a block in it.
          </p>
          <p>
            With it, all eight examples share one pass in {num(fullOn.peak_gb, 3)} GB and take {num(fullOn.seconds_per_batch, 2)} s,
            against {num(bestOff.seconds_per_batch, 2)} s for {bestOff.micro} per pass without it (Table A6).
            {r.ckM && <> On the Apple GPU, one example per pass needs {num(ck(r.ckM, false, 1)?.peak_gb, 3)} GB without checkpointing and {num(ck(r.ckM, true, 1)?.peak_gb, 3)} GB with it, and the recomputation stretches a batch from {num(ck(r.ckM, false, 1)?.seconds_per_batch, 3)} s to {num(ck(r.ckM, true, 1)?.seconds_per_batch, 3)} s.</>}
          </p>
        </>
      )}
      {r.ckC && (
        <Fig n="A6" table caption={<>LoRA training of Qwen3-0.6B on {r.ckC.examples} GSM8K examples ({int(r.ckC.tokens)} tokens) per batch, split into passes of the given size, from <code>bench/checkpointing.py</code> (§11). Peak is the GPU memory the batch needed.</>}>
          <table className="results">
            <thead><tr><th>Checkpointing</th><th className="num">Examples per pass</th><th className="num">NVIDIA peak, GB</th><th className="num">NVIDIA s per batch</th><th className="num">Apple peak, GB</th><th className="num">Apple s per batch</th></tr></thead>
            <tbody>
              {r.ckC.rows.map((x) => {
                const m = r.ckM?.rows.find((y) => y.checkpointing === x.checkpointing && y.micro === x.micro);
                return (
                  <tr key={`${x.checkpointing}${x.micro}`}>
                    <th>{x.checkpointing ? "on" : "off"}</th><td className="num">{x.micro}</td>
                    <td className="num">{x.out_of_memory ? "out of memory" : num(x.peak_gb, 3)}</td><td className="num">{x.out_of_memory ? "" : num(x.seconds_per_batch, 3)}</td>
                    <td className="num">{m ? num(m.peak_gb, 3) : ""}</td><td className="num">{m ? num(m.seconds_per_batch, 3) : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Fig>
      )}

      <h3 id="sampling">Sampling replies for GRPO (§12)</h3>
      {ro && (
        <Fig n="A7" table caption={<>Sampling replies for GRPO (§12): Qwen3-0.6B in minitorch on the RTX 3090, from <code>bench/rollout.py</code>: one decode step for the whole batch, and the reply tokens it writes per second.</>}>
          <table className="results">
            <thead><tr><th className="num">Replies at once</th><th className="num">ms per decode step</th><th className="num">Tokens / s</th></tr></thead>
            <tbody>
              {ro.rows.map((x) => (
                <tr key={x.rows}><td className="num">{x.rows}</td><td className="num">{num(x.decode_ms_per_step, 3)}</td><td className="num">{num(x.tokens_per_s, 3)}</td></tr>
              ))}
            </tbody>
          </table>
        </Fig>
      )}
      <p>
        Each group's replies share a prompt, and serving systems compute a shared prompt once. Here that would save little.
        Reading all 64 prompts takes {num(many?.prefill_ms, 3)} ms, against {num(decodeS, 3)} s for the 320 decode steps after
        it, so the prompts are {many && decodeS ? pct(many.prefill_ms / 1000 / (many.prefill_ms / 1000 + decodeS), 1) : "…"} of
        the sampling time.
      </p>

      <h3 id="decoding">Decoding speed-ups (§10)</h3>
      <DecodingAppendix />

      <h3 id="other-models">Other models</h3>
      <p>
        Like tinygrad's <code>examples/</code> folder, two more models run on minitorch's layers, one file each, loaded from
        their published weights through the safetensors reader of §8 and checked against the reference libraries.
      </p>
      {r.gpt2 && (
        <Fig n="A11" table caption={<>GPT-2 ({(r.gpt2.params / 1e6).toFixed(0)}M parameters) in <code>examples/gpt2.py</code> against transformers, from <code>tests/gpt2.py</code>: logits for a {r.gpt2.prompt_tokens}-token prompt, and whether greedy decoding reproduces transformers' first {r.gpt2.greedy_tokens} tokens.</>}>
          <table className="results">
            <thead><tr><th>minitorch on</th><th className="num">Largest gap</th><th className="num">Relative to the largest logit</th><th className="num">Same next token</th><th>Greedy text</th></tr></thead>
            <tbody>
              {r.gpt2.logits.map((l) => (
                <tr key={l.device}><th>{l.device === "cpu" ? "CPU (NumPy)" : "Apple GPU"}</th><td className="num">{sci(l.max_gap)}</td><td className="num">{sci(l.relative_gap)}</td>
                  <td className="num">{pct(l.argmax_agree, 0)}</td><td>{l.greedy_match ? "identical" : "differs"}</td></tr>
              ))}
            </tbody>
          </table>
        </Fig>
      )}
      <p>
        Stable Diffusion v1.5 <Cite k="ldm" /> turns a sentence into a 512 by 512 image with three networks: CLIP's text
        encoder <Cite k="clip" />, a {r.sdParity ? `${num(r.sdParity.unet_params / 1e6, 3)}M` : "…"}-parameter UNet that removes
        noise from a 4 by 64 by 64 latent over DDIM's steps <Cite k="ddim" /> with classifier-free guidance <Cite k="cfg" />, and a
        VAE decoder that turns the latent into pixels. <code>examples/stable_diffusion.py</code> builds all three from
        minitorch's layers; the convolutions run on the GPU through <code>im2col</code>. Its core, condensed:
      </p>
      <pre className="code-block">{`uv run python examples/stable_diffusion.py --prompt "a lighthouse at dusk"

# condensed from generate() in examples/stable_diffusion.py
tok = ClipTokenizer(path / "tokenizer")
prompts = np.array([tok.encode(""), tok.encode(prompt)])
context = load_clip(path, device)(prompts)          # 77 vectors per prompt
unet = load_unet(path, device)
for t, t_prev in zip(times, [*times[1:], -1]):     # 25 DDIM steps
    pair = Tensor(np.concatenate([x, x]), device=device)
    noise = unet(pair, t, context).numpy()
    noise = noise[:1] + guidance * (noise[1:] - noise[:1])   # guidance
    x = ddim_step(x, noise, t, t_prev)
image = load_vae(path, device).decode(Tensor(x, device=device))  # pixels`}</pre>
      {r.sdParity && (
        <Fig n="A12" table caption={<>Each network of Stable Diffusion v1.5 in minitorch against transformers (CLIP) and diffusers (UNet, VAE), from <code>tests/stable_diffusion.py</code>: the largest difference between the outputs, over the reference's largest value. The UNet sees a 32 by 32 latent and the VAE a 16 by 16 one; every layer still runs.</>}>
          <table className="results">
            <thead><tr><th>Network</th><th className="num">Parameters</th><th className="num">CPU (NumPy)</th><th className="num">Apple GPU (MLX)</th></tr></thead>
            <tbody>
              {([["clip", "CLIP text encoder"], ["unet", "UNet"], ["vae", "VAE decoder"]] as const).map(([k, name]) => (
                <tr key={k}><th>{name}</th><td className="num">{num(r.sdParity![`${k}_params`] / 1e6, 3)}M</td>
                  <td className="num">{sci(r.sdParity![`${k}_cpu`])}</td><td className="num">{sci(r.sdParity![`${k}_metal`])}</td></tr>
              ))}
            </tbody>
          </table>
        </Fig>
      )}
      {r.sd && (
        <Fig n="A2" caption={<>"{r.sd.prompt}", {r.sd.steps} DDIM steps, guidance {r.sd.guidance}, float32 on the Apple GPU from the same starting noise, from <code>bench/stable_diffusion.py</code>. The largest pixel difference is {num(r.sd.max_pixel_gap * 255, 2)} of 255 levels. A step takes {num(r.sd.minitorch_metal_s_per_step, 3)} s in minitorch and {num(r.sd.diffusers_mps_s_per_step, 3)} s in diffusers on the same chip.</>}>
          <div className="image-pair">
            <div><img src="/results/stable_diffusion_minitorch.png" alt={`Stable Diffusion's image for "${r.sd.prompt}", drawn by minitorch`} /><span>minitorch</span></div>
            <div><img src="/results/stable_diffusion_diffusers.png" alt="The same image drawn by diffusers on PyTorch" /><span>diffusers on PyTorch</span></div>
          </div>
        </Fig>
      )}

      <h3 id="reproduce">Reproducing the results</h3>
      <p>Every number in this post comes from one of these scripts, which write the files the figures read.</p>
      <pre className="code-block">{`uv run python test_grads.py                    # gradient checks (Figure 3)
uv run python bench/m1_parity.py --float64     # against PyTorch (Figure 4)
uv run python bench/lazy.py                    # lazy against eager (§2)
uv run python scripts/tinystories.py train     # the tokenizer (Table 1)
uv run python tests/backends.py --device metal # backends (Table A1)
uv run python bench/backend_overhead.py        # Figures 9 to 11
uv run python bench/flash_attention.py         # Table 3
uv run python bench/profile_step.py --device metal  # Table A3
uv run python bench/mixed_precision.py --device metal  # §6
uv run python bench/checkpointing.py --device metal    # Table A6, Apple columns
uv run python scripts/scaling.py fit               # Figure A1, from the grid below
uv run python tests/serve.py                   # Table A2
uv run python tests/qwen3.py                   # Table 4, Qwen3
uv run python tests/gpt2.py                    # Table A11, GPT-2
uv run python bench/quant.py                   # Tables A8 and A9
uv run python bench/speculative.py             # Table A10
uv run python apps/rag.py                      # Table 7, with the server running
uv run python apps/agent.py                    # §13, with the server running
uv run python tests/stable_diffusion.py        # Table A12
uv run python bench/stable_diffusion.py        # Figure A2
uv run python bench/gsm8k.py                   # Tables 5 and 6, before training

# on an NVIDIA GPU
uv run --no-dev python tests/backends.py --device cuda      # Table A1, NVIDIA column
CUPY_TF32=1 uv run --no-dev python scripts/scaling.py run --device cuda   # Figure A1's grid
CUPY_TF32=1 uv run --no-dev python train_gpt.py --run abl_baseline --device cuda --steps 10000 --warmup 500
    # the variants add --norm post, --norm none, --no-rope, or --ffn silu --d-ff 2048; then scripts/ablations.py (Table A5)
CUPY_TF32=1 uv run --no-dev python bench/mixed_precision.py --device cuda  # §6
CUPY_TF32=1 uv run --no-dev python bench/checkpointing.py --device cuda --settings off:1,off:2,off:4,off:8,on:1,on:2,on:4,on:8  # Table A6
CUPY_TF32=1 uv run --no-dev python bench/flash_attention_cuda.py         # Table A4
CUPY_TF32=1 uv run --no-dev python bench/profile_step.py --device cuda    # Table A3
CUPY_TF32=1 uv run --no-dev python train_gpt.py --run tinystories --device cuda
CUPY_TF32=1 uv run --no-dev python train_lora.py --run lora_gsm8k --device cuda
CUPY_TF32=1 uv run --no-dev python bench/rollout.py --device cuda          # Table A7
CUPY_TF32=1 uv run --no-dev python train_grpo.py --run grpo_gsm8k --device cuda --steps 200
uv run python bench/gsm8k.py --adapter runs/lora_gsm8k/adapter.npz   # Table 5, after LoRA
uv run python bench/gsm8k.py --adapter runs/grpo_no_stop/adapter.npz  # Table 6, reward ignores stopping
uv run python bench/gsm8k.py --adapter runs/grpo_gsm8k/adapter.npz   # Table 6, after GRPO
uv run python bench/gsm8k_compare.py gsm8k_base gsm8k_agent          # §13, the calculator's split
uv run python bench/gsm8k_compare.py gsm8k_base gsm8k_grpo_gsm8k     # §12, problem by problem
uv run python bench/gsm8k_compare.py gsm8k_base gsm8k_lora_gsm8k     # §12, LoRA's split`}</pre>

      <h3 id="acknowledgements">Acknowledgements</h3>
      <p>Shoutout to Karpathy, CS336 Stanford, George Hotz tinygrad.</p>

      <h3 id="references">References</h3>
      <ol className="refs">
        {REFS.map((r, i) => (
          <li key={r.k} id={`ref-${i + 1}`}><a href={r.url}>{r.text}</a></li>
        ))}
      </ol>

      <h3 id="citation">Citation</h3>
      <div className="bibtex">
        <button className="quiet" onClick={() => navigator.clipboard.writeText(BIBTEX).then(() => setCopied(true))}>
          {copied ? "Copied" : "Copy"}
        </button>
        <pre className="code-block">{BIBTEX}</pre>
      </div>
    </Section>
  );
}
