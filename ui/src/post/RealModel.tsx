import { RunCurve } from "../figures/RunCurve";
import { Aside, chance, Cite, Fig, Section, TeX, int, mean, median, num, pct, sci, useResults, type Gsm8k, type GrpoRow, type Speculative } from "./kit";

export function Qwen() {
  const r = useResults();
  const q = r.qwen;
  return (
    <Section id="qwen" n={8} title="Can it run someone else's model?">
      <p>
        Everything so far used a model I trained myself. A stronger test is someone else's model: its published weights should
        give the same numbers in minitorch as in the reference library. Qwen3-0.6B <Cite k="qwen3" /> is an open chat model from Alibaba's Qwen team, small
        enough to run on a laptop and capable enough to hold a conversation and work through grade-school math.
      </p>
      <p>
        Its weights ship as a <b>safetensors</b> file. Eight bytes give the length of a header, the header lists each tensor's
        name, number type, shape, and byte range as JSON text, and the raw bytes follow. Reading a tensor means slicing out its
        bytes and reinterpreting them, so loading the file runs no code from it, which is the reason the format exists.
      </p>
      <Aside title="bfloat16, and why converting it is exact">
        <p>
          Qwen ships its weights in <b>bfloat16</b>, a 16-bit number format. A standard 32-bit float spends 1 bit on the
          sign, 8 on the exponent (the scale), and 23 on the mantissa (the digits). bfloat16 keeps the same sign and exponent
          and the first 7 mantissa bits. It covers the same range of sizes with two to three significant digits instead of
          seven, in half the memory. Because bfloat16 is the first half of a 32-bit float, minitorch converts it by appending 16
          zero bits, and the conversion loses nothing.
        </p>
      </Aside>
      <p>
        Qwen3 is the model of §4 with four settings changed (Table 2). Each head is 128 numbers wide, set apart from the
        model's width. Qwen3 normalizes queries and keys before RoPE, which keeps attention scores in a steady range. The normalization constant is smaller. And the output layer reuses the embedding table instead of a
        separate matrix, which saves {q ? `${((q.config.vocab_size as number) * (q.config.d_model as number) / 1e6).toFixed(0)}M` : "…"}{" "}
        parameters. The loader maps each of Hugging Face's tensor names to a minitorch parameter and refuses to run if any
        name or shape fails to match, because a weight that keeps its random starting value without warning produces a model
        that runs and talks nonsense.
      </p>
      {q && (
        <Fig n={4} table caption={<>minitorch's output scores (logits) for a {q.prompt_tokens}-token prompt against Hugging Face's transformers library, the reference implementation; Qwen3's largest logit is {num(q.largest_logit, 3)}. "Same next token" is how often both pick the same likeliest next token. With greedy decoding, minitorch {q.greedy_match ? "reproduces" : "does not reproduce"} Hugging Face's first {q.greedy_tokens} tokens of Qwen3's text.</>}>
          <table className="results">
            <thead><tr><th>Model</th><th>minitorch on</th><th className="num">Largest gap</th><th className="num">Gap relative to the largest logit</th><th className="num">Same next token</th></tr></thead>
            <tbody>
              {q.logits.map((l) => (
                <tr key={l.device + l.dtype}>
                  <th>Qwen3-0.6B</th><td>{l.device === "cpu" ? "CPU (NumPy)" : "Apple GPU"}, {l.dtype}</td>
                  <td className="num">{sci(l.max_gap)}</td><td className="num">{sci(l.relative_gap)}</td><td className="num">{pct(l.argmax_agree, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Fig>
      )}
      <p>
        In 32-bit floats the two implementations agree to a few millionths of the logits' size, which again is rounding. In
        bfloat16, the format people run the model in, the gap grows to {pct(r.qwenBf16?.relative_gap)} of the largest
        logit, yet both still pick the same next token at every position.
      </p>
      <p>
        The same layers also run GPT-2 <Cite k="gpt2" />, an older design with learned positions, LayerNorm, and a bias on
        every projection, from one file in the style of tinygrad's examples (Appendix, Other models).
      </p>
    </Section>
  );
}

export function Quantization() {
  const r = useResults();
  const qt = r.quant;
  const row = (p: string) => qt?.rows.find((x) => x.precision === p);
  return (
    <Section id="quant" n={10} title="Can decoding go faster?">
      <p>
        Each reply in Figure 12 streams at the pace of the decode step. Generating one token runs the whole model once and
        reads each weight from memory once. For a 0.6B-parameter model in
        bfloat16 that is about 1.2 GB per token. The arithmetic is cheap by comparison, so memory bandwidth sets the speed of
        generation. No implementation can produce tokens faster than the bandwidth divided by the bytes of weights.
        There are two ways past that limit: read fewer bytes per token, or get more than one token from each read.
      </p>
      {qt && row("bfloat16") && row("int4") && (
        <p>
          <b>Quantization</b> reads fewer bytes. minitorch cuts each row of a weight matrix into groups of 64, keeps a scale and
          an offset per group, and rounds every weight to one of <TeX>{String.raw`2^{\text{bits}}`}</TeX> levels; its own Metal
          kernel multiplies against the packed integers and matches MLX's on the same bytes (<code>tests/quant.py</code>).
          Four bits cut the weights read per token from {int(row("bfloat16")!.weight_mb)} MB to {int(row("int4")!.weight_mb)} MB,
          for {pct(row("int4")!.perplexity.gsm8k / row("bfloat16")!.perplexity.gsm8k - 1)} higher perplexity on math text. Yet
          a decode step goes only from {num(row("bfloat16")!.decode_ms_per_token, 3)} ms to {num(row("int4")!.decode_ms_per_token, 3)} ms.
          Reading the bfloat16 weights takes {num(row("bfloat16")!.weight_mb / qt.bandwidth_gb_s!, 3)} ms of the step at the
          measured bandwidth; the rest is hundreds of small operations, each paying a kernel launch (§6), and quantizing leaves
          those as they were (Tables A8 and A9).
        </p>
      )}
      {r.spec && <SpeculativeFindings spec={r.spec} layers={Number(r.qwen?.config.n_layers)} />}
    </Section>
  );
}

export function Lora() {
  const r = useResults();
  const run = r.runs.find((x) => x.name === "lora_gsm8k");
  const rank = Number(run?.config.rank ?? 16);
  const q = r.qwen?.config;
  // LoRA's trainable numbers: rank * (inputs + outputs) for each of the seven matrices in every layer
  const perLayer = q ? [["d_model", "qh"], ["d_model", "kvh"], ["d_model", "kvh"], ["qh", "d_model"], ["d_model", "d_ff"], ["d_model", "d_ff"], ["d_ff", "d_model"]] : [];
  const width = (k: string) => {
    if (k === "qh") return Number(q!.n_heads) * Number(q!.head_dim);
    if (k === "kvh") return Number(q!.n_kv_heads) * Number(q!.head_dim);
    return Number(q![k]);
  };
  const loraParams = q ? Number(q.n_layers) * perLayer.reduce((acc, [a, b]) => acc + rank * (width(a) + width(b)), 0) : undefined;
  const base = r.gsmBase, tuned = r.gsmLora;
  const ck = (c: typeof r.ckC, on: boolean, micro: number) => c?.rows.find((x) => x.checkpointing === on && x.micro === micro);
  const fullOn = ck(r.ckC, true, 8), bestOff = r.ckC?.rows.filter((x) => !x.checkpointing && x.peak_gb).at(-1);
  const minutes = (name: string) => {
    const last = r.runs.find((x) => x.name === name)?.last;
    return last ? last.time / 60 : undefined;
  };
  return (
    <Section id="lora" n={11} title="Does fine-tuning make it better at math?">
      <p>
        A pretrained model knows a great deal but has its own habits. <b>Fine-tuning</b> trains it a little further on
        examples of the behaviour you want. Full fine-tuning updates every weight, and Adam stores two extra numbers per
        weight, so even a 0.6B model needs memory for several copies of itself.
      </p>
      <p>
        <b>LoRA</b> <Cite k="lora" /> (low-rank adaptation) freezes the original weights and learns a small correction next to
        each weight matrix. A matrix <TeX>W</TeX> with <TeX>{String.raw`n_\text{in}`}</TeX> inputs and{" "}
        <TeX>{String.raw`n_\text{out}`}</TeX> outputs gets a correction <TeX>AB</TeX>, where <TeX>A</TeX> has{" "}
        <TeX>r</TeX> columns and <TeX>B</TeX> has <TeX>r</TeX> rows for a small <b>rank</b> <TeX>r</TeX>:
      </p>
      <TeX block>{String.raw`y = xW + \frac{\alpha}{r}\,(xA)\,B, \qquad \underbrace{n_\text{in} \cdot n_\text{out}}_{\text{frozen}} \;\gg\; \underbrace{r\,(n_\text{in} + n_\text{out})}_{\text{trained}}`}</TeX>
      <p>
        <TeX>B</TeX> starts at zero, so training starts from the original model unchanged. For a 1024 by 1024 matrix at rank{" "}
        {rank}, the correction has {int(rank * 2048)} numbers against the matrix's {int(1024 * 1024)}. Across all seven
        matrices of all {int(q?.n_layers as number)} layers, Qwen3-0.6B gets {int(loraParams)} trainable numbers,{" "}
        {r.qwen && loraParams ? pct(loraParams / r.qwen.params, 2) : "…"} of the model. After training, <TeX>AB</TeX> is
        added into <TeX>W</TeX> once, and the fine-tuned model runs at the original's speed.
      </p>
      <p>
        The task is GSM8K <Cite k="gsm8k" />: grade-school math word problems that take a few steps of arithmetic, each with
        a worked solution ending in <code>#### answer</code>. The model trains on the {int(run?.config.examples as number)}{" "}
        training problems, learning to produce the solutions while the loss ignores the question. I fixed the evaluation
        protocol before training: the first {int(base?.n)} test problems, the same prompt asking for the answer after{" "}
        <code>####</code>, greedy decoding with thinking mode off, at most 320 new tokens, and a problem counts as solved when
        the final number equals the reference.
      </p>
      {r.ckC && fullOn && bestOff && (
        <p>
          Training keeps every activation until backward has used it. <b>Activation checkpointing</b> <Cite k="checkpoint" />{" "}
          keeps only each block's input and recomputes the rest during backward, which fits a batch of {int(r.ckC.examples)}{" "}
          examples into {num(fullOn.peak_gb, 3)} GB on the 3090; without it the most that fits is {bestOff.micro} per pass, at{" "}
          {num(bestOff.peak_gb, 3)} GB (appendix, Activation checkpointing).
        </p>
      )}
      {base ? (
        <Fig n={5} table caption={<>GSM8K accuracy of Qwen3-0.6B running in minitorch, on the Apple GPU in bfloat16. "Hit the limit" counts replies cut off at 320 tokens.</>}>
          <AccuracyTable rows={[["Qwen3-0.6B as released", base], ["after LoRA on GSM8K", tuned]]} />
        </Fig>
      ) : (
        <p className="pending">The baseline evaluation is running; its numbers appear here when it finishes.</p>
      )}
      {tuned && base && r.lora.length >= 100 && (
        <>
          <p>
            LoRA made the model worse at math. Accuracy falls from {pct(base.accuracy)} to {pct(tuned.accuracy)}. The loss on
            the solution tokens still fell, from an average of {mean(r.lora.slice(0, 50).map((x) => x.loss)).toFixed(3)}{" "}
            over the first 50 steps to {mean(r.lora.slice(-50).map((x) => x.loss)).toFixed(3)} over the last 50 (Figure 13), so the model got better
            at reproducing GSM8K's solutions.
          </p>
          <p>
            Reproducing them was the wrong goal. GSM8K's solutions are terse, a few lines of arithmetic and then the answer,
            and the model picked up their length along with their format. Its average reply shrinks from{" "}
            {int(base.new_tokens / base.n)} to {int(tuned.new_tokens / tuned.n)} tokens, and the replies cut off at the limit
            drop from {int(base.hit_token_limit)} to {int(tuned.hit_token_limit)}. The released model writes out more of its
            steps and gets more answers right. The loss rewards replies that read like the reference solution, while Table 5
            counts whether the final number is right, and nothing in this training run rewarded that.
          </p>
        </>
      )}
      {r.gsmLora64 && (
        <Aside title="My first run trained in float64 by mistake">
          <p>
            Each adapter's <TeX>A</TeX> started as random numbers divided by <TeX>{String.raw`\sqrt{n_\text{in}}`}</TeX>. Since
            NumPy 2, dividing a float32 array by the float64 number NumPy's square root returns gives a float64 array. The
            first adapted layer's output became float64, and so did every layer after it, in the forward pass and the backward
            pass. The 3090 does float64 arithmetic at 1/64 of its float32 rate, and the run took{" "}
            {num(minutes("lora_gsm8k_f64"), 3)} minutes. Cast after dividing, every activation takes 4 bytes instead of 8, the run
            fits two examples per pass instead of one, and takes {num(minutes("lora_gsm8k"), 3)} minutes. The float64 run's adapter
            solves {int(r.gsmLora64.correct)} of {int(r.gsmLora64.n)}, against {int(tuned?.correct)} for the float32 one.{" "}
            <code>tests/lora.py</code> now checks that the adapter and the loss stay float32.
          </p>
        </Aside>
      )}
      {r.lora.length > 0 && (
        <Fig n={13} caption={<>The LoRA run's training loss, from <code>runs/lora_gsm8k</code>, measured on the solution tokens only. The GSM8K evaluation in Table 5 plays the part of a validation curve.</>}>
          <RunCurve rows={r.lora} smooth={50} />
        </Fig>
      )}
    </Section>
  );
}

/** GSM8K accuracy on the fixed 250-problem protocol, one row per model; a missing result reads "running". */
function AccuracyTable({ rows }: { rows: [string, Gsm8k | undefined][] }) {
  return (
    <table className="results">
      <thead><tr><th>Model</th><th className="num">Solved</th><th className="num">Accuracy</th><th className="num">Average reply</th><th className="num">Hit the limit</th></tr></thead>
      <tbody>
        {rows.map(([label, x]) => (
          <tr key={label}>
            <th>{label}</th>
            <td className="num">{x ? `${x.correct} of ${x.n}` : "running"}</td><td className="num">{x ? pct(x.accuracy) : "…"}</td>
            <td className="num">{x ? `${int(x.new_tokens / x.n)} tokens` : "…"}</td><td className="num">{x ? int(x.hit_token_limit) : "…"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function Grpo() {
  const r = useResults();
  const ro = r.rollout, g = r.grpo;
  const cfg = r.runs.find((x) => x.name === "grpo_gsm8k")?.config;
  const one = ro?.rows.find((x) => x.rows === 1), many = ro?.rows.at(-1);
  const ns = r.grpoNoStop;
  const lastUncut = ns.reduce((last, x, i) => (x.hit_limit < 1 ? i : last), -1);
  const allCut = ns[lastUncut + 1]?.step;  // from this step on, every reply hit the limit
  const cut10 = (rows: GrpoRow[], from: number) => mean(rows.slice(from, from + 10).map((x) => x.hit_limit));
  const lowest = g.length >= 10 ? Math.min(...g.slice(0, -9).map((_, i) => cut10(g, i))) : undefined;
  const spike = g.length ? g.reduce((a, x) => (x.grad_norm > a.grad_norm ? x : a)) : undefined;
  const base = r.gsmBase, after = r.gsmGrpo;
  const vs = r.vsGrpo, vl = r.vsLora;
  return (
    <Section id="grpo" n={12} title="Does learning from its own answers help?">
      <p>
        LoRA in §11 trained the model to copy solutions, and it got worse at what Table 5 counts. Reinforcement learning
        trains on that count. The model writes its own replies, a checker scores each one, and training makes the replies
        that scored well more likely. The checker here is the evaluation's rule: a reply earns 1 if its final number equals
        the reference and 0 otherwise. Nothing tells the model how to reason, only whether its answer came out right.
      </p>
      <p>
        The update comes from the <b>policy gradient</b> <Cite k="reinforce" />. Write <TeX>{String.raw`\pi_\theta(o \mid q)`}</TeX>{" "}
        for the probability that the model, with weights <TeX>\theta</TeX>, answers question <TeX>q</TeX> with reply{" "}
        <TeX>o</TeX>, and <TeX>R(o)</TeX> for the reply's score. The gradient of the expected score is an expectation too,
        so sampled replies estimate it:
      </p>
      <TeX block>{String.raw`\nabla_\theta\, \mathbb{E}_{o \sim \pi_\theta}\big[R(o)\big] = \mathbb{E}_{o \sim \pi_\theta}\big[R(o)\, \nabla_\theta \log \pi_\theta(o \mid q)\big]`}</TeX>
      <p>
        Stepping along <TeX>{String.raw`R(o)\, \nabla_\theta \log \pi_\theta(o \mid q)`}</TeX> makes scoring replies
        likelier. Used raw, it pushes up every correct reply by the same amount, even on a question the model never gets
        wrong, and the estimate is noisy. <b>GRPO</b> <Cite k="grpo" /> (group relative policy optimization) samples a group
        of replies to each question and scores each reply against its siblings:
      </p>
      <TeX block>{String.raw`A_i = \frac{R(o_i) - \operatorname{mean}_j R(o_j)}{\operatorname{std}_j R(o_j) + \epsilon}, \qquad \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} A_i\, \frac{1}{B} \sum_t \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})`}</TeX>
      <p>
        A reply better than its group's average gets a positive <b>advantage</b> <TeX>A_i</TeX> and becomes likelier, and a
        worse one becomes less likely. A group whose replies all scored the same, all right or all wrong, says nothing about
        which reply was better: every advantage in it is zero, and minitorch skips those replies in the update. A reply's
        log-probabilities are summed and divided by <TeX>B</TeX>, the same 320-token budget for every reply. CS336's version
        divides each reply by its own length <TeX>{String.raw`|o_i|`}</TeX> instead, which, Liu et al. <Cite k="drgrpo" /> point
        out, gives a long wrong reply a smaller push per token than a short one. Their Dr. GRPO drops that division and the one
        by the standard deviation; minitorch drops only the first. CS336's version also clips how far one update may move a
        probability, which matters when one batch of replies trains several updates. minitorch makes one update per batch, and
        there the clipped loss and this one have the same gradient.
      </p>
      <Aside title="Four replies to one question">
        <p>
          Say one reply of four is right: scores 1, 0, 0, 0. The mean is 0.25 and the standard deviation 0.433, so the right
          reply's advantage is <TeX>{String.raw`(1 - 0.25)/0.433 = 1.73`}</TeX> and each wrong reply's is{" "}
          <TeX>{String.raw`(0 - 0.25)/0.433 = -0.58`}</TeX>. The right reply gets pushed up three times as hard as each wrong
          one gets pushed down, and the pushes cancel: <TeX>{String.raw`1.73 = 3 \times 0.58`}</TeX>.
        </p>
      </Aside>
      {cfg && (
        <p>
          Training starts from the released Qwen3-0.6B, not the LoRA model of §11, with a fresh LoRA adapter of the same
          size. Its learning rate, {cfg.lr}, is ten times CS336's rate for training every weight, the ratio "LoRA Without
          Regret" <Cite k="lorawr" /> recommends. Each step samples {cfg.group} replies to each of {cfg.questions} training
          problems at temperature {cfg.temperature} and makes one update.
        </p>
      )}

      <p>
        Sampling takes most of each step, up to 320 decode steps of one token each, and a decode step for one reply leaves most
        of the GPU idle (§6). Sampling all {int(many?.rows)} replies in one batch fills it. On the 3090 a decode step takes{" "}
        {num(one?.decode_ms_per_step, 3)} ms for one reply and {num(many?.decode_ms_per_step, 3)} ms for {int(many?.rows)}, so the
        batch writes {one && many ? (many.tokens_per_s / one.tokens_per_s).toFixed(0) : "…"} times as many tokens per second
        (Table A7). Prompts of different lengths share the batch by padding on the left, with the padding masked out of
        attention (<code>tests/grpo.py</code>). The first version still crawled. It copied all {int(many?.rows)} rows of logits,{" "}
        {int(Number(r.qwen?.config.vocab_size))} numbers each, to the CPU to draw the tokens, which took{" "}
        {num(ro?.step_ms_64_rows.sample_on_host, 3)} ms per step against the model's {num(ro?.step_ms_64_rows.model, 3)} ms;
        drawing them on the GPU takes {num(ro?.step_ms_64_rows.sample_on_device, 2)} ms.
      </p>

      <h3>The reward the model gamed</h3>
      {ns.length > 0 && (
        <p>
          The first run used the reward above, the final number alone. From step {int(allCut)} on, every reply it sampled ran to
          the 320-token limit (Figure 14). The model had found replies the reward could not tell from good ones: it wrote its
          solution, then "The final answer is <TeX>{String.raw`\boxed{18}`}</TeX>." between rows of dashes, over and over, until
          the limit cut it off. The number stayed right, and nothing in the reward cost the model anything for writing on. On the
          test problems that adapter solves {r.gsmNoStop ? `${r.gsmNoStop.correct} of ${r.gsmNoStop.n}` : "…"}, against{" "}
          {base ? base.correct : "…"} before training, and {r.gsmNoStop ? `all ${int(r.gsmNoStop.hit_token_limit)}` : "…"} of
          its replies hit the limit (Table 6).
        </p>
      )}
      <p>
        The fix makes stopping part of the answer. A reply earns 1 only if it ends with Qwen's end-of-turn token and its final
        number is right, so a reply the limit cuts off scores 0 whatever it says.
      </p>
      {g.length > 0 && (
        <Fig n={14} caption={<>The share of sampled replies the 320-token limit cut off, step by step, under each reward, from <code>runs/grpo_gsm8k</code> and <code>runs/grpo_no_stop</code>. The first run stopped at step {int(ns.at(-1)?.step)}.</>}>
          <RunCurve rows={g} field="hit_limit" label="reward requires stopping" log={false} smooth={10} compare={{ rows: ns, label: "reward ignores stopping" }} />
        </Fig>
      )}
      {g.length >= 10 && (
        <p>
          Under the new reward the share cut off fell from {pct(cut10(g, 0), 0)} over the first ten steps to {pct(lowest, 0)} at
          its lowest ten-step stretch, and ended at {pct(cut10(g, g.length - 10), 0)}. It did not fall steadily. Twice it
          climbed back, and both climbs came with spikes in the gradient's norm, up to {num(spike?.grad_norm, 3)} at step{" "}
          {int(spike?.step)} against a median of {num(median(g.map((x) => x.grad_norm)), 2)}. Clipping caps every update at a norm
          of {cfg?.clip ?? "…"}, so a spike cannot move the weights far, and these runs do not show which way the cause runs.
        </p>
      )}
      {g.length > 0 && (
        <Fig n={15} caption={<>The share of sampled replies that scored, step by step, on the training problems, from <code>runs/grpo_gsm8k</code>. {int(g.length)} of {cfg?.steps ?? "…"} steps; replies sampled at temperature {cfg?.temperature ?? "…"}.</>}>
          <RunCurve rows={g} field="reward" label="reward" log={false} smooth={20} />
        </Fig>
      )}
      {g.length > 0 && (
        <p>
          In {pct(mean(g.map((x) => x.groups_with_signal)), 0)} of groups the replies disagreed; the rest scored the same
          everywhere and trained nothing. A step took {num(mean(g.map((x) => x.rollout_s)), 2)} s of sampling and{" "}
          {num(mean(g.map((x) => x.update_s)), 2)} s of updating on average.
        </p>
      )}
      {(after || r.gsmNoStop) && (
        <Fig n={6} table caption={<>GSM8K accuracy under the protocol of §11: the first 250 test problems, greedy decoding, at most 320 new tokens, on the Apple GPU in bfloat16.</>}>
          <AccuracyTable rows={[["Qwen3-0.6B as released", base], ["after LoRA on GSM8K", r.gsmLora], ["after GRPO, reward ignores stopping", r.gsmNoStop], ["after GRPO, reward requires stopping", after]]} />
        </Fig>
      )}
      {after && base && vs && g.length >= 40 && (
        <>
          <p>
            On the training problems, sampled at temperature {cfg?.temperature ?? "…"}, the reward rose from{" "}
            {pct(mean(g.slice(0, 20).map((x) => x.reward)), 0)} over the first 20 steps to {pct(mean(g.slice(-20).map((x) => x.reward)), 0)}{" "}
            over the last 20. On the test problems, decoded greedily, the adapter solves {after.correct} of {after.n}, against{" "}
            {base.correct} before training, and {int(after.hit_token_limit)} of its replies hit the limit, against{" "}
            {int(base.hit_token_limit)}.
          </p>
          <p>
            Problem by problem, the adapter solved {vs.fixed} problems the released model missed and missed {vs.broke} that
            it had solved. If training had changed nothing, each of those {vs.fixed + vs.broke} disagreements would fall either way
            like a coin toss, and a split at least this uneven comes up {chance(vs.p_value)} of the time, so these {vs.n} problems
            cannot tell the adapter from the model it started as.
            {vl && <> LoRA's split in §11, {vl.fixed} fixed against {vl.broke} broken, comes up {chance(vl.p_value)} of the time: that drop is real.</>}
          </p>
        </>
      )}
    </Section>
  );
}

/** §10's measurements in full, for the appendix. */
export function DecodingAppendix() {
  const r = useResults();
  const qt = r.quant;
  const row = (p: string) => qt?.rows.find((x) => x.precision === p);
  return (
    <>
      {qt && (
        <Fig n="A8" table caption={<>Qwen3-0.6B on the Apple GPU at three precisions (§10). Perplexity comes from {int(qt.tokens_per_text)} tokens of each held-out text, GSM8K's worked solutions and TinyStories stories; lower is better. "Ceiling" is the measured memory bandwidth, {num(qt.bandwidth_gb_s, 3)} GB/s, divided by the bytes of weights read per token.</>}>
          <table className="results">
            <thead><tr><th>Weights</th><th className="num">MB read per token</th><th className="num">Perplexity, math</th><th className="num">Perplexity, stories</th><th className="num">Tokens / s</th><th className="num">Ceiling</th></tr></thead>
            <tbody>
              {qt.rows.map((x) => (
                <tr key={x.precision}>
                  <th>{x.precision}</th><td className="num">{int(x.weight_mb)}</td><td className="num">{num(x.perplexity.gsm8k, 4)}</td>
                  <td className="num">{num(x.perplexity.tinystories, 4)}</td><td className="num">{num(x.decode_tokens_per_s, 3)}</td><td className="num">{num(x.ceiling_tokens_per_s, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Fig>
      )}
      {qt && row("bfloat16") && row("int8") && (
        <p>
          One result is unexplained: int8 makes the decode step slower, {num(row("int8")!.decode_ms_per_token, 3)} ms against
          bfloat16's {num(row("bfloat16")!.decode_ms_per_token, 3)}, although its matrix multiplications run faster in isolation.
          Swapping in MLX's own quantized kernel gives {num(row("int8")!.decode_ms_per_token_mlx_kernel, 3)} ms, so the cause is
          not minitorch's kernel; finding it needs a profile of the whole step on the GPU.
        </p>
      )}
      {qt?.kernels && (
        <Fig n="A9" table wide caption={<>One token's matrix multiplication for three of Qwen3's matrices on the Apple GPU: the time, and under it the memory bandwidth it reaches. MLX's quantized kernel is the reference; the measured peak bandwidth is {num(qt.bandwidth_gb_s, 3)} GB/s.</>}>
          <KernelTable kernels={qt.kernels} />
        </Fig>
      )}
      {r.spec && <SpeculativeTable spec={r.spec} />}
    </>
  );
}

/** One row per matrix, one column per implementation: microseconds, with the bandwidth that implies under it. */
function KernelTable({ kernels }: { kernels: { matrix: string; impl: string; us: number; gb_s: number }[] }) {
  const impls = [...new Set(kernels.map((k) => k.impl))];
  const matrices = [...new Set(kernels.map((k) => k.matrix))];
  return (
    <table className="results">
      <thead><tr><th>Matrix</th>{impls.map((i) => <th key={i} className="num">{i}</th>)}</tr></thead>
      <tbody>
        {matrices.map((m) => (
          <tr key={m}>
            <th>{m}</th>
            {impls.map((i) => {
              const k = kernels.find((x) => x.matrix === m && x.impl === i);
              return <td key={i} className="num">{k ? <>{num(k.us, 4)} µs<small>{num(k.gb_s, 3)} GB/s</small></> : "-"}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SpeculativeTable({ spec }: { spec: Speculative }) {
  return (
    <Fig n="A10" table caption={<>Speculative decoding (§10): greedy decoding on the Apple GPU in bfloat16, 128 tokens for each of four GSM8K prompts, from <code>bench/speculative.py</code>. Speedup is against Qwen3-1.7B alone; "kept" is the share of draft tokens the target accepted.</>}>
      <table className="results">
        <thead><tr><th>Decoding</th><th className="num">Tokens / s</th><th className="num">Speedup</th><th className="num">Kept</th></tr></thead>
        <tbody>
          {spec.rows.map((x) => (
            <tr key={x.method + x.k}>
              <th>{x.method === "speculative" ? `0.6B drafts ${x.k} for 1.7B` : x.method}</th>
              <td className="num">{num(x.tokens_per_s, 3)}</td><td className="num">{x.speedup.toFixed(2)}x</td>
              <td className="num">{x.acceptance === undefined ? "" : pct(x.acceptance, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Fig>
  );
}

function SpeculativeFindings({ spec, layers }: { spec: Speculative; layers: number }) {
  const target = spec.rows.find((x) => x.method === "Qwen3-1.7B alone")!;
  const draft = spec.rows.find((x) => x.method === "Qwen3-0.6B alone")!;
  const runs = spec.rows.filter((x) => x.method === "speculative");
  const best = runs.reduce((a, x) => (x.speedup > a.speedup ? x : a));
  const exact = runs.every((x) => x.first_diff_from_target?.every((d) => d === null));
  const cost = target.tokens_per_s / draft.tokens_per_s;  // a draft step's time over a target step's
  return (
    <p>
      <b>Speculative decoding</b> <Cite k="specdec" /> <Cite k="specsample" /> gets more than one token from each pass
      instead. A small draft model guesses the next few tokens, the large model checks all of them in one pass, and it keeps
      the guesses it agrees with by a rule that leaves its output distributed exactly as before (<code>tests/speculative.py</code>).
      With Qwen3-0.6B drafting for Qwen3-1.7B, the target kept {pct(runs[0].acceptance!, 0)} of single guesses
      {exact ? " and every output matched the 1.7B's own text token for token" : ""}, yet the best speedup was{" "}
      {num(best.speedup, 3)}x, at {best.k} {best.k === 1 ? "guess" : "guesses"} per pass (Table A10). The draft costs too much: a 0.6B decode step takes{" "}
      {pct(cost, 0)} of a 1.7B step, because both models have {int(layers)} layers, launch the same number of kernels, and
      pay §6's launch cost for each.
    </p>
  );
}
