import { AttentionTable, Matmul, OpCost, StepBars } from "../figures/BenchCharts";
import { Chat } from "../figures/Chat";
import { Aside, Cite, Fig, Section, TeX, int, num, pct, sci, useResults } from "./kit";

export function Gpu() {
  const r = useResults();
  const bkM = r.bkM, bkC = r.bkC;
  const mp = (m: typeof r.mpC, precision: string) => m?.rows.find((x) => x.precision === precision);
  const c32 = mp(r.mpC, "float32"), c16 = mp(r.mpC, "float16");
  return (
    <Section id="gpu" n={6} title="Is a GPU faster?">
      <p>
        Most of a model's arithmetic repeats one operation over huge arrays, such as multiplying two matrices. A CPU has a
        handful of fast cores built to run complicated programs one step after another. A GPU has thousands of simple cores
        that all run the same small program, a <b>kernel</b>, on different pieces of an array at once. That design suits a
        transformer, which spends its time in large matrix multiplications.
      </p>
      <p>
        minitorch runs on three array libraries that share NumPy's interface: <b>NumPy</b> on the CPU as the readable
        reference, <b>MLX</b> <Cite k="mlx" /> on the Apple laptop's GPU, and <b>CuPy</b> <Cite k="cupy" /> on an NVIDIA GPU.
        The gradient code in <code>mlops.py</code> never calls any of them by name. Each operation receives a module called{" "}
        <code>xp</code> that exposes the same dozen functions over its own kind of array, so one copy of every gradient runs
        everywhere, and one call, <code>model.to("metal")</code>, moves a model.
      </p>
      <Aside title="Two quirks of MLX that shaped the engine">
        <p>
          Code cannot change an MLX array after creating it, so minitorch never updates a gradient in place: adding to a
          gradient makes a new array. MLX is also <b>lazy</b>. An operation records what to compute, and MLX runs the work when
          code reads a value. Pending work keeps growing until something reads a result, so the optimizer computes the new
          weights at the end of each step, or each step's pending work would pile onto the last.
        </p>
      </Aside>
      <p>
        <code>tests/backends.py</code> checks each GPU backend against NumPy. Of {int(bkM?.ops.total)} gradient checks,{" "}
        {int(bkM?.ops.agree)} agree on the Apple GPU and {bkC ? int(bkC.ops.agree) : "…"} on the NVIDIA GPU, and the story
        model, trained on each from the same weights and batches, stays within {sci(r.worstTrainGap)} of NumPy's loss (Table
        A1 in the appendix).
      </p>
      <h3>A GPU does not win by default</h3>
      <p>
        Starting a kernel has a fixed cost. The CPU describes the work and hands it to the GPU's driver, and the GPU schedules
        it. For a small array, that fixed cost is the whole cost. Through minitorch, one elementwise
        multiplication on a thousand numbers takes {num(r.opNumpy, 2)} µs with NumPy, {num(r.opMlx, 2)} µs on the Apple GPU,
        and {num(r.opCupy, 2)} µs on the NVIDIA GPU, so the CPU wins. On ten million numbers it takes {num(r.bigNumpy, 3)} µs,{" "}
        {num(r.bigMlx, 3)} µs, and {num(r.bigCupy, 3)} µs, and the NVIDIA GPU wins by
        {r.bigNumpy && r.bigCupy ? ` ${(r.bigNumpy / r.bigCupy).toFixed(1)}x` : " …"}. Figure 9 shows the crossover.
      </p>
      {r.ovM && (
        <Fig n={9} caption={<>The cost of one elementwise multiplication against the size of the array, both on log scales. The buttons switch between the raw array library, the library with a wait for the GPU to finish after every operation, and minitorch on top of the library. The waiting view exposes each device's round trip: {num(r.syncMlx, 3)} µs on the Apple GPU and {num(r.syncCupy, 3)} µs on the NVIDIA GPU.</>}>
          <OpCost rows={[...r.ovM.elementwise, ...(r.ovC?.elementwise ?? [])]} />
        </Fig>
      )}
      <p>
        Matrix multiplication suits a GPU. Multiplying two <TeX>n \times n</TeX> matrices takes{" "}
        <TeX>{String.raw`2n^3`}</TeX> floating-point operations (FLOPs) on <TeX>{String.raw`3n^2`}</TeX> numbers, so as{" "}
        <TeX>n</TeX> grows, the arithmetic outgrows the memory traffic and the hardware can run at full speed. Figure 10 shows
        each device's throughput in trillions of operations per second (TFLOP/s). The NVIDIA GPU peaks at{" "}
        {num(r.peakMatmul("cuda", "tf32"), 3)} TFLOP/s using TF32, a reduced-precision format that its tensor cores run in hardware,
        and the Apple GPU at {num(r.peakMatmul("metal", "bfloat16"), 3)} TFLOP/s.
      </p>
      {r.ovM && (
        <Fig n={10} caption="Square matrix multiplication throughput against matrix size, log scale. Dashed lines use a reduced-precision number format: bfloat16 on the Apple GPU, TF32 on the NVIDIA GPU.">
          <Matmul rows={[...r.ovM.matmul, ...(r.ovC?.matmul ?? [])]} />
        </Fig>
      )}
      {r.mpC && c32 && c16 && (
        <p>
          Training can spend that speed through <b>mixed precision</b> <Cite k="mixed" />, which multiplies in 16 bits while the
          weights stay in float32. On the NVIDIA GPU it takes a step of the story model from {num(c32.step_ms, 3)} ms to{" "}
          {num(c16.step_ms, 3)} ms and moves the loss by {num(r.mpC.last_50_loss_gap, 2)} (appendix, Mixed precision).
        </p>
      )}
      <p>
        A whole training step of the story model spends most of its time in large matrix multiplications, so the GPUs win there (Figure 11).
        The NVIDIA GPU trains {num(r.cudaVsCpu, 3)}x faster than the laptop's CPU. On the Apple GPU, minitorch reaches{" "}
        {pct(r.mlxShare, 0)} of the speed of the same model written in plain MLX with MLX's own hand-tuned kernels, which
        measures what minitorch's generality costs there.
      </p>
      {r.ovM && (
        <Fig n={11} caption={<>One training step of the story model, with a batch of {r.ovM.train_step[0]?.batch} sequences of 256 tokens: forward, backward, clipping, and the optimizer update. Longer bars are faster.</>}>
          <StepBars rows={[...r.ovM.train_step, ...(r.ovC?.train_step ?? [])]} />
        </Fig>
      )}
      <p>
        The <b>model FLOPs utilization</b> (MFU) measures how much of the chip a training run uses. It divides the arithmetic
        a training step must do, about six operations per parameter per token, by what the chip could do at its rated peak.
        The full training run in §5 sustained a median {num(r.tflops, 3)} TFLOP/s, an MFU of {pct(r.mfu)}. Launch overhead and
        operations that move memory without doing much arithmetic take the remaining time, and attention has many of those,
        so §7 fuses it into one kernel.
      </p>
    </Section>
  );
}

export function Attention() {
  const r = useResults();
  const p = r.profM;
  const opMs = (op: string) => p?.rows.find((x) => x.op === op)?.ms;
  const attB = opMs("Attention backward"), attF = opMs("Attention");
  const blocks = p ? p.context / 32 : undefined;
  const visited = blocks ? (blocks * (blocks + 1)) / 2 / blocks ** 2 : undefined;  // share of 32x32 tiles v3's causal loop visits
  const fa = (shape: string, impl: string) => r.fa2?.rows.find((x) => x.shape === shape && x.impl === impl);
  const faC = fa("CS336 training", "composed"), faT = fa("CS336 training", "Triton");
  const pc = r.profC;
  const share = (prof: typeof p, op: string) => {
    const row = prof?.rows.find((x) => x.op === op);
    return row && prof ? row.ms / prof.profiled_ms : undefined;
  };
  return (
    <Section id="attention" n={7} title="Can a hand-written kernel beat the library's?">
      <p>
        Recall attention from §4: compute scores <TeX>{String.raw`S = QK^\top/\sqrt{d}`}</TeX> between every pair of
        positions, apply the mask, take the softmax, and use the result to average the values. Composed from minitorch's basic
        operations, each of those steps is its own kernel. The score matrix has one entry per pair of positions, so for 256
        tokens and 16 heads it holds a million numbers per sequence, and every step writes all of them to the GPU's memory and
        the next step reads them back.
      </p>
      <p>
        Moving that matrix costs more than computing it. A GPU can do arithmetic far faster than it can move data to and from
        its memory, so a step that does one multiplication per number it reads spends most of its time waiting for memory. FlashAttention{" "}
        <Cite k="flash" /> <Cite k="flash2" /> computes attention in one kernel that never writes the score matrix anywhere. It
        rests on a trick for computing a softmax in a single pass <Cite k="online" />.
      </p>
      <Aside title="Softmax in one pass">
        <p>
          A softmax needs the largest score <TeX>m</TeX> (subtracted for numerical safety) and the sum{" "}
          <TeX>{String.raw`l = \sum_j e^{s_j - m}`}</TeX>. Both seem to need all scores before you start. But you can keep a
          running maximum and a running sum, and whenever a larger score arrives, rescale the sum so far by{" "}
          <TeX>{String.raw`e^{m_\text{old} - m_\text{new}}`}</TeX>. Take the scores 1, 3, 2. After the first,{" "}
          <TeX>m = 1</TeX> and <TeX>l = 1</TeX>. The 3 raises the maximum, so the old sum shrinks to{" "}
          <TeX>{String.raw`e^{1-3} = 0.135`}</TeX> and the new term adds <TeX>{String.raw`e^0 = 1`}</TeX>: <TeX>l = 1.135</TeX>.
          The 2 leaves the maximum alone and adds <TeX>{String.raw`e^{-1} = 0.368`}</TeX>: <TeX>l = 1.503</TeX>, the same as
          computing it all at once. The weighted sum of values is rescaled the same way, so one pass over the keys gives the
          finished output.
        </p>
      </Aside>
      <TeX block>{String.raw`\begin{aligned} m_j &= \max(m_{j-1},\, s_j) \\ l_j &= l_{j-1}\,e^{m_{j-1}-m_j} + e^{s_j-m_j} \\ o_j &= o_{j-1}\,e^{m_{j-1}-m_j} + e^{s_j-m_j}\,v_j \end{aligned} \qquad \text{output} = \frac{o_S}{l_S}`}</TeX>
      <p>
        I wrote three versions as Metal kernels for the Apple GPU (with MLX's <code>mx.fast.metal_kernel</code>), each fixing
        the previous one's bottleneck:
      </p>
      <ul>
        <li><b>v1</b> gives each query position one GPU thread, which reads every key and value straight from memory. It is simple, but neighbouring threads read the same keys again and again.</li>
        <li><b>v2</b> groups 32 query positions into a <i>threadgroup</i> that loads a tile of keys and values into fast on-chip memory once and shares it. Each key is read from main memory 32 times less often.</li>
        <li><b>v3</b> keeps v2's tiles and computes the dot products with the GPU's <i>simdgroup matrix</i> instructions, which multiply small 8 by 8 matrices in hardware instead of one number at a time.</li>
      </ul>
      <p>
        All three handle grouped-query heads and the offsets a cache needs during generation, and{" "}
        {r.bkM?.kernels ? `${r.bkM.kernels.within_tolerance} of ${r.bkM.kernels.total} runs across versions and shapes match` : "they match"} a 64-bit
        reference to within {sci(r.bkM?.kernels?.tolerance, 0)}. Table 3 times them against MLX's own fused attention kernel.
      </p>
      {r.attn && (
        <Fig n={3} table wide caption={<>Time in milliseconds to compute attention on the Apple GPU (bold is the fastest in each row). "Materialized scores" is the version built from basic operations. v3 runs {num(r.kernelWin, 2)}x faster than MLX's kernel at the training shape, where each head is 32 numbers wide, and {num(r.kernelLoss, 2)}x slower at Qwen3's shape, 128 wide, where MLX's kernel wins. For generating one token at a time (decode), v1 takes {num(r.decodeRatio, 3)}x MLX's time.</>}>
          <AttentionTable rows={r.attn} />
        </Fig>
      )}
      <p>
        Training also needs attention's backward pass. The kernel returns one extra number per row, the log of the softmax
        sum, which is enough to rebuild the softmax weights during backward instead of storing them:
      </p>
      <TeX block>{String.raw`\begin{aligned} P &= e^{S - \text{lse}} & dV &= P^\top dO & dP &= dO\,V^\top \\ dS &= P \odot \big(dP - \operatorname{rowsum}(dO \odot O)\big) & dQ &= dS\,K / \sqrt{d} & dK &= dS^\top Q / \sqrt{d} \end{aligned}`}</TeX>
      <p>
        Here <TeX>dO</TeX> is the gradient arriving at the output and <TeX>{String.raw`\odot`}</TeX> multiplies entry by entry.
        This backward lives in <code>mlops.py</code> with every other gradient, so it runs on all three backends.
      </p>
      {p && (
        <>
          <h3>Where a training step's time goes</h3>
          <p>
            A kernel is worth writing only for an operation that costs something, and a profiler says which ones do. Inside{" "}
            <code>with profile():</code> (<code>minitorch/profile.py</code>), every primitive waits for the GPU to finish
            before the next one starts, so each one's time is its own. One forward and backward pass of the story model on{" "}
            {int(p.batch)} sequences of {int(p.context)} tokens takes {num(p.profiled_ms, 3)} ms this way on the Apple GPU. A
            normal step, optimizer included, takes {num(p.step_ms, 3)} ms, because there the GPU queues work without waiting.
            Table A3 in the appendix lists every primitive.
          </p>
          <p>
            Matrix multiplications take the most time: {num(opMs("Matmul backward"), 3)} ms backward and{" "}
            {num(opMs("Matmul"), 3)} ms forward. Attention comes second, and nearly all of its time is backward:{" "}
            {num(attB, 3)} ms, against {num(attF, 2)} ms for the fused forward, {num(attB && attF && attB / attF, 3)}x.
            Counting arithmetic predicts {num(visited && 5 / (2 * visited), 2)}x. The forward makes two matrix products per
            head, the scores and the weighted sum of values, and skips the {pct(visited && 1 - visited, 0)} of 32-by-32 tiles
            that the causal mask hides completely. The backward in <code>mlops.py</code> makes five products over every pair of
            positions: the scores again, then <TeX>dV</TeX>, <TeX>dP</TeX>, <TeX>dQ</TeX> and <TeX>dK</TeX>. It also writes
            three score-sized matrices, <TeX>P</TeX>, <TeX>dP</TeX> and <TeX>dS</TeX>, to the GPU's memory and reads them back,
            the traffic the forward kernel exists to avoid. FlashAttention-2 <Cite k="flash2" /> fuses the backward as well.
          </p>
        </>
      )}
      {faC && faT && (
        <>
          <h3>The backward as a kernel, in Triton</h3>
          <p>
            On the NVIDIA GPU I wrote FlashAttention-2's forward and backward in Triton <Cite k="triton" />, a Python dialect
            for GPU kernels in which a program describes the work on one tile and the compiler handles the threads. The forward
            is the one-pass softmax above. The backward walks the tiles twice: once per block of keys, adding up{" "}
            <TeX>dK</TeX> and <TeX>dV</TeX>, and once per block of queries for <TeX>dQ</TeX>. Each pass rebuilds its tile of{" "}
            <TeX>P</TeX> from the saved log-sum-exp in on-chip memory, so no score-sized matrix reaches the GPU's main memory.
            Against a 64-bit reference, {r.bkC?.kernels ? `${r.bkC.kernels.within_tolerance} of ${r.bkC.kernels.total}` : "all"}{" "}
            checks of the kernels agree to within {sci(r.bkC?.kernels?.tolerance, 0)}.
          </p>
          <p>
            At the training shape, forward and backward together take {num(faT.forward_backward_ms, 3)} ms, against{" "}
            {num(faC.forward_backward_ms, 3)} ms for the composed version, {num(faC.forward_backward_ms / faT.forward_backward_ms, 2)}x
            faster (Table A4 lists every shape).
            {pc && <> In a profiled training step on the NVIDIA GPU, attention's backward takes {pct(share(pc, "Attention backward"), 1)} of the time, where on the Apple GPU, with the composed backward, it took {pct(share(p, "Attention backward"), 1)} (Table A3).</>}
          </p>
        </>
      )}
    </Section>
  );
}

export function Serving() {
  const r = useResults();
  const s = r.serve;
  const tight = s?.scenarios.find((x) => x.preemptions > 0);
  return (
    <Section id="serving" n={9} title="Can it serve two people at once?">
      <p>
        minitorch now computes the same scores as the reference, one forward pass at a time. A chat reply needs one forward
        pass per new token, hundreds in all. Without care, step <TeX>t</TeX> recomputes the keys and
        values of all <TeX>t</TeX> earlier tokens, although they never change. The <b>KV cache</b> stores them instead: each
        step computes keys and values for the one new token, appends them, and attends over the stored ones. The work per
        token stops growing with the length of the text. The price is memory, since each token of each conversation keeps its
        keys and values for each layer.
      </p>
      <p>
        That memory limits how many people one GPU can serve. Reserving the maximum length for each request wastes the part a
        short reply never uses. A <b>block manager</b> hands out cache memory in small fixed-size blocks as
        a sequence grows and takes them back the moment it finishes, the bookkeeping behind vLLM's PagedAttention{" "}
        <Cite k="paged" />.
      </p>
      <p>
        The second idea is <b>continuous batching</b> <Cite k="orca" />. A plain batch of requests runs until its longest
        reply finishes, while finished requests sit idle. minitorch's scheduler instead admits new requests between any two
        generation steps and lets each one leave as soon as it is done. If the cache runs out of blocks, the scheduler{" "}
        <b>preempts</b> the newest request. That request gives back its blocks, waits, and later rebuilds its cache from its
        prompt plus the tokens it had already produced.
      </p>
      {s && (
        <p>
          <code>tests/serve.py</code> checks the scheduler with greedy decoding, where each request must receive the same
          tokens it gets when it runs alone. Under each of {int(s.scenarios.length)} loads{" "}
          {s.scenarios.every((x) => x.identical) ? "every request does" : "some requests do not"}
          {tight ? `, including a run with ${int(tight.preemptions)} preemptions when the whole cache holds only ${int(tight.num_blocks)} blocks` : ""} (Table
          A2 in the appendix).
        </p>
      )}
      <p>
        In Figure 12 the engine serves Qwen3. Messages go through an endpoint shaped like OpenAI's chat API into the
        scheduler and engine, and Qwen3 runs in minitorch on the Apple GPU in bfloat16. Thinking mode is off, so it answers
        without a reasoning preamble. Open this page twice and send from both, and the engine interleaves the two
        conversations step by step.
      </p>
      <Fig n={12} caption="Chat with Qwen3-0.6B, live. The reply streams in token by token; the line under it gives the time to the first token and the speed after it.">
        <Chat />
      </Fig>
      <p>
        The two conversations take turns. The scheduler runs the model on one sequence per step, because each sequence's
        cache has a different length, and running several in one step needs an attention kernel that reads each sequence's
        keys and values through its own list of blocks. Even alone, a reply streams no faster than one decode step allows.
      </p>
    </Section>
  );
}
