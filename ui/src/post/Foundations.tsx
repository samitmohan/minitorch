import { AutogradGraph } from "../figures/AutogradGraph";
import { ParityChart } from "../figures/BenchCharts";
import { GradChecks } from "../figures/GradChecks";
import { useState } from "react";
import { Aside, Cite, Fig, Section, TeX, int, num, pct, pval, sci, useResults , type Lazy } from "./kit";

export function Intro() {
  const r = useResults();
  const lines = (name: string) => int(r.lines?.groups.find((g) => g.name === name)?.lines);
  const fa = (impl: string) => r.fa2?.rows.find((x) => x.shape === "CS336 training" && x.impl === impl)?.forward_backward_ms;
  return (
    <Section id="intro" n={1} title="Introduction">
      <p>
        Deep learning frameworks like PyTorch hide the parts worth understanding. You call <code>loss.backward()</code> and
        gradients appear. You move a model to a GPU and it gets faster, except when it gets slower. You download a chat model
        and it answers, and the steps between the bytes on disk and the words on screen stay out of sight.
      </p>
      <p>
        minitorch rebuilds those parts in plain Python and checks each one against the library it imitates. I started where
        Andrej Karpathy's micrograd <Cite k="micrograd" /> starts, with a single number that computes its own gradient, and
        took tinygrad's design <Cite k="tinygrad" />: a short list of basic operations with known gradients, and everything
        else composed from them.
      </p>
      <p>
        The work follows the order a study plan would. Autograd comes first, then a language model trained from scratch, GPUs
        and kernels, a real pretrained model, a server, fine-tuning, and last the two applications most language-model systems
        are built for: <b>retrieval-augmented generation</b>, which answers from documents it looks up, and an <b>agent</b>{" "}
        that calls tools. That last step is the goal. A question goes in through an OpenAI-style endpoint, and every number
        between it and the answer comes from code written here. Each section builds the piece the previous section's result
        showed was missing, and ends on a measurement.
      </p>
      {r.lines && (
        <p>
          The framework is {int(r.lines.package)} lines of Python, counting code and docstrings but not blank lines or
          comments: {lines("engine")} for the engine, {lines("backends and kernels")} for the three backends and their Metal and
          Triton kernels, {lines("models")} for the models, and {lines("serving")} for the server. The whole repository, with
          its tests, benchmarks, and this page, is {int(r.lines.repository)} (<code>scripts/lines.py</code>).
        </p>
      )}
      <p>The results, in the order the sections reach them:</p>
      <ul>
        <li>
          The gradients match PyTorch's to {sci(r.p64?.rows[0].abs_diff)}, and a{" "}
          {r.tsParams ? `${(r.tsParams / 1e6).toFixed(1)}M` : "…"}-parameter model trains on TinyStories to a validation loss of{" "}
          {num(r.finalVal)} in {num(r.minutes, 2)} minutes on one RTX 3090 (§2, §5).
        </li>
        <li>
          A hand-written Metal attention kernel runs {num(r.kernelWin, 2)}x faster than MLX's own at the training shape, and
          FlashAttention-2 written in Triton runs {num(fa("composed") && fa("Triton") && fa("composed")! / fa("Triton")!, 2)}x
          faster than the composed version (§7).
        </li>
        <li>Qwen3-0.6B's logits match Hugging Face's to {sci(r.qwenCpu?.max_gap)}, and minitorch serves it behind an OpenAI-style endpoint (§8, §9).</li>
        {r.vsLora && r.vsGrpo && r.vsAgent && (
          <li>
            Fine-tuning with LoRA, reinforcement learning with GRPO, and a calculator tool all fail to make it better at
            grade-school math ({pval(r.vsLora.p_value)}, {pval(r.vsGrpo.p_value)}, {pval(r.vsAgent.p_value)}; §11 to §13).
          </li>
        )}
        {r.rag && (
          <li>
            Retrieval puts the right Wikipedia paragraph among the top three for {pct(r.rag.recall_at_3, 1)} of questions, and{" "}
            {pct(r.rag.rag_contains, 0)} of answers then contain the reference (§13).
          </li>
        )}
      </ul>
      <p>
        You do not need to know any machine learning to follow along. Each section explains its idea from scratch before
        minitorch implements it. The figures read the project's result files, and several run the engine on the server when
        you click.
      </p>
    </Section>
  );
}

export function Autograd() {
  const r = useResults();
  const lz = r.lazy;
  const lazyRows = (device: string) => {
    const rows = lz?.rows.filter((x) => x.device === device);
    return rows?.length ? Object.fromEntries(rows.map((x) => [x.mode, x])) as Record<"eager" | "lazy" | "both", Lazy["rows"][number]> : undefined;
  };
  const lzCpu = lazyRows("cpu"), lzMetal = lazyRows("metal");
  const [dtype, setDtype] = useState<"float64" | "float32">("float64");
  const parity = dtype === "float64" ? r.p64 : r.p32;
  return (
    <Section id="autograd" n={2} title="Autograd: how a network learns">
      <p>
        A neural network is a function with adjustable numbers inside it. It takes numbers in, pixels or token ids, and returns
        numbers out. We call the adjustable numbers <b>parameters</b> or <b>weights</b>, and a large language model has
        billions of them. A new network starts with random weights and gives useless answers. Learning means nudging the
        weights until the answers get good.
      </p>
      <p>
        To nudge them in the right direction we need two things. First, a <b>loss</b>: one number that says how wrong the
        network is on an example, where lower is better. Second, the <b>gradient</b> of the loss with respect to every weight:
        how much the loss would change if that one weight moved a little. The gradient points uphill, so we step each weight a
        little the other way. That step is <b>gradient descent</b>, and training repeats it millions of times.
      </p>
      <Aside title="A network with one weight, trained for one step">
        <p>
          Take the smallest possible network, <TeX>{String.raw`y = w \cdot x`}</TeX>, with one weight <TeX>w</TeX>. We want it to
          output <TeX>6</TeX> when the input is <TeX>2</TeX>, so we use the squared error as the loss:{" "}
          <TeX>{String.raw`L = (w x - 6)^2`}</TeX>. Start from <TeX>w = 1</TeX>: the output is <TeX>2</TeX> and the loss is{" "}
          <TeX>{String.raw`(2-6)^2 = 16`}</TeX>.
        </p>
        <p>
          The gradient is <TeX>{String.raw`\partial L/\partial w = 2(wx - 6)\,x = 2 \cdot (-4) \cdot 2 = -16`}</TeX>. It is
          negative, which says that increasing <TeX>w</TeX> lowers the loss. Step against it with a step size (the{" "}
          <b>learning rate</b>) of <TeX>0.1</TeX>: <TeX>{String.raw`w \leftarrow 1 - 0.1 \cdot (-16) = 2.6`}</TeX>. Now the output
          is <TeX>5.2</TeX> and the loss is <TeX>{String.raw`(5.2-6)^2 = 0.64`}</TeX>, twenty-five times smaller after one step.
        </p>
      </Aside>
      <p>
        With one weight you can work the gradient out by hand. With a billion weights in dozens of layers you need a program
        that computes them all, and that program is the first part of minitorch.
      </p>
      <h3>The chain rule as a program</h3>
      <p>
        Any computation, however large, is a chain of small steps: multiply, add, take an exponential, and so on. Calculus has
        a rule for chains. If <TeX>z</TeX> depends on <TeX>y</TeX> and <TeX>y</TeX> depends on <TeX>x</TeX>, then{" "}
        <TeX>{String.raw`\partial z/\partial x = (\partial z/\partial y)\,(\partial y/\partial x)`}</TeX>: the effect of{" "}
        <TeX>x</TeX> on <TeX>z</TeX> is its effect on <TeX>y</TeX> times the effect of <TeX>y</TeX> on <TeX>z</TeX>. Automatic
        differentiation, or <b>autograd</b>, applies that rule to every step of a program.
      </p>
      <p>
        It works in two passes. The <b>forward pass</b> runs the computation and, as it goes, records every step and which
        values fed into it, building a graph. The <b>backward pass</b> walks that graph from the loss back to the inputs. It
        starts with a gradient of 1 at the loss (the loss changes one-for-one with itself) and, at each step, multiplies the
        gradient arriving from above by that step's local derivative and passes the result down to the step's inputs. A value
        that feeds into several steps receives the sum of the gradients coming back from each of them. The scheme is{" "}
        <b>reverse-mode</b> differentiation, also known as <b>backpropagation</b> <Cite k="backprop" />, and one backward pass
        costs about two forward passes for any number of weights.
      </p>
      <Fig n={2} caption={<>The autograd engine, live. The server runs your expression through minitorch. The forward pass fills in each value (blue) and the backward pass each gradient (pink). At <TeX>{String.raw`x = 2,\; w = -0.5,\; b = 3`}</TeX> the chain rule gives <TeX>{String.raw`\partial/\partial x = -2`}</TeX> and <TeX>{String.raw`\partial/\partial w = 8`}</TeX>, the numbers on the graph. Edit the expression and run backward again.</>}>
        <AutogradGraph />
      </Fig>
      <h3>Nineteen operations with gradients</h3>
      <p>
        A tensor in minitorch, an array of numbers with any number of dimensions, stores a reference to the operation that
        produced it. An operation is a small class with two methods. <code>forward</code> computes the result from the raw arrays.{" "}
        <code>backward</code> receives the gradient of the result and returns one gradient per input. For multiplication,{" "}
        <TeX>{String.raw`z = x\,y`}</TeX>, backward returns <TeX>{String.raw`\bar z \cdot y`}</TeX> for <TeX>x</TeX> and{" "}
        <TeX>{String.raw`\bar z \cdot x`}</TeX> for <TeX>y</TeX>, where <TeX>{String.raw`\bar z`}</TeX> is the gradient
        arriving at <TeX>z</TeX>. In general, backward multiplies the incoming gradient by the operation's Jacobian, the table
        of all its partial derivatives, without building that table:
      </p>
      <TeX block>{String.raw`\bar{x}_i \;=\; \sum_j \frac{\partial y_j}{\partial x_i}\,\bar{y}_j, \qquad \text{where } \bar{v} \text{ means } \frac{\partial L}{\partial v}`}</TeX>
      <p>For multiplication the whole class is this (<code>minitorch/mlops.py</code>, without its checks that skip gradients nothing needs):</p>
      <pre className="code-block">{`class Mul(Function):
    def forward(self, x, y):
        self.x, self.y = x, y          # kept for backward
        return x * y

    def backward(self, grad):          # grad is dL/dz for z = x * y
        return grad * self.y, grad * self.x`}</pre>
      <p>
        All of minitorch's gradients live in one file, <code>mlops.py</code>, in nineteen such operations: add, multiply,
        reciprocal, exponential, logarithm, sigmoid, power, sum, max, reshape, transpose, broadcast, indexing, concatenation,
        matrix multiplication, and a few more. Subtraction is adding a negation, and division is multiplying by a reciprocal.
        Softmax, normalization layers, and the loss are short compositions of the nineteen, and none of them defines a
        gradient of its own.
      </p>
      <p>
        Broadcasting needs care. Add a single number to a whole array and NumPy copies the number across the array, and
        NumPy calls this <b>broadcasting</b>. Each copy receives a gradient in the backward pass, so the backward pass has to
        add the copies' gradients back into the one original number. minitorch makes the copy an explicit operation,{" "}
        <code>Expand</code>, whose backward is that sum.
      </p>
      <h3>Checking every gradient two ways</h3>
      <p>
        A wrong gradient does not crash anything. The model trains worse instead, and nothing points you to the cause. minitorch checks
        its gradients against two independent references. The first is the definition of a derivative: nudge an input up by a
        tiny amount <TeX>h</TeX>, then down, and measure the slope.
      </p>
      <TeX block>{String.raw`\frac{\partial f}{\partial x} \;\approx\; \frac{f(x + h) - f(x - h)}{2h}, \qquad h = 10^{-5}`}</TeX>
      <p>
        The check is slow, since it reruns the function twice per input, but it needs no calculus and no trust. minitorch
        compares it with backward's result for {int(r.bkM?.ops.total)} operations and layers, and the button below runs all of
        them on the server.
      </p>
      <Fig n={3} caption="Finite-difference gradient checks, run live. Each line is one operation or layer; it passes when the gradient from backward() agrees with the measured slope for every input.">
        <GradChecks />
      </Fig>
      <p>
        The second reference is PyTorch. I built the same {r.p64 ? `${(r.p64.params / 1e6).toFixed(1)}M` : "…"}-parameter
        language model in both frameworks, copied the same starting weights into both, fed them the same batches of text, and
        trained each for {r.p64?.steps ?? "…"} steps with the same optimizer. If any gradient anywhere were wrong, the two loss
        curves would separate (Figure 4).
      </p>
      {parity && (
        <Fig n={4} caption={<>Training loss of the same model in minitorch (thin blue line) and PyTorch (wide pink line), in {dtype}. They overlap. At step 1 they differ by {sci(parity.rows[0].abs_diff)}; the largest difference over {parity.steps} steps is {sci(parity.max_abs_loss_diff)}.</>}>
          <div className="runs" role="group" aria-label="Precision">
            {(["float64", "float32"] as const).map((d) => (
              <button key={d} aria-pressed={d === dtype} onClick={() => setDtype(d)}>{d}</button>
            ))}
          </div>
          <ParityChart p={parity} />
        </Fig>
      )}
      <p>
        The curves still differ by small amounts, because computers store numbers with limited precision. A 64-bit float
        keeps about 16 significant digits, so each arithmetic step rounds away a little, and two programs that add the same
        numbers in a different order get different last digits. In 64-bit floats the two frameworks agree at step 1 to{" "}
        {sci(r.p64?.rows[0].abs_diff)}, which is rounding error alone, so the math is identical. Each training step feeds the
        previous step's tiny differences forward, and the gap grows to {sci(r.p64?.max_abs_loss_diff)} by the end. In 32-bit
        floats, which keep about 7 digits and which models use in practice, each rounding step is about half a billion times
        larger, and the gap reaches {sci(r.p32?.max_abs_loss_diff)}. Switch the figure between the two to compare.
      </p>
      <p>
        On the laptop's CPU, minitorch takes {num(r.p32?.median_ms_per_step.minitorch, 3)} ms per training step of this
        model and PyTorch takes {num(r.p32?.median_ms_per_step.torch_cpu, 3)} ms, so PyTorch runs{" "}
        {r.p32 ? (r.p32.median_ms_per_step.minitorch / r.p32.median_ms_per_step.torch_cpu).toFixed(1) : "…"} times faster.
      </p>
      <h3 id="lazy">When does an operation run?</h3>
      <p>
        Every operation so far ran the moment it was written. minitorch now waits: <code>c = a @ b</code> records the
        multiplication and its inputs and returns a tensor that knows its shape and dtype but holds no numbers. The
        numbers are computed when something reads them, through <code>.numpy()</code>, <code>.item()</code>, printing,{" "}
        <code>.backward()</code>, or <code>.realize()</code>. tinygrad works the same way <Cite k="tinygrad" />.
      </p>
      <pre className="code-block">{`c = a @ b      # recorded: c.shape is known, nothing has run
y = layer(x)   # recorded and returned: still nothing has run
y.numpy()      # realize: every operation y depends on runs, inputs first, once each`}</pre>
      <p>
        Realizing walks the recorded graph from the inputs to the tensor that was read and runs each operation once; an
        operation nobody reads never runs. For the next operation to be recorded before the last one has run, each of
        the {int(lz?.primitives)} primitives in <code>mlops.py</code> (the nineteen with gradients and the comparison{" "}
        <code>x &lt; y</code>) states its output's shape and dtype from its inputs', and realize checks every stated
        shape against the computed one. An input that already holds numbers is captured as it was, so an optimizer step
        that replaces a weight cannot change a result recorded before it.
      </p>
      {lzCpu && lzMetal && (
        <p>
          Waiting saves nothing by itself. A training step of a {int(lz?.config.n_layers)}-layer story model with{" "}
          {int(lz?.config.d_model)}-wide vectors records {int(lzCpu.both?.ops_recorded_per_step)} operations and runs all{" "}
          {int(lzCpu.both?.ops_run_per_step)}, since every result feeds the loss. On the CPU its losses match eager
          execution bit for bit, and a step takes {num(lzCpu.lazy?.ms_per_step, 3)} ms lazily against{" "}
          {num(lzCpu.eager?.ms_per_step, 3)} ms eagerly. On the Apple GPU two eager runs already differ by up to{" "}
          {sci(lzMetal.both?.loss_gap_eager_eager)}, because the GPU sums in a different order from run to run, and lazy
          and eager runs differ by {sci(lzMetal.both?.loss_gap_eager_lazy)}; a step takes{" "}
          {num(lzMetal.lazy?.ms_per_step, 3)} ms against {num(lzMetal.eager?.ms_per_step, 3)} ms. Laziness gives a
          compiler the whole graph before any of it runs, so the compiler can fuse operations into fewer kernels;
          minitorch records the graph and stops there (<code>bench/lazy.py</code>, <code>tests/lazy.py</code>).
        </p>
      )}
    </Section>
  );
}
