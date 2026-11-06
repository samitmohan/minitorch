// What every section of the post shares: the results (read once from the API), figure and
// table frames, equations, citations, and number formatting. No number is typed into the
// prose by hand; each one comes from bench/results or runs/ through useResults().
import katex from "katex";
import "katex/dist/katex.min.css";
import { createContext, useContext, useMemo, type ReactNode } from "react";
import type { Run } from "../api";
import type { AttnRow, Overhead, Parity } from "../figures/BenchCharts";
import type { Optimum, ScalingRun } from "../figures/IsoFlop";
import type { MetricRow } from "../lib";

export type Backends = {
  ops: { agree: number; total: number };
  training: { steps: number; max_gap: number };
  decode: { max_logit_gap: number };
  mnist?: { steps: number; max_gap: number };
  bfloat16?: { nodes: number; promoted: number; relative_logit_gap: number };
  kernels?: { within_tolerance: number; total: number; tolerance: number };
};
export type QwenParity = {
  prompt_tokens: number;
  largest_logit: number;
  logits: { device: string; dtype: string; max_gap: number; relative_gap: number; argmax_agree: number }[];
  greedy_tokens: number;
  greedy_match: boolean;
  params: number;
  config: Record<string, number | boolean>;
};
export type Gpt2Parity = {
  prompt_tokens: number; largest_logit: number; params: number; greedy_tokens: number;
  logits: { device: string; max_gap: number; relative_gap: number; argmax_agree: number; greedy_match: boolean }[];
};
export type Speculative = {
  rows: { method: string; k: number; tokens_per_s: number; speedup: number; acceptance?: number; first_diff_from_target?: (number | null)[] }[];
};
export type Serve = { block_size: number; scenarios: { scenario: string; requests: number; num_blocks: number; max_batch: number; steps: number; preemptions: number; identical: boolean }[] };
export type Gsm8k = { name: string; n: number; correct: number; accuracy: number; tokens_per_s: number; new_tokens: number; hit_token_limit: number; dtype: string };
export type Quant = {
  rows: { precision: string; weight_mb: number; perplexity: Record<string, number>; decode_tokens_per_s: number; decode_ms_per_token: number; decode_ms_per_token_mlx_kernel?: number; ceiling_tokens_per_s?: number }[];
  bandwidth_gb_s?: number;
  tokens_per_text?: number;
  kernels?: { matrix: string; impl: string; us: number; gb_s: number }[];
};

export type Sd = {
  prompt: string; steps: number; guidance: number; size: number;
  minitorch_metal_s_per_step: number; diffusers_mps_s_per_step: number;
  max_pixel_gap: number; mean_pixel_gap: number; pixels_off_by_more_than_1_of_255: number;
  unet_profile: { wall_ms: number; rows: { op: string; calls: number; ms: number }[] };
};

export type Fa2 = { device: string; dtype: string; rows: { shape: string; impl: string; forward_ms: number; forward_backward_ms: number }[] };

export type MixedPrecision = { device: string; steps: number; last_50_loss_gap: number; rows: { precision: string; step_ms: number; losses: number[]; skipped_steps: number }[] };

export type Checkpointing = { device: string; examples: number; tokens: number; rows: { checkpointing: boolean; micro: number; peak_gb?: number; seconds_per_batch?: number; out_of_memory?: boolean }[] };

export type Paired = { n: number; fixed: number; broke: number; p_value: number };

export type Ablation = { run: string; change: string; lr: number; warmup: number; steps: number; final_val_loss: number | null; best_val_loss: number | null; diverged_at_step: number | null; tokens_per_s: number };

export type Scaling = { runs: ScalingRun[]; optima: Optimum[]; exponent: number; cs336_budget: number; cs336_params_predicted: number; cs336_tokens_predicted: number };

export type Rag = { questions: number; paragraphs: number; k: number; recall_at_1: number; closed_exact: number; closed_f1: number; closed_contains: number;
  rag_exact: number; rag_f1: number; rag_contains: number; index_seconds: number; query_embed_ms: number; closed_answer_s: number; rag_answer_s: number } & Record<string, number>;
export type Agent = { n: number; correct: number; accuracy: number; tool_calls: number; problems_using_tool: number; model_calls: number;
  model_seconds: number; tool_seconds: number; prompt_tokens: number; reply_tokens: number; seconds: number };

export type Lazy = {
  config: Record<string, number>; primitives: number; steps: number;
  rows: { device: string; mode: string; ms_per_step?: number; loss_gap_eager_lazy?: number; loss_gap_eager_eager?: number;
    ops_recorded_per_step?: number; ops_run_per_step?: number }[];
};

export type Lines = { groups: { name: string; files: number; lines: number }[]; package: number; repository: number };

export type Profile = { device: string; batch: number; context: number; step_ms: number; profiled_ms: number; rows: { op: string; calls: number; ms: number }[] };

export type Rollout = {
  rows: { rows: number; prompt_tokens: number; prefill_ms: number; decode_ms_per_step: number; tokens_per_s: number }[];
  step_ms_64_rows: { model: number; logits_to_host: number; sample_on_host: number; sample_on_device: number };
};
export type GrpoRow = {
  step: number; reward: number; groups_with_signal: number; reply_tokens: number; hit_limit: number;
  grad_norm: number; rollout_s: number; update_s: number; rollout_tokens_per_s: number; time: number;
};

export type Results = {
  bench: Record<string, unknown>;
  runs: Run[];
  ts: MetricRow[];  // the TinyStories run's metrics
  lora: MetricRow[];  // the LoRA run's metrics
  grpo: GrpoRow[];  // the GRPO run's metrics
  grpoNoStop: GrpoRow[];  // the first GRPO run, whose reward ignored stopping
  sample?: string;  // its last sample
};

export const ResultsContext = createContext<Results>({ bench: {}, runs: [], ts: [], lora: [], grpo: [], grpoNoStop: [] });

/** Typed views of the result files, plus the numbers derived from them that the prose quotes. */
export function useResults() {
  const { bench: b, runs, ts, lora, grpo, grpoNoStop, sample } = useContext(ResultsContext);
  return useMemo(() => {
    const get = <T,>(k: string) => b[k] as T | undefined;
    const attn = get<{ rows: AttnRow[] }>("flash_attention")?.rows;
    const ovM = get<Overhead>("backend_overhead_cpu_metal");
    const ovC = get<Overhead>("backend_overhead_cuda");
    const bkM = get<Backends>("backends_metal");
    const bkC = get<Backends>("backends_cuda");
    const qwen = get<QwenParity>("qwen3_parity");
    const tsRun = runs.find((r) => r.name === "tinystories");
    const cfg = tsRun?.config as Record<string, number> | undefined;
    const evals = ts.filter((r) => r.val_loss !== undefined);
    const ms = (shape: string, impl: string, dtype = "float32") => attn?.find((r) => r.shape === shape && r.impl === impl && r.dtype === dtype)?.ms;
    const op = (o: Overhead | undefined, device: string, path: string, n = 1000) => o?.elementwise.find((r) => r.device === device && r.path === path && r.n === n)?.us_per_op;
    const steps = [...(ovM?.train_step ?? []), ...(ovC?.train_step ?? [])];
    const step = (impl: string, device: string) => steps.find((r) => r.impl.startsWith(impl) && r.device === device);
    const ratio = (a?: number, c?: number) => (a !== undefined && c !== undefined ? a / c : undefined);
    const tsParams = cfg && cfg.vocab_size * cfg.d_model * 2 + cfg.layers * (4 * cfg.d_model ** 2 + 3 * cfg.d_model * cfg.d_ff + 2 * cfg.d_model) + cfg.d_model;
    return {
      p64: get<Parity>("m1_parity_f64"), p32: get<Parity>("m1_parity"), bpe: get<Record<string, number | boolean>>("bpe"),
      ovM, ovC, attn, bkM, bkC, qwen, serve: get<Serve>("serve_cpu"), cfg, ts, lora, grpo, grpoNoStop, evals, sample, runs,
      gsmBase: get<Gsm8k>("gsm8k_base"), gsmLora: get<Gsm8k>("gsm8k_lora_gsm8k"), gsmLora64: get<Gsm8k>("gsm8k_lora_gsm8k_f64"), gsmGrpo: get<Gsm8k>("gsm8k_grpo_gsm8k"), gsmNoStop: get<Gsm8k>("gsm8k_grpo_no_stop"),
      rollout: get<Rollout>("rollout_cuda"), gpt2: get<Gpt2Parity>("gpt2_parity"),
      spec: get<Speculative>("speculative_metal"),
      quant: get<Quant>("quant"),
      profM: get<Profile>("profile_metal"), profC: get<Profile>("profile_cuda"),
      fa2: get<Fa2>("flash_attention_cuda"),
      lines: get<Lines>("lines"), lazy: get<Lazy>("lazy"),
      rag: get<Rag>("rag"), agent: get<Agent>("gsm8k_agent"), vsAgent: get<Paired>("gsm8k_base_vs_gsm8k_agent"),
      agentStep: get<Agent>("gsm8k_agent_step_by_step"), vsAgentStep: get<Paired>("gsm8k_base_vs_gsm8k_agent_step_by_step"),
      scaling: get<Scaling>("scaling"),
      abl: get<{ runs: Ablation[] }>("ablations")?.runs,
      vsGrpo: get<Paired>("gsm8k_base_vs_gsm8k_grpo_gsm8k"), vsLora: get<Paired>("gsm8k_base_vs_gsm8k_lora_gsm8k"),
      ckC: get<Checkpointing>("checkpointing_cuda"), ckM: get<Checkpointing>("checkpointing_metal"),
      mpC: get<MixedPrecision>("mixed_precision_cuda"), mpM: get<MixedPrecision>("mixed_precision_metal"),
      sd: get<Sd>("stable_diffusion"), sdParity: get<Record<string, number>>("sd_parity"),
      kernelWin: ratio(ms("CS336 training", "mx.fast.sdpa (oracle)"), ms("CS336 training", "v3 simdgroup matmul")),
      kernelLoss: ratio(ms("Qwen3 prefill 2k", "v3 simdgroup matmul"), ms("Qwen3 prefill 2k", "mx.fast.sdpa (oracle)")),
      decodeRatio: ratio(ms("decode, 2k cache", "v1 thread per row"), ms("decode, 2k cache", "mx.fast.sdpa (oracle)")),
      finalVal: evals.at(-1)?.val_loss,
      firstVal: evals[0]?.val_loss,
      minutes: ts.length ? ts[ts.length - 1].time / 60 : undefined,
      tokS: ts.length ? median(ts.map((r) => r.tokens_per_s)) : undefined,
      mfu: ts.length && ts[0].mfu !== undefined ? median(ts.map((r) => r.mfu ?? 0)) : undefined,
      tflops: ts.length ? median(ts.map((r) => r.tflops)) : undefined,
      runDone: cfg !== undefined && ts.length > 0 && ts[ts.length - 1].step === cfg.steps,
      tsParams,
      tsTokens: cfg ? cfg.steps * cfg.batch * cfg.context : undefined,
      opNumpy: op(ovM, "cpu", "minitorch"), opMlx: op(ovM, "metal", "minitorch"), opCupy: op(ovC, "cuda", "minitorch"),
      syncMlx: op(ovM, "metal", "sync every op"), syncCupy: op(ovC, "cuda", "sync every op"),
      bigNumpy: op(ovM, "cpu", "minitorch", 1e7), bigMlx: op(ovM, "metal", "minitorch", 1e7), bigCupy: op(ovC, "cuda", "minitorch", 1e7),
      mlxShare: ratio(step("minitorch", "metal")?.tokens_per_s, step("plain MLX", "metal")?.tokens_per_s),
      cudaVsCpu: ratio(step("minitorch", "cuda")?.tokens_per_s, step("minitorch", "cpu")?.tokens_per_s),
      peakMatmul: (device: string, dtype = "float32") => {
        const t = [...(ovM?.matmul ?? []), ...(ovC?.matmul ?? [])].filter((r) => r.device === device && r.dtype === dtype).map((r) => r.tflops);
        return t.length ? Math.max(...t) : undefined;  // no result file yet: "…", not 0 TFLOP/s
      },
      worstTrainGap: bkM && bkC ? Math.max(bkM.training.max_gap, bkC.training.max_gap) : bkM?.training.max_gap,
      qwenCpu: qwen?.logits.find((l) => l.device === "cpu"),
      qwenBf16: qwen?.logits.find((l) => l.dtype === "bfloat16"),
    };
  }, [b, runs, ts, lora, grpo, grpoNoStop, sample]);
}

export const REFS = [
  { k: "micrograd", text: "A. Karpathy. micrograd: a tiny scalar-valued autograd engine. 2020.", url: "https://github.com/karpathy/micrograd" },
  { k: "tinygrad", text: "G. Hotz et al. tinygrad. 2020.", url: "https://github.com/tinygrad/tinygrad" },
  { k: "backprop", text: "D. Rumelhart, G. Hinton, R. Williams. Learning representations by back-propagating errors. Nature, 1986.", url: "https://www.nature.com/articles/323533a0" },
  { k: "bpe", text: "R. Sennrich, B. Haddow, A. Birch. Neural Machine Translation of Rare Words with Subword Units. ACL 2016.", url: "https://arxiv.org/abs/1508.07909" },
  { k: "gpt2", text: "A. Radford et al. Language Models are Unsupervised Multitask Learners. 2019.", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" },
  { k: "transformer", text: "A. Vaswani et al. Attention Is All You Need. NeurIPS 2017.", url: "https://arxiv.org/abs/1706.03762" },
  { k: "tinystories", text: "R. Eldan, Y. Li. TinyStories: How Small Can Language Models Be and Still Speak Coherent English? 2023.", url: "https://arxiv.org/abs/2305.07759" },
  { k: "cs336", text: "Stanford CS336: Language Modeling from Scratch. 2025.", url: "https://stanford-cs336.github.io/" },
  { k: "rmsnorm", text: "B. Zhang, R. Sennrich. Root Mean Square Layer Normalization. NeurIPS 2019.", url: "https://arxiv.org/abs/1910.07467" },
  { k: "rope", text: "J. Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. 2021.", url: "https://arxiv.org/abs/2104.09864" },
  { k: "swiglu", text: "N. Shazeer. GLU Variants Improve Transformer. 2020.", url: "https://arxiv.org/abs/2002.05202" },
  { k: "gqa", text: "J. Ainslie et al. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP 2023.", url: "https://arxiv.org/abs/2305.13245" },
  { k: "adam", text: "D. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. ICLR 2015.", url: "https://arxiv.org/abs/1412.6980" },
  { k: "adamw", text: "I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. ICLR 2019.", url: "https://arxiv.org/abs/1711.05101" },
  { k: "online", text: "M. Milakov, N. Gimelshein. Online normalizer calculation for softmax. 2018.", url: "https://arxiv.org/abs/1805.02867" },
  { k: "flash", text: "T. Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.", url: "https://arxiv.org/abs/2205.14135" },
  { k: "flash2", text: "T. Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. 2023.", url: "https://arxiv.org/abs/2307.08691" },
  { k: "paged", text: "W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023.", url: "https://arxiv.org/abs/2309.06180" },
  { k: "orca", text: "G.-I. Yu et al. Orca: A Distributed Serving System for Transformer-Based Generative Models. OSDI 2022.", url: "https://www.usenix.org/conference/osdi22/presentation/yu" },
  { k: "qwen3", text: "Qwen Team. Qwen3 Technical Report. 2025.", url: "https://arxiv.org/abs/2505.09388" },
  { k: "specdec", text: "Y. Leviathan, M. Kalman, Y. Matias. Fast Inference from Transformers via Speculative Decoding. ICML 2023.", url: "https://arxiv.org/abs/2211.17192" },
  { k: "specsample", text: "C. Chen et al. Accelerating Large Language Model Decoding with Speculative Sampling. 2023.", url: "https://arxiv.org/abs/2302.01318" },
  { k: "lora", text: "E. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.", url: "https://arxiv.org/abs/2106.09685" },
  { k: "gsm8k", text: "K. Cobbe et al. Training Verifiers to Solve Math Word Problems. 2021.", url: "https://arxiv.org/abs/2110.14168" },
  { k: "reinforce", text: "R. Williams. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning, 1992.", url: "https://link.springer.com/article/10.1007/BF00992696" },
  { k: "grpo", text: "Z. Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. 2024.", url: "https://arxiv.org/abs/2402.03300" },
  { k: "drgrpo", text: "Z. Liu et al. Understanding R1-Zero-Like Training: A Critical Perspective. 2025.", url: "https://arxiv.org/abs/2503.20783" },
  { k: "lorawr", text: "J. Schulman and Thinking Machines Lab. LoRA Without Regret. 2025.", url: "https://thinkingmachines.ai/blog/lora/" },
  { k: "ddpm", text: "J. Ho, A. Jain, P. Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020.", url: "https://arxiv.org/abs/2006.11239" },
  { k: "ddim", text: "J. Song, C. Meng, S. Ermon. Denoising Diffusion Implicit Models. ICLR 2021.", url: "https://arxiv.org/abs/2010.02502" },
  { k: "ldm", text: "R. Rombach et al. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.", url: "https://arxiv.org/abs/2112.10752" },
  { k: "clip", text: "A. Radford et al. Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.", url: "https://arxiv.org/abs/2103.00020" },
  { k: "cfg", text: "J. Ho, T. Salimans. Classifier-Free Diffusion Guidance. 2022.", url: "https://arxiv.org/abs/2207.12598" },
  { k: "triton", text: "P. Tillet, H. T. Kung, D. Cox. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. MAPL 2019.", url: "https://github.com/triton-lang/triton" },
  { k: "mixed", text: "P. Micikevicius et al. Mixed Precision Training. ICLR 2018.", url: "https://arxiv.org/abs/1710.03740" },
  { k: "checkpoint", text: "T. Chen, B. Xu, C. Zhang, C. Guestrin. Training Deep Nets with Sublinear Memory Cost. 2016.", url: "https://arxiv.org/abs/1604.06174" },
  { k: "nope", text: "A. Haviv, O. Ram, O. Press, P. Izsak, O. Levy. Transformer Language Models without Positional Encodings Still Learn Positional Information. Findings of EMNLP 2022.", url: "https://arxiv.org/abs/2203.16634" },
  { k: "chinchilla", text: "J. Hoffmann et al. Training Compute-Optimal Large Language Models. NeurIPS 2022.", url: "https://arxiv.org/abs/2203.15556" },
  { k: "rag", text: "P. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.", url: "https://arxiv.org/abs/2005.11401" },
  { k: "squad", text: "P. Rajpurkar, J. Zhang, K. Lopyrev, P. Liang. SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP 2016.", url: "https://arxiv.org/abs/1606.05250" },
  { k: "qwen3emb", text: "Y. Zhang et al. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. 2025.", url: "https://arxiv.org/abs/2506.05176" },
  { k: "react", text: "S. Yao et al. ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.", url: "https://arxiv.org/abs/2210.03629" },
  { k: "mlx", text: "A. Hannun, J. Digani, A. Katharopoulos, R. Collobert. MLX: Efficient and flexible machine learning on Apple silicon. 2023.", url: "https://github.com/ml-explore/mlx" },
  { k: "cupy", text: "R. Okuta et al. CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations. LearningSys 2017.", url: "https://github.com/cupy/cupy" },
] as const;

export function Cite({ k }: { k: (typeof REFS)[number]["k"] }) {
  const i = REFS.findIndex((r) => r.k === k);
  return (
    <a className="cite" href={`#ref-${i + 1}`} title={REFS[i].text}
      onClick={(e) => {
        e.preventDefault();
        document.getElementById(`ref-${i + 1}`)?.scrollIntoView({ behavior: "smooth" });
      }}>
      [{i + 1}]
    </a>
  );
}

export function TeX({ children, block = false }: { children: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: block, throwOnError: false }), [children, block]);
  return block ? <div className="eq" dangerouslySetInnerHTML={{ __html: html }} /> : <span dangerouslySetInnerHTML={{ __html: html }} />;
}

/** A numbered figure (caption below) or table (caption above). `wide` lets it outgrow the text column. */
export function Fig({ id, n, caption, children, table = false, wide = false }: {
  id?: string; n: number | string; caption: ReactNode; children: ReactNode; table?: boolean; wide?: boolean;
}) {
  return (
    <figure id={id ?? `${table ? "table" : "figure"}-${n}`} className={wide ? "wide" : undefined}>
      {table && <figcaption className="above"><b>Table {n}:</b> {caption}</figcaption>}
      {children}
      {!table && <figcaption><b>Figure {n}:</b> {caption}</figcaption>}
    </figure>
  );
}

/** A plain-language aside: a definition or a worked example a newcomer may want. */
export function Aside({ title, children }: { title: string; children: ReactNode }) {
  return (
    <aside className="aside">
      <b>{title}</b>
      {children}
    </aside>
  );
}

export const median = (xs: number[]) => [...xs].sort((a, c) => a - c)[Math.floor(xs.length / 2)];
export const mean = (xs: number[]) => xs.reduce((a, x) => a + x, 0) / xs.length;
export const sci = (v: number | undefined, d = 1) => (v === undefined || !Number.isFinite(v) ? "…" : v.toExponential(d).replace("e-", "e−").replace("e+", "e"));
export const num = (v: number | undefined, d = 4) => {
  if (v === undefined || Number.isNaN(v)) return "…";
  if (Math.abs(v) >= 1e4) return Math.round(v).toLocaleString("en-US");
  if (v !== 0 && Math.abs(v) < 1e-3) return sci(v);
  return Number(v.toPrecision(d)).toString();
};
export const int = (v: number | undefined) => (v === undefined || !Number.isFinite(v) ? "…" : Math.round(v).toLocaleString("en-US"));
export const pct = (v: number | undefined, d = 1) => (v === undefined || !Number.isFinite(v) ? "…" : `${(100 * v).toFixed(d)}%`);
/** A p-value in paper notation, without rounding a tiny one to 0. */
export const pval = (p: number) => (p < 1e-4 ? "p < 0.0001" : `p = ${Number(p.toPrecision(2))}`);
/** A p-value as a share of the time, without rounding a tiny one to 0%. */
export const chance = (p: number) => (p < 1e-4 ? "less than 0.01%" : pct(p, p < 0.1 ? 1 : 0));

/** A numbered top-level section; the table of contents is built from these ids and titles. */
export function Section({ id, n, title, children }: { id: string; n?: number; title: string; children: ReactNode }) {
  return (
    <section id={id} className="post-section" aria-label={title}>
      <h2>{n !== undefined && <span>{n}</span>}{title}</h2>
      {children}
    </section>
  );
}
