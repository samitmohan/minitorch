import { AttentionHeads } from "../figures/AttentionHeads";
import { BpeTrace } from "../figures/BpeTrace";
import { Generate } from "../figures/Generate";
import { RunCurve } from "../figures/RunCurve";
import { Aside, Cite, Fig, Section, TeX, int, num, useResults } from "./kit";

export function Tokens() {
  const r = useResults();
  const b = r.bpe;
  return (
    <Section id="tokens" n={3} title="Turning text into numbers">
      <p>
        A network computes on numbers, so before a language model can read anything, text has to become a sequence of
        integers. A <b>tokenizer</b> does this. It cuts text into <b>tokens</b> from a fixed list, the <b>vocabulary</b>, and
        replaces each token with its position in that list.
      </p>
      <p>
        Single characters make a tiny vocabulary, but sentences become long sequences, and the model must learn to spell
        before it can learn anything else. Whole words make sequences short, but the vocabulary explodes, and the model has no
        way to represent a word missing from it, such as a name or a typo. Modern models use <b>subwords</b>. Common words get
        one token each, and rare words get a few.
      </p>
      <p>
        minitorch's tokenizer uses <b>byte-level byte-pair encoding</b> <Cite k="bpe" /> <Cite k="gpt2" />, the scheme behind
        GPT-2, GPT-4, and Qwen. It starts from bytes. Text in any language, emoji included, is a sequence of UTF-8 bytes, and
        only 256 different bytes exist, so no input is unknown. Then it learns <b>merges</b>. It finds the pair of adjacent
        symbols that occurs most often in the training text, adds that pair to the vocabulary as a new symbol, and repeats
        until the vocabulary reaches the size you asked for.
      </p>
      <Aside title="Four merges on a tiny corpus">
        <p>
          Suppose the training text contains <i>low</i> 5 times, <i>lower</i> 2 times, <i>newest</i> 6 times, and{" "}
          <i>widest</i> 3 times. Count every adjacent pair of letters, weighting each word by how often it appears. The pairs{" "}
          <code>e s</code> and <code>s t</code> both occur 9 times (6 in <i>newest</i>, 3 in <i>widest</i>), more than any
          other, so the first merge creates the symbol <code>es</code>. Recount, and <code>es t</code> now leads with 9, so the
          second merge creates <code>est</code>. Then <code>l o</code> and <code>lo w</code> lead with 7 each. After four
          merges, <i>low</i> is a single token and <i>newest</i> is <code>n e w est</code>.
        </p>
      </Aside>
      <p>
        Encoding new text replays the merges in training order. minitorch adds two refinements. First, a regular expression
        cuts the text into words, numbers, and punctuation before any merging, so a merge never spans two words, and{" "}
        <i>dog</i> in "dog." and "dog!" becomes the same token. Second, for speed, minitorch counts word frequencies in
        parallel across the processor's cores, and after each merge it recounts pairs in the words that contained the merged
        pair instead of rescanning the whole corpus.
      </p>
      {b && (
        <Fig n={1} table caption="Training a 10,000-token vocabulary on the TinyStories training set, then encoding the whole set with it.">
          <table className="results">
            <tbody>
              <tr><th>Vocabulary size</th><td className="num">{int(b.bpe_vocab_size as number)} tokens</td></tr>
              <tr><th>Training time</th><td className="num">{num(b.bpe_train_seconds as number)} s</td></tr>
              <tr><th>Peak memory of the largest worker process</th><td className="num">{int(b.bpe_train_peak_rss_mb_largest_worker as number)} MB</td></tr>
              <tr><th>Tokens in the training set</th><td className="num">{int(b.train_tokens as number)}</td></tr>
              <tr><th>Encoding speed</th><td className="num">{num(b.train_encode_mb_per_s as number)} MB/s</td></tr>
              <tr><th>Average bytes per token</th><td className="num">{num(b.train_bytes_per_token as number)}</td></tr>
              <tr><th>Decoding the held-out set back to text</th><td className="num">{b.valid_round_trip_exact ? "byte-for-byte exact" : "differs"}</td></tr>
            </tbody>
          </table>
        </Fig>
      )}
      <p>
        An average token covers {num(b?.train_bytes_per_token as number, 3)} bytes of text, so the model reads about a
        quarter as many tokens as there are characters. Try it on your own text.
      </p>
      <Fig n={5} caption="The tokenizer, live. Type any text and press Tokenize. Each box is one token with its id. Then pick a word below to watch its bytes merge, step by step, into the final tokens.">
        <BpeTrace />
      </Fig>
      <p>
        Two references check the tokenizer. Loaded with GPT-2's published merges, it produces the same token ids as tiktoken,
        OpenAI's tokenizer library. Loaded with Qwen3's vocabulary, Qwen's regular expression, which keeps each digit a
        separate token so numbers stay easy to do arithmetic on, and Unicode normalization, it produces the same ids as
        Hugging Face's tokenizer on each test text, including stories, Chinese, emoji, code, and chat transcripts.
      </p>
      <p>
        Chat models also use <b>special tokens</b>, reserved ids that ordinary text never produces.{" "}
        <code>&lt;|endoftext|&gt;</code> marks where one document ends and the next begins, and <code>&lt;|im_start|&gt;</code>{" "}
        and <code>&lt;|im_end|&gt;</code> mark who is speaking in a conversation. I wrote the chat template that wraps
        messages in these markers by hand, and it matches Hugging Face's character for character.
      </p>
    </Section>
  );
}

export function Model() {
  const r = useResults();
  const q = r.qwen?.config;
  const c = r.cfg;
  const yesNo = (v: number | boolean | undefined) => {
    if (v === undefined) return "…";
    return v ? "yes" : "no";
  };
  return (
    <Section id="model" n={4} title="A language model">
      <p>
        Given a sequence of tokens, a language model assigns a probability to each token in the vocabulary being the next
        one. To write text, you sample a token from those probabilities, append it, and ask again. A chat model writes stories
        and does arithmetic by repeating that step.
      </p>
      <p>
        minitorch's model is a <b>decoder-only transformer</b> <Cite k="transformer" />, the architecture behind GPT, Llama,
        and Qwen, in the modern form taught in Stanford's CS336 <Cite k="cs336" />. Follow one token through it.
      </p>
      <p>
        <b>Embedding.</b> The token id selects a row from a learned table, a vector of <TeX>d</TeX> numbers (512 for the story
        model). Think of the vector as the model's working notes about that position. The notes flow upward through a stack of
        identical layers, and each layer reads them and adds its own update. That running sum is the{" "}
        <b>residual stream</b>.
      </p>
      <p>
        <b>Attention</b> is how a position gathers information from earlier positions. Each position computes three vectors
        from its notes: a <b>query</b> ("what am I looking for?"), a <b>key</b> ("what do I contain?"), and a <b>value</b>{" "}
        ("what do I pass on if chosen?"). The score between two positions is the dot product of one's query with the other's
        key. Softmax turns the scores into weights that are positive and sum to 1, and the output is the weighted sum of the
        values:
      </p>
      <TeX block>{String.raw`\operatorname{softmax}(s)_i = \frac{e^{s_i}}{\sum_j e^{s_j}}, \qquad \text{Attention}(Q, K, V) = \operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d}} + M\Big)\, V`}</TeX>
      <p>
        The mask <TeX>M</TeX> holds <TeX>{String.raw`-\infty`}</TeX> wherever a position would look at a later one, so a
        position never sees the future it is trying to predict. Dividing by <TeX>{String.raw`\sqrt{d}`}</TeX> keeps the scores
        from growing with the vector length. A layer runs several of these attention computations side by side, the{" "}
        <b>heads</b>, each with its own weights and so its own pattern. In <b>grouped-query attention</b> <Cite k="gqa" />,
        several query heads share one set of keys and values, which shrinks the memory they take during generation (§9).
      </p>
      <p>
        Figure 6 runs the trained story model (§5) on a sentence and draws each head's weights. Most heads spread their weight over several earlier tokens, and their patterns shift from sentence to
        sentence. A few have a habit you can spot. The figure opens on layer 3, head 11, where words inside a phrase look
        back to the verb or preposition that introduces it: <i>little</i>, <i>girl</i>, and the first <i>Lily</i> look at{" "}
        <i>was</i>, and <i>red</i> and <i>ball</i> look at <i>with</i>. Layer 1, head 15 parks much of its weight on the first
        token, a resting place for a head when nothing earlier matters. Type a sentence of your own to see the patterns move.
      </p>
      <Fig n={6} caption="Attention patterns of the trained story model, live. Rows are its layers and columns its heads; click a thumbnail to enlarge it. In the enlarged view each row is a token looking back at the tokens before it, and darker cells carry more weight (on a square-root scale, so faint patterns stay visible).">
        <AttentionHeads />
      </Fig>
      <p>
        <b>Position.</b> Attention by itself ignores word order. Shuffle the earlier tokens and the weighted sum stays the same.
        Rotary position embeddings (RoPE) <Cite k="rope" /> fix this by rotating each query and key by an angle proportional
        to its position <TeX>p</TeX>, a different speed for each pair of dimensions. Because rotating both vectors changes their
        dot product only by the difference of their angles, scores end up depending on how far apart two tokens are:
      </p>
      <TeX block>{String.raw`\begin{pmatrix} x_i \\ x_{i+d/2} \end{pmatrix} \mapsto \begin{pmatrix} \cos p\theta_i & -\sin p\theta_i \\ \sin p\theta_i & \cos p\theta_i \end{pmatrix} \begin{pmatrix} x_i \\ x_{i+d/2} \end{pmatrix}, \qquad \theta_i = b^{-2i/d}`}</TeX>
      <p>
        <b>Feed-forward.</b> After attention, each position runs its notes through a small two-layer network on its own. The
        modern version is SwiGLU <Cite k="swiglu" />: two projections of the input, one passed through a smooth on-off
        function (SiLU) that acts as a gate on the other, then a projection back down. <b>Normalization</b> before each of
        these blocks rescales the notes to a steady size, so values neither vanish nor explode across dozens of layers. The
        RMSNorm <Cite k="rmsnorm" /> version divides by the root mean square and multiplies by learned gains <TeX>g</TeX>:
      </p>
      <TeX block>{String.raw`\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}} \odot g, \qquad \operatorname{FFN}(x) = W_2\big(\operatorname{SiLU}(W_1 x) \odot W_3 x\big)`}</TeX>
      <p>In code, a whole layer is two lines (<code>minitorch/gpt.py</code>):</p>
      <pre className="code-block">{`def forward(self, x, cache=None, pad=None):
    x = x + self.attn(self.attn_norm(x), cache, pad)   # attention, added to the residual stream
    return x + self.ffn(self.ffn_norm(x))              # feed-forward, added the same way`}</pre>
      <p>
        <b>Output.</b> After the last layer, one more normalization, then a multiplication by a matrix with one row per
        vocabulary entry, gives a score per token, the <b>logit</b>. Softmax turns the logits into the probabilities we started
        with. Table 2 lists the two configurations this post uses: the small story model, and Qwen3-0.6B (§8), which
        is the same design with four settings changed, so one class in minitorch covers both.
      </p>
      <Fig n={2} table caption="The two models in this post. The parameter count is the number of adjustable numbers.">
        <table className="results">
          <thead><tr><th>Setting</th><th className="num">Story model (CS336)</th><th className="num">Qwen3-0.6B</th></tr></thead>
          <tbody>
            <tr><th>Vocabulary</th><td className="num">{int(c?.vocab_size)}</td><td className="num">{int(q?.vocab_size as number)}</td></tr>
            <tr><th>Vector width <TeX>d</TeX></th><td className="num">{int(c?.d_model)}</td><td className="num">{int(q?.d_model as number)}</td></tr>
            <tr><th>Layers</th><td className="num">{int(c?.layers)}</td><td className="num">{int(q?.n_layers as number)}</td></tr>
            <tr><th>Query heads / key-value heads</th><td className="num">{c ? `${c.heads} / ${c.heads}` : "…"}</td><td className="num">{q ? `${q.n_heads} / ${q.n_kv_heads}` : "…"}</td></tr>
            <tr><th>Width of one head</th><td className="num">{c ? c.d_model / c.heads : "…"}</td><td className="num">{int(q?.head_dim as number)}</td></tr>
            <tr><th>Feed-forward width</th><td className="num">{int(c?.d_ff)}</td><td className="num">{int(q?.d_ff as number)}</td></tr>
            <tr><th>RoPE base <TeX>b</TeX></th><td className="num">10,000</td><td className="num">{int(q?.rope_theta as number)}</td></tr>
            <tr><th>Normalization on queries and keys</th><td className="num">no</td><td className="num">{yesNo(q?.qk_norm)}</td></tr>
            <tr><th>Output matrix shared with the embedding</th><td className="num">no</td><td className="num">{yesNo(q?.tie_embeddings)}</td></tr>
            <tr><th>Parameters</th><td className="num">{int(r.tsParams)}</td><td className="num">{int(r.qwen?.params)}</td></tr>
          </tbody>
        </table>
      </Fig>
    </Section>
  );
}

export function Training() {
  const r = useResults();
  const c = r.cfg;
  const abl = (run: string) => r.abl?.find((x) => x.run === run);
  const base = abl("abl_baseline"), post = abl("abl_post_norm"), silu = abl("abl_silu"), nope = abl("abl_nope");
  const noNorm = abl("abl_no_rmsnorm"), noNormLow = abl("abl_no_rmsnorm_lr3e-4");
  const cost = (x?: typeof base) => (x?.final_val_loss != null && base?.final_val_loss != null ? x.final_val_loss - base.final_val_loss : undefined);
  return (
    <Section id="training" n={5} title="Can it train a language model?">
      <p>
        Training needs a loss, and for next-token prediction the loss is <b>cross-entropy</b>: the negative logarithm of the
        probability the model gave to the token that came next. Give the right token probability 1 and the loss is 0.
        Give it 0.25 and the loss is <TeX>{String.raw`-\ln 0.25 \approx 1.39`}</TeX>. A model that guesses at random among{" "}
        {int(c?.vocab_size)} tokens scores <TeX>{String.raw`\ln 10{,}000 \approx 9.21`}</TeX>, and the untrained model's
        first step scores {num(r.ts[0]?.loss, 3)}, since random weights make near-uniform guesses. The exponential of the
        loss, the <b>perplexity</b>, is easier to read. A perplexity of 4 means the model is as unsure as a uniform choice
        among 4 tokens.
      </p>
      <p>
        The data is TinyStories <Cite k="tinystories" />: millions of short children's stories written by GPT-4 in the
        vocabulary of a small child. It is small enough to train on one GPU in an hour and simple enough that a small model
        learns to write coherent stories, which makes progress easy to see. Training repeats one <b>step</b> {int(c?.steps)}{" "}
        times: take {c?.batch ?? "…"} random windows of {c?.context ?? "…"} tokens from the text, compute the loss at every
        position, run backward, and update every weight.
      </p>
      <Aside title="The parts of a training step">
        <ul>
          <li>
            <b>AdamW</b> <Cite k="adam" /> <Cite k="adamw" /> is the update rule. Plain gradient descent uses one step size for
            every weight. Adam keeps a running average of each weight's gradient and of its square, and divides one by the
            square root of the other, so weights whose gradients stay small still move, and weights with noisy gradients move
            less. The "W" adds <b>weight decay</b>, which shrinks each weight a little toward zero at each step so weights grow
            only when the data asks for it.
          </li>
          <li>
            <b>The learning rate schedule</b> starts small and ramps up over {int(c?.warmup)} steps (<b>warmup</b>), because
            gradients from random weights are unreliable. It then decays along a cosine curve to {c?.min_lr ?? "…"}, so the
            final steps make fine adjustments instead of large jumps.
          </li>
          <li>
            <b>Gradient clipping</b> rescales the whole gradient whenever its length exceeds {c?.clip ?? "…"}, so one unusual
            batch cannot throw the weights far off course.
          </li>
          <li>
            <b>Validation</b> scores the model every {int(c?.eval_every)} steps on stories it never trains on. If training loss
            keeps falling while validation loss rises, the model is memorizing its training stories.
          </li>
        </ul>
      </Aside>
      <p>
        The full run used Stanford CS336's configuration: the {r.tsParams ? `${(r.tsParams / 1e6).toFixed(1)}M` : "…"}-parameter
        model of Table 2, {r.tsTokens ? `${(r.tsTokens / 1e6).toFixed(1)}M` : "…"} tokens in all, learning rate {c?.lr ?? "…"},
        weight decay {c?.weight_decay ?? "…"}, on one {r.ovC?.gpu ?? "…"} through minitorch's CuPy backend (§6). Figure 7
        shows its loss curves.
      </p>
      <Fig n={7} caption={<>The TinyStories run, from <code>runs/tinystories</code>: train loss in blue, validation loss in pink, on a log scale. {r.runDone ? <>The TinyStories run ends at a validation loss of {num(r.finalVal)}, a perplexity of {num(r.finalVal !== undefined ? Math.exp(r.finalVal) : undefined, 3)}.</> : null}</>}>
        <RunCurve rows={r.ts} />
      </Fig>
      <p>
        Validation loss falls from {num(r.firstVal)} after the first {int(c?.eval_every)} steps to {num(r.finalVal)} at the
        end, under CS336's target of 1.45, in {num(r.minutes, 2)} minutes at a median {int(r.tokS)} tokens per second. The
        validation curve tracks the training curve the whole way, so the model predicts stories it never saw about as well as
        the ones it trained on.
      </p>
      {r.sample && (
        <blockquote className="sample">
          <small>A sample from the final checkpoint, cut off at its length limit:</small>
          <p>{r.sample}</p>
        </blockquote>
      )}
      <p>
        <b>Temperature</b> and <b>top-p</b> control how the next token is drawn. Temperature divides the logits before the
        softmax: below 1 it sharpens the distribution toward the likeliest token, above 1 it flattens it, and 0 means always
        take the likeliest. Top-p keeps the smallest set of likeliest tokens whose probabilities add up to p, which cuts off
        the long tail of unlikely tokens. Try both on any checkpoint.
      </p>
      <Fig n={8} caption="Generate from a trained checkpoint, live. The server runs minitorch's model with a key-value cache (§9) on the CPU; generated text follows the prompt in plain type.">
        <Generate />
      </Fig>
      {base && post && silu && nope && noNorm && (
        <>
          <h3>Which parts matter</h3>
          <p>
            §4 built the model from parts that recent transformers share: RMSNorm before each sublayer, RoPE for position, and
            SwiGLU in the feed-forward layer. CS336's ablations take each one out and train again. Every variant below trains the
            story model for {int(base.steps)} steps, a quarter of the full run, with everything else unchanged (Table A5).
          </p>
          <p>
            The model as built ends at a validation loss of {num(base.final_val_loss ?? undefined)}. Moving RMSNorm after each
            sublayer, where the original transformer put it, costs {num(cost(post), 2)}. Replacing SwiGLU with a plain SiLU layer,
            widened to the same number of parameters, costs {num(cost(silu), 2)}. Dropping position information entirely costs{" "}
            {num(cost(nope), 2)}, the most of the three, yet the model still learns. Haviv et al. <Cite k="nope" /> found the
            same and traced it to the causal mask: each position sees a different number of earlier tokens, which tells it
            roughly where it is.
          </p>
          <p>
            Without RMSNorm, nothing rescales the vectors between sublayers, and at the full learning rate of {noNorm.lr} the run
            diverged: its loss stopped being a number at step {int(noNorm.diverged_at_step ?? undefined)}, {int((noNorm.diverged_at_step ?? 0) - noNorm.warmup)}{" "}
            steps after warmup reached the peak rate.
            {noNormLow && (noNormLow.diverged_at_step === null
              ? <> At {noNormLow.lr} it trains to a validation loss of {num(noNormLow.final_val_loss ?? undefined)}, {num(cost(noNormLow), 2)} behind the model with RMSNorm at its own rate.</>
              : <> At {noNormLow.lr} it diverged too, at step {int(noNormLow.diverged_at_step ?? undefined)}.</>)}
          </p>
        </>
      )}
      <p>
        One more switch changes the residual itself. <code>TransformerLM(..., attn_res=True)</code> replaces the plain sum
        of sublayer outputs with Block Attention Residuals: before every sublayer, a learned query takes a softmax over the
        outputs of earlier layers and mixes them, starting from an equal mix. The story model has not been trained with it
        here; the <a href="https://samitmohan.in/blog/attention-residuals">Attention Residuals</a> post explains it and
        measures it on a small model.
      </p>
    </Section>
  );
}
