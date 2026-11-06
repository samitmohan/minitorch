import { AgentDemo } from "../figures/AgentDemo";
import { RagDemo } from "../figures/RagDemo";
import { Cite, Fig, Section, chance, int, num, pct, useResults } from "./kit";

export function Tools() {
  const r = useResults();
  const g = r.rag, a = r.agent, base = r.gsmBase, vs = r.vsAgent;
  return (
    <Section id="tools" n={13} title="Can it answer from documents and use a tool?">
      <p>
        Most systems built on language models do one of two things: answer from documents the model looks up, or let it call
        tools. Here both run on minitorch end to end. They are programs that talk to the server of §9 through its chat
        endpoint, the way an application uses any hosted model, and minitorch computes every number between the question and
        the answer, from the embedding to the last sampled token.
      </p>
      {g && (
        <>
          <h3>Retrieval</h3>
          <p>
            A small model knows few facts. <b>Retrieval-augmented generation</b> <Cite k="rag" /> looks up passages that bear
            on a question and puts them in the prompt. The test is SQuAD v1.1 <Cite k="squad" />: questions people wrote about
            Wikipedia paragraphs, with the answers they marked in the text. Its development set has {int(g.paragraphs)}{" "}
            paragraphs.
          </p>
          <p>
            Finding the right paragraph needs a way to compare meanings. An <b>embedding model</b> maps a text to a vector so
            that texts about the same thing point the same way. Qwen3-Embedding-0.6B <Cite k="qwen3emb" /> has Qwen3's
            architecture with its own weights, so minitorch runs it with the code of §8, and its embedding of a text is the final
            hidden state of the last token, scaled to length 1. The index is a NumPy matrix with one row per paragraph
            {g.index_seconds > 0 ? `, built once in ${num(g.index_seconds, 3)} s on the Apple GPU` : ""}. Embedding a question
            takes {num(g.query_embed_ms, 2)} ms, and the search is one matrix product with the index and a sort: the largest
            dot products are the nearest paragraphs.
          </p>
          <pre className="code-block">{`query = embed(model, tok, [INSTRUCT + question], device)[0]   # 1,024 numbers, length 1
scores = index @ query                                          # one dot product per paragraph
top = np.argsort(-scores)[:3]                                   # the three nearest`}</pre>
          <p>
            For {int(g.questions)} questions drawn at random, the paragraph the question was written from ranks first{" "}
            {pct(g.recall_at_1, 1)} of the time and among the top {g.k} paragraphs {pct(g[`recall_at_${g.k}`] as number, 1)} of the time.
            The server then answers each question with the top {g.k} paragraphs in its prompt.
          </p>
          <p>
            SQuAD scores an answer by exact match and by word overlap (F1), after lowercasing and dropping punctuation and
            articles. Qwen3 answers in sentences, such as "The chloroplast double membrane is sometimes compared to the
            mitochondrial double membrane.", which is right and scores 0 on exact match. Table 7 therefore also counts whether
            an answer contains a reference answer as whole words: {pct(g.rag_contains, 0)} do. Asked without the paragraphs,
            the same questions set a floor, since SQuAD's questions presuppose their paragraph ("the event that compelled the
            protection of German private schools"): closed-book, only {pct(g.closed_contains, 0)} of answers contain the
            reference. The paragraphs cost time, since the prompt grows by three of them: an answer takes{" "}
            {num(g.rag_answer_s, 3)} s against {num(g.closed_answer_s, 2)} s. Figure 16 runs the whole pipeline on a question of
            your own.
          </p>
          <Fig n={7} table caption={<>Qwen3-0.6B on {int(g.questions)} SQuAD v1.1 questions through minitorch's server on the Apple GPU, from <code>apps/rag.py</code>. Exact match and F1 follow SQuAD's rules; "contains" asks whether a reference answer appears in the reply as whole words.</>}>
            <table className="results">
              <thead><tr><th>Prompt</th><th className="num">Exact match</th><th className="num">F1</th><th className="num">Contains</th><th className="num">Seconds per answer</th></tr></thead>
              <tbody>
                <tr><th>the question alone (a floor)</th><td className="num">{pct(g.closed_exact, 1)}</td><td className="num">{num(g.closed_f1, 2)}</td><td className="num">{pct(g.closed_contains, 0)}</td><td className="num">{num(g.closed_answer_s, 2)}</td></tr>
                <tr><th>with the top {g.k} paragraphs</th><td className="num">{pct(g.rag_exact, 1)}</td><td className="num">{num(g.rag_f1, 2)}</td><td className="num">{pct(g.rag_contains, 0)}</td><td className="num">{num(g.rag_answer_s, 3)}</td></tr>
              </tbody>
            </table>
          </Fig>
          <Fig n={16} caption={<>Retrieval, live, through minitorch's server: the question is embedded by Qwen3-Embedding-0.6B, compared with every one of SQuAD's {int(g.paragraphs)} development paragraphs, and Qwen3-0.6B answers from the three nearest. The first question loads both models, which takes a moment.</>}>
            <RagDemo />
          </Fig>
        </>
      )}
      <h3>A calculator</h3>
      <p>
        GSM8K's problems take a few steps of arithmetic, and a 0.6B model slips on arithmetic. Qwen3 was trained to call
        tools: shown a function's description in its system prompt, it can stop in the middle of a reply and write a call as
        JSON between <code>&lt;tool_call&gt;</code> tags. <code>apps/agent.py</code> gives it a calculator and runs the loop
        ReAct <Cite k="react" /> describes: when a reply calls the tool, the program evaluates the expression and sends the
        value back as a tool message, and the model continues, until it answers without a call. The expression comes from the model, so the calculator parses it and evaluates numbers
        and arithmetic only, and never runs it as code (<code>tests/agent.py</code>). minitorch writes tools and tool results
        into the prompt exactly as Hugging Face's chat template does, which <code>tests/tokenizer.py</code> checks token for
        token.
      </p>
      {r.agentStep && r.vsAgentStep && base && (
        <p>
          The first run kept §11's prompt, which says "Solve it step by step", and offered the calculator in the system
          message. In {int(r.agentStep.n)} problems the model never called it, and Hugging Face's Qwen3-0.6B ignores it the same
          way on that prompt. The offer still changed the answers. Replies shrank from {int(base.new_tokens / base.n)} tokens on
          average to {int(r.agentStep.reply_tokens / r.agentStep.n)}, and the model solved {r.agentStep.correct} problems against{" "}
          {base.correct}: it fixed {r.vsAgentStep.fixed} and broke {r.vsAgentStep.broke}, a split that chance produces{" "}
          {chance(r.vsAgentStep.p_value)} of the time. Told to solve step by step, it does, whatever the system message says. The
          second run asks for the calculator in the user's turn instead.
        </p>
      )}
      {a && base && vs ? (
        <p>
          Asked in the user's turn, the model called the calculator {int(a.tool_calls)} times, on {int(a.problems_using_tool)}{" "}
          of {a.n} problems, and solved {a.correct} against {base.correct} without it: it fixed {vs.fixed} problems and broke{" "}
          {vs.broke}, a split that chance produces {chance(vs.p_value)} of the time. A problem took{" "}
          {num(a.model_calls / a.n, 2)} calls to the model on average, and a reply averaged {int(a.reply_tokens / a.model_calls)}{" "}
          tokens, little more than the calls themselves. The model handed the calculator its arithmetic before working out
          what to compute, then answered with the number that came back (<code>bench/results/gsm8k_agent_replies.jsonl</code>{" "}
          has every trace). The request displaced the reasoning that had been getting problems right: with the plain prompt a
          reply averaged {int(base.new_tokens / base.n)} tokens of worked steps.
        </p>
      ) : (
        <p className="pending">The second calculator run is under way; its numbers appear here when it finishes.</p>
      )}
      <Fig n={17} caption={<>The agent, live: <code>apps/agent.py</code>'s loop on minitorch's server, with each model reply and each calculator call in order. It reproduces the second run's prompt, so expect the calls to come before the reasoning.</>}>
        <AgentDemo />
      </Fig>
    </Section>
  );
}
