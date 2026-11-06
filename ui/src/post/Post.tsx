import { useEffect, useState } from "react";
import { api, streamMetrics, type Run } from "../api";
import type { MetricRow } from "../lib";
import type { GrpoRow } from "./kit";
import { Appendix } from "./Appendix";
import { Conclusion } from "./Conclusion";
import { Autograd, Intro } from "./Foundations";
import { Head, Teaser, Tldr } from "./Head";
import { ResultsContext } from "./kit";
import { Model, Tokens, Training } from "./Language";
import { Grpo, Lora, Quantization, Qwen } from "./RealModel";
import { Attention, Gpu, Serving } from "./Systems";
import { Tools } from "./Using";

/** The whole site: one post. It reads the results once and every section draws from them. */
export default function Post() {
  const [bench, setBench] = useState<Record<string, unknown> | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [ts, setTs] = useState<MetricRow[]>([]);
  const [lora, setLora] = useState<MetricRow[]>([]);
  const [grpo, setGrpo] = useState<GrpoRow[]>([]);
  const [grpoNoStop, setGrpoNoStop] = useState<GrpoRow[]>([]);
  const [sample, setSample] = useState<string | undefined>();

  useEffect(() => {
    api.bench().then(setBench, () => setBench({}));
    api.runs().then(setRuns, () => {});
    api.samples("tinystories").then((s) => setSample(s.at(-1)?.text), () => {});
    const stopTs = streamMetrics("tinystories", (batch) => setTs((prev) => prev.concat(batch)));
    const stopLora = streamMetrics("lora_gsm8k", (batch) => setLora((prev) => prev.concat(batch)));
    const stopGrpo = streamMetrics("grpo_gsm8k", (batch) => setGrpo((prev) => prev.concat(batch as unknown as GrpoRow[])));
    const stopNoStop = streamMetrics("grpo_no_stop", (batch) => setGrpoNoStop((prev) => prev.concat(batch as unknown as GrpoRow[])));
    return () => {
      stopTs();
      stopLora();
      stopGrpo();
      stopNoStop();
    };
  }, []);

  if (bench === null) return <p className="muted loading">Loading the results…</p>;
  return (
    <ResultsContext.Provider value={{ bench, runs, ts, lora, grpo, grpoNoStop, sample }}>
      <article className="paper">
        <Head />
        <div className="col body">
          <Tldr />
          <Teaser />
          <Intro />
          <Autograd />
          <Tokens />
          <Model />
          <Training />
          <Gpu />
          <Attention />
          <Qwen />
          <Serving />
          <Quantization />
          <Lora />
          <Grpo />
          <Tools />
          <Conclusion />
          <Appendix />
        </div>
      </article>
    </ResultsContext.Provider>
  );
}
