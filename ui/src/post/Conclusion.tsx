import { Section, int, sci, useResults } from "./kit";

export function Conclusion() {
  const r = useResults();
  return (
    <Section id="conclusion" n={14} title="Conclusion">
      <p>
        minitorch went from a single derivative to a server that answers questions from Wikipedia and calls a calculator, in{" "}
        {int(r.lines?.package)} lines of Python. Each layer matched its reference: the gradients PyTorch's to{" "}
        {sci(r.p64?.rows[0].abs_diff)}, Qwen3's logits Hugging Face's to {sci(r.qwenCpu?.max_gap)}, and every attention kernel a
        64-bit reference to within {sci(r.bkM?.kernels?.tolerance, 0)}.
      </p>
    </Section>
  );
}
