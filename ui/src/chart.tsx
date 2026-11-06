// Chart pieces shared by the pages.

export const INK = "#1b2330";
export const RULE = "#c9d1ca";

// Device identity on the Bench page: categorical slots 1-3, checked with the dataviz
// validator against the figure background (#fafbfc). Aqua sits under 3:1 contrast, so every chart
// that uses these also labels its lines and offers a table.
export const DEVICE_COLOR: Record<string, string> = { cpu: "#2a78d6", metal: "#eb6834", cuda: "#1baf7a" };
// which machine each device ran on is in the result files; the Bench page prints it
export const DEVICE_NAME: Record<string, string> = { cpu: "CPU (NumPy)", metal: "Apple GPU (MLX)", cuda: "NVIDIA GPU (CuPy)" };
export const DEVICE_SHORT: Record<string, string> = { cpu: "NumPy", metal: "Apple GPU", cuda: "NVIDIA" };

/** Label only the last point of a series, so the line names itself next to the legend. */
export function endLabel(count: number, text: string, color: string, dy = 0) {
  return (props: { index?: number; x?: number | string; y?: number | string }) =>
    props.index === count - 1 ? (
      <text x={Number(props.x) + 8} y={Number(props.y) + dy} fill={color} fontSize={13} fontWeight={700}>
        {text}
      </text>
    ) : null;
}
