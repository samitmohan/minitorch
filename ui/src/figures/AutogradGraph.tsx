import { Graph as Dag, layout } from "@dagrejs/dagre";
import { Background, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo, useState } from "react";
import { api, type Graph, type GraphNode } from "../api";
import { formatNumber, parseVariables } from "../lib";

const NODE_W = 150;
const NODE_H = 70;

export function AutogradGraph() {
  const [expr, setExpr] = useState("relu(x * w + b) ** 2");
  const [vars, setVars] = useState("x=2, w=-0.5, b=3");
  const [graph, setGraph] = useState<Graph | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = () => {
    try {
      api.graph(expr, parseVariables(vars)).then(
        (g) => {
          setGraph(g);
          setError(null);
        },
        (e) => setError(e.message),
      );
    } catch (e) {
      setError((e as Error).message);
    }
  };
  useEffect(run, []);

  const flow = useMemo(() => (graph ? toFlow(graph) : null), [graph]);

  return (
    <section aria-label="Autograd graph">
      <form className="row" onSubmit={(e) => {
        e.preventDefault();
        run();
      }}>
        <label>
          Expression
          <input className="code" size={32} value={expr} onChange={(e) => setExpr(e.target.value)} />
        </label>
        <label>
          Variables
          <input className="code" size={24} value={vars} onChange={(e) => setVars(e.target.value)} />
        </label>
        <button type="submit">Run backward</button>
        <div className="legend" aria-hidden>
          <span className="v">value</span>
          <span className="g">gradient</span>
        </div>
      </form>
      <p className="muted">Functions: exp, log, relu, sigmoid, tanh, sum, mean. Operators: + - * / ** @.</p>
      {error && <p className="error">{error}</p>}
      {flow && (
        <div className="dag">
          <ReactFlow nodes={flow.nodes} edges={flow.edges} nodeTypes={nodeTypes} fitView nodesConnectable={false}>
            <Background gap={16} color="#c9d1ca" />
          </ReactFlow>
        </div>
      )}
    </section>
  );
}

type OpData = GraphNode & { isRoot: boolean };

function OpNode({ data }: NodeProps<Node<OpData>>) {
  const show = (v: GraphNode["value"]) => {
    if (v === null) return "…";
    if (Array.isArray(v)) return `[${v.map((x) => formatNumber(x, 3)).join(", ")}]`;
    return formatNumber(v);
  };
  return (
    <div className={`op-node${data.leaf ? " leaf" : ""}${data.isRoot ? " root" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <strong>{data.label}</strong>
      <div className="v">{show(data.value)}</div>
      {data.requires_grad && <div className="g">∂ {show(data.grad)}</div>}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
const nodeTypes = { op: OpNode };

function toFlow(g: Graph): { nodes: Node<OpData>[]; edges: Edge[] } {
  const dag = new Dag();
  dag.setGraph({ rankdir: "LR", nodesep: 24, ranksep: 64 });
  dag.setDefaultEdgeLabel(() => ({}));
  g.nodes.forEach((n) => dag.setNode(String(n.id), { width: NODE_W, height: NODE_H }));
  g.edges.forEach((e) => dag.setEdge(String(e.source), String(e.target)));
  layout(dag);
  return {
    nodes: g.nodes.map((n) => {
      const { x, y } = dag.node(String(n.id));
      return { id: String(n.id), type: "op", position: { x: x - NODE_W / 2, y: y - NODE_H / 2 }, data: { ...n, isRoot: n.id === g.root } };
    }),
    edges: g.edges.map((e) => ({ id: `${e.source}-${e.target}`, source: String(e.source), target: String(e.target), style: { stroke: "#1b2330" } })),
  };
}

