def draw_graph(root, format='svg'):
    """Render the autograd graph as a graphviz Digraph."""
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("Install graphviz: uv pip install graphviz")

    dot = Digraph(format=format, graph_attr={'rankdir': 'LR'})

    visited = set()

    def visit(v):
        if id(v) in visited:
            return
        visited.add(id(v))

        uid = str(id(v))

        # tensor node: show shape
        if v.data.ndim == 0:
            label = f'{float(v.data):.3g}'
        else:
            label = str(tuple(v.shape))

        color = '#d4edda' if v.requires_grad else '#e2e3e5'
        dot.node(uid, label, shape='ellipse', style='filled', fillcolor=color)

        if v._op:
            # operation node
            op_uid = uid + '_op'
            dot.node(op_uid, v._op, shape='box', style='filled', fillcolor='#cce5ff')
            dot.edge(op_uid, uid)

            for child in v._prev:
                visit(child)
                dot.edge(str(id(child)), op_uid)

    visit(root)
    return dot
