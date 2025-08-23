__all__ = ["draw_g"]
import matplotlib.pyplot as plt
import networkx as nx
#TODO: add node weights and names to draw function


def _as_nx_graph(obj):
    """Best-effort conversion of various pipeline graph containers to nx.Graph.
    Accepts: TimeGraph-like (get_graph/_get_graph), nx.Graph, dict of graphs,
    list/tuple of graphs. Composes multiple graphs when needed.
    """
    # Unwrap common wrappers
    if hasattr(obj, "get_graph"):
        try:
            g = obj.get_graph()
            if isinstance(g, nx.Graph):
                return g
        except Exception:
            pass
    if hasattr(obj, "_get_graph"):
        try:
            g = obj._get_graph()
            if isinstance(g, nx.Graph):
                return g
        except Exception:
            pass

    # Already an nx.Graph
    if isinstance(obj, nx.Graph):
        return obj

    # Dict of graphs (e.g., {id: TimeGraph/nx.Graph})
    if isinstance(obj, dict):
        graphs = []
        for v in obj.values():
            gv = _as_nx_graph(v)
            if gv is not None:
                graphs.append(gv)
        if not graphs:
            raise TypeError("No graphs found inside dict to compose.")
        return nx.compose_all(graphs)

    # List/tuple of graphs
    if isinstance(obj, (list, tuple)):
        graphs = []
        for v in obj:
            gv = _as_nx_graph(v)
            if gv is not None:
                graphs.append(gv)
        if not graphs:
            raise TypeError("No graphs found inside sequence to compose.")
        return nx.compose_all(graphs)

    raise TypeError(f"Unsupported graph input type: {type(obj)}")


def draw_g(timegraph):
    """Draw a graph from various input containers into 'example.png'."""
    try:
        G = _as_nx_graph(timegraph)
    except Exception as e:
        raise TypeError(f"draw_g expected a NetworkX graph or TimeGraph-like object; got error: {e}")

    pos = nx.spring_layout(G, seed=1)

    edge_colors = [G[u][v].get('color', 'gray') for u, v in G.edges()]
    node_colors = [d.get('color', 'gray') for _, d in G.nodes(data=True)]

    # Edge widths by (normalized) weight
    weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
    if weights:
        wmax, wmin = max(weights), min(weights)
        if wmax == wmin:
            norm_w = [1 for _ in weights]
        else:
            norm_w = [1 + 4 * (w - wmin) / (wmax - wmin) for w in weights]
    else:
        norm_w = []

    nx.draw(G, pos, node_size=100, node_color=node_colors, with_labels=False,
            edge_color=edge_colors, width=norm_w)
    plt.tight_layout()
    #plt.savefig('example.png')

