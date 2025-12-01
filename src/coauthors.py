import re
import pandas as pd
import networkx as nx
from rapidfuzz import process, fuzz

# ------------------------
# Name handling
# ------------------------

_whitespace_re = re.compile(r"\s+")
_punct_re = re.compile(r"[.\u00B7•]")

def normalize_author_name(name: str) -> str:
    if name is None:
        return ""
    name = name.strip()
    name = _punct_re.sub("", name)
    name = _whitespace_re.sub(" ", name)
    return name.title()

def parse_authors(authors_str: str):
    if pd.isna(authors_str) or not str(authors_str).strip():
        return []
    raw = [a.strip() for a in str(authors_str).split(",")]
    raw = [a for a in raw if a]
    return [normalize_author_name(a) for a in raw]

# ------------------------
# Search authors (for UI)
# ------------------------

def search_authors(query, all_authors, limit=10, score_cutoff=60):
    q = normalize_author_name(query)
    if not q:
        return []

    results = process.extract(
        q,
        all_authors,
        scorer=fuzz.WRatio,
        limit=limit
    )
    return [(name, score) for name, score, _ in results if score >= score_cutoff]

# ------------------------
# Co-author tables
# ------------------------

def get_coauthors_and_twohop(G, author):
    """
    Direct coauthors and coauthors-of-coauthors from the full graph G.
    Returns two DataFrames with columns:
      co_df:  Coauthor, NumSharedPapers
      two_df: CoauthorOfCoauthor, NumPathsFromAuthor
    """
    if author not in G:
        return pd.DataFrame(), pd.DataFrame()

    # direct
    coauthors = []
    for nbr in G.neighbors(author):
        w = G[author][nbr].get("weight", 1)
        coauthors.append((nbr, w))
    coauthors.sort(key=lambda x: (-x[1], x[0]))
    co_df = pd.DataFrame(coauthors, columns=["Coauthor", "NumSharedPapers"])

    direct_set = set(a for a, _ in coauthors)
    direct_set.add(author)

    # 2-hop
    twohop_counts = {}
    for co in G.neighbors(author):
        for nbr2 in G.neighbors(co):
            if nbr2 in direct_set:
                continue
            twohop_counts[nbr2] = twohop_counts.get(nbr2, 0) + 1

    twohop = sorted(twohop_counts.items(), key=lambda x: (-x[1], x[0]))
    two_df = pd.DataFrame(twohop, columns=["CoauthorOfCoauthor", "NumPathsFromAuthor"])

    return co_df, two_df

# ------------------------
# Tree layout + Plotly figure
# ------------------------

import plotly.graph_objects as go

def build_hierarchical_tree(G, author):
    if author not in G:
        return nx.DiGraph()

    direct = sorted(set(G.neighbors(author)))
    twohop = set()
    for co in direct:
        twohop.update(G.neighbors(co))
    twohop.discard(author)
    twohop -= set(direct)
    twohop = sorted(twohop)

    T = nx.DiGraph()
    T.add_node(author, level=0)

    for co in direct:
        T.add_node(co, level=1)
        T.add_edge(author, co)

    for co in direct:
        for nbr2 in G.neighbors(co):
            if nbr2 in twohop:
                T.add_node(nbr2, level=2)
                T.add_edge(co, nbr2)

    return T

def hierarchical_positions(T, x_gap=1.6, y_gap=1.6):
    levels = {}
    for n, data in T.nodes(data=True):
        lvl = data.get("level", 0)
        levels.setdefault(lvl, []).append(n)

    pos = {}
    for lvl, nodes in levels.items():
        nodes = sorted(nodes)
        ys = [(i - (len(nodes)-1)/2) * y_gap for i in range(len(nodes))]
        for n, y in zip(nodes, ys):
            pos[n] = (lvl * x_gap, y)
    return pos

def plot_hierarchical_tree(T, pos, author):
    edge_x, edge_y = [], []
    for u, v in T.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#999999"),
        hoverinfo="none"
    )

    node_x, node_y, labels, colors, sizes, fs = [], [], [], [], [], []
    for n, data in T.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        labels.append(n)
        lvl = data.get("level", 0)
        if n == author:
            colors.append("red")
            sizes.append(26)
            fs.append(26)
        elif lvl == 1:
            colors.append("green")
            sizes.append(26)
            fs.append(26)
        else:
            colors.append("blue")
            sizes.append(26)
            fs.append(26)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=fs),
        hoverinfo="text",
        cliponaxis=False,
        marker=dict(
            size=sizes,
            color="black",
            line=dict(width=1, color="grey")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs) - 0.8, max(xs) + 0.8
    y_min, y_max = min(ys) - 1.2, max(ys) + 1.2

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=80, b=40),
        height=700,
        xaxis=dict(range=[x_min, x_max], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[y_min, y_max], showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig
