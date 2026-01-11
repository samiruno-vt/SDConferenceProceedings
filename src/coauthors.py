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

def get_coauthors_by_degree(G, author, max_degree=2):
    """
    Get co-authors up to max_degree hops from author.
    Returns a list of DataFrames, one per degree level.
    
    Each DataFrame has columns appropriate to the degree:
      - Degree 1: Coauthor, NumSharedPapers
      - Degree 2+: Author, NumPaths (number of shortest paths from root author)
    """
    if author not in G:
        return []
    
    results = []
    visited = {author}
    current_level = {author}
    
    for degree in range(1, max_degree + 1):
        next_level = set()
        degree_data = []
        
        for node in current_level:
            for nbr in G.neighbors(node):
                if nbr not in visited:
                    next_level.add(nbr)
        
        # Count paths to each node at this degree
        path_counts = {}
        for node in next_level:
            # Count how many nodes from previous level connect to this node
            count = sum(1 for prev in current_level if G.has_edge(prev, node))
            path_counts[node] = count
        
        # For degree 1, include edge weight (shared papers)
        if degree == 1:
            for nbr in sorted(next_level):
                w = G[author][nbr].get("weight", 1)
                degree_data.append((nbr, w))
            degree_data.sort(key=lambda x: (-x[1], x[0]))
            df = pd.DataFrame(degree_data, columns=["Co-author", "Shared Papers"])
        else:
            for nbr in sorted(next_level):
                degree_data.append((nbr, path_counts[nbr]))
            degree_data.sort(key=lambda x: (-x[1], x[0]))
            df = pd.DataFrame(degree_data, columns=["Author", "Connections"])
        
        results.append(df)
        visited.update(next_level)
        current_level = next_level
    
    return results


def get_coauthors_and_twohop(G, author):
    """
    Direct coauthors and coauthors-of-coauthors from the full graph G.
    Returns two DataFrames with columns:
      co_df:  Coauthor, NumSharedPapers
      two_df: CoauthorOfCoauthor, NumPathsFromAuthor
    
    DEPRECATED: Use get_coauthors_by_degree instead.
    Kept for backwards compatibility.
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
# Network layout + Plotly figure
# ------------------------

import numpy as np
import plotly.graph_objects as go

def build_coauthor_network(G, author, max_degree=2):
    """
    Build a network of co-authors up to max_degree levels.
    Returns a graph with 'level' attribute on each node.
    Includes ALL edges between authors in the subgraph, not just BFS edges.
    """
    if author not in G:
        return nx.Graph()

    H = nx.Graph()
    H.add_node(author, level=0)
    
    visited = {author}
    current_level_nodes = {author}
    
    # First, collect all nodes up to max_degree using BFS
    for degree in range(1, max_degree + 1):
        next_level_nodes = set()
        
        for node in current_level_nodes:
            for nbr in G.neighbors(node):
                if nbr not in visited:
                    next_level_nodes.add(nbr)
                    H.add_node(nbr, level=degree)
        
        visited.update(next_level_nodes)
        current_level_nodes = next_level_nodes
        
        if not next_level_nodes:
            break
    
    # Now add ALL edges from G between nodes in H
    for node in H.nodes():
        for nbr in G.neighbors(node):
            if nbr in H and not H.has_edge(node, nbr):
                H.add_edge(node, nbr)
    
    return H


def plot_coauthor_network(H, author):
    """
    Plot the co-author network with force-directed layout.
    Nodes are colored by degree level.
    """
    if H.number_of_nodes() == 0:
        return None
    
    # Force-directed layout
    pos = nx.spring_layout(H, seed=42, k=1.5/np.sqrt(H.number_of_nodes()) if H.number_of_nodes() > 1 else 1)
    
    # Draw edges
    edge_x, edge_y = [], []
    for u, v in H.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="#cccccc"),
        hoverinfo="none"
    )

    # Colors for different levels
    level_colors = {0: "red", 1: "green", 2: "blue", 3: "orange", 4: "purple"}

    node_x, node_y, node_text, node_colors, node_sizes = [], [], [], [], []
    for n in H.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        
        lvl = H.nodes[n].get("level", 0)
        node_colors.append(level_colors.get(lvl, "gray"))
        
        # Size: larger for central author, smaller for outer degrees
        if lvl == 0:
            node_sizes.append(30)
        elif lvl == 1:
            node_sizes.append(22)
        elif lvl == 2:
            node_sizes.append(16)
        else:
            node_sizes.append(12)
        
        node_text.append(f"{n}<br>Degree: {lvl}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n for n in H.nodes()],
        textposition="top center",
        textfont=dict(size=11, color="black"),
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="white")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
        height=700,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    
    return fig


# Keep old functions for backwards compatibility
def build_hierarchical_tree(G, author, max_degree=2):
    """
    DEPRECATED: Use build_coauthor_network instead.
    """
    return build_coauthor_network(G, author, max_degree)


def hierarchical_positions(T, x_gap=1.6, y_gap=1.6):
    """
    DEPRECATED: No longer needed with network layout.
    """
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
    """
    DEPRECATED: Use plot_coauthor_network instead.
    """
    return plot_coauthor_network(T, author)
