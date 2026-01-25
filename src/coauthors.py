import re
import pandas as pd
import networkx as nx
from rapidfuzz import process, fuzz

# ------------------------
# Name handling
# ------------------------

_whitespace_re = re.compile(r"\s+")
_punct_re = re.compile(r"[.\u00B7•]")
_quotes_re = re.compile(r"[\"'""''`]")

def normalize_author_name(name: str) -> str:
    if name is None:
        return ""
    name = name.strip()
    name = _punct_re.sub("", name)
    name = _quotes_re.sub("", name)  # Remove all types of quotes
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

def search_authors(query, all_authors, limit=10, score_cutoff=80):
    """
    Search for authors matching the query.
    Prioritizes exact substring matches, then falls back to fuzzy matching.
    """
    q = normalize_author_name(query)
    if not q:
        return []
    
    q_lower = q.lower()
    
    # First, find exact substring matches (case-insensitive)
    exact_matches = []
    for name in all_authors:
        if q_lower in name.lower():
            # Score based on how much of the name the query covers
            score = len(q) / len(name) * 100
            exact_matches.append((name, min(100, score + 50)))  # Boost exact matches
    
    # Sort by score descending, then alphabetically
    exact_matches.sort(key=lambda x: (-x[1], x[0]))
    
    # If we have enough exact matches, return those only
    if len(exact_matches) >= limit:
        return exact_matches[:limit]
    
    # If we have some exact matches, just return those (don't add fuzzy noise)
    if len(exact_matches) > 0:
        return exact_matches[:limit]
    
    # No exact matches - fall back to fuzzy matching
    fuzzy_results = process.extract(
        q,
        all_authors,
        scorer=fuzz.WRatio,
        limit=limit
    )
    return [(name, score) for name, score, _ in fuzzy_results if score >= score_cutoff]

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
    
    # Now add ALL edges from G between nodes in H, copying edge weights
    for node in H.nodes():
        for nbr in G.neighbors(node):
            if nbr in H and not H.has_edge(node, nbr):
                # Copy edge weight (number of shared papers) from original graph
                weight = G[node][nbr].get("weight", 1)
                H.add_edge(node, nbr, weight=weight)
    
    return H


def plot_coauthor_network(H, author):
    """
    Plot the co-author network with improved layout and visual encoding.
    
    Improvements:
    - Strong repulsion layout to prevent overlap
    - Visible edge thickness indicating collaboration strength
    - Dramatic node sizing differences
    - Intuitive color scheme by degree
    """
    if H.number_of_nodes() == 0:
        return None
    
    n_nodes = H.number_of_nodes()
    
    # Use spring layout with STRONG repulsion to spread nodes apart
    pos = nx.spring_layout(
        H, 
        seed=42, 
        k=6/np.sqrt(n_nodes) if n_nodes > 1 else 1,  # High k = strong repulsion
        iterations=300,  # Many iterations for convergence
        scale=3
    )
    
    # Get edge weights for thickness scaling
    edge_weights = [H[u][v].get("weight", 1) for u, v in H.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    min_weight = min(edge_weights) if edge_weights else 1
    
    # Group edges by weight for efficient rendering with visible thickness
    edges_by_weight = {}
    
    for u, v in H.edges():
        weight = H[u][v].get("weight", 1)
        if weight not in edges_by_weight:
            edges_by_weight[weight] = {"x": [], "y": [], "pairs": []}
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edges_by_weight[weight]["x"].extend([x0, x1, None])
        edges_by_weight[weight]["y"].extend([y0, y1, None])
        edges_by_weight[weight]["pairs"].append((u, v))
    
    edge_traces = []
    for weight, data in sorted(edges_by_weight.items()):
        if max_weight > min_weight:
            normalized = (weight - min_weight) / (max_weight - min_weight)
        else:
            normalized = 0.5
        
        # Edge width: 1px to 8px - much more visible
        line_width = 1.5 + normalized * 6.5
        
        # Color: darker gray for stronger connections
        gray_val = int(170 - normalized * 90)  # 170 (light) to 80 (dark)
        
        edge_traces.append(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="lines",
            line=dict(width=line_width, color=f"rgb({gray_val},{gray_val},{gray_val})"),
            hoverinfo="skip",
            showlegend=False
        ))

    # Color scheme by degree level - more vibrant and distinct
    level_colors = {
        0: "#d62828",  # Vivid red for selected author
        1: "#2a9d8f",  # Teal for direct co-authors  
        2: "#457b9d",  # Steel blue for 2nd degree
        3: "#8338ec",  # Purple for 3rd degree
        4: "#6c757d",  # Gray for 4th degree
    }

    # Calculate node connectivity for size bonus
    node_degrees = dict(H.degree())
    max_deg = max(node_degrees.values()) if node_degrees else 1
    
    node_x, node_y, node_text, node_colors, node_sizes = [], [], [], [], []
    node_names = []
    
    for n in H.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_names.append(n)
        
        lvl = H.nodes[n].get("level", 0)
        node_colors.append(level_colors.get(lvl, "#6c757d"))
        
        # Much more dramatic size differences
        # Central author is huge, then big drop-off
        deg = node_degrees.get(n, 1)
        
        if lvl == 0:
            # Selected author - very large
            base = 60
        elif lvl == 1:
            # Direct co-authors - large
            base = 35
        elif lvl == 2:
            # 2nd degree - medium
            base = 22
        else:
            # 3rd+ degree - smaller
            base = 16
        
        # Add connectivity bonus (logarithmic)
        if max_deg > 1:
            deg_bonus = np.log1p(deg) / np.log1p(max_deg) * 12
        else:
            deg_bonus = 0
        
        node_sizes.append(base + deg_bonus)
        
        # Rich hover text
        connections = deg
        node_text.append(f"<b>{n}</b><br>Degree from center: {lvl}<br>Connections shown: {connections}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_names,
        textposition="top center",
        textfont=dict(size=9, color="#333333"),
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color="white"),
            opacity=0.9,
            sizemode="diameter"
        ),
        showlegend=False
    )

    # Edges first (behind), then nodes on top
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#f8f9fa",
        margin=dict(l=20, r=20, t=20, b=20),
        height=700,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        dragmode="pan",
        hovermode="closest"
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
