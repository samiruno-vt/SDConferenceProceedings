import pandas as pd
import numpy as np
import re
import networkx as nx
from itertools import combinations
from rapidfuzz import process, fuzz
import plotly.graph_objects as go



# functions to work with author names

def normalize_author_name(name: str) -> str:
    """
    Normalize an author name for matching / graph nodes.
    - trims
    - collapses whitespace
    - removes periods/bullets
    - title-cases (optional, but nice for display)
    """
    if name is None:
        return ""
    name = name.strip()
    # Keep hyphens/apostrophes; title-case for consistency
    name = name.title()
    return name

def parse_authors(authors_str: str):
    """
    authors_str: comma-separated list of 'First Last' names.
    Returns list of normalized author names.
    """
    if pd.isna(authors_str) or not str(authors_str).strip():
        return []
    raw = [a.strip() for a in str(authors_str).split(",")]
    raw = [a for a in raw if a]  # drop empties
    normed = [normalize_author_name(a) for a in raw]
    return normed



# build co-author graph

def build_coauthor_graph(df, authors_col = "Authors"):
    """
    Undirected weighted graph:
      node = normalized author name
      edge weight = # of shared papers
    """
    G = nx.Graph()

    for authors_str in df[authors_col]:
        authors = parse_authors(authors_str)
        if not authors:
            continue

        # add nodes
        for a in authors:
            G.add_node(a)

        # add weighted edges per paper
        uniq = sorted(set(authors))
        for a, b in combinations(uniq, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    return G



# search for and confirm author

def search_authors(query, all_authors, limit = 10, score_cutoff = 60):
    """
    Returns list of (author, score) candidates for confirmation UI.
    """
    query = normalize_author_name(query)
    if not query:
        return []

    results = process.extract(
        query,
        all_authors,
        scorer = fuzz.WRatio,
        limit = limit
    )
    return [(name, score) for name, score, _ in results if score >= score_cutoff]



# direct co-authors and co-authors of co-authors

def get_coauthors_and_twohop(G, author):
    """
    author is expected to be normalized (from confirmation step).
    Returns two DataFrames.
    """
    if author not in G:
        return pd.DataFrame(), pd.DataFrame()

    # 1-hop
    coauthors = []
    for nbr in G.neighbors(author):
        w = G[author][nbr].get("weight", 1)
        coauthors.append((nbr, w))
    coauthors.sort(key=lambda x: (-x[1], x[0]))
    co_df = pd.DataFrame(coauthors, columns=["Coauthor", "NumSharedPapers"])

    direct_set = set([a for a, _ in coauthors])
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


def format_coauthor_tables(co_df, two_df, top_n=30):
    # Direct coauthors
    co_df = co_df.copy()
    co_df = co_df.rename(columns={
        "Coauthor": "Direct co-author",
        "NumSharedPapers": "Shared papers"
    })
    co_df = co_df[["Direct co-author", "Shared papers"]]
    co_df = co_df.head(top_n)

    # 2-hop coauthors
    two_df = two_df.copy()
    two_df = two_df.rename(columns={
        "CoauthorOfCoauthor": "Connection",
        "NumPathsFromAuthor": "Number of links"
    })
    two_df["Rank"] = range(1, len(two_df) + 1)
    two_df = two_df[["Rank", "Connection", "Number of links"]]
    two_df = two_df.head(top_n)

    return co_df, two_df


def show_coauthor_report(author, df, G, co_df, two_df, top_n=25):
    """
    author: confirmed normalized author name
    df: your full papers dataframe (needs Authors + Year)
    G: coauthor graph
    co_df, two_df: outputs from get_coauthors_and_twohop()
    """

    # --- summary stats ---
    # papers the author appears on
    mask = df["Authors"].fillna("").apply(lambda s: author in [a.strip().title() for a in str(s).split(",")])
    author_papers = df[mask]
    total_papers = len(author_papers)

    num_direct = len(co_df)
    num_twohop = len(two_df)

    # --- format direct coauthors table ---
    co_tbl = (co_df.rename(columns={
                    "Coauthor": "Direct co-author",
                    "NumSharedPapers": "Shared papers"
                })
                .sort_values("Shared papers", ascending=False)
                .head(top_n)
             )

    # add their total papers in proceedings (optional but nice)
    if "Authors" in df.columns:
        counts = {}
        for s in df["Authors"].fillna(""):
            for a in [x.strip().title() for x in str(s).split(",") if x.strip()]:
                counts[a] = counts.get(a, 0) + 1
        co_tbl["Their total papers"] = co_tbl["Direct co-author"].map(lambda a: counts.get(a, 0))

    # --- format 2-hop table ---
    two_tbl = (two_df.rename(columns={
                        "CoauthorOfCoauthor": "Co-author of co-authors",
                        "NumPathsFromAuthor": "2-hop links"
                    })
                    .sort_values("2-hop links", ascending=False)
                    .head(top_n)
               )

    # add a "Via" column (intermediaries between the others)
    two_tbl["Via"] = two_tbl["Co-author of co-authors"].apply(
    lambda two_name: ", ".join(sorted([
        co for co in G.neighbors(author) if G.has_edge(co, two_name)
    ]))
    )

    if "Authors" in df.columns:
        two_tbl["Their total papers"] = two_tbl["Co-author of co-authors"].map(lambda a: counts.get(a, 0))

    # --- display nicely in Colab ---
    display(Markdown(f"## Co-author network for **{author}**"))
    display(Markdown(
        f"- **Total papers:** {total_papers}\n"
        f"- **Direct co-authors:** {num_direct}\n"
        f"- **Second-degree connections:** {num_twohop}\n"
    ))

    display(Markdown("### Direct co-authors"))
    display(
        co_tbl.style
            .hide(axis="index")
            .set_properties(**{"font-size": "14px", "text-align": "left"})
            .set_table_styles([
                {"selector":"th", "props":[("background-color","#f2f2f2"),
                                          ("font-weight","600"),
                                          ("text-align","left")]},
                {"selector":"td", "props":[("padding","6px 10px")]}
            ])
    )

    display(Markdown("### Co-authors of co-authors (2-hop)"))
    display(
        two_tbl.style
            .hide(axis="index")
            .set_properties(**{"font-size": "14px", "text-align": "left"})
            .set_table_styles([
                {"selector":"th", "props":[("background-color","#f2f2f2"),
                                          ("font-weight","600"),
                                          ("text-align","left")]},
                {"selector":"td", "props":[("padding","6px 10px")]}
            ])
    )

    return co_tbl, two_tbl


# network visualization of co-authors

def build_hierarchical_tree(G, author):
    """
    Returns a directed tree-like graph:
      level 0: author
      level 1: direct coauthors
      level 2: coauthors-of-coauthors (excluding level 0/1)
    """
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


def hierarchical_positions(T, x_gap = 1.0, y_gap = 1.0):
    """
    Assign positions:
      x = level * x_gap
      y = evenly spaced within each level
    """
    levels = {}
    for n, data in T.nodes(data = True):
        lvl = data.get("level", 0)
        levels.setdefault(lvl, []).append(n)

    pos = {}
    for lvl, nodes in levels.items():
        nodes = sorted(nodes)
        # center them vertically around y = 0
        ys = [(i - (len(nodes)-1)/2) * y_gap for i in range(len(nodes))]
        for n, y in zip(nodes, ys):
            pos[n] = (lvl * x_gap, y)

    return pos


def compute_height(T, base = 400, per_node = 35):
    # count nodes per level
    levels = {}
    for n, d in T.nodes(data = True):
        lvl = d.get("level", 0)
        levels[lvl] = levels.get(lvl, 0) + 1
    max_level_size = max(levels.values()) if levels else 1
    return base + per_node * max_level_size


def plot_hierarchical_tree(T, pos, author):
    # edges
    edge_x, edge_y = [], []
    for u, v in T.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x = edge_x, y = edge_y,
        mode = "lines",
        line = dict(width = 1, color = 'lightgrey'),
        hoverinfo = "none"
    )

    # nodes by level for styling
    node_x, node_y, labels, colors, sizes = [], [], [], [], []
    for n, data in T.nodes(data = True):
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        labels.append(n)

        lvl = data.get("level", 0)
        if n == author:
            colors.append("red")  # selected author
            sizes.append(22)
        elif lvl == 1:
            colors.append("green")  # direct coauthors
            sizes.append(22)
        else:
            colors.append("blue")  # two-hop
            sizes.append(22)

    node_trace = go.Scatter(
        x = node_x, y = node_y,
        mode = "markers+text",
        text = labels,
        textposition = "top center",
        hoverinfo = "text",
        marker = dict(size = sizes, color = colors, line = dict(width = 1)),
        textfont = dict(size = 16),
        cliponaxis = False
    )

    fig = go.Figure(data = [edge_trace, node_trace])
    fig.update_layout(
        showlegend = False,
        margin = dict(l = 20, r = 20, t = 20, b = 20),
        xaxis = dict(showgrid = False, zeroline = False, showticklabels = False),
        yaxis = dict(showgrid = False, zeroline = False, showticklabels = False),
        plot_bgcolor = "white"
    )
    return fig
