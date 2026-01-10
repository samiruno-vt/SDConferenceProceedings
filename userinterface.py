import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go


# Make sure import from src/ works
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import similarpapers as papers  # src/similarpapers.py
import coauthors  # src/coauthors.py


# ------------------------
# Cached loaders for data
# ------------------------

@st.cache_data
def load_dataframe():
    return pd.read_parquet(os.path.join("data", "papers_clean_from05_withThreads.parquet"))

@st.cache_resource
def load_embeddings():
    return np.load(os.path.join("data", "paper_embeddings_from05.npy"))

@st.cache_resource
def load_graph():
    with open(os.path.join("data", "coauthor_graph_from05_withNodesCountryOrg.pkl"), "rb") as f:
        G = pickle.load(f)
    return G

@st.cache_data
def load_author_stats():
    return pd.read_parquet(os.path.join("data", "author_stats_from05_withCountryOrg.parquet"))


df = load_dataframe()
embeddings = load_embeddings()
G = load_graph()
author_stats = load_author_stats()


st.sidebar.write(f"Papers: {len(df)}")
st.sidebar.write(f"Years: {int(df['Year'].min())}–{int(df['Year'].max())}")
st.sidebar.write(f"Authors: {G.number_of_nodes()}")



# -------------------
# Sidebar navigation
# -------------------

st.set_page_config(page_title="System Dynamics Paper Explorer", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Find Similar Papers", "Find Co-authors", "Network Overview"]
)


# -----------------------
# Page 1: Similar papers
# -----------------------

if page == "Find Similar Papers":
    st.header("Find Similar Papers")

    st.markdown(
        "Search by **title** or **author name**, then pick a paper to see similar ones "
        "based on titles and abstracts."
    )

    query = st.text_input("Search for a paper (title or author)")

    if query:
        # search_papers should return a DataFrame with columns: row_idx, Title, Year, Authors
        candidates = papers.search_papers(query, df, search_in=("Title", "Authors"), limit=20)

        if candidates.empty:
            st.info("No papers found for that search.")
        else:
            st.subheader("Matching papers")

            # Allow user to pick one paper from candidates
            options = candidates.index.tolist()

            def format_paper_option(i):
                row = candidates.loc[i]
                year = int(row["Year"]) if not pd.isna(row["Year"]) else "NA"
                return f"{row['Title']} ({year})"

            selected_idx = st.selectbox(
                "Select a paper:",
                options=options,
                format_func=format_paper_option,
            )

            # Get the underlying row index into df
            paper_row_idx = int(candidates.loc[selected_idx, "row_idx"])

            # Show selected paper details
            selected_paper = df.loc[paper_row_idx]
            st.markdown("**Selected paper**")
            st.markdown(f"**Title:** {selected_paper['Title']}")
            st.markdown(f"**Year:** {selected_paper['Year']}")
            st.markdown(f"**Authors:** {selected_paper['Authors']}")
            if "Abstract" in df.columns:
                st.markdown("**Abstract:**")
                st.write(selected_paper["Abstract"])

            # Get similar papers
            k = st.slider("Number of similar papers to show", 3, 20, 10)
            sim_tbl = papers.get_similar_papers(
                paper_idx=paper_row_idx,
                embeddings=embeddings,
                df=df,
                k=k
            )

            st.subheader("Similar papers")
            if sim_tbl.empty:
                st.info("No similar papers found.")
            else:
                st.dataframe(sim_tbl, use_container_width=True)


# ---------------------------
# Page 2: Co-author explorer
# ---------------------------

elif page == "Find Co-authors":
    st.header("Co-authors of co-authors")

    author_query = st.text_input("Search for an author")

    if author_query:
        all_authors = sorted(G.nodes())
        candidates = coauthors.search_authors(author_query, all_authors, limit=20, score_cutoff=60)

        if not candidates:
            st.info("No matching authors found.")
        else:
            author_names = [name for name, score in candidates]
            selected_author = st.selectbox("Select an author:", options=author_names)

            author = selected_author
            st.markdown(f"**Selected author:** {author}")

            co_df, two_df = coauthors.get_coauthors_and_twohop(G, author)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Direct co-authors")
                if co_df.empty:
                    st.write("No direct co-authors found.")
                else:
                    st.dataframe(co_df, use_container_width=True)

            with col2:
                st.subheader("Co-authors of co-authors")
                if two_df.empty:
                    st.write("No 2nd-degree co-authors found.")
                else:
                    st.dataframe(two_df, use_container_width=True)

            T = coauthors.build_hierarchical_tree(G, author)
            if T.number_of_nodes() == 0:
                st.info("No co-author network to display for this author.")
            else:
                pos = coauthors.hierarchical_positions(T)
                fig = coauthors.plot_hierarchical_tree(T, pos, author)
                st.subheader("Co-author network")
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Page 3: Network Overview
# ---------------------------

elif page == "Network Overview":
    st.header("Network Overview")

    # --- Filters section ---
    st.subheader("Filters")
    
    col_year, col_papers, col_coauthors = st.columns(3)
    
    with col_year:
        year_min, year_max = st.slider(
            "Year range",
            min_value=int(df["Year"].min()),
            max_value=int(df["Year"].max()),
            value=(int(df["Year"].min()), int(df["Year"].max())),
        )
    
    with col_papers:
        min_papers = st.number_input(
            "Min papers (in selected years)",
            min_value=1,
            max_value=50,
            value=1,
            step=1,
            help="Only show authors with at least this many papers in the selected year range"
        )
    
    with col_coauthors:
        min_coauthors = st.number_input(
            "Min co-authors (overall)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            help="Only show authors with at least this many total co-authors"
        )

    # Thread filter (uses Category column)
    all_threads = sorted([t for t in df["Category"].dropna().unique() if t])
    selected_threads = st.multiselect(
        "Thread (leave empty for all)",
        options=all_threads,
        default=[],
        help="Filter to authors who have papers in these threads. Leave empty to include all papers."
    )
    
    include_no_thread = st.checkbox(
        "Include papers without a thread",
        value=True,
        help="Include papers that don't have a thread assigned"
    )

    # Apply year filter first
    df_filtered = df[df["Year"].between(year_min, year_max)].copy()
    
    # Apply thread filter if any threads are selected
    if selected_threads:
        if include_no_thread:
            df_filtered = df_filtered[
                df_filtered["Category"].isin(selected_threads) | df_filtered["Category"].isna()
            ]
        else:
            df_filtered = df_filtered[df_filtered["Category"].isin(selected_threads)]
    elif not include_no_thread:
        # No threads selected but excluding papers without thread
        df_filtered = df_filtered[df_filtered["Category"].notna()]

    # explode authors only for filtering
    ap = df_filtered[["Authors"]].copy()
    ap["Author"] = ap["Authors"].apply(coauthors.parse_authors)
    ap = ap.explode("Author")
    ap["Author"] = ap["Author"].astype(str).str.strip()
    ap = ap[ap["Author"] != ""]

    author_counts = (
        ap.groupby("Author")
          .size()
          .rename("NumPapers_Filtered")
          .reset_index()
    )

    tbl = author_stats.merge(
        author_counts,
        on="Author",
        how="left"
    )

    tbl["NumPapers_Filtered"] = tbl["NumPapers_Filtered"].fillna(0).astype(int)

    # Apply the min papers and min coauthors filters
    tbl = tbl[
        (tbl["NumPapers_Filtered"] >= min_papers) &
        (tbl["NumCoauthors"] >= min_coauthors)
    ]

    st.divider()
    
    # --- Author table section ---
    st.subheader("Top Authors")
    st.caption(f"**{len(tbl)}** authors match the current filters (from **{len(df_filtered)}** papers)")
    top_n = st.slider("Number of authors to show", 10, 200, 50)

    tbl_show = (
        tbl.sort_values(["NumPapers_Filtered", "NumCoauthors"], ascending=False)
       .head(top_n)
    )

    cols = ["Author", "NumPapers_Filtered", "NumPapers", "NumCoauthors"]
    for extra in ["Country", "Organization"]:
        if extra in tbl_show.columns:
            cols.append(extra)

    st.dataframe(tbl_show[cols], use_container_width=True)


    st.subheader("Network (Top authors only)")

    max_nodes = st.slider("Max nodes to display", 50, 400, 150)
    size_mode = st.radio("Node size based on", ["NumPapers", "NumCoauthors"], horizontal=True)

    # pick top authors to display (from already-filtered tbl)
    top_authors = (
        tbl.sort_values(["NumPapers_Filtered", "NumCoauthors"], ascending=False)
           .head(max_nodes)["Author"]
           .tolist()
    )


    # Build filtered coauthor edges (using df_filtered which has year + thread filters applied)
    edges_filtered = set()

    for _, row in df_filtered.iterrows():
        authors = [a.strip() for a in coauthors.parse_authors(row["Authors"]) if a.strip()]
        for i, a in enumerate(authors):
            for b in authors[i+1:]:
                edges_filtered.add((a, b))
                edges_filtered.add((b, a))

    H = nx.Graph()

    # add nodes first
    for a in top_authors:
        if a in G:
            H.add_node(a, **G.nodes[a])

    # add only year-valid edges
    for a, b in edges_filtered:
        if a in H and b in H:
            if G.has_edge(a, b):
                H.add_edge(a, b)


    if H.number_of_nodes() == 0:
        st.info("No nodes to display.")
    else:
        # layout (force-directed)
        pos = nx.spring_layout(H, seed=42, k=None)

    # edges
    edge_x, edge_y = [], []
    for u, v in H.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#bbbbbb"),
        hoverinfo="none"
    )

    # node sizes
    def node_size(n):
        val = H.nodes[n].get("num_papers" if size_mode == "NumPapers" else "num_coauthors", 0)
        # sqrt scaling so big nodes don't dominate
        return 6 + (val ** 0.5) * 2.5

    node_x, node_y, node_text, node_sizes = [], [], [], []
    for n in H.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_sizes.append(node_size(n))

        npapers = H.nodes[n].get("num_papers", 0)
        nco = H.nodes[n].get("num_coauthors", 0)
        country = H.nodes[n].get("country")
        org = H.nodes[n].get("organization")

        hover = f"{n}<br>Papers: {npapers}<br>Coauthors: {nco}"
        if country:
            hover += f"<br>Country: {country}"
        if org:
            hover += f"<br>Org: {org}"
        node_text.append(hover)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_sizes,
            color="#1f77b4",
            line=dict(width=1, color="#333333")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=750,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Showing {H.number_of_nodes()} authors and {H.number_of_edges()} coauthorship links.")


