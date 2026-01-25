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


# -------------------
# Page config and styling
# -------------------

st.set_page_config(page_title="SD Conference Proceedings", layout="wide")

# Custom CSS for tab styling
st.markdown("""
    <style>
    /* Make tabs more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: white;
        border-radius: 8px;
        border: 1px solid #ddd;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("International System Dynamics Conference Proceedings (Demo)")

# Sidebar title and stats
st.sidebar.title("International System Dynamics Conference Proceedings")
st.sidebar.markdown("*Demo Version*")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Overview**")
st.sidebar.write(f"Years: {int(df['Year'].min())}–{int(df['Year'].max())}")
st.sidebar.write(f"Papers: {len(df):,}")
st.sidebar.write(f"Authors: {G.number_of_nodes():,}")

# -------------------
# Tab navigation
# -------------------

tab1, tab2, tab3 = st.tabs(["Network Overview", "Find Co-authors", "Find Similar Papers"])


# -----------------------
# Tab 3: Similar papers
# -----------------------

with tab3:
    st.header("Find Similar Papers")

    st.markdown(
        "Search for a paper, then select it to see similar ones based on titles and abstracts."
    )

    # Search field selection
    search_field = st.radio(
        "Search in:",
        options=["All fields", "Title", "Authors", "Abstract"],
        horizontal=True
    )
    
    # Map selection to search_in parameter
    search_in_map = {
        "All fields": ("Title", "Authors", "Abstract"),
        "Title": ("Title",),
        "Authors": ("Authors",),
        "Abstract": ("Abstract",)
    }
    search_in = search_in_map[search_field]

    query = st.text_input(f"Search for a paper")

    if query:
        # search_papers should return a DataFrame with columns: row_idx, Title, Year, Authors
        candidates = papers.search_papers(query, df, search_in=search_in, limit=20)

        if candidates.empty:
            st.info("No papers found for that search.")
        else:
            st.subheader("Matching papers")
            st.caption("Click on a row to select a paper")
            
            # Prepare display dataframe
            display_df = candidates[["Title", "Year", "Authors"]].copy()
            display_df.index = range(1, len(display_df) + 1)  # Start numbering at 1
            
            # Clickable dataframe for selection
            selection = st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=False,
                on_select="rerun",
                selection_mode="single-row",
            )
            
            # Get selected row
            selected_rows = selection.selection.rows if selection.selection else []
            
            if selected_rows:
                selected_idx = selected_rows[0]  # Get first (only) selected row
                
                # Get the underlying row index into df
                paper_row_idx = int(candidates.iloc[selected_idx]["row_idx"])

                # Show selected paper details
                selected_paper = df.loc[paper_row_idx]
                st.markdown("---")
                st.markdown("**Selected paper**")
                st.markdown(f"**Title:** {selected_paper['Title']}")
                st.markdown(f"**Year:** {selected_paper['Year']}")
                st.markdown(f"**Thread:** {selected_paper.get('Category', 'N/A')}")
                st.markdown(f"**Authors:** {selected_paper['Authors']}")
                if "Abstract" in df.columns:
                    st.markdown("**Abstract:**")
                    st.write(selected_paper["Abstract"])

                # Get similar papers
                k = st.radio("Number of similar papers to show", [5, 10, 15, 20], index=1, horizontal=True)
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
                    # Fix index to start at 1
                    sim_tbl.index = range(1, len(sim_tbl) + 1)
                    st.caption("💡 Double-click a cell to read full text. Scroll right to see Authors and Thread.")
                    st.dataframe(sim_tbl, use_container_width=True)


# ---------------------------
# Tab 2: Co-author explorer
# ---------------------------

with tab2:
    st.header("Co-author Network Explorer")

    author_query = st.text_input("Search for an author")

    if author_query:
        all_authors = sorted(G.nodes())
        candidates = coauthors.search_authors(author_query, all_authors, limit=10, score_cutoff=60)

        if not candidates:
            st.info("No matching authors found.")
        else:
            author_names = [name for name, score in candidates]
            selected_author = st.radio("Select an author:", options=author_names)

            author = selected_author
            st.markdown(f"**Selected author:** {author}")
            
            # Degrees selector - horizontal radio buttons
            max_degree = st.radio(
                "Degrees of separation",
                options=[1, 2, 3, 4],
                index=1,  # Default to 2
                horizontal=True,
                help="1 = direct co-authors only, 2 = co-authors of co-authors, etc."
            )

            # Get co-authors by degree
            degree_dfs = coauthors.get_coauthors_by_degree(G, author, max_degree=max_degree)
            
            # Display tables in columns (up to 4)
            if degree_dfs:
                degree_labels = ["1st degree (direct)", "2nd degree", "3rd degree", "4th degree"]
                cols = st.columns(min(len(degree_dfs), 4))
                
                for i, (col, degree_df) in enumerate(zip(cols, degree_dfs)):
                    with col:
                        st.subheader(degree_labels[i])
                        if degree_df.empty:
                            st.write(f"No {degree_labels[i].lower()} co-authors found.")
                        else:
                            st.caption(f"{len(degree_df)} authors")
                            # Fix index to start at 1
                            degree_df_display = degree_df.copy()
                            degree_df_display.index = range(1, len(degree_df_display) + 1)
                            st.dataframe(degree_df_display, use_container_width=True)

            # Network visualization
            H = coauthors.build_coauthor_network(G, author, max_degree=max_degree)
            if H.number_of_nodes() == 0:
                st.info("No co-author network to display for this author.")
            else:
                fig = coauthors.plot_coauthor_network(H, author)
                st.subheader("Co-author network")
                
                # Visual color legend with colored circles
                legend_colors = ["#d62828", "#2a9d8f", "#457b9d", "#8338ec", "#6c757d"]
                legend_labels = ["Selected author", "1st degree", "2nd degree", "3rd degree", "4th degree"]
                
                # Build legend HTML with colored circles
                legend_items = []
                for i in range(max_degree + 1):
                    color = legend_colors[i]
                    label = legend_labels[i]
                    legend_items.append(
                        f'<span style="display:inline-flex; align-items:center; margin-right:16px;">'
                        f'<span style="display:inline-block; width:12px; height:12px; border-radius:50%; '
                        f'background-color:{color}; margin-right:5px; border:1px solid white; '
                        f'box-shadow:0 0 2px rgba(0,0,0,0.3);"></span>'
                        f'<span style="color:#555; font-size:13px;">{label}</span></span>'
                    )
                
                legend_html = "".join(legend_items)
                stats_html = (
                    f'<span style="color:#555; font-size:13px; margin-left:8px;">'
                    f'· {H.number_of_nodes()} authors, {H.number_of_edges()} connections '
                    f'· Edge thickness = shared papers</span>'
                )
                
                st.markdown(
                    f'<div style="display:flex; flex-wrap:wrap; align-items:center; margin-bottom:8px;">'
                    f'{legend_html}{stats_html}</div>',
                    unsafe_allow_html=True
                )
                
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Tab 1: Network Overview
# ---------------------------

with tab1:
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

    # Country and Organization filters
    col_country, col_org = st.columns(2)
    
    with col_country:
        all_countries = sorted([c for c in author_stats["Country"].dropna().unique() if c])
        selected_countries = st.multiselect(
            "Author country (leave empty for all)",
            options=all_countries,
            default=[],
            help="Filter to authors from these countries. Leave empty to include all authors."
        )
    
    with col_org:
        all_orgs = sorted([o for o in author_stats["Organization"].dropna().unique() if o])
        selected_orgs = st.multiselect(
            "Author organization (leave empty for all)",
            options=all_orgs,
            default=[],
            help="Type to filter the list, then click to select. Matching options appear at the top; scroll down shows all options."
        )

    # Apply year filter first
    df_filtered = df[df["Year"].between(year_min, year_max)].copy()
    
    # Apply thread filter if any threads are selected
    if selected_threads:
        df_filtered = df_filtered[df_filtered["Category"].isin(selected_threads)]

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

    # Normalize author_stats Author names to match parsed names
    author_stats_normalized = author_stats.copy()
    author_stats_normalized["Author"] = author_stats_normalized["Author"].apply(coauthors.normalize_author_name)
    
    tbl = author_stats_normalized.merge(
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
    
    # Apply country filter
    if selected_countries:
        tbl = tbl[tbl["Country"].isin(selected_countries)]
    
    # Apply organization filter
    if selected_orgs:
        tbl = tbl[tbl["Organization"].isin(selected_orgs)]

    st.divider()
    
    # --- Author table section ---
    st.subheader("Top Authors")
    st.caption(f"**{len(tbl):,}** authors match the current filters (from **{len(df_filtered):,}** papers in selected year/thread range)")
    top_n = st.slider("Number of authors to show", 10, 200, 50)

    tbl_show = (
        tbl.sort_values(["NumPapers_Filtered", "NumCoauthors"], ascending=False)
       .head(top_n)
    )

    cols = ["Author", "NumPapers_Filtered", "NumPapers", "NumCoauthors"]
    for extra in ["Country", "Organization"]:
        if extra in tbl_show.columns:
            cols.append(extra)

    # Rename columns for display
    tbl_display = tbl_show[cols].copy()
    tbl_display = tbl_display.rename(columns={
        "NumPapers_Filtered": "Papers (filtered)",
        "NumPapers": "Total Papers",
        "NumCoauthors": "Co-authors"
    })
    tbl_display.index = range(1, len(tbl_display) + 1)  # Start at 1
    
    st.dataframe(tbl_display, use_container_width=True)


    st.subheader("Network (Top authors only)")

    max_nodes = st.slider("Max nodes to display", 50, 400, 50)
    size_mode = st.radio("Node size based on", ["Total Papers", "Co-authors"], horizontal=True)

    # pick top authors to display (from already-filtered tbl)
    top_authors = (
        tbl.sort_values(["NumPapers_Filtered", "NumCoauthors"], ascending=False)
           .head(max_nodes)["Author"]
           .tolist()
    )

    # Create mapping from normalized names to original graph node names
    normalized_to_original = {coauthors.normalize_author_name(n): n for n in G.nodes()}

    # Build filtered coauthor edges (using df_filtered which has year + thread filters applied)
    edges_filtered = set()

    for _, row in df_filtered.iterrows():
        authors = [a.strip() for a in coauthors.parse_authors(row["Authors"]) if a.strip()]
        for i, a in enumerate(authors):
            for b in authors[i+1:]:
                edges_filtered.add((a, b))
                edges_filtered.add((b, a))

    H = nx.Graph()

    # add nodes first (using normalized names, but getting attributes from original graph)
    for a in top_authors:
        original_name = normalized_to_original.get(a)
        if original_name and original_name in G:
            H.add_node(a, **G.nodes[original_name])

    # add only year-valid edges with weights from original graph
    for a, b in edges_filtered:
        if a in H and b in H:
            # Check if edge exists in original graph (using original names)
            orig_a = normalized_to_original.get(a)
            orig_b = normalized_to_original.get(b)
            if orig_a and orig_b and G.has_edge(orig_a, orig_b):
                # Copy edge weight (shared papers) from original graph
                weight = G[orig_a][orig_b].get("weight", 1)
                H.add_edge(a, b, weight=weight)


    if H.number_of_nodes() == 0:
        st.info("No nodes to display.")
    else:
        n_nodes = H.number_of_nodes()
        
        # Use Fruchterman-Reingold with strong repulsion to spread nodes
        # The key is a HIGH k value (repulsion) and many iterations
        pos = nx.spring_layout(
            H, 
            seed=42, 
            k=8/np.sqrt(n_nodes),  # Much stronger repulsion
            iterations=300,  # More iterations
            scale=3  # Larger scale
        )
        
        # Get edge weights for thickness scaling
        edge_weights = [H[u][v].get("weight", 1) for u, v in H.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        min_weight = min(edge_weights) if edge_weights else 1
        
        # Build edge data with varying thickness - use single trace with separate segments
        edge_x, edge_y, edge_widths, edge_colors, edge_hover = [], [], [], [], []
        
        for u, v in H.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            weight = H[u][v].get("weight", 1)
            
            # Scale edge width: 1 to 8 pixels based on weight (much more visible)
            if max_weight > min_weight:
                normalized_weight = (weight - min_weight) / (max_weight - min_weight)
            else:
                normalized_weight = 0.5
            width = 1 + normalized_weight * 7
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_widths.append(width)
            edge_hover.append(f"{u} ↔ {v}: {weight} shared papers")
        
        # Create multiple edge traces grouped by width for better rendering
        # Group edges into buckets for cleaner rendering
        edge_traces = []
        edges_by_weight = {}
        
        for i, (u, v) in enumerate(H.edges()):
            weight = H[u][v].get("weight", 1)
            if weight not in edges_by_weight:
                edges_by_weight[weight] = {"x": [], "y": [], "hover": []}
            
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edges_by_weight[weight]["x"].extend([x0, x1, None])
            edges_by_weight[weight]["y"].extend([y0, y1, None])
            edges_by_weight[weight]["hover"].append(f"{u} ↔ {v}: {weight} papers")
        
        for weight, data in sorted(edges_by_weight.items()):
            if max_weight > min_weight:
                normalized = (weight - min_weight) / (max_weight - min_weight)
            else:
                normalized = 0.5
            
            line_width = 1 + normalized * 7  # 1px to 8px
            # Darker color for stronger connections
            gray_val = int(180 - normalized * 100)  # 180 (light) to 80 (dark)
            
            edge_traces.append(go.Scatter(
                x=data["x"],
                y=data["y"],
                mode="lines",
                line=dict(width=line_width, color=f"rgb({gray_val},{gray_val},{gray_val})"),
                hoverinfo="skip",  # Edges don't hover well in Plotly
                showlegend=False
            ))

        # Get metric values for sizing
        metric_key = "num_papers" if size_mode == "Total Papers" else "num_coauthors"
        all_vals = [H.nodes[n].get(metric_key, 0) for n in H.nodes()]
        max_val = max(all_vals) if all_vals else 1
        min_val = min(all_vals) if all_vals else 0
        
        # Much more aggressive node sizing
        def node_size(val):
            if max_val == min_val:
                return 25
            # Normalize to 0-1
            normalized = (val - min_val) / (max_val - min_val)
            # Use power curve for dramatic difference: small nodes ~15px, large nodes ~80px
            return 15 + (normalized ** 0.5) * 65

        node_x, node_y, node_text, node_sizes = [], [], [], []
        
        for n in H.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            
            val = H.nodes[n].get(metric_key, 0)
            node_sizes.append(node_size(val))

            npapers = H.nodes[n].get("num_papers", 0)
            nco = H.nodes[n].get("num_coauthors", 0)
            country = H.nodes[n].get("country")
            org = H.nodes[n].get("organization")

            hover = f"<b>{n}</b><br>Papers: {npapers}<br>Co-authors: {nco}"
            if country:
                hover += f"<br>Country: {country}"
            if org:
                hover += f"<br>Organization: {org}"
            node_text.append(hover)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            hoverinfo="text",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color="#2a9d8f",  # Single teal color for all nodes
                line=dict(width=2, color="white"),
                opacity=0.9,
                sizemode="diameter"
            )
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="#f8f9fa",
            margin=dict(l=10, r=10, t=10, b=10),
            height=750,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            dragmode="pan",
            hovermode="closest"
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # Updated caption without color reference
        st.caption(f"Showing {H.number_of_nodes()} authors and {H.number_of_edges()} coauthorship links.")
        st.caption(f"**Node size** = {size_mode} · **Edge thickness** = shared papers (thicker = more collaborations)")


