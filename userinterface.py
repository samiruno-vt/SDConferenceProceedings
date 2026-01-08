import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st

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
    ["Find Similar Papers", "Find Co-authors"]
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

    top_n = st.slider("Top N authors (by number of papers)", 10, 200, 50)

    tbl = author_stats.sort_values(["NumPapers", "NumCoauthors"], ascending=False).head(top_n)

    cols = ["Author", "NumPapers", "NumCoauthors"]
    for extra in ["Country", "Organization"]:
        if extra in tbl.columns:
            cols.append(extra)

    st.dataframe(tbl[cols], use_container_width=True)
