
import pandas as pd
import numpy as np
import re
import networkx as nx
from itertools import combinations
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer



def search_papers(query, df, search_in = ("Title", "Authors"), limit = 20):
    """
    Search papers by keyword in Title and/or Authors.
    Returns a small table with a 'row_idx' column for selecting a paper.
    """
    q = str(query).strip()
    if not q:
        return df.head(0)

    mask = pd.Series(False, index = df.index)
    for col in search_in:
        mask |= df[col].fillna("").str.contains(q, case = False, na = False)

    out = df.loc[mask, ["Title", "Year", "Authors"]].copy()
    out = out.head(limit)
    return out.reset_index().rename(columns = {"index": "row_idx"})

def get_similar_papers(
    paper_idx,
    embeddings,
    df,
    k = 10,
    year_min = None,
    year_max = None
):
    """
    paper_idx: integer row index in df
    embeddings: numpy array (num_papers, dim), normalized
    df: DataFrame with at least Title, Abstract, Authors, Year
    Returns a DataFrame: Title, Year, Abstract, Authors
    """

    n = len(df)
    if paper_idx < 0 or paper_idx >= n:
        raise ValueError("paper_idx out of range")

    q = embeddings[paper_idx]

    mask = np.ones(n, dtype = bool)
    mask[paper_idx] = False  # exclude the query paper

    if year_min is not None:
        mask &= df["Year"].values >= year_min
    if year_max is not None:
        mask &= df["Year"].values <= year_max

    candidate_idxs = np.where(mask)[0]
    if len(candidate_idxs) == 0:
        return df.head(0)[["Title", "Year", "Abstract", "Authors"]]

    scores = embeddings[candidate_idxs] @ q  # cosine similarity

    k_eff = min(k, len(candidate_idxs))
    top_local = np.argsort(-scores)[:k_eff]
    top_idxs = candidate_idxs[top_local]

    # Build results
    result = df.loc[top_idxs, ["Title", "Year", "Abstract", "Authors"]].copy()

    # Reset index so no row_idx/index shows up in the table
    result = result.reset_index(drop = True)

    return result


