import numpy as np
import pandas as pd

from movie_rag import BASELINE, BASELINE_V2
from movie_rag.experimental_retrieval import sparse_dot, weighted_rrf


def test_v2_does_not_replace_runtime_baseline():
    assert BASELINE.version == "v1"
    assert BASELINE.title_column == "canonical_title"
    assert BASELINE_V2.version == "v2"
    assert BASELINE_V2.title_column == "tmdb_title"


def test_sparse_dot_uses_shared_lexical_features_only():
    q = {1: 2.0, 2: 3.0}
    d = {2: 4.0, 3: 10.0}
    assert sparse_dot(q, d) == 12.0


def test_rrf_uses_ranks_not_raw_score_scale():
    a = pd.DataFrame([
        {"movieId": 1, "rank": 1, "score": 1000.0},
        {"movieId": 2, "rank": 2, "score": 900.0},
    ])
    b = pd.DataFrame([
        {"movieId": 2, "rank": 1, "score": 0.5},
        {"movieId": 1, "rank": 2, "score": 0.4},
    ])
    result = weighted_rrf({"a": a, "b": b}, weights={"a": 0.5, "b": 0.5}, rrf_k=60, top_k=2)
    # Symmetric opposite rankings: both receive the same RRF score despite score scales.
    assert np.isclose(result.loc[0, "rrf_score"], result.loc[1, "rrf_score"])
