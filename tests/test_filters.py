from __future__ import annotations

import pytest

from movie_rag.filters import MovieFilters, dataframe_filter_mask


def selected_ids(dataframe, filters):
    mask = dataframe_filter_mask(dataframe, filters)
    return dataframe.loc[mask, "movieId"].tolist()


def test_structured_filters_use_exact_normalized_values(sample_catalog):
    assert selected_ids(sample_catalog, {"max_runtime": 120}) == [1, 3]
    assert selected_ids(sample_catalog, {"include_genres": ["Science Fiction"]}) == [1]
    assert selected_ids(sample_catalog, {"include_genres": ["sci-fi"]}) == [1]
    assert selected_ids(sample_catalog, {"exclude_genres": ["horror"]}) == [1, 3]
    assert selected_ids(sample_catalog, {"countries": ["it"], "languages": ["IT"]}) == [3]


def test_filter_bounds_are_validated():
    with pytest.raises(ValueError):
        MovieFilters(min_year=2020, max_year=2000)
    with pytest.raises(KeyError):
        MovieFilters.from_mapping({"unknown": 1})


def test_qdrant_filter_is_built_from_same_schema():
    pytest.importorskip("qdrant_client")
    from movie_rag.filters import build_qdrant_filter

    result = build_qdrant_filter(
        {
            "max_runtime": 120,
            "include_genres": ["Science Fiction", "Sci-Fi"],
            "exclude_genres": ["Horror"],
        }
    )
    assert len(result.must) == 2
    assert len(result.must_not) == 1

