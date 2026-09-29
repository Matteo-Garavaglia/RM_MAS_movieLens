from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_catalog() -> pd.DataFrame:
    common = {
        "tagline": None,
        "tmdb_keyword_text": "",
        "ml_tags": "",
        "directors": "",
        "writers": "",
        "composers": "",
        "cast_top": "",
        "cast_characters_top": "",
        "collection_name": None,
        "production_companies": "",
        "adult": False,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "movieId": 1,
                "imdbId": 101,
                "tmdbId": 201,
                "canonical_title": "Memory Film (1999)",
                "tmdb_genres": "Drama, Science Fiction",
                "ml_genres": "Drama|Sci-Fi",
                "overview": "A reflective story about memory and identity.",
                "runtime": 110,
                "release_year": 1999,
                "original_language": "en",
                "origin_countries": "US",
                "production_countries": "US, CA",
                "directors": "Jane Doe",
            },
            {
                **common,
                "movieId": 2,
                "imdbId": 102,
                "tmdbId": 202,
                "canonical_title": "Long Horror (2004)",
                "tmdb_genres": "Horror, Thriller",
                "ml_genres": "Horror|Thriller",
                "overview": "A tense night in an isolated house.",
                "runtime": 145,
                "release_year": 2004,
                "original_language": "en",
                "origin_countries": "GB",
                "production_countries": "GB",
            },
            {
                **common,
                "movieId": 3,
                "imdbId": 103,
                "tmdbId": 203,
                "canonical_title": "Famiglia (2010)",
                "tmdb_genres": "Drama",
                "ml_genres": "Drama",
                "overview": "Una famiglia affronta un grande cambiamento.",
                "runtime": 95,
                "release_year": 2010,
                "original_language": "it",
                "origin_countries": "IT",
                "production_countries": "IT",
            },
        ]
    )

