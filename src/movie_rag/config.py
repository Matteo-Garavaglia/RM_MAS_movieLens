"""Frozen configuration for the first Qdrant dense-retrieval baseline."""

from __future__ import annotations

from dataclasses import dataclass


DocumentField = tuple[str, str]


# Exact field order and labels used by the winning notebook configuration.
D_FULL_ENRICHED_FIELDS: tuple[DocumentField, ...] = (
    ("Title", "canonical_title"),
    ("genres", "tmdb_genres"),
    ("Overview", "overview"),
    ("Tagline", "tagline"),
    ("keywords", "tmdb_keyword_text"),
    ("user tags", "ml_tags"),
    ("Directors", "directors"),
    ("Writers", "writers"),
    ("Composers", "composers"),
    ("Main cast", "cast_top"),
    ("Main characters", "cast_characters_top"),
    ("Collection", "collection_name"),
    ("Production companies", "production_companies"),
)


# Kept in Qdrant as payload. Missing or null columns are omitted point by point.
DEFAULT_PAYLOAD_COLUMNS: tuple[str, ...] = (
    "movieId",
    "imdbId",
    "tmdbId",
    "canonical_title",
    "ml_title",
    "tmdb_title",
    "tmdb_original_title",
    "release_date",
    "release_year",
    "ml_genres",
    "tmdb_genres",
    "ml_tags",
    "tmdb_keywords",
    "overview",
    "tagline",
    "runtime",
    "original_language",
    "origin_countries",
    "production_countries",
    "spoken_languages",
    "adult",
    "directors",
    "writers",
    "composers",
    "cast_top",
    "cast_characters_top",
    "collection_name",
    "production_companies",
    "ml_rating_mean",
    "ml_rating_count",
    "ml_weighted_rating",
    "tmdb_vote_average",
    "tmdb_vote_count",
    "tmdb_popularity",
    "budget",
    "revenue",
    "poster_path",
    "backdrop_path",
    "homepage",
)


@dataclass(frozen=True)
class DenseBaselineConfig:
    """Versioned choices carried from the ablation notebook to Qdrant."""

    name: str = "D_full_enriched_dense"
    version: str = "v1"
    experiment_name: str = "D_full_enriched"
    document_column: str = "document__D_full_enriched"
    id_column: str = "movieId"
    title_column: str = "canonical_title"
    document_fields: tuple[DocumentField, ...] = D_FULL_ENRICHED_FIELDS
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    document_max_length: int = 1024
    query_max_length: int = 512
    embedding_batch_size: int = 16
    normalize_embeddings: bool = True
    vector_name: str = "dense"
    distance: str = "Cosine"
    collection_name: str = "movies_d_full_enriched_dense_v1"
    payload_schema_version: str = "movie_payload_v1"

    @property
    def representation_version(self) -> str:
        return f"{self.experiment_name}_{self.version}"


BASELINE = DenseBaselineConfig()

