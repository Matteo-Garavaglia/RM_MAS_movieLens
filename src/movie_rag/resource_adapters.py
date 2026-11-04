from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .resource_models import RangeConstraint, ResourceDocument, ResourceFilters

NULL_STRINGS = {"", "nan", "none", "null", "<na>"}

# Exact v2 document representation selected for the MovieLens/TMDb experiments.
MOVIELENS_V2_DOCUMENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("Title", "tmdb_title"),
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
MOVIELENS_V2_DOCUMENT_COLUMN = "document__D_full_enriched_v2"


def clean_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        parts = [clean_text_value(item) for item in value]
        return " | ".join(part for part in parts if part)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() in NULL_STRINGS else text


def build_labeled_document(
    row: pd.Series | Mapping[str, Any],
    fields: Sequence[tuple[str, str]],
) -> str:
    lines: list[str] = []
    for label, column in fields:
        if column not in row:
            continue
        value = clean_text_value(row[column])
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def ensure_movielens_v2_document(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the exact D_full_enriched_v2 text column when it is not present."""

    result = frame.copy()
    if MOVIELENS_V2_DOCUMENT_COLUMN not in result.columns:
        result[MOVIELENS_V2_DOCUMENT_COLUMN] = result.apply(
            build_labeled_document,
            axis=1,
            fields=MOVIELENS_V2_DOCUMENT_FIELDS,
        )
    if result[MOVIELENS_V2_DOCUMENT_COLUMN].fillna("").astype(str).str.strip().eq("").any():
        print("Found empty documents in D_full_enriched_v2")
        print("empty document rows:")
        for i, row in result.iterrows():
            if row[MOVIELENS_V2_DOCUMENT_COLUMN] == "":
                print(f"  - Row {i}: {row.to_dict()}")
        raise ValueError("D_full_enriched_v2 contains empty documents")
    return result


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [x for item in value for x in _list_value(item)]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in NULL_STRINGS:
        return []
    # The enriched catalog contains both comma- and pipe-separated list columns.
    values = re.split(r"\s*[|,;]\s*", text)
    seen: set[str] = set()
    output: list[str] = []
    for item in values:
        item = item.strip()
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output




def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_json_scalar(item) for item in value]
    return value


def _first_existing(row: Mapping[str, Any], *columns: str) -> Any:
    for column in columns:
        if column in row:
            value = row[column]
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            if str(value).strip():
                return value
    return None


@dataclass(frozen=True)
class DataFrameResourceAdapter:
    """Configurable DataFrame -> ResourceDocument adapter.

    This is the portability seam for WhoTeach: after chunking educational material,
    only column mapping / metadata configuration should change.
    """

    id_column: str
    title_column: str
    text_column: str
    domain: str
    metadata_columns: Mapping[str, str] = field(default_factory=dict)
    source_column: str | None = None
    metadata_transforms: Mapping[str, Callable[[Any], Any]] = field(default_factory=dict)

    def to_resources(self, frame: pd.DataFrame) -> list[ResourceDocument]:
        required = {self.id_column, self.title_column, self.text_column}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"missing required columns: {missing}")
        if frame[self.id_column].duplicated().any():
            raise ValueError(f"duplicate ids in {self.id_column}")

        resources: list[ResourceDocument] = []
        for row in frame.to_dict("records"):
            metadata: dict[str, Any] = {}
            for key, column in self.metadata_columns.items():
                if column not in row:
                    continue
                value = row[column]
                transform = self.metadata_transforms.get(key)
                if transform is not None:
                    value = transform(value)
                try:
                    if pd.isna(value):
                        continue
                except (TypeError, ValueError):
                    pass
                if value in (None, "", [], {}):
                    continue
                metadata[key] = _json_scalar(value)

            source = None
            if self.source_column and self.source_column in row:
                source = clean_text_value(row[self.source_column]) or None

            resources.append(
                ResourceDocument(
                    resource_id=row[self.id_column],
                    title=clean_text_value(row[self.title_column]),
                    text=clean_text_value(row[self.text_column]),
                    metadata=metadata,
                    domain=self.domain,
                    source=source,
                )
            )
        return resources


@dataclass(frozen=True)
class MovieLensV2Adapter:
    id_column: str = "movieId"
    title_column: str = "tmdb_title"
    text_column: str = MOVIELENS_V2_DOCUMENT_COLUMN
    domain: str = "movielens_movies"

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = ensure_movielens_v2_document(frame)
        required = {self.id_column, self.title_column, self.text_column}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"missing MovieLens v2 columns: {missing}")
        return frame

    def to_resources(self, frame: pd.DataFrame) -> list[ResourceDocument]:
        prepared = self.prepare(frame)
        resources: list[ResourceDocument] = []
        if prepared[self.id_column].duplicated().any():
            raise ValueError("movieId must be unique")

        for row in prepared.to_dict("records"):
            genres = _list_value(row.get("tmdb_genres")) + _list_value(row.get("ml_genres"))
            genres = list(dict.fromkeys(genres))
            countries = _list_value(row.get("origin_countries")) + _list_value(row.get("production_countries"))
            countries = list(dict.fromkeys(countries))

            metadata: dict[str, Any] = {
                "year": _first_existing(row, "release_year", "ml_year"),
                "runtime": _first_existing(row, "runtime", "tmdb_runtime"),
                "language": _first_existing(row, "original_language"),
                "genres": genres,
                "countries": countries,
                "adult": _first_existing(row, "adult"),
                "tmdb_id": _first_existing(row, "tmdbId"),
                "imdb_id": _first_existing(row, "imdbId"),
            }
            metadata = {
                key: _json_scalar(value) for key, value in metadata.items()
                if value not in (None, "", [], {})
            }

            resources.append(
                ResourceDocument(
                    resource_id=row[self.id_column],
                    title=clean_text_value(row[self.title_column]),
                    text=clean_text_value(row[self.text_column]),
                    metadata=metadata,
                    domain=self.domain,
                    source="MovieLens + TMDb",
                )
            )
        return resources


def movie_filters_to_resource_filters(filters: Mapping[str, Any] | None) -> ResourceFilters:
    """Translate the existing movie filter contract to domain-agnostic metadata filters."""

    filters = dict(filters or {})
    supported = {
        "max_runtime", "min_runtime", "min_year", "max_year", "languages",
        "countries", "include_genres", "exclude_genres", "adult",
    }
    unknown = sorted(set(filters) - supported)
    if unknown:
        raise KeyError(f"unsupported movie filters: {unknown}")

    equals: dict[str, Any] = {}
    any_of: dict[str, list[Any]] = {}
    none_of: dict[str, list[Any]] = {}
    ranges: dict[str, RangeConstraint] = {}

    runtime = RangeConstraint(
        gte=float(filters["min_runtime"]) if "min_runtime" in filters else None,
        lte=float(filters["max_runtime"]) if "max_runtime" in filters else None,
    )
    if not runtime.is_empty():
        ranges["runtime"] = runtime

    year = RangeConstraint(
        gte=float(filters["min_year"]) if "min_year" in filters else None,
        lte=float(filters["max_year"]) if "max_year" in filters else None,
    )
    if not year.is_empty():
        ranges["year"] = year

    if filters.get("languages"):
        any_of["language"] = list(filters["languages"])
    if filters.get("countries"):
        any_of["countries"] = list(filters["countries"])
    if filters.get("include_genres"):
        any_of["genres"] = list(filters["include_genres"])
    if filters.get("exclude_genres"):
        none_of["genres"] = list(filters["exclude_genres"])
    if "adult" in filters:
        equals["adult"] = bool(filters["adult"])

    return ResourceFilters(equals=equals, any_of=any_of, none_of=none_of, ranges=ranges)
