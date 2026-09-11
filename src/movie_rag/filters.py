"""One filter schema with equivalent Pandas and Qdrant implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from .documents import clean_text_value, normalized_values, split_multi_value


SUPPORTED_FILTERS = {
    "max_runtime",
    "min_runtime",
    "min_year",
    "max_year",
    "languages",
    "countries",
    "include_genres",
    "exclude_genres",
    "adult",
}


def _tuple_or_empty(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class MovieFilters:
    max_runtime: float | None = None
    min_runtime: float | None = None
    min_year: int | None = None
    max_year: int | None = None
    languages: tuple[str, ...] = ()
    countries: tuple[str, ...] = ()
    include_genres: tuple[str, ...] = ()
    exclude_genres: tuple[str, ...] = ()
    adult: bool | None = None

    def __post_init__(self) -> None:
        if (
            self.min_runtime is not None
            and self.max_runtime is not None
            and self.min_runtime > self.max_runtime
        ):
            raise ValueError("min_runtime cannot exceed max_runtime")
        if self.min_year is not None and self.max_year is not None and self.min_year > self.max_year:
            raise ValueError("min_year cannot exceed max_year")

    @classmethod
    def from_mapping(cls, filters: Mapping[str, Any] | "MovieFilters" | None) -> "MovieFilters":
        if filters is None:
            return cls()
        if isinstance(filters, cls):
            return filters
        unknown = set(filters) - SUPPORTED_FILTERS
        if unknown:
            raise KeyError(f"Unsupported filters: {sorted(unknown)}")
        values = dict(filters)
        for key in ("languages", "countries", "include_genres", "exclude_genres"):
            values[key] = _tuple_or_empty(values.get(key))
        return cls(**values)

    def is_empty(self) -> bool:
        return all(
            value is None or value == ()
            for value in (
                self.max_runtime,
                self.min_runtime,
                self.min_year,
                self.max_year,
                self.languages,
                self.countries,
                self.include_genres,
                self.exclude_genres,
                self.adult,
            )
        )


def _row_normalized_values(row: pd.Series, columns: Sequence[str]) -> set[str]:
    values: list[str] = []
    for column in columns:
        if column in row:
            values.extend(split_multi_value(row[column]))
    return set(normalized_values(values))


def _as_bool(value: Any) -> bool | None:
    if value is None or clean_text_value(value) == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return bool(value)


def dataframe_filter_mask(
    dataframe: pd.DataFrame,
    filters: Mapping[str, Any] | MovieFilters | None = None,
) -> pd.Series:
    """Apply the same exact-value semantics used by Qdrant payload filters."""

    parsed = MovieFilters.from_mapping(filters)
    mask = pd.Series(True, index=dataframe.index)

    runtime = pd.to_numeric(
        dataframe.get("runtime", pd.Series(float("nan"), index=dataframe.index)),
        errors="coerce",
    )
    year = pd.to_numeric(
        dataframe.get("release_year", pd.Series(float("nan"), index=dataframe.index)),
        errors="coerce",
    )
    if parsed.max_runtime is not None:
        mask &= runtime.notna() & runtime.le(parsed.max_runtime)
    if parsed.min_runtime is not None:
        mask &= runtime.notna() & runtime.ge(parsed.min_runtime)
    if parsed.min_year is not None:
        mask &= year.notna() & year.ge(parsed.min_year)
    if parsed.max_year is not None:
        mask &= year.notna() & year.le(parsed.max_year)

    if parsed.languages:
        allowed = set(normalized_values(parsed.languages))
        language = dataframe.get("original_language", pd.Series("", index=dataframe.index))
        mask &= language.map(lambda value: clean_text_value(value).casefold() in allowed)

    if parsed.countries:
        allowed = set(normalized_values(parsed.countries))
        mask &= dataframe.apply(
            lambda row: bool(
                _row_normalized_values(row, ("origin_countries", "production_countries"))
                & allowed
            ),
            axis=1,
        )

    if parsed.include_genres:
        allowed = set(normalized_values(parsed.include_genres))
        mask &= dataframe.apply(
            lambda row: bool(_row_normalized_values(row, ("ml_genres", "tmdb_genres")) & allowed),
            axis=1,
        )

    if parsed.exclude_genres:
        excluded = set(normalized_values(parsed.exclude_genres))
        mask &= ~dataframe.apply(
            lambda row: bool(_row_normalized_values(row, ("ml_genres", "tmdb_genres")) & excluded),
            axis=1,
        )

    if parsed.adult is not None:
        adult = dataframe.get("adult", pd.Series(None, index=dataframe.index))
        mask &= adult.map(_as_bool).eq(parsed.adult)

    return mask


def build_qdrant_filter(
    filters: Mapping[str, Any] | MovieFilters | None = None,
) -> Any | None:
    """Translate MovieFilters into qdrant-client models.Filter."""

    parsed = MovieFilters.from_mapping(filters)
    if parsed.is_empty():
        return None

    try:
        from qdrant_client import models
    except ImportError as exc:
        raise ImportError("qdrant-client is required to build Qdrant filters") from exc

    must: list[Any] = []
    must_not: list[Any] = []

    if parsed.min_runtime is not None or parsed.max_runtime is not None:
        must.append(
            models.FieldCondition(
                key="runtime",
                range=models.Range(gte=parsed.min_runtime, lte=parsed.max_runtime),
            )
        )
    if parsed.min_year is not None or parsed.max_year is not None:
        must.append(
            models.FieldCondition(
                key="release_year",
                range=models.Range(gte=parsed.min_year, lte=parsed.max_year),
            )
        )
    if parsed.languages:
        must.append(
            models.FieldCondition(
                key="original_language_norm",
                match=models.MatchAny(any=normalized_values(parsed.languages)),
            )
        )
    if parsed.countries:
        must.append(
            models.FieldCondition(
                key="countries_norm",
                match=models.MatchAny(any=normalized_values(parsed.countries)),
            )
        )
    if parsed.include_genres:
        must.append(
            models.FieldCondition(
                key="genres_norm",
                match=models.MatchAny(any=normalized_values(parsed.include_genres)),
            )
        )
    if parsed.exclude_genres:
        must_not.append(
            models.FieldCondition(
                key="genres_norm",
                match=models.MatchAny(any=normalized_values(parsed.exclude_genres)),
            )
        )
    if parsed.adult is not None:
        must.append(
            models.FieldCondition(key="adult", match=models.MatchValue(value=parsed.adult))
        )

    return models.Filter(must=must or None, must_not=must_not or None)
