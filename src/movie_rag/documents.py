"""Document construction, catalog validation, payloads and fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import BASELINE, DEFAULT_PAYLOAD_COLUMNS, DenseBaselineConfig, DocumentField


NULL_STRINGS = {"", "nan", "none", "null", "<na>", "nat"}


def clean_text_value(value: Any) -> str:
    """Serialize a scalar or list-like value exactly as document text needs it."""

    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        parts = [clean_text_value(item) for item in value]
        return " | ".join(part for part in parts if part)
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.casefold() in NULL_STRINGS else text


def build_labeled_document(
    row: Mapping[str, Any] | pd.Series,
    fields: Sequence[DocumentField] = BASELINE.document_fields,
) -> str:
    """Build one labelled document while preserving the configured field order."""

    lines: list[str] = []
    for label, column in fields:
        if column not in row:
            continue
        value = clean_text_value(row[column])
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def validate_catalog_schema(
    dataframe: pd.DataFrame,
    config: DenseBaselineConfig = BASELINE,
) -> None:
    """Fail early when the catalog cannot reproduce the frozen baseline."""

    required = {
        config.id_column,
        config.title_column,
        *(column for _, column in config.document_fields),
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise KeyError(f"Missing columns required by {config.representation_version}: {missing}")
    if dataframe.empty:
        raise ValueError("The movie catalog is empty")
    if dataframe[config.id_column].isna().any():
        raise ValueError(f"{config.id_column} contains null values")
    if dataframe[config.id_column].duplicated().any():
        duplicate_ids = dataframe.loc[
            dataframe[config.id_column].duplicated(keep=False), config.id_column
        ].tolist()
        raise ValueError(f"{config.id_column} must be unique; duplicates: {duplicate_ids}")


def prepare_catalog(
    dataframe: pd.DataFrame,
    config: DenseBaselineConfig = BASELINE,
) -> pd.DataFrame:
    """Validate, reset the index and add the frozen D document column."""

    validate_catalog_schema(dataframe, config)
    prepared = dataframe.copy().reset_index(drop=True)
    prepared[config.document_column] = prepared.apply(
        build_labeled_document,
        axis=1,
        fields=config.document_fields,
    )
    empty_documents = prepared[config.document_column].str.strip().eq("")
    if empty_documents.any():
        ids = prepared.loc[empty_documents, config.id_column].tolist()
        raise ValueError(f"Empty documents generated for movie IDs: {ids}")
    # Verify IDs can be represented as Qdrant unsigned integer point IDs.
    for value in prepared[config.id_column]:
        coerce_point_id(value)
    return prepared


def coerce_point_id(value: Any) -> int:
    """Convert MovieLens IDs to safe Qdrant integer point IDs."""

    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid movie IDs")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid movie ID: {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError(f"Movie ID must be a non-negative integer: {value!r}")
    return int(numeric)


def split_multi_value(value: Any, separators: Sequence[str] = ("|", ",")) -> list[str]:
    """Convert CSV-style multi-value fields into trimmed, unique values."""

    if isinstance(value, (list, tuple, set, np.ndarray)):
        raw_items: Iterable[Any] = value
    else:
        text = clean_text_value(value)
        if not text:
            return []
        pattern = "|".join(re.escape(separator) for separator in separators)
        raw_items = re.split(pattern, text)

    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        cleaned = clean_text_value(item)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def normalized_values(values: Iterable[str]) -> list[str]:
    """Case-fold and deduplicate keyword values used by exact Qdrant filters."""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = clean_text_value(value).casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_json_safe(item) for item in value]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _combined_values(row: Mapping[str, Any] | pd.Series, columns: Sequence[str]) -> list[str]:
    values: list[str] = []
    for column in columns:
        if column in row:
            values.extend(split_multi_value(row[column]))
    # Preserve the first human-readable spelling while deduplicating case-insensitively.
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def row_to_payload(
    row: Mapping[str, Any] | pd.Series,
    document: str,
    dataset_fingerprint: str,
    config: DenseBaselineConfig = BASELINE,
    payload_columns: Sequence[str] = DEFAULT_PAYLOAD_COLUMNS,
) -> dict[str, Any]:
    """Build a JSON-safe Qdrant payload with raw and normalized filter fields."""

    payload: dict[str, Any] = {}
    for column in payload_columns:
        if column not in row:
            continue
        value = _json_safe(row[column])
        if value is not None and value != "":
            payload[column] = value

    # CSV columns containing missing values are often promoted to floats. Keep the
    # filter and identifier payload fields typed consistently for Qdrant indexes.
    payload[config.id_column] = coerce_point_id(row[config.id_column])
    for column in ("imdbId", "tmdbId", "release_year", "runtime"):
        if column in payload:
            numeric = float(payload[column])
            if math.isfinite(numeric) and numeric.is_integer():
                payload[column] = int(numeric)
    if "adult" in payload and not isinstance(payload["adult"], bool):
        adult_text = str(payload["adult"]).strip().casefold()
        if adult_text in {"true", "1", "yes", "y"}:
            payload["adult"] = True
        elif adult_text in {"false", "0", "no", "n"}:
            payload["adult"] = False

    genres = _combined_values(row, ("ml_genres", "tmdb_genres"))
    countries = _combined_values(row, ("origin_countries", "production_countries"))

    payload.update(
        {
            "document": document,
            "genres": genres,
            "genres_norm": normalized_values(genres),
            "countries": countries,
            "countries_norm": normalized_values(countries),
            "original_language_norm": clean_text_value(
                row.get("original_language", "")
            ).casefold(),
            "retrieval_experiment": config.experiment_name,
            "representation_version": config.representation_version,
            "payload_schema_version": config.payload_schema_version,
            "embedding_model": config.embedding_model,
            "dataset_fingerprint": dataset_fingerprint,
        }
    )
    return {key: value for key, value in payload.items() if value not in (None, "")}


def catalog_fingerprint(
    prepared: pd.DataFrame,
    config: DenseBaselineConfig = BASELINE,
) -> str:
    """Hash IDs plus generated documents, independent of DataFrame row order."""

    rows = [
        {
            "id": coerce_point_id(row[config.id_column]),
            "document": str(row[config.document_column]),
        }
        for _, row in prepared.iterrows()
    ]
    rows.sort(key=lambda item: item["id"])
    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
