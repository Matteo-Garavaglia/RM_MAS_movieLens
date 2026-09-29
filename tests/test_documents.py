from __future__ import annotations

from movie_rag import BASELINE
from movie_rag.documents import catalog_fingerprint, prepare_catalog, row_to_payload


def test_frozen_document_order_and_null_omission(sample_catalog):
    prepared = prepare_catalog(sample_catalog)
    document = prepared.loc[0, BASELINE.document_column]
    assert document.splitlines() == [
        "Title: Memory Film (1999)",
        "genres: Drama, Science Fiction",
        "Overview: A reflective story about memory and identity.",
        "Directors: Jane Doe",
    ]


def test_catalog_fingerprint_does_not_depend_on_row_order(sample_catalog):
    first = prepare_catalog(sample_catalog)
    second = prepare_catalog(sample_catalog.iloc[::-1])
    assert catalog_fingerprint(first) == catalog_fingerprint(second)


def test_payload_has_typed_ids_and_normalized_filter_values(sample_catalog):
    prepared = prepare_catalog(sample_catalog)
    fingerprint = catalog_fingerprint(prepared)
    row = prepared.iloc[0]
    payload = row_to_payload(row, row[BASELINE.document_column], fingerprint)
    assert payload["movieId"] == 1
    assert isinstance(payload["runtime"], int)
    assert payload["genres_norm"] == ["drama", "sci-fi", "science fiction"]
    assert payload["countries_norm"] == ["us", "ca"]
    assert payload["original_language_norm"] == "en"
    assert payload["dataset_fingerprint"] == fingerprint

