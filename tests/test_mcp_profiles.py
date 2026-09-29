from __future__ import annotations

import json

import pytest

from movie_rag.mcp_profiles import (
    MCP_CONTRACT_VERSION,
    load_profile_catalog,
)


def test_versioned_profile_catalog_has_dense_default_and_optional_reranker():
    catalog = load_profile_catalog()

    assert catalog.contract_version == MCP_CONTRACT_VERSION
    assert catalog.default_profile_id == "dense_v1"
    assert catalog.get().pipeline == "dense"
    assert catalog.get().reranker_enabled is False
    reranked = catalog.get("dense_rerank_v1")
    assert reranked.reranker_enabled is True
    assert reranked.reranker_config is not None
    assert reranked.reranker_config.model_name == "BAAI/bge-reranker-v2-m3"


def test_unknown_profile_is_rejected_with_available_choices():
    catalog = load_profile_catalog()

    with pytest.raises(KeyError, match="dense_v1"):
        catalog.get("free-form-model-name")


def test_profile_file_rejects_unknown_fields(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "contract_version": MCP_CONTRACT_VERSION,
                "default_profile": "invalid",
                "profiles": {
                    "invalid": {
                        "pipeline": "dense",
                        "encoder_provider": "bge_m3",
                        "unexpected": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="unexpected"):
        load_profile_catalog(path)


def test_reranker_pool_must_cover_maximum_public_limit(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "contract_version": MCP_CONTRACT_VERSION,
                "default_profile": "bad",
                "profiles": {
                    "bad": {
                        "pipeline": "dense_rerank",
                        "encoder_provider": "bge_m3",
                        "max_limit": 20,
                        "reranker": {
                            "provider": "flag_embedding",
                            "candidate_pool_size": 10,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="candidate_pool_size"):
        load_profile_catalog(path)

