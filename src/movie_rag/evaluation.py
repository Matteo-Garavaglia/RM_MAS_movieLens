"""Frozen ranking metrics from the ablation notebook."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    values = np.asarray(list(relevances)[:k], dtype=float)
    if len(values) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(values) + 2))
    gains = np.power(2.0, values) - 1.0
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    ranked_relevances: Sequence[float],
    all_judged_relevances: Sequence[float],
    k: int,
) -> float:
    actual = dcg_at_k(ranked_relevances, k)
    ideal = dcg_at_k(sorted(all_judged_relevances, reverse=True), k)
    return 0.0 if ideal == 0 else actual / ideal


def evaluate_results(
    results: pd.DataFrame,
    annotated_pool: pd.DataFrame,
    k: int = 10,
    id_column: str = "movieId",
) -> pd.DataFrame:
    """Compute NDCG, recall and precision with the notebook's qrels semantics."""

    required_result_columns = {"query_id", "experiment", "system", "rank", id_column}
    required_qrel_columns = {"query_id", id_column, "relevance"}
    missing_results = required_result_columns - set(results.columns)
    missing_qrels = required_qrel_columns - set(annotated_pool.columns)
    if missing_results:
        raise KeyError(f"Missing result columns: {sorted(missing_results)}")
    if missing_qrels:
        raise KeyError(f"Missing annotation columns: {sorted(missing_qrels)}")

    judged = annotated_pool.dropna(subset=["relevance"]).copy()
    judged["relevance"] = judged["relevance"].astype(float)
    if (~judged["relevance"].isin([0.0, 1.0, 2.0])).any():
        raise ValueError("Relevance judgments must be 0, 1 or 2")

    qrels = {
        query_id: dict(zip(group[id_column], group["relevance"]))
        for query_id, group in judged.groupby("query_id")
    }

    rows: list[dict[str, float | int | str]] = []
    for (query_id, experiment, system), group in results.groupby(
        ["query_id", "experiment", "system"]
    ):
        if query_id not in qrels:
            continue
        ranking = group.sort_values("rank").head(k)
        relevance_map = qrels[query_id]
        ranked_relevances = [
            float(relevance_map.get(movie_id, 0.0)) for movie_id in ranking[id_column]
        ]
        all_relevances = [float(value) for value in relevance_map.values()]
        relevant_ids = {
            movie_id for movie_id, relevance in relevance_map.items() if relevance > 0
        }
        retrieved_ids = set(ranking[id_column])
        rows.append(
            {
                "query_id": query_id,
                "experiment": experiment,
                "system": system,
                f"ndcg@{k}": ndcg_at_k(ranked_relevances, all_relevances, k),
                f"recall@{k}": (
                    len(relevant_ids & retrieved_ids) / len(relevant_ids)
                    if relevant_ids
                    else np.nan
                ),
                f"precision@{k}": sum(value > 0 for value in ranked_relevances) / k,
                "judged_items_for_query": len(relevance_map),
            }
        )
    return pd.DataFrame(rows)
