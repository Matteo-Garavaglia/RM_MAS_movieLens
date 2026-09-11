"""Frozen ranking metrics from the ablation notebook."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def build_annotation_extension(
    results: pd.DataFrame,
    annotated_pool: pd.DataFrame,
    id_column: str = "movieId",
    title_column: str = "canonical_title",
) -> pd.DataFrame:
    """Return retrieved query-document pairs that still need a judgment.

    This is intended for incremental pooling: after adding a new system, only
    its previously unseen top results need to be judged before metrics are
    compared fairly.
    """

    required_results = {"query_id", "system", "rank", id_column}
    required_qrels = {"query_id", id_column, "relevance"}
    missing_results = required_results - set(results.columns)
    missing_qrels = required_qrels - set(annotated_pool.columns)
    if missing_results:
        raise KeyError(f"Missing result columns: {sorted(missing_results)}")
    if missing_qrels:
        raise KeyError(f"Missing annotation columns: {sorted(missing_qrels)}")

    judged = annotated_pool.dropna(subset=["relevance"])
    judged_pairs = set(zip(judged["query_id"], judged[id_column]))
    unseen_mask = [
        (query_id, movie_id) not in judged_pairs
        for query_id, movie_id in zip(results["query_id"], results[id_column])
    ]
    unseen = results.loc[unseen_mask].copy()
    if unseen.empty:
        return pd.DataFrame(
            columns=[
                "query_id",
                "query",
                id_column,
                title_column,
                "retrieved_by",
                "best_rank",
                "relevance",
                "notes",
            ]
        )

    rows: list[dict[str, object]] = []
    for (query_id, movie_id), group in unseen.groupby(
        ["query_id", id_column], sort=False
    ):
        systems = sorted(
            {
                f"{row.system}@{int(row.rank)}"
                for row in group[["system", "rank"]].itertuples(index=False)
            }
        )
        first = group.iloc[0]
        rows.append(
            {
                "query_id": query_id,
                "query": first.get("query"),
                id_column: movie_id,
                title_column: first.get(title_column),
                "retrieved_by": " | ".join(systems),
                "best_rank": int(group["rank"].min()),
                "relevance": pd.NA,
                "notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["query_id", "best_rank", id_column], kind="stable"
    ).reset_index(drop=True)


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
    expected_query_ids: Sequence[str] | None = None,
    strict_judgments: bool = False,
) -> pd.DataFrame:
    """Compute NDCG, recall and precision with the notebook's qrels semantics.

    When ``expected_query_ids`` is supplied, an absent query/system run is
    evaluated as an empty ranking rather than silently disappearing from the
    aggregate.  The default keeps backward compatibility with the original
    ablation notebook.
    """

    required_result_columns = {"query_id", "experiment", "system", "rank", id_column}
    required_qrel_columns = {"query_id", id_column, "relevance"}
    missing_results = required_result_columns - set(results.columns)
    missing_qrels = required_qrel_columns - set(annotated_pool.columns)
    if missing_results:
        raise KeyError(f"Missing result columns: {sorted(missing_results)}")
    if missing_qrels:
        raise KeyError(f"Missing annotation columns: {sorted(missing_qrels)}")

    if strict_judgments:
        unjudged = build_annotation_extension(
            results=results,
            annotated_pool=annotated_pool,
            id_column=id_column,
        )
        if not unjudged.empty:
            raise ValueError(
                f"The evaluated runs contain {len(unjudged)} unjudged "
                "query-document pairs"
            )

    judged = annotated_pool.dropna(subset=["relevance"]).copy()
    judged["relevance"] = judged["relevance"].astype(float)
    if (~judged["relevance"].isin([0.0, 1.0, 2.0])).any():
        raise ValueError("Relevance judgments must be 0, 1 or 2")

    qrels = {
        query_id: dict(zip(group[id_column], group["relevance"]))
        for query_id, group in judged.groupby("query_id")
    }

    observed_groups = {
        key: group
        for key, group in results.groupby(["query_id", "experiment", "system"])
    }
    if expected_query_ids is None:
        evaluation_keys = list(observed_groups)
    else:
        query_ids = list(dict.fromkeys(str(value) for value in expected_query_ids))
        run_configs = list(
            results[["experiment", "system"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        evaluation_keys = [
            (query_id, experiment, system)
            for query_id in query_ids
            for experiment, system in run_configs
        ]

    rows: list[dict[str, float | int | str]] = []
    for query_id, experiment, system in evaluation_keys:
        if query_id not in qrels:
            continue
        group = observed_groups.get(
            (query_id, experiment, system),
            results.iloc[0:0],
        )
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
                "retrieved_items_at_k": len(ranking),
            }
        )
    return pd.DataFrame(rows)
