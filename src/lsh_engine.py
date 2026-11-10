"""MinHash LSH with Jaccard similarity: parameter choice, join, and k-NN.

Owner: Member 2 — Distributed LSH Core & Retrieval Lead

Spark ML notes for this module:
- ``MinHashLSH`` lives in ``pyspark.ml.feature``.
- ``approxSimilarityJoin`` takes a *distance* threshold.
  Jaccard distance = 1 - Jaccard similarity.
- A self-join emits both (a, b) and (b, a) plus self-matches.
  Keep pairs with ``datasetA.id < datasetB.id``.
- Spark's built-in join uses one hash per band. Custom banding groups
  K = b * r signatures into b bands of r rows, with collision probability
  P(s) = 1 - (1 - s^r)^b.
"""

from __future__ import annotations

from pyspark.ml import Model
from pyspark.sql import DataFrame, SparkSession

from src.config import PipelineConfig


def collision_probability(similarity: float, bands: int, rows: int) -> float:
    """Theoretical probability that similarity ``s`` collides under (b, r) LSH.

    TODO(Member 2): M2-1
    Return P = 1 - (1 - s^r)^b.
    Reject similarities outside [0, 1] and non-positive band or row counts.
    """
    raise NotImplementedError("TODO(Member 2): M2-1 collision_probability")


def choose_bands_and_rows(
    threshold: float,
    target_probability: float = 0.5,
    max_hashes: int = 64,
) -> tuple[int, int]:
    """Pick ``(bands, rows)`` so the S-curve crosses ``target_probability`` near ``threshold``.

    TODO(Member 2): M2-1
    Search integer pairs with bands * rows <= max_hashes. A standard starting
    point is the smallest r such that threshold^r is near 0.5, then set
    bands so the OR construction hits ``target_probability``.
    Return ``(bands, rows)``. Document the chosen K = bands * rows.
    """
    raise NotImplementedError("TODO(Member 2): M2-1 choose_bands_and_rows")


def fit_lsh(df: DataFrame, config: PipelineConfig) -> Model:
    """Fit the Spark MinHash baseline with one hash per band.

    TODO(Member 2): M2-2
    Use ``MinHashLSH(inputCol="features", outputCol="hashes",
    numHashTables=config.num_hash_tables)`` for the baseline.
    Custom banding needs b * r signatures from a separate fit.
    Input vectors come from Member 1's ``features`` column.
    """
    raise NotImplementedError("TODO(Member 2): M2-2 fit_lsh")


def similarity_to_distance(similarity: float) -> float:
    """Convert a Jaccard similarity cutoff into the distance Spark expects.

    TODO(Member 2): M2-3
    Return ``1 - similarity``.
    """
    raise NotImplementedError("TODO(Member 2): M2-3 similarity_to_distance")


def all_pairs_similarity_join(
    model: Model,
    df: DataFrame,
    config: PipelineConfig,
) -> DataFrame:
    """Candidate pairs whose distance is within the configured threshold.

    TODO(Member 2): M2-3
    Call ``model.approxSimilarityJoin(df, df, distance_threshold, distCol="distance")``.
    Filter to ``datasetA.id < datasetB.id``.
    Return columns ``id_a``, ``id_b``, ``distance``, and ``similarity``
    (similarity = 1 - distance for Jaccard).
    """
    raise NotImplementedError("TODO(Member 2): M2-3 all_pairs_similarity_join")


def approx_knn(
    model: Model,
    df: DataFrame,
    config: PipelineConfig,
) -> DataFrame:
    """k nearest candidates for the row whose id is ``config.query_id``.

    TODO(Member 2): M2-4
    Collect that single query vector on the driver (one row, not the corpus)
    and call ``model.approxNearestNeighbors(df, key, config.neighbors_k)``.
    Drop the query itself from the neighbor list.
    """
    raise NotImplementedError("TODO(Member 2): M2-4 approx_knn")


def run_lsh(spark: SparkSession, features: DataFrame, config: PipelineConfig) -> dict[str, str]:
    """Fit LSH, write join pairs and a k-NN list, and return their paths.

    The Spark baseline uses b hash functions. Custom banding will use b * r.
    """
    bands, rows = choose_bands_and_rows(config.similarity_threshold)
    config.num_hash_tables = bands
    config.rows_per_band = rows
    model = fit_lsh(features, config)
    pairs = all_pairs_similarity_join(model, features, config)
    neighbors = approx_knn(model, features, config)

    pairs_path = str(config.output_dir / "candidate_pairs")
    knn_path = str(config.output_dir / "knn")
    pairs.write.mode("overwrite").parquet(pairs_path)
    neighbors.write.mode("overwrite").parquet(knn_path)
    return {"pairs": pairs_path, "knn": knn_path}
