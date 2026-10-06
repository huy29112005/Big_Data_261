"""Ground truth, quality metrics, S-curve check, and Spark profiling.

Owner: Member 3 — Benchmarks and Validation Lead

FAISS or scikit-learn may be used only on the driver, and only on a sample,
to cross-check the brute-force baseline. The graded join itself stays in Spark.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from src.config import PipelineConfig


def brute_force_pairs(df: DataFrame, config: PipelineConfig) -> DataFrame:
    """Exact pairs on a validation sample.

    TODO(Member 3): M3-1
    Sample at most ``config.ground_truth_sample_size`` rows with
    ``config.seed``. For MinHash features, compute Jaccard similarity between
    shingle sets (or between binary vectors). For random projection, compute
    Euclidean distance. Emit ``id_a``, ``id_b``, ``similarity`` with
    ``id_a < id_b``. This is the O(n^2) reference the LSH join must beat.
    Keep the Cartesian product on the sample only.
    """
    raise NotImplementedError("TODO(Member 3): M3-1 brute_force_pairs")


def precision_recall(
    candidates: DataFrame,
    ground_truth: DataFrame,
    config: PipelineConfig,
) -> dict[str, float]:
    """Score LSH candidates against exact pairs above the similarity threshold.

    TODO(Member 3): M3-2
    Treat ground-truth pairs with similarity >= ``config.similarity_threshold``
    as relevant. Return a dict with keys
    ``precision``, ``recall``, ``false_positive_rate``, ``false_negative_rate``.
    Also accept the labeled file ``config.pairs_path`` when the corpus is the
    checked-in sample (column ``is_duplicate``).
    """
    raise NotImplementedError("TODO(Member 3): M3-2 precision_recall")


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k retrieved ids that sit in ``relevant_ids``, over k.

    TODO(Member 3): M3-2
    Use the first ``k`` ids in ``retrieved_ids``. Return 0 when k is 0.
    """
    raise NotImplementedError("TODO(Member 3): M3-2 recall_at_k")


def time_and_profile(
    spark: SparkSession,
    exact_seconds: float,
    lsh_seconds: float,
) -> dict[str, float]:
    """Speedup and a snapshot of Spark listener metrics.

    TODO(Member 3): M3-3
    speedup = exact_seconds / lsh_seconds when lsh_seconds > 0.
    From ``spark.sparkContext.statusTracker`` or the last job's stage metrics,
    record shuffle read/write bytes and the longest stage duration.
    Executor memory can come from the Spark UI REST endpoint on port 4040
    when it is up; if it is not, record null.
    """
    raise NotImplementedError("TODO(Member 3): M3-3 time_and_profile")


def s_curve_table(
    bands_rows: list[tuple[int, int]],
    similarities: list[float],
    empirical: dict[tuple[int, int, float], float],
) -> list[dict[str, float]]:
    """Rows comparing theory and measured collision rates.

    TODO(Member 3): M3-4
    For each (b, r, s), call ``src.lsh_engine.collision_probability``.
    ``empirical[(b, r, s)]`` is the measured rate. Each result dict needs
    ``bands``, ``rows``, ``similarity``, ``theoretical``, ``empirical``.
    """
    raise NotImplementedError("TODO(Member 3): M3-4 s_curve_table")


def score_knn(
    neighbors: DataFrame,
    ground_truth: DataFrame,
    config: PipelineConfig,
) -> float:
    """Recall@k for the probe ``config.query_id``.

    TODO(Member 3): Relevant ids are ground-truth neighbors of ``query_id``
    at or above ``config.similarity_threshold``. Pass the retrieved id list
    to ``recall_at_k`` with ``k=config.neighbors_k``.
    """
    raise NotImplementedError("TODO(Member 3): score_knn")


def parameter_sweep(
    features: DataFrame,
    ground_truth: DataFrame,
    config: PipelineConfig,
) -> list[dict[str, float]]:
    """Refit LSH over ``band_grid`` x ``row_grid`` and compare with theory.

    TODO(Member 3): For each (b, r), set ``config.num_hash_tables`` and
    ``config.rows_per_band``, call ``fit_lsh`` and ``all_pairs_similarity_join``,
    and measure the empirical collision rate against ``ground_truth``.
    Feed those rates to ``s_curve_table``, then ``plot_s_curve`` under
    ``config.figures_dir / "s_curve.png"``. Similarities to test: 0.2, 0.5, 0.8.
    """
    raise NotImplementedError("TODO(Member 3): parameter_sweep")


def plot_s_curve(rows: list[dict[str, float]], output_path: Path) -> None:
    """Write the empirical-vs-theoretical S-curve figure.

    TODO(Member 3): M3-5
    Use Matplotlib. One line per (bands, rows) setting. Save to
    ``output_path`` (default ``reports/figures/s_curve.png``).
    """
    raise NotImplementedError("TODO(Member 3): M3-5 plot_s_curve")


def run_evaluate(
    spark: SparkSession,
    features: DataFrame,
    candidates: DataFrame,
    neighbors: DataFrame,
    config: PipelineConfig,
    exact_seconds: float,
    lsh_seconds: float,
) -> dict[str, float]:
    """Ground truth, join quality, Recall@k, speedup, and the S-curve sweep."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    truth = brute_force_pairs(features, config)
    metrics = precision_recall(candidates, truth, config)
    metrics["recall_at_k"] = score_knn(neighbors, truth, config)
    metrics.update(time_and_profile(spark, exact_seconds, lsh_seconds))
    parameter_sweep(features, truth, config)
    return metrics
