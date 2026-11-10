"""Shared pipeline settings.

This module is shared scaffolding. Algorithm work stays in the member modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PipelineConfig:
    """Knobs for ingestion, LSH, and evaluation.

    Spark's ``approxSimilarityJoin`` threshold is a *distance*, not a
    similarity. For Jaccard, distance = 1 - similarity. Member 2 converts
    ``similarity_threshold`` before calling Spark.
    """

    app_name: str = "distributed-lsh"
    master: str = "local[*]"
    input_path: Path = PROJECT_ROOT / "data" / "samples" / "questions_sample.csv"
    pairs_path: Path = PROJECT_ROOT / "data" / "samples" / "duplicate_pairs.csv"
    id_column: str = "id"
    text_column: str = "text"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    output_dir: Path = PROJECT_ROOT / "data" / "output"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"

    driver_memory: str = "4g"  # local mode: the driver is the only executor
    min_chars: int = 0  # arXiv run uses 100; the short sample questions need 0
    drop_exact_duplicates: bool = True
    inject_frac: float = 0.01  # share of rows cloned as synthetic near-duplicates
    max_dup_drop_rate: float = 0.30  # each clone drops uniform [0, this) of its words

    shingle_size: int = 3  # word shingles
    num_features: int = 1 << 18
    vectorizer: str = "hashing_tf"  # hashing_tf | count_vectorizer

    num_hash_tables: int = 5  # bands b; Spark uses b * r tables to generate signatures for custom banding
    rows_per_band: int = 2  # rows r in the AND construction; K = b * r
    band_grid: tuple[int, ...] = (5, 10, 20)
    row_grid: tuple[int, ...] = (2, 3, 4)
    similarity_threshold: float = 0.8
    neighbors_k: int = 5
    query_id: str = "1"

    ground_truth_sample_size: int = 200
    seed: int = 36
    shuffle_partitions: int = 8
