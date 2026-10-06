"""Entry point for the distributed LSH mini-project.

Stages:
  preprocess  -> Member 1
  lsh         -> Member 2
  evaluate    -> Member 3
  all         -> preprocess, then LSH, then evaluation

Example:
  python main.py --stage all --input data/samples/questions_sample.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.config import PROJECT_ROOT, PipelineConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed LSH mini-project pipeline")
    parser.add_argument(
        "--stage",
        choices=("preprocess", "lsh", "evaluate", "all"),
        default="all",
        help="Which stage to run",
    )
    parser.add_argument("--input", type=Path, help="Raw CSV or JSON corpus")
    parser.add_argument("--master", help="Spark master URL, default local[*]")
    parser.add_argument("--threshold", type=float, help="Jaccard similarity threshold")
    parser.add_argument("--bands", type=int, help="numHashTables / OR bands b")
    parser.add_argument("--rows", type=int, help="AND rows r inside a band")
    parser.add_argument("--k", type=int, help="Neighbors for approxNearestNeighbors")
    parser.add_argument("--query-id", help="Document id to probe")
    parser.add_argument(
        "--family",
        choices=("minhash", "random_projection"),
        help="LSH family",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig()
    if args.input:
        config.input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    if args.master:
        config.master = args.master
    if args.threshold is not None:
        config.similarity_threshold = args.threshold
    if args.bands is not None:
        config.num_hash_tables = args.bands
    if args.rows is not None:
        config.rows_per_band = args.rows
    if args.k is not None:
        config.neighbors_k = args.k
    if args.query_id:
        config.query_id = args.query_id
    if args.family:
        config.lsh_family = args.family
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    return config


def main(argv: list[str] | None = None) -> int:
    from src.evaluate import run_evaluate
    from src.lsh_engine import run_lsh
    from src.preprocess import build_spark_session, read_feature_matrix, run_preprocess

    args = parse_args(argv)
    config = config_from_args(args)
    spark = build_spark_session(config)
    spark.sparkContext.setLogLevel("WARN")

    def features():
        path = config.processed_dir / "features"
        if not path.exists():
            raise SystemExit("Feature Parquet is missing. Run --stage preprocess first.")
        return read_feature_matrix(spark, config)

    try:
        if args.stage in ("preprocess", "all"):
            print(run_preprocess(spark, config))

        lsh_seconds = 0.0
        if args.stage in ("lsh", "all"):
            started = time.perf_counter()
            print(run_lsh(spark, features(), config))
            lsh_seconds = time.perf_counter() - started

        if args.stage in ("evaluate", "all"):
            pairs_path = config.output_dir / "candidate_pairs"
            if not pairs_path.exists():
                raise SystemExit("Candidate pairs are missing. Run --stage lsh first.")
            candidates = spark.read.parquet(str(pairs_path))
            knn_path = config.output_dir / "knn"
            if not knn_path.exists():
                raise SystemExit("k-NN results are missing. Run --stage lsh first.")
            neighbors = spark.read.parquet(str(knn_path))
            # TODO(Member 3): replace 0.0 with the wall time of brute_force_pairs.
            print(
                run_evaluate(
                    spark,
                    features(),
                    candidates,
                    neighbors,
                    config,
                    exact_seconds=0.0,
                    lsh_seconds=lsh_seconds,
                )
            )
    except NotImplementedError as exc:
        print(f"\nBlocked: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
