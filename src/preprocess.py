"""Distributed ingestion, shingling, and feature construction.

Owner: Member 1 — Data Engineering & Representation Lead
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from src.config import PipelineConfig


def build_spark_session(config: PipelineConfig) -> SparkSession:
    """Create the SparkSession used by every stage.

    TODO(Member 1): Add shuffle partitions, a Kryo or default serializer choice,
    and driver/executor memory settings that match the real corpus size.
    Keep ``config.shuffle_partitions`` as the source of truth.
    """
    return (
        SparkSession.builder.appName(config.app_name)
        .master(config.master)
        .config("spark.sql.shuffle.partitions", str(config.shuffle_partitions))
        .getOrCreate()
    )


def load_raw_corpus(spark: SparkSession, config: PipelineConfig) -> DataFrame:
    """Load raw text into a DataFrame with ``config.id_column`` and ``config.text_column``.

    TODO(Member 1): M1-1
    - Read CSV and JSON from ``config.input_path``.
    - Quora Question Pairs: explode question1 and question2 into one row per
      question, and keep the original pair id for Member 3's labels.
    - ArXiv abstracts: map the abstract field into ``text``.
    - The checked-in sample is ``data/samples/questions_sample.csv``.
    - Cast ids to string so later joins do not mix int and string keys.
    """
    raise NotImplementedError("TODO(Member 1): M1-1 load_raw_corpus")


def normalize_text(df: DataFrame, config: PipelineConfig) -> DataFrame:
    """Return ``df`` with a cleaned ``text`` column.

    TODO(Member 1): M1-2
    Lowercase, strip URLs and punctuation, and collapse whitespace with Spark
    SQL functions (``regexp_replace``, ``trim``, ``lower``). Do not use a
    Python UDF unless a transformation cannot be expressed in SQL.
    """
    raise NotImplementedError("TODO(Member 1): M1-2 normalize_text")


def tokenize(df: DataFrame, config: PipelineConfig) -> DataFrame:
    """Add a ``tokens`` column of words from the cleaned text.

    TODO(Member 1): Split ``text`` with ``pyspark.ml.feature.RegexTokenizer``
    or ``split``. This is the tokenization step before k-shingles. Do not
    collect documents to the driver.
    """
    raise NotImplementedError("TODO(Member 1): tokenize")


def add_shingles(df: DataFrame, config: PipelineConfig) -> DataFrame:
    """Add an array column ``shingles`` of k-grams.

    TODO(Member 1): M1-3
    For a normalized string, emit the set of contiguous k-shingles
    (``config.shingle_size``). Example for k=3 on "abcd": ["abc", "bcd"].
    Deduplicate shingles inside each document so the row represents a set,
    which is what Jaccard and MinHash assume.
    """
    raise NotImplementedError("TODO(Member 1): M1-3 add_shingles")


def vectorize(df: DataFrame, config: PipelineConfig) -> DataFrame:
    """Add a ``features`` column of Spark ML vectors.

    TODO(Member 1): M1-4
    - ``hashing_tf``: ``pyspark.ml.feature.HashingTF`` on ``shingles``,
      ``numFeatures=config.num_features``, ``binary=True`` for sets.
    - ``count_vectorizer``: ``CountVectorizer`` fit on the corpus, also binary.
    - ``dense``: optional pre-trained embedding column assembled with
      ``VectorAssembler``. Use this path only for BucketedRandomProjectionLSH.
    MinHashLSH needs non-negative set-like vectors. Do not L2-normalize them.
    """
    raise NotImplementedError("TODO(Member 1): M1-4 vectorize")


def write_feature_matrix(df: DataFrame, config: PipelineConfig) -> str:
    """Cache the feature matrix and write partitioned Parquet.

    TODO(Member 1): M1-5
    Repartition before the write, ``cache`` (or ``persist``) the frame that
    Member 2 will scan twice for the self-join, and write to
    ``config.processed_dir / "features"``. Return that path as a string.
    """
    raise NotImplementedError("TODO(Member 1): M1-5 write_feature_matrix")


def run_preprocess(spark: SparkSession, config: PipelineConfig) -> str:
    """Run Member 1's stage and return the Parquet path."""
    raw = load_raw_corpus(spark, config)
    cleaned = normalize_text(raw, config)
    tokens = tokenize(cleaned, config)
    shingled = add_shingles(tokens, config)
    featured = vectorize(shingled, config)
    return write_feature_matrix(featured, config)


def read_feature_matrix(spark: SparkSession, config: PipelineConfig) -> DataFrame:
    """Read the Parquet matrix produced by ``run_preprocess``."""
    path = str(config.processed_dir / "features")
    return spark.read.parquet(path)
