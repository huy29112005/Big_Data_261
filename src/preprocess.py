"""Distributed ingestion, shingling, and feature construction.

Owner: Member 1 — Data Engineering & Representation Lead

Ported from ``big-data.ipynb`` (arXiv run on Kaggle). Output columns:
``id, source_id, is_synthetic, n_shingles, features`` plus ``text`` and,
for arXiv, ``primary_category``.
"""

from __future__ import annotations

import random

from pyspark.ml.feature import CountVectorizer, HashingTF, NGram
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.config import PipelineConfig

# Only the arXiv fields we use; skipping schema inference saves a full pass over the JSON.
ARXIV_SCHEMA = T.StructType([
    T.StructField("id", T.StringType()),
    T.StructField("title", T.StringType()),
    T.StructField("abstract", T.StringType()),
    T.StructField("categories", T.StringType()),
    T.StructField("update_date", T.StringType()),
])
WITHDRAWN_RE = r"^.{0,200}(has been|was|is being) withdrawn"


def build_spark_session(config: PipelineConfig) -> SparkSession:
    """Create the SparkSession used by every stage.

    ``spark.driver.memory`` only applies if no session exists yet in this process.
    """
    return (
        SparkSession.builder.appName(config.app_name)
        .master(config.master)
        .config("spark.driver.memory", config.driver_memory)
        .config("spark.sql.shuffle.partitions", str(config.shuffle_partitions))
        .getOrCreate()
    )


def load_raw_corpus(spark: SparkSession, config: PipelineConfig) -> DataFrame:
    """Load raw text into a DataFrame with ``id`` and ``text`` columns.

    - ``.json``: arXiv metadata snapshot; ``abstract`` becomes ``text`` and the
      first entry of ``categories`` becomes ``primary_category``.
    - ``.csv``: needs ``config.id_column`` and ``config.text_column`` (the samples).
    """
    path = str(config.input_path)
    if path.endswith(".json"):
        # shortcut: re-parses the JSON every run (~1 min for 5.6 GB); cache as Parquet if runs get frequent.
        df = (spark.read.schema(ARXIV_SCHEMA).json(path)
              .select("id",
                      F.col("abstract").alias("text"),
                      F.split("categories", " ")[0].alias("primary_category")))
    else:
        df = (spark.read.option("header", True).option("multiLine", True).option("escape", '"')
              .csv(path)
              .select(F.col(config.id_column).alias("id"), F.col(config.text_column).alias("text")))
    return df.withColumn("id", F.col("id").cast("string")).filter(F.col("id").isNotNull())


def normalize_text(df: DataFrame, config: PipelineConfig) -> DataFrame:
    """Clean ``text``, drop junk rows, and make ``id`` unique.

    Lowercase, strip inline LaTeX and ``\\commands``, punctuation, extra
    whitespace. Drop texts shorter than ``config.min_chars`` and arXiv
    "withdrawn" stubs. Optionally drop exact duplicate texts (keep the smallest
    id), then keep one row per id, so Member 2's ``id_a < id_b`` filter is sound.
    """
    x = F.lower(F.col("text"))
    x = F.regexp_replace(x, r"\$[^$]*\$", " ")
    x = F.regexp_replace(x, r"\\[a-z]+", " ")
    x = F.regexp_replace(x, r"[^a-z0-9\s]", " ")
    x = F.trim(F.regexp_replace(x, r"\s+", " "))

    cleaned = (df.withColumn("text", x)
                 .filter(F.col("text").isNotNull() & (F.col("text") != ""))
                 .filter(F.length("text") >= config.min_chars)
                 .filter(~F.col("text").rlike(WITHDRAWN_RE)))

    # Deterministic: struct ordering compares ``id`` first, so reruns keep the same rows.
    rest = [c for c in cleaned.columns if c not in ("id", "text")]
    if config.drop_exact_duplicates:
        cleaned = (cleaned.groupBy("text")
                          .agg(F.min(F.struct("id", *rest)).alias("r"))
                          .select("r.id", *[f"r.{c}" for c in rest], "text"))
    return (cleaned.groupBy("id")
                   .agg(F.min(F.struct("text", *rest)).alias("r"))
                   .select("id", "r.text", *[f"r.{c}" for c in rest]))


def inject_near_duplicates(df: DataFrame, config: PipelineConfig) -> DataFrame:
    """Add ``source_id`` / ``is_synthetic`` and clone ``config.inject_frac`` of rows.

    arXiv has no duplicate labels, so clones are Member 3's known positives.
    Each clone ``<id>_dup`` deletes a per-row rate in ``[0, max_dup_drop_rate)``
    of its words (stored in ``drop_rate``), so the clones span similarities on
    both sides of the join threshold instead of clustering at one value.
    """
    seed = config.seed

    def mutate(text: str, row_id: str, rate: float) -> str:
        rng = random.Random(f"{seed}-{row_id}")
        kept = [w for w in text.split(" ") if rng.random() >= rate]
        return " ".join(kept) if kept else text

    mutate_udf = F.udf(mutate, T.StringType())
    base = (df.withColumn("source_id", F.lit(None).cast("string"))
              .withColumn("is_synthetic", F.lit(False))
              .withColumn("drop_rate", F.lit(None).cast("double")))
    if config.inject_frac <= 0:
        return base

    pick = (F.abs(F.xxhash64("id", F.lit(seed + 1))) % 10000) < int(config.inject_frac * 10000)
    rate = (F.abs(F.xxhash64("id", F.lit(seed + 2))) % 10000) / 10000.0 * config.max_dup_drop_rate
    dups = (df.filter(pick)
              .withColumn("drop_rate", rate)
              .withColumn("source_id", F.col("id"))
              .withColumn("text", mutate_udf("text", "id", "drop_rate"))
              .withColumn("id", F.concat(F.col("id"), F.lit("_dup")))
              .withColumn("is_synthetic", F.lit(True)))
    return base.unionByName(dups)


def tokenize(df: DataFrame, config: PipelineConfig) -> DataFrame:
    """Add a ``tokens`` column; ``text`` is already single-space separated."""
    return df.withColumn("tokens", F.split("text", " "))


def add_shingles(df: DataFrame, config: PipelineConfig) -> DataFrame:
    """Add ``shingles``: the set of contiguous k-word shingles (k = ``config.shingle_size``).

    Rows with no shingle (fewer than k words) are dropped: ``MinHashLSH``
    rejects all-zero vectors.
    """
    d = NGram(n=config.shingle_size, inputCol="tokens", outputCol="shingles").transform(df)
    return (d.withColumn("shingles", F.array_distinct("shingles"))
             .withColumn("n_shingles", F.size("shingles"))
             .filter(F.col("n_shingles") > 0))


def vectorize(df: DataFrame, config: PipelineConfig) -> DataFrame:
    """Add a binary ``features`` vector (a set indicator, as MinHash expects)."""
    if config.vectorizer == "hashing_tf":
        return HashingTF(inputCol="shingles", outputCol="features",
                         numFeatures=config.num_features, binary=True).transform(df)
    if config.vectorizer == "count_vectorizer":
        model = CountVectorizer(inputCol="shingles", outputCol="features",
                                vocabSize=config.num_features, binary=True).fit(df)
        return model.transform(df)
    raise ValueError(f"Unknown vectorizer: {config.vectorizer!r}")


def write_feature_matrix(df: DataFrame, config: PipelineConfig) -> str:
    """Write the feature matrix as Parquet, partitioned by ``id``; return the path."""
    path = str(config.processed_dir / "features")
    cols = ["id", "source_id", "is_synthetic", "drop_rate", "primary_category", "text",
            "n_shingles", "features"]
    (df.select(*[c for c in cols if c in df.columns])
       .repartition(config.shuffle_partitions, "id")
       .write.mode("overwrite").parquet(path))
    return path


def run_preprocess(spark: SparkSession, config: PipelineConfig) -> str:
    """Run Member 1's stage and return the Parquet path."""
    raw = load_raw_corpus(spark, config)
    cleaned = normalize_text(raw, config)
    corpus = inject_near_duplicates(cleaned, config)
    tokens = tokenize(corpus, config)
    shingled = add_shingles(tokens, config)
    featured = vectorize(shingled, config)
    return write_feature_matrix(featured, config)


def read_feature_matrix(spark: SparkSession, config: PipelineConfig) -> DataFrame:
    """Read the Parquet matrix produced by ``run_preprocess``."""
    path = str(config.processed_dir / "features")
    return spark.read.parquet(path)
