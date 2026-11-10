# Distributed LSH mini-project

Spark pipeline for near-duplicate search. Unfinished work is marked `TODO(Member N)` in the owning module.

## Project Overview

This project implements a distributed near-duplicate search pipeline using Apache Spark and Locality-Sensitive Hashing (LSH). The goal is to efficiently identify similar or near-duplicate text documents without comparing every possible pair exactly.

The pipeline consists of three main stages:

* **Preprocessing:** load and normalize text, tokenize documents, generate k-shingles, convert text into feature vectors, and store the processed data in Parquet.
* **LSH-based search:** configure LSH parameters, build the approximate similarity index, perform similarity joins, and retrieve nearest neighbors.
* **Evaluation:** compare LSH results with an exact similarity baseline and measure precision, recall, Recall@k, execution speedup, and Spark performance metrics.

The project also evaluates different LSH configurations by varying the number of bands (`b`) and rows per band (`r`) while keeping `K = b × r`. The theoretical probability of detecting a candidate pair with similarity `s` is:

`P = 1 - (1 - s^r)^b`

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── samples/
│       ├── questions_sample.csv
│       └── duplicate_pairs.csv
├── src/
│   ├── preprocess.py
│   ├── lsh_engine.py
│   └── evaluate.py
└── README.md
```

### Responsibilities

1. **Member 1 — Preprocessing**

   * File: `src/preprocess.py`
   * Load text data
   * Normalize and tokenize text
   * Generate k-shingles
   * Create feature vectors using HashingTF / CountVectorizer or dense vectors
   * Save processed data as Parquet

2. **Member 2 — LSH Engine**

   * File: `src/lsh_engine.py`
   * Choose `b` and `r` such that `K = b * r`
   * Use MinHash signatures for custom Jaccard banding, and use Spark MinHash for the baseline join and k-NN
   * Perform similarity join
   * Perform k-NN search

3. **Member 3 — Evaluation**

   * File: `src/evaluate.py`
   * Build an exact similarity baseline
   * Calculate precision and recall
   * Calculate Recall@k
   * Measure speedup
   * Collect Spark performance metrics
   * Sweep different `b` and `r` configurations
   * Compare experimental results against:

   `P = 1 - (1 - s^r)^b`

## Setup

Create a virtual environment and install the required dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Put the full corpus in:

```text
data/raw/
```

The sample dataset is:

```text
data/samples/questions_sample.csv
```

The sample columns are:

```text
id
text
```

Ground-truth duplicate pairs are stored in:

```text
data/samples/duplicate_pairs.csv
```

with the following columns:

```text
id_a
id_b
is_duplicate
```

## Running the Project

### 1. Run Preprocessing — Member 1

Run only the preprocessing stage:

```bash
python main.py --stage preprocess --input data/samples/questions_sample.csv
```

This stage should:

1. Read the input CSV.
2. Normalize the text.
3. Tokenize the text.
4. Generate k-shingles.
5. Convert shingles into feature vectors.
6. Save the processed dataset in Parquet format.

The resulting Parquet data will be used by the LSH stage.

---

### 2. Run LSH Search — Member 2

Run the LSH stage:

```bash
python main.py --stage lsh --input data/samples/questions_sample.csv
```

This stage should:

1. Load the preprocessed vectors.
2. Configure `b` and `r`.
3. Ensure:

```text
K = b × r
```

4. Build the selected LSH model.
5. Perform similarity join.
6. Perform k-NN / nearest-neighbor search.
7. Save or return the LSH search results.

For `MinHashLSH`, the distance is **Jaccard distance**:

```text
Jaccard distance = 1 - Jaccard similarity
```

Therefore, when using a similarity threshold, remember that Spark's LSH join API expects a **distance threshold**, not a similarity threshold.

For example:

```text
similarity ≥ 0.8
```

corresponds to:

```text
distance ≤ 0.2
```

---

### 3. Run Evaluation — Member 3

Run the evaluation stage:

```bash
python main.py --stage evaluate --input data/samples/questions_sample.csv
```

The evaluation stage compares the approximate LSH search against an exact baseline.

It should report metrics such as:

```text
Precision
Recall
Recall@k
Exact search time
LSH search time
Speedup
Spark execution metrics
```

The ground-truth labels are loaded from:

```text
data/samples/duplicate_pairs.csv
```

where:

```text
id_a       First document ID
id_b       Second document ID
is_duplicate   Ground-truth duplicate label
```

---

### 4. Run the Complete Pipeline

To run all stages sequentially:

```bash
python main.py --stage all --input data/samples/questions_sample.csv
```

This is the recommended command for testing the complete project.

The pipeline follows:

```text
Raw CSV
   │
   ▼
Preprocessing
   │
   ▼
Feature Vectors / Parquet
   │
   ▼
LSH Engine
   │
   ▼
Candidate / Nearest Neighbors
   │
   ▼
Evaluation
   │
   ▼
Precision / Recall / Recall@k / Speedup
```

## LSH Parameter Sweep

For the LSH experiment, Member 3 evaluates different combinations of `b` and `r` while maintaining:

```text
K = b × r
```

For a pair of documents with Jaccard similarity `s`, the theoretical probability that LSH identifies the pair as a candidate is:

```text
P = 1 - (1 - s^r)^b
```

Example:

```text
K = 20

b = 5,  r = 4
b = 10, r = 2
b = 20, r = 1
```

The experiment should compare the theoretical probability with the experimentally observed candidate/duplicate detection rate.

## Important Notes

Spark's LSH similarity join uses a **distance threshold**.

For Jaccard similarity:

```text
distance = 1 - similarity
```

Therefore:

| Desired similarity | Spark distance threshold |
| -----------------: | -----------------------: |
|               0.90 |                     0.10 |
|               0.80 |                     0.20 |
|               0.70 |                     0.30 |
|               0.60 |                     0.40 |

Use the full corpus in `data/raw/` for the final experiment and the sample dataset for development and debugging.

## Quick Commands

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Member 1

```bash
python main.py --stage preprocess --input data/samples/questions_sample.csv
```

### Member 2

```bash
python main.py --stage lsh --input data/samples/questions_sample.csv
```

### Member 3

```bash
python main.py --stage evaluate --input data/samples/questions_sample.csv
```

### Complete pipeline

```bash
python main.py --stage all --input data/samples/questions_sample.csv
```
