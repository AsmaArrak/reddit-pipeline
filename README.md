
<div align="center">

# 🔍 Reddit Search + Label Pipeline

DuckDB → Postgres (BM25) → SentenceTransformer embeddings → hybrid search  
+ optional **gold labels** for arbitrary categories (`category_1`, `label_group_1`, …)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![DuckDB](https://img.shields.io/badge/db-DuckDB-green)
![Postgres](https://img.shields.io/badge/db-Postgres-orange)
![Status](https://img.shields.io/badge/status-experimental-purple)

</div>

---

## 🚀 Features

- Ingest Reddit-style posts into **DuckDB** (`bronze.reddit_raw`).
- Build a cleaned, deduplicated **silver table** (`silver.posts`).
- Export to **Postgres** and build a **BM25** full-text index.
- Add **SentenceTransformer** embeddings (no `pgvector` required).
- Run **hybrid search** (BM25 + cosine similarity).
- Store arbitrary annotations in **`gold.label_events`**  
  (`category_1`, `category_2`, `label_group_1`, …; single or multiple annotators).

---

## 🧩 Requirements

- Python 3.10+
- DuckDB
- PostgreSQL 13+
- A SentenceTransformer model  
  (default: `sentence-transformers/all-MiniLM-L6-v2`)

`requirements.txt`:

```txt
duckdb
pandas
numpy
psycopg2-binary
sentence-transformers
openpyxl
```

Install:

```bash
pip install -r requirements.txt
```

---

## ⚡ Quick start

```bash
# 1. Build bronze + silver + Postgres BM25
python reddit_pipeline.py build

# 2. Add embeddings
python reddit_pipeline.py embed

# 3. Run a hybrid search
python reddit_pipeline.py search "your query here" 20

# 4. (optional) Load annotations into gold.label_events
python reddit_pipeline.py gold
```

---

All paths are controlled by `CONFIG` in `reddit_pipeline.py`.

---

## ⚙️ Configuration

### Core config

```python
CONFIG = {
    "RAW_POSTS_PATH": r"data/raw/posts_raw.xlsx",          # raw posts (.csv / .xlsx)
    "DUCKDB_PATH":   r"data/warehouse/reddit.duckdb",      # DuckDB file
    "EXPORT_DIR":    r"data/exports",                      # Postgres export dir
    "PG_DSN":        "postgresql://USER:PASS@localhost:5432/DB_NAME",
    "MODEL_NAME":    "sentence-transformers/all-MiniLM-L6-v2",
}
```
---

## 🧱 Pipeline overview

### 1. Bronze (raw → bronze.reddit_raw)

- Reads `CONFIG["RAW_POSTS_PATH"]` (.csv or .xlsx).
- Normalizes column names via `CSV_COL_MAP`.
- Adds metadata: `run_id`, `ingested_at`, `source_file`.
- Writes to `bronze.reddit_raw` in DuckDB (table is cleared on each run).

### 2. Silver (bronze.reddit_raw → silver.posts)

- Deduplicates by `post_id` (latest `created_utc` / `ingested_at` wins).
- Cleans text:
  - empty strings → `NULL`
  - `[deleted]`, `[removed]` → `NULL`
- Writes to `silver.posts` with:

  - `post_id`
  - `subreddit`
  - `author_id`
  - `created_utc`
  - `title_clean`
  - `text_clean`
  - `url`
  - `score`
  - `num_comments`
  - `flair`
  - `run_id`
  - `source_file`

### 3. Export to Postgres

- Exports `silver.posts` to `EXPORT_DIR/silver_posts_psql.csv`.
- Creates schema/table:

```sql
CREATE SCHEMA IF NOT EXISTS serving_test;

CREATE TABLE IF NOT EXISTS serving_test.posts_search_test (
    post_id        TEXT PRIMARY KEY,
    subreddit      TEXT,
    author_id      TEXT,
    created_utc    TEXT,
    title_clean    TEXT,
    text_clean     TEXT,
    url            TEXT,
    score          TEXT,
    num_comments   TEXT,
    flair          TEXT,
    run_id         TEXT,
    source_file    TEXT,
    lexical        tsvector,
    embedding_text TEXT
);
```

- Loads CSV into `serving_test.posts_search_test`.
- Builds BM25 vector + GIN index:

```sql
UPDATE serving_test.posts_search_test
SET lexical =
    to_tsvector(
        'english',
        COALESCE(title_clean, '') || ' ' || COALESCE(text_clean, '')
    );

CREATE INDEX IF NOT EXISTS idx_posts_search_test_lexical
ON serving_test.posts_search_test
USING GIN (lexical);
```

### 4. Embeddings

`python reddit_pipeline.py embed`:

- Reads all rows from `silver.posts`.
- Builds text: `title_clean + "\n\n" + text_clean`.
- Encodes with `SentenceTransformer(CONFIG["MODEL_NAME"])`.
- Normalizes embeddings.
- Stores as strings (e.g. `"[0.01,0.02,...]"`) in  
  `serving_test.posts_search_test.embedding_text`.

### 5. Hybrid search

`python reddit_pipeline.py search "query" [top_k]`:

1. Encodes the query to a normalized vector.
2. Asks Postgres for BM25 top-N candidates (lexical match + `ts_rank_cd`).
3. Parses `embedding_text` into NumPy arrays.
4. Computes cosine similarity with the query.
5. Normalizes BM25 scores and cosine scores.
6. Combines them:

   ```text
   hybrid = bm25_weight * bm25_norm + (1 - bm25_weight) * cosine_norm
   ```

7. Prints ranked results with `post_id`, subreddit, title, snippet and scores.

---

## 🥇 Gold labels

Annotations are stored in a long-format DuckDB table: `gold.label_events`.

```sql
CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS gold.label_events (
    item_type     TEXT,
    item_id       TEXT,
    task_name     TEXT,
    label_value   TEXT,
    annotator_id  TEXT,
    label_source  TEXT,
    confidence    DOUBLE,
    guideline_ver TEXT,
    created_at    TIMESTAMP,
    run_id        TEXT
);
```

### GOLD_CONFIG

Configure how your annotation file is interpreted:

```python
GOLD_CONFIG = {
    "LABELS_PATH": r"data/annotations/labels.csv",  # .csv or .xlsx
    "ITEM_ID_COL": "post_id",                       # must match silver.posts.post_id
    "ITEM_TYPE":   "post",                          # e.g. "post", "comment", "image"

    "ANNOTATOR_COL": None,                          # or "annotator", "__annotator__", ...
    "DEFAULT_ANNOTATOR": None,                      # e.g. "rater_1" if you want a default

    "IGNORE_COLS": [
        "post_id", "Post ID", "id",
        "annotator", "__annotator__", "coder",
        "title", "text", "selftext", "subreddit",
        "created_at", "run_id", "notes",
    ],

    "LABEL_SOURCE":  "human",
    "GUIDELINE_VER": "v1",
}
```

All non-ignored, non-ID, non-annotator columns become separate `task_name`s  
(e.g. `category_1`, `category_2`, `label_group_1`, …).

---

### Example A — single annotator

Annotation file:

```text
post_id,category_1,category_2,label_group_1
1a2b3c,option_a,option_x,label_1
9x8y7z,option_b,option_y,label_3
```

Config:

```python
GOLD_CONFIG = {
    "LABELS_PATH":       r"data/annotations/labels.csv",
    "ITEM_ID_COL":       "post_id",
    "ITEM_TYPE":         "post",
    "ANNOTATOR_COL":     None,
    "DEFAULT_ANNOTATOR": "rater_1",
    "IGNORE_COLS":       ["post_id"],
    "LABEL_SOURCE":      "human",
    "GUIDELINE_VER":     "v1_single",
}
```

This produces rows in `gold.label_events`:

```text
item_type | item_id | task_name     | label_value | annotator_id
--------- |--------|-------------- |------------ |------------
post      | 1a2b3c | category_1    | option_a    | rater_1
post      | 1a2b3c | category_2    | option_x    | rater_1
post      | 1a2b3c | label_group_1 | label_1     | rater_1
post      | 9x8y7z | category_1    | option_b    | rater_1
post      | 9x8y7z | category_2    | option_y    | rater_1
post      | 9x8y7z | label_group_1 | label_3     | rater_1
```

### Example B — multiple annotators

Annotation file:

```text
post_id,annotator,category_1,category_2
1a2b3c,rater_1,option_a,option_x
1a2b3c,rater_2,option_b,option_y
```

Config:

```python
GOLD_CONFIG = {
    "LABELS_PATH":       r"data/annotations/labels_multi.csv",
    "ITEM_ID_COL":       "post_id",
    "ITEM_TYPE":         "post",
    "ANNOTATOR_COL":     "annotator",
    "DEFAULT_ANNOTATOR": None,
    "IGNORE_COLS":       ["post_id"],
    "LABEL_SOURCE":      "human",
    "GUIDELINE_VER":     "v1_multi",
}
```

---

### Loading labels

```bash
python reddit_pipeline.py gold
```

Inspect:

```python
import duckdb
from reddit_pipeline import CONFIG

con = duckdb.connect(CONFIG["DUCKDB_PATH"])
print(con.execute("SELECT * FROM gold.label_events LIMIT 20;").df())
```

---

## 🧪 Debugging

### DuckDB

```bash
python
```

```python
import duckdb
from reddit_pipeline import CONFIG

con = duckdb.connect(CONFIG["DUCKDB_PATH"])
print(con.execute("SELECT COUNT(*) FROM silver.posts").fetchone())
```

### Postgres

```bash
psql "postgresql://USER:PASS@localhost:5432/DB_NAME"
```

```sql
SELECT COUNT(*) FROM serving_test.posts_search_test;
```

---

## 🏁 Summary

- Configure paths + DSN in `CONFIG`.
- Configure annotation behavior in `GOLD_CONFIG`.
- Run: `build → embed → search → gold`.
- Reuse `gold.label_events` for any taxonomy  
  (`category_1`, `category_2`, `label_group_1`, …) without changing pipeline code.
