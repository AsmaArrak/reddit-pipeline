import sys
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
import psycopg2
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer

# ============================================
# CONFIG – EDIT THESE FOR YOUR ENVIRONMENT
# ============================================

CONFIG = {
    # Path to the raw Reddit posts file (.csv or .xlsx)
    "RAW_POSTS_PATH": r"data/raw/raw-data.xlsx",

    # DuckDB file (bronze / silver / gold live here)
    "DUCKDB_PATH":   r"data/warehouse/reddit.duckdb",

    # Directory where CSV exports for Postgres will be written
    "EXPORT_DIR":    r"data/exports",

    # Postgres DSN (update user, password, host, port, dbname)
    # Example: "postgresql://postgres:password@localhost:5432/reddit_db"
    "PG_DSN":        "postgresql://USER:PASSWORD@localhost:5432/DB_NAME",

    # Embedding model for document + query encodings
    "MODEL_NAME":    "sentence-transformers/all-MiniLM-L6-v2",
}

# ===================================================
# GOLD CONFIG – HOW YOUR ANNOTATION FILE IS SHAPED
# ===================================================
# Two common patterns:
#  1) Single annotator file (no annotator column):
#       - set ANNOTATOR_COL = None
#       - optionally set DEFAULT_ANNOTATOR = "rater_1"
#  2) Multi-annotator file:
#       - set ANNOTATOR_COL = name of the annotator column
#       - DEFAULT_ANNOTATOR can stay None

GOLD_CONFIG = {
    # Path to your annotation file (.csv or .xlsx)
    "LABELS_PATH": r"data/annotations/labels.csv",

    # Column that lines up with silver.posts.post_id
    # (e.g., "post_id", "Post ID", "id", etc.)
    "ITEM_ID_COL": "post_id",

    # What kind of item is being labeled (for joins / future extensions)
    "ITEM_TYPE": "post",

    # Name of the annotator column in your file, or None for single-annotator files
    "ANNOTATOR_COL": None,

    # Used only if ANNOTATOR_COL is None or missing
    "DEFAULT_ANNOTATOR": None,  # e.g. "rater_1"

    # Columns that are NOT label tasks (IDs, metadata, free-text notes, etc.)
    # Any column not listed here, and not ITEM_ID_COL / ANNOTATOR_COL,
    # will be treated as a task_name in gold.label_events.
    "IGNORE_COLS": [
        "post_id", "Post ID", "id",          # ID / linkage
        "annotator", "__annotator__", "coder",
        "title", "text", "selftext", "subreddit",
        "created_at", "run_id", "notes",
        "stance_details",
    ],

    # Extra metadata stored with label_events
    "LABEL_SOURCE": "human",
    "GUIDELINE_VER": "v1",
}

# ============================================
# RAW → CANONICAL COLUMN NAME MAP
# ============================================

CSV_COL_MAP = {
    # IDs
    "Post ID":         "post_id",
    "post_id":         "post_id",
    "id":              "post_id",

    # Subreddit
    "Subreddit":       "subreddit",
    "subreddit":       "subreddit",

    # Author
    "Author":          "author",
    "author":          "author",

    # Created time
    "Created (UTC)":   "created_utc",
    "created_utc":     "created_utc",

    # Title / text
    "Title":           "title",
    "title":           "title",
    "Text":            "selftext",
    "selftext":        "selftext",

    # URL
    "URL":             "url",
    "url":             "url",

    # Score / num_comments
    "Score":           "score",
    "score":           "score",
    "Number of Comments": "num_comments",
    "num_comments":    "num_comments",

    # Flair
    "Flair":           "flair",
    "link_flair_text": "flair",
}

# ======================
# DUCKDB HELPERS
# ======================

def setup_duckdb():
    db_path = Path(CONFIG["DUCKDB_PATH"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    return con


def load_bronze(con: duckdb.DuckDBPyConnection):
    raw_path = Path(CONFIG["RAW_POSTS_PATH"])
    print(f"[bronze] reading raw file from {raw_path}")

    # detect extension and read appropriately
    suffix = raw_path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(raw_path, dtype=str, engine="openpyxl")
    elif suffix == ".csv":
        df = pd.read_csv(raw_path, dtype=str, low_memory=False)
    else:
        raise ValueError(f"Unsupported file extension {suffix!r} for RAW_POSTS_PATH")

    # rename columns according to CSV_COL_MAP (only if present)
    rename_map = {src: dst for src, dst in CSV_COL_MAP.items() if src in df.columns}
    df = df.rename(columns=rename_map)

    # ingestion metadata
    df["run_id"]      = "run_" + dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    df["ingested_at"] = dt.datetime.utcnow().isoformat()
    df["source_file"] = str(raw_path)

    # ensure we have post_id and id columns
    if "post_id" not in df.columns:
        id_candidates = [
            c for c in df.columns
            if c.lower().replace(" ", "") in ("id", "postid", "post_id")
        ]
        if id_candidates:
            base_id_col = id_candidates[0]
            print(f"[bronze] using column {base_id_col!r} as post_id")
            df["post_id"] = df[base_id_col].astype(str)
        else:
            raise ValueError(
                f"No ID-like column found. Got columns: {list(df.columns)}"
            )

    if "id" not in df.columns:
        df["id"] = df["post_id"]

    # make sure expected logical columns exist
    for col in ["subreddit", "author", "created_utc", "title", "selftext",
                "url", "score", "num_comments", "flair"]:
        if col not in df.columns:
            df[col] = None

    con.execute("CREATE SCHEMA IF NOT EXISTS bronze;")
    con.execute("""
        CREATE TABLE IF NOT EXISTS bronze.reddit_raw (
            run_id          TEXT,
            ingested_at     TIMESTAMP,
            source_file     TEXT,
            id              TEXT,
            post_id         TEXT,
            subreddit       TEXT,
            author          TEXT,
            created_utc     TIMESTAMP,
            title           TEXT,
            selftext        TEXT,
            url             TEXT,
            num_comments    INTEGER,
            score           INTEGER,
            flair           TEXT
        );
    """)

    # clear old data for repeatable pipeline
    con.execute("DELETE FROM bronze.reddit_raw;")

    con.register("df_bronze_view", df)
    con.execute("""
        INSERT INTO bronze.reddit_raw
        SELECT
            run_id,
            ingested_at::TIMESTAMP,
            source_file,
            id,
            post_id,
            subreddit,
            author,
            TRY_CAST(created_utc AS TIMESTAMP),
            title,
            selftext,
            url,
            TRY_CAST(num_comments AS INTEGER),
            TRY_CAST(score AS INTEGER),
            flair
        FROM df_bronze_view;
    """)

    n = con.execute("SELECT COUNT(*) FROM bronze.reddit_raw").fetchone()[0]
    print(f"[bronze] inserted {n} rows into bronze.reddit_raw")


def build_silver(con: duckdb.DuckDBPyConnection):
    con.execute("CREATE SCHEMA IF NOT EXISTS silver;")
    con.execute("DROP TABLE IF EXISTS silver.posts;")

    con.execute("""
        CREATE TABLE silver.posts AS
        WITH ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY post_id
                    ORDER BY created_utc DESC, ingested_at DESC
                ) AS rn
            FROM bronze.reddit_raw
            WHERE post_id IS NOT NULL
        )
        SELECT
            post_id,
            subreddit,
            author       AS author_id,
            created_utc,
            CASE WHEN title IS NULL OR TRIM(title) = '' THEN NULL ELSE title END AS title_clean,
            CASE
                WHEN selftext IS NULL THEN NULL
                WHEN TRIM(selftext) = '' THEN NULL
                WHEN LOWER(TRIM(selftext)) IN ('[deleted]', '[removed]') THEN NULL
                ELSE selftext
            END AS text_clean,
            url,
            score,
            num_comments,
            flair,
            run_id,
            source_file
        FROM ranked
        WHERE rn = 1;
    """)

    n = con.execute("SELECT COUNT(*) FROM silver.posts").fetchone()[0]
    print(f"[silver] created silver.posts with {n} rows")


def export_silver(con: duckdb.DuckDBPyConnection) -> Path:
    export_dir = Path(CONFIG["EXPORT_DIR"])
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / "silver_posts_psql.csv"
    print(f"[export] writing {out_path}")

    con.execute(f"""
        COPY (
            SELECT
                post_id,
                subreddit,
                author_id,
                created_utc,
                title_clean,
                text_clean,
                url,
                score,
                num_comments,
                flair,
                run_id,
                source_file
            FROM silver.posts
        )
        TO '{out_path}'
        (FORMAT 'csv', HEADER true, DELIMITER '|', QUOTE '"', ESCAPE '"');
    """)
    return out_path

# ======================
# POSTGRES HELPERS
# ======================

PG_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS serving;

CREATE TABLE IF NOT EXISTS serving.posts_search (
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
"""

def init_postgres():
    print("[postgres] connecting...")
    conn = psycopg2.connect(CONFIG["PG_DSN"])
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(PG_SCHEMA_SQL)
    cur.close()
    conn.close()
    print("[postgres] serving.posts_search is ready")


def load_into_postgres(csv_path: Path):
    print(f"[postgres] loading from {csv_path}")
    conn = psycopg2.connect(CONFIG["PG_DSN"])
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("TRUNCATE TABLE serving.posts_search;")

    with csv_path.open("r", encoding="utf-8") as f:
        cur.copy_expert(
            """
            COPY serving.posts_search (
                post_id,
                subreddit,
                author_id,
                created_utc,
                title_clean,
                text_clean,
                url,
                score,
                num_comments,
                flair,
                run_id,
                source_file
            )
            FROM STDIN
            WITH (FORMAT csv, HEADER true, DELIMITER '|', QUOTE '"', ESCAPE '"');
            """,
            f,
        )

    cur.close()
    conn.close()
    print("[postgres] loaded data into serving.posts_search")


def build_lexical_index():
    conn = psycopg2.connect(CONFIG["PG_DSN"])
    conn.autocommit = True
    cur = conn.cursor()

    print("[postgres] building lexical column...")
    cur.execute("""
        UPDATE serving.posts_search
        SET lexical =
            to_tsvector(
                'english',
                COALESCE(title_clean, '') || ' ' || COALESCE(text_clean, '')
            );
    """)

    print("[postgres] creating GIN index...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_posts_search_lexical
        ON serving.posts_search
        USING GIN (lexical);
    """)

    cur.close()
    conn.close()
    print("[postgres] lexical + index ready")

# ======================
# GOLD (DuckDB) – label_events
# ======================

def setup_gold_schema(con: duckdb.DuckDBPyConnection):
    con.execute("CREATE SCHEMA IF NOT EXISTS gold;")
    con.execute("""
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
    """)


def load_labels_into_gold(con: duckdb.DuckDBPyConnection):
    path = Path(GOLD_CONFIG["LABELS_PATH"])
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    print(f"[gold] reading labels from {path}")
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, dtype=str, engine="openpyxl")
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str, low_memory=False)
    else:
        raise ValueError(f"Unsupported labels extension {suffix!r}")

    item_id_col = GOLD_CONFIG["ITEM_ID_COL"]
    if item_id_col not in df.columns:
        raise ValueError(f"ITEM_ID_COL {item_id_col!r} not in labels file. Columns: {list(df.columns)}")

    annot_col = GOLD_CONFIG.get("ANNOTATOR_COL")
    default_annot = GOLD_CONFIG.get("DEFAULT_ANNOTATOR")
    ignore_cols = set(GOLD_CONFIG.get("IGNORE_COLS", []))

    # task columns = all non-ignored, non-item-id, non-annotator
    task_cols = [
        c for c in df.columns
        if c not in ignore_cols and c != item_id_col and c != annot_col
    ]
    if not task_cols:
        print("[gold] no task columns found (all columns ignored?)")
        return

    print(f"[gold] treating these columns as tasks: {task_cols}")

    now = dt.datetime.utcnow()
    run_id = "gold_" + now.strftime("%Y%m%d_%H%M%S")

    rows = []
    for _, row in df.iterrows():
        item_id_val = row[item_id_col]
        if pd.isna(item_id_val):
            continue
        item_id = str(item_id_val)

        annotator_id = None
        if annot_col and annot_col in df.columns:
            val = row[annot_col]
            if not pd.isna(val):
                annotator_id = str(val)
        if annotator_id is None and default_annot:
            annotator_id = default_annot

        for task_name in task_cols:
            value = row[task_name]
            if pd.isna(value):
                continue
            label_value = str(value).strip()
            if not label_value:
                continue

            rows.append({
                "item_type":     GOLD_CONFIG["ITEM_TYPE"],
                "item_id":       item_id,
                "task_name":     task_name,
                "label_value":   label_value,
                "annotator_id":  annotator_id,
                "label_source":  GOLD_CONFIG.get("LABEL_SOURCE", "human"),
                "confidence":    None,
                "guideline_ver": GOLD_CONFIG.get("GUIDELINE_VER", None),
                "created_at":    now.isoformat(),
                "run_id":        run_id,
            })

    if not rows:
        print("[gold] no label rows to insert (check IGNORE_COLS / ITEM_ID_COL).")
        return

    df_out = pd.DataFrame(rows)
    print(f"[gold] prepared {len(df_out)} label_events rows")

    setup_gold_schema(con)

    con.register("df_gold_view", df_out)
    con.execute("""
        INSERT INTO gold.label_events
        SELECT
            item_type,
            item_id,
            task_name,
            label_value,
            annotator_id,
            label_source,
            NULLIF(confidence, '')::DOUBLE,
            guideline_ver,
            created_at::TIMESTAMP,
            run_id
        FROM df_gold_view;
    """)

    n = con.execute("SELECT COUNT(*) FROM gold.label_events").fetchone()[0]
    print(f"[gold] gold.label_events now has {n} rows (including previous runs)")

# ======================
# EMBEDDINGS (no pgvector)
# ======================

def run_embeddings():
    print("[embed] loading model:", CONFIG["MODEL_NAME"])
    model = SentenceTransformer(CONFIG["MODEL_NAME"])

    con = setup_duckdb()
    df = con.execute("""
        SELECT
            post_id,
            COALESCE(title_clean, '') AS title_clean,
            COALESCE(text_clean, '')  AS text_clean
        FROM silver.posts
    """).df()

    print(f"[embed] got {len(df)} posts from silver.posts")

    texts = (df["title_clean"] + "\n\n" + df["text_clean"]).tolist()

    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    def vec_to_str(vec):
        return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    df["embedding_text"] = [vec_to_str(v) for v in emb]

    conn = psycopg2.connect(CONFIG["PG_DSN"])
    conn.autocommit = False
    cur = conn.cursor()

    print("[embed] updating embedding_text in serving.posts_search...")
    execute_batch(
        cur,
        """
        UPDATE serving.posts_search
        SET embedding_text = %(embedding_text)s
        WHERE post_id = %(post_id)s
        """,
        df[["post_id", "embedding_text"]].to_dict("records"),
        page_size=500,
    )

    conn.commit()
    cur.close()
    conn.close()
    print("[embed] embeddings stored in embedding_text")

# ======================
# HYBRID SEARCH (BM25 + cosine in Python)
# ======================

def parse_vec(s: str) -> np.ndarray:
    s = s.strip()
    if not s:
        return None
    if s[0] == "[" and s[-1] == "]":
        s = s[1:-1]
    return np.fromstring(s, sep=",", dtype=float)


def normalize_array(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    x_min, x_max = float(x.min()), float(x.max())
    if x_max == x_min:
        return np.ones_like(x)
    return (x - x_min) / (x_max - x_min)


def hybrid_search(query: str, top_k: int = 20, bm25_weight: float = 0.6):
    print(f"[search] query = {query!r}")
    model = SentenceTransformer(CONFIG["MODEL_NAME"])
    q_vec = model.encode([query], normalize_embeddings=True)[0]

    conn = psycopg2.connect(CONFIG["PG_DSN"])
    cur = conn.cursor()

    n_candidates = max(top_k * 5, top_k)

    cur.execute(
        """
        WITH q AS (
          SELECT plainto_tsquery('english', %s) AS tsq
        )
        SELECT
          p.post_id,
          p.subreddit,
          p.title_clean,
          p.text_clean,
          ts_rank_cd(p.lexical, q.tsq) AS bm25_score,
          p.embedding_text
        FROM serving.posts_search p, q
        WHERE p.lexical @@ q.tsq
          AND p.embedding_text IS NOT NULL
        ORDER BY bm25_score DESC
        LIMIT %s;
        """,
        (query, n_candidates),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("[search] no hits")
        return []

    post_ids, subs, titles, texts, bm25_scores, emb_texts = zip(*rows)
    bm25_scores = np.array(bm25_scores, dtype=float)

    doc_vecs = np.stack([parse_vec(s) for s in emb_texts])
    cosine_sim = doc_vecs @ q_vec

    bm25_norm = normalize_array(bm25_scores)
    cos_norm = normalize_array(cosine_sim)

    hybrid = bm25_weight * bm25_norm + (1.0 - bm25_weight) * cos_norm

    order = np.argsort(-hybrid)  # descending
    results = []
    for idx in order[:top_k]:
        results.append({
            "post_id": post_ids[idx],
            "subreddit": subs[idx],
            "title": titles[idx],
            "snippet": (texts[idx] or "")[:300],
            "bm25": float(bm25_scores[idx]),
            "cosine": float(cosine_sim[idx]),
            "hybrid": float(hybrid[idx]),
        })

    print(f"[search] top {top_k} results:")
    for i, r in enumerate(results, start=1):
        print(f"\n#{i} [{r['subreddit']}] {r['post_id']}")
        print(f" hybrid={r['hybrid']:.3f}  bm25={r['bm25']:.3f}  cos={r['cosine']:.3f}")
        print(f" title: {r['title']}")
        print(f" text : {r['snippet']}")
    return results

# ======================
# MAIN CLI
# ======================

def run_build():
    con = setup_duckdb()
    load_bronze(con)
    build_silver(con)
    csv_path = export_silver(con)

    init_postgres()
    load_into_postgres(csv_path)
    build_lexical_index()

    print(" build step finished: DuckDB + Postgres BM25 ready.")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python reddit_pipeline.py build")
        print("  python reddit_pipeline.py embed")
        print("  python reddit_pipeline.py search \"your query\" [top_k]")
        print("  python reddit_pipeline.py gold   # load annotations into gold.label_events")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "build":
        run_build()
    elif mode == "embed":
        run_embeddings()
    elif mode == "search":
        if len(sys.argv) < 3:
            print("Usage: python reddit_pipeline.py search \"your query\" [top_k]")
            sys.exit(1)
        query = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) >= 4 else 20
        hybrid_search(query, top_k=top_k)
    elif mode == "gold":
        con = setup_duckdb()
        load_labels_into_gold(con)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
