"""Database bootstrap: connection, schema, migrations, and the FTS5 index.

Design notes
------------
* The DB path is resolved relative to the project root, not the current
  working directory, so `python terminal-agent/agent.py` and
  `python agent.py` from inside the folder hit the same file.
* FTS5 gives us real lexical retrieval for free: porter stemming, prefix
  matching and BM25 ranking in C. The hand-rolled TF-IDF ranker still runs
  alongside it because it is a genuinely different signal (see
  `memory/retrieve.py`).
* Migration is additive: old v1 tables gain columns via ALTER TABLE, so no
  user data is destroyed.
"""

import os
import sqlite3
import threading
from pathlib import Path

SCHEMA_VERSION = 2

_LOCK = threading.RLock()
_CONN = None

# Columns added after v1. Each entry is (column, DDL type, default literal).
_V2_COLUMNS = [
    ("norm_hash", "TEXT", None),
    ("scope", "TEXT", "'global'"),
    ("subject", "TEXT", None),
    ("importance", "REAL", "0.5"),
    ("access_count", "INTEGER", "0"),
    ("updated_at", "TEXT", None),
    ("expires_at", "TEXT", None),
    ("archived_reason", "TEXT", None),
    ("supersedes", "INTEGER", None),
    ("superseded_by", "INTEGER", None),
    ("source", "TEXT", "'tool'"),
    ("session_id", "TEXT", None),
]


def project_root():
    return Path(__file__).resolve().parent.parent


def db_path():
    """Where the SQLite file lives.

    Override with RETAIN_DB_PATH (relative paths resolve against the project
    root, so tests can point at a temp file without depending on CWD).
    """
    override = os.environ.get("RETAIN_DB_PATH")
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = project_root() / candidate
        return candidate
    return project_root() / "app.db"


def connect(path=None):
    """Return the process-wide connection, creating the schema on first use."""
    global _CONN

    with _LOCK:
        if _CONN is not None:
            return _CONN

        target = Path(path) if path else db_path()
        target.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(target), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")

        _CONN = conn
        migrate(conn)
        return conn


def reset_connection():
    """Drop the cached handle. Used by tests and after a DB swap."""
    global _CONN
    with _LOCK:
        if _CONN is not None:
            try:
                _CONN.close()
            except sqlite3.Error:
                pass
        _CONN = None


def has_fts5(conn):
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts_probe USING fts5(x)")
        conn.execute("DROP TABLE temp.__fts_probe")
        return True
    except sqlite3.Error:
        return False


def migrate(conn):
    with _LOCK:
        _create_memory_table(conn)
        _upgrade_memory_table(conn)
        _create_access_table(conn)
        _create_meta_table(conn)
        if has_fts5(conn):
            _create_fts(conn)
        _stamp_version(conn)
        conn.commit()


def _create_memory_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            memory_type TEXT NOT NULL DEFAULT 'fact',
            confidence_score REAL NOT NULL DEFAULT 1.0,
            decay_rate REAL NOT NULL DEFAULT 0.1,
            created_at TEXT,
            last_accessed_at TEXT,
            is_archived INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def _upgrade_memory_table(conn):
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(memory)")}
    if not existing:
        return

    for column, ddl_type, default in _V2_COLUMNS:
        if column in existing:
            continue
        clause = f"ADD COLUMN {column} {ddl_type}"
        if default is not None:
            clause += f" DEFAULT {default}"
        conn.execute(f"ALTER TABLE memory {clause}")

    # Backfill updated_at from created_at for rows that predate the column.
    conn.execute(
        "UPDATE memory SET updated_at = created_at "
        "WHERE updated_at IS NULL AND created_at IS NOT NULL"
    )
    conn.execute(
        "UPDATE memory SET is_archived = 0 WHERE is_archived IS NULL"
    )
    conn.execute("UPDATE memory SET access_count = 0 WHERE access_count IS NULL")
    conn.execute("UPDATE memory SET importance = 0.5 WHERE importance IS NULL")

    _create_indexes(conn)


def _create_indexes(conn):
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_archived ON memory(is_archived)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(memory_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory(scope)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_hash ON memory(norm_hash)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_subject ON memory(subject)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_expiry ON memory(expires_at)"
    )
    # Partial index: every hot-path query filters on is_archived = 0.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_live "
        "ON memory(confidence_score DESC, last_accessed_at DESC) "
        "WHERE is_archived = 0"
    )


def _create_access_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_access (
            id INTEGER PRIMARY KEY,
            memory_id INTEGER NOT NULL,
            accessed_at TEXT,
            query TEXT,
            score REAL,
            FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_access_memory ON memory_access(memory_id)"
    )


def _create_meta_table(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT)"
    )


def _create_fts(conn):
    """External-content FTS table kept in sync by triggers.

    `content='memory'` means the index stores only the tokenized terms, not a
    second copy of the text, so the two tables cannot drift on size.
    """
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            text,
            content='memory',
            content_rowid='id',
            tokenize="porter unicode61"
        )
        """
    )

    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memory_fts_ai AFTER INSERT ON memory BEGIN
            INSERT INTO memory_fts(rowid, text) VALUES (new.id, new.text);
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memory_fts_ad AFTER DELETE ON memory BEGIN
            INSERT INTO memory_fts(memory_fts, rowid, text)
            VALUES ('delete', old.id, old.text);
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memory_fts_au AFTER UPDATE ON memory BEGIN
            INSERT INTO memory_fts(memory_fts, rowid, text)
            VALUES ('delete', old.id, old.text);
            INSERT INTO memory_fts(rowid, text) VALUES (new.id, new.text);
        END
        """
    )

    # Backfill: rows that existed before the index did are invisible to FTS
    # until we rebuild, and we cannot know which ones those are cheaply.
    indexed = conn.execute("SELECT count(*) AS c FROM memory_fts").fetchone()["c"]
    total = conn.execute("SELECT count(*) AS c FROM memory").fetchone()["c"]
    if indexed < total:
        conn.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")


def _stamp_version(conn):
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
        (str(SCHEMA_VERSION),),
    )


def schema_version(conn=None):
    conn = conn or connect()
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ).fetchone()
    return int(row["value"]) if row else 1
