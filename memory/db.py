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

import atexit
import os
import sqlite3
import threading
import time
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
    ("ttl_days", "REAL", None),
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


class _DurableConnection(sqlite3.Connection):
    """A connection that folds the write-ahead log into the main file on commit.

    The whole point of this class is one sentence of the product requirement:
    stop the application, start it again, the memory is still there. Commits
    alone already achieve that - but only if the reader also has the `-wal`
    file, and a user who stops the app on a Friday and expects a real backup on
    Saturday is going to copy the file called `app.db`.

    In WAL mode a freshly created store can leave `app.db` a few kilobytes of
    schema with *every memory still sitting in `app.db-wal`*: the file with the
    memory bank's name on it reads as an empty table, while the running app
    works fine, so nothing looks broken until the day the copy is needed.
    `wal_autocheckpoint` does not fix this - its default of 1000 pages is tuned
    for a write-heavy server and is never reached by a personal memory bank,
    and lowering it still left the rows in the log.

    Checkpointing on every commit makes `app.db` complete after each write, and
    at a handful of writes per conversation turn the cost is a couple of
    milliseconds. Correctness of the artifact the user can see beats a
    micro-optimisation nobody can observe.
    """

    def commit(self):
        super().commit()
        try:
            super().execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            # A busy or unavailable checkpoint is not data loss: the log is
            # still valid and replays on the next open. Never fail a write
            # because the tidy-up could not run.
            pass


def _open(target):
    """Open one connection and put it into WAL mode."""
    conn = sqlite3.connect(
        str(target), check_same_thread=False, factory=_DurableConnection
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # FULL, not NORMAL. NORMAL is the right trade for a scratch index but it
    # only guarantees durability against a *process* crash - a power cut can
    # still lose the last commits. This store is the user's own memory bank;
    # there is no stream of writes to protect at this rate, so paying the extra
    # fsync to make "written" mean "on disk" is the correct call.
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _discard_stale_logs(target):
    """Delete a `-shm`/`-wal` pair left behind by a process that was killed.

    `-shm` is a memory-mapped index of the write-ahead log. A process killed
    with the terminal window closed leaves one that the next process can fail
    to map, and the symptom is `PRAGMA journal_mode=WAL` raising "disk I/O
    error" - the memory store then refuses to start at all, which is the worst
    possible time to discover it.

    Removing the pair is only safe while nobody else has the database open, and
    `connect()` holds the process-wide lock with `_CONN is None` at this point,
    so this store is the only user. SQLite itself treats an unreferenced log and
    index as disposable and rebuilds both.
    """
    removed = []
    for suffix in ("-shm", "-wal"):
        stray = Path(f"{target}{suffix}")
        try:
            stray.unlink()
            removed.append(suffix)
        except OSError:
            pass
    return removed


def connect(path=None):
    """Return the process-wide connection, creating the schema on first use."""
    global _CONN

    with _LOCK:
        if _CONN is not None:
            return _CONN

        target = Path(path) if path else db_path()
        target.parent.mkdir(parents=True, exist_ok=True)

        conn = None
        # A just-killed predecessor can still be releasing its file handles for
        # a moment, so a transient failure here is worth a second chance before
        # concluding anything is actually wrong with the file.
        for attempt in range(3):
            try:
                conn = _open(target)
                break
            except sqlite3.OperationalError:
                if attempt == 1 and target.exists():
                    _discard_stale_logs(target)
                if attempt == 2:
                    raise
                time.sleep(0.2)

        _CONN = conn
        migrate(conn)

        # Belt and braces alongside the explicit `shutdown()` the agent calls.
        # A terminal closed with the window X'd, a Ctrl-C that propagates, or
        # an unhandled exception all reach here, and the point is to leave the
        # single `.db` file self-contained rather than trailing a WAL that holds
        # the newest memories.
        atexit.register(shutdown)
        return conn


def shutdown():
    """Flush the WAL into the main file and close. Safe to call repeatedly.

    Every write is already committed and therefore already durable - a
    checkpoint is not needed to *save* anything. It is needed so that the
    `.db` file on its own holds the full history: without it the newest
    memories sit in `-wal` until the next process opens the database, so
    copying, backing up or shipping `app.db` would quietly lose them.
    """
    global _CONN
    with _LOCK:
        if _CONN is None:
            return False
        try:
            _CONN.commit()
            _CONN.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            # A checkpoint failure is not a data loss - the WAL is still valid
            # and replays on the next open - so do not block the exit on it.
            pass
        try:
            _CONN.close()
        except sqlite3.Error:
            pass
        _CONN = None
        return True


def reset_connection():
    """Drop the cached handle. Used by tests and after a DB swap."""
    shutdown()


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
    _create_decision_table(conn)


def _create_decision_table(conn):
    """Cache of model verdicts on "are these two memories the same fact?".

    A verdict of *different* leaves both memories live, so the same pair is
    offered to the model again on the next pass - and the next. Without this
    table, consolidation re-buys the same questions every 25 turns forever.
    Keyed on the pair's text, so editing either memory reopens the question
    instead of inheriting a stale answer.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS consolidation_decisions (
            pair_hash TEXT PRIMARY KEY,
            same INTEGER NOT NULL,
            decided_at TEXT
        )
        """
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

    # Rows that predate the index are invisible to FTS until it is rebuilt.
    _ensure_fts_index(conn)


def _ensure_fts_index(conn):
    """Make the FTS index agree with the content table.

    The obvious check - comparing `count(*)` on the FTS table against the
    content table - does not work here. For an *external content* table,
    `SELECT count(*) FROM memory_fts` reads the content table, so it always
    agrees and the index is silently never built for pre-existing rows. The
    result is an FTS table that reports rows it cannot match, and any later
    write fires the AFTER UPDATE trigger, whose 'delete' of the missing index
    entry raises "database disk image is malformed".

    So ask FTS5 itself. `integrity-check` compares the index against the
    content table, and rebuild only when they disagree. Both are cheap at
    personal-remember scale and run once per process start.
    """
    if fts_index_is_sane(conn):
        return

    conn.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")


def fts_index_is_sane(conn):
    """True when the FTS index matches its content table."""
    try:
        conn.execute(
            "INSERT INTO memory_fts(memory_fts, rank) VALUES ('integrity-check', 1)"
        )
        return True
    except sqlite3.Error:
        # FTS5 reports an inconsistent index by raising here.
        return False


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


def persistence_report():
    """Everything needed to answer "will my memories still be here tomorrow?".

    Deliberately answers the question a user actually has - is this the file I
    think it is, is it writable, is it intact, how much is in it - rather than
    echoing back the pragmas. Surfaced by `/memory persist` so that "my memory
    disappeared" is a diagnosis instead of a guess.
    """
    path = db_path()
    conn = connect()

    def _count(sql):
        try:
            return conn.execute(sql).fetchone()[0]
        except sqlite3.Error:
            return None

    def _size(suffix):
        try:
            return (path.parent / f"{path.name}{suffix}").stat().st_size
        except OSError:
            return 0

    writable = False
    try:
        with open(path, "r+b"):
            writable = True
    except OSError:
        writable = os.access(str(path), os.W_OK)

    return {
        "path": str(path),
        "exists": path.exists(),
        "writable": writable,
        "bytes": path.stat().st_size if path.exists() else 0,
        "wal_bytes": _size("-wal"),
        "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
        "synchronous": conn.execute("PRAGMA synchronous").fetchone()[0],
        "integrity": conn.execute("PRAGMA integrity_check").fetchone()[0],
        "fts_sane": fts_index_is_sane(conn),
        "schema_version": schema_version(conn),
        "total": _count("SELECT COUNT(*) FROM memory"),
        "live": _count("SELECT COUNT(*) FROM memory WHERE is_archived = 0"),
        "archived": _count("SELECT COUNT(*) FROM memory WHERE is_archived = 1"),
        "newest": _count(
            "SELECT MAX(created_at) FROM memory WHERE is_archived = 0"
        ),
    }
