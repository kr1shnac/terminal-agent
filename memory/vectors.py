"""The vector half of memory: persistent embeddings and cosine search.

Kept apart from `store.py` on purpose. `store.py` is the truth about what a
memory *is*; this is one derived, rebuildable index over it. Every row here can
be thrown away and regenerated from `memory.text`, which is what makes it safe
to change the embedder, and why a failure in this module is never allowed to
stop a write from succeeding.
"""

from . import db, embed
from .clock import now_iso

# How many vectors one scan will consider. A personal memory bank is hundreds of
# rows; this exists so a pathological store cannot turn a query into a long
# linear scan.
MAX_SCAN = 5000

# Below this cosine, a hit is noise. Two unrelated sentences of English share
# enough subword material to sit around 0.0-0.1, and returning those as
# "related" is how a retriever fills a context window with plausible-sounding
# rubbish.
MIN_SIMILARITY = 0.12


def _conn():
    return db.connect()


def table_count(conn=None):
    conn = conn or _conn()
    row = conn.execute("SELECT COUNT(*) FROM memory_vector").fetchone()
    return int(row[0]) if row else 0


def index_memory(memory_id, text, embedder=None, conn=None):
    """Store (or refresh) the vector for one memory.

    Never raises, and never sees corpus IDF - see `search`. A memory that cannot
    be embedded is a memory that cannot be *found by meaning*; refusing to
    record it at all would turn a degraded search feature into data loss, which
    is a far worse failure than falling back to the lexical rankers that always
    work.
    """
    embedder = embedder or embed.current()
    try:
        vector = embedder.embed(text)
    except Exception:
        return False

    if not vector or not any(vector):
        # No usable signal. Drop any stale vector rather than leaving one that
        # points at the old text.
        delete_memory(memory_id, embedder.name, conn=conn)
        return False

    conn = conn or _conn()
    try:
        conn.execute(
            "INSERT INTO memory_vector(memory_id, model, dim, vec, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(memory_id, model) DO UPDATE SET "
            "dim = excluded.dim, vec = excluded.vec, updated_at = excluded.updated_at",
            (
                memory_id,
                embedder.name,
                len(vector),
                embed.to_blob(vector),
                now_iso(),
            ),
        )
        conn.commit()
        return True
    except Exception:
        return False


def index_many(items, embedder=None, conn=None):
    """Index a batch. `items` is an iterable of objects with `.id` and `.text`."""
    embedder = embedder or embed.current()
    count = 0
    for item in items:
        if index_memory(item.id, item.text, embedder=embedder, conn=conn):
            count += 1
    return count


def delete_memory(memory_id, model=None, conn=None):
    conn = conn or _conn()
    model = model or embed.current_name()
    try:
        conn.execute(
            "DELETE FROM memory_vector WHERE memory_id = ? AND model = ?",
            (memory_id, model),
        )
        conn.commit()
        return True
    except Exception:
        return False


def delete_for_model(model, conn=None):
    conn = conn or _conn()
    try:
        conn.execute("DELETE FROM memory_vector WHERE model = ?", (model,))
        conn.commit()
        return True
    except Exception:
        return False


def stale_count(model=None, conn=None):
    """Rows whose text has changed since it was embedded, or is missing.

    The trigger-based FTS index cannot drift because SQLite maintains it. This
    one can: a row edited by raw SQL, a model swapped in, an interrupted sweep.
    `reindex_stale` uses this to catch up.
    """
    conn = conn or _conn()
    model = model or embed.current_name()
    row = conn.execute(
        "SELECT COUNT(*) FROM memory m WHERE m.is_archived = 0 AND ("
        "  NOT EXISTS (SELECT 1 FROM memory_vector v"
        "             WHERE v.memory_id = m.id AND v.model = ?)"
        "  OR EXISTS (SELECT 1 FROM memory_vector v"
        "             WHERE v.memory_id = m.id AND v.model = ?"
        "               AND v.updated_at IS NOT NULL"
        "               AND v.updated_at < m.updated_at)"
        ")",
        (model, model),
    ).fetchone()
    return int(row[0]) if row else 0


def reindex(conn=None, limit=None):
    """Rebuild every live memory's vector for the current model.

    Returns (indexed, skipped, already_current).
    """
    conn = conn or _conn()
    embedder = embed.current()
    rows = conn.execute(
        "SELECT id, text FROM memory WHERE is_archived = 0 ORDER BY id"
    ).fetchall()

    have = {
        int(r[0])
        for r in conn.execute(
            "SELECT memory_id FROM memory_vector WHERE model = ?", (embedder.name,)
        )
    }

    indexed = skipped = current = 0
    for row in rows:
        if limit is not None and indexed >= limit:
            break
        memory_id = int(row[0])
        if memory_id in have:
            current += 1
            continue
        if index_memory(memory_id, row[1], embedder=embedder, conn=conn):
            indexed += 1
        else:
            skipped += 1

    _stamp(conn)
    return indexed, skipped, current


def _stamp(conn):
    try:
        conn.execute(
            "INSERT OR REPLACE INTO memory_vector_meta(key, value) VALUES (?, ?)",
            ("last_reindex_at", now_iso()),
        )
        conn.commit()
    except Exception:
        pass


def last_reindex_at(conn=None):
    conn = conn or _conn()
    row = conn.execute(
        "SELECT value FROM memory_vector_meta WHERE key = 'last_reindex_at'"
    ).fetchone()
    return row[0] if row else None


def search(query, limit=40, model=None, idf=None, conn=None):
    """Top-`limit` live memories by cosine to `query`, as (memory_id, score).

    A semantic safety net, not the primary ranker: it can surface a memory that
    shares no word with the question, but it has no notion of importance,
    recency or negation, so the caller still has to blend and filter what comes
    back.

    `idf` must stay `None` here and at index time. Corpus IDF weights drift as
    memories are added, and a cosine between two vectors built with different
    weightings is a number that looks like a similarity and means nothing. The
    stored vectors therefore all use the neutral prior, which is stable by
    construction; discriminating power comes from BM25, which has its own
    corpus statistics and is free to use them.
    """
    conn = conn or _conn()
    embedder = embed.get(model) if model else embed.current()
    if embedder is None:
        return []

    try:
        probe = embedder.embed(query)
    except Exception:
        return []
    if not probe or not any(probe):
        return []

    rows = conn.execute(
        "SELECT v.memory_id, v.dim, v.vec FROM memory_vector v "
        "JOIN memory m ON m.id = v.memory_id "
        "WHERE v.model = ? AND m.is_archived = 0 "
        "ORDER BY v.memory_id DESC LIMIT ?",
        (embedder.name, MAX_SCAN),
    ).fetchall()

    scored = []
    for memory_id, dim, blob in rows:
        stored = embed.from_blob(blob, int(dim))
        if stored is None or len(stored) != len(probe):
            # Written by a different embedder. Skipping is correct: comparing
            # across embedding spaces produces confident nonsense.
            continue
        score = embed.dot(probe, stored)
        if score >= MIN_SIMILARITY:
            scored.append((int(memory_id), round(score, 4)))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit]


def coverage(conn=None, model=None):
    """How much of the live store has a vector. For `/memory persist`."""
    conn = conn or _conn()
    model = model or embed.current_name()
    live = conn.execute(
        "SELECT COUNT(*) FROM memory WHERE is_archived = 0"
    ).fetchone()[0]
    indexed = conn.execute(
        "SELECT COUNT(*) FROM memory_vector v JOIN memory m ON m.id = v.memory_id "
        "WHERE v.model = ? AND m.is_archived = 0",
        (model,),
    ).fetchone()[0]
    return {
        "model": model,
        "live": int(live),
        "indexed": int(indexed),
        "missing": int(live) - int(indexed),
        "rows_total": table_count(conn),
    }
