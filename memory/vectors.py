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
#
# Calibrated on the 262-memory benchmark, comparing each labelled question's
# wanted memory against the best unwanted one:
#
#   true-answer cosines  min 0.0000  median 0.4525  max 0.7453
#   best-wrong cosines   min 0.1110  median 0.2859  max 0.5615
#
# 0.12 sits just above the weakest false positive observed, so a hit below it is
# treated as noise. Note what that leaves on the table: the cheapest true match
# scores 0.2139 ("do I have any allergies"), while the purely synonymic ones
# score ~0.00 and are unreachable at any threshold. This embedder is a
# character-overlap index - see the module docstring in `embed.py` for the full
# measurement of what it cannot do.
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


def index_rows(rows, embedder=None, conn=None, batch_size=64):
    """Bulk-index `(id, text)` pairs, embedding in batches.

    Re-indexing a store one memory at a time costs one HTTP round trip per
    memory when the embedder is a hosted model: 262 memories took 269 calls and
    6.6s this way, against 5 calls for the same work batched. Writes are
    committed per batch rather than per row, so a failure costs one batch
    instead of the whole run, and the rows already committed stay valid.
    """
    conn = conn or _conn()
    embedder = embedder or embed.current()
    rows = [(int(mid), text) for mid, text in rows if str(text or "").strip()]
    if not rows:
        return 0, 0

    indexed = skipped = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        try:
            vectors_out = embedder.embed_many([text for _mid, text in chunk])
        except Exception:
            skipped += len(chunk)
            continue
        stamp = now_iso()
        payload = []
        for (mid, text), vector in zip(chunk, vectors_out):
            if vector is None or not any(vector):
                continue
            payload.append((mid, embedder.name, len(vector), embed.to_blob(vector), stamp))
        if not payload:
            skipped += len(chunk)
            continue
        try:
            conn.executemany(
                """INSERT INTO memory_vector (memory_id, model, dim, vec, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(memory_id, model)
                   DO UPDATE SET dim = excluded.dim,
                                 vec = excluded.vec,
                                 updated_at = excluded.updated_at""",
                payload,
            )
            conn.commit()
            indexed += len(payload)
            skipped += len(chunk) - len(payload)
        except Exception:
            skipped += len(chunk)
    return indexed, skipped


def reindex(conn=None, limit=None):
    """Rebuild every live memory's vector for the current model.

    Returns (indexed, skipped, already_current).

    A row counts as current only if its stored vector is at least as new as the
    memory text it describes. Skipping on "a vector exists" alone would leave a
    permanently stale vector behind whenever a memory was edited by a process
    that did not re-embed - a second agent instance, or a write made while this
    one was not running - and a memory findable only by wording the user has
    already replaced is worse than one that is merely unindexed, because nothing
    about it looks wrong.
    """
    conn = conn or _conn()
    embedder = embed.current()
    rows = conn.execute(
        "SELECT id, text, updated_at FROM memory WHERE is_archived = 0 ORDER BY id"
    ).fetchall()

    have = {
        int(row["memory_id"]): row["updated_at"]
        for row in conn.execute(
            "SELECT memory_id, updated_at FROM memory_vector WHERE model = ?",
            (embedder.name,),
        )
    }

    indexed = skipped = current = 0
    todo = []
    for row in rows:
        if limit is not None and len(todo) >= limit:
            break
        memory_id = int(row["id"])
        stamped = have.get(memory_id)
        if stamped is not None and not _is_stale(row["updated_at"], stamped):
            current += 1
            continue
        todo.append((memory_id, row["text"]))

    if todo:
        indexed, skipped = index_rows(todo, embedder=embedder, conn=conn)

    _stamp(conn)
    return indexed, skipped, current


def _is_stale(memory_updated_at, vector_updated_at):
    """True when the text is newer than the vector built from it.

    Unparseable or missing timestamps are treated as stale: re-embedding is
    cheap next to being wrong, and a row with no usable timestamp is a row we
    cannot claim is up to date.
    """
    if not memory_updated_at:
        return not vector_updated_at
    if not vector_updated_at:
        return True
    return str(memory_updated_at) > str(vector_updated_at)


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
        probe = (
            embedder.embed_query(query)
            if hasattr(embedder, "embed_query")
            else embedder.embed(query)
        )
    except Exception:
        # The embedder may be a hosted model. A store whose embedder needs the
        # network must still answer from the lexical rankers when the network is
        # down, so an unreachable API means "no vector", not an exception -
        # raising here would take the whole agent down because an embedding
        # call timed out.
        return []
    if not probe or not any(probe):
        return []

    floor = getattr(embedder, "min_similarity", MIN_SIMILARITY)

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
        if score >= floor:
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
