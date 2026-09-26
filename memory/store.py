"""Persistence for memories: insert, read, update, archive, supersede.

What this module adds over the v1 store:

* `insert_memory` returns the row id, so callers can dedupe, link, and undo.
* exact-duplicate detection via a normalized-text hash, plus near-duplicate
  detection via token/trigram overlap. Without this the agent re-stores the
  same fact every time it is mentioned and floods its own context.
* contradiction handling: a new memory with the same `subject` but different
  text supersedes the old one instead of coexisting with it. "uses npm" and
  "uses pnpm" must never both reach the prompt.
* an access log, which is what makes "which memories are actually earning
  their place" answerable.
"""

import sqlite3

from . import db
from .clock import now_iso, plus_days
from .models import (
    DEFAULT_TYPE,
    MemoryItem,
    decay_rate_for,
    get_type,
    importance_for,
    normalize_type,
)
from .text import (
    build_idf,
    fts_query,
    identity_tokens,
    norm_hash,
    similarity,
    tokenize,
)
from . import vectors as _vectors

# Weighted-similarity bar for treating a second statement as the same fact.
# Calibrated so paraphrases clear it and one-word-substituted contradictions
# ("pnpm" -> "yarn") stay well under.
NEAR_DUPE_THRESHOLD = 0.78

LIVE_PREDICATE = "is_archived = 0 AND superseded_by IS NULL"


def _conn():
    return db.connect()


def _rows_to_items(rows):
    return [MemoryItem.from_row(row) for row in rows]


# ---------------------------------------------------------------- insert


def is_refinement(existing, new_text):
    """True when `new_text` states the same fact with strictly more detail.

    "The user's name is Krishna." followed by "The user's name is Krishna C."
    is the user sharpening a memory, not making a new one. Dropping the second
    would silently lose the correction, and creating a second row would put
    two names in the prompt.

    The test is whether the new wording *adds content words and loses none*,
    not how many characters it grew by. A character delta gets the threshold
    wrong in both directions: it rejects a genuinely fuller last name that adds
    one word, and it accepts a stray clause that adds no information.
    """
    old_tokens = set(tokenize(existing.text))
    new_tokens = set(tokenize(new_text))
    if not old_tokens <= new_tokens:
        return False  # dropped something the original said
    return bool(new_tokens - old_tokens)  # and added something new


def _apply_refinement(existing, new_text):
    """Sharpen a stored memory in place, keeping its id and subject links."""
    conn = _conn()
    conn.execute(
        "UPDATE memory SET text = ?, norm_hash = ?, updated_at = ? WHERE id = ?",
        (new_text, norm_hash(new_text), now_iso(), existing.id),
    )
    conn.commit()


def insert_memory(
    text,
    memory_type=DEFAULT_TYPE,
    scope="global",
    subject=None,
    importance=None,
    confidence=1.0,
    ttl_days=None,
    source="tool",
    session_id=None,
    allow_duplicate=False,
):
    """Store one memory. Returns the created :class:`MemoryItem`.

    When the text duplicates something already stored, the *existing* item is
    returned and nothing new is written, so "the same fact" stays one row.

    A duplicate is not treated as a recall. Restating something is weak
    evidence, and counting it as a use would let a memory that is merely
    repeated earn permanent decay relief and distort the access log that
    "which memories earn their place" is measured from. If the new wording is
    a refinement, the stored text is sharpened in place instead.
    """
    conn = _conn()
    text = str(text or "").strip()
    if not text:
        return None

    memory_type = normalize_type(memory_type)
    if importance is None:
        importance = importance_for(memory_type)
    if ttl_days is None:
        ttl_days = get_type(memory_type).ttl_days

    if not allow_duplicate:
        existing = find_duplicate(text)
        if existing is not None:
            if is_refinement(existing, text):
                _apply_refinement(existing, text)
            return get(existing.id)

    digest = norm_hash(text)
    stamp = now_iso()
    expires_at = plus_days(ttl_days) if ttl_days else None
    rate = decay_rate_for(memory_type)

    cursor = conn.execute(
        """
        INSERT INTO memory (
            text, norm_hash, memory_type, confidence_score, decay_rate,
            importance, ttl_days, scope, subject, created_at, updated_at,
            last_accessed_at, expires_at, is_archived, access_count, source,
            session_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
        """,
        (
            text,
            digest,
            memory_type,
            float(confidence),
            float(rate),
            float(importance),
            float(ttl_days) if ttl_days else None,
            scope or "global",
            subject,
            stamp,
            stamp,
            stamp,
            expires_at,
            source or "tool",
            session_id,
        ),
    )
    conn.commit()

    new_id = cursor.lastrowid
    _supersede_conflicts(new_id, text, subject, scope, memory_type)
    conn.commit()

    item = get(new_id)
    if item is not None:
        # Index by meaning on the way in. Derived and rebuildable, so a failure
        # here must not fail the write - `index_memory` swallows its own errors
        # for exactly that reason.
        _vectors.index_memory(new_id, item.text)
    return item


def find_duplicate(text, window=200):
    """Cheap-then-thorough duplicate check.

    Step 1 is an exact hash hit, which is O(1) and catches the common case
    (the model re-emitting a memory verbatim). Step 2 only runs when the hash
    misses, and it narrows the rows worth scoring two ways:

    * the FTS index, which returns every live row sharing a term with the
      incoming text. Previously this step only looked at the `window` most
      recently *accessed* rows, so a duplicate went undetected as soon as the
      original had not been recalled recently - the same fact came back in a
      new row and the store grew a duplicate per re-derivation. Recency is not
      evidence of distinctness.
    * the recency window, kept as a floor so behaviour is unchanged when FTS5
      is unavailable or the text has no indexable terms (non-Latin script).

    Similarity is IDF-weighted (see `text.weighted_jaccard`) so a paraphrase
    merges while a contradiction does not.

    Returns a :class:`MemoryItem` or ``None``.
    """
    conn = _conn()

    digest = norm_hash(text)
    row = conn.execute(
        f"SELECT * FROM memory WHERE norm_hash = ? AND {LIVE_PREDICATE} "
        "ORDER BY id DESC LIMIT 1",
        (digest,),
    ).fetchone()
    if row:
        return MemoryItem.from_row(row)

    new_tokens = tokenize(text)
    if not new_tokens:
        # Nothing to compare on. The hash above is the only defence left, and
        # it is exact - so rather than guess, decline the near-dupe check.
        return None

    candidates = _dupe_candidates(conn, text, window)
    if not candidates:
        return None

    idf = build_idf([tokenize(row["text"]) for row in candidates])

    # Tokens the index cannot see but which still tell two memories apart.
    # Without this guard, "notebook in slot 0" and "notebook in slot 1" are the
    # same token set, score a perfect 1.0, and one of the two is dropped -
    # silently, and at exactly the moment the store is asked to be reliable.
    mine = identity_tokens(text)

    best, best_score = None, 0.0
    for row in candidates:
        if row["norm_hash"] == digest:
            return MemoryItem.from_row(row)

        if mine != identity_tokens(row["text"]):
            continue

        score = similarity(new_tokens, tokenize(row["text"]), idf)
        if score > best_score:
            best, best_score = row, score

    return MemoryItem.from_row(best) if best_score >= NEAR_DUPE_THRESHOLD else None


# How many rows a single near-dupe comparison will score. Beyond this the cost
# of a false merge outweighs the chance of catching one more paraphrase.
MAX_DUPE_CANDIDATES = 400


def _dupe_candidates(conn, text, window):
    """Live rows worth scoring against `text`: index hits plus a recency floor."""
    seen = {}

    match = fts_query(text, mode="OR")
    if match:
        try:
            rows = conn.execute(
                f"""
                SELECT m.* FROM memory_fts
                JOIN memory m ON m.id = memory_fts.rowid
                WHERE memory_fts MATCH ? AND {LIVE_PREDICATE}
                LIMIT ?
                """,
                (match, MAX_DUPE_CANDIDATES),
            ).fetchall()
        except sqlite3.Error:
            rows = []
        for row in rows:
            seen[row["id"]] = row

    for row in conn.execute(
        f"SELECT * FROM memory WHERE {LIVE_PREDICATE} "
        "ORDER BY last_accessed_at DESC LIMIT ?",
        (window,),
    ).fetchall():
        seen.setdefault(row["id"], row)

    return list(seen.values())


def _supersede_conflicts(new_id, text, subject, scope, memory_type):
    """Retire memories that the new one invalidates.

    Only acts when a `subject` is supplied: "user.editor" is a slot that holds
    one value, so a second value for the same slot is a replacement. Without a
    subject there is no way to know two statements conflict, and guessing
    would delete good data.
    """
    if not subject:
        return

    conn = _conn()
    rows = conn.execute(
        f"SELECT id, text FROM memory WHERE subject = ? AND id != ? "
        f"AND {LIVE_PREDICATE}",
        (subject, new_id),
    ).fetchall()

    for row in rows:
        if row["text"].strip().lower() == text.strip().lower():
            continue
        conn.execute(
            "UPDATE memory SET is_archived = 1, superseded_by = ?, "
            "archived_reason = 'superseded', updated_at = ? WHERE id = ?",
            (new_id, now_iso(), row["id"]),
        )
        conn.execute(
            "UPDATE memory SET supersedes = ? WHERE id = ?",
            (row["id"], new_id),
        )


# ------------------------------------------------------------------- read


def get(memory_id):
    conn = _conn()
    row = conn.execute("SELECT * FROM memory WHERE id = ?", (memory_id,)).fetchone()
    return MemoryItem.from_row(row)


def get_many(memory_ids):
    ids = [int(i) for i in memory_ids if i is not None]
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    conn = _conn()
    rows = conn.execute(
        f"SELECT * FROM memory WHERE id IN ({placeholders})", ids
    ).fetchall()
    order = {mid: pos for pos, mid in enumerate(ids)}
    rows = sorted(rows, key=lambda r: order.get(r["id"], 0))
    return _rows_to_items(rows)


def get_active(limit=None):
    """Every live memory, most confident and most recently used first."""
    conn = _conn()
    sql = (
        f"SELECT * FROM memory WHERE {LIVE_PREDICATE} "
        "ORDER BY confidence_score DESC, last_accessed_at DESC"
    )
    params = ()
    if limit:
        sql += " LIMIT ?"
        params = (int(limit),)
    return _rows_to_items(conn.execute(sql, params).fetchall())


def get_by_subject(subject, scope=None):
    conn = _conn()
    sql = f"SELECT * FROM memory WHERE subject = ? AND {LIVE_PREDICATE}"
    params = [subject]
    if scope:
        sql += " AND scope = ?"
        params.append(scope)
    return _rows_to_items(conn.execute(sql, params).fetchall())


def get_by_type(memory_type, limit=None):
    conn = _conn()
    sql = (
        f"SELECT * FROM memory WHERE memory_type = ? AND {LIVE_PREDICATE} "
        "ORDER BY confidence_score DESC"
    )
    params = [normalize_type(memory_type)]
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    return _rows_to_items(conn.execute(sql, params).fetchall())


def get_archived(limit=None):
    conn = _conn()
    sql = (
        "SELECT * FROM memory WHERE is_archived = 1 "
        "ORDER BY updated_at DESC"
    )
    params = ()
    if limit:
        sql += " LIMIT ?"
        params = (int(limit),)
    return _rows_to_items(conn.execute(sql, params).fetchall())


def get_expired(now=None):
    """Live rows whose TTL has passed. Archived by the decay sweep."""
    conn = _conn()
    now_iso_value = now or now_iso()
    return _rows_to_items(
        conn.execute(
            f"SELECT * FROM memory WHERE expires_at IS NOT NULL "
            f"AND expires_at <= ? AND {LIVE_PREDICATE}",
            (now_iso_value,),
        ).fetchall()
    )


# ---------------------------------------------------------------- update


def update_confidence(
    memory_id, new_confidence, new_decay_rate=None, last_accessed_at=None
):
    """Write a new confidence.

    The store maintains one invariant: `confidence_score` is the confidence
    *as of* `last_accessed_at`. Any writer that changes one must move the
    other, or the decay projection compounds every time it is applied and
    memories evaporate far faster than the curve says they should.
    """
    new_confidence = max(0.0, min(1.0, float(new_confidence)))
    conn = _conn()

    if new_decay_rate is not None and last_accessed_at is not None:
        conn.execute(
            "UPDATE memory SET confidence_score = ?, decay_rate = ?, "
            "last_accessed_at = ?, updated_at = ? WHERE id = ?",
            (
                new_confidence,
                float(new_decay_rate),
                last_accessed_at,
                now_iso(),
                memory_id,
            ),
        )
    elif new_decay_rate is not None:
        conn.execute(
            "UPDATE memory SET confidence_score = ?, decay_rate = ?, "
            "updated_at = ? WHERE id = ?",
            (new_confidence, float(new_decay_rate), now_iso(), memory_id),
        )
    elif last_accessed_at is not None:
        conn.execute(
            "UPDATE memory SET confidence_score = ?, last_accessed_at = ?, "
            "updated_at = ? WHERE id = ?",
            (new_confidence, last_accessed_at, now_iso(), memory_id),
        )
    else:
        conn.execute(
            "UPDATE memory SET confidence_score = ?, updated_at = ? WHERE id = ?",
            (new_confidence, now_iso(), memory_id),
        )
    conn.commit()


def update_last_accessed(memory_id):
    conn = _conn()
    conn.execute(
        "UPDATE memory SET last_accessed_at = ?, access_count = "
        "COALESCE(access_count, 0) + 1 WHERE id = ?",
        (now_iso(), memory_id),
    )
    conn.commit()


def update_text(memory_id, new_text):
    conn = _conn()
    new_text = str(new_text or "").strip()
    if not new_text:
        return None
    conn.execute(
        "UPDATE memory SET text = ?, norm_hash = ?, updated_at = ? WHERE id = ?",
        (new_text, norm_hash(new_text), now_iso(), memory_id),
    )
    conn.commit()
    return get(memory_id)


def update_fields(memory_id, **fields):
    """Whitelist-based partial update. Ignores unknown keys."""
    allowed = {
        "text",
        "memory_type",
        "scope",
        "subject",
        "importance",
        "confidence",
        "ttl_days",
        "source",
    }

    sets, params = [], []
    for key, value in fields.items():
        if key not in allowed or value is None:
            continue
        if key == "confidence":
            column, value = "confidence_score", max(0.0, min(1.0, float(value)))
        elif key == "text":
            value = str(value).strip()
            if not value:
                continue
            column = "text"
        else:
            column = key
        sets.append(f"{column} = ?")
        params.append(value)

    if not sets:
        return get(memory_id)

    if "text" in fields:
        conn = _conn()
        params.append(norm_hash(str(fields["text"]).strip()))
        sets.append("norm_hash = ?")

    conn = _conn()
    params.extend([now_iso(), memory_id])
    conn.execute(
        f"UPDATE memory SET {', '.join(sets)}, updated_at = ? WHERE id = ?",
        params,
    )
    conn.commit()

    item = get(memory_id)
    # The text may have changed underneath the old vector, which would leave a
    # memory findable only by its previous wording.
    if item is not None and "text" in fields:
        _vectors.index_memory(memory_id, item.text)
    return item


def reinforce(memory_id):
    """Bump confidence and slow decay for a memory that was actually used."""
    from .decay import reinforce as _reinforce

    item = get(memory_id)
    if item is None:
        return None
    return _reinforce(item)


def log_access(memory_id, query=None, score=None):
    conn = _conn()
    conn.execute(
        "INSERT INTO memory_access (memory_id, accessed_at, query, score) "
        "VALUES (?, ?, ?, ?)",
        (memory_id, now_iso(), query, score),
    )
    conn.commit()


def access_history(memory_id, limit=20):
    conn = _conn()
    rows = conn.execute(
        "SELECT accessed_at, query, score FROM memory_access "
        "WHERE memory_id = ? ORDER BY id DESC LIMIT ?",
        (memory_id, int(limit)),
    ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------- archive


def archive(memory_id, reason=None):
    conn = _conn()
    conn.execute(
        "UPDATE memory SET is_archived = 1, archived_reason = ?, "
        "updated_at = ? WHERE id = ?",
        (reason, now_iso(), memory_id),
    )
    conn.commit()


def restore(memory_id):
    conn = _conn()
    conn.execute(
        "UPDATE memory SET is_archived = 0, archived_reason = NULL, "
        "superseded_by = NULL, updated_at = ? WHERE id = ?",
        (now_iso(), memory_id),
    )
    conn.commit()
    return get(memory_id)


def delete(memory_id):
    conn = _conn()
    conn.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
    conn.commit()


def forget_all(include_archived=True):
    """Wipe the store. Used by `/memory reset`."""
    conn = _conn()
    if include_archived:
        conn.execute("DELETE FROM memory_access")
        conn.execute("DELETE FROM memory")
        # Verdicts are keyed on the text of a pair that no longer exists. Left
        # behind, they would be inherited by whatever is stored under those
        # words next, answering a question about memories that are gone.
        conn.execute("DELETE FROM consolidation_decisions")
        # Captures that have not been folded in yet are memories the store does
        # not have but the user asked to keep. Leaving them queued would make
        # "reset" lie: the next agent start would drain them straight back in,
        # restoring precisely the rows that were just erased. `memory_vector`
        # needs no explicit delete - it cascades from `memory`.
        conn.execute("DELETE FROM memory_inbox")
    else:
        conn.execute("DELETE FROM memory WHERE is_archived = 0")
    conn.commit()


# ----------------------------------------------------------------- stats


def stats():
    conn = _conn()

    totals = conn.execute(
        f"SELECT count(*) AS live, "
        f"COALESCE(sum(access_count), 0) AS uses, "
        f"COALESCE(avg(confidence_score), 0) AS avg_conf "
        f"FROM memory WHERE {LIVE_PREDICATE}"
    ).fetchone()

    by_type = conn.execute(
        f"SELECT memory_type, count(*) AS n FROM memory WHERE {LIVE_PREDICATE} "
        "GROUP BY memory_type ORDER BY n DESC"
    ).fetchall()

    archived = conn.execute(
        "SELECT count(*) AS n FROM memory WHERE is_archived = 1"
    ).fetchone()["n"]

    top = conn.execute(
        f"SELECT id, text, memory_type, access_count, confidence_score "
        f"FROM memory WHERE {LIVE_PREDICATE} "
        "ORDER BY access_count DESC, confidence_score DESC LIMIT 5"
    ).fetchall()

    return {
        "live": totals["live"] or 0,
        "archived": archived,
        "total_uses": totals["uses"] or 0,
        "avg_confidence": round(totals["avg_conf"] or 0.0, 3),
        "by_type": {row["memory_type"]: row["n"] for row in by_type},
        "most_used": [dict(row) for row in top],
        "schema_version": db.schema_version(conn),
    }
