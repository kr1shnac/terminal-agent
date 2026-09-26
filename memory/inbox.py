"""Capture memories while the agent is not running.

A conversation that happens while the agent is closed cannot be observed by a
process that does not exist, so "the app was off" and "the fact was never
recorded" are the same event from the store's point of view. This module is the
other half of persistence: a durable queue that accepts a memory at any time
from any process, and folds it into the real store the next time the agent
opens the database.

The queue is deliberately dumb - no extraction, no model calls, no merge
decisions. Whatever the caller hands over is stored verbatim on the way in and
classified on the way out, so capturing a memory can never fail because a
network call did, and re-running it can never invent a memory the user did not
state. Capturing text that turns out to be already known is not an error either:
:meth:`drain` reports the existing row's id and marks the entry consumed, so
the same fact queued twice costs one no-op rather than a duplicate.
"""

from . import db, store
from .clock import now_iso

#: Sources are free-form labels recorded on both the queue entry and the memory
#: it becomes, so a row can always be traced back to the process that wrote it.
DEFAULT_SOURCE = "cli"

#: Refuse absurd entries rather than let one bad argument eat the queue's
#: memory and slow every future drain. A real memory is a sentence or two.
MAX_CAPTURE_CHARS = 4000


class CaptureError(ValueError):
    """Raised when a capture request is not something we can store."""


def _conn():
    return db.connect()


def capture(text, source=DEFAULT_SOURCE, session_id=None, note=None):
    """Queue one memory for later. Returns the queue entry id.

    Safe to call from any process, including while the agent is running: the
    queue is a table in the same database, and every statement below is
    short-lived and transactional, so this does not contend with the agent's
    own writes beyond ordinary SQLite locking.
    """
    if text is None:
        raise CaptureError("nothing to capture")

    cleaned = " ".join(str(text).split()).strip()
    if not cleaned:
        raise CaptureError("nothing to capture")
    if len(cleaned) > MAX_CAPTURE_CHARS:
        raise CaptureError(
            f"capture is {len(cleaned)} characters; the limit is {MAX_CAPTURE_CHARS}"
        )

    conn = _conn()
    cursor = conn.execute(
        """INSERT INTO memory_inbox (text, source, session_id, captured_at, status)
           VALUES (?, ?, ?, ?, 'pending')""",
        (cleaned, str(source or DEFAULT_SOURCE), session_id, now_iso()),
    )
    conn.execute(
        "UPDATE memory_inbox SET note = ? WHERE id = ?",
        (note, cursor.lastrowid),
    )
    conn.commit()
    return int(cursor.lastrowid)


def pending(limit=50):
    """Oldest queued entries first, so draining cannot starve old captures."""
    rows = _conn().execute(
        """SELECT id, text, source, session_id, captured_at
             FROM memory_inbox
            WHERE status = 'pending'
            ORDER BY id
            LIMIT ?""",
        (int(limit),),
    ).fetchall()
    return [dict(row) for row in rows]


def count_pending():
    row = _conn().execute(
        "SELECT COUNT(*) AS n FROM memory_inbox WHERE status = 'pending'"
    ).fetchone()
    return int(row["n"] if row else 0)


def drain(limit=100, memory_type="fact", scope="global"):
    """Turn queued entries into stored memories.

    Returns a report dict::

        {"processed": n, "stored": n, "duplicates": n, "failed": n, ...}

    Each entry is marked ``drained`` with the resulting ``memory_id``, or
    ``error`` with the reason, so a failure is visible in ``/memory inbox`` and
    can be retried rather than silently lost. Entries that fail are left in the
    queue as ``error`` rather than deleted: losing a fact the user explicitly
    asked to keep, because one insert hit a locked file, is the one outcome this
    module must never produce.
    """
    conn = _conn()
    rows = conn.execute(
        """SELECT id, text, source, session_id FROM memory_inbox
            WHERE status = 'pending' ORDER BY id LIMIT ?""",
        (int(limit),),
    ).fetchall()

    report = {
        "processed": 0,
        "stored": 0,
        "duplicates": 0,
        "failed": 0,
        "entries": [],
    }

    for row in rows:
        report["processed"] += 1
        entry = {
            "inbox_id": int(row["id"]),
            "text": row["text"],
            "source": row["source"],
        }
        try:
            # `insert_memory` returns the *existing* row when the text is
            # already known, so the write is a no-op either way; asking first is
            # what lets the report tell the two apart.
            duplicate = store.find_duplicate(row["text"]) is not None
            item = store.insert_memory(
                row["text"],
                memory_type,
                scope=scope,
                source=f"inbox:{row['source'] or DEFAULT_SOURCE}",
            )
            if item is None:
                raise CaptureError("store rejected the entry")
            entry["memory_id"] = item.id
            entry["duplicate"] = duplicate
            report["duplicates" if duplicate else "stored"] += 1
            status, memory_id, note = "drained", item.id, None
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the queue
            entry["error"] = str(exc)
            report["failed"] += 1
            status, memory_id, note = "error", None, str(exc)[:500]

        conn.execute(
            """UPDATE memory_inbox
                  SET status = ?, drained_at = ?, memory_id = ?, note = ?
                WHERE id = ?""",
            (status, now_iso(), memory_id, note, row["id"]),
        )
        conn.commit()
        report["entries"].append(entry)

    return report


def summarize(report):
    """One-line human summary of a :func:`drain` report."""
    if not report or not report.get("processed"):
        return "inbox: nothing queued"
    parts = [f"inbox: {report['processed']} captured"]
    if report["stored"]:
        parts.append(f"{report['stored']} stored")
    if report["duplicates"]:
        parts.append(f"{report['duplicates']} already known")
    if report["failed"]:
        parts.append(f"{report['failed']} failed")
    return ", ".join(parts)


def stats():
    """Counts per status, for `/memory inbox` and the persistence report."""
    rows = _conn().execute(
        "SELECT status, COUNT(*) AS n FROM memory_inbox GROUP BY status"
    ).fetchall()
    counts = {row["status"]: int(row["n"]) for row in rows}
    counts.setdefault("pending", 0)
    return counts
