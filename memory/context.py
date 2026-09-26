"""Turning memories into prompt text.

Retrieval decides *what* is relevant; this decides *how much* of it the model
gets to see. The v1 code appended every retrieved memory to the system prompt
with no ceiling, so a chatty agent could inject an unbounded block and blow
the context window for a question that needed two facts.

Rules here:
* a hard character budget, spent in score order (not insertion order, which
  would starve the best match if an early filler took the budget)
* grouping by type, so the model reads identity/instruction before trivia
* ids included, so `forget` works and so the model can tell two similar
  memories apart
"""

from .text import summarize

# Generous enough for ~25 short memories, small enough to be invisible next to
# the system prompt.
DEFAULT_BUDGET = 2400

# Never let low-value trivia crowd out high-value identity/instruction lines.
MIN_IMPORTANCE = 0.25

_TYPE_ORDER = [
    "identity",
    "instruction",
    "preference",
    "goal",
    "project",
    "fact",
    "event",
    "session",
]


def sort_entries(results):
    """Best score first, ties broken by importance then recency.

    Recency is applied as a *separate* stable sort rather than as a third
    ascending key on the same tuple. ISO-8601 strings sort chronologically, so
    putting `last_accessed_at` in ascending position in a tuple that is
    otherwise descending made the *oldest* of two equally-scored memories win
    the tie - the opposite of what the tie-break is for. Two stable sorts give
    the intended order without having to negate a string.
    """
    by_recency = sorted(
        results,
        key=lambda entry: entry["item"].last_accessed_at or "",
        reverse=True,
    )
    return sorted(
        by_recency,
        key=lambda entry: (
            -entry["score"],
            -(entry["item"].importance or 0.0),
        ),
    )


def build_context(
    results,
    budget=DEFAULT_BUDGET,
    min_importance=MIN_IMPORTANCE,
    include_ids=True,
    heading="What you know about the user",
):
    """Render retrieval results as a prompt block. Returns "" if nothing fits."""
    if not results:
        return ""

    ordered = sort_entries(results)

    # Low-importance entries are dropped first, but only if something better
    # survives: hard-dropping them unconditionally would throw away genuinely
    # relevant session state on a quiet query.
    preferred = [
        entry for entry in ordered if float(entry["item"].importance or 0.0) >= min_importance
    ]
    candidates = preferred or ordered

    lines = []
    used = len(heading) + 1

    for entry in candidates:
        line = _format_line(entry, include_ids=include_ids)
        cost = len(line) + 1
        if lines and used + cost > budget:
            break
        lines.append(line)
        used += cost

    if not lines:
        return ""

    return f"{heading}:\n" + "\n".join(lines)


def _format_line(entry, include_ids=True):
    item = entry["item"]
    tag = f"[{item.id}]" if include_ids and item.id else "[ ]"
    parts = [f"{tag} ({item.memory_type})"]

    if item.subject:
        parts.append(f"<{item.subject}>")

    text = summarize(item.text, limit=220)
    parts.append(text)

    confidence = float(item.confidence or 0.0)
    if confidence < 0.5:
        # Tell the model when it is working from a memory it should hedge on.
        parts.append(f"(confidence: {confidence:.0%})")

    return "- " + " ".join(parts)


def build_system_memory_block(results, budget=DEFAULT_BUDGET):
    """The block the agent injects into the system prompt."""
    if not results:
        return ""

    block = build_context(results, budget=budget)

    guidance = (
        "Use these only when relevant to the current request. "
        "If one contradicts what the user just said, trust the user. "
        "When you learn something durable and non-obvious that is not "
        "derivable from the files, call remember()."
    )
    return f"{block}\n{guidance}"


def group_by_type(results):
    """Type-grouped view, for the terminal `/memories` command."""
    buckets = {}
    for entry in sort_entries(results):
        buckets.setdefault(entry["item"].memory_type, []).append(entry)

    ordered = [t for t in _TYPE_ORDER if t in buckets]
    ordered += [t for t in buckets if t not in _TYPE_ORDER]
    return [(t, buckets[t]) for t in ordered]


def render_table(results, include_archived=False):
    """Plain-text listing for the CLI."""
    lines = []
    for memory_type, entries in group_by_type(results):
        lines.append(f"{memory_type.upper()}  ({len(entries)})")
        for entry in entries:
            item = entry["item"]
            flag = " " if item.is_live else "x"
            lines.append(
                f"  {flag} #{item.id:<4} {item.confidence:.0%} "
                f"used={item.access_count:<3} {summarize(item.text, limit=90)}"
            )
        lines.append("")

    if include_archived:
        archived = [i for i in results if not i.is_live]
        if archived:
            lines.append(f"ARCHIVED  ({len(archived)})")
            for item in archived:
                lines.append(
                    f"  x #{item.id:<4} {item.archived_reason or 'archived'}: "
                    f"{summarize(item.text, limit=80)}"
                )
    return "\n".join(lines).rstrip()
