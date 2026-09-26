"""Consolidation: housekeeping that keeps the store from rotting.

Retrieval quality is mostly a storage-quality problem. If the same fact exists
in five rows, the retriever spends its whole budget on one idea and the model
reads it as five independent pieces of evidence.

The hard part is telling a *paraphrase* from a *contradiction*:

    "The user prefers pnpm over npm"
    "User prefers pnpm, not npm"       -> same fact, merge
    "The user prefers yarn over npm"    -> different fact, keep both

No bag-of-words measure settles that. Token overlap calls all three similar
(~0.6-0.7). IDF weighting helps and is used as a filter, but on a small store
the IDF estimates are too noisy to trust. So consolidation is tiered:

1. near-verbatim restatements (very high weighted overlap) merge immediately,
   no model call;
2. same-`subject` rows are resolved deterministically by recency - a subject
   is a slot that holds one value, so a newer value supersedes;
3. the ambiguous middle band is adjudicated by a cheap LLM call, batched, and
   only when a client is available. Without a key, that band is simply left
   alone: fewer merges, but never a wrong one.
"""

import json
import re

from . import store
from .clock import now_iso
from .text import build_idf, keywords, norm_hash, similarity, tokenize

# At or above this, merge without asking anything.
AUTO_MERGE_THRESHOLD = 0.88
# Below this, not even worth asking.
ASK_FLOOR = 0.58
# Pairs handed to the model in one call. Consolidation is rare; batching keeps
# it to a single request.
MAX_PAIRS = 12

# New questions bought in one pass. Verdicts already on file are free, so this
# caps the recurring cost of a store whose pairs sit permanently in the
# ambiguous band rather than the first pass that discovers them.
MAX_NEW_QUESTIONS = 12

# Below this confidence a memory is not worth spending a prompt slot on.
LOW_VALUE_THRESHOLD = 0.12

ADJUDICATE_PROMPT = """\
You are deduplicating a long-term memory store.

Each numbered candidate is a PAIR of statements that may or may not assert the
same fact about the user. For each pair, decide SAME or DIFFERENT.

SAME: paraphrases, rewordings, or a more specific restatement of the other.
"The user prefers pnpm over npm" and "User prefers pnpm, not npm" are the same
fact.

DIFFERENT: the value, name, or claim actually differs. "The user prefers pnpm
over npm" and "The user prefers yarn over npm" are different facts. So are
statements about different subjects that merely share vocabulary.

Return ONLY JSON of this exact shape:

{"same": [1, 3]}

listing the numbers of the candidates that assert the SAME fact. Return
{"same": []} if none do. Judge only what is written; do not guess.
"""


def consolidate(dry_run=False, client=None, model=None):
    """Run one consolidation pass. Returns a report dict."""
    report = {
        "checked": 0,
        "merged": [],
        "superseded": [],
        "pruned": [],
        "adjudicated": 0,
        "skipped": [],
        "dry_run": dry_run,
    }

    items = store.get_active()
    report["checked"] = len(items)

    # One IDF table for the whole pass, so a word is weighted identically in
    # every comparison.
    idf = build_idf([tokenize(item.text) for item in items])

    auto, ambiguous = _plan(items, idf)
    report["adjudicated"] = len(ambiguous)

    if ambiguous and client is not None and model and not dry_run:
        auto += _adjudicate(ambiguous, client, model)

    merges, skipped = _apply_merges(auto)
    report["skipped"] = [
        {"drop_id": drop.id, "drop_text": drop.text, "reason": "keeper_retired"}
        for _keep, drop in skipped
    ]

    for keep, drop in merges:
        report["merged"].append(
            {
                "kept_id": keep.id,
                "kept_text": keep.text,
                "merged_id": drop.id,
                "merged_text": drop.text,
            }
        )
        if not dry_run:
            _merge_into(keep, drop)

    if not dry_run:
        report["superseded"] = _resolve_subject_conflicts()
        report["pruned"] = _prune_debris()

    return report


def _apply_merges(pairs):
    """Order the merges so nothing is ever folded into a retired row.

    `_plan` guarantees the *automatic* merges are well formed - the survivor
    is always a keeper and never itself a drop. The adjudicated pairs break
    that: an ambiguous pair's `drop` is also left in the survivor list, so it
    can later be chosen as somebody else's keeper. Applying such a pair after
    its keeper was already archived silently destroys the memory: the row is
    gone and the text that was folded into it went with it.

    So a pair whose keeper is itself being retired is re-pointed at whatever
    survives that chain, and only dropped if the whole chain collapses.
    """
    by_id = {}
    parent = {}
    for keep, drop in pairs:
        by_id[keep.id] = keep
        by_id[drop.id] = drop
        parent[drop.id] = keep.id

    retired = set()
    applied, skipped = [], []

    for keep, drop in pairs:
        if keep.id == drop.id:
            continue

        target = keep
        seen = set()
        while target.id in parent and target.id in retired:
            if target.id in seen:  # cycle: give up rather than spin
                break
            seen.add(target.id)
            target = by_id.get(parent[target.id], target)
            if target is None or target.id in seen:
                break

        if target.id in retired or target.id == drop.id:
            skipped.append((keep, drop))
            continue

        applied.append((target, drop))
        retired.add(drop.id)

    return applied, skipped


def _plan(items, idf):
    """Split candidate pairs into 'certain' merges and 'ask the model' ones.

    Pairs are formed against the survivors only, so three copies of one fact
    collapse to a single row instead of ping-ponging between merges.
    """
    auto, ambiguous = [], []

    survivors = []
    for item in items:
        tokens = tokenize(item.text)
        if not tokens:
            survivors.append(item)
            continue

        scored = []
        for candidate in survivors:
            if candidate.memory_type != item.memory_type:
                continue
            if item.subject and item.subject == candidate.subject:
                # A shared subject means a shared slot: recency decides, not
                # similarity. Handled by _resolve_subject_conflicts.
                continue
            scored.append(
                (candidate, similarity(tokens, tokenize(candidate.text), idf))
            )

        scored = [(c, s) for c, s in scored if s >= ASK_FLOOR]
        if not scored:
            survivors.append(item)
            continue

        best, best_score = max(scored, key=lambda pair: pair[1])

        if best_score >= AUTO_MERGE_THRESHOLD:
            auto.append((best, item))
        else:
            ambiguous.append((best, item))
            survivors.append(item)

    return auto, ambiguous[:MAX_PAIRS]


def _pair_key(keep, drop):
    """Stable key for a pair's *text*, so rewording reopens the question."""
    left, right = sorted([norm_hash(keep.text), norm_hash(drop.text)])
    return f"{left}:{right}"


def _cached_verdict(key):
    from . import db

    row = db.connect().execute(
        "SELECT same FROM consolidation_decisions WHERE pair_hash = ?", (key,)
    ).fetchone()
    if row is None:
        return None
    return bool(row[0])


def _remember_verdict(key, same):
    from . import db

    conn = db.connect()
    conn.execute(
        "INSERT OR REPLACE INTO consolidation_decisions(pair_hash, same, decided_at) "
        "VALUES (?, ?, ?)",
        (key, 1 if same else 0, now_iso()),
    )
    conn.commit()


def _adjudicate(pairs, client, model):
    """Ask a cheap model which ambiguous pairs assert the same fact.

    Verdicts already on file are reused. A "different" answer leaves both
    memories live, so without the cache the identical question is re-bought on
    every future pass and the store never stops paying for it.
    """
    verdicts = []

    pending = []
    for pair in pairs:
        key = _pair_key(*pair)
        known = _cached_verdict(key)
        if known is None:
            pending.append((pair, key))
        elif known:
            verdicts.append(pair)

    # No model, or nothing new to ask: keep the cached merges, drop the rest.
    if client is None or not pending:
        return verdicts

    limited = pending[:MAX_NEW_QUESTIONS]
    for _pair, key in pending[MAX_NEW_QUESTIONS:]:
        # Remember the cap so the leftover is not re-offered next pass.
        _remember_verdict(key, False)

    fresh = _ask_model([pair for pair, _ in limited], client, model)

    if fresh is None:
        # The model was never reached. Caching now would turn a transient
        # outage into a permanent "these are different facts" verdict on a pair
        # nobody ever judged, so leave it retryable.
        return verdicts

    for index in fresh:
        verdicts.append(limited[index - 1][0])

    for index, (_pair, key) in enumerate(limited, 1):
        _remember_verdict(key, index in fresh)

    return verdicts


def _ask_model(pairs, client, model):
    """Return the 1-based candidate numbers the model called the same fact.

    Returns None when the call failed, which is deliberately distinct from an
    empty list meaning "the model answered: none of them are the same".
    """
    if not pairs:
        return []

    lines = []
    for index, (keep, drop) in enumerate(pairs, 1):
        lines.append(f"{index}. A: {keep.text}")
        lines.append(f"   B: {drop.text}")

    prompt = ADJUDICATE_PROMPT + "\nCandidates:\n" + "\n".join(lines)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        content = response.choices[0].message.content or ""
    except Exception:
        # A failed adjudication means "leave them alone", not "merge them".
        return None

    verdicts = _parse_same_pairs(content)
    return [n for n in verdicts if 1 <= n <= len(pairs)]


def _parse_same_pairs(content):
    """Pull the list of candidate numbers out of a model reply.

    Tolerant of fenced code blocks, surrounding prose, and the older nested
    [[1, 2]] shape, which collapsed to its first index.
    """
    text = str(content or "").strip()
    if not text:
        return []

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()

    payload = None
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                payload = json.loads(text[start: end + 1])
            except (ValueError, TypeError):
                return []

    if not isinstance(payload, dict):
        return []

    same = payload.get("same")
    if same is None:
        return []
    if not isinstance(same, list):
        return []

    numbers = []
    for entry in same:
        if isinstance(entry, list) and entry:
            # Legacy [[1, 2]] form: the first element is the candidate.
            try:
                entry = entry[0]
            except (TypeError, ValueError):
                continue
        try:
            numbers.append(int(entry))
        except (TypeError, ValueError):
            continue
    return numbers


def _merge_into(keep, drop):
    """Fold `drop` into `keep`: keep the stronger statement of the two."""
    if len(drop.text) > len(keep.text) * 1.5:
        # A much longer restatement usually carries detail the shorter one
        # lost, so promote it rather than discarding it.
        store.update_text(keep.id, drop.text)

    store.update_fields(
        keep.id,
        importance=max(keep.importance or 0.0, drop.importance or 0.0),
    )
    store.archive(drop.id, reason=f"merged into #{keep.id}")


def _resolve_subject_conflicts():
    """Keep the newest live memory per subject; archive the rest."""
    resolved = []

    subjects = {}
    for item in store.get_active():
        if not item.subject:
            continue
        subjects.setdefault(item.subject, []).append(item)

    for subject, group in subjects.items():
        if len(group) < 2:
            continue

        group.sort(key=lambda i: (i.created_at or "", i.id or 0), reverse=True)
        winner, losers = group[0], group[1:]

        for loser in losers:
            resolved.append(
                {
                    "subject": subject,
                    "kept_id": winner.id,
                    "kept_text": winner.text,
                    "archived_id": loser.id,
                }
            )
            store.archive(loser.id, reason=f"superseded by #{winner.id}")

    return resolved


def _prune_debris():
    """Archive live rows that are neither trusted nor useful."""
    pruned = []
    for item in store.get_active():
        confidence = float(item.confidence or 0.0)
        importance = float(item.importance or 0.0)

        if not item.text or not item.text.strip():
            pruned.append({"id": item.id, "text": "(empty)"})
            store.archive(item.id, reason="empty")
        elif confidence < LOW_VALUE_THRESHOLD and importance < 0.35:
            pruned.append({"id": item.id, "text": item.text})
            store.archive(item.id, reason="low_value")

    return pruned


def summarize(report):
    """One-line summary for the terminal."""
    bits = [f"checked {report['checked']}"]
    if report["merged"]:
        bits.append(f"merged {len(report['merged'])}")
    if report["superseded"]:
        bits.append(f"superseded {len(report['superseded'])}")
    if report["pruned"]:
        bits.append(f"pruned {len(report['pruned'])}")
    if report.get("adjudicated"):
        bits.append(f"{report['adjudicated']} pairs left alone (ambiguous)")
    return ", ".join(bits)


def compact():
    """Housekeeping summary line for the CLI."""
    summary = store.stats()
    return (
        "{live} live, {archived} archived, avg confidence {avg_confidence}, "
        "{total_uses} recalls, schema v{schema_version}".format(**summary)
    )


def describe(item, limit=90):
    keys = ", ".join(keywords(item.text, limit=4))
    return f"#{item.id} ({item.memory_type}) [{keys}] {item.text[:limit]}"


def mark_run():
    from . import db

    conn = db.connect()
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) "
        "VALUES ('last_consolidation', ?)",
        (now_iso(),),
    )
    conn.commit()


def last_run():
    from . import db

    row = db.connect().execute(
        "SELECT value FROM schema_meta WHERE key = 'last_consolidation'"
    ).fetchone()
    return row["value"] if row else None
