"""Retrieval: hybrid lexical search, salience reranking, and diversification.

The v1 retriever was a single TF-IDF cosine over whole memories, filtered by
`score > 0`. That is weak in three specific ways:

* IDF went to zero for any term appearing in every memory, so the most common
  topic in the store became invisible.
* There was no floor, so recall was all-or-nothing and a slightly-off query
  returned nothing while reinforcement still fired.
* Ranking ignored *whether a memory is any good*. A fact relevant five months
  ago outranked a preference the user stated last week.

So: two independent lexical rankers, fused by Reciprocal Rank Fusion, then
reranked by a salience prior, then diversified so the prompt does not receive
five paraphrases of the same fact.

    BM25 (FTS5, porter stemming)  ─┐
                                    ├─ RRF ─→ lexical ─→ × salience ─→ MMR
    TF-IDF cosine (hand-rolled)   ─┘

RRF is used deliberately instead of adding the raw scores: BM25 and cosine
live on incomparable scales, and rank-fusion needs no normalization constants.
"""

import math

from . import db, store
from .clock import days_since
from .text import fts_query, tokenize

# Reciprocal-rank-fusion smoothing constant. 60 is the value from the original
# RRF paper and is insensitive within a wide band.
RRF_K = 60

# Final score gate. Below this a memory is not worth prompt tokens.
MIN_SCORE = 0.05

# Salience prior bounds: the lexical signal always dominates, salience can
# swing the result by at most +/-30%.
SALIENCE_WEIGHT = 0.30
SALIENCE_BIAS = 1.0 - SALIENCE_WEIGHT

# The lexical term is squared before the prior is applied.
#
# `lexical` is normalised against the best match *in this query*, so it sits
# near 1.0 for anything that matched at all - a 0.97 and a 1.00 differ by 3%.
# A prior of +/-30% therefore used to decide those cases on its own, and since
# identity memories carry the highest importance they outranked the actually
# relevant memory on unrelated questions: asked about package managers, the
# user's *name* came first and the package manager was pushed out of the
# prompt entirely. Squaring turns "matched weakly" into a real gap while
# leaving near-ties alone, so salience breaks ties and never carries a weak
# match past a strong one.
LEXICAL_SHARPNESS = 2.0

# Half-life (days) of the recency term inside salience. Independent of a
# memory's own decay, since this measures "how stale is this in the user's
# current focus".
RECENCY_HALFLIFE = 45.0

# Maximal Marginal Relevance trade-off. 1.0 = pure relevance, 0 = pure
# diversity. 0.7 keeps strong matches while still avoiding near-duplicates.
MMR_LAMBDA = 0.7

MAX_CANDIDATES = 200


# ------------------------------------------------------------ lexical: FTS


def fts_search(query, limit=MAX_CANDIDATES):
    """BM25 rank over the FTS5 index. Returns [(id, rank_index)].

    Tries an AND chain first for precision, then falls back to OR. A prefix
    query can only extend the query term, so a memory stored as "allergic"
    is unreachable from the query "allergy" under AND - exactly the case a
    user hits when they phrase a question differently from last time.

    `bm25()` in SQLite returns *lower is better* and negative values, so the
    ordering is ascending and we invert to a 0-based rank afterwards.
    """
    for mode in ("AND", "OR"):
        match = fts_query(query, mode=mode)
        if not match:
            continue

        rows = _run_fts(match, limit)
        if rows:
            return [(row["id"], pos) for pos, row in enumerate(rows)]

    return []


def _run_fts(match, limit):
    conn = db.connect()
    try:
        return conn.execute(
            f"""
            SELECT m.id AS id, bm25(memory_fts) AS rank
            FROM memory_fts
            JOIN memory m ON m.id = memory_fts.rowid
            WHERE memory_fts MATCH ?
              AND m.is_archived = 0
              AND m.superseded_by IS NULL
            ORDER BY rank ASC
            LIMIT ?
            """,
            (match, int(limit)),
        ).fetchall()
    except Exception:
        # A malformed MATCH or a missing index must never take down the agent.
        return []


def fts_available():
    return db.has_fts5(db.connect())


# ---------------------------------------------------------- lexical: TF-IDF


def _term_frequency(tokens):
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _document_frequency(documents):
    frequency = {}
    for tokens in documents:
        for token in set(tokens):
            frequency[token] = frequency.get(token, 0) + 1
    return frequency


def _inverse_document_frequency(frequency, total):
    """Smoothed IDF.

    v1 used `log(N / df)`, which is exactly 0 for a term present in every
    document, and would go negative if df ever exceeded N. `log(1 + N / df)`
    is always positive, so shared vocabulary still contributes evidence
    instead of being silently erased.
    """
    idf = {}
    for term, df in frequency.items():
        idf[term] = math.log(1.0 + (total / max(df, 1)))
    return idf


def _vectorize(tokens, idf):
    """Sublinear TF weighting, restricted to the known vocabulary."""
    tf = _term_frequency(tokens)
    vector = {}
    for term, idf_value in idf.items():
        count = tf.get(term, 0)
        if count:
            vector[term] = (1.0 + math.log(count)) * idf_value
    return vector


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0

    dot = 0.0
    magnitude_a = 0.0
    magnitude_b = 0.0

    for term, value in a.items():
        magnitude_a += value * value
        other = b.get(term)
        if other is not None:
            dot += value * other

    for value in b.values():
        magnitude_b += value * value

    if magnitude_a <= 0 or magnitude_b <= 0:
        return 0.0

    return dot / (math.sqrt(magnitude_a) * math.sqrt(magnitude_b))


def tfidf_search(query, items, limit=MAX_CANDIDATES):
    """Cosine rank over the in-memory candidate set. Returns [(id, rank)]."""
    query_tokens = tokenize(query)
    if not query_tokens or not items:
        return []

    documents = [tokenize(item.text) for item in items]
    idf = _inverse_document_frequency(_document_frequency(documents), len(documents))
    query_vector = _vectorize(query_tokens, idf)

    if not query_vector:
        return []

    scored = []
    for item, tokens in zip(items, documents):
        score = cosine_similarity(query_vector, _vectorize(tokens, idf))
        if score > 0:
            scored.append((item.id, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [(mid, pos) for pos, (mid, _score) in enumerate(scored[:limit])]


# ----------------------------------------------------------------- fusion


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """RRF: sum 1/(k + rank) across rankers.

    Only ranks matter, not the underlying scores, which is what lets BM25 and
    cosine be combined without pretending they share a scale.
    """
    fused = {}
    for ranking in rankings:
        for memory_id, rank in ranking:
            if rank < 0:
                continue
            fused[memory_id] = fused.get(memory_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


# --------------------------------------------------------------- salience


def salience(item, now=None):
    """How much this memory deserves attention, independent of the query.

    Four signals, all normalized to [0, 1]:
      confidence  - has it survived contradiction and time
      importance  - what kind of memory it is
      recency     - is it in the user's current focus
      usage       - has it proven useful before
    """
    confidence = max(0.0, min(1.0, float(item.confidence or 0.0)))
    importance = max(0.0, min(1.0, float(item.importance or 0.0)))
    recency = math.exp(-days_since(item.last_accessed_at, now=now) / RECENCY_HALFLIFE)
    usage = math.log1p(item.access_count or 0) / math.log(11.0)  # ~11 uses = 1.0
    usage = max(0.0, min(1.0, usage))

    score = (
        0.40 * confidence
        + 0.30 * importance
        + 0.20 * recency
        + 0.10 * usage
    )

    return score, {
        "confidence": round(confidence, 3),
        "importance": round(importance, 3),
        "recency": round(recency, 3),
        "usage": round(usage, 3),
    }


# ------------------------------------------------------------------- MMR


def _similarity(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mmr_diversify(candidates, top_k, lambda_=MMR_LAMBDA):
    """Maximal Marginal Relevance.

    Greedily take the item maximizing `lambda * relevance - (1 - lambda) *
    max similarity to what is already selected`. Stops five paraphrases of the
    same preference from all occupying the context window.
    """
    if len(candidates) <= top_k:
        return candidates

    tokenized = [(entry, tokenize(entry["item"].text)) for entry in candidates]
    selected = []
    remaining = list(tokenized)

    while remaining and len(selected) < top_k:
        best_index, best_value = 0, float("-inf")

        for index, (entry, tokens) in enumerate(remaining):
            relevance = entry["score"]
            if selected:
                redundancy = max(
                    _similarity(tokens, other_tokens) for _, other_tokens in selected
                )
            else:
                redundancy = 0.0
            value = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if value > best_value:
                best_index, best_value = index, value

        entry, tokens = remaining.pop(best_index)
        entry = dict(entry)
        entry["mmr"] = round(best_value, 4)
        selected.append((entry, tokens))

    return [entry for entry, _ in selected]


# -------------------------------------------------------------- retrieval


def retrieve(
    query,
    top_k=6,
    items=None,
    scope=None,
    min_score=MIN_SCORE,
    reinforce=True,
    diversify=True,
    now=None,
):
    """Find the memories most worth putting in front of the model.

    Returns a list of dicts: ``{"item": MemoryItem, "score": float,
    "signals": {...}}`` ordered best-first.
    """
    pool = items if items is not None else store.get_active(limit=MAX_CANDIDATES)

    if scope:
        pool = [
            item
            for item in pool
            if item.scope == scope or item.scope == "global"
        ]
    if not pool:
        return []

    query_tokens = tokenize(query)

    # No usable query terms (greeting, "/help", pure punctuation): fall back to
    # "what is most worth knowing about this user right now".
    if not query_tokens:
        return _salience_only(pool, top_k, reinforce=reinforce, now=now)

    by_id = {item.id: item for item in pool}

    rankings = [fts_search(query, limit=MAX_CANDIDATES)]
    rankings.append(tfidf_search(query, pool))

    fused = reciprocal_rank_fusion(rankings)

    if not fused:
        return []

    best_raw = max(fused.values()) or 1.0

    candidates = []
    for memory_id, raw in fused.items():
        item = by_id.get(memory_id)
        if item is None:
            # Matched the index but is not in the live pool (stale row, or it
            # was filtered out by scope). Drop it.
            continue

        lexical = raw / best_raw  # top-ranked item in this query -> 1.0
        salience_score, signals = salience(item, now=now)
        final = (lexical ** LEXICAL_SHARPNESS) * (
            SALIENCE_BIAS + SALIENCE_WEIGHT * salience_score
        )

        signals["lexical"] = round(lexical, 3)
        signals["salience"] = round(salience_score, 3)

        if final < min_score:
            continue

        candidates.append(
            {
                "item": item,
                "score": round(final, 4),
                "signals": signals,
            }
        )

    if not candidates:
        return []

    candidates.sort(key=lambda entry: entry["score"], reverse=True)
    candidates = candidates[:MAX_CANDIDATES]

    if diversify:
        candidates = mmr_diversify(candidates, top_k)

    results = candidates[:top_k]

    if reinforce:
        for entry in results:
            store.reinforce(entry["item"].id)
            store.log_access(
                entry["item"].id, query=query, score=entry["score"]
            )
            entry["item"] = store.get(entry["item"].id)

    return results


def _salience_only(pool, top_k, reinforce=True, now=None):
    """Fallback ranking with no query: pure salience, no reinforcement.

    Reinforcing on a fallback would mark every core memory as "used" just
    because the user said hello, which corrupts the signal that tells us which
    memories matter.
    """
    ranked = []
    for item in pool:
        salience_score, signals = salience(item, now=now)
        signals["lexical"] = 0.0
        signals["salience"] = round(salience_score, 3)
        ranked.append(
            {"item": item, "score": round(salience_score, 4), "signals": signals}
        )

    ranked.sort(key=lambda entry: entry["score"], reverse=True)
    return ranked[:top_k]


def explain(results):
    """Human-readable justification, for the `/recall` debug command."""
    lines = []
    for rank, entry in enumerate(results, 1):
        signals = entry.get("signals", {})
        lines.append(
            "#{rank} score={score:.3f} lexical={lex:.3f} sal={sal:.3f} "
            "[conf={c} imp={i} rec={r} use={u}] type={t} id={id} :: {text}".format(
                rank=rank,
                score=entry["score"],
                lex=signals.get("lexical", 0.0),
                sal=signals.get("salience", 0.0),
                c=signals.get("confidence", 0.0),
                i=signals.get("importance", 0.0),
                r=signals.get("recency", 0.0),
                u=signals.get("usage", 0.0),
                t=entry["item"].memory_type,
                id=entry["item"].id,
                text=entry["item"].text,
            )
        )
    return "\n".join(lines)


def keyword_overlap(query, text):
    """Shared-token ratio. Cheap sanity check that a hit is not a coincidence."""
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)
