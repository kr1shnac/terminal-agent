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

# How much of the final lexical score comes from score magnitude rather than
# rank position. See `fuse`: RRF on its own is too flat to separate a good match
# from a merely-early one.
MAGNITUDE_WEIGHT = 0.5

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

# Search strategies, most precise first. The first one that returns anything
# wins.
#
# Order matters more than it looks. The ladder used to be ("AND", "OR") with a
# prefix match on every term, and the prefix is what broke it: `"use"*` also
# matches "user", so a query as ordinary as "which package manager do I use"
# could never satisfy its AND chain, fell through to OR, and matched every
# memory in the store. BM25 has nothing to rank when everything matches, so
# the "best" row was just the first one SQLite emitted - which is how an
# unrelated note about "threads or processes" took the top slot for questions
# about editors, tabs and search tools.
#
# Leading with the exact, non-prefix AND chain fixes the common case outright:
# both sides of the comparison are stemmed by the same tokenizer, so "tabs"
# still finds "tabs" and "allergy" still finds "allergic" without a prefix.
# The prefix variants stay in the ladder purely as a recall backstop.
FTS_LADDER = (
    ("AND", False),
    ("AND", True),
    ("OR", False),
    ("OR", True),
)


def fts_search(query, limit=MAX_CANDIDATES):
    """BM25 rank over the FTS5 index. Returns [(id, rank, magnitude)].

    Walks :data:`FTS_LADDER` and stops at the first strategy that matches
    anything, so a query that fully matches is never diluted by a laxer
    strategy. Only the AND-to-OR widening is unconditional: a query where the
    user phrased things differently from last time would otherwise return
    nothing at all.

    `bm25()` in SQLite returns *lower is better* and negative values, so the
    ordering is ascending and we invert to a 0-based rank afterwards. The raw
    magnitude is carried out too: rank alone throws away how much better the
    top hit was, which is the difference between a real match and noise.
    """
    for mode, prefix in FTS_LADDER:
        match = fts_query(query, mode=mode, prefix=prefix)
        if not match:
            continue

        rows = _run_fts(match, limit)
        if rows:
            return [
                (row["id"], pos, abs(float(row["rank"] or 0.0)))
                for pos, row in enumerate(rows)
            ]

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
    """Sublinear TF weighting, restricted to the known vocabulary.

    Walks the document's tokens and looks each one up, rather than walking the
    whole vocabulary and asking whether the document contains it. Same result,
    but the cost drops from `len(doc) * len(vocab)` to `len(doc)` per document
    - which is the difference between a retriever that fits in a turn's budget
    and one that does not once the store grows past a few hundred memories.
    """
    vector = {}
    for term, count in _term_frequency(tokens).items():
        idf_value = idf.get(term)
        if idf_value:
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
    """Cosine rank over the in-memory candidate set.

    Returns [(id, rank, cosine)].
    """
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
    return [
        (mid, pos, score) for pos, (mid, score) in enumerate(scored[:limit])
    ]


# ----------------------------------------------------------------- fusion

# A ranker whose top hit is not meaningfully better than its worst is handing
# back noise, not a ranking. This is the relative spread below which its
# magnitudes are ignored and only its positions are trusted. Measured against
# a query that matched every row in the store, where BM25 returns ~0 for all of
# them and the "ranking" is really just insertion order.
MIN_MAGNITUDE_SPREAD = 0.02


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """RRF: sum 1/(k + rank) across rankers.

    Only ranks matter, not the underlying scores, which is what lets BM25 and
    cosine be combined without pretending they share a scale.
    """
    fused = {}
    for ranking in rankings:
        for entry in ranking:
            memory_id, rank = entry[0], entry[1]
            if rank < 0:
                continue
            fused[memory_id] = fused.get(memory_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _magnitude_signal(ranking):
    """Per-id strength in [0, 1] from a ranking's own score magnitudes.

    Returns None when the ranking does not actually discriminate, so that a
    degenerate result set cannot masquerade as a confident one.
    """
    if not ranking:
        return None

    magnitudes = [abs(entry[2]) for entry in ranking if entry[1] >= 0]
    if not magnitudes:
        return None

    peak, trough = max(magnitudes), min(magnitudes)
    if peak <= 0 or (peak - trough) / peak < MIN_MAGNITUDE_SPREAD:
        return None

    return {entry[0]: abs(entry[2]) / peak for entry in ranking if entry[1] >= 0}


def fuse(rankings, k=RRF_K):
    """Combine the rankers into one 0..1 lexical score per memory.

    Rank fusion alone turned out to be too flat to rank on. RRF only sees
    positions, so a row that BM25 scored a tenth as highly as the winner and a
    row that merely happened to come first in an unrankable result set both
    land within a factor of two of the leader - and the +/-30% salience prior
    then decided between them. Scoring close to half the weight on the
    normalized magnitudes lets an actually-better match win.

    Only rankers that discriminate contribute a magnitude term (see
    :func:`_magnitude_signal`), so a query that matched the whole store
    degrades to plain rank fusion rather than to noise.
    """
    fused = reciprocal_rank_fusion(rankings, k=k)

    strength = {}
    for ranking in rankings:
        signal = _magnitude_signal(ranking)
        if not signal:
            continue
        for memory_id, value in signal.items():
            strength[memory_id] = strength.get(memory_id, 0.0) + value

    if not fused:
        return {}

    best_rank = max(fused.values()) or 1.0
    best_strength = max(strength.values()) if strength else 0.0

    lexical = {}
    for memory_id, raw in fused.items():
        by_rank = raw / best_rank
        if best_strength > 0:
            by_score = strength.get(memory_id, 0.0) / best_strength
            lexical[memory_id] = (1.0 - MAGNITUDE_WEIGHT) * by_rank + MAGNITUDE_WEIGHT * by_score
        else:
            lexical[memory_id] = by_rank

    return lexical


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

    if not pool:
        return []

    query_tokens = tokenize(query)

    # No usable query terms (greeting, "/help", pure punctuation): fall back to
    # "what is most worth knowing about this user right now".
    if not query_tokens:
        return _salience_only(pool, top_k, reinforce=reinforce, now=now)

    by_id = {item.id: item for item in pool}

    fts_ranking = fts_search(query, limit=MAX_CANDIDATES)
    rankings = [fts_ranking, tfidf_search(query, pool)]

    # The candidate pool is capped so the in-memory cosine stays cheap, but
    # BM25 legitimately finds the best match in a store that is larger than the
    # cap. `by_id` was built from the pool alone, so every such hit hit the
    # "matched the index but is not in the live pool" branch below and was
    # dropped - which silently capped recall at MAX_CANDIDATES no matter how
    # good the match was, and got worse the more the user had stored. Pull the
    # out-of-pool winners in explicitly. The scope filter still runs after
    # this, so a filtered row cannot sneak back in through the index.
    if items is None:
        outside = [mid for mid, _r, _m in fts_ranking if mid not in by_id]
        if outside:
            for item in store.get_many(outside):
                by_id[item.id] = item

    if scope:
        by_id = {
            mid: item
            for mid, item in by_id.items()
            if item.scope == scope or item.scope == "global"
        }
        if not by_id:
            return []

    lexical_scores = fuse(rankings)
    if not lexical_scores:
        return []

    candidates = []
    for memory_id, lexical in lexical_scores.items():
        item = by_id.get(memory_id)
        if item is None:
            # Matched the index but is not in the live pool (stale row, or it
            # was filtered out by scope). Drop it.
            continue

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
