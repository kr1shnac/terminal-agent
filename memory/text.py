"""Text utilities: tokenizing, normalization, dedupe hashing, FTS escaping.

The tokenizer is intentionally small and hand-written. FTS5 handles the
heavy stemming for the lexical path; this tokenizer feeds the TF-IDF ranker,
the near-duplicate detector, and the salience keywords, so it favours
recall over precision and keeps short domain-meaningful tokens.
"""

import hashlib
import re

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS = {
    # articles / conjunctions / prepositions
    "the", "and", "but", "for", "nor", "yet", "so", "with", "from", "into",
    "onto", "upon", "than", "that", "this", "these", "those", "there", "here",
    # pronouns
    "you", "your", "yours", "she", "her", "hers", "him", "his", "they", "them",
    "their", "theirs", "we", "our", "ours", "ourself", "myself", "itself",
    # auxiliaries
    "am", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "doing", "done", "have", "has", "had", "having", "will", "would",
    "shall", "should", "can", "could", "may", "might", "must", "ought",
    # filler that adds nothing to retrieval
    "just", "also", "very", "really", "quite", "some", "any", "all", "more",
    "most", "other", "such", "only", "own", "same", "too", "s", "t", "don",
    "now", "get", "got", "make", "made", "want", "like", "well", "back",
}

# Short tokens that carry real meaning in a coding context and would be lost
# to a blanket "len > 2" filter.
SHORT_TOKENS = {
    "ai", "ml", "js", "ts", "go", "py", "os", "ui", "ux", "db", "ci", "cd",
    "qa", "vm", "io", "db", "cd", "ip", "qa", "sh", "cd", "r", "c", "npm",
    "aws", "gcp", "api", "css", "sql", "gpu", "cpu", "ram", "ssh", "ftp",
}

MIN_TOKEN_LEN = 2
MAX_QUERY_TERMS = 12


def tokenize(text, keep_stopwords=False):
    """Lowercase word tokens, minus noise.

    Keeps `len >= 2` (rather than the old `> 2`) so domain tokens like "js"
    and "ci" survive, and compensates with a much larger stopword list.
    """
    if not text:
        return []

    tokens = []
    for match in _WORD_RE.finditer(str(text).lower()):
        word = match.group(0)
        if not keep_stopwords:
            if word in STOPWORDS:
                continue
            if len(word) < MIN_TOKEN_LEN and word not in SHORT_TOKENS:
                continue
            if word.isdigit() and len(word) > 4:
                # long bare numbers are noise (timestamps, hashes, ids)
                continue
        tokens.append(word)

    return tokens


def normalize(text):
    """Canonical form used for exact-duplicate detection.

    "User's name is Krishna." and "user name is krishna" must hash the same,
    otherwise dedupe silently misses and the same fact accumulates.
    """
    if not text:
        return ""
    tokens = tokenize(text, keep_stopwords=True)
    return " ".join(tokens)


def norm_hash(text):
    """Stable hash of the normalized text (exact-dupe key)."""
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def jaccard(a_tokens, b_tokens):
    """Jaccard similarity of two token sets, used for near-dupe detection."""
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def trigrams(text):
    """Character trigrams, a second near-dupe signal that survives typos."""
    squashed = " ".join(tokenize(text))
    if len(squashed) < 3:
        return {squashed} if squashed else set()
    return {squashed[i: i + 3] for i in range(len(squashed) - 2)}


def trigram_jaccard(a, b):
    return jaccard(a, b)


def fts_query(text, prefix=True, max_terms=MAX_QUERY_TERMS):
    """Build a safe FTS5 MATCH expression from free-form user text.

    FTS5 has its own mini query language, so raw input breaks it: an unbalanced
    quote is a syntax error, and bare words like AND/OR/NOT change the parse.
    Every term is therefore re-emitted as a quoted string, which neutralises
    both problems.
    """
    terms = [t for t in tokenize(text) if t]
    if not terms:
        return None

    if len(terms) > max_terms:
        # Keep the longest terms: they carry the most specificity, and an
        # over-long AND chain returns nothing when one term is missing.
        terms = sorted(terms, key=len, reverse=True)[:max_terms]

    parts = []
    for term in terms:
        safe = term.replace('"', "")
        if not safe:
            continue
        parts.append(f'"{safe}"*' if prefix else f'"{safe}"')

    if not parts:
        return None
    return " AND ".join(parts)


def keywords(text, limit=8):
    """Highest-signal tokens for a memory, used for salience and dedupe."""
    counts = {}
    for token in tokenize(text):
        counts[token] = counts.get(token, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [token for token, _ in ranked[:limit]]


def summarize(text, limit=120):
    """Short single-line preview for logs and terminal output."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
