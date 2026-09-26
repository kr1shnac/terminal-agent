"""Text utilities: tokenizing, normalization, dedupe hashing, FTS escaping.

The tokenizer is intentionally small and hand-written. FTS5 handles the
heavy stemming for the lexical path; this tokenizer feeds the TF-IDF ranker,
the near-duplicate detector, and the salience keywords, so it favours
recall over precision and keeps short domain-meaningful tokens.
"""

import hashlib
import math
import re
import unicodedata

# Word characters from *any* script, with the internal apostrophe kept
# ("user's" is one token) and the underscore excluded so identifiers still
# split ("foo_bar" -> "foo", "bar").
#
# The old pattern was `[a-z0-9]+`, ASCII only, which made every token from
# non-Latin text disappear. That is not a cosmetic problem: `normalize` is what
# `norm_hash` is computed from, so *any* text with no ASCII word characters
# normalized to the empty string and therefore hashed to sha1("") - the same
# value for every one of them. Storing a second memory in Japanese, Chinese,
# Cyrillic or Greek did not create a row, it silently returned the first one,
# and the new fact was dropped without a trace. Matching the FTS5
# `unicode61` tokenizer's notion of a token keeps the lexical path and the
# BM25 path agreeing with each other as well.
_WORD_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
_POSSESSIVE_RE = re.compile(r"['\u2019]s\b")

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
    # function words that separate a paraphrase from a contradiction. These
    # carry no assertion, so they must not count toward similarity or
    # "prefers pnpm over npm" stops resembling "prefers pnpm, not npm".
    "over", "not", "instead", "rather", "than", "off", "again", "further",
    "once", "no", "nor", "both", "each", "few", "about", "after", "before",
    "during", "through", "under", "above", "below", "between", "because",
    "why", "how", "what", "when", "where", "which", "who", "whom", "does",
    "did", "doing", "would", "could", "should", "may", "might", "must",
    "let", "lets", "please", "thanks", "hello", "hi", "hey", "yes", "sure",
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


def fold(text):
    """NFKC-normalize and casefold, so equivalent spellings meet.

    NFKC is what makes a fullwidth "Ｄｅｖｅｌｏｐｅｒ" and an accented "café"
    survive at all, and it folds the compatibility forms that IME input and
    copy-paste produce. `casefold` rather than `lower` because it is the
    Unicode-aware operation (`lower` leaves "STRASSE" and "straße" distinct).
    """
    return unicodedata.normalize("NFKC", str(text or "")).casefold()


def tokenize(text, keep_stopwords=False):
    """Lowercase word tokens, minus noise.

    Keeps `len >= 2` (rather than the old `> 2`) so domain tokens like "js"
    and "ci" survive, and compensates with a much larger stopword list.
    """
    if not text:
        return []

    tokens = []
    for match in _WORD_RE.finditer(fold(text)):
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
    otherwise dedupe silently misses and the same fact accumulates. Apostrophes
    are dropped first, since the word regex otherwise keeps "user's" as one
    token and it will never match a bare "user".
    """
    if not text:
        return ""

    flat = str(text)
    # Drop the possessive first, so "user's" becomes "user" and not "users",
    # then remove any apostrophe that is left over.
    flat = _POSSESSIVE_RE.sub("", flat)
    flat = flat.replace("'", "").replace("\u2019", "")

    tokens = tokenize(flat, keep_stopwords=True)
    if tokens:
        return " ".join(tokens)

    # No word tokens at all - text in an alphabet the regex does not match, or
    # nothing but punctuation. Falling back to "" here would give every such
    # text the same hash and merge unrelated memories into one row, so use a
    # form that is still canonical (so real duplicates still collide) but
    # cannot be empty (so distinct ones never do).
    collapsed = " ".join(fold(flat).split())
    return collapsed or _PUNCT_HASH_SALT


# Appended to the empty-token fallback so that two *different* punctuation-only
# strings stay distinct. It is unreachable by any real text, because the
# fallback only runs on strings that produced no tokens at all.
_PUNCT_HASH_SALT = "\x00punctuation-only\x00"


def norm_hash(text):
    """Stable hash of the normalized text (exact-dupe key)."""
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def jaccard(a_tokens, b_tokens):
    """Jaccard similarity of two token sets, used for near-dupe detection."""
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_idf(documents):
    """IDF table over a corpus of token lists, smoothed so nothing is zero."""
    total = max(1, len(documents))
    frequency = {}
    for tokens in documents:
        for token in set(tokens):
            frequency[token] = frequency.get(token, 0) + 1

    return {
        token: math.log(1.0 + total / max(count, 1))
        for token, count in frequency.items()
    }


def weighted_jaccard(a_tokens, b_tokens, idf=None, default_idf=1.0):
    """IDF-weighted Jaccard similarity.

    Plain Jaccard cannot tell a paraphrase from a contradiction:

        "The user prefers pnpm over npm"
        "User prefers pnpm, not npm"        <- paraphrase, should merge
        "The user prefers yarn over npm"     <- different fact, must not

    All three sit around 0.67 plain Jaccard, because the function words
    ("over", "not") carry as much weight as the discriminative one. Weighting
    each token by inverse document frequency fixes exactly that: the filler
    words count for almost nothing, while pnpm-vs-yarn dominates the score and
    pushes it below any sane merge threshold.
    """
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return 0.0

    idf = idf or {}

    def weight(token):
        return idf.get(token, default_idf)

    shared = sum(weight(token) for token in a & b)
    union = sum(weight(token) for token in a | b)
    if union <= 0:
        return 0.0
    return shared / union


def similarity(a_tokens, b_tokens, idf=None):
    """Best of token overlap and character trigram overlap.

    Trigrams catch the typos and word-splitting that token overlap misses, so
    the two signals are combined rather than one being trusted alone.

    A raw string is accepted and tokenized here. This matters more than it
    looks: `trigrams_from` joins its argument, and joining a *string* joins it
    per character. Two contradictory facts then reduce to near-identical
    character soup and score ~0.92 instead of ~0.60:

        similarity("The user prefers pnpm over npm",
                   "The user prefers yarn over npm")     -> 0.92, auto-merged

    Since this score gates automatic merging and archiving of real user data,
    that mistake would quietly delete a preference. Coercing instead of
    raising keeps a plausible caller error from becoming data loss.
    """
    a_tokens = tokenize(a_tokens) if isinstance(a_tokens, str) else a_tokens
    b_tokens = tokenize(b_tokens) if isinstance(b_tokens, str) else b_tokens

    token_score = weighted_jaccard(a_tokens, b_tokens, idf)
    trigram_score = weighted_jaccard(trigrams_from(a_tokens), trigrams_from(b_tokens), idf)
    return max(token_score, trigram_score * 0.92)


def trigrams_from(tokens):
    return trigrams(" ".join(tokens))


def trigrams(text):
    """Character trigrams, a second near-dupe signal that survives typos."""
    squashed = " ".join(tokenize(text))
    if len(squashed) < 3:
        return {squashed} if squashed else set()
    return {squashed[i: i + 3] for i in range(len(squashed) - 2)}


def trigram_jaccard(a, b):
    return jaccard(a, b)


def fts_query(text, prefix=True, max_terms=MAX_QUERY_TERMS, mode="AND"):
    """Build a safe FTS5 MATCH expression from free-form user text.

    FTS5 has its own mini query language, so raw input breaks it: an unbalanced
    quote is a syntax error, and bare words like AND/OR/NOT change the parse.
    Every term is therefore re-emitted as a quoted string, which neutralises
    both problems.

    `mode="OR"` is the fallback for when an AND chain returns nothing. Prefix
    matching only extends the *query* term, so "allergy"* cannot match the
    indexed stem "allerg" - a shorter document stem is unreachable, and any
    query where the user used a different inflection than the memory silently
    returns nothing. Falling back to OR lets BM25 rank the partial matches.
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
    return f" {mode} ".join(parts)


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
