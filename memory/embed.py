"""Turning memory text into a vector you can compare by meaning.

The design constraint that shaped everything here: the application ships with
no runtime dependency beyond the API client, and it has to keep working with no
network. So the default embedder is local, deterministic and dependency-free -
it is *not* a neural model, and it does not pretend to be one. What it does buy
is the part lexical search structurally cannot do: robustness to inflection and
to word order.

Concretely, `"allergies"` and `"allergic"` share no token with each other, and
a bag-of-words ranker scores them as unrelated. Both contain the character
trigrams `all`, `lle`, `ler`, `eri`, `rie`, so a subword vector puts them next
to each other without ever having seen a training set. That is a real,
measurable improvement and it is honestly bounded.

**The bound, measured rather than assumed.** On the 262-memory benchmark, whose
questions are labelled, this embedder was scored on the wanted memory's cosine
against the best unwanted one:

    true-answer cosines  min 0.0000  median 0.4525  max 0.7453
    best-wrong cosines   min 0.1110  median 0.2859  max 0.5615
    true answers ranked above the worst false positive: 4 of 21

Only four of twenty-one questions put the right memory above every wrong one,
and eight put a wrong memory *first*; the two distributions overlap heavily. So
this is a **character-overlap index, not a semantic model**, and on its own it
is a poor ranker. The near-zero true answers are all the same kind of question
- "do you know who I am" 0.0000, "what is my job" 0.0000, "which package
manager do I use" 0.0020 - where the question and the answer share not one
character. No amount of hashing closes that gap.

Where it earns its place is inside the retrieval cascade, as a safety net that
only *adds* candidates when the lexical rankers came back weak or empty, never
displacing a strong match. The benchmark agrees: hit@6 56.5% -> 60.9% with the
net in place, at ~3ms per escalated query.

For genuine synonymy, a trained model is the only real answer.
:class:`ExternalEmbedder` is the seam for one, and vectors are cached in the
database, so a store indexed with a real model keeps working with the network
switched off. The retrieval cascade in `retrieve.py` does not care which
embedder produced the vectors.
"""

import hashlib
import math
import os
import re
from array import array
from collections import OrderedDict
from operator import mul

from .text import STOPWORDS, fold, tokenize

# 256 dimensions. Bigger costs a linear scan and buys nothing at the scale of a
# personal memory bank; measured on this machine, a 256-dim dot product over
# 1000 memories takes ~13ms, which is why the cascade only pays for it when the
# cheap rankers have already failed.
DEFAULT_DIM = 256

# Character n-grams are what carry inflection ("migrate"/"migrating"). Shorter
# n-grams add noise, longer ones stop matching ordinary spelling changes.
_NGRAM_MIN = 3
_NGRAM_MAX = 4

_NGRAM_STRIP = re.compile(r"[^a-z0-9]+")

MODEL_NAME = "local-hash-v1"
PROVIDER_VERSION = 1


def _hash_signed(token, dim):
    """Map a token to one dimension, with a stable sign.

    A plain modulo sends every collision to the same slot, so two unrelated
    words that collide reinforce each other instead of partly cancelling.
    Signing by a second hash halves the damage on average: colliding tokens
    cancel about half the time rather than always adding.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    slot = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return slot, sign


def _char_ngrams(word):
    """Padded character n-grams, the morphological safety net."""
    cleaned = _NGRAM_STRIP.sub(" ", fold(word))
    padded = f"^{cleaned}$"
    grams = []
    for size in range(_NGRAM_MIN, _NGRAM_MAX + 1):
        if len(padded) < size:
            continue
        for start in range(len(padded) - size + 1):
            grams.append(padded[start : start + size])
    return grams


class HashingEmbedder:
    """IDF-weighted, subword-augmented, L2-normalised sparse hashing.

    IDF weighting is what keeps it from being a synonym for "stop hashing
    everything into the same bucket": a term in every memory contributes almost
    nothing, so a match on a rare word still dominates the vector direction.
    """

    name = MODEL_NAME
    dim = DEFAULT_DIM

    # Cosine thresholds belong to the model that produced the vectors, because
    # the scale is a property of the model, not of the store: a learned model
    # puts related pairs around 0.3-0.9 where this one puts them around
    # 0.15-0.55. Calibrated on the 262-memory benchmark by comparing each
    # labelled question's wanted memory against the best unwanted one:
    #
    #   true-answer cosines  min 0.0000  median 0.4525  max 0.7453
    #   best-wrong cosines   min 0.1110  median 0.2859  max 0.5615
    #
    # 0.12 sits just above the weakest false positive seen, and 0.55 is where a
    # genuinely good match tends to land. See the module docstring for what this
    # embedder cannot do.
    min_similarity = 0.12
    strong = 0.55

    def __init__(self, dim=DEFAULT_DIM, idf=None):
        self.dim = dim
        # A neutral prior until the caller has real corpus statistics. Query
        # time supplies the store's own IDF, which is strictly better.
        self.idf = idf or {}

    def _weight(self, token):
        return self.idf.get(token, 1.0)

    def embed(self, text, idf=None):
        """Return an L2-normalised `array('f')` of length `dim`.

        A zero vector is returned for text with no usable signal (pure
        punctuation, a stopword-only sentence). Cosine against it is 0, which is
        the correct "no opinion" rather than a crash or a random direction.
        """
        if idf is not None:
            self.idf = idf

        vector = array("f", bytes(4 * self.dim))
        tokens = tokenize(text)

        for token in tokens:
            slot, sign = _hash_signed(token, self.dim)
            vector[slot] += sign * self._weight(token)

            for gram in _char_ngrams(token):
                gslot, gsign = _hash_signed(f"#{gram}", self.dim)
                # Subwords count for less than whole words: they exist to catch
                # an inflection, not to outvote a real term match.
                vector[gslot] += gsign * self._weight(token) * 0.35

        return _normalise(vector)

    def embed_many(self, texts, idf=None):
        return [self.embed(text, idf=idf) for text in texts]


def _normalise(vector):
    """Scale to unit length in place, so cosine is a plain dot product."""
    total = 0.0
    for value in vector:
        total += value * value
    if total <= 0:
        return vector
    scale = 1.0 / math.sqrt(total)
    for index in range(len(vector)):
        vector[index] *= scale
    return vector


def to_blob(vector):
    """Pack a vector for storage. Little-endian float32."""
    return array("f", vector).tobytes()


def from_blob(blob, dim):
    """Unpack a stored vector, defensively.

    A truncated or foreign blob must not take the process down: a dimension
    mismatch means the row was written by a different embedder, and the honest
    answer is "this vector is not comparable", not an exception mid-query.
    """
    if not blob:
        return None
    values = array("f")
    usable = len(blob) - (len(blob) % 4)
    if usable != dim * 4:
        return None
    try:
        values.frombytes(blob[:usable])
    except ValueError:
        return None
    return values


def dot(a, b):
    """Cosine similarity of two unit vectors.

    `sum(map(mul, ...))` rather than a Python loop: at these sizes the map runs
    in C and this is roughly five times faster, which is the difference between
    a vector search being affordable on every query and only on escalation.
    """
    if len(a) != len(b):
        return 0.0
    return sum(map(mul, a, b))


def cosine(a, b):
    """Cosine similarity for vectors that may not be normalised."""
    if not a or not b or len(a) != len(b):
        return 0.0
    left = math.sqrt(sum(x * x for x in a))
    right = math.sqrt(sum(x * x for x in b))
    if left <= 0 or right <= 0:
        return 0.0
    return dot(a, b) / (left * right)


class ExternalEmbedder:
    """Provider-agnostic seam: wrap any callable that returns a vector.

    :class:`OpenAICompatibleEmbedder` covers hosted models, and
    :class:`HashingEmbedder` covers the offline default. This remains for
    anything else - a local model, a private endpoint, a test double - where
    all that is needed is a name, a dimension, and a function from text to
    numbers. Vectors are cached per model name, so swapping models re-indexes
    rather than invalidates.

    `encode` is the single method an adapter has to provide. Thresholds default
    to the hashing embedder's measured values; a model with a different cosine
    scale should override `min_similarity` and `strong`.
    """

    min_similarity = HashingEmbedder.min_similarity
    strong = HashingEmbedder.strong

    def __init__(self, name, dim, encode):
        self.name = name
        self.dim = dim
        self._encode = encode

    def embed(self, text, idf=None):
        # IDF is meaningless for a learned model; accepted only so this is a
        # drop-in for `HashingEmbedder`.
        return _normalise(array("f", self._encode(text)))

    def embed_many(self, texts, idf=None):
        return [self.embed(text) for text in texts]


class OpenAICompatibleEmbedder:
    """Real embeddings from any OpenAI-compatible ``/embeddings`` endpoint.

    This is the embedder that can actually answer "what is my job" against "the
    user is a backend engineer", which the hashing embedder scores at 0.0000
    because the two sentences share no characters.

    Three things it does that matter for a memory store:

    **Batching.** Embedding one memory per HTTP round trip would dominate the
    cost of storing it, so writes go out in batches of `batch_size`.

    **Dimension truncation.** The provider's full-width vector is 1536-3072
    dims, and the search is a linear scan, so width is paid on *every query*.
    `text-embedding-3` is trained Matryoshka-style and supports an explicit
    `dimensions` argument, which keeps almost all of the retrieval quality for a
    fraction of the scan time - measured here at roughly 1.5ms per 262 rows per
    256 dims against ~9ms at 1536.

    **Query-vector caching.** The escalation path embeds the user's question, so
    a repeated question would otherwise be paid for twice. Recent query vectors
    are kept in memory; the identity of a question is its exact text, which is
    safe because a cached vector is a pure function of that text.
    """

    #: Learned models put unrelated English sentences in a much tighter, higher
    #: band than the hashing embedder, so the thresholds are its own. Calibrated
    #: on the 262-memory benchmark with `text-embedding-3-small` truncated to 512
    #: dims, comparing each labelled question's wanted memory against the best
    #: unwanted one:
    #:
    #:   true cosines  min 0.2854  median 0.5071  max 0.7539
    #:   wrong cosines min 0.2505  median 0.3286  max 0.5271
    #:   true ranked above every wrong memory: 19 of 21
    #:
    #: The same measurement on the hashing embedder gives 4 of 21, with the
    #: synonym questions at ~0.00. That difference is the whole reason this class
    #: exists.
    #:
    #: The floor sits just above the weakest false positive and below the weakest
    #: true answer. The two distributions still overlap between roughly 0.25 and
    #: 0.53, so no single threshold separates them - which is why the vector stage
    #: only ever *adds* candidates on escalation and never displaces a strong
    #: lexical match.
    min_similarity = 0.26
    strong = 0.51

    def __init__(
        self,
        name,
        client,
        model,
        dim=512,
        batch_size=64,
        max_chars=8000,
        query_cache=256,
    ):
        self.name = name
        self.model = model
        self.dim = int(dim)
        self.batch_size = int(batch_size)
        self.max_chars = int(max_chars)
        self._client = client
        self._query_cache = OrderedDict()
        self._query_cache_max = int(query_cache)
        self.calls = 0
        self.tokens = 0

    # -------------------------------------------------------------- encoding

    def _request(self, texts):
        payload = [t[: self.max_chars] for t in texts]
        response = self._client.embeddings.create(
            model=self.model, input=payload, dimensions=self.dim
        )
        self.calls += 1
        self.tokens += int(getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0)
        # The API is documented to preserve input order, but it returns objects
        # with an `index`, and trusting a positional assumption here would
        # silently attach memories to each other's vectors.
        ordered = sorted(response.data, key=lambda row: getattr(row, "index", 0))
        return [array("f", row.embedding) for row in ordered]

    def embed(self, text, idf=None):
        return _normalise(self._request([str(text)])[0])

    def embed_many(self, texts, idf=None):
        texts = [str(t) for t in texts]
        if not texts:
            return []
        out = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._request(texts[start : start + self.batch_size]))
        return [_normalise(vector) for vector in out]

    def embed_query(self, text, idf=None):
        """Embed a question, reusing the vector if this exact text was just asked.

        Escalation is the only query-time caller, and it fires on the queries
        that are hardest to cache-cache by meaning, so a small exact-text cache
        is what keeps a repeated question from costing a second round trip.
        """
        key = " ".join(str(text).split())
        cached = self._query_cache.get(key)
        if cached is not None:
            self._query_cache.move_to_end(key)
            return cached
        vector = self.embed(text)
        self._query_cache[key] = vector
        while len(self._query_cache) > self._query_cache_max:
            self._query_cache.popitem(last=False)
        return vector

    def calibrate(self, min_similarity=None, strong=None):
        """Set this model's thresholds from measurement. Returns self."""
        if min_similarity is not None:
            self.min_similarity = float(min_similarity)
        if strong is not None:
            self.strong = float(strong)
        return self

    def usage(self):
        return {"calls": self.calls, "prompt_tokens": self.tokens}


def from_environment(client=None, environ=None):
    """Build a network embedder from the environment, or return None.

    Returns None when no model is configured, which is the default and the
    reason the application still runs with no network at all. Turning this on is
    a deliberate act with two costs worth stating plainly: every *write* and
    every *escalated query* becomes an HTTP call, and memory text leaves the
    machine. Reads stay local and free, because vectors are cached in the
    database - so a store indexed with a real model keeps answering offline.
    """
    environ = os.environ if environ is None else environ
    model = (environ.get("RETAIN_EMBEDDING_MODEL") or "").strip()
    if not model:
        return None

    base_url = (environ.get("RETAIN_EMBEDDING_BASE_URL") or "").strip()
    api_key = (environ.get("RETAIN_EMBEDDING_API_KEY") or "").strip()
    if not api_key:
        for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
            if environ.get(name):
                api_key = environ[name]
                break
    if not api_key:
        return None

    if client is None:
        from openai import OpenAI

        client = OpenAI(
            base_url=base_url or "https://openrouter.ai/api/v1", api_key=api_key
        )

    dim = int(environ.get("RETAIN_EMBEDDING_DIMENSIONS") or 512)
    batch = int(environ.get("RETAIN_EMBEDDING_BATCH") or 64)
    return OpenAICompatibleEmbedder(
        name=f"{model}:{dim}",
        client=client,
        model=model,
        dim=dim,
        batch_size=batch,
    )


# ------------------------------------------------------------------ registry

_DEFAULT = HashingEmbedder()
_BY_NAME = {MODEL_NAME: _DEFAULT}
_CURRENT = {"name": MODEL_NAME}


def current():
    """The embedder the store is currently indexed with."""
    return _BY_NAME[_CURRENT["name"]]


def current_name():
    return _CURRENT["name"]


def use(embedder):
    """Adopt an embedder process-wide. Returns its name."""
    _BY_NAME[embedder.name] = embedder
    _CURRENT["name"] = embedder.name
    return embedder.name


def get(name):
    return _BY_NAME.get(name)


def reset():
    """Back to the built-in embedder. Used by tests."""
    _CURRENT["name"] = MODEL_NAME
    _BY_NAME[MODEL_NAME] = HashingEmbedder()


def is_compatible(blob, dim):
    """Cheap guard so a dimension change invalidates stale rows loudly."""
    return bool(blob) and len(blob) == dim * 4
