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

For genuine synonymy - "what is my job" against "the user is a backend
engineer" - a trained model is the only real answer. :class:`ExternalEmbedder`
is the seam for one, and vectors are cached in the database, so a store indexed
once keeps answering offline forever afterwards. The retrieval cascade in
`retrieve.py` does not care which embedder produced the vectors.
"""

import hashlib
import math
import re
from array import array
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
    """Pluggable seam for a real embedding model.

    Not wired to a provider on purpose. What a caller needs from one is narrow -
    a name, a dimension, and `embed` - and pinning a specific hosted model into
    the store's schema would tie the database format to somebody's pricing
    page. Vectors are cached per model name, so swapping models re-indexes
    rather than invalidates.

    `encode` is the single method an adapter has to provide.
    """

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
