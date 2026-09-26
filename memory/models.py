"""Memory types and their lifecycle policy.

The v1 code set a decay rate by `if memory_type == "fact"` and lumped
everything else together, so a stated life goal decayed exactly as fast as
"today I fixed a typo". The whole point of typed memory is that different
kinds of knowledge have different lifetimes, so the policy lives here as
data and `decay.py` just reads it.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class MemoryType:
    name: str
    description: str
    # Days for confidence to halve if never touched. Larger = stickier.
    half_life_days: float
    # Default importance in [0, 1]; scales how much decay is tolerated
    # before archival, and how much this memory can outrank others.
    importance: float
    # How many days until the memory is considered expired. None = never.
    ttl_days: Optional[float] = None
    # Whether the model is allowed to write this type from a `remember` call.
    user_writable: bool = True


MEMORY_TYPES = {
    "identity": MemoryType(
        name="identity",
        description=(
            "Stable facts about who the user is: name, role, timezone, "
            "location, languages they speak."
        ),
        half_life_days=720.0,
        importance=0.95,
    ),
    "preference": MemoryType(
        name="preference",
        description=(
            "How the user likes things done: editor, package manager, code "
            "style, tone, notification habits."
        ),
        half_life_days=365.0,
        importance=0.8,
    ),
    "instruction": MemoryType(
        name="instruction",
        description=(
            "Standing rules the user has given the agent, e.g. 'always run "
            "tests before saying done'."
        ),
        half_life_days=540.0,
        importance=0.9,
    ),
    "goal": MemoryType(
        name="goal",
        description=(
            "Things the user is trying to achieve, with a direction that "
            "persists across weeks."
        ),
        half_life_days=270.0,
        importance=0.85,
    ),
    "project": MemoryType(
        name="project",
        description=(
            "Context about a codebase or product: its purpose, stack, "
            "conventions that are not derivable from the files themselves."
        ),
        half_life_days=180.0,
        importance=0.75,
    ),
    "fact": MemoryType(
        name="fact",
        description="General durable knowledge about the user's world.",
        half_life_days=120.0,
        importance=0.6,
    ),
    "event": MemoryType(
        name="event",
        description=(
            "Something that happened at a point in time. Useful for "
            "recency, not for identity."
        ),
        half_life_days=21.0,
        importance=0.4,
    ),
    "session": MemoryType(
        name="session",
        description=(
            "Short-lived working state from the current conversation. "
            "Expires quickly on purpose."
        ),
        half_life_days=3.0,
        importance=0.3,
        ttl_days=2.0,
    ),
}

DEFAULT_TYPE = "fact"

# Types the auto-extractor may write without an explicit model request.
AUTO_WRITABLE = {
    "identity",
    "preference",
    "instruction",
    "goal",
    "project",
    "fact",
    "event",
}

# Aliases so a sloppy `memory_type` string from the model still lands
# somewhere sensible instead of falling back to "fact".
TYPE_ALIASES = {
    "personal": "identity",
    "bio": "identity",
    "profile": "identity",
    "user": "identity",
    "persona": "identity",
    "prefs": "preference",
    "style": "preference",
    "settings": "preference",
    "rule": "instruction",
    "guideline": "instruction",
    "constraint": "instruction",
    "objective": "goal",
    "todo": "goal",
    "task": "goal",
    "repo": "project",
    "codebase": "project",
    "context": "project",
    "info": "fact",
    "knowledge": "fact",
    "memory": "fact",
    "log": "event",
    "activity": "event",
    "history": "event",
    "temporary": "session",
    "temp": "session",
    "ephemeral": "session",
}


def normalize_type(value):
    """Map any incoming label onto a known type. Never raises."""
    if not value:
        return DEFAULT_TYPE

    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key in MEMORY_TYPES:
        return key
    if key in TYPE_ALIASES:
        return TYPE_ALIASES[key]

    # No exact match: sweep the parts. Scanning from the right handles
    # qualified labels correctly - in "user_preference" the trailing word is
    # the specific one, while "user" is a generic prefix that would otherwise
    # win and mislabel everything as identity.
    for token in reversed(key.split("_")):
        if token in MEMORY_TYPES:
            return token
        if token in TYPE_ALIASES:
            return TYPE_ALIASES[token]

    return DEFAULT_TYPE


def get_type(name):
    return MEMORY_TYPES[normalize_type(name)]


def decay_rate_for(name):
    """Convert a half-life into the continuous rate used by the decay curve.

    confidence(t) = confidence(0) * e^(-rate * t)  and  we want
    confidence(half_life) = 0.5 * confidence(0), which gives
    rate = ln(2) / half_life.
    """
    import math

    half_life = get_type(name).half_life_days
    return math.log(2) / max(half_life, 0.5)


def importance_for(name):
    return get_type(name).importance


def _clamp(value, default):
    """Coerce a possibly-missing numeric field into [0, 1]."""
    if value is None:
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if value != value:  # NaN
        return float(default)
    return max(0.0, min(1.0, value))


def ttl_for(name):
    return get_type(name).ttl_days


@dataclass
class MemoryItem:
    """One remembered thing. Mirrors a row in the `memory` table."""

    text: str
    memory_type: str = DEFAULT_TYPE
    scope: str = "global"
    subject: Optional[str] = None
    importance: Optional[float] = None
    confidence: float = 1.0
    decay_rate: float = 0.1
    ttl_days: Optional[float] = None
    source: str = "tool"
    id: Optional[int] = None
    access_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_accessed_at: Optional[str] = None
    expires_at: Optional[str] = None
    is_archived: int = 0
    archived_reason: Optional[str] = None
    supersedes: Optional[int] = None
    superseded_by: Optional[int] = None
    session_id: Optional[str] = None

    # filled in by the retriever, never persisted
    score: float = 0.0
    signals: dict = field(default_factory=dict)

    def __post_init__(self):
        self.memory_type = normalize_type(self.memory_type)
        if self.importance is None:
            self.importance = importance_for(self.memory_type)
        if self.ttl_days is None:
            self.ttl_days = ttl_for(self.memory_type)
        if not self.scope:
            self.scope = "global"
        # Coerced here, once, so that no caller has to defend against a NULL
        # column. The row carries `confidence_score REAL NOT NULL`, but a
        # caller can still build `MemoryItem(confidence=None)` by hand, and
        # the read paths that print it - `f"{item.confidence:.0%}"` in the
        # CLI table and the `recall` tool - raised TypeError on that instead of
        # treating it as the unset value it is.
        self.confidence = _clamp(self.confidence, 1.0)
        self.importance = _clamp(self.importance, importance_for(self.memory_type))
        if self.access_count is None:
            self.access_count = 0

    @property
    def is_live(self):
        return not self.is_archived and not self.superseded_by

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_row(cls, row):
        if row is None:
            return None
        data = dict(row)
        data.pop("norm_hash", None)
        return cls(
            id=data.get("id"),
            text=data.get("text") or "",
            memory_type=data.get("memory_type") or DEFAULT_TYPE,
            scope=data.get("scope") or "global",
            subject=data.get("subject"),
            importance=data.get("importance"),
            confidence=data.get("confidence_score", 1.0),
            decay_rate=data.get("decay_rate", 0.1),
            ttl_days=data.get("ttl_days"),
            source=data.get("source") or "tool",
            access_count=data.get("access_count") or 0,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            last_accessed_at=data.get("last_accessed_at"),
            expires_at=data.get("expires_at"),
            is_archived=data.get("is_archived") or 0,
            archived_reason=data.get("archived_reason"),
            supersedes=data.get("supersedes"),
            superseded_by=data.get("superseded_by"),
            session_id=data.get("session_id"),
        )
