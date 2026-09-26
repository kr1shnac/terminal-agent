"""Forgetting.

The v1 model was `confidence * e^(-rate * days)` with a hand-set rate, and it
applied the same curve to everything. Two changes make it behave sanely:

1. Half-life instead of a raw rate. "This decays by 0.2/day" is not something
   anyone can reason about; "this halves every 21 days" is, and it is
   equivalent: rate = ln(2) / half_life.
2. Importance gates archival. A trivial memory that is never recalled should
   quietly fade, but a stated goal must not evaporate just because the user
   was quiet for a month.

Decay is driven off `last_accessed_at`, not `created_at`, so using a memory
resets its clock. That is the entire feedback loop: recall reinforces, silence
forgets.
"""

import math

from . import store
from .clock import days_since, now_iso, to_iso
from .models import get_type

# Below this a memory is a candidate for archival.
ARCHIVE_THRESHOLD = 0.05
# A memory's importance sets the floor it decays toward, so important things
# plateau instead of dropping to zero.
IMPORTANCE_FLOOR_WEIGHT = 0.35
# Each successful recall adds this much confidence (capped at 1.0) ...
REINFORCE_BOOST = 0.12
# ... and shrinks the decay rate by this factor, so a well-used memory becomes
# progressively more durable rather than repeatedly re-learning its strength.
REINFORCE_DECAY_FACTOR = 0.85
# Must sit well below the slowest natural rate (identity: ln(2)/720 =
# 0.00096), or it clamps the initial rate instead of providing a backstop.
MIN_DECAY_RATE = 0.0001
# Re-inforcing a memory beyond this many times stops giving decay relief, so a
# popular memory cannot become effectively immortal through churn.
MAX_DECAY_RELIEF_USES = 8
# Base window before a fully-decayed memory is released. Scaled per memory by
# its importance; see `idle_limit_days`.
ARCHIVE_AFTER_IDLE_DAYS = 365.0


def retention_floor(item):
    """The confidence an item asymptotes toward instead of zero."""
    return IMPORTANCE_FLOOR_WEIGHT * float(item.importance or 0.0)


def decayed_confidence(item, now=None):
    """Confidence this memory should have right now, before reinforcement.

    The curve is `floor + (confidence - floor) * e^(-rate * days)`: the part
    above the floor decays, the floor holds. A zero floor therefore means
    "no plateau", and must still decay to zero - it must not short-circuit.
    """
    if item.confidence is None:
        return 1.0

    if item.last_accessed_at is None:
        return float(item.confidence)

    base = retention_floor(item)
    above_floor = max(0.0, float(item.confidence) - base)
    days = days_since(item.last_accessed_at, now=now)
    survived = math.exp(-float(item.decay_rate) * days)
    return base + above_floor * survived


def idle_limit_days(item):
    """How long this memory may sit completely unused before it is dropped.

    Without this the importance floor is permanent: every memory asymptotes
    to `0.35 * importance`, which never crosses the archive threshold, so
    nothing is ever forgotten and the decay machinery is cosmetic. The limit
    scales with importance, so an identity memory survives roughly twice the
    abandonment of an event before it is released.
    """
    importance = max(0.0, min(1.0, float(item.importance or 0.0)))
    return ARCHIVE_AFTER_IDLE_DAYS * (0.5 + importance)


def is_abandoned(item, now=None):
    """Fully decayed to its floor and left alone past its idle limit."""
    floor = retention_floor(item)
    current = decayed_confidence(item, now=now)
    at_floor = current <= floor + 0.01
    return at_floor and days_since(item.last_accessed_at, now=now) > idle_limit_days(item)


def reinforce(item, boost=REINFORCE_BOOST):
    """Record a successful use: raise confidence, slow decay, reset the clock."""
    if item is None:
        return None

    current = decayed_confidence(item)
    new_confidence = min(1.0, current + boost)

    uses = int(item.access_count or 0)
    if uses < MAX_DECAY_RELIEF_USES:
        new_rate = max(
            MIN_DECAY_RATE, float(item.decay_rate) * REINFORCE_DECAY_FACTOR
        )
    else:
        new_rate = float(item.decay_rate)

    store.update_confidence(item.id, new_confidence, new_rate)
    store.update_last_accessed(item.id)
    return store.get(item.id)


def apply_decay(now=None):
    """Sweep every live memory. Returns a summary of what changed.

    Cheap enough to call once per turn, which the v1 code could not do because
    it only ran at startup.

    Each write moves `last_accessed_at` forward with the confidence, upholding
    the store invariant that confidence_score is the confidence *as of*
    last_accessed_at. Without that, the next sweep would decay the value
    again from the same starting point and decay would compound.
    """
    stamp = to_iso(now) if now else now_iso()
    summary = {
        "checked": 0,
        "decayed": 0,
        "archived": 0,
        "expired": 0,
        "details": [],
    }

    for item in store.get_active():
        summary["checked"] += 1

        if _is_expired(item, now=now):
            store.archive(item.id, reason="ttl_expired")
            summary["expired"] += 1
            continue

        target = decayed_confidence(item, now=now)
        current = float(item.confidence or 0.0)

        # Abandonment is checked *before* the no-op guard below, not after it.
        # A memory that has finished decaying sits exactly on its importance
        # floor, so `target` equals `current` and the guard would skip it -
        # which meant `is_abandoned` was unreachable and every memory in the
        # store became immortal. The decay machinery looked like it worked
        # because confidences did fall, but nothing was ever released.
        if is_abandoned(item, now=now):
            store.archive(item.id, reason="idle")
            summary["archived"] += 1
            summary["details"].append(
                {"id": item.id, "text": item.text, "reason": "idle"}
            )
            continue

        # Only write when the drift is real; otherwise this becomes a commit
        # per row per turn for no reason.
        if abs(target - current) < 0.005:
            continue

        summary["decayed"] += 1

        if target < ARCHIVE_THRESHOLD:
            store.archive(item.id, reason="decayed")
            summary["archived"] += 1
            summary["details"].append(
                {"id": item.id, "text": item.text, "reason": "decayed"}
            )
        else:
            store.update_confidence(item.id, target, last_accessed_at=stamp)

    return summary


def _is_expired(item, now=None):
    if not item.expires_at:
        return False
    from .clock import parse_iso, utcnow

    expiry = parse_iso(item.expires_at)
    if expiry is None:
        return False
    return expiry <= (now or utcnow())


def half_life_days(memory_type):
    return get_type(memory_type).half_life_days


def forecast(memory_id, days=30):
    """Projected confidence over time. Useful for `/memory stats`."""
    item = store.get(memory_id)
    if item is None:
        return []

    from .clock import parse_iso, to_iso, utcnow

    base_time = parse_iso(item.last_accessed_at) or utcnow()
    current = decayed_confidence(item)
    floor = retention_floor(item)
    above_floor = max(0.0, current - floor)

    points = []
    for step in (0, 7, 14, 30, 60, 90):
        horizon = min(step, days) if days < 90 else step
        survived = math.exp(-float(item.decay_rate) * horizon)
        points.append(
            {
                "day": horizon,
                "at": to_iso(base_time) if horizon == 0 else None,
                "confidence": round(floor + above_floor * survived, 3),
            }
        )
    return points
