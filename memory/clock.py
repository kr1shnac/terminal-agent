"""Time helpers.

Every timestamp in the store is an ISO-8601 string in UTC with an explicit
offset, e.g. "2026-09-25T10:00:00+00:00". The old v1 code mixed naive
`datetime.utcnow()` values with `fromisoformat()` parsing, which silently
compares "naive" against "aware" and raises TypeError once any row carries a
real offset. Everything funnels through here so that never happens again.
"""

from datetime import datetime, timezone

UTC = timezone.utc


def utcnow():
    """Current time as an aware UTC datetime."""
    return datetime.now(UTC)


def to_iso(dt):
    """Serialize an aware datetime to a sortable UTC ISO string."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def now_iso():
    return to_iso(utcnow())


def parse_iso(value):
    """Parse a stored timestamp back into an aware UTC datetime.

    Tolerates the legacy v1 rows that were written without an offset; those
    are interpreted as UTC, which is what they were meant to be.
    """
    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def days_since(value, now=None):
    """Whole-or-fractional days elapsed since `value`. Never negative.

    Clamping at zero matters: a clock skew or a future-dated row should not
    produce a negative age and *increase* a memory's confidence.
    """
    then = parse_iso(value)
    if then is None:
        return 0.0
    now = now or utcnow()
    return max(0.0, (now - then).total_seconds() / 86400.0)


def plus_days(days, now=None):
    now = now or utcnow()
    return to_iso(now + _days(days))


def _days(days):
    from datetime import timedelta

    return timedelta(days=days)
