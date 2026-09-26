"""Automatic memory capture.

In v1 the only path to a memory was the model choosing to call `remember()`.
That is unreliable in exactly the situation it matters: a user says "btw I'm on
Windows and I hate yarn" in passing, and the model is busy debugging and never
calls the tool. The fact is gone.

So memories are extracted from the *user's own words* every turn, on two
paths:

1. A cheap LLM call that returns strict JSON. Catches nuance, infers the
   subject slot ("I moved to Berlin" -> identity/user.location).
2. Regex heuristics. Catches the high-frequency phrasings with no API call at
   all, and keeps the system useful when the key is missing or the model is
   down.

Only user text is mined. Extracting from the assistant's own replies creates a
feedback loop where the agent paraphrases a retrieved memory back at the user,
re-stores the paraphrase, and slowly drifts.
"""

import json
import re

from . import store
from .models import AUTO_WRITABLE, normalize_type
from .text import summarize, tokenize

# Bounded on purpose: a long conversation is a sign the user is describing a
# task, not volunteering durable facts.
MAX_CHARS = 4000
MAX_PER_TURN = 3

# Token overlap at which a model candidate is treated as an echo of a
# subject-less heuristic hit rather than a new fact.
#
# Low, and deliberately so: both candidates come from the same sentence, and
# Jaccard punishes exactly the case worth catching. Asked about "I use pnpm and
# yarn", the model answers "uses pnpm as their package manager" - it replaces
# the value `yarn` with a description, so the two texts differ at both ends and
# score 0.43 while describing one fact. Two genuinely different subject-less
# preferences ("prefers pnpm", "dislikes npm") share only the boilerplate and
# score around 0.2, so they still survive.
_ECHO_THRESHOLD = 0.4

# Nothing here is a memory on its own.
MIN_CHARS = 12
MIN_CONTENT_TOKENS = 2

SYSTEM_PROMPT = """\
You extract durable facts about a user from their message for long-term memory.

Return ONLY a JSON array. Each element must be an object with these keys:
  "text"        - one self-contained sentence, third person ("The user's
                  editor is Neovim"), stating only what the message supports
  "memory_type" - one of: identity, preference, instruction, goal, project,
                  fact, event
  "subject"     - short dotted key for the thing being described, e.g.
                  "user.name", "user.editor", "goal.python". Use a subject
                  ONLY when the value is singular and could be replaced later
                  (a name, a preference, a tool choice). Use null otherwise.
  "importance"  - 0.0 to 1.0, how much this should be trusted and remembered

Store: identity (who they are), preference (how they like things done),
instruction (standing rules for you), goal (what they are trying to achieve),
project (context about their codebase not visible in the files), fact (durable
knowledge about their world), event (something that happened, with a time).

Do NOT store: anything about the current task in progress, code or file
contents, secrets, credentials or API keys, transient state, or anything the
message only speculates about. If the message contains nothing durable, return
[].

Examples:
"my name is Krishna and I use pnpm" ->
[{"text":"The user's name is Krishna.","memory_type":"identity",
  "subject":"user.name","importance":0.95},
 {"text":"The user uses pnpm as their package manager.","memory_type":"preference",
  "subject":"tool.package_manager","importance":0.8}]

"fix the bug in auth.py" ->
[]

"always run the tests before you say you're done" ->
[{"text":"The user wants tests run before any task is reported as done.",
  "memory_type":"instruction","subject":"workflow.verify","importance":0.9}]
"""


# ------------------------------------------------------------------ gating


def should_extract(user_text):
    """Cheap pre-filter. False means "do not spend a call on this"."""
    if not user_text:
        return False

    text = str(user_text).strip()
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return False

    # Code and stack traces are the agent's problem, not the user's biography.
    if "```" in text or "Traceback (most recent call last)" in text:
        return False

    # Pure questions rarely state durable facts.
    if text.endswith("?") and "?" not in text[:-1]:
        return False

    # Agent-directed commands ("run tests", "ls the folder") are instructions
    # to the agent, not memories about the user.
    if re.match(
        r"^\s*(run|execute|open|create|delete|remove|write|read|edit|list|"
        r"show|fix|commit|push|pull|build|install|search|find|explain|"
        r"refactor|add|update|print|cat|cd|ls|grep|make)\b",
        text,
        re.IGNORECASE,
    ) and not _looks_self_referential(text):
        return False

    return len(tokenize(text)) >= MIN_CONTENT_TOKENS


def _looks_self_referential(text):
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in ("my ", "i ", "i'm", "i've", "call me", "prefer", "always")
    )


# --------------------------------------------------------------- LLM path


def extract_with_llm(user_text, client, model, timeout=12):
    """Ask a cheap model for candidate memories. Never raises."""
    if client is None or not model:
        return []

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(user_text)[:MAX_CHARS]},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        content = response.choices[0].message.content or ""
    except Exception:
        # An extraction failure is not a conversation failure. The heuristic
        # path still runs.
        return []

    return _parse_candidates(content)


def _parse_candidates(content):
    """Pull a JSON array out of a model response, whatever wrapper it used."""
    if not content:
        return []

    text = str(content).strip()

    # Strip markdown fences.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()

    candidates = None

    try:
        candidates = json.loads(text)
    except (ValueError, TypeError):
        # Fall back to the outermost bracketed region.
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                candidates = json.loads(text[start: end + 1])
            except (ValueError, TypeError):
                candidates = None

    if candidates is None:
        # Last resort: a single object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                candidates = [json.loads(text[start: end + 1])]
            except (ValueError, TypeError):
                return []

    if isinstance(candidates, dict):
        candidates = [candidates]
    if not isinstance(candidates, list):
        return []

    cleaned = []
    for raw in candidates:
        if not isinstance(raw, dict):
            continue

        text_value = str(raw.get("text") or "").strip()
        if len(text_value) < MIN_CHARS:
            continue

        memory_type = normalize_type(raw.get("memory_type"))
        if memory_type not in AUTO_WRITABLE:
            memory_type = normalize_type(raw.get("memory_type"))

        try:
            importance = float(raw.get("importance", 0.6))
        except (TypeError, ValueError):
            importance = 0.6
        importance = max(0.0, min(1.0, importance))

        subject = raw.get("subject")
        subject = str(subject).strip().lower() if subject else None

        cleaned.append(
            {
                "text": text_value,
                "memory_type": memory_type,
                "subject": subject,
                "importance": importance,
                "source": "auto",
            }
        )

    return cleaned[:MAX_PER_TURN]


# --------------------------------------------------------- heuristic path


# Each pattern is written with inline `(?i:...)` groups around the literal
# parts rather than a global re.IGNORECASE. With a global flag, `[A-Z]` also
# matches lowercase, so "my name is Krishna and I use pnpm" captures the name
# as "Krishna and". The case-sensitivity of the capture classes is load-bearing.
#
# `verb` is the template used to turn a captured fragment into a standalone
# statement. It is per-pattern rather than per-type because the trigger phrase
# carries the polarity: a single "instruction" template would turn "never
# commit to main" into "the user has asked to always commit to main", storing
# the exact opposite of what the user said in a rule the agent will obey for
# months. `verb` receives the cleaned value and returns the full sentence, or
# None to drop the candidate.
#
# (pattern, memory_type, subject, importance, verb, keep_links)
_HEURISTICS = [
    (
        r"\b(?i:my name(?:'s| is)\s+)([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)",
        "identity", "user.name", 0.95, _identity("user.name"),
    ),
    (
        r"\b(?i:call me\s+)([A-Z][\w'-]*)",
        "identity", "user.name", 0.95, _identity("user.name"),
    ),
    (
        r"\b(?i:i(?:'m| am)\s+called\s+)([A-Z][\w'-]*)",
        "identity", "user.name", 0.95, _identity("user.name"),
    ),
    (
        r"\b(?i:i (?:live|reside) in\s+)([A-Za-z][\w .'-]{2,40})",
        "identity", "user.location", 0.8, _identity("user.location"),
    ),
    (
        r"\b(?i:i(?:'m| am) (?:based|located) in\s+)([A-Za-z][\w .'-]{2,40})",
        "identity", "user.location", 0.8, _identity("user.location"),
    ),
    (
        r"\b(?i:i(?:'m| am) from\s+)([A-Z][\w .'-]{2,40})",
        "identity", "user.location", 0.75, _identity("user.location"),
    ),
    (
        r"\b(?i:i(?:'m| am) an?\s+)([a-z][\w -]{0,30}?(?:developer|engineer|"
        r"designer|student|manager|founder|researcher|analyst))",
        "identity", "user.role", 0.85, _identity("user.role"),
    ),
    # "I use pnpm and yarn" is one preference about two tools. The plain
    # two-word capture stopped at the conjunction and recorded only pnpm, so
    # the capture is extended across conjunctions. It is tried before the
    # optional second word so "pnpm and yarn" is taken whole rather than
    # swallowing "and" as the second word.
    (
        r"\b(?i:i (?:use|prefer)\s+)((?:the\s+)?[a-z0-9.+#-]{2,20}"
        r"(?:\s*(?:,|and|or)\s*[a-z0-9.+#-]{2,20})*"
        r"(?:\s+[a-z0-9.+#-]{2,20})?)",
        "preference", None, 0.7, None,
    ),
    (
        r"\b(?i:my favou?rite \w+ is\s+)([\w .+#-]{2,40})",
        "preference", None, 0.75, None,
    ),
    (
        r"\b(?i:i (?:really )?(?:hate|can't stand|dislike)\s+)([\w .+#-]{2,40})",
        "preference", None, 0.7, None,
    ),
    (
        r"\b(?i:always\s+)(.{8,120})",
        "instruction", None, 0.85, _instruction("always"),
    ),
    (
        r"\b(?i:never\s+)(.{8,120})",
        "instruction", None, 0.9, _instruction("never"),
    ),
    (
        r"\b(?i:from now on,?\s+)(.{8,120})",
        "instruction", None, 0.85, _instruction("from_now_on"),
    ),
    (
        r"\b(?i:don'?t ever\s+)(.{8,120})",
        "instruction", None, 0.9, _instruction("never"),
    ),
    (
        r"\b(?i:i(?:'m| am) (?:trying|working) to\s+)(.{5,120})",
        "goal", None, 0.8, None,
    ),
    (
        r"\b(?i:my goal is to\s+)(.{5,120})",
        "goal", "goal.primary", 0.85, None,
    ),
    (r"\b(?i:i want to\s+)(.{5,120})", "goal", None, 0.65, None),
    (
        r"\b(?i:i(?:'m| am) (?:working on|building|developing)\s+)(.{5,120})",
        "project", None, 0.75, None,
    ),
    (
        r"\b(?i:remember that\s+)(.{8,200})",
        "fact", None, 0.7, None,
    ),
    (
        r"\b(?i:my (?:timezone|time zone) is\s+)([\w/ +-]{3,30})",
        "identity", "user.timezone", 0.7, _identity("user.timezone"),
    ),
]

# A captured fragment stops at the first of these: "I live in Berlin and I use
# pnpm" must not record a location of "Berlin and I".
_TRUNCATE_AT = {    "and", "but", "or", "so", "because", "then", "while", "when", "which",
    "who", "that", "although", "though", "however", "but", "since", "until",
    "with", "for",
}

# Filler that tends to land at the end of a greedy capture.
_TRIM_WORDS = {
    "the", "a", "an", "is", "am", "are", "i", "we", "my", "our", "me", "us",
    "use", "uses", "using", "prefer", "prefers", "live", "lives", "based",
    "work", "works", "working", "called", "call", "very", "really", "just",
}

_TRAILING = re.compile(r"[.,;:!?]\s*$")
_STRIP = ".,;:!?\"'"

# Conjunctions that may join two values of the same kind.
_CONJUNCTIONS = {"and", "or", "&", "+"}

# Words that begin a new clause, so a conjunction before one ends the value.
_CLAUSE_STARTERS = {
    "i", "we", "you", "he", "she", "they", "it", "my", "our", "me", "us",
    "im", "ive", "id", "the", "a", "an", "this", "that", "there", "then",
    "so", "but", "because",
}


def _clean_value(value, keep_links=()):
    """Trim a captured fragment to the part that is actually the fact.

    A conjunction is ambiguous: in "Berlin and I" it ends the value, in "pnpm
    and yarn" it joins two values of the same kind. Breaking at every one of
    them recorded half of what the user said, so the decision is made by
    looking ahead - the conjunction is kept only when what follows looks like
    another bare value rather than the start of a new clause.
    """
    words = str(value).split()
    if not words:
        return ""

    links = {link.lower() for link in keep_links} or _CONJUNCTIONS
    kept = []

    for index, word in enumerate(words):
        bare = word.lower().strip(_STRIP)

        if bare in links:
            # Keep the conjunction only when another value follows it.
            if not _continues_a_list(words, index):
                break
            kept.append(word)
            continue

        if bare in _TRUNCATE_AT:
            break

        kept.append(word)

    while kept:
        bare = kept[-1].lower().strip(_STRIP)
        if bare in links or bare in _TRIM_WORDS:
            kept.pop()
        else:
            break

    return " ".join(kept).strip(_STRIP)


def _continues_a_list(words, index):
    """True when the conjunction at `index` introduces another value.

    Requires a bare value token after it, followed by nothing or another
    conjunction. "pnpm and yarn" qualifies; "Berlin and I" does not, because
    "I" is a pronoun and starts the next clause.
    """
    if index + 1 >= len(words):
        return False

    nxt = words[index + 1].lower().strip(_STRIP)
    if len(nxt) < 2 or nxt in _TRUNCATE_AT or nxt in _CLAUSE_STARTERS:
        return False

    following = words[index + 2].lower().strip(_STRIP) if index + 2 < len(words) else ""
    return not following or following in _CONJUNCTIONS or following in _TRUNCATE_AT


def extract_with_heuristics(user_text):
    """Pattern-match the phrasings that show up constantly in real use."""
    if not user_text:
        return []

    text = str(user_text).strip()
    found = []
    seen = set()

    for entry in _HEURISTICS:
        pattern, memory_type, subject, importance = entry[:4]
        keep_links = entry[4] if len(entry) > 4 else ()
        match = re.search(pattern, text)
        if not match:
            continue

        value = _clean_value(match.group(1), keep_links=keep_links)
        if len(value) < 3:
            continue

        key = value.lower()
        if key in seen:
            continue
        seen.add(key)

        sentence = _phrase_to_sentence(text, value, memory_type, subject)
        if not sentence:
            continue

        found.append(
            {
                "text": sentence,
                "memory_type": memory_type,
                "subject": subject,
                "importance": importance,
                "source": "auto",
            }
        )

        if len(found) >= MAX_PER_TURN:
            break

    return found


def _phrase_to_sentence(original, value, memory_type, subject):
    """Turn a captured fragment into a standalone third-person statement.

    Memories are read out of context, months later, by a model that was not
    there. "yarn" is useless; "The user uses yarn." is not.
    """
    lowered = original.lower()

    if memory_type == "identity":
        if subject == "user.name":
            return f"The user's name is {value}."
        if subject == "user.location":
            return f"The user lives in {value}."
        if subject == "user.timezone":
            return f"The user's timezone is {value}."
        if subject == "user.role":
            return f"The user is a {value}."

    if memory_type == "preference":
        if "prefer" in lowered or "favou" in lowered or "favorite" in lowered:
            return f"The user prefers {value}."
        if "hate" in lowered or "dislike" in lowered or "can't stand" in lowered:
            return f"The user dislikes {value}."
        return f"The user uses {value}."

    if memory_type == "instruction":
        return f"The user has asked to always {value}."

    if memory_type == "goal":
        return f"The user is trying to {value}."

    if memory_type == "project":
        return f"The user is working on {value}."

    return f"The user stated: {value}."


# ---------------------------------------------------------------- pipeline


def extract(user_text, client=None, model=None, use_llm=True):
    """Best-effort candidate extraction. Always returns a list.

    Runs the heuristic path first and the LLM path second, then merges. The
    heuristic hits are kept when they say something the LLM did not, which
    matters when the cheap model is unavailable.
    """
    if not should_extract(user_text):
        return []

    candidates = extract_with_heuristics(user_text)

    if use_llm and client is not None and model:
        for candidate in extract_with_llm(user_text, client, model):
            if _restates_existing(candidate, candidates):
                continue
            candidates.append(candidate)

    return _dedupe(candidates)[:MAX_PER_TURN]


def _restates_existing(candidate, existing):
    """True when `candidate` only re-covers something the pattern table caught.

    Both passes read the same sentence, so the model tends to return its own
    phrasing of a fact the heuristics already matched - and its version is
    often the worse one. Asked about "I use pnpm and yarn", the heuristics
    store the whole list while the model answers "The user uses pnpm as their
    package manager", silently dropping yarn.

    A heuristic hit that carries a subject is left alone: a subject is a slot,
    and a later statement about that slot is a correction the store arbitrates
    through supersede. A subject-less hit has no such slot, so an echoing
    candidate that covers the same span should not displace it.
    """
    new_tokens = set(tokenize(candidate["text"]))
    if not new_tokens:
        return True

    for other in existing:
        if candidate["memory_type"] != other["memory_type"]:
            continue
        if other.get("subject"):
            continue

        other_tokens = set(tokenize(other["text"]))
        if not other_tokens:
            continue

        if other_tokens <= new_tokens or new_tokens <= other_tokens:
            return True

        shared = len(other_tokens & new_tokens)
        union = len(other_tokens | new_tokens)
        if union and shared / union >= _ECHO_THRESHOLD:
            return True

    return False


def _dedupe(candidates):
    seen = set()
    result = []
    for candidate in candidates:
        key = candidate["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def store_candidates(candidates, session_id=None, verbose=False):
    """Persist extracted candidates. Returns the items actually created.

    Anything already known is skipped, so extraction stays idempotent across
    repeated turns.
    """
    created = []
    for candidate in candidates:
        text = candidate["text"]
        if not text:
            continue

        existing = store.find_duplicate(text)
        if existing is not None:
            continue

        item = store.insert_memory(
            text=text,
            memory_type=candidate.get("memory_type", "fact"),
            subject=candidate.get("subject"),
            importance=candidate.get("importance"),
            scope=candidate.get("scope", "global"),
            source=candidate.get("source", "auto"),
            session_id=session_id,
        )
        if item is not None:
            created.append(item)

    return created


def explain_candidates(candidates):
    """Preview extraction without writing. Used by the `/extract` debug command."""
    if not candidates:
        return "  (nothing worth remembering)"

    lines = []
    for candidate in candidates:
        subject = candidate.get("subject") or "-"
        lines.append(
            f"  [{candidate['memory_type']}/{candidate['importance']:.2f}] "
            f"<{subject}> {summarize(candidate['text'], 100)}"
        )
    return "\n".join(lines)
