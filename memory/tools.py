"""Memory tool schemas and dispatch.

The v1 agent exposed a single `remember` tool with two parameters. That is
enough to store something and not enough to manage it: there is no way to
recall on demand, correct a mistake, or remove something. Four tools:

  remember  - store a durable fact, with type/subject/importance
  recall    - search memory explicitly, for when the model suspects it is
              missing something
  forget    - archive or hard-delete by id
  memories  - list what is stored, so the model can see its own state

The `subject` parameter is what makes corrections work: a new memory with the
same subject supersedes the old one, so "I moved to Berlin" cleanly replaces
"I live in London" instead of leaving both in the prompt.
"""

import json

from . import store
from .context import render_table
from .models import AUTO_WRITABLE, MEMORY_TYPES, get_type
from .retrieve import explain, retrieve
from .text import summarize

_TYPE_HELP = ", ".join(
    f"{name} ({spec.description.split('.')[0].lower()})"
    for name, spec in MEMORY_TYPES.items()
)

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Store something durable about the user or their work for "
                "future conversations. Only store what will still matter "
                "later: identity, preferences, standing instructions, goals, "
                "project context, or notable events. Do NOT store secrets, "
                "credentials, file contents, or the state of the current "
                "task. Provide a 'subject' for anything singular that could "
                "be replaced later (a name, a preferred tool) so that a newer "
                "value supersedes the old one instead of both being kept."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "One self-contained third-person sentence, e.g. "
                            "'The user prefers pnpm over npm.'"
                        ),
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": sorted(
                            name
                            for name in MEMORY_TYPES
                            if MEMORY_TYPES[name].user_writable
                        ),
                        "description": f"Type of memory. Options: {_TYPE_HELP}",
                    },
                    "subject": {
                        "type": "string",
                        "description": (
                            "Optional dotted key for a singular replaceable "
                            "value, e.g. 'user.name', 'user.editor', "
                            "'tool.package_manager'. Reuse a subject when "
                            "updating a previously stored value."
                        ),
                    },
                    "importance": {
                        "type": "number",
                        "description": (
                            "0.0 to 1.0. How much this should be trusted "
                            "and remembered. Default 0.6."
                        ),
                    },
                },
                "required": ["text", "memory_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Search long-term memory. Use when you need information you "
                "were not given in this conversation, or to check what is "
                "already known before storing something new."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return. Default 5.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": (
                "Remove a memory by id. Use 'archive' to hide it while "
                "keeping the record, or 'delete' to remove it entirely. Use "
                "this when the user asks you to forget something, or to "
                "correct something you stored by mistake."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "integer",
                        "description": "The id shown in memories.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["archive", "delete"],
                        "description": "Default 'archive'.",
                    },
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memories",
            "description": (
                "List what is currently stored in long-term memory, grouped "
                "by type with ids, confidence and usage counts. Use before "
                "'forget' if you are unsure of the id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "description": "Optional type to filter by.",
                    },
                    "include_archived": {
                        "type": "boolean",
                        "description": "Default false.",
                    },
                },
            },
        },
    },
]


def dispatch(name, args, memory):
    """Route a memory tool call. Returns the string fed back to the model."""
    if name == "remember":
        return _remember(args, memory)
    if name == "recall":
        return _recall(args)
    if name == "forget":
        return _forget(args)
    if name == "memories":
        return _memories(args)
    return f"Error: unknown memory tool '{name}'"


def _remember(args, memory):
    text = str(args.get("text") or "").strip()
    if not text:
        return "Error: nothing to remember (text was empty)."

    memory_type = args.get("memory_type") or "fact"
    if memory_type not in AUTO_WRITABLE and memory_type not in MEMORY_TYPES:
        return (
            f"Error: '{memory_type}' is not a valid memory_type. "
            f"Use one of: {', '.join(sorted(MEMORY_TYPES))}"
        )

    # `user_writable` is a real gate, not documentation. `session` rows are
    # the agent's own short-lived scratch state, and letting `remember` create
    # them would let the model file a transient observation as if it were
    # something worth keeping.
    if not get_type(memory_type).user_writable:
        return (
            f"Error: '{memory_type}' memories are managed by the agent, not "
            f"written with remember(). Use one of: {', '.join(sorted(AUTO_WRITABLE))}"
        )

    importance = args.get("importance")
    try:
        importance = float(importance) if importance is not None else None
    except (TypeError, ValueError):
        importance = None

    subject = args.get("subject")
    subject = str(subject).strip().lower() if subject else None

    # Look before inserting so the reply can say "already known", but let the
    # store do the dedupe. Reinforcing here would count a restatement as a
    # recall and hand the memory permanent decay relief for mere repetition.
    existing = store.find_duplicate(text)

    item = memory.add(
        text=text,
        memory_type=memory_type,
        subject=subject,
        importance=importance,
        source="tool",
    )

    if item is None:
        return "Error: could not store that memory."

    if existing is not None:
        if store.is_refinement(existing, text):
            return (
                f"Refined memory #{item.id} with the more precise wording: "
                f"{summarize(item.text, 100)}"
            )
        return (
            f"Already remembered (id {item.id}, no change needed): "
            f"{summarize(item.text, 100)}"
        )

    # Report a supersede explicitly, so the model knows the old value is gone
    # rather than assuming both are live.
    note = ""
    if item.supersedes:
        previous = store.get(item.supersedes)
        if previous is not None:
            note = (
                f" (superseded id {previous.id}: "
                f"{summarize(previous.text, 60)})"
            )

    return f"Memory saved as id {item.id}{note}: {summarize(item.text, 100)}"


def _recall(args):
    query = str(args.get("query") or "").strip()
    if not query:
        return "Error: recall needs a query."

    try:
        limit = int(args.get("limit") or 5)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, 20))

    results = retrieve(query, top_k=limit)
    if not results:
        return f"No memories matched '{summarize(query, 60)}'."

    lines = [f"{len(results)} memories for '{summarize(query, 60)}':"]
    for entry in results:
        item = entry["item"]
        subject = f" <{item.subject}>" if item.subject else ""
        lines.append(
            f"  #{item.id} ({item.memory_type}{subject}, "
            f"{float(item.confidence or 0.0):.0%} conf, score {entry['score']:.2f}) "
            f"{item.text}"
        )
    return "\n".join(lines)


def _forget(args):
    try:
        memory_id = int(args.get("memory_id"))
    except (TypeError, ValueError):
        return "Error: forget needs an integer memory_id."

    mode = str(args.get("mode") or "archive").lower()
    item = store.get(memory_id)

    if item is None:
        return f"Error: no memory with id {memory_id}."

    if mode == "delete":
        store.delete(memory_id)
        return f"Deleted memory #{memory_id}: {summarize(item.text, 80)}"

    if mode != "archive":
        return "Error: mode must be 'archive' or 'delete'."

    store.archive(memory_id, reason="user_requested")
    return f"Archived memory #{memory_id}: {summarize(item.text, 80)}"


def _memories(args):
    memory_type = args.get("memory_type")
    include_archived = bool(args.get("include_archived"))

    if memory_type:
        items = store.get_by_type(memory_type)
    else:
        items = store.get_active()

    if include_archived:
        seen = {item.id for item in items}
        items = items + [i for i in store.get_archived() if i.id not in seen]

    if not items:
        return "Long-term memory is empty."

    entries = [{"item": item, "score": float(item.confidence or 0.0)} for item in items]
    return render_table(entries, include_archived=include_archived)


def describe_types():
    """Help text for the `remember` tool."""
    return json.dumps(
        {name: spec.description for name, spec in MEMORY_TYPES.items()}, indent=2
    )
