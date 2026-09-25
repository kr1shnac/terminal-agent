"""Long-term memory for Retain.

Public surface (unchanged where it matters, so `from memory import Memory`
keeps working):

    memory = Memory()
    memory.add("The user prefers pnpm", "preference")
    results = memory.search("what package manager")   # list of MemoryItem
    memory.forget(3)
    memory.stats()

Everything else is available on the submodules: `store`, `retrieve`, `decay`,
`extract`, `consolidate`, `context`.

Design in one paragraph: memories are typed rows in SQLite with a
confidence that decays on a per-type half-life and is restored by use; they
are found by fusing BM25 with TF-IDF cosine and reranking on salience; they
are captured automatically from what the user says; and they are deduplicated
and superseded rather than allowed to accumulate.
"""

from . import consolidate as _consolidate
from . import context as _context
from . import db as _db
from . import decay as _decay
from . import extract as _extract
from . import retrieve as _retrieve
from . import store as _store
from .clock import now_iso
from .models import MEMORY_TYPES, MemoryItem, get_type, normalize_type
from .tools import MEMORY_TOOLS
from .tools import dispatch as _dispatch


class Memory:
    """The agent-facing memory object.

    A long-lived instance is expected: it keeps counters for the per-turn
    maintenance cycle (decay, consolidation) so a chatty session does not run
    a full sweep on every message.
    """

    # Maintenance cadence.
    DECAY_EVERY_TURNS = 1
    CONSOLIDATE_EVERY_TURNS = 25

    def __init__(self, path=None, session_id=None, auto_maintain=True, client=None, model=None):
        self._conn = _db.connect(path)
        self.session_id = session_id
        self.auto_maintain = auto_maintain
        self._turns = 0
        self._session_id = session_id
        # Optional, used only for the ambiguous-band merge adjudication in
        # consolidation. Everything else works without a model.
        self.client = client
        self.extract_model = model

        # Bring the store up to date the moment we attach, so a store that was
        # last touched a month ago does not hand stale confidences to the
        # first query. The v1 code did this too, but only at startup and with
        # the naive-datetime bug.
        self._decay_summary = _decay.apply_decay()

    # ---------------------------------------------------------------- write

    def add(
        self,
        text,
        memory_type="fact",
        scope="global",
        subject=None,
        importance=None,
        source="tool",
        confidence=1.0,
    ):
        """Store a memory. Returns the :class:`MemoryItem`.

        Returns the *existing* item when the text is already known, so
        repeated calls are idempotent rather than duplicating.
        """
        return _store.insert_memory(
            text=text,
            memory_type=memory_type,
            scope=scope,
            subject=subject,
            importance=importance,
            confidence=confidence,
            source=source,
            session_id=self.session_id,
        )

    def add_many(self, items):
        created = []
        for raw in items or []:
            if isinstance(raw, MemoryItem):
                created.append(self.add(raw.text, raw.memory_type, raw.subject))
            elif isinstance(raw, dict):
                created.append(self.add(**raw))
            elif isinstance(raw, str):
                created.append(self.add(raw))
        return [item for item in created if item is not None]

    def forget(self, memory_id, mode="archive"):
        """Remove a memory. ``archive`` keeps the row; ``delete`` erases it."""
        if mode == "delete":
            _store.delete(memory_id)
            return True
        if _store.get(memory_id) is None:
            return False
        _store.archive(memory_id, reason="user_requested")
        return True

    def update(self, memory_id, **fields):
        return _store.update_fields(memory_id, **fields)

    def restore(self, memory_id):
        return _store.restore(memory_id)

    # ----------------------------------------------------------------- read

    def search(self, query, top_k=6, scope=None, reinforce=True, diversify=True):
        """Recall memories for a query.

        Returns a list of :class:`MemoryItem` (not the retriever's internal
        dicts) so the common case stays a one-liner. Use
        :meth:`search_detailed` when you want the score breakdown.
        """
        return [
            entry["item"]
            for entry in self.search_detailed(
                query,
                top_k=top_k,
                scope=scope,
                reinforce=reinforce,
                diversify=diversify,
            )
        ]

    def search_detailed(
        self, query, top_k=6, scope=None, reinforce=True, diversify=True
    ):
        """Same as :meth:`search` but returns ``{"item", "score", "signals"}``.

        Used by the `/recall` command to show *why* something surfaced, which
        is the only practical way to tune a retriever.
        """
        return _retrieve.retrieve(
            query,
            top_k=top_k,
            scope=scope,
            reinforce=reinforce,
            diversify=diversify,
        )

    def get(self, memory_id):
        return _store.get(memory_id)

    def get_all(self, include_archived=False):
        items = _store.get_active()
        if include_archived:
            seen = {item.id for item in items}
            items = items + [i for i in _store.get_archived() if i.id not in seen]
        return items

    def by_type(self, memory_type):
        return _store.get_by_type(memory_type)

    def by_subject(self, subject):
        return _store.get_by_subject(subject)

    def history(self, memory_id, limit=20):
        return _store.access_history(memory_id, limit)

    def stats(self):
        return _store.stats()

    def forecast(self, memory_id, days=30):
        return _decay.forecast(memory_id, days)

    # ------------------------------------------------------------- context

    def context(self, query, budget=_context.DEFAULT_BUDGET, top_k=6):
        """The prompt block for a query, ready to splice into a system prompt.

        Retrieval and rendering are bundled here so the caller cannot forget
        the budget.
        """
        results = self.search_detailed(query, top_k=top_k)
        return _context.build_system_memory_block(results, budget=budget), results

    def render(self, results=None, include_archived=False):
        entries = results or [
            {"item": item, "score": float(item.confidence or 0.0)}
            for item in self.get_all(include_archived=include_archived)
        ]
        return _context.render_table(entries, include_archived=include_archived)

    def explain(self, results):
        return _retrieve.explain(results)

    # ----------------------------------------------------------- extraction

    def observe(self, user_text, client=None, model=None, use_llm=True):
        """Extract memories from what the user just said, and store them.

        Runs after every user turn. Returns the newly created items, so the
        caller can show "noted: ..." without a second lookup.
        """
        candidates = _extract.extract(
            user_text, client=client, model=model, use_llm=use_llm
        )
        if not candidates:
            return []
        return _extract.store_candidates(
            candidates, session_id=self.session_id
        )

    def preview(self, user_text, client=None, model=None, use_llm=True):
        """What `observe` *would* store, without storing it."""
        return _extract.extract(
            user_text, client=client, model=model, use_llm=use_llm
        )

    # ------------------------------------------------------------ lifecycle

    def tick(self):
        """Per-turn maintenance. Called once per user message."""
        self._turns += 1

        if not self.auto_maintain:
            return {"decayed": 0, "archived": 0}

        if self._turns % self.DECAY_EVERY_TURNS == 0:
            self._decay_summary = _decay.apply_decay()

        if self._turns % self.CONSOLIDATE_EVERY_TURNS == 0:
            self.consolidate(client=self.client, model=self.extract_model)

        return self._decay_summary

    def consolidate(self, dry_run=False, client=None, model=None):
        report = _consolidate.consolidate(
            dry_run=dry_run, client=client, model=model
        )
        if not dry_run:
            _consolidate.mark_run()
        return report

    def decay_now(self):
        return _decay.apply_decay()

    def reset(self):
        _store.forget_all()

    # ---------------------------------------------------------------- tools

    @staticmethod
    def tools():
        return MEMORY_TOOLS

    def run_tool(self, name, args):
        return _dispatch(name, args or {}, self)


def get_memory(path=None, session_id=None):
    """Shared instance, so the agent and its tools never hold two objects."""
    global _INSTANCE

    if _INSTANCE is None:
        _INSTANCE = Memory(path=path, session_id=session_id)
    return _INSTANCE


_INSTANCE = None


__all__ = [
    "Memory",
    "MemoryItem",
    "MEMORY_TYPES",
    "MEMORY_TOOLS",
    "get_memory",
    "get_type",
    "normalize_type",
    "now_iso",
    "store",
    "retrieve",
    "decay",
    "extract",
    "consolidate",
    "context",
    "db",
]
