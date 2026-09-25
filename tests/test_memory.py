"""Offline test suite for the memory system.

No API key and no network required: extraction, consolidation and retrieval
are all testable through their deterministic paths, and the LLM paths are
exercised with a stub client.

    python -m unittest discover -s tests -v
"""

import json
import os
import shutil
import tempfile
import types
import unittest
from datetime import timedelta

# Point the store at a temp file *before* memory is imported, since db.connect
# caches a module-level handle.
_TMP_DIR = tempfile.mkdtemp(prefix="retain-tests-")
os.environ["RETAIN_DB_PATH"] = os.path.join(_TMP_DIR, "test.db")

from memory import Memory, consolidate, context, db, decay, extract, models, retrieve, store  # noqa: E402
from memory.clock import days_since, now_iso, parse_iso, to_iso, utcnow  # noqa: E402
from memory.text import (  # noqa: E402
    fts_query,
    norm_hash,
    similarity,
    tokenize,
    weighted_jaccard,
    build_idf,
)


def tearDownModule():
    db.reset_connection()
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


class MemoryTestCase(unittest.TestCase):
    """Gives every test a private, empty store."""

    def setUp(self):
        db.reset_connection()
        store.forget_all()
        self.memory = Memory(auto_maintain=False)

    def seed(self, text, memory_type="fact", **kwargs):
        """Store one memory and return it."""
        return self.memory.add(text, memory_type, **kwargs)

    def seed_many(self, *texts, memory_type="fact", **kwargs):
        return [self.memory.add(text, memory_type, **kwargs) for text in texts]


# ------------------------------------------------------------------- clock


class TestClock(unittest.TestCase):
    def test_parse_utc_offset(self):
        parsed = parse_iso("2026-09-25T10:00:00+00:00")
        self.assertIsNotNone(parsed.tzinfo)
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)

    def test_parse_legacy_naive_as_utc(self):
        # v1 rows were written without an offset; they must still load.
        parsed = parse_iso("2026-09-25T10:00:00")
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)

    def test_parse_z_suffix(self):
        self.assertIsNotNone(parse_iso("2026-09-25T10:00:00Z"))

    def test_parse_garbage_returns_none(self):
        self.assertIsNone(parse_iso("not a timestamp"))
        self.assertIsNone(parse_iso(""))
        self.assertIsNone(parse_iso(None))

    def test_aware_and_naive_timestamps_are_comparable(self):
        # The v1 crash: comparing a naive stamp to an aware one.
        aware = days_since("2026-09-20T00:00:00+00:00")
        naive = days_since("2026-09-20T00:00:00")
        self.assertAlmostEqual(aware, naive, places=3)

    def test_days_since_never_negative(self):
        # A future-dated row must not gain confidence.
        future = to_iso(utcnow() + timedelta(days=5))
        self.assertEqual(days_since(future), 0.0)

    def test_days_since_missing_is_zero(self):
        self.assertEqual(days_since(None), 0.0)


# -------------------------------------------------------------------- text


class TestText(unittest.TestCase):
    def test_tokenize_drops_stopwords(self):
        self.assertNotIn("the", tokenize("The user is here"))
        self.assertNotIn("is", tokenize("The user is here"))

    def test_tokenize_keeps_short_domain_tokens(self):
        for token in ("js", "ci", "ai"):
            self.assertIn(token, tokenize(f"I work with {token} daily"))

    def test_tokenize_drops_long_bare_numbers(self):
        self.assertEqual(tokenize("build 1234567 rust"), ["build", "rust"])

    def test_normalize_collapses_punctuation_and_case(self):
        # The apostrophe must be stripped, or "user's" never matches "user".
        self.assertEqual(
            norm_hash("User's name is Krishna."),
            norm_hash("user name is krishna"),
        )

    def test_norm_hash_differs_for_different_text(self):
        self.assertNotEqual(norm_hash("prefers pnpm"), norm_hash("prefers yarn"))

    def test_fts_query_quotes_every_term(self):
        # "what" and "is" are stopwords, so only the content terms survive.
        self.assertEqual(fts_query("what is my name"), '"my"* AND "name"*')

    def test_fts_query_or_mode(self):
        self.assertEqual(
            fts_query("peanut allergy", mode="OR"), '"peanut"* OR "allergy"*'
        )

    def test_fts_query_neutralises_fts_operators(self):
        # Bare AND/OR/NOT would otherwise change the parse.
        query = fts_query("a AND OR NOT b")
        self.assertNotIn(" NOT ", query)
        for part in query.split(" AND "):
            self.assertTrue(part.startswith('"'), part)

    def test_fts_query_handles_hostile_input(self):
        for bad in ['unbalanced "quote', "drop table memory;--", "*(){}", "a" * 500]:
            query = fts_query(bad)
            if query is not None:
                self.assertNotIn('"', query[1:-1].replace('"', ""), query)

    def test_fts_query_returns_none_for_no_terms(self):
        self.assertIsNone(fts_query("*** ???"))

    def test_fts_query_caps_term_count(self):
        self.assertLessEqual(len(fts_query(" ".join(f"word{i}" for i in range(50))).split(" AND ")), 12)

    def test_weighted_jaccard_separates_paraphrase_from_contradiction(self):
        corpus = [
            tokenize(t)
            for t in [
                "The user prefers pnpm over npm",
                "The user prefers pnpm instead of npm",
                "The user prefers yarn over npm",
            ]
        ]
        idf = build_idf(corpus)
        base = corpus[0]

        paraphrase = weighted_jaccard(base, corpus[1], idf)
        contradiction = weighted_jaccard(base, corpus[2], idf)

        self.assertGreater(paraphrase, contradiction)
        # A contradiction must sit clearly below the auto-merge bar.
        self.assertLess(contradiction, consolidate.AUTO_MERGE_THRESHOLD)

    def test_similarity_is_zero_for_disjoint(self):
        self.assertEqual(similarity(tokenize("alpha"), tokenize("beta")), 0.0)


# ------------------------------------------------------------------- store


class TestStore(MemoryTestCase):
    def test_insert_returns_id(self):
        item = self.memory.add("The user likes tea", "preference")
        self.assertIsNotNone(item.id)

    def test_repeated_add_is_idempotent(self):
        first = self.memory.add("The user likes tea", "preference")
        second = self.memory.add("The user likes tea", "preference")
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(store.get_active()), 1)

    def test_near_duplicate_is_rejected(self):
        self.memory.add("The user prefers pnpm over npm", "preference")
        self.memory.add("The user prefers pnpm, not npm", "preference")
        self.assertEqual(len(store.get_active()), 1)

    def test_contradiction_is_not_a_duplicate(self):
        self.memory.add("The user prefers pnpm over npm", "preference")
        self.memory.add("The user prefers yarn over npm", "preference")
        self.assertEqual(len(store.get_active()), 2)

    def test_empty_text_is_rejected(self):
        self.assertIsNone(self.memory.add("   "))
        self.assertIsNone(self.memory.add(""))

    def test_subject_supersedes_previous_value(self):
        old = self.memory.add("The user lives in London", "identity", subject="user.location")
        new = self.memory.add("The user lives in Tokyo", "identity", subject="user.location")

        self.assertEqual(new.supersedes, old.id)
        self.assertTrue(store.get(old.id).is_archived)
        self.assertEqual(
            [i.text for i in store.get_by_subject("user.location")],
            ["The user lives in Tokyo"],
        )

    def test_supersede_does_not_archive_itself(self):
        self.memory.add("The user's editor is vim", "preference", subject="tool.editor")
        again = self.memory.add("The user's editor is vim", "preference", subject="tool.editor")
        self.assertEqual(again.archived_reason, None)

    def test_type_aliases_normalize(self):
        self.assertEqual(models.normalize_type("user_preference"), "preference")
        self.assertEqual(models.normalize_type("RULE"), "instruction")
        self.assertEqual(models.normalize_type("nonsense"), "fact")
        self.assertEqual(models.normalize_type(None), "fact")

    def test_importance_defaults_by_type(self):
        identity = self.memory.add("The user is a developer", "identity")
        event = self.memory.add("The user shipped a release", "event")
        self.assertGreater(identity.importance, event.importance)

    def test_update_text_reindexes(self):
        item = self.memory.add("The user enjoys hiking", "preference")
        store.update_text(item.id, "The user enjoys long hikes in the hills")
        self.assertEqual(len(self.memory.search("hiking")), 1)
        self.assertEqual(len(self.memory.search("kayaking")), 0)

    def test_update_fields_ignores_unknown_keys(self):
        item = self.memory.add("The user likes tea", "preference")
        updated = store.update_fields(item.id, nonsense="x", importance=0.9)
        self.assertEqual(updated.importance, 0.9)

    def test_forget_archive_then_restore(self):
        item = self.memory.add("The user likes tea", "preference")
        self.memory.forget(item.id)
        self.assertEqual(len(store.get_active()), 0)
        self.assertEqual(len(store.get_archived()), 1)

        self.memory.restore(item.id)
        self.assertEqual(len(store.get_active()), 1)

    def test_forget_delete_is_permanent(self):
        item = self.memory.add("The user likes tea", "preference")
        self.memory.forget(item.id, mode="delete")
        self.assertIsNone(store.get(item.id))

    def test_access_history_is_recorded(self):
        item = self.memory.add("The user's name is Krishna", "identity")
        store.log_access(item.id, query="name", score=0.9)
        history = self.memory.history(item.id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["query"], "name")

    def test_stats(self):
        self.seed_many("a fact about tea", "another fact about coffee")
        stats = self.memory.stats()
        self.assertEqual(stats["live"], 2)
        self.assertEqual(stats["schema_version"], db.SCHEMA_VERSION)


# ------------------------------------------------------------------- decay


class TestDecay(MemoryTestCase):
    def _age(self, item, days):
        stamp = to_iso(utcnow() - timedelta(days=days))
        conn = db.connect()
        conn.execute(
            "UPDATE memory SET last_accessed_at = ?, created_at = ? WHERE id = ?",
            (stamp, stamp, item.id),
        )
        conn.commit()

    def test_decay_is_idempotent(self):
        item = self.memory.add("The user drank coffee", "event")
        self._age(item, 30)

        seen = []
        for _ in range(5):
            decay.apply_decay()
            seen.append(round(store.get(item.id).confidence, 6))

        self.assertEqual(len(set(seen)), 1, f"decay compounded: {seen}")

    def test_decay_matches_half_life(self):
        item = self.memory.add("The user drank coffee", "event")
        # Pin the importance floor out of the way for a clean comparison.
        conn = db.connect()
        conn.execute("UPDATE memory SET importance = 0.0 WHERE id = ?", (item.id,))
        conn.commit()

        self._age(item, 21)
        decay.apply_decay()
        # event half-life is 21 days, so one half-life must halve confidence.
        self.assertAlmostEqual(store.get(item.id).confidence, 0.5, delta=0.03)

    def test_importance_floor_protects_important_memories(self):
        item = self.memory.add("The user is allergic to peanuts", "identity")
        self._age(item, 900)
        decay.apply_decay()
        self.assertGreater(store.get(item.id).confidence, 0.5)
        self.assertFalse(store.get(item.id).is_archived)

    def test_reinforcement_always_increases_confidence(self):
        item = self.seed("The user prefers dark mode", "preference")
        # Aged well past the cap so there is headroom to observe the increase.
        self._age(item, 400)

        previous = None
        for _ in range(4):
            decay.reinforce(store.get(item.id))
            current = store.get(item.id).confidence
            if previous is not None:
                self.assertGreater(current, previous, "reinforcement lowered confidence")
            previous = current

    def test_reinforcement_resets_the_decay_clock(self):
        item = self.memory.add("The user prefers dark mode", "preference")
        self._age(item, 60)
        decay.reinforce(store.get(item.id))
        self.assertLess(days_since(store.get(item.id).last_accessed_at), 0.01)

    def test_reinforcement_slows_decay_rate(self):
        item = self.memory.add("The user prefers dark mode", "preference")
        before = store.get(item.id).decay_rate
        decay.reinforce(store.get(item.id))
        self.assertLess(store.get(item.id).decay_rate, before)

    def test_decay_relief_is_capped(self):
        item = self.memory.add("The user prefers dark mode", "preference")
        for _ in range(decay.MAX_DECAY_RELIEF_USES + 6):
            decay.reinforce(store.get(item.id))
        rate = store.get(item.id).decay_rate
        floor_rate = decay.MIN_DECAY_RATE
        self.assertGreaterEqual(rate, floor_rate)

    def test_expired_session_memory_is_archived(self):
        item = self.memory.add("Working on the parser bug", "session")
        past = to_iso(utcnow() - timedelta(days=5))
        conn = db.connect()
        conn.execute("UPDATE memory SET expires_at = ? WHERE id = ?", (past, item.id))
        conn.commit()

        decay.apply_decay()
        self.assertTrue(store.get(item.id).is_archived)

    def test_stale_event_is_archived_but_identity_is_not(self):
        stale = self.seed("The user fixed a typo today", "event")
        kept = self.seed("The user's name is Krishna", "identity")
        self._age(stale, 400)
        self._age(kept, 400)

        decay.apply_decay()

        self.assertTrue(store.get(stale.id).is_archived)
        self.assertFalse(store.get(kept.id).is_archived)

    def test_idle_limit_scales_with_importance(self):
        # Without an importance-scaled idle window, the retention floor makes
        # every memory permanent and nothing is ever actually forgotten.
        event = self.seed("The user fixed a typo today", "event")
        identity = self.seed("The user's name is Krishna", "identity")

        self.assertLess(decay.idle_limit_days(event), decay.idle_limit_days(identity))

    def test_abandoned_memory_is_eventually_released(self):
        # A low-importance memory that plateaus at its floor must still be
        # droppable, or "decay" is purely cosmetic.
        item = self.seed("The user bought a new keyboard", "event")
        self._age(item, decay.idle_limit_days(item) + 10)

        decay.apply_decay()

        self.assertTrue(store.get(item.id).is_archived)
        self.assertEqual(store.get(item.id).archived_reason, "idle")

    def test_recently_used_memory_at_its_floor_is_kept(self):
        item = self.seed("The user bought a new keyboard", "event")
        self._age(item, decay.idle_limit_days(item) + 10)
        decay.reinforce(store.get(item.id))
        decay.apply_decay()
        self.assertFalse(store.get(item.id).is_archived)

    def test_forecast_is_monotonically_decreasing(self):
        item = self.memory.add("The user prefers dark mode", "preference")
        points = decay.forecast(item.id)
        values = [p["confidence"] for p in points]
        self.assertEqual(values, sorted(values, reverse=True))


# ---------------------------------------------------------------- retrieve


class TestRetrieve(MemoryTestCase):
    def test_finds_stemmed_match(self):
        self.seed("The user is working on a parser bug", "event")
        # "parser" vs "parsers" only matches via porter stemming in FTS5.
        self.assertEqual(len(self.memory.search("parsers", reinforce=False)), 1)

    def test_prefers_relevant_over_irrelevant(self):
        self.seed("The user drinks oat milk", "preference")
        self.seed("The user is building a web scraper", "goal")
        self.seed("The user prefers dark mode in the editor", "preference")

        results = self.memory.search("what editor theme do they like", reinforce=False)
        self.assertTrue(results)
        self.assertIn("dark mode", results[0].text)

    def test_empty_query_falls_back_to_salience(self):
        self.seed("The user's name is Krishna", "identity")
        self.seed("The user drank a coffee", "event")
        results = self.memory.search("!!!", reinforce=False)
        self.assertTrue(results)
        self.assertEqual(results[0].text, "The user's name is Krishna")

    def test_fallback_does_not_reinforce(self):
        # Saying "hi" must not mark every core memory as "used".
        item = self.seed("The user's name is Krishna", "identity")
        self.memory.search("hi", reinforce=True)
        self.assertEqual(store.get(item.id).access_count, 0)

    def test_real_recall_does_reinforce_and_log(self):
        item = self.seed("The user's name is Krishna", "identity")
        self.memory.search("what is my name", top_k=1)
        fresh = store.get(item.id)
        self.assertEqual(fresh.access_count, 1)
        self.assertEqual(len(self.memory.history(item.id)), 1)

    def test_min_score_gates_weak_matches(self):
        self.seed("The user plays the cello", "preference")
        self.assertEqual(len(self.memory.search("quantum chromodynamics", reinforce=False)), 0)

    def test_mmr_diversifies_near_duplicates(self):
        self.seed_many(
            "The user prefers pnpm over npm",
            "The user prefers pnpm instead of npm",
            "The user prefers yarn over npm",
            "The user works in the terminal",
            "The user's timezone is Europe/London",
            memory_type="preference",
        )
        results = self.memory.search("pnpm", top_k=2, reinforce=False)
        self.assertEqual(len(results), 2)
        # The two selected must not be paraphrases of each other.
        self.assertNotEqual(results[0].text, results[1].text)

    def test_superseded_memory_is_not_retrievable(self):
        old = self.memory.add("The user lives in London", "identity", subject="user.location")
        self.memory.add("The user lives in Tokyo", "identity", subject="user.location")

        texts = [i.text for i in self.memory.search("where does the user live", reinforce=False)]
        self.assertNotIn("The user lives in London", texts)
        self.assertIn("The user lives in Tokyo", texts)

    def test_archived_memory_is_not_retrievable(self):
        item = self.memory.add("The user plays the cello", "preference")
        self.memory.forget(item.id)
        self.assertEqual(len(self.memory.search("cello", reinforce=False)), 0)

    def test_signals_are_exposed_for_debugging(self):
        self.seed("The user's name is Krishna", "identity")
        entry = self.memory.search_detailed("name", reinforce=False)[0]
        for key in ("lexical", "salience", "confidence", "importance", "recency", "usage"):
            self.assertIn(key, entry["signals"])
        self.assertGreater(entry["score"], 0)

    def test_explain_is_readable(self):
        self.seed("The user's name is Krishna", "identity")
        text = self.memory.explain(self.memory.search_detailed("name", reinforce=False))
        self.assertIn("score=", text)
        self.assertIn("Krishna", text)

    def test_search_does_not_mutate_the_query_pool(self):
        self.seed("The user's name is Krishna", "identity")
        self.memory.search("name", reinforce=True)
        self.assertEqual(len(store.get_active()), 1)


# ----------------------------------------------------------------- extract


class TestExtract(unittest.TestCase):
    def test_should_extract_rejects_trivial_and_agent_commands(self):
        for text in ["", "ok", "run the tests", "ls the folder", "fix auth.py"]:
            self.assertFalse(extract.should_extract(text), text)

    def test_should_extract_rejects_code(self):
        self.assertFalse(extract.should_extract("```python\nprint('x')\n```"))
        self.assertFalse(extract.should_extract("Traceback (most recent call last): boom"))

    def test_should_extract_accepts_self_disclosure(self):
        for text in [
            "my name is Krishna and I use pnpm",
            "always run the tests before saying done",
            "I am trying to learn Rust this year",
        ]:
            self.assertTrue(extract.should_extract(text), text)

    def test_heuristic_name_capture_is_case_sensitive(self):
        # A global re.IGNORECASE made [A-Z] match lowercase and captured
        # "Krishna and".
        found = extract.extract_with_heuristics("my name is Krishna and I use pnpm")
        names = [c for c in found if c["subject"] == "user.name"]
        self.assertEqual(names[0]["text"], "The user's name is Krishna.")

    def test_heuristic_location_stops_at_conjunction(self):
        found = extract.extract_with_heuristics("I live in Berlin and I use pnpm")
        locations = [c for c in found if c["subject"] == "user.location"]
        self.assertEqual(locations[0]["text"], "The user lives in Berlin.")

    def test_heuristics_produce_third_person_sentences(self):
        for text in extract.extract_with_heuristics("my name is Krishna"):
            self.assertTrue(text["text"].startswith("The user"))
            self.assertTrue(text["text"].endswith("."))

    def test_heuristics_do_not_extract_from_commands(self):
        self.assertEqual(extract.extract_with_heuristics("fix the bug in auth.py"), [])

    def test_llm_json_parsing(self):
        payload = json.dumps(
            [
                {
                    "text": "The user's name is Krishna.",
                    "memory_type": "identity",
                    "subject": "user.name",
                    "importance": 0.95,
                }
            ]
        )
        for wrapper in (payload, f"```json\n{payload}\n```", f"Sure!\n{payload}\nHope that helps."):
            parsed = extract._parse_candidates(wrapper)
            self.assertEqual(len(parsed), 1, wrapper[:30])
            self.assertEqual(parsed[0]["memory_type"], "identity")

    def test_llm_json_parsing_rejects_garbage(self):
        for junk in ("", "not json", "[1, 2, 3]", "null", '{"a": 1}'):
            self.assertEqual(extract._parse_candidates(junk), [], junk)

    def test_llm_failure_falls_back_to_heuristics(self):
        class Boom:
            class chat:
                @property
                def completions(self):
                    class C:
                        def create(self, **kw):
                            raise RuntimeError("api down")
                    return C()

        found = extract.extract("my name is Krishna", client=Boom(), model="m")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["subject"], "user.name")

    def test_no_client_uses_heuristics_only(self):
        found = extract.extract("my name is Krishna", client=None, use_llm=False)
        self.assertEqual(len(found), 1)

    def test_max_candidates_per_turn(self):
        text = "my name is Krishna, I live in Berlin, I use pnpm, and I always run tests"
        self.assertLessEqual(len(extract.extract(text, use_llm=False)), extract.MAX_PER_TURN)

    def test_store_candidates_is_idempotent(self):
        db.reset_connection()
        store.forget_all()
        memory = Memory(auto_maintain=False)
        candidates = extract.extract_with_heuristics("my name is Krishna")

        first = extract.store_candidates(candidates)
        second = extract.store_candidates(candidates)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 0)
        self.assertEqual(len(store.get_active()), 1)


# ------------------------------------------------------------- consolidate


class FakeClient:
    """Minimal stand-in for the OpenAI client."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.prompts = []

    def _completions(self):
        client = self

        class Completions:
            def create(self, **kwargs):
                client.calls += 1
                client.prompts.append(kwargs["messages"][-1]["content"])
                message = types.SimpleNamespace(content=client.reply)
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        return Completions()

    @property
    def chat(self):
        return types.SimpleNamespace(completions=self._completions())


class TestConsolidate(MemoryTestCase):
    def test_merges_verbatim_restatement(self):
        # Exact restatements never reach consolidation: the dedupe on insert
        # already collapsed them.
        self.seed("The user's timezone is Europe/London", "identity")
        self.seed("The user's timezone is Europe/London", "identity")

        report = self.memory.consolidate()
        self.assertEqual(report["merged"], [])
        self.assertEqual(len(store.get_active()), 1)

    def test_contradiction_is_never_merged_without_a_model(self):
        self.seed("The user prefers pnpm over npm", "preference")
        self.seed("The user prefers yarn over npm", "preference")

        self.memory.consolidate()
        self.assertEqual(len(store.get_active()), 2)

    def test_model_adjudication_merges_the_paraphrase(self):
        self.seed("The user prefers pnpm over npm", "preference")
        self.seed("The user prefers pnpm instead of npm", "preference")

        client = FakeClient('```json\n{"same": [1]}\n```')
        self.memory.consolidate(client=client, model="fake")

        self.assertEqual(client.calls, 1)
        self.assertEqual(len(store.get_active()), 1)

    def test_the_model_is_shown_both_statements(self):
        # The prompt used to list only one side of each candidate while asking
        # the model to judge "the two statements", so the model was being asked
        # a question it could not answer from what it was given.
        self.seed("The user prefers pnpm over npm", "preference")
        self.seed("The user prefers pnpm instead of npm", "preference")

        client = FakeClient('{"same": []}')
        self.memory.consolidate(client=client, model="fake")

        self.assertEqual(client.calls, 1)
        prompt = client.prompts[0]
        self.assertIn("The user prefers pnpm over npm", prompt)
        self.assertIn("The user prefers pnpm instead of npm", prompt)

    def test_hostile_model_output_merges_nothing(self):
        self.seed("The user prefers pnpm over npm", "preference")
        self.seed("The user prefers yarn over npm", "preference")

        for reply in ("", "not json", '{"same": [[99, 42]]}', '{"same": "nope"}', '{"same": [["a","b"]]}'):
            client = FakeClient(reply)
            report = self.memory.consolidate(client=client, model="fake")
            self.assertEqual(len(report["merged"]), 0, reply)
            self.assertEqual(len(store.get_active()), 2, reply)

    def test_model_api_failure_merges_nothing(self):
        self.seed("The user prefers pnpm over npm", "preference")
        self.seed("The user prefers yarn over npm", "preference")

        class Down:
            @property
            def chat(self):
                class C:
                    @property
                    def completions(self):
                        raise RuntimeError("down")
                return C()

        report = self.memory.consolidate(client=Down(), model="fake")
        self.assertEqual(len(report["merged"]), 0)
        self.assertEqual(len(store.get_active()), 2)

    def test_a_verdict_is_never_paid_for_twice(self):
        # A "different" verdict leaves both memories live, so the same pair
        # comes back on every future pass. Consolidation must not re-buy it.
        self.seed("The user prefers pnpm over npm", "preference")
        self.seed("The user prefers pnpm instead of npm", "preference")

        first = FakeClient('{"same": []}')
        report = self.memory.consolidate(client=first, model="fake")
        self.assertEqual(first.calls, 1, "the ambiguous pair was never asked")
        self.assertEqual(len(report["merged"]), 0)
        self.assertEqual(len(store.get_active()), 2)

        for _ in range(4):
            again = FakeClient('{"same": []}')
            self.memory.consolidate(client=again, model="fake")
            self.assertEqual(again.calls, 0, "a cached verdict was re-asked")
            self.assertEqual(len(store.get_active()), 2)

    def test_reworded_pair_is_asked_afresh(self):
        # The cache is keyed on the pair's text, so a reworded pair is a new
        # question rather than an inherited answer to the old one.
        self.seed("The user prefers pnpm over npm", "preference")
        self.seed("The user prefers pnpm instead of npm", "preference")

        first = FakeClient('{"same": []}')
        self.memory.consolidate(client=first, model="fake")
        self.assertEqual(first.calls, 1)
        self.assertEqual(len(store.get_active()), 2)

        for item in store.get_active():
            if "instead" in item.text:
                store.archive(item.id, reason="test")

        # Bypass dedupe: the point of the test is the cache key, and insert
        # would otherwise sharpen the surviving row in place rather than
        # producing a new pair to judge.
        store.insert_memory(
            "The user prefers pnpm over npm a lot", "preference", allow_duplicate=True
        )

        second = FakeClient('{"same": [1]}')
        report = self.memory.consolidate(client=second, model="fake")
        self.assertEqual(second.calls, 1, "a reworded pair inherited a stale verdict")
        self.assertEqual(len(report["merged"]), 1)

    def test_a_failed_call_is_not_cached_as_a_verdict(self):
        # A network failure must stay retryable. Caching it would permanently
        # freeze a pair the model never actually judged.
        self.seed("The user prefers pnpm over npm", "preference")
        self.seed("The user prefers pnpm instead of npm", "preference")

        class Down:
            @property
            def chat(self):
                class C:
                    @property
                    def completions(self):
                        raise RuntimeError("down")
                return C()

        self.memory.consolidate(client=Down(), model="fake")
        self.assertEqual(len(store.get_active()), 2)

        retry = FakeClient('{"same": [1]}')
        report = self.memory.consolidate(client=retry, model="fake")
        self.assertEqual(retry.calls, 1, "a failed call poisoned the cache")
        self.assertEqual(len(report["merged"]), 1)

    def test_subject_conflicts_resolve_at_write_time(self):
        # Supersede happens on insert, so the store never holds two live
        # values for one subject in the first place.
        self.memory.add("The user uses vim", "preference", subject="tool.editor")
        self.memory.add("The user uses emacs", "preference", subject="tool.editor")
        new = self.memory.add(
            "The user uses neovim", "preference", subject="tool.editor"
        )

        self.assertEqual(
            [i.text for i in store.get_by_subject("tool.editor")],
            ["The user uses neovim"],
        )
        self.assertEqual(len(store.get_active()), 1)
        self.assertTrue(new.supersedes)

    def test_dry_run_writes_nothing(self):
        self.seed("The user prefers pnpm over npm", "preference")
        self.seed("The user prefers pnpm instead of npm", "preference")
        self.seed("The user prefers yarn over npm", "preference")
        before = len(store.get_active())

        report = self.memory.consolidate(dry_run=True)

        self.assertTrue(report["dry_run"])
        self.assertEqual(len(store.get_active()), before)

    def test_consolidation_is_idempotent(self):
        self.seed("The user prefers pnpm over npm", "preference")
        self.seed("The user prefers yarn over npm", "preference")

        first = self.memory.consolidate()
        second = self.memory.consolidate()

        self.assertEqual(len(first["merged"]), 0)
        self.assertEqual(len(second["merged"]), 0)
        self.assertEqual(len(store.get_active()), 2)


# ----------------------------------------------------------------- context


class TestContext(MemoryTestCase):
    def test_empty_results_render_nothing(self):
        self.assertEqual(context.build_context([]), "")

    def test_includes_ids_and_types(self):
        item = self.seed("The user's name is Krishna", "identity")
        block = context.build_context(
            self.memory.search_detailed("name", reinforce=False)
        )
        self.assertIn(f"[{item.id}]", block)
        self.assertIn("identity", block)
        self.assertIn("Krishna", block)

    def test_budget_is_respected(self):
        self.seed_many(*[f"The user has preference number {i} about tooling" for i in range(60)])
        results = self.memory.search_detailed("tooling preferences", top_k=40, reinforce=False)
        block = context.build_context(results, budget=400)
        self.assertLessEqual(len(block), 400 + len("What you know about the user:\n"))

    def test_best_score_survives_a_tight_budget(self):
        self.seed("The user is allergic to peanuts", "identity")
        for i in range(40):
            self.seed(f"Unrelated note number {i} about compilers and build flags")

        results = self.memory.search_detailed("peanut allergy", top_k=40, reinforce=False)
        block = context.build_context(results, budget=300)
        self.assertIn("peanuts", block)

    def test_low_confidence_is_flagged(self):
        item = self.seed("The user is allergic to peanuts", "identity")
        db.connect().execute(
            "UPDATE memory SET confidence_score = 0.3 WHERE id = ?", (item.id,)
        )
        db.connect().commit()
        block = context.build_context(self.memory.search_detailed("peanuts", reinforce=False))
        self.assertIn("confidence:", block)

    def test_group_by_type_ordering(self):
        self.seed("The user shipped a release", "event")
        self.seed("The user's name is Krishna", "identity")
        entries = [{"item": i, "score": 0.5} for i in store.get_active()]
        groups = [name for name, _ in context.group_by_type(entries)]
        self.assertLess(groups.index("identity"), groups.index("event"))

    def test_system_block_has_guidance(self):
        self.seed("The user's name is Krishna", "identity")
        block, _ = self.memory.context("name")
        self.assertIn("remember()", block)


# ------------------------------------------------------------------- tools


class TestTools(MemoryTestCase):
    def test_remember_tool(self):
        out = self.memory.run_tool(
            "remember",
            {"text": "The user's name is Krishna", "memory_type": "identity", "subject": "user.name"},
        )
        self.assertIn("Memory saved", out)
        self.assertEqual(len(store.get_active()), 1)

    def test_remember_rejects_bad_type(self):
        out = self.memory.run_tool("remember", {"text": "something", "memory_type": "nope"})
        self.assertIn("Error", out)

    def test_remember_rejects_empty_text(self):
        self.assertIn("Error", self.memory.run_tool("remember", {"text": "  "}))

    def test_remember_reports_duplicate(self):
        self.memory.run_tool("remember", {"text": "The user likes tea", "memory_type": "preference"})
        out = self.memory.run_tool("remember", {"text": "The user likes tea", "memory_type": "preference"})
        self.assertIn("Already remembered", out)

    def test_remember_reports_supersede(self):
        self.memory.run_tool("remember", {"text": "User lives in London", "memory_type": "identity", "subject": "user.location"})
        out = self.memory.run_tool("remember", {"text": "User lives in Tokyo", "memory_type": "identity", "subject": "user.location"})
        self.assertIn("superseded", out)

    def test_recall_tool(self):
        self.seed("The user's name is Krishna", "identity")
        out = self.memory.run_tool("recall", {"query": "name"})
        self.assertIn("Krishna", out)
        self.assertIn("#", out)

    def test_recall_tool_miss(self):
        out = self.memory.run_tool("recall", {"query": "underwater basket weaving"})
        self.assertIn("No memories", out)

    def test_recall_requires_query(self):
        self.assertIn("Error", self.memory.run_tool("recall", {"query": ""}))

    def test_forget_tool_archive_and_delete(self):
        item = self.seed("The user likes tea", "preference")
        self.assertIn("Archived", self.memory.run_tool("forget", {"memory_id": item.id}))
        self.assertIn("Deleted", self.memory.run_tool("forget", {"memory_id": item.id, "mode": "delete"}))

    def test_forget_unknown_id(self):
        self.assertIn("Error", self.memory.run_tool("forget", {"memory_id": 9999}))

    def test_memories_tool(self):
        self.seed("The user's name is Krishna", "identity")
        out = self.memory.run_tool("memories", {})
        self.assertIn("IDENTITY", out)
        self.assertIn("Krishna", out)

    def test_memories_tool_empty(self):
        self.assertIn("empty", self.memory.run_tool("memories", {}))

    def test_unknown_tool(self):
        self.assertIn("Error", self.memory.run_tool("nope", {}))

    def test_all_tools_have_valid_schemas(self):
        for tool in Memory.tools():
            self.assertEqual(tool["type"], "function")
            function = tool["function"]
            self.assertIn("name", function)
            self.assertIn("description", function)
            self.assertEqual(function["parameters"]["type"], "object")
            for required in function["parameters"].get("required", []):
                self.assertIn(required, function["parameters"]["properties"])


# --------------------------------------------------------------- migration


class TestMigration(unittest.TestCase):
    """A v1 database must gain the v2 columns without losing rows."""

    def test_v1_schema_is_upgraded_in_place(self):
        import sqlite3

        legacy_dir = tempfile.mkdtemp(prefix="retain-legacy-")
        legacy_path = os.path.join(legacy_dir, "legacy.db")

        try:
            conn = sqlite3.connect(legacy_path)
            conn.row_factory = sqlite3.Row
            # Exactly the v1 schema from the original store.py.
            conn.execute(
                """
                CREATE TABLE memory (
                    id INTEGER PRIMARY KEY,
                    text TEXT,
                    memory_type TEXT,
                    confidence_score REAL,
                    decay_rate REAL,
                    created_at TEXT,
                    last_accessed_at TEXT,
                    is_archived INTEGER
                )
                """
            )
            stamp = to_iso(utcnow())
            conn.execute(
                "INSERT INTO memory (text, memory_type, confidence_score, decay_rate,"
                " created_at, last_accessed_at, is_archived) VALUES (?,?,?,?,?,?,?)",
                ("User's name is Krishna", "fact", 1.0, 0.2, stamp, stamp, 0),
            )
            conn.commit()
            conn.close()

            previous = os.environ["RETAIN_DB_PATH"]
            os.environ["RETAIN_DB_PATH"] = legacy_path
            db.reset_connection()
            try:
                conn = db.connect()
                self.assertEqual(db.schema_version(conn), db.SCHEMA_VERSION)

                columns = {r["name"] for r in conn.execute("PRAGMA table_info(memory)")}
                for column in ("norm_hash", "scope", "subject", "importance",
                               "access_count", "expires_at", "superseded_by"):
                    self.assertIn(column, columns)

                items = store.get_active()
                self.assertEqual(len(items), 1)
                self.assertEqual(items[0].text, "User's name is Krishna")

                # The legacy row must also be searchable, and the index must
                # be writable afterwards.
                self.assertEqual(len(retrieve.retrieve("name", top_k=5)), 1)
                decay.reinforce(items[0])
                self.assertGreater(store.get(items[0].id).access_count, 0)
            finally:
                os.environ["RETAIN_DB_PATH"] = previous
                db.reset_connection()
        finally:
            shutil.rmtree(legacy_dir, ignore_errors=True)

    def test_rows_inserted_behind_the_store_are_indexed(self):
        """Regression: a fresh FTS index must pick up existing rows.

        `SELECT count(*)` on an external-content FTS table reads the content
        table, so a count-based "is the index stale?" check always says no.
        The index then silently stays empty and the first write fails with
        "database disk image is malformed".
        """
        directory = tempfile.mkdtemp(prefix="retain-index-")
        path = os.path.join(directory, "index.db")

        try:
            previous = os.environ["RETAIN_DB_PATH"]
            os.environ["RETAIN_DB_PATH"] = path
            db.reset_connection()
            try:
                conn = db.connect()
                stamp = now_iso()
                # Straight to SQL, bypassing the store and its triggers.
                conn.execute(
                    "INSERT INTO memory (text, memory_type, confidence_score,"
                    " decay_rate, created_at, last_accessed_at, is_archived)"
                    " VALUES (?,?,?,?,?,?,0)",
                    ("The user collects vinyl records", "preference", 1.0, 0.01, stamp, stamp),
                )
                conn.commit()

                # Reconnecting runs migration, which must notice the gap.
                db.reset_connection()
                conn = db.connect()

                self.assertTrue(db.fts_index_is_sane(conn))
                hits = conn.execute(
                    "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?",
                    ("vinyl*",),
                ).fetchall()
                self.assertEqual(len(hits), 1)

                # And the triggers must be able to maintain it.
                conn.execute("UPDATE memory SET confidence_score = 0.5")
                conn.commit()
                self.assertTrue(db.fts_index_is_sane(conn))
            finally:
                os.environ["RETAIN_DB_PATH"] = previous
                db.reset_connection()
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
