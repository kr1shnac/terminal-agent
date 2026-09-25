"""End-to-end walkthrough of the memory system. No API key required.

    python demo.py

Each section prints what the memory layer actually did, so you can see the
behaviour rather than take it on faith. Uses a throwaway database.
"""

import json
import os
import re
import shutil
import tempfile
import types
from datetime import timedelta

_DEMO_DIR = tempfile.mkdtemp(prefix="retain-demo-")
os.environ["RETAIN_DB_PATH"] = os.path.join(_DEMO_DIR, "demo.db")

from memory import Memory, consolidate, decay, store  # noqa: E402
from memory.clock import to_iso, utcnow  # noqa: E402

LINE = "-" * 72


def header(title):
    print(f"\n{LINE}\n{title}\n{LINE}")


def show_items(items, label="live"):
    if not items:
        print(f"  ({label}: none)")
    for item in items:
        print(
            f"  #{item.id:<3} {item.memory_type:<11} conf={item.confidence:.2f} "
            f"imp={item.importance:.2f} used={item.access_count:<2} {item.text}"
        )


# ---------------------------------------------------------------------------
# A stub model, standing in for the cheap LLM used by extraction and by
# consolidation's ambiguity check.
# ---------------------------------------------------------------------------


def stub_client(reply_for):
    client = types.SimpleNamespace()
    client.calls = 0

    class Completions:
        def create(self, **kwargs):
            client.calls += 1
            content = kwargs["messages"][-1]["content"]
            body = reply_for(content)
            message = types.SimpleNamespace(content=body)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    client.chat = types.SimpleNamespace(completions=Completions())
    return client


def main():
    memory = Memory(session_id="demo")

    # ---------------------------------------------------------------- 1
    header("1. CAPTURE - what the user says becomes memory, with no tool call")
    print("Each line below is a raw user message. Extraction runs on the")
    print("user's own words; the model never has to remember to call remember().\n")

    messages = [
        "my name is Krishna and I use pnpm",
        "always run the tests before saying a task is done",
        "I am trying to learn Rust this year",
        "I am working on a terminal agent called Retain",
        "btw I am a backend developer",
    ]
    for message in messages:
        learned = memory.observe(message, use_llm=False)
        for item in learned:
            print(f"  {message!r}")
            print(f"      -> #{item.id} {item.memory_type:<11} {item.text}")

    # ---------------------------------------------------------------- 2
    header("2. DEDUPE - the same fact stated again does not become a second row")
    before = len(store.get_active())
    for repeat in ("my name is Krishna", "my name is Krishna."):
        learned = memory.observe(repeat, use_llm=False)
        print(f"  {repeat!r} -> {'stored ' + str(len(learned)) + ' new' if learned else 'nothing new'}")
    print(f"  live memories: {before} -> {len(store.get_active())}")
    show_items(store.get_by_subject("user.name"))

    # ---------------------------------------------------------------- 3
    header("3. REFINE and CORRECT - sharpening a memory, and replacing a value")
    print("  a) the user gives a fuller version of a name already stored:")
    memory.add("The user's name is Krishna Chinni", "identity", subject="user.name")
    named = store.get_by_subject("user.name")[0]
    print(f"     kept the same row #{named.id}, sharpened the text:")
    print(f"     -> {named.text}")

    print("\n  b) the user states a different value for the same subject:")
    before = store.get_by_subject("user.name")[0]
    memory.add("The user's name is Arjun", "identity", subject="user.name")
    print(f"     before: #{before.id} {before.text}")
    for item in store.get_by_subject("user.name"):
        print(f"     live   -> #{item.id} {item.text}")
    stale = store.get(before.id)
    print(f"     old #{stale.id}: archived={bool(stale.is_archived)} "
          f"reason={stale.archived_reason} superseded_by={stale.superseded_by}")
    print("     a superseded memory is filtered out of recall, so the prompt")
    print("     can never show two different names")

    # ---------------------------------------------------------------- 4
    header("4. RECALL - hybrid BM25 + TF-IDF, reranked by salience")
    print("  Scores need not run in order: MMR re-ranks to avoid handing the")
    print("  model two near-duplicate memories.\n")
    for question in (
        "what is my name?",
        "which package manager should I use?",
        "peanut allergy",
    ):
        results = memory.search_detailed(question, top_k=3)
        print(f"\n  {question!r}")
        if not results:
            print("      (nothing relevant)")
        for entry in results:
            item = entry["item"]
            signals = entry["signals"]
            print(
                f"      {entry['score']:.3f}  #{item.id:<3} {item.memory_type:<11} "
                f"lex={signals['lexical']:.2f} sal={signals['salience']:.2f}  {item.text[:52]}"
            )

    # ---------------------------------------------------------------- 5
    header("5. NOISE GATE - an unrelated question recalls nothing")
    for question in ("what is the airspeed velocity of a swallow?", "quantum chromodynamics"):
        results = memory.search_detailed(question, top_k=3)
        print(f"  {question!r} -> {len(results)} memories")

    # ---------------------------------------------------------------- 6
    header("6. FORGETTING - confidence decays, use restores it")
    print("Half-lives are per type, so an event fades and an identity does not.\n")
    event = memory.add("The user fixed a typo in the parser", "event")
    identity = memory.add("The user is allergic to peanuts", "identity")

    def age(item, days):
        stamp = to_iso(utcnow() - timedelta(days=days))
        conn = store._conn()
        conn.execute(
            "UPDATE memory SET last_accessed_at = ?, created_at = ? WHERE id = ?",
            (stamp, stamp, item.id),
        )
        conn.commit()

    print(f"  {'days unused':<13}{'event':<22}{'identity'}")
    for days in (0, 14, 30, 60, 120, 300):
        age(event, days)
        age(identity, days)
        decay.apply_decay()
        e = store.get(event.id)
        i = store.get(identity.id)
        print(
            f"  {days:<13}{e.confidence:<22.2f}{i.confidence:.2f}"
            f"{'   <- event archived' if e.is_archived else ''}"
        )

    print("\n  reinforce the event, and it comes back:")
    decay.reinforce(store.get(event.id))
    revived = store.get(event.id)
    print(f"    confidence {revived.confidence:.2f}, accesses {revived.access_count}")

    # ---------------------------------------------------------------- 7
    header("7. IDEMPOTENT DECAY - repeated sweeps must not compound")
    stable = memory.add("The user prefers ripgrep over grep", "preference")
    age(stable, 200)
    seen = []
    for _ in range(5):
        decay.apply_decay()
        seen.append(round(store.get(stable.id).confidence, 6))
    print(f"  5 sweeps -> {seen}")
    print(f"  stable: {len(set(seen)) == 1}")

    # ---------------------------------------------------------------- 8
    header("8. CONTEXT BUDGET - what actually reaches the prompt")
    topics = ["caching", "linting", "typing", "logging", "packaging", "release",
              "testing", "profiling", "refactors", "schemas", "timeouts", "retries"]
    for index in range(36):
        memory.add(
            f"Build note {index}: the {topics[index % len(topics)]} step in the "
            f"release pipeline was reworked in week {index + 1}",
            "event",
        )
    block, results = memory.context("what is my name?", budget=600)
    print(f"  {len(store.get_active())} memories exist; the question retrieved "
          f"{len(results)}, rendered within 600 chars:\n")
    for line in block.splitlines():
        print(f"  {line}")
    print("\n  the same question on a narrow budget, to show the trim:")
    narrow, _ = memory.context("what is my name?", budget=120)
    for line in narrow.splitlines():
        print(f"  {line}")

    # ---------------------------------------------------------------- 9
    header("9. CONSOLIDATION - paraphrase vs contradiction")
    print("  A clean store, so the numbers mean something. Three statements")
    print("  about package managers: 'instead of' and 'over' should fold into")
    print("  one fact, while 'prefers yarn' must survive as its own fact.\n")

    from memory.text import similarity, tokenize

    # Memory instances share one store, so clear it to keep these counts honest.
    store.forget_all()

    for text in (
        "The user prefers pnpm over npm",
        "The user prefers pnpm instead of npm",
        "The user prefers yarn over npm",
    ):
        memory.add(text, "preference")

    print("  stored all three:")
    for item in store.get_by_type("preference"):
        print(f"    #{item.id:<3} {item.text}")

    print("\n  pairwise similarity (auto-merge at 0.88, ask the model at 0.58):")
    prefs = store.get_by_type("preference")
    for i in range(len(prefs)):
        for j in range(i + 1, len(prefs)):
            score = similarity(tokenize(prefs[i].text), tokenize(prefs[j].text))
            verdict = (
                "MERGE" if score >= 0.88
                else "ask" if score >= 0.58
                else "leave alone"
            )
            print(f"    {score:.3f}  {verdict:<11} {prefs[i].text!r} <-> {prefs[j].text!r}")

    print("\n  a) without a model, nothing is guessed:")
    report = memory.consolidate()
    print(f"     {consolidate.summarize(report)}")

    # A stub that judges the way a competent model would: same fact only when
    # the package manager actually agrees.
    def judge(prompt):
        statements = []
        for line in prompt.splitlines():
            found = re.match(r"^(?:\d+\.\s*)?([AB]):\s*(.*)$", line.strip())
            if found:
                statements.append(found.group(2).strip())
        same = []
        for number, block in enumerate(
            [statements[i: i + 2] for i in range(0, len(statements) - 1, 2)], 1
        ):
            managers = [
                {w for w in ("pnpm", "yarn", "npm") if w in text} for text in block
            ]
            if len(managers) == 2 and managers[0] == managers[1]:
                same.append(number)
        return json.dumps({"same": same})

    print("\n  b) with a model adjudicating the ambiguous pair:")
    model = stub_client(judge)
    report = memory.consolidate(client=model, model="stub")
    print(f"     {consolidate.summarize(report)}  (model asked {model.calls}x)")
    for entry in report["merged"]:
        print(f"       merged #{entry['merged_id']} into #{entry['kept_id']}:")
        print(f"         was: {entry['merged_text']}")
        print(f"         now: {entry['kept_text']}")

    print("\n  final live preferences (the yarn/pnpm contradiction is preserved):")
    for item in store.get_by_type("preference"):
        print(f"    #{item.id:<3} {item.text}")

    print("\n  c) verdicts are cached, so the same question is never re-bought:")
    for attempt in range(2, 5):
        again = stub_client(judge)
        memory.consolidate(client=again, model="stub")
        print(f"     pass {attempt}: model asked {again.calls}x")

    print("\n  d) a contradiction never gets merged even when asked directly:")
    hostile = stub_client(lambda _: '{"same": [1, 2]}')
    store.forget_all()
    memory.add("The user prefers pnpm over npm", "preference")
    memory.add("The user prefers yarn over npm", "preference")
    report = memory.consolidate(client=hostile, model="stub")
    print(f"     {consolidate.summarize(report)}")
    print(f"     the model called both SAME, and it changed nothing: "
          f"{len(store.get_by_type('preference'))} live")
    print("     a contradiction never reaches the model - it scores below the")
    print("     ask floor, so the budget is never spent second-guessing it")

    # ---------------------------------------------------------------- 10
    header("10. STATS")
    stats = memory.stats()
    print(f"  live={stats['live']} archived={stats['archived']} "
          f"avg_confidence={stats['avg_confidence']} recalls={stats['total_uses']}")
    print(f"  by type: {stats['by_type']}")
    print(f"  most used:")
    for row in stats["most_used"]:
        print(f"    #{row['id']} {row['access_count']}x  {row['text'][:56]}")

    print(f"\n  demo database: {os.environ['RETAIN_DB_PATH']}")
    shutil.rmtree(_DEMO_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
