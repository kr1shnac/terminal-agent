"""Retrieval-quality benchmark for the memory system.

Builds a realistic store (a persona plus a long tail of unrelated facts) and
measures whether the right memory comes back for realistic questions.

    python bench_memory.py            # summary
    python bench_memory.py --verbose  # per-query breakdown
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_TMP = tempfile.mkdtemp(prefix="retain-bench-")
os.environ["RETAIN_DB_PATH"] = os.path.join(_TMP, "bench.db")

from memory import Memory, consolidate, db, retrieve, store  # noqa: E402
from memory.clock import to_iso, utcnow  # noqa: E402

# --------------------------------------------------------------- the store
#
# A persona, plus ~240 distractor facts that share vocabulary with the
# questions. The distractors matter: a retriever that only looks good against
# five hand-picked memories is not being measured.

PERSONA = [
    ("The user's name is Krishna Raman.", "identity", "user.name"),
    ("The user's timezone is Asia/Kolkata.", "identity", "user.timezone"),
    ("The user lives in Berlin, Germany.", "identity", "user.location"),
    ("The user is a backend engineer.", "identity", "user.role"),
    ("The user prefers pnpm over npm and yarn.", "preference", "tool.package_manager"),
    ("The user's editor is Neovim with lazy.nvim.", "preference", "tool.editor"),
    ("The user prefers dark mode in every application.", "preference", "ui.theme"),
    ("The user prefers ripgrep over grep for searching.", "preference", "tool.search"),
    ("The user dislikes writing unit tests before the design is settled.", "preference", None),
    ("The user has asked to always run the test suite before reporting done.", "instruction", "workflow.verify"),
    ("The user has asked to never commit directly to the main branch.", "instruction", "workflow.branch"),
    ("The user has asked to never force-push.", "instruction", "workflow.git"),
    ("The user's primary goal is shipping the payments rewrite this quarter.", "goal", "goal.primary"),
    ("The user is working on a Rust telemetry agent called retaind.", "project", "project.retaind"),
    ("The retaind project stores metrics in ClickHouse.", "project", "project.retaind"),
    ("The user prefers tabs over spaces.", "preference", "style.indent"),
    ("The user prefers descriptive commit messages over conventional commits.", "preference", "style.commits"),
    ("The user is allergic to peanuts.", "identity", "health.allergy"),
    ("The user's dog is called Biscuit.", "identity", "pet.dog"),
    ("The user prefers fzf over find.", "preference", "tool.search"),
]

DISTRACTORS = [
    "The build server runs on Ubuntu 22.04 with 16GB of RAM.",
    "The frontend is written in TypeScript with React 18.",
    "The database migrations live in the db/migrations folder.",
    "The team standup happens at 09:30 CET every weekday.",
    "The staging environment is torn down nightly at 03:00 UTC.",
    "The CI pipeline runs lint, then unit tests, then integration tests.",
    "The documentation is generated with MkDocs from the docs folder.",
    "The API version is pinned at v2 and is backwards compatible.",
    "The analytics dashboard loads slowly when the date range exceeds 90 days.",
    "The user asked about caching strategies for the search endpoint.",
    "The user reported a bug where the sidebar collapses on narrow viewports.",
    "The user wanted a dark theme toggle added to the settings page.",
    "The user reviewed the pull request for the auth refactor.",
    "The user asked why the test suite takes eleven minutes to finish.",
    "The user mentioned the payment provider rejected a retry.",
    "The user compared Postgres and SQLite for the local cache.",
    "The user asked about reading Docker logs from a stopped container.",
    "The user wanted the CLI to print colours when stdout is a TTY.",
    "The user asked how to profile a slow Python function.",
    "The user tried Python 3.13 and hit a removed stdlib module.",
    "The user asked whether to use threads or processes for IO bound work.",
    "The user wanted structured logging with request ids.",
    "The user asked about rotating secrets without downtime.",
    "The user read a paper about consensus protocols.",
    "The user evaluated vector databases for semantic search.",
    "The user asked about embedding models and chunking strategies.",
    "The user wanted a retry budget instead of fixed retries.",
    "The user asked how to detect memory leaks in a long running service.",
    "The user compared gRPC and REST for internal services.",
    "The user asked about idempotency keys for the checkout endpoint.",
    "The user asked why the linter flags an unused import in a test file.",
    "The user wanted fixtures scoped per test module.",
    "The user asked about snapshot testing the rendered component tree.",
    "The user asked for a health check endpoint that checks dependencies.",
    "The user wanted structured error codes in the API response.",
    "The user asked about rate limiting per API key.",
    "The user wanted the metrics renamed to follow OpenTelemetry conventions.",
    "The user asked about cardinality explosion in label sets.",
    "The user wanted histograms instead of summaries for latency.",
    "The user asked about tail latency and percentiles.",
    "The user wanted trace propagation across the queue boundary.",
    "The user asked about sampling strategies in a tracer.",
    "The user wanted a flame graph in the profiling dashboard.",
    "The user asked about pprof output in a container.",
    "The user wanted the schema registry to validate at write time.",
    "The user asked about consumer lag and rebalancing.",
    "The user wanted exactly-once semantics discussion resolved.",
    "The user asked about dead letter queues and replay.",
    "The user wanted the consumer to be idempotent by key.",
    "The user asked about partition key choice for ordering.",
    "The user wanted backpressure in the ingestion path.",
    "The user asked about batching and micro-batching tradeoffs.",
    "The user wanted a bulkhead pattern around the third party API.",
    "The user asked about circuit breaker hysteresis.",
    "The user wanted timeouts lowered to fail fast.",
    "The user asked about connection pool sizing.",
    "The user wanted prepared statements to avoid query plan churn.",
    "The user asked about N+1 queries in the ORM layer.",
    "The user wanted an index on the foreign key column.",
    "The user asked about vacuum and bloat in Postgres.",
    "The user wanted logical replication between regions.",
    "The user asked about read replicas and staleness tradeoffs.",
    "The user wanted columnar storage for the event table.",
    "The user asked about compression codecs for time series.",
    "The user wanted retention policies per metric name.",
    "The user asked about downsampling rollups.",
    "The user wanted a recording rule for the error rate.",
    "The user asked about alert fatigue and routing.",
    "The user wanted a silence window for deploys.",
    "The user asked about SLO error budgets.",
    "The user wanted burn rate alerts for the checkout SLO.",
    "The user asked about the on-call rotation handover.",
    "The user wanted a runbook for the top five alerts.",
    "The user asked about paging only for actionable pages.",
    "The user wanted the deploy pipeline gated on the smoke test.",
    "The user asked about blue green versus canary.",
    "The user wanted feature flags with a cleanup date.",
    "The user asked about database migration during a rolling deploy.",
    "The user wanted the old column dropped after backfill.",
    "The user asked about zero downtime schema changes.",
    "The user wanted a dual write shim removed once verified.",
    "The user asked about the read path during a backfill.",
    "The user wanted a kill switch for the new writer.",
    "The user asked about verifying the backfill row counts.",
    "The user wanted the migration idempotent on re-run.",
    "The user asked about the transaction log growth.",
    "The user wanted compaction scheduled off peak.",
    "The user asked about snapshot restore time.",
    "The user wanted point-in-time recovery tested.",
    "The user asked about cross region failover.",
    "The user wanted the runbook linked from the alert.",
    "The user asked about an incident review blameless format.",
    "The user wanted the timeline written from the logs.",
    "The user asked about paging the secondary on call.",
    "The user wanted the error budget policy documented.",
    "The user asked about capacity planning for peak traffic.",
    "The user wanted load tests at 3x current peak.",
    "The user asked about the autoscaling policy on CPU.",
    "The user wanted a warm pool to absorb bursts.",
    "The user asked about the pod disruption budget.",
    "The user wanted the readiness probe to check the cache.",
    "The user asked about graceful shutdown draining in flight requests.",
    "The user wanted SIGTERM handling in the entrypoint.",
    "The user asked about the sidecar proxy timeouts.",
    "The user wanted mTLS between services.",
    "The user asked about certificate rotation.",
    "The user wanted secrets mounted from a file not env.",
    "The user asked about least privilege IAM policies.",
    "The user wanted audit logs shipped off the host.",
    "The user asked about encrypting data at rest.",
    "The user wanted PII redacted from logs.",
    "The user asked about GDPR deletion requests.",
    "The user wanted a data retention matrix.",
    "The user asked about the right to be forgotten.",
    "The user wanted export in a portable format.",
    "The user asked about consent tracking.",
    "The user wanted the privacy policy updated.",
    "The user asked about the cookie banner.",
    "The user wanted analytics with IP anonymisation.",
    "The user asked about the difference between sampling and censoring.",
    "The user wanted the histogram buckets widened.",
    "The user asked about the leaky bucket algorithm.",
    "The user wanted a token bucket for the outbound limiter.",
    "The user asked about fairness between tenants.",
    "The user wanted per tenant quotas.",
    "The user asked about noisy neighbour on shared clusters.",
    "The user wanted the noisy tenant detection.",
    "The user asked about the difference between throughput and goodput.",
    "The user wanted the SLA error definition tightened.",
    "The user asked about measurement error in the metric.",
    "The user wanted the dashboard to show a confidence band.",
    "The user asked about the difference between median and p99.",
    "The user wanted the aggregation window aligned.",
    "The user asked about counter resets and rate computation.",
    "The user wanted a gauge for in flight requests.",
    "The user asked about the cost of high cardinality labels.",
    "The user wanted the metric names reviewed for consistency.",
    "The user asked about the scrape interval tradeoff.",
    "The user wanted remote write batching tuned.",
    "The user asked about the WAL segment size.",
    "The user wanted the read path to use a head block.",
    "The user asked about the query queue sharding.",
    "The user wanted the ingester scaled horizontally.",
    "The user asked about deduplication by tenant and time.",
    "The user wanted the out of order window widened.",
    "The user asked about clock skew between agents.",
    "The user wanted the timestamps normalised to UTC.",
    "The user asked about leap seconds in the parser.",
    "The user wanted the timezone handling centralized.",
    "The user asked about daylight saving transitions.",
    "The user wanted the dashboard in the user's local time.",
    "The user asked about date formatting in the export.",
    "The user wanted ISO 8601 everywhere.",
    "The user asked about relative time labels.",
    "The user wanted a human readable summary of the last 24 hours.",
    "The user asked about anomaly detection thresholds.",
    "The user wanted seasonal decomposition of the daily series.",
    "The user asked about the forecast confidence interval.",
    "The user wanted the alert to fire on the forecast, not the raw value.",
    "The user asked about a seasonality of 7 days.",
    "The user wanted the model retrained nightly.",
    "The user asked about feature drift detection.",
    "The user wanted the training data window to be explicit.",
    "The user asked about label leakage in the features.",
    "The user wanted the evaluation set frozen.",
    "The user asked about backtesting with a time based split.",
    "The user wanted the metric to be precision at k.",
    "The user asked about recall of the relevant docs.",
    "The user wanted hybrid retrieval evaluated separately.",
    "The user asked about reranking with a cross encoder.",
    "The user wanted the context budget measured in tokens.",
    "The user asked about truncating a long memory.",
    "The user wanted the summary generated once at write time.",
    "The user asked about summarisation drift over months.",
    "The user wanted a fact type separate from a summary type.",
    "The user asked about the difference between an event and a fact.",
    "The user wanted decay applied per type.",
    "The user asked about a half life of 21 days for events.",
    "The user wanted reinforcement to slow the decay rate.",
    "The user asked about archiving below a threshold.",
    "The user wanted an idle limit before release.",
    "The user asked about the retention floor from importance.",
    "The user wanted decay to be idempotent.",
    "The user asked about the compounding decay bug.",
    "The user wanted the confidence tied to last accessed time.",
    "The user asked about restoring an archived memory.",
    "The user wanted a hard delete path.",
    "The user asked about the access log for auditing.",
    "The user wanted the score stored with each access.",
    "The user asked about which memories earn their place.",
    "The user wanted a review of unused memories.",
    "The user asked about superseding a corrected value.",
    "The user wanted the old row archived with a reason.",
    "The user asked about the subject slot semantics.",
    "The user wanted a subject for singular replaceable values.",
    "The user asked about merging paraphrases.",
    "The user wanted contradictions left alone.",
    "The user asked about adjudicating with a model.",
    "The user wanted verdicts cached to avoid re buying them.",
    "The user asked about the cost of consolidation.",
    "The user wanted a dry run mode.",
    "The user asked about pruning low value rows.",
    "The user wanted the empty row cleaned up.",
    "The user asked about the schema version stamp.",
    "The user wanted the migration to be additive.",
    "The user asked about the FTS external content table.",
    "The user wanted triggers to keep the index in sync.",
    "The user asked about the integrity check command.",
    "The user wanted the index rebuilt when it disagrees.",
    "The user asked about the WAL journal mode.",
    "The user wanted the connection cached process wide.",
    "The user asked about thread safety on one connection.",
    "The user wanted the busy timeout set.",
    "The user asked about the partial index on live rows.",
    "The user wanted an index on the normalised hash.",
    "The user asked about the subject index.",
    "The user wanted the expiry index.",
    "The user asked about the type index.",
    "The user wanted a covering index for the hot query.",
    "The user asked about explain query plan.",
    "The user wanted the retrieval latency measured.",
    "The user asked about the candidate cap.",
    "The user wanted the pool to scale with the store.",
    "The user asked about BM25 versus cosine.",
    "The user wanted reciprocal rank fusion.",
    "The user asked about the RRF smoothing constant.",
    "The user wanted prefix matching for inflections.",
    "The user asked about the AND to OR fallback.",
    "The user wanted the porter stemmer.",
    "The user asked about unicode tokenization.",
    "The user wanted the query escaped against the FTS grammar.",
    "The user asked about an unbalanced quote in a query.",
    "The user wanted the search to never raise.",
    "The user asked about a minimum score gate.",
    "The user wanted salience to break ties.",
    "The user asked about lexical sharpening.",
    "The user wanted MMR diversification.",
    "The user asked about near duplicate prompts.",
    "The user wanted the context budget spent in score order.",
    "The user asked about grouping by type in the prompt.",
    "The user wanted ids included so forget works.",
    "The user asked about the minimum importance filter.",
    "The user wanted low confidence marked in the block.",
    "The user asked about hedging on a stale memory.",
]

# (question, index into PERSONA of the expected memory, acceptable alternates)
QUERIES = [
    ("what is my name", 0, []),
    ("do you know who I am", 0, []),
    ("which timezone am I in", 1, []),
    ("where do I live", 2, []),
    ("what do I do for work", 3, []),
    ("what is my job", 3, []),
    ("which package manager do I use", 4, []),
    ("what editor do I use", 5, []),
    ("do I like dark mode or light mode", 6, []),
    ("what do I search with", 7, []),
    ("which search tool do I prefer", 7, [19]),
    ("do I like unit tests", 8, []),
    ("what should you do before telling me you are done", 9, []),
    ("should you commit to main", 10, []),
    ("what are the git rules", 10, [11]),
    ("what is my main goal this quarter", 12, []),
    ("what am I building right now", 13, []),
    ("what database does retaind use", 14, []),
    ("spaces or tabs", 15, []),
    ("how should I write commit messages", 16, []),
    ("do I have any allergies", 17, []),
    ("what is my dog called", 18, []),
    ("fzf or find", 19, [7]),
]


def build():
    memory = Memory(auto_maintain=False)
    for text, mtype, subject in PERSONA:
        memory.add(text, mtype, subject=subject)
    for text in DISTRACTORS:
        memory.add(text, "fact")
    return memory


def age_rows(memory, days_by_type):
    """Backdate rows so recency and confidence are not all identical."""
    conn = db.connect()
    for item in store.get_active():
        days = days_by_type.get(item.memory_type, 5)
        stamp = to_iso(utcnow() - timedelta(days=days))
        conn.execute(
            "UPDATE memory SET created_at = ?, last_accessed_at = ?, "
            "access_count = ?, confidence_score = ? WHERE id = ?",
            (stamp, stamp, 3, 0.8, item.id),
        )
    conn.commit()


def measure(memory, top_k, verbose):
    by_text = {text: index for index, (text, _t, _s) in enumerate(PERSONA)}
    hits_at_1 = 0
    hits_at_k = 0
    misses = []

    for question, expected, alternates in QUERIES:
        results = memory.search_detailed(question, top_k=top_k, reinforce=False)
        ids = [entry["item"].id for entry in results]
        expected_id = None
        for cand in [expected] + alternates:
            text = PERSONA[cand][0]
            match = [i.id for i in store.get_active() if i.text == text]
            if match:
                expected_id = match[0]
                break

        if not expected_id:
            continue

        rank = ids.index(expected_id) + 1 if expected_id in ids else 0
        if rank == 1:
            hits_at_1 += 1
        if rank:
            hits_at_k += 1
        else:
            misses.append((question, PERSONA[expected][0], [results[i]["item"].text for i in range(len(results))]))

        if verbose:
            mark = "OK " if rank == 1 else ("~  " if rank else "MISS")
            print(f"  {mark} rank={rank or '-'}  {question}")
            if rank != 1:
                for i, entry in enumerate(results[:3], 1):
                    print(f"        {i}. {entry['score']:.3f} {entry['item'].text[:70]}")

    total = len(QUERIES)
    return {
        "total": total,
        "hit@1": hits_at_1 / total,
        f"hit@{top_k}": hits_at_k / total,
        "misses": misses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    memory = build()
    age_rows(memory, {"identity": 30, "preference": 12, "instruction": 20, "goal": 40, "project": 25})
    live = len(store.get_active())
    print(f"store: {live} live memories ({len(PERSONA)} persona, {len(DISTRACTORS)} distractors)")
    print(f"index: FTS5 available = {retrieve.fts_available()}\n")

    t0 = time.perf_counter()
    result = measure(memory, args.top_k, args.verbose)
    elapsed = time.perf_counter() - t0

    print(f"\nquestions        : {result['total']}")
    print(f"hit@1            : {result['hit@1']:.1%}")
    print(f"hit@{args.top_k:<12}: {result[f'hit@{args.top_k}']:.1%}")
    print(f"query time       : {elapsed / result['total'] * 1000:.1f} ms/query")

    if result["misses"]:
        print(f"\n{len(result['misses'])} miss(es):")
        for question, want, got in result["misses"]:
            print(f"  Q: {question}\n     want: {want[:70]}")
            for i, text in enumerate(got, 1):
                print(f"     got{i}: {text[:70]}")

    t0 = time.perf_counter()
    report = memory.consolidate()
    print(
        f"\nconsolidate      : {consolidate.summarize(report)} "
        f"({(time.perf_counter() - t0) * 1000:.0f} ms)"
    )
    print(f"live after       : {len(store.get_active())}")

    db.reset_connection()
    shutil.rmtree(_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
