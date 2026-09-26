"""Diagnose why the right memory loses to a distractor."""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, r"C:\all projects\clone terminal agent\terminal-agent")
_TMP = tempfile.mkdtemp(prefix="retain-diag-")
os.environ["RETAIN_DB_PATH"] = os.path.join(_TMP, "d.db")

from memory import db, store, retrieve  # noqa: E402
from memory.text import fts_query, tokenize  # noqa: E402

conn = db.connect()

rows = [
    "The user prefers tabs over spaces.",
    "The user asked whether to use threads or processes for IO bound work.",
    "The user is a backend engineer.",
    "The user's editor is Neovim with lazy.nvim.",
    "The user prefers ripgrep over grep for searching.",
]
for r in rows:
    store.insert_memory(r, "preference")

print("=== how FTS5 actually stores the terms ===")
for phrase in ("tabs over spaces", "or", "spaces", "tabs"):
    got = conn.execute(
        "SELECT memory_fts FROM memory_fts WHERE memory_fts MATCH ?", (f'"{phrase}"',)
    ).fetchall()
    print(f"  {phrase!r:22} -> {[g[0].split(':')[0] for g in got]}")

print("\n=== what the query builder emits vs what matches ===")
for q in ["spaces or tabs", "which package manager do I use", "what is my job"]:
    match = fts_query(q, mode="AND")
    or_match = fts_query(q, mode="OR")
    print(f"  q={q!r}")
    print(f"    tokens  = {tokenize(q)}")
    print(f"    AND     = {match}")
    if match:
        hits = conn.execute(
            "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH ?", (match,)
        ).fetchone()[0]
        print(f"    AND hits= {hits}")
    print(f"    OR      = {or_match}")
    if or_match:
        hits = conn.execute(
            "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH ?", (or_match,)
        ).fetchone()[0]
        print(f"    OR hits = {hits}")

print("\n=== porter stem of each query term vs indexed stem ===")
for term in ("tabs", "spaces", "or", "use", "editor", "manager"):
    hits = conn.execute(
        "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH ?", (f'"{term}"*',)
    ).fetchone()[0]
    print(f'  "{term}"*  -> {hits} rows')

print("\n=== RRF dynamic range on a real query ===")
for r in rows:
    store.insert_memory(r, "preference", allow_duplicate=True)
pool = store.get_active(limit=retrieve.MAX_CANDIDATES)
q = "which package manager do I use"
fts_rank = retrieve.fts_search(q)
tfidf_rank = retrieve.tfidf_search(q, pool)
fused = retrieve.reciprocal_rank_fusion([fts_rank, tfidf_rank])
best = max(fused.values())
by_id = {i.id: i for i in pool}
name = {i.id: i.text[:45] for i in pool}
for mid, raw in sorted(fused.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  rrf={raw:.5f} lex={raw / best:.3f}  {name.get(mid)}")

print("\n=== what MIN_TOKEN_LEN / stopwords do to 'or' ===")
print("  'or' in STOPWORDS   :", "or" in __import__("memory.text", fromlist=["STOPWORDS"]).STOPWORDS)
print("  tokenize('or')      :", tokenize("or"))
print("  tokenize('spaces or tabs'):", tokenize("spaces or tabs"))

db.reset_connection()
shutil.rmtree(_TMP, ignore_errors=True)
