import os, tempfile
os.environ["RETAIN_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "x.db")
from memory.extract import extract, extract_with_heuristics

for msg in ["my name is Krishna and I use pnpm",
            "I use pnpm and yarn",
            "I prefer dark mode and I am a backend developer"]:
    print(f"\n{msg!r}")
    print("  heuristics:")
    for c in extract_with_heuristics(msg):
        print(f"    {c['memory_type']:<11} subject={c.get('subject')!r:<28} {c['text']!r}")
    print("  merged (no llm):")
    for c in extract(msg, use_llm=False):
        print(f"    {c['memory_type']:<11} subject={c.get('subject')!r:<28} {c['text']!r}")
