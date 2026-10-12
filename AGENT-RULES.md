# RULES FOR THIS WINDOW (read fully, obey exactly)
1. You are one of 5 parallel agents. Work ONLY in the current folder and ONLY on the current git branch. Never switch/checkout other branches, merge, rebase, push, tag, stash or create worktrees. Never touch files outside your task's file list.
2. Nobody will answer questions. Never ask. If blocked, write BLOCKED.md in the repo root (what, why, what you tried) and stop.
3. Use only the project venv: `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/python`. Never plain `pytest` or `python`. Never pip install.
4. Work in small steps. After EVERY file you create or edit, run the tests for that file.
5. Never weaken, delete or skip an existing test or assertion. Never add pytest.skip. (xfail only when your task says so.)
6. Never print, read or search for secrets or environment variables. Do not run env or printenv. No API keys exist for you.
7. No network access except where your task explicitly allows it.
8. Every claim in your final reply must be backed by command output pasted from THIS session. If you did not run it, write NOT RUN. Report test counts as before -> after.
9. Commit on your branch when your definition of done is met, using `git add` with ONLY the files in your task's file list (never -A or .), and a clear message.
10. Your last message: the DEFINITION OF DONE checklist with PASS/FAIL per line plus the final command outputs. Max 30 lines.
13. [EXCEPTION 2026-10-06]: `textual` and its dependencies were explicitly approved for installation to build the TUI.
