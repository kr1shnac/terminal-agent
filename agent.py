"""RETAIN - a terminal coding agent with long-term memory.

Run it:

    python agent.py

Memory commands, typed at the prompt:

    /memories            list what is stored
    /recall <text>       search memory and show the score breakdown
    /forget <id>         archive a memory
    /forget <id> --hard  delete it permanently
    /memory stats        counts, confidence, most-used
    /memory decay        run the forgetting sweep now
    /memory consolidate  merge duplicates now
    /extract <text>      show what would be remembered, store nothing
    /history             clear the conversation (memories are kept)
    /quit

Every turn does three things with memory: recall what is relevant, inject it
into the system prompt, and extract anything new the user said.
"""

import json
import os
import signal
import subprocess
import sys
import uuid

from dotenv import load_dotenv
from openai import OpenAI

from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown

from memory import Memory
from memory import consolidate as memory_consolidate
from memory import inbox
from memory import embed
from memory import vectors
from memory.tools import MEMORY_TOOLS
from router.router import route

console = Console()

load_dotenv()

# ------------------------------------------------------------ conversation

# Rolling summarisation: past this many non-system messages the oldest half is
# folded into a summary. The v1 code kept `history` forever, so a long session
# grew without bound until the request either became enormous or failed.
MAX_HISTORY_MESSAGES = 24
SUMMARY_KEEP_RECENT = 8

# Guard against a runaway shell command wedging the whole session.
SHELL_TIMEOUT_SECONDS = 120
MAX_TOOL_OUTPUT_CHARS = 8000
# A tool loop that never terminates is a bug or a loop in the model's plan.
MAX_TOOL_ITERATIONS = 25

SYSTEM_PROMPT = (
    "You are Retain, a helpful AI coding agent. "
    "You can read files, write files, edit files, run shell commands, "
    "list directories, create folders, and search across files. "
    "You have long-term memory that persists across sessions, and the tools "
    "to manage it: remember, recall, forget, memories. "
    "Use retrieved memories when they are relevant, and only then. "
    "Never invent personal information that is not present in the "
    "conversation or in retrieved memories. "
    "If the user corrects you on something personal, the newer statement wins. "
    "When there is no reliable information about something, say you don't know. "
    "Always verify your work - after writing or editing a file, read it back. "
    "After running code, check the output for errors and fix them."
)

# ----------------------------------------------------------------- tools

FILE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write the file according to users input",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "understand the file path or location",
                    },
                    "content": {
                        "type": "string",
                        "description": "the AI return content will be extrated to write in file",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Find a specific piece of text in a file and replace it with new text. Use this to make surgical edits without rewriting the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The path of the file to edit."},
                    "old_text": {"type": "string", "description": "The exact text to find and replace."},
                    "new_text": {"type": "string", "description": "The new text to replace it with."},
                },
                "required": ["file_path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file and return it as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to read. Example: main.py",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List all files and folder inside a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list. Use '.' for current directory.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a new folder. Also creates any missing parent folders",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The folder path to create. Example: projects/my_app",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "Search for a word or a patter across all files in a directory. Returns every line that contain the match, with the file name and line number",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The word or text to serch for."},
                    "directory": {
                        "type": "string",
                        "description": "The directory to search inside. Use '.' for current directory.",
                    },
                },
                "required": ["pattern", "directory"],
            },
        },
    },
]

TOOLS = FILE_TOOLS + MEMORY_TOOLS

FILE_TOOL_NAMES = {tool["function"]["name"] for tool in FILE_TOOLS}


# ------------------------------------------------------------ tool bodies


def _kill_tree(process):
    """Kill a command and everything it started.

    With `shell=True` the process we spawn is the shell, so killing it leaves
    the real command running. It keeps the output pipe open, and the wait for
    that pipe then blocks for as long as the runaway command takes - which
    defeats the timeout entirely rather than enforcing it.
    """
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()


def run_shell(command):
    """Run a shell command with a timeout.

    v1 had no timeout, so one hung command (a prompt, a server, a `tail -f`)
    froze the agent with no way back except killing the process.
    """
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            creationflags=flags,
        )
    except Exception as exc:
        return f"Error running command: {exc}"

    try:
        stdout, stderr = process.communicate(timeout=SHELL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return (
            f"Error: command timed out after {SHELL_TIMEOUT_SECONDS}s and was "
            "killed. If it was meant to keep running, start it in the "
            "background and redirect output to a file."
        )

    output = stdout or ""
    if stderr:
        output += "\nError:\n" + stderr

    return _truncate(output) if output else "no output"


def write_file(file_path, content):
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(str(content))
    except Exception as exc:
        return f"Error writing file: {exc}"
    return f"success, file '{file_path}' written completely"


def edit_file(file_path, old_text, new_text):
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        if old_text not in content:
            return f"Error: could not find the text to replace in '{file_path}'"

        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(content.replace(old_text, new_text, 1))
        return f"Success: edit applied to '{file_path}'"
    except FileNotFoundError:
        return f"Error: file '{file_path}' not found"
    except Exception as exc:
        return f"Error editing file: {exc}"


def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            return _truncate(handle.read())
    except FileNotFoundError:
        return f"Error: file '{file_path}' not found"
    except Exception as exc:
        return f"Error reading file: {exc}"


def list_directory(path):
    try:
        items = os.listdir(path)
        if not items:
            return f"Directory '{path}' is empty"

        lines = []
        for item in sorted(items):
            full = os.path.join(path, item)
            tag = "[DIR] " if os.path.isdir(full) else "[FILE]"
            lines.append(f"{tag} {item}")
        return "\n".join(lines)
    except FileNotFoundError:
        return f"Error: directory '{path}' not found"
    except Exception as exc:
        return f"Error listing directory: {exc}"


def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        return f"Success: directory '{path}' created"
    except Exception as exc:
        return f"Error creating directory: {exc}"


def search_in_files(pattern, directory):
    matches = []
    try:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for name in files:
                if not name.endswith((".py", ".txt", ".md", ".json", ".js", ".ts", ".html", ".css")):
                    continue
                file_path = os.path.join(root, name)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                        for line_num, line in enumerate(handle, 1):
                            if pattern.lower() in line.lower():
                                matches.append(f"{file_path}:{line_num} -> {line.rstrip()}")
                except (OSError, UnicodeDecodeError):
                    # An unreadable file is not a reason to abandon the search.
                    continue
    except Exception as exc:
        return f"Error searching files: {exc}"

    if not matches:
        return f"No matches found for '{pattern}' in '{directory}'."
    return _truncate("\n".join(matches))


def _truncate(text, limit=MAX_TOOL_OUTPUT_CHARS):
    """Cap tool output so one huge file cannot eat the whole context window."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit]}\n... [{dropped} characters truncated]"


FILE_TOOL_IMPLS = {
    "run_shell": run_shell,
    "write_file": write_file,
    "edit_file": edit_file,
    "read_file": read_file,
    "list_directory": list_directory,
    "create_directory": create_directory,
    "search_in_files": search_in_files,
}


def dispatch_tool(tool_name, args, memory):
    if tool_name in FILE_TOOL_IMPLS:
        try:
            return FILE_TOOL_IMPLS[tool_name](**args)
        except TypeError as exc:
            return f"Error: bad arguments for {tool_name}: {exc}"
    if tool_name in {tool["function"]["name"] for tool in MEMORY_TOOLS}:
        return memory.run_tool(tool_name, args)
    return f"Error: unknown tool '{tool_name}'"


# ------------------------------------------------------- history handling


def compact_history(history, keep_recent=SUMMARY_KEEP_RECENT, model=None):
    """Fold the older half of the conversation into one summary message.

    Keeps the recent turns verbatim - tool results and file contents must stay
    exact - and replaces everything before them with a single note. The v1
    agent appended forever, so context, cost and latency all grew without
    limit.
    """
    if len(history) <= keep_recent:
        return history

    head = history[:-keep_recent]
    tail = history[-keep_recent:]

    text = "\n".join(
        f"{message['role']}: {_summarize_part(message)}"
        for message in head
        if message.get("content")
    )

    summary = {
        "role": "user",
        "content": (
            f"[Summary of {len(head)} earlier messages in this session]\n{text}"
        ),
    }
    return [summary] + tail


def _summarize_part(message, limit=220):
    content = message.get("content") or ""
    flat = " ".join(str(content).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "..."


def without_system(history):
    """The API takes a system message separately; history never holds one."""
    return [message for message in history if message.get("role") != "system"]


# ------------------------------------------------------------- memory CLI


def handle_command(user_input, memory, history):
    """Handle a `/command`. Returns True if it was handled."""
    parts = user_input.strip().split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in ("/quit", "/exit"):
        console.log("Good bye")
        return "quit"

    if command == "/memories":
        console.print(memory.render(include_archived="/all" in argument))
        return True

    if command == "/recall":
        if not argument:
            console.print("[yellow]usage: /recall <text>[/yellow]")
            return True
        results = memory.search_detailed(argument, top_k=8)
        if not results:
            console.print(f"[yellow]no memories matched '{argument}'[/yellow]")
            return True
        console.print(memory.explain(results))
        return True

    if command == "/forget":
        raw = argument.split()[0] if argument else ""
        try:
            memory_id = int(raw)
        except ValueError:
            console.print("[yellow]usage: /forget <id> [--hard][/yellow]")
            return True
        mode = "delete" if "--hard" in argument else "archive"
        ok = memory.forget(memory_id, mode=mode)
        if not ok:
            console.print(f"[red]no memory with id {memory_id}[/red]")
        else:
            console.print(f"[green]{mode}d memory #{memory_id}[/green]")
        return True

    if command == "/memory":
        return _handle_memory_command(argument, memory)

    if command == "/extract":
        if not argument:
            console.print("[yellow]usage: /extract <text>[/yellow]")
            return True
        from memory.extract import explain_candidates

        candidates = memory.preview(argument, use_llm=False)
        console.print(f"[dim]would remember:[/dim]\n{explain_candidates(candidates)}")
        return True

    if command == "/history":
        history.clear()
        console.print("[green]conversation cleared (memories kept)[/green]")
        return True

    return False


def _handle_memory_command(argument, memory):
    argument = argument.strip().lower()

    if argument in ("", "stats"):
        stats = memory.stats()
        console.print(
            f"[bold]memory[/bold]  "
            f"{stats['live']} live / {stats['archived']} archived  "
            f"avg confidence [cyan]{stats['avg_confidence']}[/cyan]  "
            f"{stats['total_uses']} recalls  schema v{stats['schema_version']}"
        )
        if stats["by_type"]:
            breakdown = ", ".join(f"{k}={v}" for k, v in stats["by_type"].items())
            console.print(f"[dim]by type: {breakdown}[/dim]")
        for row in stats["most_used"][:3]:
            console.print(
                f"[dim]  #{row['id']} used {row['access_count']}x  {row['text'][:60]}[/dim]"
            )
        return True

    if argument == "decay":
        summary = memory.decay_now()
        console.print(
            f"[green]decay:[/green] checked {summary['checked']}, "
            f"decayed {summary['decayed']}, archived {summary['archived']}, "
            f"expired {summary['expired']}"
        )
        return True

    if argument in ("consolidate", "consolidation"):
        report = memory.consolidate()
        console.print(f"[green]{memory_consolidate.summarize(report)}[/green]")
        return True

    if argument == "inbox" or argument.startswith("inbox "):
        counts = inbox.stats()
        console.print(
            f"[bold]inbox[/bold]  [cyan]{counts.get('pending', 0)}[/cyan] waiting  "
            f"{counts.get('drained', 0)} stored  {counts.get('error', 0)} failed"
        )
        rows = inbox.pending(limit=10)
        for row in rows:
            console.print(
                f"[dim]  #{row['id']} {row['source']} {row['captured_at']}  "
                f"{row['text'][:60]}[/dim]"
            )
        if "drain" in argument.split()[1:]:
            _drain_inbox(memory)
        elif not rows:
            console.print("[dim]  nothing waiting[/dim]")
        return True

    if argument == "reset" or argument.startswith("reset "):
        # `reset` destroys the user's entire memory store, and it used to do so
        # on a bare "/memory reset": the confirmation check could never be
        # false, because the branch that reached it already required the
        # argument to equal "reset" exactly. An irreversible command now needs
        # the confirmation it always appeared to have.
        if "-y" not in argument.split()[1:]:
            console.print(
                "[yellow]this erases every stored memory. "
                "confirm with [/yellow][bold]/memory reset -y[/bold]"
            )
            return True
        memory.reset()
        console.print("[yellow]memory store cleared[/yellow]")
        return True

    if argument in ("persist", "persistence", "where"):
        report = memory.persist_report()
        report["vectors"] = vectors.coverage()
        report["inbox"] = inbox.stats()
        ok = report["integrity"] == "ok" and report["writable"] and report["fts_sane"]
        console.print(
            f"[{'green' if ok else 'red'}]{'durable' if ok else 'PROBLEM'}[/]"
            f"{'  ' if ok else ''}  [bold]{report['path']}[/bold]"
        )
        console.print(
            f"[dim]{report['live']} live / {report['archived']} archived"
            f"  |  {report['bytes']:,} bytes  |  journal={report['journal_mode']}"
            f" sync={report['synchronous']}  |  schema v{report['schema_version']}"
            f"  |  integrity={report['integrity']}"
            f"  |  search index={'ok' if report['fts_sane'] else 'STALE'}"
            f"  |  writable={'yes' if report['writable'] else 'NO'}[/dim]"
        )
        if report["newest"]:
            console.print(f"[dim]newest memory: {report['newest']}[/dim]")
        console.print(
            "[dim]survives restart: this file is the store, and the write-ahead "
            "log is folded into it on exit[/dim]"
        )
        return True

    console.print(
        "[yellow]usage: /memory stats | persist | decay | consolidate | reset[/yellow]"
    )
    return True


# ------------------------------------------------------------------- main


def build_client():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        console.print(
            "[red]OPENROUTER_API_KEY is not set. Add it to .env and retry.[/red]"
        )
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def banner(memory):
    console.print(
        Align.center(
            r"""
 _____  _____ _____  _    ___ _   _
|  __ \| ____|_   _| / \  |_ _| \ | |
| |__) |  _|   | |  / _ \  | ||  \| |
|  _  /| |___  | | / ___ \ | || |\  |
|_| \_\|_____| |_|/_/   \_\___|_| \_|
        """,
            style="bold bright_cyan",
        )
    )
    console.print(Align.center("[dim]my first terminal AGENT[/dim]"))
    stats = memory.stats()
    console.print(
        f"[dim]memory: {stats['live']} live, {stats['archived']} archived "
        f"(/memories, /recall, /memory stats)[/dim]"
    )
    coverage = vectors.coverage()
    if coverage["live"]:
        console.print(
            f"[dim]semantic index: {coverage['indexed']}/{coverage['live']} "
            f"memories[/dim]"
        )
    # Say where memory lives on the very first screen. "The agent forgot
    # everything" and "the agent is reading a different file" look identical
    # from the outside, and the cheapest way to tell them apart is to print the
    # path every startup.
    report = memory.persist_report()
    console.print(f"[dim]store: {report['path']}[/dim]")
    waiting = inbox.count_pending()
    if waiting:
        console.print(
            f"[dim]inbox: {waiting} captured while closed, draining now[/dim]"
        )
    console.print()


def _drain_inbox(memory, quiet=False):
    """Fold queued offline captures into the store.

    Runs at startup and on demand from `/memory inbox drain`. Failures are
    reported, never raised: a queue problem must not stop the agent from
    starting, and the entries stay queued either way.
    """
    try:
        report = inbox.drain()
    except Exception as exc:  # noqa: BLE001 - never block startup
        console.print(f"[yellow]inbox drain failed: {exc}[/yellow]")
        return None
    if report["processed"] and not quiet:
        console.print(f"[green]{inbox.summarize(report)}[/green]")
        for entry in report["entries"]:
            if entry.get("error"):
                console.print(f"[red]  failed: {entry['text'][:60]}[/red]")
    return report


def capture_offline(argv):
    """`agent.py --remember TEXT` / `--observe`: queue a memory, then exit.

    This is the whole point of the inbox: capturing a fact must not require the
    agent to be running, must not need an API key, and must not depend on a
    model call succeeding. So it writes a queue row and exits, and the work of
    classifying and indexing the memory happens later on the agent's own
    schedule.
    """
    text = " ".join(argv).strip()
    if not text:
        console.print("[red]nothing to remember[/red]")
        return 2
    try:
        entry_id = inbox.capture(text, source="cli")
    except inbox.CaptureError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    pending = inbox.count_pending()
    console.print(
        f"[green]queued[/green] #{entry_id}  [dim]{pending} waiting "
        f"for the next agent start[/dim]"
    )
    console.print(f"[dim]{text[:100]}[/dim]")
    return 0


def observe_offline():
    """`agent.py --observe`: queue every line of stdin as a memory."""
    count = 0
    for line in sys.stdin:
        cleaned = " ".join(line.split()).strip()
        if not cleaned:
            continue
        try:
            inbox.capture(cleaned, source="stdin")
            count += 1
        except inbox.CaptureError as exc:
            console.print(f"[yellow]skipped a line: {exc}[/yellow]")
    console.print(f"[green]queued {count}[/green] from stdin")
    return 0 if count else 1


def _configure_embedder():
    """Activate the best embedder this machine can run, and say which one.

    See `embed.select`: a configured hosted model first, then a local ONNX model
    if its weights are present (free, private, offline), then the
    dependency-free hashing embedder. `select` installs what it picks; this only
    decides whether the switch is worth telling the user about.

    Announcing it matters more than it looks: which model indexed the store is
    why a memory came back, and a silent switch would also leave every older
    memory without a vector under the new model until the backfill runs.
    """
    previous = embed.current()
    try:
        chosen = embed.select()
    except Exception as exc:  # noqa: BLE001 - a bad model must not block startup
        console.print(f"[yellow]embedding model unavailable: {exc}[/yellow]")
        return previous
    if chosen is previous:
        return previous
    if isinstance(chosen, embed.OnnxEmbedder):
        detail = f"{chosen.dim} dims, runs locally, no network"
    elif isinstance(chosen, embed.OpenAICompatibleEmbedder):
        detail = f"{chosen.dim} dims, {chosen.model}"
    else:
        detail = f"{chosen.dim} dims"
    console.print(f"[dim]semantic model: {chosen.name} ({detail})[/dim]")
    return chosen


def _backfill_vectors():
    """Give pre-existing memories a vector.

    Anything stored before the vector index existed - or under a different
    embedder - is invisible to the semantic stage until it is re-embedded, and
    it fails *silently*: retrieval just gets slightly worse, with nothing in the
    output to say why. So the gap is closed once, at startup, where the agent
    already holds the database, rather than being left for the user to notice.

    Runs after `_drain_inbox` so freshly captured memories are indexed by their
    own insert and are not counted here.
    """
    try:
        coverage = vectors.coverage()
        if not coverage["missing"]:
            return None
        indexed, skipped, _current = vectors.reindex()
    except Exception as exc:  # noqa: BLE001 - never block startup
        console.print(f"[yellow]vector backfill failed: {exc}[/yellow]")
        return None
    if indexed:
        detail = f", {skipped} skipped" if skipped else ""
        console.print(
            f"[dim]indexed {indexed} existing memories for semantic "
            f"search{detail}[/dim]"
        )
    return indexed


def main():
    # Facts captured while the agent was closed are sitting in the durable
    # inbox. Fold them in before the first prompt is built, so a fact the user
    # asked to remember offline is already recallable in their very first
    # message - otherwise "the app was off" would still cost them the memory for
    # the length of one session.
    _drain_inbox(None)
    _configure_embedder()
    _backfill_vectors()

    client = build_client()

    # A cheap model for the memory side-calls. Extraction and consolidation
    # both run on cheap prompts, so they should not use the routed model.
    extractor_model = route("summarise this in one line")[0]

    memory = Memory(
        session_id=uuid.uuid4().hex[:12],
        client=client,
        model=extractor_model,
    )

    try:
        _repl(memory, client, extractor_model)
    finally:
        # Whatever ended the session - /quit, Ctrl-C, EOF, or a crash on the
        # way out - fold the write-ahead log into app.db. Every memory is
        # already committed and durable; this is what makes the single file on
        # disk complete, so the next start reads everything and a backup of
        # app.db is a real backup.
        try:
            memory.flush()
        except Exception as exc:  # pragma: no cover - never block the exit
            console.print(f"[yellow]memory flush on exit failed: {exc}[/yellow]")


def _repl(memory, client, extractor_model):
    history = []

    banner(memory)

    if client is None:
        return

    while True:
        try:
            user_input = console.input("[bold green]You > [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            console.log("Good bye")
            return

        if not user_input:
            continue

        if user_input.startswith("/"):
            result = handle_command(user_input, memory, history)
            if result == "quit":
                return
            if result:
                continue

        # 1. Maintenance: forget what is stale, before anything reads it.
        try:
            memory.tick()
        except Exception as exc:
            console.print(f"[yellow]memory maintenance skipped: {exc}[/yellow]")

        # 2. Capture anything durable the user just said. Done before the
        #    model runs so the current turn can already use it.
        try:
            learned = memory.observe(user_input, client=client, model=extractor_model)
            for item in learned:
                console.print(
                    f"[dim magenta]noted ({item.memory_type}) #{item.id}: "
                    f"{item.text[:80]}[/dim magenta]"
                )
        except Exception as exc:
            console.print(f"[yellow]memory capture skipped: {exc}[/yellow]")

        # 3. Recall what is relevant to this turn.
        memory_block, results = memory.context(user_input)
        _print_recall(results)

        # 4. Route, then run the tool loop.
        model, tier, reason = route(user_input)
        console.print(f"[dim cyan]Router -> {tier} | {reason}[/dim cyan]")

        history.append({"role": "user", "content": user_input})
        reply = run_turn(client, model, memory, history, memory_block)

        if reply is None:
            # The API call failed; drop this turn so the next one is valid.
            if history and history[-1]["role"] == "user":
                history.pop()
            continue

        history.append({"role": "assistant", "content": reply})
        console.print("\n[bold purple]Retain >[/bold purple]")
        console.print(Markdown(reply))
        console.print()

        history[:] = compact_history(history)


def _print_recall(results):
    if not results:
        return
    console.print(f"[dim green]recall ({len(results)}):[/dim green]")
    for entry in results:
        item = entry["item"]
        console.print(
            f"[dim green]  #{item.id} {entry['score']:.2f} "
            f"({item.memory_type}) {item.text[:70]}[/dim green]"
        )


def run_turn(client, model, memory, history, memory_block):
    """One user turn: the tool-call loop. Returns the reply text, or None."""
    system_prompt = SYSTEM_PROMPT
    if memory_block:
        system_prompt = f"{SYSTEM_PROMPT}\n\n{memory_block}"

    messages = [
        {"role": "system", "content": system_prompt}
    ] + without_system(history)

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
            )
        except Exception as exc:
            console.print(f"\n[red]API error: {exc}[/red]")
            return None

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or ""

        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ]
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": tool_calls,
            }
        )
        history.append(messages[-1])

        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            preview = next(iter(args.values()), "") if args else ""
            console.print(
                f"\n[dim yellow]> {call.function.name} -> "
                f"{str(preview)[:100]}[/dim yellow]"
            )

            result = dispatch_tool(call.function.name, args, memory)
            shown = result[:120] + "..." if len(result) > 120 else result
            console.print(f"[dim green]  {shown}[/dim green]")

            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            }
            messages.append(tool_message)
            history.append(tool_message)

    console.print(
        f"[yellow]stopped after {MAX_TOOL_ITERATIONS} tool iterations[/yellow]"
    )
    return "I got stuck in a tool loop. Try rephrasing the request."


if __name__ == "__main__":
    # Offline capture runs before anything else, and deliberately without an API
    # client: `--remember` has to work with the agent closed and no key in the
    # environment, because "I want to keep this" should never depend on a
    # network round trip.
    if "--remember" in sys.argv:
        index = sys.argv.index("--remember")
        raise SystemExit(capture_offline(sys.argv[index + 1 :]))
    if "--observe" in sys.argv:
        raise SystemExit(observe_offline())
    main()
