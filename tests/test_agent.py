"""Tests for the agent's memory wiring and tool loop.

`agent.py` had no tests at all. These drive `run_turn` with a fake client, so
they check the loop's shape - that memory reaches the system prompt, that tool
results are fed back, that history stays consistent - without a network call.

    python -m unittest discover -s tests -t .
"""

import json
import os
import shutil
import tempfile
import types
import unittest
from io import StringIO

_TEST_DIR = tempfile.mkdtemp(prefix="retain-agent-tests-")
os.environ["RETAIN_DB_PATH"] = os.path.join(_TEST_DIR, "agent-tests.db")

import agent  # noqa: E402
from memory import Memory  # noqa: E402
from memory import db, store  # noqa: E402
from memory.tools import MEMORY_TOOLS  # noqa: E402

# The agent narrates every tool call to the terminal, which is unreadable in
# test output - the runaway-loop test alone prints 25 directory listings.
_QUIET = StringIO()
agent.console = agent.Console(file=_QUIET, width=200)


def tool_call(name, arguments, call_id="call_1"):
    return types.SimpleNamespace(
        id=call_id,
        function=types.SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def assistant(tool_calls=None, content=None):
    return types.SimpleNamespace(
        tool_calls=tool_calls or [], content=content
    )


def reply(message):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class ScriptedClient:
    """Replays a fixed list of assistant turns, recording every request."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.sent = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = self.turns.pop(0) if self.turns else assistant(content="done")
        self.sent.append(kwargs["messages"])
        return reply(message)

    @property
    def chat(self):
        return types.SimpleNamespace(completions=self)

    @property
    def last_messages(self):
        return self.requests[-1]["messages"]

    @property
    def system_prompt(self):
        return self.last_messages[0]["content"]


class AgentTestCase(unittest.TestCase):
    def setUp(self):
        db.reset_connection()
        store.forget_all()
        self.memory = Memory(auto_maintain=False)

    def run_turn(self, client, history, memory_block=""):
        return agent.run_turn(client, "fake-model", self.memory, history, memory_block)


class TestMemoryReachesThePrompt(AgentTestCase):
    def test_the_retrieved_block_is_injected(self):
        self.memory.add("The user's name is Krishna", "identity", subject="user.name")
        block, _results = self.memory.context("what is my name?")

        client = ScriptedClient([assistant(content="You are Krishna.")])
        self.run_turn(client, [{"role": "user", "content": "what is my name?"}], block)

        self.assertIn("Krishna", client.system_prompt)
        self.assertIn("What you know about the user", client.system_prompt)

    def test_with_no_memories_the_prompt_is_unchanged(self):
        client = ScriptedClient([assistant(content="hi")])
        self.run_turn(client, [{"role": "user", "content": "hi"}], "")

        self.assertEqual(client.system_prompt, agent.SYSTEM_PROMPT)
        self.assertNotIn("What you know about the user", client.system_prompt)

    def test_the_recent_turns_are_kept(self):
        self.memory.add("The user prefers pnpm", "preference")
        block, _ = self.memory.context("package manager")

        history = [
            {"role": "user", "content": "older question"},
            {"role": "assistant", "content": "older answer"},
            {"role": "user", "content": "which package manager?"},
        ]
        client = ScriptedClient([assistant(content="pnpm")])
        self.run_turn(client, history, block)

        contents = [m["content"] for m in client.last_messages if m["role"] != "system"]
        self.assertIn("older question", contents)
        self.assertIn("which package manager?", contents)


class TestToolLoop(AgentTestCase):
    def test_a_tool_call_is_run_and_fed_back(self):
        client = ScriptedClient([
            assistant(tool_calls=[tool_call("read_file", {"file_path": "notes.txt"})]),
            assistant(content="The file says hello."),
        ])

        path = os.path.join(_TEST_DIR, "notes.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("hello")
        previous = os.getcwd()
        os.chdir(_TEST_DIR)
        try:
            history = [{"role": "user", "content": "read notes.txt"}]
            result = self.run_turn(client, history)
        finally:
            os.chdir(previous)

        self.assertEqual(result, "The file says hello.")
        self.assertEqual(len(client.requests), 2)

        second = client.last_messages
        roles = [m["role"] for m in second]
        self.assertIn("tool", roles)
        tool_message = [m for m in second if m["role"] == "tool"][0]
        self.assertEqual(tool_message["content"], "hello")
        self.assertEqual(tool_message["tool_call_id"], "call_1")

    def test_the_assistant_tool_call_precedes_its_results(self):
        # OpenAI rejects a tool result that does not immediately follow the
        # assistant message carrying the matching tool_calls.
        client = ScriptedClient([
            assistant(tool_calls=[
                tool_call("list_directory", {"path": "."}, "a"),
                tool_call("list_directory", {"path": "."}, "b"),
            ]),
            assistant(content="ok"),
        ])

        self.run_turn(client, [{"role": "user", "content": "ls"}])

        messages = client.last_messages
        assistant_index = next(
            i for i, m in enumerate(messages)
            if m["role"] == "assistant" and "tool_calls" in m
        )
        self.assertEqual(messages[assistant_index + 1]["role"], "tool")
        self.assertEqual(messages[assistant_index + 2]["role"], "tool")
        self.assertEqual(
            [m["tool_call_id"] for m in messages[assistant_index + 1: assistant_index + 3]],
            ["a", "b"],
        )

    def test_history_records_the_exchange(self):
        client = ScriptedClient([
            assistant(tool_calls=[tool_call("read_file", {"file_path": "missing.txt"})]),
            assistant(content="not there"),
        ])

        history = [{"role": "user", "content": "read missing.txt"}]
        self.run_turn(client, history)

        roles = [m["role"] for m in history]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        self.assertIn("tool_calls", history[1])

    def test_bad_json_arguments_do_not_crash(self):
        call = types.SimpleNamespace(
            id="call_x",
            function=types.SimpleNamespace(name="read_file", arguments="{not json"),
        )
        client = ScriptedClient([
            assistant(tool_calls=[call]),
            assistant(content="recovered"),
        ])

        result = self.run_turn(client, [{"role": "user", "content": "read it"}])
        self.assertEqual(result, "recovered")

    def test_an_unknown_tool_is_reported_not_raised(self):
        client = ScriptedClient([
            assistant(tool_calls=[tool_call("teleport", {})]),
            assistant(content="ok"),
        ])
        result = self.run_turn(client, [{"role": "user", "content": "go"}])
        self.assertEqual(result, "ok")
        tool_message = [m for m in client.last_messages if m["role"] == "tool"][0]
        self.assertIn("unknown tool", tool_message["content"])

    def test_a_runaway_tool_loop_stops(self):
        client = ScriptedClient([
            assistant(tool_calls=[tool_call("list_directory", {"path": "."}, f"c{i}")])
            for i in range(agent.MAX_TOOL_ITERATIONS + 10)
        ])

        result = self.run_turn(client, [{"role": "user", "content": "loop"}])

        self.assertIn("tool loop", result)
        self.assertEqual(len(client.requests), agent.MAX_TOOL_ITERATIONS)

    def test_an_api_error_returns_none_and_leaves_history_clean(self):
        class Down:
            @property
            def chat(self):
                class C:
                    @property
                    def completions(self):
                        raise RuntimeError("network down")
                return C()

        history = [{"role": "user", "content": "hello"}]
        self.assertIsNone(self.run_turn(Down(), history))
        # The caller pops the user turn; the point is that no half-built tool
        # exchange is left behind.
        self.assertEqual([m["role"] for m in history], ["user"])


class TestMemoryToolsAreWired(AgentTestCase):
    def test_the_memory_tools_are_registered(self):
        names = {tool["function"]["name"] for tool in agent.TOOLS}
        for tool in MEMORY_TOOLS:
            self.assertIn(tool["function"]["name"], names)
        self.assertIn("read_file", names)

    def test_remember_is_reachable_through_dispatch(self):
        result = agent.dispatch_tool(
            "remember",
            {"text": "The user ships on Fridays", "memory_type": "preference"},
            self.memory,
        )
        stored = store.get_active()
        self.assertEqual(len(stored), 1)
        self.assertIn(str(stored[0].id), result)
        self.assertIn("Friday", stored[0].text)

    def test_recall_is_reachable_through_dispatch(self):
        self.memory.add("The user's name is Krishna", "identity")
        result = agent.dispatch_tool("recall", {"query": "name"}, self.memory)
        self.assertIn("Krishna", result)

    def test_forget_is_reachable_through_dispatch(self):
        item = self.memory.add("The user dislikes tabs", "preference")
        result = agent.dispatch_tool("forget", {"memory_id": item.id}, self.memory)
        self.assertIn(str(item.id), result)
        self.assertTrue(store.get(item.id).is_archived)


class TestToolBodies(AgentTestCase):
    def test_shell_output_is_truncated(self):
        out = agent.run_shell("python -c \"print('x' * 50000)\"")
        self.assertLess(len(out), agent.MAX_TOOL_OUTPUT_CHARS + 200)
        self.assertIn("truncated", out)

    def test_a_hanging_command_times_out(self):
        original = agent.SHELL_TIMEOUT_SECONDS
        agent.SHELL_TIMEOUT_SECONDS = 1
        try:
            out = agent.run_shell("python -c \"import time; time.sleep(30)\"")
        finally:
            agent.SHELL_TIMEOUT_SECONDS = original

        self.assertIn("timed out", out)

    def test_a_failing_shell_command_reports_the_error(self):
        out = agent.run_shell("python -c \"raise SystemExit(3)\"")
        self.assertTrue(out.strip())

    def test_search_does_not_die_on_unreadable_files(self):
        binary = os.path.join(_TEST_DIR, "blob.bin")
        with open(binary, "wb") as handle:
            handle.write(b"\xff\xfe\x00\x01needle\x00")
        out = agent.search_in_files("needle", _TEST_DIR)
        self.assertIsInstance(out, str)

    def test_edit_reports_a_missing_anchor(self):
        path = os.path.join(_TEST_DIR, "edit-me.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("hello world")
        out = agent.edit_file(path, "not there", "x")
        self.assertIn("could not find", out)

    def test_write_then_read_round_trips(self):
        path = os.path.join(_TEST_DIR, "nested", "deep", "out.txt")
        self.assertIn("success", agent.write_file(path, "content here"))
        self.assertEqual(agent.read_file(path), "content here")


class TestHistoryCompaction(AgentTestCase):
    def test_short_history_is_untouched(self):
        history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]
        self.assertEqual(agent.compact_history(history), history)

    def test_a_long_history_collapses_to_a_summary_plus_the_recent_turns(self):
        history = []
        for index in range(40):
            history.append({"role": "user" if index % 2 == 0 else "assistant",
                            "content": f"message {index}"})

        compacted = agent.compact_history(history)

        self.assertLess(len(compacted), len(history))
        self.assertEqual(
            len(compacted), agent.SUMMARY_KEEP_RECENT + 1
        )
        self.assertIn("Summary of", compacted[0]["content"])
        # The recent turns must survive verbatim: tool payloads and file
        # contents cannot be summarised away.
        self.assertEqual(
            [m["content"] for m in compacted[-agent.SUMMARY_KEEP_RECENT:]],
            [m["content"] for m in history[-agent.SUMMARY_KEEP_RECENT:]],
        )

    def test_compaction_is_idempotent(self):
        history = [{"role": "user", "content": f"m{i}"} for i in range(30)]
        once = agent.compact_history(history)
        twice = agent.compact_history(once)
        self.assertEqual(len(once), len(twice))

    def test_the_system_message_is_never_duplicated(self):
        history = [{"role": "system", "content": "stale"}] + [
            {"role": "user", "content": f"m{i}"} for i in range(5)
        ]
        self.assertEqual(
            [m for m in agent.without_system(history) if m["role"] == "system"], []
        )


class TestCommands(AgentTestCase):
    def test_unknown_command_is_not_swallowed(self):
        self.assertFalse(agent.handle_command("/nonsense", self.memory, []))

    def test_history_command_clears_the_conversation_only(self):
        self.memory.add("The user prefers pnpm", "preference")
        history = [{"role": "user", "content": "hi"}]
        self.assertTrue(agent.handle_command("/history", self.memory, history))
        self.assertEqual(history, [])
        self.assertEqual(len(store.get_active()), 1)

    def test_forget_requires_an_id(self):
        self.assertTrue(agent.handle_command("/forget", self.memory, []))

    def test_recall_requires_a_query(self):
        self.assertTrue(agent.handle_command("/recall", self.memory, []))

    def test_quit_reports_quit(self):
        self.assertEqual(agent.handle_command("/quit", self.memory, []), "quit")


def tearDownModule():
    shutil.rmtree(_TEST_DIR, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
