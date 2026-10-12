import os
import queue
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Input
from textual import work
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown
from tui.events import TurnStart, MessageDelta, ToolStart, ToolEnd, TurnEnd, StatusUpdate

TCSS = """
Screen {
    background: #0d0d0d;
    layout: vertical;
    color: #e8e8e8;
}

#transcript-area {
    height: 1fr;
    padding: 1 2;
    overflow-y: auto;
    align: center top;
}

.turn {
    height: auto;
    margin-bottom: 1;
    max-width: 120;
}

.user-turn-container {
    height: auto;
    margin-bottom: 1;
    padding-bottom: 1;
    max-width: 120;
}

.user-turn {
    color: #f8fafc;
    text-style: bold;
}

.agent-turn {
    color: #e2e8f0;
}

.gutter-icon {
    width: auto;
    margin-right: 1;
    content-align: left top;
}
.user-icon { color: #00ff9f; }
.agent-icon { color: #00b8ff; }

/* Tool Cards */
.tool-card {
    background: rgba(255, 255, 255, 0.03);
    border: round rgba(255, 255, 255, 0.1);
    border-top: round rgba(255, 255, 255, 0.3);
    padding: 0 1;
    margin: 1 0 1 4;
    width: 80%;
    color: #94a3b8;
}

.tool-active {
    border: round #00b8ff;
    border-top: round #00ff9f;
    color: #f8fafc;
}

.tool-finished {
    background: transparent;
    border: none;
    padding: 0 1;
    margin: 0 0 0 4;
    color: #64748b;
}

.tool-header {
    layout: horizontal;
    height: 1;
}

.tool-name { text-style: bold; width: 1fr; }
.tool-status { width: auto; }
.tool-args { color: #64748b; }

#header-bar {
    layout: horizontal;
    padding: 0 2 0 2;
    height: 2;
    background: #1a1a1a;
    border-bottom: solid #3a3a3a;
}

#wordmark {
    width: 1fr;
    content-align: center middle;
    text-style: bold;
    padding: 0 4;
    color: #00b8ff;
}

#model-display {
    width: auto;
    color: #00ff9f;
    content-align: right middle;
    padding: 0 4;
    font-size: 1;
    text-style: bold;
    min-width: 80;
}

/* Composer */
#composer {
    height: 3;
    padding: 0 2;
    background: #1a1a1a;
    border-top: solid #3a3a3a;
    color: #e8e8e8;
}

.composer-prompt {
    color: #00ff9f;
    text-style: bold;
    width: 4;
    content-align: center middle;
}

#user-input {
    width: 1fr;
    background: #0d0d0d;
    border: solid #3a3a3a;
    color: #f8fafc;
}

#user-input:focus {
    border: solid #00b8ff;
    background: #1a1a1a;
}
"""

class TAMIASApp(App):
    CSS = TCSS

    def __init__(self, ui_queue: queue.Queue, agent_queue: queue.Queue):
        super().__init__()
        self.ui_queue = ui_queue
        self.agent_queue = agent_queue
        self.current_agent_turn = None
        self.current_agent_text = ""
        self.active_tools = {}
        self.current_model = os.environ.get("AGENT_MODEL", "nvidia/nemotron-3.5-lightning:free")
        self.current_effort = os.environ.get("AGENT_EFFORT_POLICY", "auto")

    def compose(self) -> ComposeResult:
        with Horizontal(id="header-bar"):
            yield Static(create_gradient_text("T A M I A S   A G E N T", "#00b8ff", "#00ff9f"), id="wordmark")
            yield Static(f"🟢 ⟨ {self.current_model} · {self.current_effort} ⟩", id="model-display")
        
        with ScrollableContainer(id="transcript-area"):
            pass

        with Horizontal(id="composer"):
            yield Static("│ ❯ ", classes="composer-prompt")
            yield Input(placeholder="type your message here...", id="user-input")

    def on_mount(self) -> None:
        self.listen_to_agent()
        self.query_one(Input).focus()
        
        self.set_interval(0.1, self.tick_spinners)
        
        # Inject welcome message
        transcript = self.query_one("#transcript-area")
        welcome_turn = Horizontal(
            Static("◆", classes="gutter-icon"),
            Static(RichMarkdown("Welcome to **TAMIAS**. Type your task to begin, or use `/model` to change the engine."), classes="agent-turn"),
            classes="turn"
        )
        self.call_later(transcript.mount, welcome_turn)

    def tick_spinners(self) -> None:
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = (self.spinner_idx + 1) % len(frames)
        if getattr(self, "agent_busy", False) and getattr(self, "agent_busy_msg", ""):
            if getattr(self, "current_agent_turn", None) and hasattr(self, "agent_markdown"):
                self.agent_markdown.update(f"{frames[self.spinner_idx]} {self.agent_busy_msg}")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if user_text == "/quit":
            self.exit()
            return
            
        if user_text.startswith("/model"):
            event.input.value = ""
            def switch_model(selected_model: str):
                if selected_model:
                    self.ui_queue.put(f"/model {selected_model}")
            
            from tui.model_picker import ModelPicker
            self.push_screen(ModelPicker(), switch_model)
            return
            
        event.input.value = ""
        
        # Display user turn
        transcript = self.query_one("#transcript-area")
        turn = Horizontal(
            Static("❯", classes="gutter-icon user-icon"),
            Static(user_text, classes="user-turn"),
            classes="user-turn-container"
        )
        await transcript.mount(turn)
        turn.scroll_visible()
        
        # Send to agent
        self.ui_queue.put(user_text)

    @work(thread=True)
    def listen_to_agent(self) -> None:
        self.listening = True
        while getattr(self, "listening", True):
            try:
                event = self.agent_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            except Exception:
                break

            # Check dict for internal messages
            if isinstance(event, dict):
                if event.get("type") == "clear":
                    self.call_from_thread(self.clear_transcript)
                elif event.get("type") == "init":
                    self.call_from_thread(self.update_model_display, event.get("model", ""))
                elif event.get("type") == "model_update":
                    self.call_from_thread(self.update_model_display, event.get("model", ""))
                continue

            if isinstance(event, TurnStart):
                self.call_from_thread(self.start_agent_turn)
            elif isinstance(event, MessageDelta):
                self.call_from_thread(self.append_agent_text, event.content)
            elif isinstance(event, ToolStart):
                self.call_from_thread(self.start_tool, event.tool_call_id,
                                      event.tool_name, event.arguments)
            elif isinstance(event, ToolEnd):
                self.call_from_thread(self.end_tool, event.tool_call_id,
                                      event.result, event.duration_sec)
            elif isinstance(event, TurnEnd):
                self.call_from_thread(self.end_agent_turn)
            elif isinstance(event, StatusUpdate):
                self.call_from_thread(self.handle_status_update, event.message, event.is_busy)

    def on_unmount(self):
        self.listening = False

    def clear_transcript(self):
        self.query_one("#transcript-area").remove_children()

    def update_model_display(self, model_name: str):
        self.current_model = model_name
        self.query_one("#model-display").update(f"🟢 ⟨ {model_name} · {self.current_effort} ⟩")

    def start_agent_turn(self):
        self.current_agent_text = ""
        transcript = self.query_one("#transcript-area")
        
        self.agent_markdown = Static("", classes="agent-turn")
        self.agent_vertical = Vertical(self.agent_markdown)
        
        self.current_agent_turn = Horizontal(
            Static("⬡", classes="gutter-icon agent-icon"),
            self.agent_vertical,
            classes="turn"
        )
        
        transcript.mount(self.current_agent_turn)
        self.current_agent_turn.scroll_visible()
        
        inp = self.query_one(Input)
        inp.disabled = True
        inp.placeholder = "Agent is working..."

    def _format_agent_text(self, text: str) -> str:
        import re
        if "<thought>" in text:
            def replacer(match):
                inner = match.group(1)
                return "> **Thinking:**\n" + "\n".join("> " + line for line in inner.split("\n"))
            return re.sub(r"<thought>(.*?)(?:</thought>|$)", replacer, text, flags=re.DOTALL)
        return text

    def handle_status_update(self, message: str, is_busy: bool):
        if not self.current_agent_turn:
            self.start_agent_turn()
            
        self.agent_busy = is_busy
        self.agent_busy_msg = message
            
        if is_busy and message:
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            self.agent_markdown.update(f"{frames[self.spinner_idx]} {message}")
            self.agent_markdown.styles.color = "#00b8ff"
        else:
            self.agent_markdown.styles.color = "#e2e8f0"
            if message and message.startswith("Error:"):
                # Append the error message so it doesn't get erased
                self.current_agent_text += f"\n\n**{message}**"
                
            self.agent_markdown.update(RichMarkdown(self._format_agent_text(self.current_agent_text)))
            self.agent_markdown.scroll_visible()

    def append_agent_text(self, text: str):
        self.current_agent_text += text
        self.agent_markdown.update(RichMarkdown(self._format_agent_text(self.current_agent_text)))
        self.agent_markdown.scroll_visible()

    def start_tool(self, tool_id: str, name: str, args: str):
        args_preview = args if len(args) < 60 else args[:60] + "..."
        
        status_spinner = Spinner("⠧ running...", classes="tool-status active-spinner")
        header = Horizontal(
            Static(name, classes="tool-name"),
            status_spinner,
            classes="tool-header"
        )
        
        card = Vertical(
            header,
            Static(args_preview, classes="tool-args"),
            classes="tool-card tool-active",
            id=f"tool-{tool_id}"
        )
        
        self.agent_vertical.mount(card)
        self.active_tools[tool_id] = card
        card.scroll_visible()

    def end_tool(self, tool_id: str, result: str, duration_sec: float = 0.0):
        if tool_id in self.active_tools:
            card = self.active_tools.pop(tool_id)
            card.remove_class("tool-active")
            card.add_class("tool-finished")
            status = card.query(".tool-status").first(Spinner)

            failed = result.startswith("Error:")
            if failed:
                status.stop("✗ failed", "red")
            else:
                final = f"✓ {duration_sec:.1f}s" if duration_sec else "✓ done"
                status.stop(final, "#64748b")
                
            args_static = card.query(".tool-args").first(Static)
            args_static.display = False

    def end_agent_turn(self):
        self.current_agent_turn = None
        self.current_agent_text = ""
        
        inp = self.query_one(Input)
        inp.disabled = False
        inp.placeholder = "type your message here..."
        inp.focus()

def create_gradient_text(text: str, start_color: str, end_color: str) -> Text:
    from textual.color import Color
    rich_text = Text()
    c1 = Color.parse(start_color)
    c2 = Color.parse(end_color)
    rich_text = Text()
    length = len(text)
    
    for i, char in enumerate(text):
        ratio = i / max(1, length - 1)
        r = int(c1.r + (c2.r - c1.r) * ratio)
        g = int(c1.g + (c2.g - c1.g) * ratio)
        b = int(c1.b + (c2.b - c1.b) * ratio)
        rich_text.append(char, style=f"rgb({r},{g},{b}) bold")
    return rich_text