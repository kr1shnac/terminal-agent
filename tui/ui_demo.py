import sys
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, Container, ScrollableContainer
from textual.widgets import Static, Markdown, Header, Footer
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown

def create_gradient_text(text: str, start_color: str, end_color: str) -> Text:
    from textual.color import Color
    rich_text = Text()
    c1 = Color.parse(start_color)
    c2 = Color.parse(end_color)
    length = len(text)
    
    for i, char in enumerate(text):
        ratio = i / max(1, length - 1)
        r = int(c1.r + (c2.r - c1.r) * ratio)
        g = int(c1.g + (c2.g - c1.g) * ratio)
        b = int(c1.b + (c2.b - c1.b) * ratio)
        rich_text.append(char, style=f"rgb({r},{g},{b}) bold")
    return rich_text

TCSS = """
Screen {
    background: transparent; /* Allows terminal bg to bleed through if supported */
    layout: vertical;
}

#transcript-area {
    height: 1fr;
    padding: 1 4;
    overflow-y: auto;
}

/* Transcript-first unboxed flow */
.turn {
    margin-bottom: 2;
}

.user-turn {
    color: #f8fafc;
    text-style: bold;
}

.agent-turn {
    color: #e2e8f0;
}

.gutter-icon {
    width: 3;
    content-align: left top;
}

.user-icon { color: #00ff9f; }
.agent-icon { color: #00b8ff; }

/* Tool Cards (Glass Surfaces) */
.tool-card {
    background: rgba(255, 255, 255, 0.05);
    border: round rgba(255, 255, 255, 0.15);
    border-top: round rgba(255, 255, 255, 0.4); /* specular edge */
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

.tool-header {
    layout: horizontal;
    height: 1;
}

.tool-name { text-style: bold; width: 1fr; }
.tool-status { width: auto; }

/* Plan / Todo List Overlay */
.plan-card {
    background: rgba(255, 255, 255, 0.08);
    border: hkey rgba(255, 255, 255, 0.2);
    padding: 1 2;
    margin: 1 0 2 4;
    width: 80%;
}

.plan-title {
    text-style: bold italic;
    color: #a78bfa;
    margin-bottom: 1;
}

#wordmark {
    padding: 1 4;
}

/* Composer */
#composer {
    height: 3;
    padding: 0 4;
    background: rgba(0, 0, 0, 0.3);
    border-top: solid rgba(255, 255, 255, 0.1);
}

.composer-prompt {
    color: #00ff9f;
    text-style: bold;
    width: 4;
    content-align: center middle;
}

.composer-placeholder {
    color: #64748b;
    content-align: left middle;
}

.active-spinner {
    color: #00ff9f;
}
"""

MARKDOWN_CONTENT = """Here is the vulnerability analysis based on the scan.

The primary issue is a hardcoded password found in the authentication module. You should never store secrets directly in the source code.

```python
def authenticate(username):
    # CRITICAL VULNERABILITY
    if username == "admin":
        return "12345password" 
    return None
```

I will now write a patch to load this from an environment variable using `os.environ` instead.
"""

class Screen2_ActiveTurn(App):
    CSS = TCSS

    def compose(self) -> ComposeResult:
        # Wordmark
        yield Static(create_gradient_text("TAMIAS AGENT", "#00b8ff", "#00ff9f"), id="wordmark")
        
        with ScrollableContainer(id="transcript-area"):
            # Turn 1: User
            with Horizontal(classes="turn"):
                yield Static("❯", classes="gutter-icon user-icon")
                yield Static("Scan the project for security vulnerabilities and fix them.", classes="user-turn")
            
            # Turn 1: Agent
            with Horizontal(classes="turn"):
                yield Static("⬡", classes="gutter-icon agent-icon")
                with Vertical():
                    # Plan Overlay
                    with Vertical(classes="plan-card"):
                        yield Static("Execution Plan", classes="plan-title")
                        yield Static("[x] Run grep_search for 'password'\n[/] Analyze results\n[ ] Generate patch")
                    
                    # Tool 1: Finished
                    with Vertical(classes="tool-card"):
                        with Horizontal(classes="tool-header"):
                            yield Static("grep_search", classes="tool-name")
                            yield Static("✓ 1.2s", classes="tool-status")
                        yield Static("{'Query': 'password'}", classes="tool-args")
                        
                    # Agent Prose
                    yield Static(RichMarkdown(MARKDOWN_CONTENT), classes="agent-turn")
                    
                    # Tool 2: Active / Running
                    with Vertical(classes="tool-card tool-active"):
                        with Horizontal(classes="tool-header"):
                            yield Static("write_to_file", classes="tool-name")
                            yield Static("⠧ running...", classes="tool-status active-spinner")
                        yield Static("{'TargetFile': 'auth.py', 'Content': '...'}", classes="tool-args")

        # Composer
        with Horizontal(id="composer"):
            yield Static("│ ❯ ", classes="composer-prompt")
            yield Static("type your message here...", classes="composer-placeholder")

if __name__ == "__main__":
    app = Screen2_ActiveTurn()
    app.run(inline=True)
