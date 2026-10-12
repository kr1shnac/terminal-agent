import os
import time
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Markdown

FAKE_EVENTS = [
    {"role": "user", "text": "Please scan the project for security vulnerabilities."},
    {"role": "tool", "text": "⮑  grep_search({'Query': 'password'}) ✓ 1.2s"},
    {"role": "agent", "text": "I found some potential security issues.\n```python\npassword = '123'\n```"},
]

def make_app(tcss, out_file):
    class UIDemo(App):
        CSS = tcss
        def compose(self) -> ComposeResult:
            with Vertical(id="transcript"):
                yield Static(FAKE_EVENTS[0]["text"], classes="bubble user")
                yield Static(FAKE_EVENTS[1]["text"], classes="tool-card")
                yield Static(FAKE_EVENTS[2]["text"], classes="bubble agent")
            yield Static("│ ❯ type your message here...", id="composer")

        def on_mount(self):
            self.set_timer(0.2, self.capture)

        def capture(self):
            svg = self.export_screenshot(title="Iteration")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, "w") as f:
                f.write(svg)
            self.exit()
    
    app = UIDemo()
    app.run()

# Iteration 02
tcss_02 = """
Screen { layout: vertical; background: transparent; }
#transcript { height: 1fr; padding: 1 2; }
.bubble { width: auto; max-width: 80%; padding: 1 2; margin-bottom: 1; border: round rgba(255,255,255,0.1); background: rgba(255,255,255,0.05); }
.user { dock: right; border-top: round #00ff9f; }
.agent { dock: left; border-top: round #00b8ff; }
.tool-card { dock: left; color: #94a3b8; margin-left: 2; margin-bottom: 1; }
#composer { height: 3; padding: 1 2; border-top: solid rgba(255,255,255,0.2); }
"""
make_app(tcss_02, "docs/tui/iterations/02/hero2_120.svg")

# Iteration 03 (Fixing right-alignment and bubbles)
tcss_03 = """
Screen { layout: vertical; background: transparent; }
#transcript { height: 1fr; padding: 1 2; }
/* dock: right pulls out of flow, so we use content-align instead for flow */
.bubble { width: auto; padding: 1 2; margin-bottom: 1; border: round rgba(255,255,255,0.1); background: rgba(255,255,255,0.05); }
.user { content-align: right middle; border-top: round #00ff9f; margin-left: 20%; }
.agent { content-align: left middle; border-top: round #00b8ff; margin-right: 20%; }
.tool-card { color: #94a3b8; margin-left: 2; margin-bottom: 1; }
#composer { height: 3; padding: 1 2; border-top: solid rgba(255,255,255,0.2); }
"""
make_app(tcss_03, "docs/tui/iterations/03/hero2_120.svg")

# Iteration 04 (Refining liquid glass and typography)
tcss_04 = """
Screen { layout: vertical; background: transparent; }
#transcript { height: 1fr; padding: 0 2; }
.bubble { width: 100%; padding: 1 2; margin-bottom: 1; border: round rgba(255,255,255,0.15); background: rgba(255,255,255,0.08); }
.user { border-top: round #00ff9f; color: #f8fafc; }
.agent { border-top: round #00b8ff; color: #e2e8f0; }
.tool-card { color: #94a3b8; margin-left: 2; margin-bottom: 1; text-style: dim; }
#composer { height: 3; padding: 1 2; border-top: round rgba(255,255,255,0.3); background: rgba(0,0,0,0.2); }
"""
make_app(tcss_04, "docs/tui/iterations/04/hero2_120.svg")

# Iteration 05 (Peak design: tighter spacing, subtle glow borders)
tcss_05 = """
Screen { layout: vertical; background: transparent; }
#transcript { height: 1fr; padding: 1 4; }
.bubble { padding: 1 2; margin-bottom: 1; border: round rgba(255,255,255,0.1); background: rgba(255,255,255,0.05); }
.user { border-top: round rgba(0,255,159,0.8); color: #f8fafc; text-style: bold; }
.agent { border-top: round rgba(0,184,255,0.8); color: #f1f5f9; }
.tool-card { color: #64748b; margin-left: 1; margin-bottom: 1; text-style: dim; }
#composer { height: 3; padding: 1 4; border-top: hkey rgba(255,255,255,0.1); color: #00ff9f; text-style: bold; }
"""
make_app(tcss_05, "docs/tui/iterations/05/hero2_120.svg")

print("Generated 5 iterations.")
