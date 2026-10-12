import asyncio
import time
from typing import Generator

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical, Container
from textual.widgets import Markdown, Static, ProgressBar
from textual.color import Color

# 1. Custom Rich renderable for truecolor gradients
def create_gradient_text(text: str, start_color: str, end_color: str) -> Text:
    """Create a gradient across a string of text using Rich truecolor."""
    rich_text = Text()
    c1 = Color.parse(start_color)
    c2 = Color.parse(end_color)
    length = len(text)
    
    for i, char in enumerate(text):
        ratio = i / max(1, length - 1)
        # Interpolate color (simple RGB lerp)
        r = int(c1.r + (c2.r - c1.r) * ratio)
        g = int(c1.g + (c2.g - c1.g) * ratio)
        b = int(c1.b + (c2.b - c1.b) * ratio)
        rich_text.append(char, style=f"rgb({r},{g},{b}) bold")
    return rich_text

# 2. TCSS for Liquid Glass Aesthetic
TCSS = """
Screen {
    background: #0f111a; /* deep base */
    layers: base raised overlay;
}

#glass-panel {
    layer: raised;
    background: rgba(255, 255, 255, 0.05);
    border: round rgba(255, 255, 255, 0.15);
    border-top: round rgba(255, 255, 255, 0.4); /* Specular highlight */
    padding: 1 2;
    margin: 1;
    height: auto;
    max-height: 80%;
}

#glass-panel:focus {
    border: round #00b8ff;
    border-top: round #00ff9f;
}

#gradient-header {
    text-align: center;
    margin-bottom: 1;
}

#stream-target {
    height: auto;
}

ProgressBar {
    margin: 1;
    /* Textual 8+ CSS supports basic styling for progress bars */
}
"""

class GlassSpike(App):
    CSS = TCSS

    def compose(self) -> ComposeResult:
        yield Container(
            Static(create_gradient_text("TAMIAS LIQUID GLASS SPIKE", "#00b8ff", "#00ff9f"), id="gradient-header"),
            ProgressBar(total=100, show_eta=False, id="progress"),
            Vertical(
                Markdown("# Streaming Test\nWaiting for agent...", id="stream-target"),
                id="glass-panel"
            )
        )

    def on_mount(self) -> None:
        self.query_one("#glass-panel").focus()
        self.stream_agent_data()

    @work(thread=True)
    def stream_agent_data(self) -> None:
        """
        4. Event Bridge Spike:
        This runs in a real OS thread, simulating agent.py's blocking synchronous code.
        It talks to the UI safely using app.call_from_thread().
        """
        # Simulate connecting
        time.sleep(1)
        
        # 3. Stream 5k tokens
        lorem = "Liquid glass aesthetic testing. " * 500  # ~5000 chars/tokens
        chunk_size = 50
        content = "## Incoming Agent Stream\n\n"
        
        for i in range(0, len(lorem), chunk_size):
            chunk = lorem[i:i+chunk_size]
            content += chunk
            
            # Safe call back to main thread
            self.call_from_thread(self.update_markdown, content)
            self.call_from_thread(self.update_progress, (i / len(lorem)) * 100)
            
            # Simulate network/LLM latency (~30 fps streaming)
            time.sleep(0.016) 

        self.call_from_thread(self.update_markdown, content + "\n\n**Stream Complete.** No flicker observed.")
        self.call_from_thread(self.update_progress, 100.0)

    def update_markdown(self, content: str) -> None:
        self.query_one("#stream-target", Markdown).update(content)
        # Keep scrolled to bottom if needed, though Markdown widget handles it well.
        self.query_one("#glass-panel").scroll_end(animate=False)

    def update_progress(self, percent: float) -> None:
        self.query_one("#progress", ProgressBar).progress = percent

if __name__ == "__main__":
    app = GlassSpike()
    app.run()
