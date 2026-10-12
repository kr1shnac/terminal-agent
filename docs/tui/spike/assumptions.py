import sys
import os
import time
from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static

class InlineTest(App):
    CSS = """
    Screen { background: transparent; }
    """
    def compose(self) -> ComposeResult:
        yield Static("Textual Inline Mode Test", id="title")
        self.log_widget = RichLog(highlight=True, markup=True)
        yield self.log_widget

    def on_mount(self) -> None:
        self.log_widget.write("Streaming 10 lines to test inline mode...")
        for i in range(10):
            self.log_widget.write(f"Line {i} - OSC 11 background transparency check.")
        
        # Test if DEC 2026 is supported by the terminal by checking env or textual features
        self.log_widget.write(f"Terminal capabilities: {self.app.features}")
        
        # Auto-exit for automated testing
        self.set_timer(2, self.exit)

if __name__ == "__main__":
    import sys
    # Run inline if requested
    inline = "--inline" in sys.argv
    app = InlineTest()
    app.run(inline=inline)
