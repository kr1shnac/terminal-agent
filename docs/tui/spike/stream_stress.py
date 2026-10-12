import asyncio
import time
import os
from textual.app import App, ComposeResult
from textual.widgets import RichLog, Header, Footer
from textual import work

MARKDOWN_CHUNK = """
## Subsection {i}
Here is a list of things:
* Item 1
* Item 2
* Item 3

And here is some code:
```python
def foo(x):
    return x * {i}
```
"""

class StreamStress(App):
    CSS = "RichLog { height: 100%; }"
    
    def compose(self) -> ComposeResult:
        yield Header()
        self.log_widget = RichLog(highlight=True, markup=True, wrap=True)
        yield self.log_widget
        yield Footer()

    def on_mount(self) -> None:
        self.start_time = time.time()
        self.frame_times = []
        self.stream_data()

    @work(thread=True)
    def stream_data(self) -> None:
        # Generate 10k tokens (~40,000 chars)
        total_chunks = 200 # 200 * ~200 chars = 40,000 chars
        
        for i in range(total_chunks):
            start = time.perf_counter()
            
            chunk = MARKDOWN_CHUNK.format(i=i)
            # Safe call to main thread
            self.call_from_thread(self.log_widget.write, chunk)
            
            # Simulate 30 fps network latency (33ms per chunk)
            time.sleep(0.033)
            
            end = time.perf_counter()
            self.frame_times.append((end - start) * 1000) # in ms

        # Calculate stats
        self.frame_times.sort()
        p50 = self.frame_times[len(self.frame_times)//2]
        p95 = self.frame_times[int(len(self.frame_times)*0.95)]
        
        try:
            cpu_usage = float(os.popen(f"ps -p {os.getpid()} -o %cpu=").read().strip())
        except Exception:
            cpu_usage = 0.0
        
        stats = f"\n\n[bold green]STRESS TEST COMPLETE[/bold green]\n" \
                f"p50 frame time: {p50:.2f} ms\n" \
                f"p95 frame time: {p95:.2f} ms\n" \
                f"CPU usage at end: {cpu_usage:.1f}%\n"
                
        self.call_from_thread(self.log_widget.write, stats)

if __name__ == "__main__":
    app = StreamStress()
    app.run()
