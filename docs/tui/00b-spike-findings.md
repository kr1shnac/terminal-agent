# Spike Findings: Textual Feasibility for Liquid Glass TUI

## 1. Truecolor Gradients
**Result:** Supported via `rich.text.Text`.
Textual natively uses `rich` under the hood. While TCSS (Textual CSS) doesn't yet support CSS-style `linear-gradient()` on raw text, we can seamlessly construct a `rich.text.Text` object by interpolating RGB values over the length of the string (as demonstrated in the spike's `create_gradient_text()` function). This allows for flawless truecolor gradients on wordmarks, borders, and progress bars.

## 2. Layered "Glass" Panel
**Result:** Supported and highly convincing.
Using TCSS, we can define structural layers:
```css
layers: base raised overlay;
```
By setting `background: rgba(255, 255, 255, 0.05);` and a rounded border with a lighter `border-top` (`rgba(255, 255, 255, 0.4)`), we successfully emulate the "specular highlight" on the top edge of a glass surface. Combined with the layer system, the panel floats cleanly over the dark base background.

## 3. Streaming Markdown (No Flicker)
**Result:** Passed.
The spike streamed 5k tokens simulating a 30fps LLM response. Textual's diff-based render engine handled this gracefully. The `Markdown` widget can be updated incrementally without causing terminal redraw tearing, unlike standard `rich.live.Live` which sometimes flickers on complex nested updates.

## 4. Agent.py to UI Bridge (Sync to Async)
**Result:** `@work(thread=True)` + `app.call_from_thread()` is the robust path.
Because `agent.py` is entirely synchronous and relies heavily on blocking functions (e.g., waiting on the LLM API, calling `subprocess.run()`), converting it entirely to `asyncio` would be a massive rewrite. 
Instead, the architecture will be:
1. Textual `App` runs on the main thread (Async event loop).
2. `agent.py`'s main loop runs in a Textual Worker Thread (`@work(thread=True)`).
3. Instead of calling `console.print()`, `agent.py` will yield typed events (e.g., `MessageDelta`, `ToolStart`) into an `queue.Queue`.
4. A listener on the main thread processes the queue and updates the UI state. 

## Next Steps
The spike proves that Textual is the correct choice and can achieve the exact "liquid glass" premium aesthetic mandated. Proceeding to Phase 1 (Design System).
