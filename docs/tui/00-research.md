# Phase 0: Research and Decisions

## 1. Study of Existing TUIs

### Claude Code (Anthropic)
- **Great**: Extremely clean, stream-first approach. Uses React+Ink to render components inline while preserving terminal scrollback. Excellent at collapsing completed tool calls into one-line summaries to prevent clutter.
- **Weak**: Does not feel like a true "application" with persistent spatial zones (status bars, sidebars). Relies heavily on the terminal's default flow, making complex interactions (like full-screen diffs) feel like context switches rather than integrated views.

### OpenCode
- **Great**: Full-screen application feel. Uses a persistent timeline, dedicated input dock, and floating modals for file picking. Excellent visual hierarchy for tool states (pending, running, success, fail).
- **Weak**: Can feel cluttered with metadata. Full-screen mode (alt-screen) traps the user, and older iterations suffered from redraw flicker during rapid streaming.

### Charm's Crush (and Glamour/Lip Gloss ecosystem)
- **Great**: The gold standard for terminal aesthetics. Beautiful, effortless gradients, rounded corners, and soft padding. Embraces truecolor and degrades perfectly gracefully.
- **Weak**: Written in Go (Bubble Tea). While the architecture (Elm architecture) is flawless, binding it to a Python-native agent core requires either rewriting the agent in Go or managing a messy cross-language IPC layer.

### Aider
- **Great**: Rock-solid Prompt Toolkit integration. Great `/` command system and robust file-handling. Very fast.
- **Weak**: Visually spartan. Looks and feels like a standard command-line script rather than a premium "liquid glass" product. Contrast and styling are often hardcoded or limited to standard ANSI 16 colors.

### Gemini CLI / Codex CLI
- **Great**: Fast, raw access to models.
- **Weak**: Almost zero TUI elements. Pure standard I/O streams with basic ANSI colors. Completely lacks interactive tool cards, timelines, or layout management.

## 2. Framework Selection

**Selected Framework: Textual (Python)**

**Justification**:
The agent core (`agent.py`) is written in Python. To achieve a seamless, low-latency, single-process architecture, the TUI must also be in Python.

While `rich` + `prompt_toolkit` is great for simple CLI scripts, achieving the **"liquid glass" aesthetic** (layered surfaces, truecolor gradients, soft glow, rounded overlapping borders) and an **app-like UX** (fixed input composers, floating modals, side-by-side diffs) requires a true UI engine.

**Textual** is the undisputed leader for this in Python:
- **Native to Python**: Direct memory access to the agent's event stream; no IPC overhead.
- **CSS-like Styling (TCSS)**: Crucial for the "liquid glass" requirement. We can define our 3 themes (`aurora`, `ember`, `mono`) purely in CSS, using layers, transparencies (opacity), and transitions.
- **Advanced Capabilities**: Out-of-the-box support for mouse tracking, bracketed paste, terminal resize (SIGWINCH), and truecolor.
- **Flicker-Free**: Uses a highly optimized diff-based rendering engine under the hood to ensure that even at 30fps streaming, the screen does not tear or flicker.

*(Note: To integrate Textual while keeping the strict rules of this repository—which prohibit `pip install` in the venv unless allowed—we will need explicit permission to add `textual` to the dependencies, or we will have to build a pure `rich` + `prompt_toolkit` pseudo-TUI, though Textual is vastly superior for this mandate).*

## 3. Platform Matrix

The TUI will be tested against the following terminal matrix to ensure graceful degradation:

### macOS
- **Ghostty / WezTerm / Kitty**: Tier 1 (Truecolor, smooth resize, full Unicode/Nerd Fonts, bracketed paste, mouse).
- **iTerm2**: Tier 1.
- **Terminal.app**: Tier 3 (256 colors only, basic Unicode, no truecolor).

### Linux
- **Kitty / Alacritty / foot**: Tier 1 (GPU accelerated, flawless truecolor and font rendering).
- **GNOME Terminal**: Tier 2 (Solid truecolor, standard VTE engine quirks).
- **Konsole**: Tier 2.

### Multiplexers & Remote
- **tmux**: Tier 2/3 (Requires `set -g default-terminal "tmux-256color"` and `Tc` overrides for truecolor; we must degrade gracefully if gradients band).
- **SSH**: Must handle higher latency (input latency budget <16ms locally, but render loop must not choke on SSH backpressure).

### Windows
- **Windows Terminal**: Tier 1 (Excellent modern support for truecolor, Unicode, and ConPTY).
