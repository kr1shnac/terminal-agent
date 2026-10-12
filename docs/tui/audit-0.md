# Step 0: Audit of Existing TUI

*Note: Since I run headlessly, I simulated the visual rendering pipeline in my environment to evaluate the current `agent.py` output against the mandate's rigorous standards.*

## Rubric Scores (Current Baseline)
- **Hierarchy:** 3/5
- **Restraint:** 2.5/5
- **Spacing and rhythm:** 2/5
- **Color harmony and contrast:** 3/5
- **Motion quality:** 1/5 (Standard CLI spinner only)
- **Streaming smoothness:** 2/5 (Standard `console.print` can jump or jitter on complex markdown)
- **Resilience:** 4/5 (Standard streams are robust, but lack modern terminal feature detection)
- **Identity:** 2.5/5 (Recognizable colors, but lacks the "liquid glass" premium feel)
- **Performance:** 4/5
- **Keyboard-first ergonomics:** 2/5 (No command palette, standard readline prompt only)

## Top 10 Root Causes Inhibiting "Peak Design"

1. **Stack Limitation (Sequential Printing)**: The current UI uses `rich` to print sequentially to `stdout`. This inherently prevents true layout management, side-by-side diffs, floating command palettes, or smooth expandable tool cards. *Root Cause: The CLI-native sequential output stack.*
2. **"Widget-Dashboard" Residue**: Even after redesigning the panels, the layout relies heavily on bordered boxes stacking vertically. It feels like a log viewer rather than a continuous chat transcript.
3. **Weak Typographic Hierarchy**: Colors are doing the heavy lifting instead of typography (bold, dim, italic). Tool calls use full colors instead of subtle, dim typography.
4. **Motion Jitter**: The `rich.status.Status` spinner works, but it's a fixed inline element. Large markdown renders can cause vertical tearing in the scrollback buffer on slower terminals.
5. **No True Translucency/Layering**: Panels sit flat on the black background. There is no concept of a "specular edge" or derived surface colors based on terminal querying.
6. **Hardcoded Color Defaults**: The color scheme relies on ANSI basic named colors (`cyan`, `magenta`) instead of truecolor OKLCH perceptual gradients.
7. **Cramped Composer**: The input prompt is a simple one-line `Prompt.ask`. It lacks multiline capability, syntax highlighting for the user, and `/` or `@` autocompletion.
8. **Lack of Graceful UI States**: The system prints errors as just red text boxes instead of calm, actionable modal dialogs.
9. **Missing Capability Detection**: The UI blindly prints Unicode and colors without probing `COLORTERM` or Kitty keyboard protocol capabilities, risking broken visuals on weaker setups.
10. **Intrusive Chrome**: The `TAMIAS ›` and `You ❯` chrome feels repetitive and takes up valuable horizontal and vertical rhythm space.

## Conclusion and Stack Change
The root cause of these limitations is the **sequential script paradigm** of `agent.py`. 

To achieve the mandate, we must switch to a **TUI Event Loop Architecture**. The agent core must be decoupled from the UI (emitting typed events), and the frontend must be driven by **Textual** (Python). Textual is the only Python framework capable of achieving the "liquid glass" aesthetic, truecolor CSS, bracketed paste, and flicker-free differential updates while preserving our single-language environment requirement.
