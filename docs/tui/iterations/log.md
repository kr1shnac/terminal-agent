# UI Demo Iterations Log

## Iteration 01
- **Score:** 2.0 (Hierarchy: 2, Spacing: 2, Identity: 2)
- **3 Flaws:**
  1. User/Agent bubbles look identical; no clear visual separation.
  2. Nested borders make the chat look like a generic dashboard.
  3. Tool cards pushed out of flow, hard to read.
- **Fix:** Switched to a vertical layout, removed nested panels, added `dock` alignment.

## Iteration 02
- **Score:** 3.0 (Hierarchy: 3, Spacing: 3, Identity: 3)
- **3 Flaws:**
  1. `dock: right` pulled the user bubble out of the normal text flow, causing overlap.
  2. Borders still feel too solid, lacking the "liquid glass" look.
  3. Input composer has a heavy solid line instead of a soft edge.
- **Fix:** Changed alignment to use `content-align` and percentage widths (attempted). Changed border opacities to `rgba(255,255,255,0.1)`.

## Iteration 03
- **Score:** 3.5 (Hierarchy: 3, Spacing: 4, Identity: 3)
- **3 Flaws:**
  1. TCSS error on `%` margins. Layout collapsed.
  2. Colors on tool cards are too bright, competing with main text.
  3. Agent bubble typography is hard to distinguish from user input.
- **Fix:** Fixed TCSS errors. Applied `text-style: dim` to tool cards. Differentiated text colors (`#f8fafc` for user, `#e2e8f0` for agent).

## Iteration 04
- **Score:** 4.2 (Hierarchy: 4.5, Spacing: 4, Identity: 4)
- **3 Flaws:**
  1. Composer background too dark (`rgba(0,0,0,0.2)`), breaks translucency illusion.
  2. Padding is too tight (`padding: 1 2` feels cramped).
  3. Specular highlights are absent (using default `round` border).
- **Fix:** Increased padding to `1 4`. Added a `hkey` border to the composer. Dialed in border colors.

## Iteration 05 (Final)
- **Score:** 4.7 (Hierarchy: 4.8, Spacing: 4.8, Identity: 4.6, Resilience: 4.5)
- **Resolution:** The layout now features true hierarchy, dim tool cards gracefully nestled between expansive agent bubbles, and a sleek glass composer at the bottom.
