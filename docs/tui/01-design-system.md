# Phase 1: Design System

## 1. Design Tokens (TCSS)

The UI is strictly built from design tokens mapped to TCSS variables. No hardcoded colors exist in the components.

```css
/* Core Structural Tokens */
$padding-base: 1 2;
$radius-base: round;
$z-base: 10;
$z-raised: 20;
$z-overlay: 30;

/* Translucency Tokens (Glass) */
$glass-opacity-base: 5%;
$glass-opacity-raised: 15%;
$glass-specular: 40%;
```

## 2. The Themes (Color Palettes)

Themes are implemented via TCSS overriding the base `$accent`, `$success`, `$warning`, and `$error` variables.

### Aurora (Default, Cool Blue-Violet Glass)
```css
$accent-start: #00b8ff;
$accent-end: #8b5cf6;
$success: #00ff9f;
$warning: #f59e0b;
$error: #ff003c;
$base-bg: #0f111a;
$text-main: #f8fafc;
$text-dim: #94a3b8;
```

### Ember (Warm, Terracotta/Orange Glass)
```css
$accent-start: #f97316;
$accent-end: #ef4444;
$success: #10b981;
$warning: #eab308;
$error: #b91c1c;
$base-bg: #1c1917;
$text-main: #fafaf9;
$text-dim: #a8a29e;
```

### Mono (High Contrast, Accessible, No Gradients)
```css
$accent-start: #ffffff;
$accent-end: #ffffff;
$success: #ffffff;
$warning: #ffffff;
$error: #ffffff;
$base-bg: #000000;
$text-main: #ffffff;
$text-dim: #888888;
```

## 3. The "Liquid Glass" Render Pipeline

A terminal cannot natively blur background layers (like CSS `backdrop-filter: blur`). We construct the illusion of liquid glass using:

1. **Layer Hierarchy**: 
   - `Screen` (layer: base)
   - `Panel` (layer: raised, `background: rgba(255,255,255, $glass-opacity-base)`)
   - `Modal` (layer: overlay)
2. **Specular Highlights**: 
   - Focused panels use a brighter border color natively.
   - The top edge of a panel receives a lighter `border-top: round rgba(255,255,255, $glass-specular)` to mimic light hitting a curved glass edge.
3. **Half-Block Shadows**:
   - `▀ ▄ █ ▌ ▐` are used in custom renderables if pseudo-drop-shadows are needed beneath modals.
4. **Solid Mode Fallback**: 
   - `--solid` flag ignores `$glass-opacity-*` and renders solid hues (e.g., `#1e293b`).

## 4. Capability Degradation

The system utilizes `rich`'s automatic capability detection mapping, combined with custom degradation paths:
- **Truecolor (Tier 1)**: Full `$accent-start` to `$accent-end` gradient interpolation across wordmarks and progress bars.
- **256/16 Colors (Tier 2/3)**: Gradients are disabled. `$accent-start` is applied universally. Alpha channels (`rgba()`) fallback to their nearest opaque pre-computed color.
- **No Color**: Controlled via `NO_COLOR=1`. Strips all style attributes; hierarchy maintained purely by bold/italic rendering.

## 5. Typography and Motion

- **Typography**: 
  - Primary text: Normal weight, `$text-main`.
  - Secondary/metadata: Dim weight, `$text-dim`.
  - Keys/Shortcuts: Italicized or enclosed in `<kbd>` style reverse-video blocks.
- **Motion**:
  - Max frame rate: ~30fps to avoid SSH choking.
  - Spinner: A single tastefully sweeping `dots2` (braille) or a custom block-fill sweep.
  - Toggled off globally if `--reduce-motion` is supplied.
