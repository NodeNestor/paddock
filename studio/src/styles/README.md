# Studio CSS

Four global files, then per-component scoped styles. Nothing else is global.

| file | owns | loaded by |
|---|---|---|
| `variables.css` | design tokens only - colours, fonts, radii, layout constants, both themes | `base.css` |
| `base.css` | reset, document defaults, scrollbars, selection, focus ring | `main.ts` |
| `components.css` | the shared primitives: `.pk-btn*`, `.pk-icon-btn`, `.pk-input*` | `main.ts` |
| `markstream-overrides.css` | dark-theme corrections for markstream's hardcoded-light blocks | `lib/markstream.ts` |

Everything else lives in the component that renders it, `<style scoped>`.

## The rule that keeps this from rotting

**A class belongs in `components.css` only if two or more unrelated components
use it and no component owns it.** The moment a component owns the markup, the
styles go with it.

This is not tidiness. `components.css` and a component's own `<style>` are both
global, at the same specificity, so a name in both is decided by bundle order -
and the loser's leftover properties still apply. Three of these had accumulated
and were removed:

- `.pk-tabs` / `.pk-tab*` - `ui/Tabs.vue` owns tabs and sets `gap: 4px`;
  `components.css` was still contributing `gap: 0`, `padding: 0 8px` and
  `overflow-x: auto` that nothing asked for.
- `.pk-resizer*` - `ui/ResizeHandle.vue` owns the handle and defines its own
  `::before`/`::after`; the global copy fought them, and its `--h`/`--v`
  variants were referenced by nothing.
- `.pk-tooltip` - superseded by `.pk-tip` in `ui/Tooltip.vue`.

Same failure mode, already paid for once: the legacy native-`<select>` block
used the same `.pk-select` class as the Reka wrapper, painted a **second**
chevron onto every trigger, and the ghost-hover `background` shorthand then
wiped one - the "one arrow disappears on hover" bug.

Removed at the same time because nothing referenced them at all: `.pk-badge`,
`.pk-panel`, `.pk-spinner` (+ `@keyframes pk-spin`), `.pk-dot*`,
`.pk-modal-overlay` and its transitions (`ui/Dialog.vue` is Reka-backed and
owns `.pk-dialog__*`), `.pk-btn--icon`, `.pk-icon-btn--active`.

## Unscoped `<style>` in a component

Legitimate in exactly one case: **Reka portals the element to `<body>`**, so a
scoped hash would never reach it. Those blocks prefix every class with the
component's own namespace (`pk-dialog`, `pk-menu`, `pk-pop`, `pk-select`,
`pk-tip`) and are the only global styles outside this directory.

`ui/`: Dialog, MenuContent, MenuItem, MenuSeparator, Popover, Select, Toaster,
Tooltip, Tabs. Plus the three panes that host portalled third-party content
(DocxPane, FilePreview, InsightPane).

Anything else unscoped is a bug.

## Tokens

Never hardcode a colour in a `<style>` block - use a `--pk-*` token so both
themes track. The exception is a chart **option object in JS** (ECharts takes
literal colours, not `var()`); those read the theme through
`composables/useTheme` and are the only place literals are correct.

Layout constants that two components must agree on are tokens too:

- `--pk-header-height: 48px` and `--pk-activitybar-width` derived from it -
  one fact so they cannot drift. The brand mark sits where they meet, and at
  55 × 44 it was a rectangle pretending to be a logo tile.
  48 fits both jobs: a 40px icon target keeps a 4px gutter, and the header
  gains the 4px it was slightly starved of.
- `--pk-panel-width: 960px` - one content width for every panel
  (Models/MCP/Prompts/Settings).
- `--pk-chat-width: 818px` - the chat column: the thread, the composer, and
  any notice inside the composer, which must match `composer__box` and so
  subtracts the composer's own 24px side padding. It is a token because
  it had been three hard-coded 768s plus a fourth derived
  from them, and widening it to stop the tool row overflowing once every icon
  carried a label meant editing all four in step or leaving the composer
  standing proud of the messages above it.

## What actually reaches a user's browser

Measured against the built output: **no comment in `src/` ships.**
esbuild's CSS minifier strips `/* */`, Vue's production compiler drops
template `<!-- -->`, esbuild strips JS comments, and no source maps are
emitted - `grep '/*' index-*.css` and `grep '2026-' index-*.js` both
return 0.

`index.html` is the exception: it is served verbatim, so a comment there is
the one that genuinely is disclosed. It was the only leak in the whole bundle,
and it is now guarded.

`<!-- -->` still never goes in a `<template>` - rationale goes in `<script>`
or in the commit (rule). That rule is about the markup an end user
could meet, not about bundle size.

## Guards

`npm run build` runs `scripts/check-shipped-ui.mjs`, which fails on any of:

- **a native `title=` tooltip in a template.** `ui/Tooltip.vue` is the only
  tooltip in the Studio; `title` on a lowercase tag is a browser tooltip we
  cannot theme, delay, or place. `iframe` is exempt (there `title` is the
  required accessible name) and uppercase tags are components where `title`
  is an ordinary prop.
- **an HTML comment in `index.html`**, per above.
- **a hand-written widget role** (`role="radio"`, `"tablist"`, `"checkbox"`,
  `"progressbar"`...) outside `components/ui/`. See `components/ui/README.md` -
  the role names a contract Reka keeps and hand-written markup does not.

Run it alone with `npm run lint:shipped-ui`.
