// Monaco, loaded lazily and dressed in the Studio's own palette.
//
// This module is only ever reached through `await import('@/lib/monaco')`, so
// everything it pulls in lands in a chunk that is fetched the first time
// somebody opens a source view - Monaco is several megabytes and the chat must
// not pay for it.
//
// Language SERVICES (IntelliSense, validation) are deliberately not wired - see
// monaco-entry.ts for the accounting; the short version is 8.9 MB of web
// workers inside paddock.exe forever, 6.7 of it the TypeScript compiler, to get
// completions in a side panel. What we do want, correct highlighting for
// whatever language the artifact claims, comes from the tokenizers, which run
// on the main thread and cost nothing. Monaco still wants its base worker for
// diffing and word completions, and that one is small.
import * as monaco from './monaco-entry'
import EditorWorker from 'monaco-vs/editor/editor.worker?worker'

export type Monaco = typeof monaco

/** A theme name of our own so re-defining it re-skins every live editor. */
const THEME = 'paddock'

let ready: Promise<Monaco> | null = null

/** Monaco reads this global to spawn workers; Vite gives us real worker URLs. */
function installEnvironment(): void {
  const w = self as unknown as { MonacoEnvironment?: unknown }
  w.MonacoEnvironment = { getWorker: () => new EditorWorker() }
}

function css(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}

/** Monaco's TOKEN theme parses colours with `/^#?([0-9A-Fa-f]{6})([0-9A-Fa-f]{2})?$/`
 *  - six digits or eight, shorthand rejected - and `editor.foreground` /
 *  `editor.background` are folded into a token rule, so they go through it too
 *  (standaloneThemeService, "Pick up default colors from ...").
 *
 *  This bit us in the built Studio only: we author
 *  `--pk-bg-surface: #FFFFFF`, the CSS minifier ships `#fff`, and light mode
 *  died on "Illegal value for token color: #fff" with the editor never
 *  loading. `npm run dev` serves the long form and is fine, which is why it
 *  survived. So expand shorthand rather than trusting what a stylesheet
 *  happens to be written as - and rgba() still converts rather than dropping
 *  the rule. */
function hex(value: string, fallback: string): string {
  const v = value.trim()
  if (v.startsWith('#')) {
    const d = v.slice(1)
    // #rgb / #rgba -> double every digit; the 6/8 forms already parse
    if (d.length === 3 || d.length === 4) {
      return `#${[...d].map((c) => c + c).join('')}`
    }
    return d.length === 6 || d.length === 8 ? v : fallback
  }
  const m = v.match(/^rgba?\(([^)]+)\)$/i)
  if (!m) return fallback
  const parts = m[1]!.split(/[,/\s]+/).filter(Boolean)
  const [r, g, b] = parts.slice(0, 3).map((p) => Math.round(parseFloat(p)))
  if (r === undefined || g === undefined || b === undefined) return fallback
  const a = parts[3] === undefined ? 1 : parseFloat(parts[3])
  const byte = (n: number): string => Math.max(0, Math.min(255, n)).toString(16).padStart(2, '0')
  const alpha = a >= 1 ? '' : byte(Math.round(a * 255))
  return `#${byte(r)}${byte(g)}${byte(b)}${alpha}`
}

/** Re-skin from the live CSS variables. Called on load and on every theme
 *  flip, because the variables themselves change under `data-theme`. */
export function applyTheme(dark: boolean): void {
  const surface = hex(css('--pk-bg-surface', '#121C26'), '#121C26')
  const base = hex(css('--pk-bg-base', '#0A1118'), '#0A1118')
  const elevated = hex(css('--pk-bg-elevated', '#1A2834'), '#1A2834')
  const text = hex(css('--pk-text-primary', '#ECE8E0'), '#ECE8E0')
  const muted = hex(css('--pk-text-muted', '#5A6A78'), '#5A6A78')
  const accent = hex(css('--pk-accent', '#0EA5E9'), '#0EA5E9')
  monaco.editor.defineTheme(THEME, {
    base: dark ? 'vs-dark' : 'vs',
    // Keep the stock token colours - they are tuned per language and we have no
    // palette of our own for syntax. Only the chrome follows the Studio.
    inherit: true,
    rules: [],
    colors: {
      'editor.background': surface,
      'editor.foreground': text,
      'editorGutter.background': surface,
      'editorLineNumber.foreground': muted,
      'editorLineNumber.activeForeground': text,
      'editorCursor.foreground': accent,
      'editorWidget.background': elevated,
      'editorSuggestWidget.background': elevated,
      'input.background': base,
      'focusBorder': accent,
    },
  })
  monaco.editor.setTheme(THEME)
}

export async function loadMonaco(dark: boolean): Promise<Monaco> {
  if (!ready) {
    installEnvironment()
    applyTheme(dark)
    ready = Promise.resolve(monaco)
  }
  return ready
}

/** The artifact's `language` is whatever the model called it ("js", "py",
 *  "svg"). Monaco's own registry already knows the aliases and extensions, so
 *  ask it rather than keeping a hand-written map that rots. */
export function resolveLanguage(want: string | undefined | null): string {
  const w = (want ?? '').trim().toLowerCase()
  if (!w) return 'plaintext'
  for (const l of monaco.languages.getLanguages()) {
    if (l.id.toLowerCase() === w) return l.id
    if (l.aliases?.some((a) => a.toLowerCase() === w)) return l.id
    if (l.extensions?.some((e) => e.toLowerCase() === `.${w}`)) return l.id
  }
  return 'plaintext'
}

export { monaco }
