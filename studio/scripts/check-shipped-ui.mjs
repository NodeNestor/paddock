// Three things that must never reach a user's browser, checked in
// `npm run build` because they all fail silently rather than visibly.
//
// 1. A native `title=` tooltip in an SFC template. Reka's <Tooltip> is the only
//    tooltip in the Studio (ui/Tooltip.vue): shared delay, themed,
//    collision-aware, touch-capable. A native title= is none of those and
//    cannot be styled.
//
//    The rule keys off Vue's own casing convention: a lowercase tag is a DOM
//    element, so `title` lands on it as the browser attribute. An uppercase tag
//    is a component, where `title` is an ordinary prop (Dialog, ConfirmDialog)
//    and means nothing to the browser. `iframe` is exempt because there `title`
//    is the required accessible name (WCAG 4.1.2), not a tooltip.
//
// 2. An HTML comment in index.html. Vue drops template comments and the
//    minifier strips CSS/JS ones, so SFC comments never ship - but index.html
//    is served verbatim, which makes it the one file where a comment really is
//    disclosed (measured: it was the only leak in the whole bundle).
//
// 3. A hand-written ARIA widget role. Writing role="radio" or role="tablist"
//    by hand gets you the NAME of a widget and none of its behaviour - the
//    quality cards had role="radio" with no roving focus and no arrow keys,
//    the area switcher had role="tablist" with no tabpanel, and the sidebar's
//    checkboxes were <span>s with no role at all. Reka ships every one of
//    these; the wrappers live in components/ui/, which is why that directory
//    is the only place the roles may appear (in prose, explaining what was
//    replaced). Roles that describe STRUCTURE rather than a widget (list,
//    status, img, separator, alert...) are not listed here and stay fine.
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const SRC = fileURLToPath(new URL('../src', import.meta.url))
const INDEX_HTML = fileURLToPath(new URL('../index.html', import.meta.url))
const EXEMPT_TAGS = new Set(['iframe'])

// role -> the ui/ wrapper that owns it
const WIDGET_ROLES = new Map([
  ['radio', 'RadioGroup + RadioItem'],
  ['radiogroup', 'RadioGroup + RadioItem'],
  ['checkbox', 'Checkbox'],
  ['switch', 'Switch'],
  ['tab', 'Tabs (or ToggleGroup for a mode strip)'],
  ['tablist', 'Tabs (or ToggleGroup for a mode strip)'],
  ['progressbar', 'Progress'],
  ['combobox', 'Select'],
  ['listbox', 'Select'],
  ['option', 'Select'],
  ['menu', 'Menu + MenuContent + MenuItem'],
  ['menuitem', 'Menu + MenuContent + MenuItem'],
  ['menuitemcheckbox', 'Menu + MenuContent + MenuItem'],
  ['menuitemradio', 'Menu + MenuContent + MenuItem'],
  ['slider', 'Slider'],
  ['dialog', 'Dialog'],
  ['alertdialog', 'Dialog role="alertdialog"'],
  ['tooltip', 'Tooltip'],
  ['spinbutton', 'NumberField'],
  // a div that pretends to be a button: use a real <button>
  ['button', 'a real <button> element'],
])
const ROLE_ATTR = /(?:^|\s)role\s*=\s*"([a-z]+)"/

// 4. A caller class on a wrapper whose Reka primitive DROPS the scope
//    attribute. Vue normally puts the caller's `data-v-xxx` on a child
//    component's root, which is what lets `.sf__qcard {}` reach it - but Reka
//    renders these through an asChild/roving-focus clone, so the class lands
//    and the attribute does not. The rule then matches nothing and the element
//    falls back to native chrome. This shipped once and took the
//    Manager's and Studio's buttons with it.
//    Verify the list with `node scripts/probe-reka-scope.mjs` after any
//    reka-ui upgrade - it is a property of their internals, not ours.
const CLONED_WRAPPERS = new Set(['ToggleGroupItem', 'RadioItem', 'Checkbox', 'Switch'])
const CLASS_ATTR = /(?:^|\s)class\s*=\s*"([^"]*)"/

// tag name, then attrs up to the closing angle - quoted values may hold
// angle brackets and are consumed whole so they can't end the match early.
const TAG = /<([a-zA-Z][\w.-]*)((?:[^<>'"]|'[^']*'|"[^"]*")*?)\/?>/gs
const TITLE_ATTR = /(?:^|\s)(?::|v-bind:)?title\s*=/

function vueFiles(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) return vueFiles(path)
    return name.endsWith('.vue') ? [path] : []
  })
}

/** The <template> block only - `title:` inside <script> is unrelated. */
function templateOf(source) {
  const open = source.indexOf('<template>')
  if (open === -1) return null
  const close = source.lastIndexOf('</template>')
  if (close <= open) return null
  return { text: source.slice(open, close), offset: open }
}

/** Scoped <style> text of an SFC (unscoped blocks are already global).
 *  Comments are blanked, not dropped, so offsets stay put - a note that
 *  mentions `.foo {}` as prose must not read as a use of it. */
function scopedStyleOf(source) {
  return [...source.matchAll(/<style[^>]*\bscoped\b[^>]*>([\s\S]*?)<\/style>/g)]
    .map((m) => m[1])
    .join('\n')
    .replace(/\/\*[\s\S]*?\*\//g, (c) => ' '.repeat(c.length))
}
/** Spans covered by a `:deep(...)`, so we can ask whether a use sits inside. */
function deepSpans(css) {
  const spans = []
  for (const m of css.matchAll(/:deep\(/g)) {
    let depth = 1
    let i = m.index + m[0].length
    for (; i < css.length && depth > 0; i++) {
      if (css[i] === '(') depth++
      else if (css[i] === ')') depth--
    }
    spans.push([m.index, i])
  }
  return spans
}

/** Every selector list in the sheet, at any nesting depth (at-rule preludes
 *  skipped). The check is PER BLOCK: a class shared between plain elements and
 *  a Reka item legitimately needs both `.cls` and `:deep(.cls)` - but they must
 *  live in the same block, or one of the two renderings goes unstyled. */
function selectorLists(css) {
  const out = []
  let start = 0
  for (let i = 0; i < css.length; i++) {
    const c = css[i]
    if (c === '{' || c === '}' || c === ';') {
      if (c === '{') {
        const sel = css.slice(start, i)
        if (!sel.trimStart().startsWith('@')) out.push({ text: sel, offset: start })
      }
      start = i + 1
    }
  }
  return out
}

const offenders = []
const roleOffenders = []
const scopeOffenders = []
for (const file of vueFiles(SRC)) {
  const source = readFileSync(file, 'utf8')
  const block = templateOf(source)
  if (!block) continue
  const rel = relative(SRC, file).replaceAll('\\', '/')
  // ui/ IS the place these roles are allowed - that is where the wrappers are
  const isWrapper = rel.startsWith('components/ui/')

  for (const match of block.text.matchAll(TAG)) {
    const [, tag, attrs] = match
    const line = () => source.slice(0, block.offset + match.index).split('\n').length

    if (tag[0] === tag[0].toLowerCase() && !EXEMPT_TAGS.has(tag) && TITLE_ATTR.test(attrs)) {
      offenders.push({ file: rel, line: line(), tag })
    }

    // Same casing rule as `title`: on an uppercase tag `role` is a prop the
    // wrapper interprets (<Dialog role="alertdialog">), not a DOM attribute.
    const role = tag[0] === tag[0].toLowerCase() ? ROLE_ATTR.exec(attrs)?.[1] : undefined
    if (role && !isWrapper && WIDGET_ROLES.has(role)) {
      roleOffenders.push({ file: rel, line: line(), tag, role, use: WIDGET_ROLES.get(role) })
    }

    if (!isWrapper && CLONED_WRAPPERS.has(tag)) {
      const css = scopedStyleOf(source)
      const spans = deepSpans(css)
      const lists = selectorLists(css)
      for (const cls of (CLASS_ATTR.exec(attrs)?.[1] ?? '').split(/\s+/).filter(Boolean)) {
        // NOT \b: a class name may contain `-`, so `.sf__qcard\b` would match
        // inside `.sf__qcard-title` and report a use that isn't one.
        const esc = cls.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
        const re = new RegExp(`\\.${esc}(?![\\w-])`, 'g')
        for (const list of lists) {
          const uses = [...list.text.matchAll(re)].map((u) => u.index + list.offset)
          if (!uses.length) continue
          // this block styles the class: at least one of its selectors must
          // reach the cloned element, i.e. hold the class inside a :deep()
          if (!uses.some((i) => spans.some(([a, b]) => i > a && i < b))) {
            scopeOffenders.push({ file: rel, line: line(), tag, cls })
            break
          }
        }
      }
    }
  }
}

let failed = false

if (offenders.length) {
  failed = true
  console.error(`\nNative title= tooltip in ${offenders.length} place(s):\n`)
  for (const o of offenders) console.error(`  src/${o.file}:${o.line}  <${o.tag} title=...>`)
  console.error(`
Wrap the element in <Tooltip label="..."> instead:

  import Tooltip from '@/components/ui/Tooltip.vue'
  <Tooltip label="Send"><button>...</button></Tooltip>
`)
}

if (roleOffenders.length) {
  failed = true
  console.error(`\nHand-written widget role in ${roleOffenders.length} place(s):\n`)
  for (const o of roleOffenders) {
    console.error(`  src/${o.file}:${o.line}  <${o.tag} role="${o.role}">  ->  use ${o.use}`)
  }
  console.error(`
The role names the widget; Reka supplies the behaviour behind it - roving
focus, arrow keys, aria-checked, the escape/dismiss wiring. Writing the role
by hand claims the contract without keeping it. Wrappers: components/ui/.
`)
}

if (scopeOffenders.length) {
  failed = true
  console.error(
    `\nScoped rule that cannot reach its element in ${scopeOffenders.length} place(s):\n`,
  )
  for (const o of scopeOffenders) {
    console.error(`  src/${o.file}:${o.line}  <${o.tag} class="${o.cls}">  styled as .${o.cls} {}`)
  }
  console.error(`
Reka renders these through an asChild/roving-focus clone, which keeps the
class but DROPS the scope attribute - so the rule matches nothing and the
element falls back to native chrome. Wrap the selector:

  .parent :deep(.${scopeOffenders[0].cls}) { ... }

Re-check which primitives do this with: node scripts/probe-reka-scope.mjs
`)
}

// 4. The Advanced tab's field list vs the runner's Config struct.
//
// AF_CARDS is not just a form definition - it is also the Advanced tab's
// SERIALIZER (`tomlFromForm` writes only these keys), so a key it omits is
// DELETED from servers/<port>.toml by a round trip through that tab. Found
// live: `vram_budget` was missing, and the Simple tab writes it, so
// opening Advanced and saving silently removed the endpoint's VRAM cage. The
// other direction is a lie of a different kind - a field for a key the runner
// would refuse at startup (deny_unknown_fields).
//
// Two files in two languages with no compile-time link between them, so the
// link is here.
const CONFIG_RS = fileURLToPath(
  new URL('../../crates/paddock-runner/src/config.rs', import.meta.url),
)
const SERVER_FORM = join(SRC, 'components/manage/ServerForm.vue')
try {
  const rs = readFileSync(CONFIG_RS, 'utf8')
  const start = rs.indexOf('pub struct Config {')
  const body = rs.slice(start, rs.indexOf('\n}', start))
  const runnerKeys = new Set([...body.matchAll(/^ {4}pub ([a-z0-9_]+):/gm)].map((m) => m[1]))

  const vue = readFileSync(SERVER_FORM, 'utf8')
  const cards = vue.slice(vue.indexOf('const AF_CARDS'), vue.indexOf('const AF_ALL'))
  const formKeys = new Set([...cards.matchAll(/key: '([a-z0-9_]+)'/g)].map((m) => m[1]))

  const missing = [...runnerKeys].filter((k) => !formKeys.has(k))
  const extra = [...formKeys].filter((k) => !runnerKeys.has(k))
  if (!runnerKeys.size || !formKeys.size) {
    failed = true
    console.error(
      `\nConfig-parity check could not read one of its inputs (runner ${runnerKeys.size} keys, form ${formKeys.size}) - the parse anchors moved.\n`,
    )
  } else if (missing.length || extra.length) {
    failed = true
    console.error(`\nAdvanced tab and the runner's Config have drifted apart:\n`)
    for (const k of missing) {
      console.error(`  ${k}  in config.rs, NO field in AF_CARDS - a round trip through Advanced deletes it`)
    }
    for (const k of extra) {
      console.error(`  ${k}  a field in AF_CARDS, NOT in config.rs - the runner refuses it at startup`)
    }
    console.error(`
Add the row to AF_CARDS in src/components/manage/ServerForm.vue (or drop it),
in the same change that touches crates/paddock-runner/src/config.rs.
`)
  }
} catch (e) {
  failed = true
  console.error(`\nConfig-parity check failed to run: ${e.message}\n`)
}

const indexHtml = readFileSync(INDEX_HTML, 'utf8')
const htmlComments = [...indexHtml.matchAll(/<!--[\s\S]*?-->/g)]
if (htmlComments.length) {
  failed = true
  console.error(`\nHTML comment in index.html (served verbatim, so users see it):\n`)
  for (const m of htmlComments) {
    console.error(`  index.html:${indexHtml.slice(0, m.index).split('\n').length}`)
  }
  console.error(`
Move the note inside the <script> instead - JS comments are minified away.
`)
}

if (failed) process.exit(1)
