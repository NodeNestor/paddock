// Task tags: the canned instructions a model's chat template expands.
//
// granite-vision is the first model we serve with these. Putting `<chart2csv>`
// in a message makes its template DISCARD the message and substitute IBM's
// exact 300-character extraction prompt, which the model was tuned on. Asking
// the same thing in your own words lands on the open-ended path IBM's own model
// card warns may not generalize well. So these are not shortcuts - for chart
// and table extraction they are the interface, and without a UI the only way in
// is to know the literal and type it.
//
// The list itself comes from the endpoint (/api/server `task_tags`), read out
// of the loaded template, so nothing here decides which tags exist. This file
// only decides what to call them.

export interface TaskTag {
  /** the literal the template matches, angle brackets included */
  tag: string
  /** exactly what the model receives instead of the user's message */
  prompt: string
}

/** Plain-language names for the tags we know. The tag itself is never shown:
 *  a person extracting a table should not have to learn a markup token to do
 *  it. An unknown tag falls back to `humanize` rather than being hidden. */
const LABELS: Record<string, string> = {
  '<chart2csv>': 'Chart to CSV',
  '<chart2code>': 'Chart to code',
  '<chart2summary>': 'Summarize chart',
  '<tables_json>': 'Table to JSON',
  '<tables_html>': 'Table to HTML',
  '<tables_otsl>': 'Table to OTSL',
}

const TAG_RE = /^<[a-z0-9_]+>$/

const ACRONYMS = new Set(['csv', 'json', 'html', 'otsl', 'xml', 'svg', 'yaml', 'md', 'pdf'])

/** Best effort name for a tag nothing here has curated copy for - a new model
 *  shipping `<tables_xml>` should still get a usable button. */
function humanize(tag: string): string {
  const words = tag
    .replace(/^</, '')
    .replace(/>$/, '')
    .split(/[_\s]+/)
    // `chart2csv` is the family's own "X to Y" spelling
    .flatMap((w) => {
      const m = /^(.+?)2(.+)$/.exec(w)
      return m ? [m[1], 'to', m[2]] : [w]
    })
    .filter(Boolean)
    .map((w) => (ACRONYMS.has(w) ? w.toUpperCase() : w))
  if (!words.length) return tag
  words[0] = words[0].charAt(0).toUpperCase() + words[0].slice(1)
  return words.join(' ')
}

/** What to call a task tag, or null when this text isn't one.
 *
 *  A curated name is proof enough on its own; anything else has to be in the
 *  endpoint's advertised list. That second gate is what keeps a user who typed
 *  `<image>` (which granite's template tests for exactly like a task tag, but
 *  passes straight through) from seeing their message rendered as an action
 *  they never ran. */
export function taskLabel(text: string, advertised?: TaskTag[]): string | null {
  const t = text.trim()
  if (!TAG_RE.test(t)) return null
  if (LABELS[t]) return LABELS[t]
  return advertised?.some((x) => x.tag === t) ? humanize(t) : null
}

/** Is this turn a task-tag ACTION rather than a conversation?
 *
 *  Matters because a task tag and a tool list are two contradictory jobs. The
 *  tag tells the model "your entire output is this extraction"; the tools tell
 *  it "you may call one of these". Given both, granite-vision fuses them and
 *  emits a tool call as its extraction OUTPUT - a bare
 *  `{"name": ..., "arguments": ...}` blob in the chat, with no `<tool_call>`
 * wrapper, because it was never invoking anything.
 *
 *  Measured on granite-vision-4.1-4b, temp 0, one image, all six tags. With
 *  tools declared: five produce the fused blob and `<chart2summary>` calls
 *  artifact_create instead of summarising. Without: every one returns its
 *  proper CSV / code / JSON / HTML / OTSL. So this is the whole feature, not
 *  one tag.
 *
 *  Reads the text the same way `taskLabel` does - a curated name, or the
 *  endpoint's own advertised list - so a tag the model does not actually have
 *  is treated as ordinary text and keeps its tools. */
export function isTaskTurn(text: string, advertised?: TaskTag[]): boolean {
  return taskLabel(text, advertised) !== null
}

/** One line of what the model will actually be asked, for the button's
 *  tooltip. Taken from the endpoint's own copy of the prompt rather than
 *  written here, so it can't describe an instruction the model isn't getting. */
export function taskHint(prompt: string): string {
  const line = prompt.split('\n')[0].trim().replace(/\s+/g, ' ')
  return line.length > 150 ? `${line.slice(0, 147)}...` : line
}
