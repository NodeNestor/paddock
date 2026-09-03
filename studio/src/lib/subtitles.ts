// SRT and WebVTT writers for exporting a transcription turn (the
// host became a chat message rather than a page).
//
// This is a deliberate second implementation of crates/paddock-runner/src/
// subtitles.rs, and the reason is correctness, not convenience. The endpoint
// can return srt/vtt directly, but asking for one would re-run the whole
// transcription: a second GPU decode of a file that may be forty minutes long,
// and - because timestamps change the decode - a subtitle whose words need not
// match the transcript on screen. An export must be what the user is looking at.
//
// The rules below are the Rust file's, quirk for quirk. Its tests pin them;
// the Studio has no test runner, so keep the two in step by hand.
//
//   |                   | SRT  | WebVTT             |
//   | header            | none | `WEBVTT` + a blank |
//   | cue number        | 1-based, required | omitted |
//   | decimal separator | `,`  | `.`                |
//   | text              | plain | `&`, `<`, `>` escaped (it is markup) |

export interface Cue {
  start: number
  end: number
  text: string
}

/** `HH:MM:SS<sep>mmm`, hours always written so a >1 h file does not change
 *  field count mid-document. Rounds to milliseconds first: formatting 59.9996
 *  directly would print 00:00:60.000, which players reject outright. */
export function stamp(t: number, sep: ',' | '.'): string {
  const ms = Math.round(Math.max(0, Number.isFinite(t) ? t : 0) * 1000)
  const p = (n: number, w = 2): string => String(n).padStart(w, '0')
  return `${p(Math.floor(ms / 3_600_000))}:${p(Math.floor(ms / 60_000) % 60)}:${p(
    Math.floor(ms / 1000) % 60,
  )}${sep}${p(ms % 1000, 3)}`
}

/** One cue is one paragraph: a BLANK line inside it terminates the cue in both
 *  formats, so an embedded one would silently split the file's structure. */
function body(text: string): string {
  return text
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
    .join(' ')
}

export function srt(cues: Cue[]): string {
  let out = ''
  // counted on EMITTED cues, not the input index: a skipped blank one would
  // leave a gap, and a file numbered 1, 3 is malformed
  let n = 0
  for (const c of cues) {
    const line = body(c.text)
    if (!line) continue
    n += 1
    out += `${n}\n${stamp(c.start, ',')} --> ${stamp(c.end, ',')}\n${line}\n\n`
  }
  return out
}

export function vtt(cues: Cue[]): string {
  let out = 'WEBVTT\n\n'
  for (const c of cues) {
    const line = body(c.text)
    if (!line) continue
    const escaped = line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    out += `${stamp(c.start, '.')} --> ${stamp(c.end, '.')}\n${escaped}\n\n`
  }
  return out
}

/** `m:ss` for the player and the segment list - a transport reading
 *  `00:01:07.480` is noise at this scale. Hours appear only when there are. */
export function clock(t: number): string {
  const s = Math.max(0, Math.floor(Number.isFinite(t) ? t : 0))
  const m = Math.floor(s / 60)
  const tail = `${String(m % 60).padStart(m >= 60 ? 2 : 1, '0')}:${String(s % 60).padStart(2, '0')}`
  return m >= 60 ? `${Math.floor(m / 60)}:${tail}` : tail
}
