// Small display formatters shared across the UI.

/** A number as the viewer's locale writes it - grouping separator included.
 *  Never hand-roll this: a 2 TB disk rendered "1999 GB" is both unreadable and
 *  wrong for every locale that doesn't group with commas. */
function num(v: number, maxFrac: number): string {
  return v.toLocaleString(undefined, { maximumFractionDigits: maxFrac })
}

/** Storage size in DECIMAL units, the way drive vendors and file managers
 *  count, and stepping up to TB so a big array doesn't read as four digits of
 *  gigabytes. */
export function fmtBytes(bytes: number): string {
  if (!bytes) return '0 GB'
  const tb = bytes / 1e12
  if (tb >= 1) return `${num(tb, tb >= 10 ? 0 : 2)} TB`
  const gb = bytes / 1e9
  if (gb >= 1) return `${num(gb, gb >= 10 ? 0 : 1)} GB`
  return `${num(bytes / 1e6, 0)} MB`
}

/** One file's size, the way a file manager writes it: binary units, down to KB.
 *  `fmtBytes` is the DRIVE formatter and bottoms out at "0 MB", which is what a
 *  162 KB photo would read as. Never "0 KB" either - a file that exists has a
 *  size, and rounding it away looks like a bug. */
export function fmtFileSize(bytes: number): string {
  const mb = bytes / (1024 * 1024)
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`
}

/** VRAM in BINARY units - matches the GPU dock and the card's marketed size
 *  (a 48 GB card reads "48 GB", not "52"). */
export function fmtVram(bytes: number): string {
  if (!bytes) return '0 GB'
  const gib = bytes / 1024 ** 3
  if (gib >= 1024) return `${num(gib / 1024, 2)} TB`
  return `${num(gib, gib >= 10 ? 0 : 1)} GB`
}

/** Token counts: 262144 -> "256K", 1048576 -> "1M". */
export function fmtTokens(tokens: number): string {
  if (!tokens) return '-'
  if (tokens >= 1024 * 1024) {
    const m = tokens / 1024 / 1024
    return `${num(m, tokens % (1024 * 1024) ? 1 : 0)}M`
  }
  if (tokens >= 1024) return `${num(tokens / 1024, 0)}K`
  return num(tokens, 0)
}

/** Context length as something a person can picture: pages of text. A token is
 *  about 3/4 of an English word and a book page holds ~400 words, so ~500
 *  tokens/page. Rounded hard - this is a mental image, not a measurement. */
export function fmtPages(tokens: number): string {
  const p = tokens / 500
  if (p >= 100) return `~${num(Math.round(p / 10) * 10, 0)} pages`
  if (p >= 10) return `~${num(Math.round(p / 5) * 5, 0)} pages`
  if (p >= 1.5) return `~${num(Math.round(p), 0)} pages`
  return '~1 page'
}

// ── timestamps: fixed ISO 8601, LOCAL time ──────────────────────────────────
// Browser locales can't see the OS format preference (only the browser
// language), so "locale formatting" guessed wrong - en-US browsers printed
// "Aug 1 7:52:43 PM" at a user whose format is 2026-08-01 07:52:43. An ops
// surface speaks ISO: unambiguous, sortable, culture-neutral.

function pad2(n: number): string {
  return String(n).padStart(2, '0')
}
/** Local wall clock, 24h: "07:52:43". */
export function fmtClock(d: Date): string {
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`
}
/** Full local stamp: "2026-08-01 07:52:43". */
export function fmtStamp(d: Date): string {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${fmtClock(d)}`
}

/** Human duration: sub-second in ms, otherwise seconds with one decimal. */
export function fmtDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return ''
  if (ms < 1000) return `${Math.round(ms)}ms`
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.round(ms / 1000)}s`
}

/** Round a tokens/sec figure for display. */
export function fmtTps(tps: number | undefined): string {
  return tps && tps > 0 ? `${Math.round(tps)} tok/s` : ''
}

/** Transfer rate: "12.4 MB/s" (KB/s below a meg - dial-up honesty). */
export function fmtRate(bps: number): string {
  if (bps >= 1e6) return `${(bps / 1e6).toFixed(1)} MB/s`
  return `${Math.max(1, Math.round(bps / 1e3))} KB/s`
}

/** Provider-reported request cost: "$0.011" - enough decimals that typical
 *  per-turn prices don't collapse to $0.00; a genuinely tiny turn reads as
 *  under a tenth of a cent rather than free. */
export function fmtCost(usd: number | undefined): string {
  if (usd === undefined || !Number.isFinite(usd)) return ''
  if (usd < 0.0005) return '<$0.001'
  return usd >= 0.1 ? `$${usd.toFixed(2)}` : `$${usd.toFixed(3)}`
}

/** Remaining time, coarse deliberately - an ETA pretending to second-accuracy
 *  reads as wrong the moment it ticks. "about 4 min left" / "38 s left". */
export function fmtEta(s: number): string {
  if (!Number.isFinite(s) || s < 0) return ''
  if (s < 60) return `${Math.max(1, Math.round(s))} s left`
  if (s < 3600) return `about ${Math.round(s / 60)} min left`
  return `about ${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min left`
}

/** The tight variant for table cells: "~38 s" / "~4 min" / "~1 h 5 min" -
 *  every character must fit in a fixed column. */
export function fmtEtaShort(s: number): string {
  if (!Number.isFinite(s) || s < 0) return ''
  if (s < 60) return `~${Math.max(1, Math.round(s))} s`
  if (s < 3600) return `~${Math.round(s / 60)} min`
  return `~${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`
}
