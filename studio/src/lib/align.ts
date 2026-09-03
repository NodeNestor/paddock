// Forced alignment: word times for a transcript that arrived without them
// The aligner (Qwen3-ForcedAligner through the runner's
// /v1/audio/alignments) answers word/start/end for its own splitting of the
// text - whitespace words with punctuation dropped, CJK one char at a time.
// Those are not the words the transcript renders: the flat-words path shows
// `word` verbatim (transcript-diff.ts), so replacing the lane's words with
// the aligner's would silently strip the punctuation out of the visible
// text. The merge therefore only ever ATTACHES times to the words the lane
// already has (or to the plain text's own whitespace tokens, which is
// exactly what the renderer falls back to when a lane sent none), and walks
// the two streams by folded-text agreement so a time can never land on a
// word it does not belong to. A walk that stops agreeing aborts the whole
// merge - an untimed transcript is honest, a mistimed one is not.

import type { TranscriptMeta, TranscriptWord } from '@/types/chat'

export interface AlignedWord {
  word: string
  start: number
  end: number
}

interface AlignResponse {
  duration?: number
  language?: string | null
  /** false = the clip's language is outside the model's trained eleven - the
   *  times still came back and may be useful (that is a measurement),
   *  but the caller deserves the flag. */
  language_supported?: boolean
  words?: AlignedWord[]
}

export interface AlignOutcome {
  words: AlignedWord[]
  languageSupported: boolean
}

/** Languages the runner REFUSES outright (morphological tokenizers it does
 *  not carry) - checked here too so the enrichment pass skips quietly instead
 *  of collecting a guaranteed 400. Both value spaces, because the lanes speak
 * both: ISO codes on whisper, English names on the generative
 *  families. */
export function alignmentRefused(language: string | undefined): boolean {
  const l = language?.toLowerCase()
  return l === 'ja' || l === 'japanese' || l === 'ko' || l === 'korean'
}

/** POST the clip + transcript to an aligner endpoint (always the manager
 *  relay - runner keys stay server-side). Throws with the server's own words
 *  on refusal; the caller decides how quiet to be about it. */
export async function alignClip(
  url: string,
  blob: Blob,
  filename: string,
  text: string,
  language?: string,
  signal?: AbortSignal,
): Promise<AlignOutcome> {
  const form = new FormData()
  form.append('file', blob, filename)
  form.append('text', text)
  if (language) form.append('language', language)
  const res = await fetch(url, { method: 'POST', body: form, signal })
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    try {
      const body = (await res.json()) as { error?: { message?: string } }
      if (body.error?.message) msg = body.error.message
    } catch {
      // non-JSON error body: the status is all there is
    }
    throw new Error(msg)
  }
  const body = (await res.json()) as AlignResponse
  return { words: body.words ?? [], languageSupported: body.language_supported !== false }
}

/** Casefold + drop punctuation/symbols - the same folding the compare diff
 *  uses, and (not by accident) the same character classes the runner's word
 *  splitter keeps, so both sides of the walk normalise to the same string. */
function fold(word: string): string {
  return [...word.toLowerCase().normalize('NFKC')]
    .filter((ch) => !/[\p{P}\p{S}\s]/u.test(ch))
    .join('')
}

/** Attach the aligner's spans to `targets` (the words the transcript will
 *  actually render). Returns one span per target - `undefined` on a target
 *  that is pure punctuation - or null when the two streams stop agreeing,
 *  which is the signal to keep the transcript untimed rather than guess.
 *  A target may consume SEVERAL aligner words (CJK: "你好," is one rendered
 *  token, two aligned chars) - its span is first start to last end. */
export function matchSpans(
  targets: string[],
  aligned: AlignedWord[],
): ({ start: number; end: number } | undefined)[] | null {
  const out: ({ start: number; end: number } | undefined)[] = []
  let j = 0
  for (const t of targets) {
    const want = fold(t)
    if (!want) {
      out.push(undefined)
      continue
    }
    let acc = ''
    let start = 0
    let end = 0
    while (acc.length < want.length) {
      if (j >= aligned.length) return null
      const fw = fold(aligned[j].word)
      if (!fw) {
        j++
        continue
      }
      if (want.slice(acc.length, acc.length + fw.length) !== fw) return null
      if (!acc) start = aligned[j].start
      end = aligned[j].end
      acc += fw
      j++
    }
    out.push({ start, end })
  }
  // trailing aligned words nothing claimed = the streams disagreed after all
  while (j < aligned.length && !fold(aligned[j].word)) j++
  return j === aligned.length ? out : null
}

/** The enriched flat word list for a transcript, or undefined when the merge
 *  cannot be done safely. `plain` is the rendered text - the fallback word
 *  source when the lane sent no word list, split exactly the way the
 *  renderer splits it so enrichment changes when words light up and never
 *  what the transcript says. */
export function mergeWordTimes(
  meta: TranscriptMeta,
  plain: string,
  aligned: AlignedWord[],
): TranscriptWord[] | undefined {
  if (!aligned.length) return undefined
  const existing = meta.words?.length ? meta.words : undefined
  const targets = existing?.map((w) => w.word) ?? plain.split(/\s+/).filter(Boolean)
  if (!targets.length) return undefined
  const spans = matchSpans(targets, aligned)
  if (!spans) return undefined
  return targets.map((word, i) => ({
    ...(existing ? existing[i] : { word }),
    start: spans[i]?.start,
    end: spans[i]?.end,
  }))
}
