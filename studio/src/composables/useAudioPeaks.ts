// The shape of a clip, as a peak envelope, for the player's waveform track.
//
// Why A WAVEFORM AND not A LINE. This player is not a music transport, it is
// the audit surface for a transcript: the questions someone brings to it are
// "where is there speech", "where did it go quiet", "is that pause where the
// model hallucinated a sentence", "which burst is the word I am reading". A
// progress line answers none of them and a level animation answers none of
// them either - both only say the clip is playing, which the play button
// already said. The envelope answers all four before anything is clicked, and
// it turns seeking from a guess at a percentage into aiming at a sound.
//
// It also gives the decode guards somewhere to live. The runner already tells
// us which seconds it refused, cut, or dropped as no-speech (`paddock_guards`,
// - painted over the envelope, "it wrote a fluent sentence over
// that silence" stops being a note under the transcript and becomes a shape
// you can see, which is the whole no-silent-failures thesis rendered.
//
// COST, and the ceiling it earns. Peaks need the samples, so this fetches the
// attachment and decodes it - a second read of bytes the <audio> element is
// already streaming, which the HTTP cache serves (same origin, same URL, same
// method). `decodeAudioData` is off-thread; the reduction below is not, but it
// is one linear pass and a few hundred thousand samples for the clips this
// product is about. Past the ceiling there is no waveform and the player falls
// back to its plain track rather than freezing a tab to draw a picture.

import { ref, shallowRef } from 'vue'

/** Above this, no waveform. A transcription studio is about clips and
 *  meetings, not archives: at 16 kHz mono a 20-minute file is ~19M samples,
 *  which decodes to ~77 MB and reduces in well under a second - and an hour
 *  is three times that in memory for a picture nobody is studying at that
 *  zoom anyway. */
const MAX_SECONDS = 20 * 60
/** And a hard byte ceiling for the fetch, since duration is only known after
 *  decoding: a compressed hour is small, a WAV hour is not. */
const MAX_BYTES = 96 * 1024 * 1024

/** How many buckets to reduce to. Not the pixel width: the canvas is redrawn
 *  at whatever size it happens to be and resamples from this, so a resize does
 *  not re-decode. Generous enough that a wide player still looks sampled
 *  rather than blocky. */
const BUCKETS = 2048

/** One clip's envelope: `min[i]`/`max[i]` are the extremes in bucket `i`,
 *  both in -1..1. Two arrays rather than one RMS because a waveform drawn
 *  from RMS loses the transients that make speech legible as speech. */
export interface Peaks {
  min: Float32Array
  max: Float32Array
}

/** Decoded envelopes, by attachment id. Module-level and never evicted on
 *  purpose: it is two Float32Arrays of 2048 (16 KB a clip), and a conversation
 *  someone is auditing gets scrolled through repeatedly. Dies with the page. */
const cache = new Map<string, Peaks>()
/** In-flight decodes, so N lanes mounting at once do not fetch N times. */
const inflight = new Map<string, Promise<Peaks | null>>()

function reduce(buf: AudioBuffer): Peaks {
  const n = BUCKETS
  const min = new Float32Array(n).fill(0)
  const max = new Float32Array(n).fill(0)
  const len = buf.length
  if (!len) return { min, max }
  // Every channel, folded: a clip recorded from a stereo interface can carry
  // the voice on one side only, and a waveform drawn from channel 0 alone
  // would show silence for audio the model transcribed perfectly.
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const data = buf.getChannelData(c)
    for (let i = 0; i < n; i++) {
      const from = Math.floor((i * len) / n)
      const to = Math.max(from + 1, Math.floor(((i + 1) * len) / n))
      let lo = 0
      let hi = 0
      // Stride on long clips: at 20 minutes a bucket holds ~9400 samples and
      // reading every one buys precision no pixel can show. The peak of a
      // 1-in-8 sample of a 9400-sample window is within a hair of the true
      // peak for anything that is not a single-sample click.
      const step = Math.max(1, Math.floor((to - from) / 1200))
      for (let s = from; s < to; s += step) {
        const v = data[s]
        if (v < lo) lo = v
        else if (v > hi) hi = v
      }
      if (lo < min[i]) min[i] = lo
      if (hi > max[i]) max[i] = hi
    }
  }
  return { min, max }
}

async function decode(url: string): Promise<Peaks | null> {
  const res = await fetch(url)
  if (!res.ok) return null
  const len = Number(res.headers.get('content-length') ?? 0)
  if (len > MAX_BYTES) return null
  const bytes = await res.arrayBuffer()
  if (bytes.byteLength > MAX_BYTES) return null
  const Ctx =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  // An OFFLINE context: this decodes, it does not play, and a live
  // AudioContext here would count against the browser's per-page limit and
  // (on some versions) start suspended outside a user gesture.
  const ac = new Ctx()
  try {
    const buf = await ac.decodeAudioData(bytes)
    if (buf.duration > MAX_SECONDS) return null
    return reduce(buf)
  } finally {
    void ac.close().catch(() => {})
  }
}

/** The envelope for one clip, or null while there is none to draw.
 *
 *  Null covers every honest reason: no id, a container the browser decodes
 *  for playback but not through WebAudio, a file past the ceiling, a failed
 *  fetch. The player treats all of them the same - it draws its plain track -
 *  because from the reader's side they are the same fact: this clip has no
 *  picture, and the transcript is unaffected either way. */
export function useAudioPeaks() {
  const peaks = shallowRef<Peaks | null>(null)
  const loading = ref(false)

  async function load(clip: string | undefined, url: string): Promise<void> {
    peaks.value = null
    if (!clip || !url) return
    const hit = cache.get(clip)
    if (hit) {
      peaks.value = hit
      return
    }
    loading.value = true
    try {
      let job = inflight.get(clip)
      if (!job) {
        job = decode(url).catch(() => null)
        inflight.set(clip, job)
      }
      const got = await job
      inflight.delete(clip)
      if (got) cache.set(clip, got)
      peaks.value = got
    } finally {
      loading.value = false
    }
  }

  return { peaks, loading, load }
}
