// What a live audio graph is carrying, right now, as a row of bars.
//
// Two callers, two jobs. The composer's RECORDING state used to be a stop
// button and a number of seconds - both true, and neither answering the
// question someone actually has while talking to a machine: "is it picking me
// up?" And the PLAYER's scrub track used to be a line, which says where you
// are and nothing about what is there.
//
// One implementation because the shaping is the hard part and it is identical
// either way: log-spaced bands, a pink-noise tilt, and an attack/release
// envelope. Getting that right once is what makes both of them look like
// instruments rather than like animated divs.

import { readonly, ref } from 'vue'

/** The range a voice occupies, and the reason the bands are LOG-spaced across
 *  it. An FFT's bins are linear in frequency: over a 24 kHz Nyquist, most of
 *  a row of equal-width bands would sit above 3 kHz, where speech has very
 *  little energy - the meter would barely move while someone talked into it.
 *  Log spacing puts a band where the ear puts one. */
const LO_HZ = 80
const HI_HZ = 6000

/** Smoothing on the analyser itself, kept LOW because the envelope below does
 *  the shaping. The default 0.8 is built for spectrum displays and would slow
 *  the attack the envelope is trying to keep fast; this only takes the
 *  bin-to-bin jitter off. */
const SMOOTHING = 0.3

/** The window the byte values are mapped across. The defaults (-100..-30 dB)
 *  are a general-purpose spectrum range: on a microphone with AGC almost all
 *  of speech lands in the top third of it, so the bars sit high and barely
 *  move. -85..-25 puts a normal speaking voice across the middle of the scale,
 *  which is what makes the difference between talking and not talking visible
 *  rather than merely present. */
const MIN_DB = -85
const MAX_DB = -25

/** Envelope time constants, in seconds. Fast up, slow down - this is the one
 *  thing that separates a meter that looks designed from a row of flickering
 *  bars, and it is what every physical VU meter does mechanically. A syllable
 *  hits immediately; the fall is slow enough to read.
 *
 *  Done here rather than as a CSS transition, deliberately: a transition
 *  duration on top of a ~60 Hz update is a second smoothing stage that fights
 *  the first, and it damps the attack as much as the release - precisely
 *  backwards. */
const ATTACK_S = 0.025
const RELEASE_S = 0.22

/** Per-band gain, as an exponent on the band's centre frequency ratio.
 *
 *  Speech energy falls steeply with frequency, so an honest FFT readout leaves
 *  the right-hand bars permanently near the floor and the meter looks broken
 *  rather than quiet. This is the pink-noise tilt every spectrum display
 *  applies for the same reason: it does not invent signal, it puts the bands
 *  on a comparable footing so a voice moves all of them. Clamped, so the top
 *  band cannot amplify room hiss into a bar that is always lit. */
const TILT = 0.34
const TILT_MAX = 2.4

/** One meter over one graph. */
export function createLevelMeter(bandCount: number) {
  const bars = ref<number[]>([])

  let ctx: AudioContext | null = null
  let node: AnalyserNode | null = null
  // Explicitly backed by a plain ArrayBuffer: getByteFrequencyData will not
  // take the SharedArrayBuffer-capable default type.
  let bins: Uint8Array<ArrayBuffer> | null = null
  let raf = 0
  /** Precomputed [start, end) bin index and gain per band - the frequency
   *  maths depends only on the context's sample rate, so it is done once per
   *  attach rather than sixty times a second. */
  let spans: { i0: number; i1: number; gain: number }[] = []
  /** The envelope's current value per band, which is what actually renders. */
  let env: number[] = []
  let last = 0

  function plan(): void {
    const nyquist = (ctx?.sampleRate ?? 48000) / 2
    const n = bins?.length ?? 0
    spans = []
    for (let b = 0; b < bandCount; b++) {
      const lo = LO_HZ * (HI_HZ / LO_HZ) ** (b / bandCount)
      const hi = LO_HZ * (HI_HZ / LO_HZ) ** ((b + 1) / bandCount)
      const i0 = Math.min(n - 1, Math.max(0, Math.floor((lo / nyquist) * n)))
      // at least one bin, always: the lowest bands are narrower than a bin at
      // this fftSize and would otherwise average over nothing
      const i1 = Math.min(n, Math.max(i0 + 1, Math.ceil((hi / nyquist) * n)))
      const mid = Math.sqrt(lo * hi)
      spans.push({ i0, i1, gain: Math.min(TILT_MAX, (mid / LO_HZ) ** TILT) })
    }
    env = new Array(bandCount).fill(0)
  }

  function read(now: number): void {
    if (!node || !bins) return
    node.getByteFrequencyData(bins)
    // Time-based rather than per-frame, and clamped: a backgrounded tab
    // resumes with an enormous gap, and a per-frame coefficient would either
    // snap or crawl depending on which way the level went.
    const dt = Math.min(0.1, last ? (now - last) / 1000 : 1 / 60)
    last = now
    const up = 1 - Math.exp(-dt / ATTACK_S)
    const down = 1 - Math.exp(-dt / RELEASE_S)
    const out: number[] = []
    for (let b = 0; b < spans.length; b++) {
      const { i0, i1, gain } = spans[b]
      let sum = 0
      for (let i = i0; i < i1; i++) sum += bins[i]
      const raw = Math.min(1, (sum / (i1 - i0) / 255) * gain)
      const prev = env[b] ?? 0
      env[b] = prev + (raw - prev) * (raw > prev ? up : down)
      out.push(env[b])
    }
    bars.value = out
  }

  function pump(now: number): void {
    if (!node) {
      raf = 0
      return
    }
    read(now)
    raf = requestAnimationFrame(pump)
  }

  const meter = {
    /** Per-band level, 0..1, newest read. Empty when nothing is attached,
     *  which is what a caller renders nothing from - an empty meter and a
     *  silent one must not look the same. */
    levels: readonly(bars),
    /** Tap a live graph. `src` is whatever the caller already built; the
     *  analyser hangs off it and connects to nothing further, so this cannot
     *  route audio anywhere it was not already going.
     *
     *  Safe to call over an existing attach (the old one is dropped) and safe
     *  to fail: a browser with no analyser simply shows no bars, which is the
     *  behaviour before this existed rather than a new way to break. */
    attach(ac: AudioContext, src: AudioNode): void {
      meter.detach()
      try {
        const an = ac.createAnalyser()
        // 1024 is 512 bins - enough resolution for a few dozen bands, and half
        // the per-frame copy of the 2048 the recorder's wake probe wants for
        // its own reasons.
        an.fftSize = 1024
        an.smoothingTimeConstant = SMOOTHING
        an.minDecibels = MIN_DB
        an.maxDecibels = MAX_DB
        src.connect(an)
        ctx = ac
        node = an
        bins = new Uint8Array(an.frequencyBinCount)
        plan()
        last = 0
        raf = requestAnimationFrame(pump)
      } catch {
        meter.detach()
      }
    },
    /** Release it. Called on every teardown path - a meter left running holds
     *  a rAF loop and would go on reporting a graph that is gone. */
    detach(): void {
      if (raf) cancelAnimationFrame(raf)
      raf = 0
      node?.disconnect()
      node = null
      bins = null
      ctx = null
      spans = []
      env = []
      last = 0
      bars.value = []
    },
  }
  return meter
}

/** How many bars the composer's mic meter shows. Odd, so there is a middle
 *  one, and small enough to sit in the tool row without crowding it. */
const MIC_BANDS = 9

/** The microphone's meter - one instance, module-level, for the same reason
 *  `useClipPlayback` is: there is one microphone. Both mic paths (record via
 *  `useRecorder`, live/dictate via `useMicTranscribe`) feed this one, so the
 *  composer reads a single thing and the bars look identical whichever mode
 *  the arming picked.
 *
 *  Players do not share it - each one meters its own clip, so each makes its
 *  own with `createLevelMeter`. */
const micMeter = createLevelMeter(MIC_BANDS)

export function useMicLevels() {
  return micMeter
}
