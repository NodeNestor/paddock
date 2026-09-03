// Record a clip, hand back a file. Nothing else.
//
// The other microphone path (`useMicTranscribe`) opens a realtime socket per
// armed lane and pushes the same PCM to all of them, so two models transcribe
// the identical audio as it is spoken. That is the better comparison when it
// is available - and it is available only against LOCAL runners: no provider
// takes a socket, so a cloud speech model can only ever hear a finished file
//
// So this is the second mode, and it is deliberately its own path rather than
// a flag inside the live one: with no lanes to feed there is no capture graph,
// no resample, no worklet, no per-lane backlog - just MediaRecorder. Keeping
// them apart means the record mode cannot regress the live one, which has cost
// two bugs of its own already.
//
// What comes out is an ordinary File, which is the point: it goes into the
// composer's tray and sends down the exact path a dragged clip takes, so every
// lane - local and cloud alike - hears the same recording.
//
// The MICROPHONE is not AWAKE when getUserMedia RESOLVES. Measured:
// a clip recorded straight after the promise settled began with
// 0.90 s of exact zeros - not a quiet ramp, digital silence - and the words
// spoken into that hole were simply not in the file. Chrome hands over a live
// track and fills it with silence while the capture device actually opens.
// So we do not start the recorder, start the clock, or tell the user we are
// listening until a non-zero sample has come through. It cannot recover speech
// the device never delivered, but it stops the composer inviting you to talk
// into a microphone that is not on yet, which is the same "no silent failures"
// rule the rest of the product follows.

import { computed, ref, shallowRef } from 'vue'

import { openMic } from './useAudioDevices'
import { useMicLevels } from './useMicLevels'

/** Container preference, best first. Opus in WebM is the small, well-supported
 *  one; Safari answers mp4. Whatever the browser accepts, the runner and the
 * providers both decode it. */
const TYPES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus']
const EXT: Record<string, string> = { webm: 'webm', mp4: 'm4a', ogg: 'ogg' }

/** How long to wait for the device to wake before recording anyway. Generous
 *  deliberately: a wireless (DECT) headset has to bring up its radio link to the
 *  base station when an app opens the microphone, and one that has been idle
 *  takes noticeably longer than the ~0.9 s a warm one does. Past this we would
 *  rather capture a clip that starts with silence than refuse to record -
 *  some virtual devices genuinely emit exact zeros until the first sound. */
const WAKE_CEILING_MS = 4000
/** How often to look for the first real sample. Cheap: one analyser read. */
const WATCH_MS = 25

/** Hard ceiling on one recording, seconds.
 *
 *  There was no ceiling at all, and nothing downstream supplied one: the
 *  100 MB attachment guard never sees a recording, and MAX_BODY (192 MB) is
 *  hours of Opus. So an afternoon of talking uploaded, decoded to ~460 MB of
 *  f32, and only then met whatever refused it.
 *
 *  An hour is the most a whisper lane can be asked for without the wait
 *  becoming the story - it windows, so length costs time and nothing else.
 *  A generative ASR lane's own ceiling is far lower and arrives per model
 *  through `maxSeconds`; this is the floor under every case, including the
 *  one where the server never said. */
export const RECORD_MAX_S = 60 * 60

export function useRecorder() {
  // The meter runs off the same graph the wake probe already builds, so
  // showing it costs one analyser and no second microphone tap.
  const meter = useMicLevels()
  /** `arming` is the device-wake wait: the microphone is open but has not
   *  produced a sample yet, so nothing is being captured. */
  const state = ref<'idle' | 'arming' | 'recording'>('idle')
  /** Busy either way - the composer's button is a stop button from the click. */
  const recording = computed(() => state.value !== 'idle')
  const arming = computed(() => state.value === 'arming')
  /** Seconds captured so far - counted from the first real sample, not from
   *  the click, because that is the audio that exists. */
  const elapsed = ref(0)
  /** This take's ceiling in seconds - the hard one, or lower when the model
   *  that will hear it says so. */
  const limit = ref(RECORD_MAX_S)
  /** The ceiling has been reached. The caller ends the take; see the tick. */
  const capped = ref(false)
  /** Seconds left, floored at 0 - what a countdown renders. */
  const remaining = computed(() => Math.max(0, limit.value - elapsed.value))
  const error = ref<string | null>(null)
  const rec = shallowRef<MediaRecorder | null>(null)

  let stream: MediaStream | null = null
  let chunks: BlobPart[] = []
  let tick: ReturnType<typeof setInterval> | null = null
  let watch: ReturnType<typeof setInterval> | null = null
  let ac: AudioContext | null = null
  let analyser: AnalyserNode | null = null
  // Explicitly backed by a plain ArrayBuffer: getFloatTimeDomainData will not
  // take the SharedArrayBuffer-capable default.
  let probe: Float32Array<ArrayBuffer> | null = null
  /** The device has delivered at least one non-zero sample. */
  let heard = false
  /** ...and we were actually able to look, so `!heard` means something. */
  let watched = false
  let opened = 0

  function teardown(): void {
    meter.detach()
    if (tick) clearInterval(tick)
    tick = null
    if (watch) clearInterval(watch)
    watch = null
    analyser?.disconnect()
    analyser = null
    probe = null
    ac?.close().catch(() => {})
    ac = null
    stream?.getTracks().forEach((t) => t.stop())
    stream = null
    rec.value = null
    state.value = 'idle'
  }

  /** Tap the stream so we can see whether samples are flowing. Best-effort:
   *  where there is no AudioContext we simply record immediately, which is the
   *  old behaviour rather than a new failure. */
  function listen(): void {
    try {
      const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      ac = new Ctx()
      ac.resume().catch(() => {})
      const src = ac.createMediaStreamSource(stream as MediaStream)
      analyser = ac.createAnalyser()
      analyser.fftSize = 2048
      src.connect(analyser)
      probe = new Float32Array(analyser.fftSize)
      // From here, not from `begin`: the bars are how someone sees the device
      // wake up, and a meter that only starts once capture does would leave
      // the arming state as the same dead pause it was before.
      meter.attach(ac, src)
    } catch {
      analyser = null
      probe = null
    }
  }

  function look(): void {
    if (analyser && probe) {
      analyser.getFloatTimeDomainData(probe)
      watched = true
      for (let i = 0; i < probe.length; i++) {
        if (probe[i] !== 0) {
          heard = true
          break
        }
      }
    }
    if (state.value !== 'arming') return
    // No way to watch, or the device took too long: start regardless.
    if (heard || !analyser || performance.now() - opened > WAKE_CEILING_MS) begin()
  }

  function begin(): void {
    if (!stream || state.value !== 'arming') return
    const mime = TYPES.find((t) => MediaRecorder.isTypeSupported(t))
    const r = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
    r.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data)
    }
    // A timeslice, so a long recording is not one enormous blob held whole in
    // memory until stop.
    r.start(1000)
    rec.value = r
    state.value = 'recording'
    const from = performance.now()
    tick = setInterval(() => {
      elapsed.value = (performance.now() - from) / 1000
      // The ceiling is REPORTED, not enforced here: stop() hands back a File
      // and only the caller knows what to do with it. Flipping a flag keeps
      // this composable's one job ("record a clip, hand back a file") intact
      // and lets the composer end the take the same way a click does.
      if (!capped.value && elapsed.value >= limit.value) capped.value = true
    }, 200)
  }

  /** Open the microphone. Returns false when it was refused or is missing -
   *  the caller has to know, because it may have opened a turn to fill.
   *
   *  True means the device is open, not that capture has begun: watch `arming`
   *  for that, and do not tell the user to speak until it clears.
   *
   *  `maxSeconds` lowers this take's ceiling - what the model about to hear it
   *  can actually take. Clamped to RECORD_MAX_S, so a server that says nothing
   *  (or says something absurd) still lands inside the hard limit. */
  async function start(maxSeconds?: number): Promise<boolean> {
    if (state.value !== 'idle') return false
    limit.value =
      maxSeconds && maxSeconds > 0 ? Math.min(maxSeconds, RECORD_MAX_S) : RECORD_MAX_S
    capped.value = false
    // Missing entirely on plain-http origins (browser secure-context rule) -
    // that is a different fact from "no microphone", so say the real one.
    if (!navigator.mediaDevices?.getUserMedia) {
      error.value =
        'The browser blocks the microphone on this address. See Trust this computer in the Manager.'
      return false
    }
    error.value = null
    chunks = []
    elapsed.value = 0
    heard = false
    watched = false
    try {
      // The chosen device, and the same processing the live path asks for -
      // both live in `openMic` precisely so a recording and a dictation of the
      // same sentence cannot arrive at the model through different front ends.
      // A chosen device that has gone away falls back to the system default
      // and says so (`micLost`), which the composer shows before the first
      // word.
      stream = await openMic()
    } catch (e) {
      error.value =
        e instanceof DOMException && e.name === 'NotAllowedError'
          ? 'The microphone was blocked. Allow it for this page and try again.'
          : 'No microphone is available.'
      return false
    }
    state.value = 'arming'
    listen()
    opened = performance.now()
    watch = setInterval(look, WATCH_MS)
    // A device that is already awake (a second recording in the same session)
    // should not pay a frame for it.
    look()
    return true
  }

  /** Stop and hand back what was said. `null` when there is nothing worth
   *  sending, and `error` then says why - a clip of pure silence would
   *  otherwise travel to every armed lane and come back as an empty transcript
   *  with no reason attached. */
  async function stop(): Promise<File | null> {
    if (state.value === 'arming') {
      teardown()
      error.value = 'Nothing was recorded - the microphone had not started yet.'
      return null
    }
    const r = rec.value
    if (!r || r.state === 'inactive') {
      teardown()
      return null
    }
    const silent = watched && !heard
    const type = r.mimeType || 'audio/webm'
    const ext = EXT[type.split('/')[1]?.split(';')[0] ?? 'webm'] ?? 'webm'
    const file = await new Promise<File | null>((resolve) => {
      r.onstop = () => {
        const parts = chunks
        chunks = []
        const size = parts.reduce((n, p) => n + (p instanceof Blob ? p.size : 0), 0)
        // The name says what this is rather than pretending to be a file
        // someone chose - it is what the turn falls back to if the transcript
        // comes back empty.
        resolve(size ? new File(parts, `recording.${ext}`, { type }) : null)
      }
      r.stop()
    })
    teardown()
    if (silent) {
      error.value = 'That recording is silent - check the microphone is not muted or in use elsewhere.'
      return null
    }
    return file
  }

  /** Throw the recording away - the mic is released and nothing is returned. */
  function cancel(): void {
    const r = rec.value
    if (r && r.state !== 'inactive') {
      r.onstop = null
      r.stop()
    }
    chunks = []
    teardown()
  }

  return {
    recording,
    arming,
    elapsed,
    limit,
    remaining,
    capped,
    error,
    levels: meter.levels,
    start,
    stop,
    cancel,
  }
}
