<script setup lang="ts">
// The player's scrub track, drawn as the clip's own waveform.
//
// It is here because a line answers none of the questions someone brings to an
// ASR studio and a shape answers all of them: where is there speech, where did
// it go quiet, is that pause the one the model wrote a sentence over, and
// which burst is the word I am reading. It also turns seeking from aiming at a
// percentage into aiming at a sound.
//
// The guard spans are painted on the same timeline, and that is the part no
// other player can do: the runner already reports which seconds it refused,
// cut, or dropped as no-speech (`paddock_guards`). Over the
// envelope, "it hallucinated over that silence" stops being a footnote under
// the transcript and becomes a flat stretch with a band on it.
//
// CANVAS, not elements. The playhead moves at ~60 Hz and the envelope has a
// few hundred columns; as DOM that is a few hundred style writes a frame. Two
// cached bitmaps and a blit is the same picture for a rounding error of the
// cost, and it keeps this affordable in a message list where several can exist
// at once.
//
// PURELY VISUAL, and pointer-transparent. It draws; it is not a widget. The
// seeking, the keyboard, the ARIA and the focus ring stay with the real
// `ui/Slider` lying on top of it - the reka-ui reuse rule, which the
// shipped-UI lint enforces. An earlier draft of this file claimed
// `role="slider"` on its own div and would have owed every part of that
// contract by hand.
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { Peaks } from '@/composables/useAudioPeaks'
import type { TranscriptGuard } from '@/types/chat'
import { clock } from '@/lib/subtitles'

const props = withDefaults(
  defineProps<{
    peaks: Peaks | null
    /** Seconds. 0 means "not measured yet" - the track renders idle. */
    duration: number
    /** Playhead, in seconds. */
    current: number
    /** Spans the decode refused or cut, painted over the envelope. */
    guards?: TranscriptGuard[]
    /** Where the cursor sits on the timeline, in seconds; null when it is not
     *  over the track. Passed in rather than read here, because this layer
     *  takes no pointer events - the wrapper above is the only thing that can
     *  see the cursor without stealing it from the Slider. */
    hover?: number | null
  }>(),
  { guards: () => [], hover: null },
)

const host = ref<HTMLElement | null>(null)
const cv = ref<HTMLCanvasElement | null>(null)

/** The two cached layers: the envelope in its resting colour, and the same
 *  envelope in the played colour. A frame is `base`, then `fill` clipped to
 *  the playhead, then the playhead itself. Rebuilt only when the picture
 *  changes - size, peaks, guards, theme - never per frame. */
let base: HTMLCanvasElement | null = null
let fill: HTMLCanvasElement | null = null
let ro: ResizeObserver | null = null
let themeWatch: MutationObserver | null = null
let raf = 0
/** css pixels */
let w = 0
let h = 0

const progress = computed(() =>
  props.duration > 0 ? Math.min(1, Math.max(0, props.current / props.duration)) : 0,
)

function css(name: string, fallback: string): string {
  const el = host.value
  if (!el) return fallback
  return getComputedStyle(el).getPropertyValue(name).trim() || fallback
}

/** Draw the envelope into `c`, in one colour. */
function envelope(c: HTMLCanvasElement, colour: string): void {
  const g = c.getContext('2d')
  if (!g) return
  const dpr = window.devicePixelRatio || 1
  g.setTransform(dpr, 0, 0, dpr, 0, 0)
  g.clearRect(0, 0, w, h)
  const mid = h / 2
  const p = props.peaks
  g.fillStyle = colour
  if (!p || !p.max.length) {
    // No envelope to draw - a flat rule, so the control still reads as a
    // track rather than as a hole where one failed to appear.
    g.fillRect(0, mid - 0.5, w, 1)
    return
  }
  // One column per device-independent pixel, resampled from the stored
  // buckets: the envelope is decoded once at a fixed resolution, so a resize
  // costs a redraw and never a re-decode.
  const cols = Math.max(1, Math.floor(w))
  const n = p.max.length
  for (let x = 0; x < cols; x++) {
    const i0 = Math.floor((x * n) / cols)
    const i1 = Math.max(i0 + 1, Math.floor(((x + 1) * n) / cols))
    let lo = 0
    let hi = 0
    for (let i = i0; i < i1; i++) {
      if (p.min[i] < lo) lo = p.min[i]
      if (p.max[i] > hi) hi = p.max[i]
    }
    // A floor of half a pixel each way: a silent column must still be part of
    // the line, or the track breaks into islands and stops reading as one clip.
    const top = mid - Math.max(0.5, hi * (h / 2 - 1))
    const bot = mid + Math.max(0.5, -lo * (h / 2 - 1))
    g.fillRect(x, top, 1, Math.max(1, bot - top))
  }
}

/** Guard bands, under the envelope. Drawn into `base` only: a span the decode
 *  refused is a fact about the CLIP, not about how far you have played. */
function bands(c: HTMLCanvasElement): void {
  const g = c.getContext('2d')
  if (!g || props.duration <= 0) return
  const dpr = window.devicePixelRatio || 1
  g.setTransform(dpr, 0, 0, dpr, 0, 0)
  const dropped = css('--pk-status-error-subtle', 'rgba(220,80,80,0.18)')
  const cut = css('--pk-status-warning-subtle', 'rgba(220,170,60,0.18)')
  for (const gd of props.guards) {
    const x0 = (gd.start / props.duration) * w
    const x1 = (gd.end / props.duration) * w
    // Two weights, and the difference is what the reader must act on: a
    // DROPPED span had its text discarded (the audio held nothing), a cut one
    // kept what it said up to the cut. Same shape, different colour, and both
    // explained in full by the notice list under the transcript.
    g.fillStyle = gd.dropped ? dropped : cut
    g.fillRect(x0, 0, Math.max(2, x1 - x0), h)
  }
}

function rebuild(): void {
  const el = cv.value
  const box = host.value
  if (!el || !box) return
  const dpr = window.devicePixelRatio || 1
  w = box.clientWidth
  h = box.clientHeight
  if (w <= 0 || h <= 0) return
  el.width = Math.round(w * dpr)
  el.height = Math.round(h * dpr)
  el.style.width = `${w}px`
  el.style.height = `${h}px`
  const mk = (): HTMLCanvasElement => {
    const c = document.createElement('canvas')
    c.width = el.width
    c.height = el.height
    return c
  }
  base = mk()
  fill = mk()
  bands(base)
  envelope(base, css('--pk-border-strong', '#8a8a8a'))
  envelope(fill, css('--pk-accent', '#5b8cff'))
  draw()
}

function draw(): void {
  const el = cv.value
  const g = el?.getContext('2d')
  if (!el || !g || !base || !fill) return
  const dpr = window.devicePixelRatio || 1
  g.setTransform(1, 0, 0, 1, 0, 0)
  g.clearRect(0, 0, el.width, el.height)
  g.drawImage(base, 0, 0)
  const played = Math.round(progress.value * w * dpr)
  if (played > 0) {
    g.save()
    g.beginPath()
    g.rect(0, 0, played, el.height)
    g.clip()
    g.drawImage(fill, 0, 0)
    g.restore()
  }
  g.setTransform(dpr, 0, 0, dpr, 0, 0)
  // Where a click would land. The number rides in the DOM label above; this is
  // the line that makes it a position rather than a guess.
  if (props.hover !== null && props.duration > 0) {
    g.fillStyle = css('--pk-text-muted', '#8a8a8a')
    g.globalAlpha = 0.55
    g.fillRect(Math.min(w - 1, (props.hover / props.duration) * w), 0, 1, h)
    g.globalAlpha = 1
  }
}

/** Coalesce redraws to one a frame: the playhead ticks at ~60 Hz and so does
 *  the pointer, and both landing in one frame must not draw twice. */
function schedule(): void {
  if (raf) return
  raf = requestAnimationFrame(() => {
    raf = 0
    draw()
  })
}

watch(() => [props.current, props.duration, props.hover], schedule)
watch(
  () => [props.peaks, props.guards],
  () => rebuild(),
)

onMounted(() => {
  rebuild()
  if (host.value && 'ResizeObserver' in window) {
    ro = new ResizeObserver(() => rebuild())
    ro.observe(host.value)
  }
  // The envelope is painted in theme colours, so a theme flip has to repaint
  // the cached layers - they would otherwise keep yesterday's palette until
  // something else resized them.
  themeWatch = new MutationObserver(() => rebuild())
  themeWatch.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme', 'class'],
  })
})
onBeforeUnmount(() => {
  ro?.disconnect()
  themeWatch?.disconnect()
  if (raf) cancelAnimationFrame(raf)
})
</script>

<template>
  <div ref="host" class="wt" aria-hidden="true">
    <canvas ref="cv" class="wt__cv" />
    <span
      v-if="hover !== null && duration > 0"
      class="wt__at"
      :style="{ left: `${(hover / duration) * 100}%` }"
      >{{ clock(hover) }}</span
    >
  </div>
</template>

<style scoped>
.wt {
  position: absolute;
  inset: 0;
  border-radius: var(--pk-radius-sm);
  /* the Slider on top owns every pointer event; this layer only draws */
  pointer-events: none;
}
.wt__cv {
  display: block;
  width: 100%;
  height: 100%;
}
/* the time under the cursor, so a click is aimed rather than estimated */
.wt__at {
  position: absolute;
  bottom: calc(100% + 3px);
  transform: translateX(-50%);
  padding: 2px 5px;
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-elevated);
  border: 1px solid var(--pk-border-strong);
  color: var(--pk-text-primary);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
</style>
