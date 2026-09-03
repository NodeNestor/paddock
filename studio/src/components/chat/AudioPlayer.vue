<script lang="ts">
// Listening preferences shared by every transport on the page - module scope
// for the same reason useClipPlayback's registry is: a reader who set half
// volume or 1.5x on one clip means it for the next clip too, and a compare
// re-render must not reset it mid-listen. Transient view state, dies with the
// page, nothing wants it persisted.
import { ref as sharedRef } from 'vue'

const sharedVolume = sharedRef(1)
const sharedMuted = sharedRef(false)
const sharedRate = sharedRef(1)
</script>

<script setup lang="ts">
// The transcription transport (became the chat's when the page
// folded into conversations). A native <audio> does the decoding; the
// controls are ours because the transcript has to drive them - clicking a word
// seeks, and playing walks the highlight.
//
// The asymmetry that shapes this file: the SERVER accepts nine containers, the
// BROWSER plays a different set (Firefox's m4a/AAC depends on system codecs,
// and some containers simply will not play). A clip can transcribe perfectly
// and still not preview. A control that looks like a player and never plays is
// the silent failure the product principles ban, so an unplayable file says so
// and the transcript stands on its own.
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import Icon from '@/components/Icon.vue'
import Slider from '@/components/ui/Slider.vue'
import WaveTrack from './WaveTrack.vue'
import { useAudioPeaks } from '@/composables/useAudioPeaks'
import Menu from '@/components/ui/Menu.vue'
import MenuTrigger from '@/components/ui/MenuTrigger.vue'
import MenuContent from '@/components/ui/MenuContent.vue'
import MenuItem from '@/components/ui/MenuItem.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import { clock } from '@/lib/subtitles'
import type { TranscriptGuard } from '@/types/chat'

const props = withDefaults(
  defineProps<{
    src: string
    type: string
    /** Seconds, from the transcription's own `duration`. Used when the element
     *  will not say: a MediaRecorder blob carries no duration in its header,
     *  so Chrome reports Infinity for a clip recorded in this very page, and a
     *  scrubber with an infinite track is a dead control. The server measured
     *  the clip to transcribe it, so once a transcript exists we know. */
    fallback?: number
    /** The attachment this is playing. Only used to key the decoded waveform
     *  so several lanes on one clip share one decode - absent means no
     *  waveform, which is the honest state for a clip whose bytes were never
     *  stored. */
    clip?: string
    /** Spans the decode refused or cut, painted onto the track. They live on
     *  the LANE's transcript and the player lives on the user's turn, so they
     *  arrive the same way the measured duration does. */
    guards?: TranscriptGuard[]
  }>(),
  { fallback: 0, guards: () => [] },
)
const emit = defineEmits<{ (e: 'time', t: number): void }>()

// The shared refs re-bound as setup bindings: only setup bindings unwrap in
// the template (a module-scope ref would render as the object and refuse
// assignment).
const volume = sharedVolume
const muted = sharedMuted
const rate = sharedRate
const RATES = [0.75, 1, 1.25, 1.5, 2]

const el = ref<HTMLAudioElement | null>(null)
const playing = ref(false)
const at = ref(0)
const duration = ref(0)
/** null = the browser has not committed yet; the element decides on load. */
const playable = ref<boolean | null>(null)
/** the Infinity-duration probe runs once per source, not per loadedmetadata */
let resolving = false
/** How to call the duration probe off while its forced seek is outstanding.
 *
 *  It exists because the probe RESTORES the playhead when it completes, and
 *  restoring over a position somebody else chose is a bug rather than a
 *  courtesy - see `onLoaded`. Null whenever no probe is in flight. */
let cancelProbe: (() => void) | null = null

/** `canPlayType` answers '' | 'maybe' | 'probably'. Only '' is a decision -
 *  the other two are guesses, so those wait for the element itself. */
function guess(): boolean | null {
  if (!props.type) return null
  return document.createElement('audio').canPlayType(props.type) === '' ? false : null
}
watch(
  () => props.src,
  () => {
    // A probe still in flight belongs to the old source; letting it complete
    // would restore a playhead on audio that is no longer loaded.
    cancelProbe?.()
    playing.value = false
    at.value = 0
    duration.value = 0
    resolving = false
    playable.value = guess()
  },
  { immediate: true },
)

// The shared preferences reach the element here. Re-applied on every load
// because loading resets playbackRate to defaultPlaybackRate - volume alone
// would survive, rate would not.
function applyPrefs(): void {
  const a = el.value
  if (!a) return
  a.volume = volume.value
  a.muted = muted.value
  a.defaultPlaybackRate = rate.value
  a.playbackRate = rate.value
}
watch([el, volume, muted, rate], applyPrefs)

function publish(): void {
  const t = el.value?.currentTime ?? 0
  if (t === at.value) return
  at.value = t
  emit('time', t)
}

// The playhead ticks on requestAnimationFrame while playing, not on the
// element's timeupdate: timeupdate fires at ~4 Hz, which is a jerking scrubber
// and - once the transcript highlights the SPOKEN WORD - a word shorter than
// the firing period that never lights at all. The ~60 Hz loop is the pattern
// KBLab's easytranscriber player uses, for the same stated reason. timeupdate
// stays bound below for the paused states (a seek while paused still moves the
// clock) - while playing it just re-publishes a value this loop already did.
let raf = 0
function tick(): void {
  const a = el.value
  if (!a || a.paused) {
    raf = 0
    return
  }
  publish()
  raf = requestAnimationFrame(tick)
}
function onPlay(): void {
  playing.value = true
  if (!raf) raf = requestAnimationFrame(tick)
}
onBeforeUnmount(() => {
  if (raf) cancelAnimationFrame(raf)
  raf = 0
  cancelProbe?.()
})

function onLoaded(): void {
  const a = el.value
  const d = a?.duration ?? 0
  duration.value = Number.isFinite(d) && d > 0 ? d : 0
  playable.value = true
  applyPrefs()
  // A clip recorded in this very page reports Infinity, and an element whose
  // duration is Infinity has no seekable range: the transport renders but
  // nothing can be scrubbed. Forcing a seek past the end makes the browser
  // resolve the real duration, and then everything works - the widely used
  // workaround for MediaRecorder output, done once and undone immediately.
  if (a && !duration.value && !resolving) {
    resolving = true
    // ONE-SHOT AND CANCELLABLE, and both halves are load-bearing (found
    // live). A recording whose duration never resolves never fires
    // `seeked` for the forced seek below - and the old listener was removed
    // only from inside its own handler, so it stayed armed indefinitely. The
    // user's first real seek then fired it, and this handler put the playhead
    // back to 0: clicking a word in the transcript snapped a playing clip to
    // 0:00 and looked like the click had done nothing at all.
    const done = () => {
      cancelProbe = null
      const real = a.duration
      if (Number.isFinite(real) && real > 0) {
        duration.value = real
      }
      // Back where it was: nothing has played yet at load, which is the only
      // moment this runs uncancelled.
      a.currentTime = 0
      at.value = 0
    }
    cancelProbe = () => {
      a.removeEventListener('seeked', done)
      cancelProbe = null
      // The probe may well have WORKED already - the browser resolves the
      // duration on the assignment, not on the event - so take the number
      // even though we are giving up on the restore.
      const real = a.duration
      if (Number.isFinite(real) && real > 0) {
        duration.value = real
      }
    }
    a.addEventListener('seeked', done, { once: true })
    try {
      a.currentTime = 1e101
    } catch {
      cancelProbe()
    }
  }
}
const total = computed(() => duration.value || props.fallback)

// The waveform. Keyed by ATTACHMENT so several lanes on one clip share the one
// decode, and loaded lazily off the src - a clip that never gets looked at
// never gets fetched twice.
const { peaks } = ((): ReturnType<typeof useAudioPeaks> => {
  const p = useAudioPeaks()
  watch(
    () => [props.clip, props.src] as const,
    ([clip, src]) => void p.load(clip, src),
    { immediate: true },
  )
  return p
})()

function seek(t: number): void {
  const a = el.value
  if (!a) return
  // Whoever asked for this wins over the duration probe: its completion would
  // restore the playhead to 0 the instant this assignment lands (see
  // `onLoaded`), which is the whole of what "clicking a word does nothing"
  // was.
  cancelProbe?.()
  a.currentTime = t
  // READ it BACK. A MediaRecorder blob has no duration in its header, so its
  // seekable range is empty until the duration resolves and the element simply
  // IGNORES the assignment above - leaving the audio at 0 while `at` claimed
  // the end. That is what "the scrubber goes backwards when I press play"
  // was: the first timeupdate corrected the display, not the audio.
  at.value = Number.isFinite(a.currentTime) ? a.currentTime : 0
  emit('time', at.value)
}
defineExpose({ seek })

/** ±5 s. The element clamps the assignment to its seekable range, and `seek`
 *  reads the result back, so the edges need no arithmetic here. */
function skip(d: number): void {
  seek(Math.max(0, at.value + d))
}

async function toggle(): Promise<void> {
  const a = el.value
  if (!a) return
  if (a.paused) {
    try {
      await a.play()
    } catch {
      playable.value = false
    }
  } else {
    a.pause()
  }
}

// Keyboard transport, on the root so it works from any focused control inside:
// space toggles, arrows skip. Exemptions keep the inner controls honest -
// space on a focused button must press that button, and arrows inside the
// volume group are reka's own fine-stepping. Arrows on the scrub thumb are
// deliberately taken over: reka would step it by the 10 ms drag granularity,
// which is no skip at all.
const volWrap = ref<HTMLElement | null>(null)
function onKey(e: KeyboardEvent): void {
  if (playable.value !== true || e.ctrlKey || e.metaKey || e.altKey) return
  if (volWrap.value?.contains(e.target as Node)) return
  if (e.key === ' ' && !(e.target instanceof HTMLButtonElement)) {
    e.preventDefault()
    void toggle()
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault()
    skip(-5)
  } else if (e.key === 'ArrowRight') {
    e.preventDefault()
    skip(5)
  }
}

// live-seeking while dragging is what a scrubber is; the element's clock
// then writes the same value back, so there is no fight between the two
const scrub = computed<number>({ get: () => at.value, set: (v) => seek(v) })
/** The scrubbable length, or null while nothing has measured one.
 *
 *  It used to fall back to 0.01 so the slider always had a range. That is a
 *  FAKE range: a two-stop track that normalises a real position against a
 *  hundredth of a second, and the moment the true length arrives the thumb has
 *  already been snapped against the wrong scale. A length we do not know yet
 *  is a control that is not ready, which is what `disabled` is for. */
const span = computed(() => (total.value > 0 ? total.value : null))

/** The volume slider shows 0 while muted, and dragging it up unmutes - the
 *  slider is the truth of what you will hear, not of a stored setting. */
const vol = computed<number>({
  get: () => (muted.value ? 0 : volume.value),
  set: (v) => {
    volume.value = v
    if (v > 0) muted.value = false
  },
})
const silent = computed(() => muted.value || volume.value === 0)

/** Where the cursor is over the track, in seconds. Read on the WRAPPER: the
 *  Slider owns pointer events and the waveform underneath takes none, so this
 *  is the only layer that can watch the cursor without taking it away from
 *  either. */
const hoverAt = ref<number | null>(null)
function onHover(e: PointerEvent): void {
  const box = e.currentTarget as HTMLElement
  const total = span.value
  if (!total) {
    hoverAt.value = null
    return
  }
  const r = box.getBoundingClientRect()
  hoverAt.value = (Math.min(r.width, Math.max(0, e.clientX - r.left)) / r.width) * total
}
</script>

<template>
  <div
    class="ap"
    :class="{ 'ap--dead': playable === false }"
    role="group"
    aria-label="Audio player"
    tabindex="0"
    @keydown="onKey"
  >
    <audio
      ref="el"
      :src="src"
      preload="metadata"
      @loadedmetadata="onLoaded"
      @timeupdate="publish"
      @play="onPlay"
      @pause="playing = false"
      @ended="playing = false"
      @error="playable = false"
    />
    <template v-if="playable === false">
      <Icon name="eye-off" :size="15" />
      <span class="ap__dead">This browser cannot play {{ type || 'this format' }}</span>
    </template>
    <template v-else>
      <button
        class="ap__play"
        :disabled="playable === null"
        :aria-label="playing ? 'Pause' : 'Play'"
        @click="toggle"
      >
        <Icon :name="playing ? 'pause' : 'play'" :size="15" />
      </button>
      <Tooltip label="Back 5 seconds">
        <button
          class="ap__btn"
          :disabled="playable === null"
          aria-label="Back 5 seconds"
          @click="skip(-5)"
        >
          <Icon name="rewind" :size="13" />
        </button>
      </Tooltip>
      <Tooltip label="Forward 5 seconds">
        <button
          class="ap__btn"
          :disabled="playable === null"
          aria-label="Forward 5 seconds"
          @click="skip(5)"
        >
          <Icon name="fast-forward" :size="13" />
        </button>
      </Tooltip>
      <span class="ap__t">{{ clock(at) }}</span>
      <!-- The waveform is the TRACK; the Slider on top of it is the control.
           Reka keeps the keyboard, the ARIA and the focus ring (the reka-ui
           reuse rule), and its own rail is made transparent so the envelope
           shows through - one widget, drawn properly, rather than a second
           one wearing a slider's role. -->
      <div
        class="ap__wave"
        @pointermove="onHover"
        @pointerleave="hoverAt = null"
      >
        <WaveTrack
          :peaks="peaks"
          :duration="span ?? 0"
          :current="at"
          :guards="guards"
          :hover="hoverAt"
        />
        <Slider
          v-model="scrub"
          class="ap__scrub"
          :min="0"
          :max="span ?? 1"
          :step="0.01"
          :disabled="playable === null || span === null"
        />
      </div>
      <span class="ap__t ap__t--end">{{ total ? clock(total) : '--:--' }}</span>
      <Menu>
        <MenuTrigger>
          <Tooltip label="Playback speed">
            <button class="ap__rate" aria-label="Playback speed">{{ rate }}&times;</button>
          </Tooltip>
        </MenuTrigger>
        <MenuContent align="end">
          <MenuItem v-for="r in RATES" :key="r" @select="rate = r">
            <Icon v-if="r === rate" name="check" :size="14" />
            <span :class="{ 'ap__rate-pad': r !== rate }">{{ r }}&times;</span>
          </MenuItem>
        </MenuContent>
      </Menu>
      <Tooltip :label="silent ? 'Unmute' : 'Mute'">
        <button
          class="ap__btn"
          :aria-pressed="muted"
          :aria-label="silent ? 'Unmute' : 'Mute'"
          @click="muted = !muted"
        >
          <Icon :name="silent ? 'volume-off' : 'volume'" :size="15" />
        </button>
      </Tooltip>
      <div ref="volWrap" class="ap__vol">
        <Slider v-model="vol" :min="0" :max="1" :step="0.05" />
      </div>
    </template>
  </div>
</template>

<style scoped>
.ap {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
}
.ap:focus-visible {
  outline: none;
  border-color: var(--pk-accent);
}
.ap--dead {
  color: var(--pk-text-muted);
}
.ap__dead {
  font-size: var(--pk-font-size-xs);
}
.ap__play {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  flex: none;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-surface);
  color: var(--pk-text-primary);
  cursor: pointer;
}
.ap__play:hover:not(:disabled) {
  border-color: var(--pk-accent);
  color: var(--pk-accent-text);
}
.ap__play:disabled {
  opacity: 0.45;
  cursor: default;
}
.ap__btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  flex: none;
  border: none;
  border-radius: var(--pk-radius-sm);
  background: none;
  color: var(--pk-text-muted);
  cursor: pointer;
}
.ap__btn:hover:not(:disabled) {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
.ap__btn:disabled {
  opacity: 0.45;
  cursor: default;
}
.ap__rate {
  flex: none;
  min-width: 34px;
  padding: 2px 4px;
  border: none;
  border-radius: var(--pk-radius-sm);
  background: none;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
  font-variant-numeric: tabular-nums;
  cursor: pointer;
}
.ap__rate:hover {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
/* keeps the labels of unchecked rates aligned under the checked one's text */
.ap__rate-pad {
  margin-left: 24px;
}
.ap__vol {
  width: 60px;
  flex: none;
}
.ap__wave {
  position: relative;
  flex: 1;
  min-width: 60px;
  height: 34px;
  display: flex;
  align-items: center;
}
/* the rail disappears so the envelope is the track; the thumb stays, because
   it is the part that says this is draggable and holds the focus ring */
.ap__wave :deep(.pk-slider) {
  position: relative;
  z-index: 1;
}
.ap__wave :deep(.pk-slider__track) {
  background: transparent;
  height: 34px;
  border-radius: var(--pk-radius-sm);
}
.ap__wave :deep(.pk-slider__range) {
  background: transparent;
}
.ap__wave :deep(.pk-slider__thumb) {
  width: 3px;
  height: 34px;
  border-radius: 2px;
  box-shadow: 0 0 0 1px var(--pk-bg-inset);
}
.ap__wave :deep(.pk-slider__thumb:hover) {
  box-shadow: 0 0 0 1px var(--pk-bg-inset), 0 0 0 4px var(--pk-accent-subtle);
}
.ap__t {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
  min-width: 34px;
  /* the times sit against a 34px waveform now, not a 4px rail; keeping them
     off the top edge is what stops the row looking hung from its numbers */
  align-self: center;
}
.ap__t--end {
  text-align: right;
}
</style>
