<script setup lang="ts">
// The log viewer over the §11.3 stream - one component for the Instrument
// Logs tab and the model page's tail. What "SOTA log UI" means here
// (Vercel/Railway-class): parsed lines (dim time · colored level chip · dim
// module · message), min-level + substring filtering, sticky follow that
// pauses when you scroll up (with a "Latest" jump pill + live indicator),
// and auto-reconnect when the stream drops. The stream itself is live: one
// HTTP response held open by the manager, lines pushed as they are written.
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useStickyScroll } from '@/composables/useStickyScroll'
import { fmtClock } from '@/lib/format'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

const props = withDefaults(
  defineProps<{
    /** 'all' | 'manager' | a runner port (as a string) */
    target: string
    /** viewport height (css); ignored when `fill` is set */
    height?: string
    /** stretch to the bottom of the screen (the last element on its page) */
    fill?: boolean
    /** extra space to leave below in fill mode - card/page padding under
     *  the viewer (px) */
    fillOffset?: number
    /** compact chrome for embeds: live dot + search only, no level pills */
    compact?: boolean
  }>(),
  { height: '60vh', fill: false, fillOffset: 20, compact: false },
)

type Level = 'TRACE' | 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'
interface Line {
  raw: string
  /** merged-mode source prefix: "manager" / "11540" */
  source?: string
  time?: string
  level?: Level
  /** filter-effective level for an unparsed line: inherited from the line
   *  above (continuations), so filtering to Warn+ hides banner art instead
   *  of showing it. Panic/backtrace text is promoted to a real ERROR. */
  eff?: Level
  module?: string
  msg?: string
}

// tracing's default format, with the merged stream's optional [source] prefix:
//   [11540] 2026-08-02T09:58:54.631489Z  INFO paddock_runner::drain: message
const RE =
  /^(?:\[([^\]]+)\]\s+)?(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z?)\s+(TRACE|DEBUG|INFO|WARN|ERROR)\s+([\w:.-]+):\s?(.*)$/

// A runner's stdout is its log file - the manager opens runner-<port>.log and
// hands over the handle at spawn - and tracing's terminal layer used to colour
// it regardless, so runner lines arrive wrapped in SGR escapes. The engine now
// only colours a real terminal, but every log already on disk carries them and
// .prev.log is never rewritten, so clean here as well. Without this the ESC
// bytes draw as missing-glyph boxes AND the line never matches RE at all: no
// clock, no level chip, and nothing for the level filter to select on.
// CSI (what tracing emits), then OSC, then the two-character escapes.
const ANSI = /\u001b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\u0007\u001b]*(?:\u0007|\u001b\\)|[@-Z\\-_])/g
// Anything that survives - a bare ESC, a stray C0 from a crashing child - is
// still an unprintable box. Tab stays; it is the only one that means something.
const CTRL = /[\u0000-\u0008\u000b-\u001f\u007f]/g
function clean(raw: string): string {
  return raw.replace(ANSI, '').replace(CTRL, '')
}

// the log file speaks UTC; the viewer speaks the user's LOCAL clock, in
// fixed ISO 24h form (browser locales can't see the OS format preference)
function localTime(iso: string): string {
  const d = new Date(iso.endsWith('Z') ? iso : `${iso}Z`)
  return Number.isNaN(d.getTime()) ? iso.slice(11, 19) : fmtClock(d)
}

const PANIC = /panicked at|RUST_BACKTRACE|stack backtrace|^thread '/i

/** `lastLevel` threads the previous parsed line's level into continuations. */
function parse(raw: string, lastLevel: Level | undefined): Line {
  const m = RE.exec(raw)
  if (!m) {
    if (PANIC.test(raw)) return { raw, level: 'ERROR' }
    return { raw, eff: lastLevel }
  }
  return { raw, source: m[1], time: localTime(m[2]), level: m[3] as Level, module: m[4], msg: m[5] }
}

const MAX_LINES = 4000 // retained in memory
const MAX_SHOWN = 1500 // rendered rows (newest matching)
const lines = ref<Line[]>([])
const connected = ref(false)

const search = ref('')
const minLevel = ref<'all' | 'info' | 'warn' | 'error'>('all')
const LEVELS = [
  { id: 'all', label: 'All' },
  { id: 'info', label: 'Info+' },
  { id: 'warn', label: 'Warn+' },
  { id: 'error', label: 'Errors' },
] as const
const RANK: Record<string, number> = { TRACE: 0, DEBUG: 1, INFO: 2, WARN: 3, ERROR: 4 }
const MIN: Record<string, number> = { all: 0, info: 2, warn: 3, error: 4 }

const visible = computed(() => {
  const min = MIN[minLevel.value]
  const q = search.value.trim().toLowerCase()
  let out = lines.value
  if (min > 0)
    out = out.filter((l) => {
      const lv = l.level ?? l.eff
      // no level anywhere (banner art, blank noise): only the All view
      return lv !== undefined && RANK[lv] >= min
    })
  if (q) out = out.filter((l) => l.raw.toLowerCase().includes(q))
  return out.length > MAX_SHOWN ? out.slice(-MAX_SHOWN) : out
})

const box = ref<HTMLElement | null>(null)
const content = ref<HTMLElement | null>(null)
const { stuck, toBottom } = useStickyScroll(box, content)

// fill mode: measure where the viewer starts and take everything below it.
// The content above can settle after mount (fleet data landing), so a body
// ResizeObserver re-measures; it converges because measure() is idempotent.
const wrap = ref<HTMLElement | null>(null)
const fillH = ref('60vh')
let bodyRo: ResizeObserver | null = null
let lastH = 0
function measure(): void {
  const el = wrap.value
  if (!el || !props.fill) return
  const top = el.getBoundingClientRect().top
  // scrolled past the viewer: keep the settled height (re-measuring against
  // a scrolled viewport would only ever grow it - a feedback loop)
  if (top < 0) return
  // Integer + hysteresis, or the observer loops: `top` is SUBPIXEL, so the
  // unconditional fractional write resized the box, which resized the body,
  // which re-fired the body ResizeObserver - a silent 60fps relayout storm
  // that held a renderer at ~12-16% of a core with nothing moving on screen
  // (the telemetry view). A 2px dead-band converges every
  // oscillation, scrollbar flips included.
  const next = Math.max(280, Math.round(window.innerHeight - top - props.fillOffset))
  if (Math.abs(next - lastH) <= 2) return
  lastH = next
  fillH.value = `${next}px`
}
onMounted(() => {
  if (!props.fill) return
  measure()
  window.addEventListener('resize', measure)
  bodyRo = new ResizeObserver(() => measure())
  bodyRo.observe(document.body)
})
onUnmounted(() => {
  window.removeEventListener('resize', measure)
  bodyRo?.disconnect()
})
const boxHeight = computed(() => (props.fill ? fillH.value : props.height))

let abort: AbortController | null = null
let retry: number | undefined
let carry = ''
async function start(): Promise<void> {
  stop()
  lines.value = []
  carry = ''
  const a = new AbortController()
  abort = a
  try {
    const res = await fetch(
      `/api/logs?target=${props.target}&follow=true&tail=300&history=true`,
      { signal: a.signal },
    )
    if (!res.ok || !res.body) throw new Error(String(res.status))
    connected.value = true
    const reader = res.body.getReader()
    const dec = new TextDecoder()
    let lastLevel: Level | undefined
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      carry += dec.decode(value, { stream: true })
      const parts = carry.split('\n')
      carry = parts.pop() ?? ''
      const fresh: Line[] = []
      for (const p of parts) {
        // clean before the empty check: a line that was nothing but a colour
        // reset would otherwise render as a blank row
        const raw = clean(p)
        if (!raw.length) continue
        const l = parse(raw, lastLevel)
        if (l.level) lastLevel = l.level
        fresh.push(l)
      }
      if (fresh.length) {
        const next = [...lines.value, ...fresh]
        lines.value = next.length > MAX_LINES ? next.slice(-MAX_LINES) : next
      }
    }
  } catch {
    /* aborted, or the stream dropped */
  }
  connected.value = false
  // stream ended without us leaving (manager restart): reconnect quietly
  if (abort === a) retry = window.setTimeout(() => void start(), 2000)
}
function stop(): void {
  if (retry !== undefined) {
    clearTimeout(retry)
    retry = undefined
  }
  const a = abort
  abort = null
  a?.abort()
  connected.value = false
}
watch(() => props.target, () => void start(), { immediate: true })
onUnmounted(stop)
</script>

<template>
  <div class="lv">
    <div class="lv__bar">
      <span class="lv__live" :class="{ 'lv__live--on': connected }">
        <span class="lv__livedot" /> {{ connected ? 'Live' : 'Connecting...' }}
      </span>
      <div v-if="!compact" class="lv__levels">
        <button
          v-for="l in LEVELS"
          :key="l.id"
          type="button"
          class="lv__level"
          :class="{ 'lv__level--on': minLevel === l.id }"
          @click="minLevel = l.id"
        >
          {{ l.label }}
        </button>
      </div>
      <input v-model="search" class="lv__search" type="search" placeholder="Filter lines..." />
      <span class="lv__count">
        {{ visible.length.toLocaleString() }}<template v-if="visible.length !== lines.length">
          of {{ lines.length.toLocaleString() }}</template>
        lines
      </span>
    </div>

    <div ref="wrap" class="lv__wrap" :style="{ height: boxHeight }">
      <div ref="box" class="lv__scroll">
        <div ref="content" class="lv__lines">
          <div
            v-for="(l, i) in visible"
            :key="i"
            class="lv__line"
            :class="l.level && `lv__line--${l.level.toLowerCase()}`"
          >
            <span v-if="l.source" class="lv__srcc">{{ l.source }}</span>
            <span class="lv__time">{{ l.time ?? '' }}</span>
            <span v-if="l.level" class="lv__lvl" :class="`lv__lvl--${l.level.toLowerCase()}`">
              {{ l.level }}
            </span>
            <Tooltip :label="l.module">
              <span class="lv__msg">{{ l.msg ?? l.raw }}</span>
            </Tooltip>
          </div>
          <p v-if="!visible.length" class="lv__empty">
            {{ lines.length ? 'Nothing matches the filter.' : 'Waiting for log lines...' }}
          </p>
        </div>
      </div>
      <!-- scrolled up = paused (nothing yanks the view); one tap re-arms -->
      <button v-if="!stuck" type="button" class="lv__jump" @click="toBottom">
        <Icon name="arrow-down" :size="13" /> Latest
      </button>
    </div>
  </div>
</template>

<style scoped>
.lv {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}
.lv__bar {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.lv__live {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-muted);
  white-space: nowrap;
}
.lv__livedot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--pk-text-muted);
}
.lv__live--on {
  color: var(--pk-status-success, #4a9);
}
.lv__live--on .lv__livedot {
  background: var(--pk-status-success, #4a9);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--pk-status-success, #4a9) 22%, transparent);
  animation: lv-pulse 2s ease infinite;
}
@keyframes lv-pulse {
  50% {
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--pk-status-success, #4a9) 10%, transparent);
  }
}
.lv__levels {
  display: inline-flex;
  gap: 2px;
  padding: 2px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
}
.lv__level {
  padding: 3px 10px;
  border: none;
  border-radius: var(--pk-radius-sm);
  background: transparent;
  color: var(--pk-text-secondary);
  font: inherit;
  font-size: var(--pk-font-size-xs);
  font-weight: 500;
  cursor: pointer;
}
.lv__level--on {
  background: var(--pk-bg-elevated);
  color: var(--pk-text-primary);
  box-shadow: var(--pk-shadow-sm);
}
.lv__search {
  flex: 1;
  min-width: 140px;
  max-width: 280px;
  padding: 5px 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  color: var(--pk-text-primary);
  font: inherit;
  font-size: var(--pk-font-size-xs);
}
.lv__search:focus {
  outline: none;
  border-color: var(--pk-accent);
}
.lv__count {
  margin-left: auto;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.lv__wrap {
  position: relative;
  min-height: 0;
}
.lv__scroll {
  height: 100%;
  overflow-y: auto;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  /* a CARD surface, like the tables - the inset grey read as a slab in
     light theme */
  background: var(--pk-bg-surface);
}
.lv__lines {
  padding: 8px 0;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  line-height: 1.55;
}
.lv__line {
  display: flex;
  align-items: baseline;
  gap: 10px;
  padding: 1px 12px;
}
.lv__line:hover {
  background: var(--pk-bg-hover);
}
.lv__line--error {
  background: var(--pk-bg-danger-subtle);
}
.lv__srcc {
  flex: none;
  min-width: 5.5ch;
  color: var(--pk-accent-text, var(--pk-accent));
}
.lv__time {
  flex: none;
  min-width: 8ch;
  color: var(--pk-text-muted);
}
.lv__lvl {
  flex: none;
  min-width: 5ch;
  font-weight: 600;
}
.lv__lvl--trace,
.lv__lvl--debug {
  color: var(--pk-text-muted);
}
.lv__lvl--info {
  color: var(--pk-status-success, #4a9);
}
.lv__lvl--warn {
  color: var(--pk-status-warning);
}
.lv__lvl--error {
  color: var(--pk-text-danger);
}
.lv__msg {
  flex: 1;
  min-width: 0;
  color: var(--pk-text-secondary);
  white-space: pre-wrap;
  word-break: break-word;
}
.lv__line--error .lv__msg {
  color: var(--pk-text-danger);
}
.lv__empty {
  padding: 24px;
  text-align: center;
  color: var(--pk-text-muted);
  font-family: var(--pk-font-sans);
}
.lv__jump {
  position: absolute;
  right: 16px;
  bottom: 12px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border: 1px solid var(--pk-border-strong);
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-elevated);
  color: var(--pk-text-primary);
  font: inherit;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  cursor: pointer;
  box-shadow: var(--pk-shadow-lg);
}
.lv__jump:hover {
  border-color: var(--pk-accent);
}
</style>
