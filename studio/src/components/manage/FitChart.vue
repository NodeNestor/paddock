<script setup lang="ts">
// The will-it-fit picture, drawn to scale (Apache ECharts, same tree-shaken
// core as the telemetry dock). One bar = the whole card, segmented into what
// other apps already hold, the model's resident floor (weights + serving
// state), what the SELECTED workload's KV cache costs, and the rest of the
// pool the engine would still take for longer/more conversations. Same
// honesty rule as the estimator: resident decides the fit; the KV pool only
// spends what's left. A does-not-fit model simply overflows the card's edge.
import { computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import { useTheme } from '@/composables/useTheme'
import { fmtVram, fmtTokens } from '@/lib/format'
import Icon from '@/components/Icon.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import type { Estimate, EstimateDevice } from '@/stores/registry'

use([CanvasRenderer, BarChart, GridComponent, TooltipComponent])

const props = defineProps<{
  est: Estimate
  device: EstimateDevice
  ctx: number
  batch: number
  /** KV precision the estimate was PRICED at - the envelope's `kv_dtype`, never
   *  the form control, which can differ on a card that has no FP8 tensor cores.
   *  Only the downgrade note below reads it: the width used to ride along in the
   *  cache band's label, which just repeated the KV precision control sitting a
   *  few rows up the same page. */
  kv?: string
  /** Why the requested width was overridden, when it was. */
  kvDowngraded?: string | null
  /** The ceiling these numbers were priced under (`vram_budget`), in bytes.
   *  Load-bearing for the refusal copy: when a limit is set, "won't fit" is a
   *  fact about the LIMIT, not about the card, and blaming the card on a 48 GB
   *  box with a 20 GB limit sends the reader to buy hardware they already have
    */
  budgetBytes?: number | null
  /** Forensics, when this endpoint serves it. `shared` = it runs on
   *  the model's own GPU (the default) and so adds no resident VRAM here - the
   *  "Shared between models" case. `device` names the card when it was pinned
   *  to a different one, where its footprint lives instead of on this bar. */
  forensics?: { shared: boolean; device: number | null } | null
}>()

const { theme } = useTheme()

const kvWidth = computed(() => (props.kv === 'fp8_e4m3' ? '8-bit' : '16-bit'))

/** The card's maker, for its mark - read off the reported NAME rather than
 *  assumed, so an AMD or Intel box shows no NVIDIA badge. `VendorLogo` already
 *  carries these marks (simple-icons, drawn in currentColor so they read in
 *  both themes); the panel simply never asked for one. Unknown silicon returns
 *  null and the row renders exactly as it did. */
const gpuVendor = computed<string | null>(() => {
  const n = (props.device.name ?? '').toLowerCase()
  if (/nvidia|geforce|rtx|quadro|tesla/.test(n)) return 'NVIDIA'
  return null
})

/** Below this, "Left free" is the crumb the KV pool could not round up into a
 *  page, not spare capacity - reporting 0.7 GB as free next to an 18 GB pool
 *  invited the reading that the card was full when it is in fact fully used on
 *  PURPOSE. */
const FREE_WORTH_SAYING = 1024 ** 3
// What the forensics CUDA context costs when it cannot share the model's - i.e.
// pinned to a different GPU (~557 MiB standalone; 0 incremental when
// it shares the engine's primary context, which is the default). Shown only in
// the cross-GPU tooltip; the shared case adds nothing to this card's bar.
const FORENSICS_FOOTPRINT_MIB = 557

/** Does this configuration's conversation memory actually fit in the pool?
 *
 *  The estimator's per-token rate × context × concurrency is what the workload
 *  wants; `kv_pool` is what the engine will allocate. The old chart took
 *  `min()` of the two, which made overflow invisible: the panel reported the
 *  same "fits" and the same free space for 16K × 1 (a quarter of the pool) and
 *  16K × 64 (four times it). Nothing here clamps any more - past the pool the
 *  engine pages, evicting and refilling sessions, which costs time to first
 *  token rather than failing to start, and that is what gets said. */
const residency = computed(() => {
  const each = props.est.kv_bytes_per_token * props.ctx
  const pool = props.est.kv_pool
  if (!pool || !each) return null
  // How many conversations of this length the pool holds at once. Reported as
  // a count rather than bytes because that is the decision a person is making
  // - "18 GB of cache" answers nothing on its own.
  return {
    want: each * props.batch,
    pool,
    over: each * props.batch > pool,
    holds: Math.max(1, Math.floor(pool / each)),
  }
})

/** The answer the panel is titled with, so nobody has to add the legend up.
 *
 *  It deliberately does not report free VRAM. The KV pool is sized from what
 *  is left after the model loads, so free space is ~0 by construction on every
 *  configuration and quoting it made the panel look identical no matter what
 *  was toggled. What actually moves is how much conversation the card holds,
 *  so that is what gets said. */
const verdict = computed(() => {
  if (props.est.fit?.verdict === 'does_not_fit') {
    // Weights are never the whole story and are usually the part the reader
    // already knows; what surprises is the rest, so name it. This is also the
    // answer to "surely KV offloading fixes this" - none of these terms is
    // the KV cache, which is elastic and never decides the verdict.
    const need = props.est.resident ?? 0
    const weights = (props.est.weights ?? 0) + (props.est.tower ?? 0)
    const rest = Math.max(0, need - weights)
    const ceiling = props.budgetBytes ?? null
    const against = ceiling
      ? `the ${fmtVram(ceiling)} limit set for it`
      : `this card's ${fmtVram(props.device.free ?? 0)} free`
    return {
      tone: 'bad' as const,
      icon: 'x-circle',
      text: "Won't fit",
      detail:
        `needs ${fmtVram(need)} - ${fmtVram(weights)} of weights plus ` +
        `${fmtVram(rest)} of working memory - against ${against}`,
    }
  }
  const r = residency.value
  if (!r) return { tone: 'good' as const, icon: 'check-circle', text: 'Fits', detail: '' }
  // One conversation that does not fit is not a swapping story - there is
  // nothing to swap with. It means the context on this form is longer than the
  // memory left after loading can hold, and on the families that require a
  // full window per slot (qwen3.5/3.6/3.8, gpt-oss) the endpoint REFUSES to
  // START rather than serving a shorter one. Saying "Fits, but not all at
  // once" there reads as "Fits" at a glance and was still on screen for a
  // configuration that had just failed to start.
  if (r.over && props.batch <= 1)
    return {
      tone: 'bad' as const,
      icon: 'x-circle',
      text: "Won't fit at this context",
      detail: `memory left after loading holds about ${fmtTokens(props.est.max_ctx)}, not the ${fmtTokens(props.ctx)} configured - lower the context, or raise how much of the card this model may use`,
    }
  if (r.over)
    return {
      tone: 'warn' as const,
      icon: 'alert-triangle',
      text: 'Fits, but not all at once',
      detail: `memory holds about ${r.holds} of your ${props.batch} conversations at ${fmtTokens(props.ctx)}; the rest are swapped in as they become active, which slows the first reply`,
    }
  return {
    tone: 'good' as const,
    icon: 'check-circle',
    text: 'Fits',
    detail: `room for about ${r.holds} conversation${r.holds === 1 ? '' : 's'} at ${fmtTokens(props.ctx)}`,
  }
})

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

interface Seg {
  name: string
  bytes: number
  color: string
  /** one plain sentence on what this memory is - chart hover + legend title */
  desc: string
}

/** `total - free`: what the card is holding that this model cannot have.
 *
 *  Named for the STATE, not the verdict. "Unavailable" said what it meant to
 *  the estimator and nothing about the world - memory is unavailable because
 *  something is using it, and on a box running several paddock endpoints that
 *  something is mostly the user's own other models (the row's tooltip splits
 *  it). "Currently used" is the same number said from the user's side, and it
 *  is also honest about being a snapshot: stop a model and it drops.
 *  (Earlier still it read "Other apps", which mis-attributed your own fleet to
 *  foreign programs.) */
const FOREIGN = 'Currently used'
/** The part of the card the endpoint's own limit puts out of reach.
 *
 *  Split out because `free` is min(total - in_use_by_others, budget),
 *  and this chart drew `total - free` as "Currently used" - so setting a
 *  25 GB limit on a 48 GB card reported 23 GB "currently used" while the card
 *  actually held 4.3. A policy number wearing a measurement's label, which any
 *  reader will check against nvidia-smi and catch (the maintainer did). Reserving a
 *  ceiling is not consumption, and the two must never share a row. */
const CAPPED = 'Outside your limit'

/** The legend, folded to what a person is actually weighing up: someone else's
 *  memory, this model's, and the conversations'. Weights / encoder / working
 *  memory / engine overhead are all "what having this model loaded costs" and
 *  nobody trades one against another, so they collapse into one row that names
 *  its parts inline. Six numbers to add up became three to read; the bar above
 *  still carries every segment in its own colour. */
/** The overhead row's tooltip: every term that is actually there, largest
 *  first, so the reader can see which one to act on. A term the model does not
 *  have (no recurrent state, no speculation) is omitted rather than shown as
 *  zero - a list of zeroes is noise, and the ones that are there are the
 *  point. Falls back to the old sentence when an older manager sends no
 *  breakdown. */
const overheadDesc = computed(() => {
  const o = props.est.overhead_parts
  const fallback =
    'the fixed cost of serving besides the weights: CUDA runtime, the decode graph, and per-conversation working state'
  if (!o) return fallback
  const terms: [string, number][] = [
    ['CUDA runtime + decode graph', o.fixed],
    ['prefix checkpoints', o.prefix_checkpoints],
    ['allocator rounding', o.allocator_slack],
    ['speculation state', o.spec_state],
    ['convolution scratch', o.conv_scratch],
    ['logits', o.logits],
    ['block tables', o.block_tables],
    ['KV offload staging', o.offload_staging],
  ]
  const shown = terms.filter(([, b]) => b > 0).sort((a, b) => b[1] - a[1])
  if (!shown.length) return fallback
  return shown.map(([n, b]) => `${n} ${fmtVram(b)}`).join(' · ')
})

const rows = computed(() => {
  const segs = parts.value.segs
  const at = (n: string) => segs.find((s) => s.name === n)
  // Everything that is this MODEL's footprint. Both of the not-ours
  // segments have to be named here: when `CAPPED` was added and only FOREIGN
  // was excluded, the headroom outside the user's own limit was summed into
  // the model's own bar - a 27B whose parts read 17 + 7.5 + 0.9 reported a
  // 51 GB total, because a 24 GB cap on a 48 GB card put 22.6 GB of "not
  // yours" on the model's side of the ledger.
  const NOT_OURS = [FOREIGN, CAPPED]
  const modelSegs = segs.filter(
    (s) => !NOT_OURS.includes(s.name) && !s.name.startsWith('Conversation'),
  )
  const out: { name: string; bytes: number; color: string; desc: string }[] = []
  const other = at(FOREIGN)
  if (other) {
    const pinned = props.device.held_by_pinned ?? 0
    out.push({
      name: other.name,
      bytes: other.bytes,
      color: other.color,
      // the split belongs in the tooltip, where someone who wonders can look,
      // rather than on a row everyone has to read past
      desc:
        pinned > 0
          ? `${fmtVram(pinned)} is your other models; the rest is other programs. This model cannot have any of it.`
          : other.desc,
    })
  }
  // its own row, right after the other not-ours one: a band in the bar with
  // no legend entry is exactly the kind of unexplained number this panel is
  // supposed to stop producing
  const cap = at(CAPPED)
  if (cap && cap.bytes > 0) {
    out.push({ name: cap.name, bytes: cap.bytes, color: cap.color, desc: cap.desc })
  }
  const total = modelSegs.reduce((t, s) => t + s.bytes, 0)
  if (total > 0)
    out.push({
      name: 'This model',
      bytes: total,
      color: at('Model weights')?.color ?? '',
      desc: modelSegs.map((m) => `${m.name.toLowerCase()} ${fmtVram(m.bytes)}`).join(' · '),
    })
  const cache = segs.find((s) => s.name.startsWith('Conversation'))
  if (cache)
    out.push({
      name: 'Conversations',
      bytes: cache.bytes,
      color: cache.color,
      desc: 'allocated in full when the server starts and unavailable to anything else - it holds the conversations in flight, and takes whatever the model leaves',
    })
  return out
})

const parts = computed<{ segs: Seg[]; free: number }>(() => {
  void theme.value // recompute colours on theme toggle
  // Five distinct hues from the theme's categorical ramp, not one accent at
  // four alphas - these segments touch, and a tinted single hue made weights,
  // encoder and overhead read as one band. The ramp is defined per theme, so
  // this stays legible on both grounds.
  const cWeights = cssVar('--pk-chart-weights') || '#38bdf8'
  const cEncoder = cssVar('--pk-chart-encoder') || '#a78bfa'
  const cWorkspace = cssVar('--pk-chart-workspace') || '#f472b6'
  const cOverhead = cssVar('--pk-chart-overhead') || '#7c93a8'
  const green = cssVar('--pk-chart-cache') || '#34d399'
  const muted = cssVar('--pk-chart-foreign') || '#64748b'
  const e = props.est
  const d = props.device
  // total - free = whatever paddock can't reclaim for this model (desktop
  // session, other processes) - the fit is judged against `free`, so this is
  // the part of the card that was never on offer.
  // What is genuinely OCCUPIED, from the manager's own measurement, and what
  // is merely out of reach because this endpoint was capped. `total - free`
  // is both added together, and reporting the sum as "used" is a lie the
  // reader can disprove in one glance at nvidia-smi.
  const unavailable = Math.max(0, d.total - d.free)
  const occupied = Math.min(unavailable, Math.max(0, d.used_by_others ?? unavailable))
  const capped = Math.max(0, unavailable - occupied)
  const segs: Seg[] = [
    // Not "other apps": this is memory the card is really holding - which on
    // a box running several paddock endpoints is mostly your own other
    // models. Naming it after foreign programs mis-attributed your own fleet,
    // and the manager had already broken out `held_by_pinned` so a panel
    // could say which is which.
    {
      name: FOREIGN,
      bytes: occupied,
      color: muted,
      desc: 'held by your other models and by other programs - this model cannot have it',
    },
    {
      name: CAPPED,
      bytes: capped,
      color: muted,
      desc: 'free on the card, but outside the memory limit set for this model - raise the limit to use it',
    },
    {
      name: 'Model weights',
      bytes: e.weights,
      color: cWeights,
      desc: 'the model file itself, loaded onto the GPU',
    },
    // The vision tower is weights too, and big enough to matter (~1.1 GB on
    // granite-vision), so it gets its own segment - folding it into "engine
    // overhead" would be a true total telling a false story about where the
    // memory went. Filtered out at 0 bytes, so text-only models are unchanged.
    {
      name: 'Vision/audio encoder',
      bytes: e.tower ?? 0,
      color: cEncoder,
      desc: 'the image or speech encoder, loaded alongside the model and held for as long as it runs - a model that can see or hear always pays this, whether or not you send pictures or audio',
    },
    // Declared serving scratch (mixture-of-experts staging) - its own segment
    // for the same reason as the tower: ~5.8 GB on gemma-4-26B-A4B is too big
    // to hide inside "engine overhead". Filtered out at 0 for everyone else.
    {
      name: 'Model working memory',
      bytes: e.workspace ?? 0,
      color: cWorkspace,
      desc: 'working memory this model pins for serving beyond its weights (expert staging on mixture-of-experts models) - measured, and held for as long as it runs',
    },
    // state + scratch + CUDA context, lumped: the rest of the must-fit floor.
    // One bar, but its tooltip names every term - "how can a 17GB model have
    // 11GB engine overhead" is a fair question that an unbroken total cannot
    // answer, and the parts are already computed.
    {
      name: 'Engine overhead',
      bytes: Math.max(0, e.resident - e.weights - (e.tower ?? 0) - (e.workspace ?? 0)),
      color: cOverhead,
      desc: overheadDesc.value,
    },
    // One segment, not the old cache/headroom pair. The pool is a single
    // allocation taken in full at start - splitting it into "what your setting
    // uses" and "the rest" invited "why is 7.5 GB wasted?" when the answer is
    // "it isn't, that's your room to grow", and it hid the case that matters:
    // the old `min(needed, pool)` clamp meant a 4x overcommit rendered as a
    // full bar with the headroom row simply vanishing, which reads as less
    // cost. Residency is now said in words instead (`fitLine`).
    {
      name: 'Conversation memory',
      bytes: e.kv_pool,
      color: green,
      desc: 'the KV pool, allocated in full when the server starts and unavailable to anything else from that moment - it holds the conversations in flight',
    },
  ].filter((s) => s.bytes > 0)
  const free = Math.max(0, d.total - unavailable - e.resident - e.kv_pool)
  return { segs, free }
})

const option = computed(() => {
  void theme.value
  const inset = cssVar('--pk-bg-inset') || 'rgba(128,128,128,0.12)'
  const elevated = cssVar('--pk-bg-elevated') || '#1b1b1b'
  const strong = cssVar('--pk-border-strong') || '#333'
  const primary = cssVar('--pk-text-primary') || '#fff'
  const muted = cssVar('--pk-text-muted') || '#888'
  const { segs, free } = parts.value
  const n = segs.length
  return {
    animation: false,
    grid: { left: 0, right: 0, top: 0, bottom: 0 },
    // the axis is the card: a resident floor bigger than the card clips at
    // the right edge, which is exactly the honest picture
    xAxis: { type: 'value', min: 0, max: props.device.total, show: false },
    yAxis: { type: 'category', data: [''], show: false },
    tooltip: {
      trigger: 'item',
      backgroundColor: elevated,
      borderColor: strong,
      borderWidth: 1,
      padding: [6, 10],
      textStyle: { color: primary, fontSize: 12 },
      extraCssText: 'max-width: 240px; white-space: normal;',
      formatter: (p: { seriesName: string; value: number }) => {
        const desc = segs.find((s) => s.name === p.seriesName)?.desc ?? ''
        return (
          `<span style="color:${muted};font-size:11px">${p.seriesName}</span><br/>` +
          `<b>${fmtVram(p.value)}</b>` +
          (desc ? `<br/><span style="color:${muted};font-size:11px">${desc}</span>` : '')
        )
      },
    },
    series: segs.map((s, i) => ({
      type: 'bar',
      stack: 'vram',
      name: s.name,
      data: [s.bytes],
      barWidth: 16,
      itemStyle: {
        color: s.color,
        borderRadius:
          n === 1
            ? [4, 4, 4, 4]
            : i === 0
              ? [4, 0, 0, 4]
              : i === n - 1 && free === 0
                ? [0, 4, 4, 0]
                : 0,
      },
      // the free remainder shows as the track, not a segment
      ...(i === 0 ? { showBackground: true, backgroundStyle: { color: inset, borderRadius: 4 } } : {}),
    })),
  }
})
</script>

<template>
  <div class="fitc">
    <div class="fitc__gpu">
      <span class="fitc__gpu-name">
        <VendorLogo v-if="gpuVendor" :vendor="gpuVendor" :size="13" class="fitc__gpu-mark" />
        {{ device.name ?? 'GPU' }}
      </span>
      <span class="fitc__gpu-total">{{ fmtVram(device.total) }}</span>
    </div>
    <VChart class="fitc__bar" :option="option" autoresize />
    <p class="fitc__verdict" :class="`fitc__verdict--${verdict.tone}`">
      <Icon :name="verdict.icon" :size="16" class="fitc__verdict-icon" />
      <span class="fitc__verdict-body">
        <span class="fitc__verdict-word">{{ verdict.text }}</span>
        <span v-if="verdict.detail" class="fitc__verdict-detail">{{ verdict.detail }}</span>
      </span>
    </p>
    <p v-if="kvDowngraded" class="fitc__note">
      Priced at {{ kvWidth }}: {{ kvDowngraded }}, so the server will serve
      {{ kvWidth }} whatever is asked.
    </p>
    <ul class="fitc__legend">
      <Tooltip v-for="s in rows" :key="s.name" :label="s.desc">
        <li class="fitc__row">
          <span class="fitc__chip" :style="{ background: s.color }" />
          <span class="fitc__name">{{ s.name }}</span>
          <span class="fitc__val">{{ fmtVram(s.bytes) }}</span>
        </li>
      </Tooltip>
      <Tooltip
        v-if="forensics"
        :label="
          forensics.shared
            ? 'Forensics shares the model\'s GPU context - it adds no resident VRAM on this card.'
            : `Forensics is pinned to GPU ${forensics.device} - its ~${FORENSICS_FOOTPRINT_MIB} MiB context lives there, not on this card.`
        "
      >
        <li class="fitc__row">
          <span class="fitc__chip fitc__chip--forensics" />
          <span class="fitc__name">Forensics</span>
          <span class="fitc__val">{{
            forensics.shared ? 'Shared between models' : `On GPU ${forensics.device}`
          }}</span>
        </li>
      </Tooltip>
      <Tooltip v-if="parts.free > FREE_WORTH_SAYING" label="VRAM this configuration leaves untouched">
        <li class="fitc__row">
          <span class="fitc__chip fitc__chip--free" />
          <span class="fitc__name">Left free</span>
          <span class="fitc__val">{{ fmtVram(parts.free) }}</span>
        </li>
      </Tooltip>
    </ul>
  </div>
</template>

<style scoped>
.fitc {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 10px;
}
.fitc__gpu {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}
.fitc__gpu-mark {
  flex-shrink: 0;
  /* the mark sits with the name, not as a second thing beside it - muted to
     the same weight as the text so it reads as a typographic detail rather
     than a badge competing with the verdict below */
  color: var(--pk-text-muted);
}
.fitc__gpu-name {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fitc__gpu-total {
  flex-shrink: 0;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.fitc__bar {
  width: 100%;
  height: 16px;
}
/* No box, no tint, no rule down the side - an accent rail on a tinted card is
   the house style of every dashboard ever generated, and it dresses a sentence
   up as a component. The verdict earns its place typographically instead: it
   is the largest text in the panel, it is the only coloured text, and it has
   air above it. Colour + scale + space, which is what separates a headline
   from a caption on paper too. */
.fitc__verdict {
  display: flex;
  align-items: flex-start;
  gap: 7px;
  margin: 10px 0 4px;
}
.fitc__verdict--good {
  --vt: var(--pk-status-success);
}
.fitc__verdict--warn {
  --vt: var(--pk-status-warning);
}
.fitc__verdict--bad {
  --vt: var(--pk-status-error);
}
.fitc__verdict-icon {
  flex-shrink: 0;
  margin-top: 3px;
  color: var(--vt);
}
.fitc__verdict-body {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.fitc__verdict-word {
  font-size: var(--pk-font-size-lg);
  font-weight: 620;
  line-height: 1.15;
  letter-spacing: -0.01em;
  color: var(--vt);
}
.fitc__verdict-detail {
  font-size: var(--pk-font-size-xs);
  line-height: 1.4;
  color: var(--pk-text-secondary);
}
.fitc__note {
  margin: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.fitc__legend {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.fitc__row {
  display: grid;
  grid-template-columns: 10px minmax(0, 1fr) auto;
  align-items: center;
  gap: 8px;
  font-size: var(--pk-font-size-xs);
}
/* the residency line is a reading of the pool above it, not a cost of its own */
.fitc__row--sub {
  color: var(--pk-text-muted);
}
.fitc__chip--none {
  background: none;
}
.fitc__chip {
  width: 10px;
  height: 10px;
  border-radius: 3px;
}
.fitc__chip--free {
  background: var(--pk-bg-inset);
  border: 1px solid var(--pk-border-subtle);
}
.fitc__chip--forensics {
  /* hollow, dashed: an informational row that costs no VRAM on this card */
  background: transparent;
  border: 1px dashed var(--pk-text-muted);
}
.fitc__name {
  color: var(--pk-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fitc__val {
  font-family: var(--pk-font-mono);
  color: var(--pk-text-primary);
}
</style>
