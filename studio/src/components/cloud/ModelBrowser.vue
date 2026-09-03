<script setup lang="ts">
// The provider model browser: smart search + sort tabs + capability filters
// over a normalized model list. Owns presentation and selection only - no
// fetching - so the Cloud page's always-open OpenRouter browse and each
// endpoint card's picker are the same component. All hover facts ride the
// Reka Tooltip wrapper (ui reuse rule), never native title attributes.
import { computed, reactive } from 'vue'
import { cloudVendor, type CloudModelPick } from '@/stores/models'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'

const props = defineProps<{
  models: CloudModelPick[]
  /** the list order is last-week usage - label the default tab Trending */
  ranked?: boolean
  loading?: boolean
  error?: string | null
  /** maker implied by the tab's endpoint when the id names none (native
   *  OpenAI/Anthropic lists are bare ids - o3-mini says nothing) */
  vendor?: string
  /** ids already enabled - rendered as "in pickers" instead of Add */
  enabled: string[]
  /** provider name for the loading line */
  name?: string
  /** stretch to the container: the list takes the remaining height instead
   *  of a fixed cap (the tabbed card's main browse; pickers stay capped) */
  fill?: boolean
  /** rows expand into the per-provider breakdown (OpenRouter only: one model,
   *  many providers, each with its own price/context/quant/throughput) */
  expandable?: boolean
}>()
const emit = defineEmits<{ add: [m: CloudModelPick] }>()

// ── the expandable provider breakdown (lazy, cached per model) ──────────────
interface ProviderRow {
  name: string
  /** the endpoint's unique slug (regional/tier variants of one brand differ
   *  only here) - also what a pin routes on */
  tag?: string
  ctx?: number
  promptPrice?: number
  completionPrice?: number
  quant?: string
  maxOut?: number
  tps?: number
}
/** what a pin stores/routes on */
function provSlug(p: ProviderRow): string {
  return p.tag ?? p.name
}
/** "amazon-bedrock/us-east-1" -> the variant part that tells twins apart */
function provVariant(p: ProviderRow): string | null {
  const t = p.tag ?? ''
  const slash = t.indexOf('/')
  return slash > 0 ? t.slice(slash + 1) : null
}
const expanded = reactive<Record<string, boolean>>({})
const details = reactive<
  Record<string, { loading: boolean; error: string | null; provs: ProviderRow[] }>
>({})
async function toggleRow(m: CloudModelPick): Promise<void> {
  if (!props.expandable) return
  expanded[m.id] = !expanded[m.id]
  if (!expanded[m.id] || details[m.id]) return
  details[m.id] = { loading: true, error: null, provs: [] }
  try {
    const res = await fetch(`/api/cloud/browse/endpoints?model=${encodeURIComponent(m.id)}`)
    const j = (await res.json()) as { providers?: ProviderRow[]; error?: { message?: string } }
    if (!res.ok) {
      details[m.id].error = j?.error?.message ?? `OpenRouter did not answer (HTTP ${res.status}).`
      return
    }
    details[m.id].provs = j.providers ?? []
  } catch (e) {
    details[m.id].error = e instanceof Error ? e.message : String(e)
  } finally {
    details[m.id].loading = false
  }
}

// `enabled` carries PICK KEYS: the bare model id for an auto-routed pick,
// `id@Provider` for a provider-pinned one - both can coexist for one model.
const enabledSet = computed(() => new Set(props.enabled))

// Ordering is the TABLE's job (the Instrument th-sort idiom): Model, price
// and context sort by clicking their headers; the chips keep only the two
// orders that aren't columns (trending, newest) plus the capability filters.
type SortKey = 'trending' | 'newest' | 'model' | 'price' | 'ctx'
const view = reactive<{
  query: string
  sort: SortKey
  dir: 1 | -1
  fVision: boolean
  fReasoning: boolean
  fAsr: boolean
  fFree: boolean
}>({
  query: '',
  sort: 'trending',
  dir: 1,
  fVision: false,
  fReasoning: false,
  fAsr: false,
  fFree: false,
})

function setSort(k: SortKey): void {
  if (view.sort === k) {
    view.dir = view.dir === 1 ? -1 : 1
  } else {
    view.sort = k
    // sensible first direction per column: cheapest first, biggest ctx first
    view.dir = k === 'ctx' ? -1 : 1
  }
}
/** aria-sort + the header arrow for a sortable column. */
function sortState(k: SortKey): 'ascending' | 'descending' | undefined {
  if (view.sort !== k) return undefined
  return view.dir === 1 ? 'ascending' : 'descending'
}

const sortTabs = computed<{ key: SortKey; label: string }[]>(() => {
  const t: { key: SortKey; label: string }[] = [
    { key: 'trending', label: props.ranked ? 'Trending' : 'All' },
  ]
  if (props.models.some((m) => m.created)) t.push({ key: 'newest', label: 'Newest' })
  return t
})
const hasVision = computed(() => props.models.some((m) => m.vision !== undefined))
const hasReasoning = computed(() => props.models.some((m) => m.reasoning))
// Speech models are a NEEDLE: 14 of them among 400+, and nothing in an id like
// `deepgram/nova-3` or `google/chirp-3` says "this transcribes".
// They get the same mark-and-chip treatment as vision.
const hasAsr = computed(() => props.models.some((m) => m.asr))
const hasFree = computed(() => props.models.some((m) => m.free))
// Columns exist only where the list carries the data: Anthropic/OpenAI model
// lists have bare ids, and a table of empty price/context/capability cells
// under full headers reads as broken.
const hasPrice = computed(() =>
  props.models.some((m) => m.free || m.promptPrice != null || m.completionPrice != null),
)
const hasCtx = computed(() => props.models.some((m) => m.ctx))
const visionCol = computed(() => props.models.some((m) => m.vision))
const colCount = computed(
  () =>
    3 +
    (hasPrice.value ? 1 : 0) +
    (hasCtx.value ? 1 : 0) +
    (visionCol.value ? 1 : 0) +
    (hasReasoning.value ? 1 : 0) +
    (hasAsr.value ? 1 : 0),
)

// ── smart search: multi-token, alias-expanded, ranked ───────────────────────
// Users type the model brand, not the org slug - "gemini" should surface
// google/*, "grok" x-ai/* - so each query token also tries its org aliases.
const ALIASES: Record<string, string[]> = {
  claude: ['anthropic'],
  gpt: ['openai'],
  oai: ['openai'],
  gemini: ['google'],
  gemma: ['google'],
  llama: ['meta'],
  grok: ['x-ai'],
  kimi: ['moonshotai'],
  glm: ['z-ai'],
  qwen: ['alibaba'],
}
/** Words someone types when they mean "speech to text".
 *
 *  The capability is nowhere in the name - `deepgram/nova-3`,
 *  `nvidia/parakeet-tdt-0.6b-v3` and `google/chirp-3` are unguessable, and
 *  only the four `openai/whisper-*` are self-describing. So the FLAG is
 *  searchable and not merely filterable: the chip finds them once you know it
 *  is there, this finds them when you type what you were already thinking. */
const ASR_WORDS = ['speech', 'voice', 'audio', 'transcribe', 'transcription', 'stt', 'asr']

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
/** 0 = no match; higher = better (prefix > word boundary > substring > blurb). */
function tokenScore(m: CloudModelPick, tok: string): number {
  const id = m.id.toLowerCase()
  const name = (m.display ?? '').toLowerCase()
  // Scored like a word-boundary hit, not a prefix: a model actually CALLED
  // whisper should still outrank one that merely does the same job.
  let best = m.asr && ASR_WORDS.includes(tok) ? 2 : 0
  for (const t of [tok, ...(ALIASES[tok] ?? [])]) {
    if (id.startsWith(t) || name.startsWith(t)) return 3
    const bound = new RegExp(`[/\\-_ .:]${escapeRe(t)}`)
    if (bound.test(id) || bound.test(name)) best = Math.max(best, 2)
    else if (id.includes(t) || name.includes(t)) best = Math.max(best, 1)
    else if ((m.blurb ?? '').toLowerCase().includes(t)) best = Math.max(best, 0.5)
  }
  return best
}
function costOf(m: CloudModelPick): number {
  // A per-minute audio rate is not on the same scale as a per-token price, so
  // it does not belong in a cheapest-first ordering of token costs - sorting
  // them together would rank whisper as the most expensive model on the page.
  // Unknown is the honest answer, and it already sorts to the end.
  if (billsByAudio(m)) return Number.POSITIVE_INFINITY
  if (m.promptPrice == null && m.completionPrice == null) return Number.POSITIVE_INFINITY
  return (m.promptPrice ?? 0) + (m.completionPrice ?? 0)
}

// The list is capped for the DOM and SAYS so (no silent truncation).
const CAP = 60
const visible = computed(() => {
  let rows = props.models.map((m, i) => ({ m, i, s: 0 }))
  if (view.fVision) rows = rows.filter((x) => x.m.vision)
  if (view.fReasoning) rows = rows.filter((x) => x.m.reasoning)
  if (view.fAsr) rows = rows.filter((x) => x.m.asr)
  if (view.fFree) rows = rows.filter((x) => x.m.free)
  const toks = view.query.trim().toLowerCase().split(/\s+/).filter(Boolean)
  if (toks.length) {
    rows = rows.flatMap((x) => {
      let sum = 0
      for (const t of toks) {
        const v = tokenScore(x.m, t)
        if (!v) return []
        sum += v
      }
      return [{ ...x, s: sum }]
    })
  }
  rows.sort((a, b) => {
    if (b.s !== a.s) return b.s - a.s
    switch (view.sort) {
      case 'newest':
        return (b.m.created ?? 0) - (a.m.created ?? 0)
      case 'model':
        return view.dir * rowName(a.m).localeCompare(rowName(b.m))
      case 'price':
        return view.dir * (costOf(a.m) - costOf(b.m))
      case 'ctx':
        return view.dir * ((a.m.ctx ?? 0) - (b.m.ctx ?? 0))
      default:
        return a.i - b.i
    }
  })
  return { shown: rows.slice(0, CAP).map((x) => x.m), total: rows.length }
})

// ── row facts: $/M, context, capability marks ───────────────────────────────
function fmtPerM(perToken: number): string {
  const v = perToken * 1e6
  if (v === 0) return '$0'
  if (v >= 100) return `$${Math.round(v)}`
  return `$${Number(v.toFixed(v >= 1 ? 2 : 3))}`
}
/** A speech model that bills by AUDIO rather than by token.
 *
 *  OpenRouter puts an audio rate in `pricing.prompt` and leaves `completion`
 *  at 0, and states the unit nowhere - checked against every
 *  pricing key it publishes (`audio`, `audio_output`, `completion`, `image`,
 *  `image_output`, `input_audio_cache`, `input_cache_*`, `internal_reasoning`,
 *  `overrides`, `prompt`, `web_search`); not one carries a unit. So the shape
 *  of the row is the only signal there is, and it is a clean one: the two
 *  models that genuinely bill per token (gpt-4o-transcribe and its mini) are
 *  exactly the two with a non-zero completion price.
 *
 *  Reading these as per-token printed "$6000" beside whisper-1, live.
 *
 *  AND the UNIT is not KNOWABLE from the API. It looked per-minute - whisper-1
 *  at 0.006 and deepgram/nova-3 at 0.0043 match those providers' published
 *  per-minute rates to the digit. But OpenRouter's own model pages show
 *  whisper-1 as "$0.006/minute" and microsoft/mai-transcribe-1.5 as
 *  "$0.36/hour": the same field, the same unscaled number, two different
 *  units, chosen per model and published nowhere the API can reach. Printing
 *  "/min" on all of them would read 60x cheap on the per-hour ones.
 *
 *  So the rate rides with its billing basis named and its unit not invented.
 *  The number that actually settles it is measured, not listed: the usage
 *  ledger records the provider's own `cost` on the first real transcription. */
function billsByAudio(m: CloudModelPick): boolean {
  return !!m.asr && !m.completionPrice
}

/** An audio rate, kept legible down to the very cheap ones - 0.000035 must
 *  not round away to $0.00. No unit: see `billsByAudio`. */
function fmtRate(v: number): string {
  if (v === 0) return '$0'
  return v >= 0.1 ? `$${v.toFixed(2)}` : `$${Number(v.toPrecision(2))}`
}

/** What the audio rate's hover says - the whole caveat, once, where someone
 *  deciding on price will meet it. */
function audioPriceTip(m: CloudModelPick): string {
  const r = m.promptPrice != null ? fmtRate(m.promptPrice) : 'an unlisted rate'
  return `Billed by audio duration at ${r}. OpenRouter publishes the unit - per minute for some models, per hour for others - only on the model's own page, so it is not shown here. The exact cost of a transcription is recorded when you run one.`
}

function priceLabel(m: CloudModelPick): string | null {
  if (m.free) return null
  if (billsByAudio(m)) return m.promptPrice != null ? `${fmtRate(m.promptPrice)} audio` : null
  if (m.promptPrice == null || m.completionPrice == null) return null
  return `${fmtPerM(m.promptPrice)} · ${fmtPerM(m.completionPrice)}`
}
function fmtCtx(n: number): string {
  return n >= 1_000_000 ? `${Number((n / 1e6).toFixed(1))}M` : `${Math.round(n / 1000)}K`
}
/** The row shows the model's NAME only - when the vendor mark already names
 *  the maker, OpenRouter's "DeepSeek: DeepSeek V4 Flash" prefix is noise. The
 *  technical id lives in the name's hover, with the description. */
function rowName(m: CloudModelPick): string {
  const d = m.display ?? m.id
  return cloudVendor(m.id) ? d.replace(/^[^:]{2,24}:\s+/, '') : d
}
function rowTip(m: CloudModelPick): string {
  return m.blurb ? `${m.id} · ${m.blurb}` : m.id
}
function rowVendor(m: CloudModelPick): string | undefined {
  return cloudVendor(m.id) ?? props.vendor
}

// A dozen models need no search hero or filter chips - Anthropic's list is
// ten rows, and chrome taller than the content it filters is noise. The
// whole catalog (OpenRouter's hundreds) keeps the full apparatus.
const slim = computed(() => props.models.length <= 12)

</script>

<template>
  <div class="mb" :class="{ 'mb--fill': fill }">
    <input
      v-if="!slim"
      v-model="view.query"
      class="pk-input mb__search"
      placeholder="Search"
      spellcheck="false"
    />
    <p v-if="loading" class="mb__hint">
      <Icon name="spinner" :size="13" class="mb__spin" /> Asking {{ name ?? 'the provider' }}...
    </p>
    <p v-else-if="error" class="mb__error">{{ error }}</p>
    <template v-else>
      <div v-if="!slim && (sortTabs.length > 1 || hasVision || hasReasoning || hasFree)" class="mb__tools">
        <button
          v-for="t in sortTabs"
          :key="t.key"
          type="button"
          class="mb__chip"
          :class="{ 'mb__chip--on': view.sort === t.key }"
          @click="view.sort = t.key"
        >
          {{ t.label }}
        </button>
        <span v-if="hasVision || hasReasoning || hasAsr || hasFree" class="mb__tools-sep" />
        <button
          v-if="hasVision"
          type="button"
          class="mb__chip"
          :class="{ 'mb__chip--on': view.fVision }"
          :aria-pressed="view.fVision"
          @click="view.fVision = !view.fVision"
        >
          <Icon name="eye" :size="12" /> Vision
        </button>
        <button
          v-if="hasReasoning"
          type="button"
          class="mb__chip"
          :class="{ 'mb__chip--on': view.fReasoning }"
          :aria-pressed="view.fReasoning"
          @click="view.fReasoning = !view.fReasoning"
        >
          <Icon name="brain" :size="12" /> Reasoning
        </button>
        <button
          v-if="hasAsr"
          type="button"
          class="mb__chip"
          :class="{ 'mb__chip--on': view.fAsr }"
          :aria-pressed="view.fAsr"
          @click="view.fAsr = !view.fAsr"
        >
          <Icon name="microphone" :size="12" /> Speech
        </button>
        <button
          v-if="hasFree"
          type="button"
          class="mb__chip"
          :class="{ 'mb__chip--on': view.fFree }"
          :aria-pressed="view.fFree"
          @click="view.fFree = !view.fFree"
        >
          Free
        </button>
      </div>
      <div class="mb__tablewrap">
        <table class="mb__table">
          <thead>
            <tr>
              <th class="mb__th-logo" />
              <th class="th-sort" :aria-sort="sortState('model')" @click="setSort('model')">
                Model
                <Icon
                  v-if="view.sort === 'model'"
                  :name="view.dir === 1 ? 'chevron-up' : 'chevron-down'"
                  :size="11"
                />
              </th>
              <th
                v-if="hasPrice"
                class="th-sort mb__th-price"
                :aria-sort="sortState('price')"
                @click="setSort('price')"
              >
                {{ hasAsr ? 'Price' : '$ in · out /M' }}
                <Icon
                  v-if="view.sort === 'price'"
                  :name="view.dir === 1 ? 'chevron-up' : 'chevron-down'"
                  :size="11"
                />
              </th>
              <th
                v-if="hasCtx"
                class="th-sort mb__th-ctx"
                :aria-sort="sortState('ctx')"
                @click="setSort('ctx')"
              >
                Context
                <Icon
                  v-if="view.sort === 'ctx'"
                  :name="view.dir === 1 ? 'chevron-up' : 'chevron-down'"
                  :size="11"
                />
              </th>
              <th v-if="visionCol" class="mb__th-mark">
                <Tooltip label="Reads images"><span><Icon name="eye" :size="13" /></span></Tooltip>
              </th>
              <th v-if="hasReasoning" class="mb__th-mark">
                <Tooltip label="Reasoning model">
                  <span><Icon name="brain" :size="13" /></span>
                </Tooltip>
              </th>
              <th v-if="hasAsr" class="mb__th-mark">
                <Tooltip label="Turns speech into text">
                  <span><Icon name="microphone" :size="13" /></span>
                </Tooltip>
              </th>
              <th class="mb__th-act" />
            </tr>
          </thead>
          <tbody>
            <template v-for="m in visible.shown" :key="m.id">
              <tr :class="{ 'mb__row-x': expandable }" @click="toggleRow(m)">
                <td class="mb__td-logo">
                  <VendorLogo v-if="rowVendor(m)" :vendor="rowVendor(m)!" :size="15" />
                </td>
                <td class="mb__td-name">
                  <Icon
                    v-if="expandable"
                    :name="expanded[m.id] ? 'chevron-down' : 'chevron-right'"
                    :size="12"
                    class="mb__caret"
                  />
                  <Tooltip :label="rowTip(m)">
                    <span class="mb__name">{{ rowName(m) }}</span>
                  </Tooltip>
                </td>
                <td v-if="hasPrice" class="mb__td-r">
                  <span v-if="m.free" class="mb__free">free</span>
                  <Tooltip v-else-if="billsByAudio(m) && priceLabel(m)" :label="audioPriceTip(m)">
                    <span class="mb__num">{{ priceLabel(m) }}</span>
                  </Tooltip>
                  <span v-else-if="priceLabel(m)" class="mb__num">{{ priceLabel(m) }}</span>
                </td>
                <td v-if="hasCtx" class="mb__td-r">
                  <span v-if="m.ctx" class="mb__num">{{ fmtCtx(m.ctx) }}</span>
                </td>
                <td v-if="visionCol" class="mb__td-c">
                  <span v-if="m.vision" class="mb__mark"><Icon name="eye" :size="14" /></span>
                </td>
                <td v-if="hasReasoning" class="mb__td-c">
                  <span v-if="m.reasoning" class="mb__mark"><Icon name="brain" :size="14" /></span>
                </td>
                <td v-if="hasAsr" class="mb__td-c">
                  <span v-if="m.asr" class="mb__mark"><Icon name="microphone" :size="14" /></span>
                </td>
                <td class="mb__td-act">
                  <Tooltip
                    v-if="!enabledSet.has(m.id)"
                    :label="expandable ? 'Adds with automatic routing: the provider is picked per request' : undefined"
                  >
                    <button class="pk-btn pk-btn--sm" type="button" @click.stop="emit('add', m)">
                      Add
                    </button>
                  </Tooltip>
                  <span v-else class="mb__in"><Icon name="check" :size="13" /> added</span>
                </td>
              </tr>
              <tr v-if="expanded[m.id]" class="mb__detail">
                <td :colspan="colCount">
                  <p v-if="details[m.id]?.loading" class="mb__hint">
                    <Icon name="spinner" :size="13" class="mb__spin" /> Asking OpenRouter...
                  </p>
                  <p v-else-if="details[m.id]?.error" class="mb__error">
                    {{ details[m.id].error }}
                  </p>
                  <table v-else class="mb__sub">
                    <thead>
                      <tr>
                        <th>Provider</th>
                        <th class="mb__sub-r">$ in · out /M</th>
                        <th class="mb__sub-r">Context</th>
                        <th class="mb__sub-r">Max out</th>
                        <th class="mb__sub-r">tok/s</th>
                        <th>Quant</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="p in details[m.id]?.provs ?? []" :key="provSlug(p)">
                        <td>
                          {{ p.name }}
                          <span v-if="provVariant(p)" class="mb__variant">{{ provVariant(p) }}</span>
                        </td>
                        <td class="mb__sub-r">
                          <span v-if="billsByAudio(m) && p.promptPrice != null" class="mb__num">
                            {{ fmtRate(p.promptPrice) }} audio
                          </span>
                          <span
                            v-else-if="!billsByAudio(m) && p.promptPrice != null && p.completionPrice != null"
                            class="mb__num"
                          >
                            {{ fmtPerM(p.promptPrice) }} · {{ fmtPerM(p.completionPrice) }}
                          </span>
                        </td>
                        <td class="mb__sub-r">
                          <span v-if="p.ctx" class="mb__num">{{ fmtCtx(p.ctx) }}</span>
                        </td>
                        <td class="mb__sub-r">
                          <span v-if="p.maxOut" class="mb__num">{{ fmtCtx(p.maxOut) }}</span>
                        </td>
                        <td class="mb__sub-r">
                          <span v-if="p.tps" class="mb__num">{{ p.tps }}</span>
                        </td>
                        <td>
                          <span v-if="p.quant" class="mb__num">{{ p.quant }}</span>
                        </td>
                        <td class="mb__sub-act">
                          <Tooltip
                            v-if="!enabledSet.has(`${m.id}@${provSlug(p)}`)"
                            :label="`Always use ${provSlug(p)}`"
                          >
                            <button
                              class="pk-btn pk-btn--sm"
                              type="button"
                              @click.stop="
                                emit('add', {
                                  id: m.id,
                                  display: m.display,
                                  ctx: p.ctx ?? m.ctx,
                                  maxOut: p.maxOut ?? m.maxOut,
                                  vision: m.vision,
                                  reasoning: m.reasoning,
                                  asr: m.asr,
                                  provider: provSlug(p),
                                })
                              "
                            >
                              Add
                            </button>
                          </Tooltip>
                          <span v-else class="mb__in"><Icon name="check" :size="13" /> added</span>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
      <p v-if="visible.total > CAP" class="mb__hint">
        Showing {{ CAP }} of {{ visible.total }}.
      </p>
      <p v-else-if="!visible.total" class="mb__hint">
        Nothing matches.
        <button
          v-if="view.query.trim()"
          type="button"
          class="mb__addid"
          @click="emit('add', { id: view.query.trim() })"
        >
          Add "{{ view.query.trim() }}" as a model id
        </button>
      </p>
    </template>
  </div>
</template>

<style scoped>
/* the search is the control of this page - full width, tall, big type
   (explicit height: the pk-input base sizes must not win) */
.mb__search {
  width: 100%;
  max-width: none;
  height: 52px;
  margin-bottom: 12px;
  padding: 0 18px;
  font-size: var(--pk-font-size-lg);
  font-weight: 600;
  border-radius: var(--pk-radius-lg);
}
.mb__search::placeholder {
  font-weight: 500;
}
.mb__hint {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
  margin: 0 0 8px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.mb__error {
  color: var(--pk-status-error, #d33);
  font-size: var(--pk-font-size-sm);
  margin: 0 0 8px;
}
.mb__spin {
  animation: mb-spin 0.9s linear infinite;
}
@keyframes mb-spin {
  to {
    transform: rotate(360deg);
  }
}
.mb__tools {
  display: flex;
  align-items: center;
  gap: 5px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.mb__chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-full);
  background: transparent;
  color: var(--pk-text-secondary);
  font: inherit;
  font-size: var(--pk-font-size-xs);
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
}
.mb__chip:hover {
  border-color: var(--pk-accent);
  color: var(--pk-text-primary);
}
.mb__chip--on {
  border-color: var(--pk-accent);
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text, var(--pk-accent));
}
.mb__tools-sep {
  width: 1px;
  height: 16px;
  background: var(--pk-border-default);
  margin: 0 3px;
}
/* the real table (the Instrument th-sort idiom): sortable headers, sticky
   while the body scrolls, fixed column widths so every cell is a cell */
.mb__tablewrap {
  overflow: auto;
  max-height: 340px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  margin-bottom: 8px;
}
.mb__table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: var(--pk-font-size-sm);
}
.mb__table thead th {
  text-align: left;
  font-weight: 600;
  font-size: var(--pk-font-size-xs);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
  padding: 9px 12px;
  border-bottom: 1px solid var(--pk-border-default);
  white-space: nowrap;
  background: var(--pk-bg-surface);
  position: sticky;
  top: 0;
  z-index: 1;
}
.th-sort {
  cursor: pointer;
  user-select: none;
}
.th-sort:hover {
  color: var(--pk-text-primary);
}
.mb__th-logo {
  width: 36px;
}
.mb__th-price {
  width: 132px;
  text-align: right;
}
.mb__th-ctx {
  width: 86px;
  text-align: right;
}
.mb__th-mark {
  width: 40px;
  text-align: center;
}
.mb__th-act {
  width: 92px;
}
.mb__table td {
  padding: 7px 12px;
  border-top: 1px solid var(--pk-border-subtle);
  vertical-align: middle;
  white-space: nowrap;
}
.mb__table tbody tr:first-child td {
  border-top: none;
}
.mb__table tbody tr:hover td {
  background: var(--pk-bg-hover);
}
.mb__td-logo {
  text-align: center;
}
.mb__td-name {
  overflow: hidden;
  text-overflow: ellipsis;
}
.mb__name {
  font-weight: 500;
  cursor: default;
}
.mb__td-r {
  text-align: right;
}
.mb__td-c {
  text-align: center;
}
.mb__td-act {
  text-align: right;
}
.mb__num {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  cursor: default;
}
.mb__free {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-status-ok, var(--pk-accent));
}
.mb__mark {
  display: inline-flex;
  color: var(--pk-text-secondary);
}
/* expandable rows: the caret is the affordance, the whole row the target */
.mb__row-x {
  cursor: pointer;
}
.mb__caret {
  color: var(--pk-text-muted);
  margin-right: 6px;
  vertical-align: -1px;
}
.mb__detail td {
  padding: 4px 12px 12px 40px;
  white-space: normal;
  background: var(--pk-bg-base);
}
/* the per-provider breakdown: one model, many providers, each its own deal */
.mb__sub {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-xs);
}
.mb__sub thead th {
  text-align: left;
  font-weight: 600;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  padding: 6px 10px;
  border-bottom: 1px solid var(--pk-border-subtle);
  white-space: nowrap;
  background: none;
  position: static;
  text-transform: none;
  letter-spacing: 0;
}
.mb__sub td {
  padding: 5px 10px;
  border-top: 1px solid var(--pk-border-subtle);
  color: var(--pk-text-secondary);
  white-space: nowrap;
}
.mb__sub tbody tr:first-child td {
  border-top: none;
}
/* the detail area is a reading surface, not a hover target - keep the outer
   table's row hover out of it */
.mb__detail:hover td,
.mb__sub tbody tr:hover td {
  background: var(--pk-bg-base);
}
.mb__sub-r {
  text-align: right;
}
.mb__sub-act {
  text-align: right;
  width: 84px;
}
/* the part of the endpoint slug that tells same-brand twins apart
   (us-east-1, global, europe...) */
.mb__variant {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  margin-left: 4px;
}
.mb__in {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--pk-status-ok, var(--pk-accent));
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
}
/* the search's no-match fallback replaces the old manual-id input: what the
   user typed can just BE the model id */
.mb__addid {
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  font-weight: 600;
  color: var(--pk-accent);
  cursor: pointer;
}
.mb__addid:hover {
  text-decoration: underline;
}
/* fill mode: the list takes the remaining card height, the page stops
   ending in dead space */
.mb--fill {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}
.mb--fill .mb__tablewrap {
  flex: 1;
  min-height: 0;
  max-height: none;
}
</style>
