<script setup lang="ts">
// One artifact viewer: a tab strip for the artifacts it is responsible for, a
// header for the one on screen, and the preview itself. The panel renders a
// single pane normally and two side by side during a compare.
//
// Everything about the frame protocol lives here exactly ONCE. The split used
// to be a second, half-copied set of refs in the panel, and every blank-preview
// bug came from the two copies disagreeing - one shared nonce remounted both
// frames while only one of them re-armed, so a resize across the split
// threshold reliably emptied the preview.
//
// Scripting content (html, svg) renders in a frame served by the manager at
// /artifact-frame, never in a srcdoc: `frame-ancestors` and `sandbox` are
// header-only CSP directives that a <meta> tag cannot express, and the served
// shell is the only way to attach a real header. The body is posted in after
// load, so the frame URL carries nothing. Everything else renders inline.
import { copyText } from '@/lib/clipboard'
import { computed, defineAsyncComponent, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import CodeEditor from '@/components/ui/CodeEditor.vue'
// The graph panel drags sigma + the 7.4 MB traverse wasm with it, so it loads
// only when a graph artifact is actually on screen.
const GraphArtifact = defineAsyncComponent(() => import('@/components/chat/graph/GraphArtifact.vue'))
/** Live graph panel instance, for kind-aware download (tvdb, not script). */
const graphRef = ref<{ downloadTvdb: () => Promise<void> } | null>(null)
import Icon from '@/components/Icon.vue'
import Markdown from '@/components/chat/Markdown.vue'
import Select from '@/components/ui/Select.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import { fleetLabel, fleetVendor } from '@/lib/model-name'
import { type ArtifactMeta, type ArtifactVersion, useArtifactsStore } from '@/stores/artifacts'
import { useChatStore } from '@/stores/chat'
import { useSettingsStore } from '@/stores/settings'

const props = defineProps<{
  /** The artifacts this pane can show - one model's during a compare. */
  items: ArtifactMeta[]
  selected: string
}>()
const emit = defineEmits<{ select: [string] }>()

const artifacts = useArtifactsStore()
const chat = useChatStore()
const settings = useSettingsStore()

// Monaco is 3.8 MB of chunk, and fetching it at the instant somebody clicks
// Source is a wait you can watch. This panel only exists
// when a chat has artifacts, so warm it while the browser is idle - by the
// time anyone reaches for Source it is already parsed.
onMounted(() => {
  const warm = (): void => void import('@/lib/monaco')
  if ('requestIdleCallback' in window) window.requestIdleCallback(warm, { timeout: 4000 })
  else setTimeout(warm, 1500)
})
const view = ref<'preview' | 'source'>('preview')
const seq = ref(0)
const body = ref('')
const versions = ref<ArtifactVersion[]>([])
const loading = ref(false)

const meta = computed(() => props.items.find((a) => a.id === props.selected) ?? null)
// Who wrote this artifact, marked and named exactly as the chat's lane header
// does it - the artifact carries its own model, so this is right in a single
// pane too, not only in a compare.
const writer = computed(() => fleetLabel(meta.value?.model))
const writerVendor = computed(() => fleetVendor(meta.value?.model))
/** Frames are for content that can execute; the rest renders natively. */
const framed = computed(() => meta.value?.kind === 'html' || meta.value?.kind === 'svg')

// Refetch when the selection moves, when a version is pinned, and when the
// artifact itself changes underneath us - an update tool call bumps the meta's
// version count without touching the id, and that is the mid-conversation edit
// the panel is supposed to follow.
let gen = 0
watch(
  () => [props.selected, seq.value, meta.value?.versions, meta.value?.updatedAt] as const,
  async ([id, s]) => {
    const mine = ++gen
    // Clear first: the id changes synchronously but the body arrives a fetch
    // later, so leaving the old text renders the previous artifact under the
    // new one's name for a frame or two.
    body.value = ''
    versions.value = []
    if (!id) return
    loading.value = true
    try {
      const r = await artifacts.fetchOne(id, s)
      if (mine !== gen) return
      body.value = r.body
      versions.value = r.versions
    } catch (e) {
      console.error('failed to load artifact', e)
    } finally {
      if (mine === gen) loading.value = false
    }
  },
  { immediate: true },
)
// A different artifact starts at its own latest, never on the last one's pin.
watch(
  () => props.selected,
  () => {
    seq.value = 0
    view.value = 'preview'
  },
)

// ── editing ──────────────────────────────────────────────────────────────
// What is on screen is the DRAFT: the saved body until somebody types, the
// edited text after. Preview renders it too, so editing the HTML and flipping
// to Preview shows the change before it is saved.
const draft = computed<string>({
  get: () => artifacts.drafts[props.selected] ?? body.value,
  set: (v) => artifacts.setDraft(props.selected, v, body.value),
})
const dirty = computed(() => artifacts.drafts[props.selected] !== undefined)
/** Only the newest version is editable - an older one is history, and writing
 *  through it would quietly fork it. Pin it to read, return to latest to edit. */
const editable = computed(() => seq.value === 0 && !!meta.value)
const saving = ref(false)
const saveError = ref('')
async function save(): Promise<void> {
  if (!dirty.value || !editable.value || saving.value) return
  saving.value = true
  saveError.value = await artifacts.saveEdit(props.selected, draft.value)
  saving.value = false
}
function revert(): void {
  artifacts.clearDraft(props.selected)
  saveError.value = ''
}

// ── the preview frame ────────────────────────────────────────────────────
// Remounting is the only way to make the frame load again: the sandbox has no
// allow-same-origin, so it runs at an opaque origin and reload() from here is a
// cross-origin access that throws. postMessage is allowed, which is why the
// first load always worked and later switches showed an empty panel.
const frame = ref<HTMLIFrameElement | null>(null)
const nonce = ref(0)
const posted = ref(-1)

// What the sandbox refused to fetch. The frame has no network at all, so an
// artifact that links a remote image just renders wrong - and saying nothing
// is how "why is this blank?" happens.
const blocked = ref<string[]>([])
/** Fetched and never arrived - a dead host, not our policy. Allowing pictures
 *  does nothing for these, so they get their own sentence. */
const failed = ref<string[]>([])
/** Per-preview opt-in to remote PICTURES (nothing else - see FRAME_CSP_IMG).
 *  Off every time the artifact changes: consent is for what you just looked
 *  at, not a setting you forget you turned on. */
const allowImages = ref(false)
const hosts = (urls: string[]): string[] => [
  ...new Set(
    urls.map((u) => {
      try {
        return new URL(u).host
      } catch {
        return u
      }
    }),
  ),
]
const blockedHosts = computed(() => hosts(blocked.value))
const failedHosts = computed(() => hosts(failed.value))
function onFrameMessage(e: MessageEvent): void {
  if (e.source !== frame.value?.contentWindow) return
  const m = (e.data as { paddockArtifactMissing?: { blocked?: unknown; failed?: unknown } })
    ?.paddockArtifactMissing
  if (!m) return
  const strings = (v: unknown): string[] =>
    Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : []
  blocked.value = strings(m.blocked)
  failed.value = strings(m.failed)
}
onMounted(() => window.addEventListener('message', onFrameMessage))
onBeforeUnmount(() => window.removeEventListener('message', onFrameMessage))
// The theme is in the payload (the scrollbar colour), so a flip has to reach
// the frame the only way anything reaches it: a fresh element and a fresh post.
watch([draft, view, () => settings.theme], () => {
  nonce.value++
  blocked.value = []
  failed.value = []
})
watch(
  () => props.selected,
  () => (allowImages.value = false),
)
/** Turning it on has to load a different frame URL, so it is a remount. */
function loadImages(): void {
  allowImages.value = true
  blocked.value = []
  failed.value = []
  nonce.value++
}

function post(): void {
  // Self-arming by construction: a nonce bump is a new element and enables
  // exactly one post, while a SECOND @load on the same element - a document
  // that navigated itself somewhere - is refused, so the body can never reach
  // a page we did not serve. The old separate `pending` flag had to be re-armed
  // by hand at every remount site and kept being forgotten at one of them.
  if (posted.value === nonce.value) return
  posted.value = nonce.value
  // The opaque origin means the target must be a wildcard; the message still
  // reaches only this contentWindow.
  frame.value?.contentWindow?.postMessage(
    {
      type: 'paddock:artifact',
      html: draft.value,
      // The frame is a document of its own and cannot see our stylesheet, so
      // the one thing we do dress - its scrollbar - has to travel with the
      // body. Same token base.css uses on everything else.
      scrollbar: getComputedStyle(document.documentElement)
        .getPropertyValue('--pk-border-default')
        .trim(),
    },
    '*',
  )
}

// ── rendering the non-framed kinds ───────────────────────────────────────
/** A fence long enough to survive a body that contains fences of its own. */
function fenceFor(s: string): string {
  let longest = 0
  for (const m of s.matchAll(/`+/g)) longest = Math.max(longest, m[0].length)
  return '`'.repeat(Math.max(3, longest + 1))
}
const srcLang = computed(() => {
  const a = meta.value
  if (!a) return ''
  if (a.language) return a.language
  return a.kind === 'html' ? 'html'
    : a.kind === 'svg' ? 'xml'
    : a.kind === 'mermaid' ? 'mermaid'
    : a.kind === 'graph' ? 'cypher'
    : ''
})
// ── csv ──────────────────────────────────────────────────────────────────
/** RFC4180 enough for what a model writes: quoted fields, doubled quotes
 *  inside them, and commas or newlines that only count when unquoted. */
function parseCsv(src: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let cell = ''
  let quoted = false
  for (let i = 0; i < src.length; i++) {
    const c = src[i]
    if (quoted) {
      if (c === '"' && src[i + 1] === '"') {
        cell += '"'
        i++
      } else if (c === '"') quoted = false
      else cell += c
      continue
    }
    if (c === '"') quoted = true
    else if (c === ',') {
      row.push(cell)
      cell = ''
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && src[i + 1] === '\n') i++
      row.push(cell)
      rows.push(row)
      row = []
      cell = ''
    } else cell += c
  }
  if (cell || row.length) {
    row.push(cell)
    rows.push(row)
  }
  return rows.filter((r) => r.length > 1 || r[0] !== '')
}
/** Rendering ten thousand rows into the DOM would wedge the panel, so the
 *  table stops - and SAYS it stopped, with the real count. */
const CSV_ROWS = 500
const csv = computed(() => (meta.value?.kind === 'csv' ? parseCsv(draft.value) : []))
const csvHead = computed(() => csv.value[0] ?? [])
const csvBody = computed(() => csv.value.slice(1, CSV_ROWS + 1))
const csvHidden = computed(() => Math.max(0, csv.value.length - 1 - csvBody.value.length))

/** Preview of a non-executing kind. Markdown and text render as themselves;
 *  code needs a fence, long enough that a body containing one cannot break out
 *  of it, before Markdown will highlight it. */
const doc = computed(() => {
  const a = meta.value
  if (!a) return ''
  if (a.kind === 'markdown' || a.kind === 'text') return draft.value
  return sourceDoc.value
})
/** The same text as highlighted, read-only markup. It stands in for the editor
 *  while Monaco's chunk is still arriving, so Source is never a blank wait. */
const sourceDoc = computed(() => {
  const f = fenceFor(draft.value)
  return `${f}${srcLang.value}\n${draft.value}\n${f}`
})

const versionOptions = computed(() =>
  versions.value
    .slice()
    .reverse()
    .map((v) => ({
      value: v.seq,
      label: v.seq === versions.value.length ? `v${v.seq} · latest` : `v${v.seq} · ${v.op}`,
    })),
)
/** 0 means "follow the latest"; show it as the newest version's number. */
const shownSeq = computed({
  get: () => (seq.value > 0 ? seq.value : versions.value.length),
  set: (s: number) => (seq.value = s === versions.value.length ? 0 : s),
})

// Same feedback as every other copy in the Studio (MessageBubble, Instrument):
// the icon becomes a checkmark in place for a moment.
const copied = ref(false)
async function copy(): Promise<void> {
  try {
    await copyText(draft.value)
    copied.value = true
    setTimeout(() => (copied.value = false), 1200)
  } catch (e) {
    console.error('clipboard write failed', e)
  }
}

function download(): void {
  const a = meta.value
  if (!a) return
  // Downloading a GRAPH means the graph: the live session exports the same
  // .tvdb bytes a Traverse server opens. The script is Source + copy.
  if (a.kind === 'graph' && graphRef.value) {
    void graphRef.value.downloadTvdb()
    return
  }
  const ext =
    a.kind === 'html' ? 'html'
    : a.kind === 'svg' ? 'svg'
    : a.kind === 'markdown' ? 'md'
    : a.kind === 'mermaid' ? 'mmd'
    : a.kind === 'csv' ? 'csv'
    : a.kind === 'graph' ? 'cypher'
    : a.language || 'txt'
  const url = URL.createObjectURL(new Blob([draft.value], { type: 'text/plain' }))
  const el = document.createElement('a')
  el.href = url
  el.download = `${a.title.replace(/[^\w.-]+/g, '-') || a.id}.${ext}`
  el.click()
  URL.revokeObjectURL(url)
}
</script>

<template>
  <section class="pane">
    <div v-if="items.length > 1" class="pane__strip">
      <Tooltip v-for="a in items" :key="a.id" :label="a.title">
        <button
          class="pane__tab"
          :class="{ 'pane__tab--on': a.id === selected }"
          @click="emit('select', a.id)"
        >
          <span v-if="artifacts.drafts[a.id] !== undefined" class="pane__dot" />
          {{ a.title || a.id }}
        </button>
      </Tooltip>
    </div>

    <header v-if="meta" class="pane__head">
      <Tooltip v-if="writer" :label="`Written by ${meta.model}`">
        <span class="pane__who">
          <VendorLogo v-if="writerVendor" :vendor="writerVendor" :size="13" />
          <span class="pane__whoname">{{ writer }}</span>
        </span>
      </Tooltip>
      <span class="pane__title">{{ meta.title || meta.id }}</span>
      <Select v-if="versions.length > 1" v-model="shownSeq" ghost :options="versionOptions" />
      <div class="pane__spacer" />
      <div class="pane__seg">
        <button
          class="pane__segbtn"
          :class="{ 'pane__segbtn--on': view === 'preview' }"
          :aria-pressed="view === 'preview'"
          @click="view = 'preview'"
        >
          Preview
        </button>
        <button
          class="pane__segbtn"
          :class="{ 'pane__segbtn--on': view === 'source' }"
          :aria-pressed="view === 'source'"
          @click="view = 'source'"
        >
          Source
          <span v-if="dirty" class="pane__dot" />
        </button>
      </div>
      <template v-if="dirty">
        <button class="pane__act pane__act--go" :disabled="saving" @click="save">
          {{ saving ? 'Saving...' : 'Save' }}
        </button>
        <button class="pane__act" @click="revert">Revert</button>
      </template>
      <Tooltip :label="copied ? 'Copied' : 'Copy'">
        <button class="pane__icon" @click="copy">
          <Icon :name="copied ? 'check' : 'copy'" :size="14" />
        </button>
      </Tooltip>
      <Tooltip :label="meta?.kind === 'graph' ? 'Download as .tvdb' : 'Save to a file'">
        <button class="pane__icon" @click="download">
          <Icon name="arrow-down" :size="14" />
        </button>
      </Tooltip>
      <Tooltip label="Hide the artifacts">
        <button class="pane__icon" aria-label="Hide the artifacts" @click="chat.setArtifactsPane(false)">
          <Icon name="panel-left" :size="14" />
        </button>
      </Tooltip>
    </header>

    <p v-if="saveError" class="pane__err">{{ saveError }}</p>
    <template v-if="view === 'preview'">
      <p v-if="blocked.length" class="pane__warn">
        <Icon name="alert-triangle" :size="12" />
        <span>
          {{ blocked.length }} external
          {{ blocked.length === 1 ? 'picture' : 'pictures' }} did not load
          ({{ blockedHosts.join(', ') }}) - the preview has no network access.
        </span>
        <button v-if="!allowImages" class="pane__act" @click="loadImages">Load pictures</button>
      </p>
      <p v-if="failed.length" class="pane__warn">
        <Icon name="alert-triangle" :size="12" />
        <span>
          {{ failed.length }}
          {{ failed.length === 1 ? 'picture' : 'pictures' }} could not be fetched
          ({{ failedHosts.join(', ') }}) - that address is not answering, so the page is
          missing what it drew on top of.
        </span>
      </p>
    </template>

    <div class="pane__body">
      <CodeEditor
        v-if="view === 'source' && meta"
        v-model="draft"
        :language="srcLang"
        :readonly="!editable"
        readonly-message="This is an older version. Switch to the latest to edit it."
        @save="save"
      >
        <template #placeholder>
          <div class="pane__doc"><Markdown :content="sourceDoc" /></div>
        </template>
      </CodeEditor>
      <iframe
        v-else-if="framed && draft"
        :key="nonce"
        ref="frame"
        class="pane__frame"
        :src="allowImages ? '/artifact-frame?img=1' : '/artifact-frame'"
        sandbox="allow-scripts"
        referrerpolicy="no-referrer"
        title="Artifact preview"
        @load="post"
      />
      <div v-else-if="meta?.kind === 'csv' && csv.length" class="pane__table">
        <table>
          <thead>
            <tr>
              <th v-for="(h, i) in csvHead" :key="i">{{ h }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(r, ri) in csvBody" :key="ri">
              <td v-for="(c, ci) in csvHead" :key="ci">{{ r[ci] ?? '' }}</td>
            </tr>
          </tbody>
        </table>
        <p v-if="csvHidden" class="pane__more">
          {{ csvHidden }} more {{ csvHidden === 1 ? 'row' : 'rows' }} - open Source or save the
          file to see all {{ csv.length - 1 }}.
        </p>
      </div>
      <GraphArtifact
        v-else-if="meta?.kind === 'graph' && draft"
        ref="graphRef"
        :content="draft"
        @import-result="(r) => meta && artifacts.reportGraphImport(meta.id, meta.versions, r)"
        :title="meta.title || meta.id"
      />
      <div v-else-if="draft" class="pane__doc"><Markdown :content="doc" /></div>
      <p v-else class="pane__empty">{{ loading ? 'Loading...' : 'Nothing here yet.' }}</p>
    </div>
  </section>
</template>

<style scoped>
.pane {
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  background: var(--pk-bg-surface);
}
.pane__strip {
  display: flex;
  align-items: center;
  gap: 2px;
  padding: 6px 8px 0;
  overflow-x: auto;
}
/* Same mark + friendly name the chat's lane header uses, so the artifact and
   the column of replies it came from read as one thing. */
.pane__who {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  flex: 0 1 auto;
  max-width: 190px;
  padding: 2px 7px;
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-base);
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.pane__who svg {
  flex: none;
}
/* The name needs its own box: ellipsis does not apply to a flex container's
   anonymous text child, so it has to be a real element that can shrink. */
.pane__whoname {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.pane__tab {
  max-width: 160px;
  padding: 4px 10px;
  border: 0;
  background: none;
  border-radius: var(--pk-radius-sm) var(--pk-radius-sm) 0 0;
  color: var(--pk-text-muted);
  font: inherit;
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  cursor: pointer;
}
.pane__tab--on {
  background: var(--pk-bg-base);
  color: var(--pk-text-primary);
}
.pane__head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-bottom: 1px solid var(--pk-border-subtle);
}
.pane__title {
  /* min-width:0 beats a flex item's implicit auto floor, which is what makes
     the ellipsis engage instead of shoving the controls off the row. */
  flex: 0 1 auto;
  min-width: 0;
  font-size: var(--pk-font-size-sm);
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.pane__spacer {
  flex: 1;
}
.pane__seg {
  display: flex;
  gap: 4px;
}
.pane__segbtn {
  padding: 3px 10px;
  border: 1px solid var(--pk-border-default);
  background: none;
  border-radius: var(--pk-radius-sm);
  color: var(--pk-text-muted);
  font: inherit;
  font-size: var(--pk-font-size-xs);
  cursor: pointer;
}
.pane__segbtn:hover,
.pane__segbtn--on {
  color: var(--pk-text-primary);
}
/* Unsaved-edit marker: on the tab it belongs to and on the Source button. */
.pane__dot {
  display: inline-block;
  width: 5px;
  height: 5px;
  margin-left: 4px;
  border-radius: 50%;
  background: var(--pk-accent);
  vertical-align: middle;
}
.pane__act {
  padding: 3px 9px;
  border: 1px solid var(--pk-border-default);
  background: none;
  border-radius: var(--pk-radius-sm);
  color: var(--pk-text-muted);
  font: inherit;
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
  cursor: pointer;
}
.pane__act:hover:not(:disabled) {
  color: var(--pk-text-primary);
}
.pane__act--go {
  border-color: var(--pk-accent);
  color: var(--pk-accent);
}
.pane__act:disabled {
  opacity: 0.6;
  cursor: default;
}
.pane__err {
  margin: 0;
  padding: 6px 10px;
  background: var(--pk-status-error-subtle);
  color: var(--pk-status-error);
  font-size: var(--pk-font-size-xs);
}
.pane__warn {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  margin: 0;
  padding: 6px 10px;
  background: var(--pk-status-warning-subtle);
  color: var(--pk-status-warning);
  font-size: var(--pk-font-size-xs);
}
.pane__warn svg {
  flex: none;
  margin-top: 1px;
}
.pane__warn span {
  flex: 1 1 auto;
  min-width: 0;
}
.pane__warn .pane__act {
  border-color: currentColor;
  color: inherit;
}
.pane__icon {
  display: grid;
  place-items: center;
  width: 26px;
  height: 26px;
  border: 0;
  background: none;
  border-radius: var(--pk-radius-sm);
  color: var(--pk-text-muted);
  cursor: pointer;
}
.pane__icon:hover {
  background: var(--pk-bg-base);
  color: var(--pk-text-primary);
}
.pane__body {
  flex: 1;
  min-height: 0;
  overflow: auto;
}
.pane__frame {
  display: block;
  width: 100%;
  height: 100%;
  border: 0;
  background: var(--pk-bg-base);
}
.pane__doc {
  padding: 12px 14px;
}
/* A csv is a table, not a code block - the whole reason it is its own kind. */
.pane__table {
  padding: 10px 12px;
  overflow: auto;
}
.pane__table table {
  border-collapse: collapse;
  font-size: var(--pk-font-size-xs);
}
.pane__table th,
.pane__table td {
  padding: 4px 9px;
  border: 1px solid var(--pk-border-subtle);
  text-align: left;
  white-space: nowrap;
}
.pane__table th {
  position: sticky;
  top: 0;
  background: var(--pk-bg-elevated);
  font-weight: 600;
}
.pane__table td {
  font-family: var(--pk-font-mono);
}
.pane__more {
  margin: 8px 0 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
}
.pane__empty {
  margin: 0;
  padding: 14px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
</style>
