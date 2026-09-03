<script setup lang="ts">
// The graph artifact panel: a live in-browser Traverse database seeded from
// the artifact's Cypher script.
//
// Session semantics, stated here because the UI must never imply otherwise:
// the script SEEDS the session; queries typed below mutate only the worker;
// leaving the panel discards mutations; the exports are explicit. Edits to
// the artifact itself re-seed - which also means the Source view is a live
// editor for the graph.
//
// This component is reached only through defineAsyncComponent in
// ArtifactPane, so sigma + the 7.4 MB wasm land in a chunk nobody who never
// opens a graph pays for.
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import DataTable from '@/components/ui/DataTable.vue'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import GraphCanvas from '@/components/chat/graph/GraphCanvas.vue'
import QueryStrip from '@/components/chat/graph/QueryStrip.vue'
import {
  buildGraph,
  cellText,
  GraphSession,
  hasEntities,
  mergeIntoGraph,
  withInducedEdges,
  type GraphResult,
  type ImportResult,
  type QueryResponse,
} from '@/lib/graph/session'
import { registerCypherCompletions, updateSchemaCompletions } from '@/lib/graph/cypher-language'
import { useSettingsStore } from '@/stores/settings'

const props = defineProps<{
  content: string
  title: string
}>()
const emit = defineEmits<{
  /** Every seed's outcome, success included - the pane forwards it to the
   *  auto-repair loop, and a clean import CLEARS a queued failure report. */
  (e: 'import-result', r: { errors: ImportResult['errors']; executed: number }): void
}>()

const settings = useSettingsStore()
const dark = computed(() => settings.theme === 'dark')

const session = new GraphSession()
const booting = ref(true)
const bootError = ref('')
const importErrors = ref<ImportResult['errors']>([])
const executed = ref(0)
const view = ref<GraphResult | null>(null)
const live = ref({ nodes: 0, edges: 0 })
/** True when the canvas shows a query's entities instead of the whole graph. */
const queryView = ref(false)
const modified = ref(false)
const running = ref(false)
const queryError = ref('')
const lastRun = ref<QueryResponse | null>(null)
/** What buildGraph last consumed - re-run on a theme flip, since the node
 *  colors are baked into the graphology attributes. */
let lastShown: QueryResponse | null = null

async function seed(script: string): Promise<void> {
  const imp = await session.seed(script)
  importErrors.value = imp.errors
  executed.value = imp.executed
  emit('import-result', { errors: imp.errors, executed: imp.executed })
  modified.value = false
  lastRun.value = null
  queryError.value = ''
  await showAll()
  void refreshCompletions()
}

async function showAll(): Promise<void> {
  const res = await session.renderAll()
  lastShown = res
  view.value = buildGraph(res, dark.value)
  queryView.value = false
  live.value = await session.stats()
}

/** Schema-aware Cypher completions for the strip. Monaco is already on its
 *  way (the strip mounts a CodeEditor), so this await costs nothing extra. */
async function refreshCompletions(): Promise<void> {
  try {
    const [lib, schema] = await Promise.all([import('@/lib/monaco'), session.schema()])
    const monaco = await lib.loadMonaco(dark.value)
    registerCypherCompletions(monaco)
    updateSchemaCompletions(
      monaco,
      schema.labels,
      schema.relationship_types,
      schema.labels_detail,
      schema.relationship_types_detail,
      schema.property_keys.map((p) => p.name),
    )
  } catch (e) {
    // Completions are a comfort, not a requirement - the strip works without.
    console.warn('cypher completions unavailable', e)
  }
}

onMounted(async () => {
  try {
    await session.open()
    await seed(props.content)
  } catch (e) {
    bootError.value = e instanceof Error ? e.message : String(e)
  } finally {
    booting.value = false
  }
})
onBeforeUnmount(() => session.close())

// Re-seed when the artifact text changes (a save, a version switch, or live
// typing in Source). Debounced so keystrokes coalesce into one import.
let seedTimer: ReturnType<typeof setTimeout> | undefined
watch(
  () => props.content,
  (c) => {
    clearTimeout(seedTimer)
    seedTimer = setTimeout(() => {
      seed(c).catch((e) => (bootError.value = e instanceof Error ? e.message : String(e)))
    }, 600)
  },
)

// Colors are baked into node attributes at build time, so a theme flip
// rebuilds the graph from the same response rather than restyling in place.
watch(dark, () => {
  if (lastShown) view.value = buildGraph(lastShown, dark.value)
})

async function runQuery(cypher: string): Promise<void> {
  if (running.value) return
  running.value = true
  queryError.value = ''
  try {
    const res = await session.run(cypher)
    lastRun.value = res
    const wrote = res.query_type !== 'Read'
    if (wrote) {
      modified.value = true
      await showAll()
    } else if (hasEntities(res)) {
      const linked = await withInducedEdges(session, res)
      lastShown = linked
      view.value = buildGraph(linked, dark.value)
      queryView.value = true
    }
  } catch (e) {
    queryError.value = e instanceof Error ? e.message : String(e)
    lastRun.value = null
  } finally {
    running.value = false
  }
}

async function reset(): Promise<void> {
  booting.value = true
  try {
    await seed(props.content)
  } finally {
    booting.value = false
  }
}

async function downloadTvdb(): Promise<void> {
  const bytes = await session.exportTvdb()
  const buf = new Uint8Array(bytes)
  const url = URL.createObjectURL(new Blob([buf.buffer as ArrayBuffer]))
  const el = document.createElement('a')
  el.href = url
  el.download = `${props.title.replace(/[^\w.-]+/g, '-') || 'graph'}.tvdb`
  el.click()
  URL.revokeObjectURL(url)
}
// The pane header's download button calls this for graph artifacts -
// downloading a GRAPH means the .tvdb (opens in a Traverse server), not the
// script; the Cypher stays reachable via Source and copy.
defineExpose({ downloadTvdb })

const canvas = ref<InstanceType<typeof GraphCanvas> | null>(null)
/** Double-click expansion, same behavior as the attached-graph pane. */
async function expandNode(fgId: number, nodeKey: string): Promise<void> {
  const g = view.value?.graph
  if (!g) return
  try {
    const res = await session.run(`MATCH (n)-[r]-(m) WHERE id(n) = ${fgId} RETURN n, r, m LIMIT 50`)
    if (mergeIntoGraph(g, res, dark.value, nodeKey) > 0) canvas.value?.relayoutMerged()
  } catch (e) {
    queryError.value = e instanceof Error ? e.message : String(e)
  }
}

/** Mutation chips for a write result - only the non-zero ones say anything. */
const mutations = computed(() => {
  const st = lastRun.value?.stats
  if (!st) return []
  const out: string[] = []
  if (st.nodes_created) out.push(`${st.nodes_created} nodes created`)
  if (st.nodes_deleted) out.push(`${st.nodes_deleted} nodes deleted`)
  if (st.relationships_created) out.push(`${st.relationships_created} rels created`)
  if (st.relationships_deleted) out.push(`${st.relationships_deleted} rels deleted`)
  if (st.properties_set) out.push(`${st.properties_set} properties set`)
  return out
})

/** Rendering thousands of rows would wedge the panel; the cap says so. */
const ROW_CAP = 100
const shownRows = computed(() => (lastRun.value?.rows ?? []).slice(0, ROW_CAP))
const tableCols = computed(() =>
  (lastRun.value?.columns ?? []).map((label, i) => ({
    label,
    numeric: Number.isFinite(Number(cellText(lastRun.value?.rows?.[0]?.[i]))),
  })),
)
const hiddenRows = computed(() =>
  Math.max(0, (lastRun.value?.total_rows ?? 0) - shownRows.value.length),
)
const errorCap = 10

/** A one-statement script fails as one 4 KB "statement" (seen live: the EU
 *  commissioners graph) - show the window around the engine's reported
 *  position instead of the wall. */
function errStatement(e: { statement: string; error: string }): string {
  const s = e.statement
  if (s.length <= 220) return s
  let at = -1
  const m = e.error.match(/position (\d+)/)
  if (m && Number(m[1]) > 0) at = Math.min(Number(m[1]), s.length)
  if (at < 0) {
    // Semantic errors report position 0 but name the offending relationship
    // type (traverse >= 0.8.5) - locate the clause by that name instead,
    // preferring an arrowless (undirected) use since that is the usual sin.
    const t = e.error.match(/\[:(\w+)/)
    if (t) {
      const undirected = new RegExp('-\\[:' + t[1] + '[^\\]]*\\]-(?!>)').exec(s)
      at = undirected ? undirected.index : s.indexOf('[:' + t[1])
    }
  }
  if (at < 0) {
    // "Variable \`X\` already bound" - the first occurrence is the legal
    // declaration; the SECOND is the rebind the engine refused.
    const v = e.error.match(/Variable `(\w+)`/)
    if (v) {
      const first = s.indexOf(v[1])
      at = first >= 0 ? s.indexOf(v[1], first + v[1].length) : -1
    }
  }
  if (at < 0) return s.slice(0, 217) + '...'
  const lo = Math.max(0, at - 100)
  const hi = Math.min(s.length, at + 120)
  return (lo > 0 ? '...' : '') + s.slice(lo, hi) + (hi < s.length ? '...' : '')
}
</script>

<template>
  <div class="ga">
    <div class="ga__bar">
      <span class="ga__stat">
        {{ live.nodes.toLocaleString() }} nodes · {{ live.edges.toLocaleString() }} edges
      </span>
      <span v-if="view?.truncated" class="ga__note">view capped at 1024 entities</span>
      <span v-if="modified" class="ga__mod">modified - not saved to the artifact</span>
      <button v-if="queryView" class="ga__link" @click="showAll">Show full graph</button>
      <div class="ga__spacer" />
      <Tooltip v-if="modified" label="Discard changes, reload the script">
        <button class="ga__icon" @click="reset"><Icon name="rotate-left" :size="14" /></button>
      </Tooltip>
    </div>

    <div v-if="importErrors.length" class="ga__errors">
      <p class="ga__errhead">
        <Icon name="alert-triangle" :size="12" />
        {{ importErrors.length }} of {{ executed + importErrors.length }} statements failed - the
        graph shows what imported.
      </p>
      <div v-for="(e, i) in importErrors.slice(0, errorCap)" :key="i" class="ga__err">
        <code>{{ errStatement(e) }}</code>
        <span>{{ e.error }}</span>
      </div>
      <p v-if="importErrors.length > errorCap" class="ga__errhead">
        and {{ importErrors.length - errorCap }} more.
      </p>
    </div>

    <div class="ga__canvas">
      <p v-if="booting" class="ga__empty">Starting the graph engine...</p>
      <p v-else-if="bootError" class="ga__empty">{{ bootError }}</p>
      <p v-else-if="!view || view.nodeCount === 0" class="ga__empty">
        The script created no nodes yet.
      </p>
      <GraphCanvas
        v-else
        ref="canvas"
        :graph="view.graph"
        :dark="dark"
        :export-name="props.title"
        @expand="expandNode"
      />
    </div>

    <div class="ga__query">
      <QueryStrip :running="running" @run="runQuery" />
      <p v-if="queryError" class="ga__qerr">{{ queryError }}</p>
      <template v-if="lastRun && !queryError">
        <div class="ga__chips">
          <span class="ga__chip">{{ lastRun.query_type }}</span>
          <span class="ga__chip">{{ lastRun.time_ms.toFixed(1) }} ms</span>
          <span class="ga__chip">{{ lastRun.total_rows }} rows</span>
          <span v-for="m in mutations" :key="m" class="ga__chip ga__chip--write">{{ m }}</span>
        </div>
        <div v-if="shownRows.length" class="ga__table">
          <DataTable :columns="tableCols" :rows="shownRows" :format="(c) => cellText(c)" />
          <p v-if="hiddenRows" class="ga__note">
            {{ hiddenRows.toLocaleString() }} more rows - add a LIMIT or aggregate.
          </p>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.ga {
  display: flex;
  flex-direction: column;
  min-height: 0;
  height: 100%;
}
.ga__bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 10px;
  border-bottom: 1px solid var(--pk-border-subtle);
  font-size: var(--pk-font-size-xs);
}
.ga__stat {
  color: var(--pk-text-secondary);
  font-variant-numeric: tabular-nums;
}
.ga__note {
  color: var(--pk-text-muted);
}
.ga__mod {
  color: var(--pk-warning, #b45309);
}
.ga__link {
  padding: 0;
  border: 0;
  background: none;
  color: var(--pk-accent, var(--pk-text-primary));
  font: inherit;
  cursor: pointer;
}
.ga__spacer {
  flex: 1;
}
.ga__icon {
  display: grid;
  place-items: center;
  width: 24px;
  height: 24px;
  padding: 0;
  border: 0;
  background: none;
  border-radius: var(--pk-radius-sm);
  color: var(--pk-text-secondary);
  cursor: pointer;
}
.ga__icon:hover {
  color: var(--pk-text-primary);
  background: var(--pk-bg-base);
}
.ga__errors {
  padding: 6px 10px;
  border-bottom: 1px solid var(--pk-border-subtle);
  max-height: 140px;
  overflow-y: auto;
  font-size: var(--pk-font-size-xs);
}
.ga__errhead {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 4px;
  color: var(--pk-warning, #b45309);
}
.ga__err {
  display: flex;
  gap: 8px;
  margin: 2px 0;
  color: var(--pk-text-secondary);
}
.ga__err code {
  max-width: 45%;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  color: var(--pk-text-muted);
}
.ga__canvas {
  flex: 1;
  min-height: 220px;
  position: relative;
}
.ga__empty {
  display: grid;
  place-items: center;
  height: 100%;
  margin: 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.ga__query {
  border-top: 1px solid var(--pk-border-subtle);
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 45%;
  overflow-y: auto;
}
.ga__qerr {
  margin: 0;
  color: var(--pk-danger, #b91c1c);
  font-size: var(--pk-font-size-xs);
  overflow-wrap: anywhere;
}
.ga__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.ga__chip {
  padding: 1px 8px;
  border-radius: var(--pk-radius-full, 999px);
  background: var(--pk-bg-base);
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-xs);
  font-variant-numeric: tabular-nums;
}
.ga__chip--write {
  color: var(--pk-warning, #b45309);
}
.ga__table {
  display: flex;
  flex-direction: column;
  max-height: 240px;
  min-height: 0;
}
</style>
