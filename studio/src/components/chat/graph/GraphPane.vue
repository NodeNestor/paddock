<script setup lang="ts">
// The conversation's ATTACHED graph (a dropped .tvdb) - phase 2's read case.
//
// Differences from GraphArtifact, which this deliberately mirrors: the seed
// is the stored tvdb (via the graphs store, which owns the session and the
// model bridge), there is no script to re-seed from, and the pane paints the
// MODEL's queries as they happen - the user literally watches the model
// explore. The user's own strip queries run unrestricted; only the model is
// read-only (store doc).
import { computed, ref, watch } from 'vue'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import GdsDrawer from '@/components/chat/graph/GdsDrawer.vue'
import GraphCanvas from '@/components/chat/graph/GraphCanvas.vue'
import QueryStrip from '@/components/chat/graph/QueryStrip.vue'
import {
  applyEmphasis,
  buildGraph,
  cellText,
  clearEmphasis,
  hasEntities,
  mergeIntoGraph,
  withInducedEdges,
  type GdsAlgorithm,
  type GraphResult,
  type QueryResponse,
} from '@/lib/graph/session'
import DataTable from '@/components/ui/DataTable.vue'
import { fmtFileSize } from '@/lib/format'
import { fleetLabel } from '@/lib/model-name'
import { useGraphsStore } from '@/stores/graphs'
import { useSettingsStore } from '@/stores/settings'

const graphs = useGraphsStore()
const settings = useSettingsStore()
const dark = computed(() => settings.theme === 'dark')

const view = ref<GraphResult | null>(null)
/** What the canvas + table show: the whole graph, one of the model's runs
 *  (by index), or the user's own strip query. Model runs are PERSISTENT
 *  chips - switching between the queries the model generated must be one
 *  click, not something that vanishes when the next query lands. */
const sel = ref<'all' | 'user' | number>('all')
const running = ref(false)
const queryError = ref('')
const gdsOpen = ref(false)
const canvas = ref<InstanceType<typeof GraphCanvas> | null>(null)
const lastRun = ref<QueryResponse | null>(null)
/** The whole graph, kept mounted: a query result is EMPHASIZED on it (pulse
 *  + fade, camera untouched) rather than replacing the view - pointing at
 *  the answer beats teleporting to it. Standalone replacement only when the
 *  result's nodes are not in the rendered graph at all. */
let fullView: GraphResult | null = null
let fullRes: QueryResponse | null = null

async function showAll(): Promise<void> {
  const s = graphs.sessionFor(graphs.conversationId)
  if (!s) return
  if (!fullView) {
    fullRes = await s.renderAll()
    fullView = buildGraph(fullRes, dark.value)
  }
  if (view.value !== fullView) view.value = fullView
  clearEmphasis(fullView.graph)
  sel.value = 'all'
  lastRun.value = null
}

/** Point a result out on the full graph; stand alone only when none of its
 *  nodes are rendered there (a truncated big graph). */
async function paintResult(res: QueryResponse): Promise<void> {
  if (fullView) {
    if (view.value !== fullView) view.value = fullView
    const hits = applyEmphasis(fullView.graph, res, dark.value)
    if (hits > 0) return
    clearEmphasis(fullView.graph)
  }
  const s = graphs.sessionFor(graphs.conversationId)
  const linked = s ? await withInducedEdges(s, res) : res
  view.value = buildGraph(linked, dark.value)
}

/** Show model run `i` - canvas emphasis if it returned entities, rows either
 *  way. */
async function showRun(i: number): Promise<void> {
  const run = graphs.modelRuns[i]
  if (!run) return
  sel.value = i
  queryError.value = ''
  if (!run.response) {
    // Restored from the conversation's stored tool calls: run it now.
    // Deterministic - the graph came from the same tvdb the model queried.
    const s = graphs.sessionFor(graphs.conversationId)
    if (!s) return
    try {
      run.response = await s.run(run.cypher)
    } catch (e) {
      queryError.value = e instanceof Error ? e.message : String(e)
      return
    }
  }
  const r = run.response
  if (hasEntities(r)) {
    await paintResult(r)
  } else if (fullView) {
    // Property-only result: nothing to point at, so un-point the previous.
    if (view.value !== fullView) view.value = fullView
    clearEmphasis(fullView.graph)
  }
  lastRun.value = run.response
}

/** Short chip label: the query's verb + pattern start, not a bare number.
 *  In a compare, several lanes share this one pane's history, so the chip
 *  leads with who asked - but only when more than one model actually has. */
const multiModel = computed(
  () => new Set(graphs.modelRuns.map((r) => r.model).filter(Boolean)).size > 1,
)
function runLabel(run: { cypher: string; model: string }): string {
  const t = run.cypher.replace(/\s+/g, ' ').trim()
  const q = t.length > 24 ? t.slice(0, 23) + '...' : t
  return multiModel.value && run.model ? `${fleetLabel(run.model)} · ${q}` : q
}

watch(
  () => graphs.status,
  (st) => {
    if (st === 'ready') void showAll()
    else {
      view.value = null
      fullView = null
      fullRes = null
      lastRun.value = null
      sel.value = 'all'
    }
  },
  { immediate: true },
)

// Follow the model live: a new run selects itself, and stays reachable as a
// chip after the next one lands. History-restored chips (no response yet)
// are not auto-run - the user clicks the one they want back.
watch(
  () => graphs.modelRuns.length,
  (n) => {
    if (n > 0 && graphs.modelRuns[n - 1]?.response) void showRun(n - 1)
  },
)

// A theme flip rebuilds the full graph (colors are baked into attributes)
// and re-applies whatever was selected.
watch(dark, async () => {
  if (!fullRes) return
  fullView = buildGraph(fullRes, dark.value)
  view.value = fullView
  if (typeof sel.value === 'number') await showRun(sel.value)
  else if (sel.value === 'user' && lastRun.value) await paintResult(lastRun.value)
})

async function runQuery(cypher: string): Promise<void> {
  const s = graphs.sessionFor(graphs.conversationId)
  if (!s || running.value) return
  running.value = true
  queryError.value = ''
  try {
    const res = await s.run(cypher)
    lastRun.value = res
    if (hasEntities(res)) {
      await paintResult(res)
    }
    sel.value = 'user'
  } catch (e) {
    queryError.value = e instanceof Error ? e.message : String(e)
    lastRun.value = null
  } finally {
    running.value = false
  }
}

/** Double-click expansion (upstream behavior): pull the node's neighborhood
 *  into the CURRENT view and let the layout absorb it. */
async function expandNode(fgId: number, nodeKey: string): Promise<void> {
  const s = graphs.sessionFor(graphs.conversationId)
  const g = view.value?.graph
  if (!s || !g) return
  try {
    const res = await s.run(`MATCH (n)-[r]-(m) WHERE id(n) = ${fgId} RETURN n, r, m LIMIT 50`)
    if (mergeIntoGraph(g, res, dark.value, nodeKey) > 0) canvas.value?.relayoutMerged()
  } catch (e) {
    queryError.value = e instanceof Error ? e.message : String(e)
  }
}

/** A GDS stream result painted onto the graph: scalar scores as a size+heat
 *  ramp, communities as palette colors. Everything else shows as rows only.
 *  "Full graph" rebuilds from scratch, which is also the reset. */
function onGdsRan(alg: GdsAlgorithm, res: QueryResponse): void {
  lastRun.value = res
  sel.value = 'user'
  const g = fullView?.graph
  if (!g) return
  if (view.value !== fullView) view.value = fullView
  const rows = res.rows ?? []
  const idCol = res.columns.findIndex((c) => /nodeid/i.test(c))
  if (idCol < 0 || rows.length === 0) return
  const valCol = res.columns.findIndex((c, i) => i !== idCol)
  if (alg.outputKind === 'nodeScalar' && valCol >= 0) {
    const vals = rows.map((r) => Number(r[valCol])).filter(Number.isFinite)
    const lo = Math.min(...vals)
    const hi = Math.max(...vals)
    const span = hi - lo || 1
    for (const r of rows) {
      const key = String(r[idCol])
      if (!g.hasNode(key)) continue
      const t = (Number(r[valCol]) - lo) / span
      const heat = lerpHex(dark.value ? '#5B9BD5' : '#9AA1AB', '#E8716E', t)
      g.mergeNodeAttributes(key, { color: heat, originalColor: heat, size: 5 + Math.round(t * 9) })
    }
  } else if (alg.outputKind === 'nodeCommunity' && valCol >= 0) {
    for (const r of rows) {
      const key = String(r[idCol])
      if (!g.hasNode(key)) continue
      const c = communityColor(Number(r[valCol]), dark.value)
      g.mergeNodeAttributes(key, { color: c, originalColor: c })
    }
  }
}

const COMMUNITY_LIGHT = ['#0369A1', '#C53030', '#B8860B', '#7C3AED', '#0284C7', '#D44040', '#6366f1', '#2A6BB5']
const COMMUNITY_DARK = ['#38BDF8', '#E8716E', '#D4A030', '#a78bfa', '#7DD3FC', '#f87171', '#818cf8', '#5B9BD5']
function communityColor(id: number, isDark: boolean): string {
  const p = isDark ? COMMUNITY_DARK : COMMUNITY_LIGHT
  return p[Math.abs(Math.trunc(id)) % p.length]
}

function lerpHex(a: string, b: string, t: number): string {
  const pa = parseInt(a.slice(1), 16)
  const pb = parseInt(b.slice(1), 16)
  const mix = (x: number, y: number): number => Math.round(x + (y - x) * t)
  const r = mix((pa >> 16) & 255, (pb >> 16) & 255)
  const gg = mix((pa >> 8) & 255, (pb >> 8) & 255)
  const bl = mix(pa & 255, pb & 255)
  return `#${((r << 16) | (gg << 8) | bl).toString(16).padStart(6, '0')}`
}

const memText = computed(() =>
  graphs.memBytes > 0 ? `${fmtFileSize(graphs.memBytes)} in memory` : '',
)

async function downloadTvdb(): Promise<void> {
  const s = graphs.sessionFor(graphs.conversationId)
  if (!s) return
  const bytes = await s.exportTvdb()
  const buf = new Uint8Array(bytes)
  const url = URL.createObjectURL(new Blob([buf.buffer as ArrayBuffer]))
  const el = document.createElement('a')
  el.href = url
  el.download = graphs.name.replace(/\.tvdb$/i, '').replace(/[^\w.-]+/g, '-') + '.tvdb'
  el.click()
  URL.revokeObjectURL(url)
}

const ROW_CAP = 100
const shownRows = computed(() => (lastRun.value?.rows ?? []).slice(0, ROW_CAP))
/** Column meta for the table: right-align what reads as a number. */
const tableCols = computed(() =>
  (lastRun.value?.columns ?? []).map((label, i) => ({
    label,
    numeric: Number.isFinite(Number(cellText(lastRun.value?.rows?.[0]?.[i]))),
  })),
)
const hiddenRows = computed(() =>
  Math.max(0, (lastRun.value?.total_rows ?? 0) - shownRows.value.length),
)
</script>

<template>
  <div class="gp">
    <div class="gp__bar">
      <Icon name="hard-drive" :size="13" />
      <span class="gp__name">{{ graphs.name }}</span>
      <span class="gp__stat">
        {{ graphs.counts.nodes.toLocaleString() }} nodes ·
        {{ graphs.counts.edges.toLocaleString() }} edges
      </span>
      <span v-if="memText" class="gp__note">{{ memText }}</span>
      <div class="gp__spacer" />
      <Tooltip label="Download as .tvdb - opens in a Traverse server">
        <button class="gp__fold" type="button" aria-label="Download as tvdb" @click="downloadTvdb">
          <Icon name="download" :size="14" />
        </button>
      </Tooltip>
      <Tooltip label="Run a graph algorithm (PageRank, communities, paths...)">
        <button class="gp__fold" type="button" aria-label="Graph algorithms" @click="gdsOpen = !gdsOpen">
          <Icon name="activity" :size="14" />
        </button>
      </Tooltip>
      <Tooltip label="The model can read this graph with queries, never change it">
        <span class="gp__ro"><Icon name="lock" :size="11" /> Read-only</span>
      </Tooltip>
      <Tooltip label="Hide the graph">
        <button
          class="gp__fold"
          type="button"
          aria-label="Hide the graph"
          @click="graphs.folded = true"
        >
          <Icon name="panel-left" :size="14" />
        </button>
      </Tooltip>
    </div>

    <div v-if="graphs.modelRuns.length || sel === 'user'" class="gp__runs">
      <button class="gp__chip" :class="{ 'gp__chip--on': sel === 'all' }" @click="showAll">
        Full graph
      </button>
      <Tooltip
        v-for="(r, i) in graphs.modelRuns"
        :key="i"
        :label="r.model ? `${fleetLabel(r.model)}: ${r.cypher}` : r.cypher"
      >
        <button class="gp__chip" :class="{ 'gp__chip--on': sel === i }" @click="showRun(i)">
          {{ runLabel(r) }}
        </button>
      </Tooltip>
      <span v-if="sel === 'user'" class="gp__chip gp__chip--on">your query</span>
    </div>

    <div class="gp__canvas">
      <p v-if="graphs.status === 'loading'" class="gp__empty">
        <Icon name="spinner" :size="14" /> {{ graphs.phase || 'Loading...' }}
      </p>
      <p v-else-if="graphs.status === 'error'" class="gp__empty gp__empty--err">
        {{ graphs.error }}
      </p>
      <p v-else-if="!view || view.nodeCount === 0" class="gp__empty">Nothing to draw yet.</p>
      <GraphCanvas
        v-else
        ref="canvas"
        :graph="view.graph"
        :dark="dark"
        :export-name="graphs.name"
        @expand="expandNode"
      />
      <GdsDrawer
        v-if="gdsOpen && graphs.status === 'ready'"
        :session="graphs.sessionFor(graphs.conversationId)!"
        @ran="onGdsRan"
        @close="gdsOpen = false"
      />
    </div>

    <div v-if="graphs.status === 'ready'" class="gp__query">
      <QueryStrip :running="running" @run="runQuery" />
      <p v-if="queryError" class="gp__qerr">{{ queryError }}</p>
      <div v-if="lastRun && !queryError && shownRows.length" class="gp__table">
        <DataTable :columns="tableCols" :rows="shownRows" :format="(c) => cellText(c)" />
        <p v-if="hiddenRows" class="gp__note">
          {{ hiddenRows.toLocaleString() }} more rows - add a LIMIT or aggregate.
        </p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.gp {
  display: flex;
  flex-direction: column;
  min-height: 0;
  width: var(--pk-graphpane-width, 420px);
  flex-shrink: 0;
  background: var(--pk-bg-surface);
}
.gp__bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-bottom: 1px solid var(--pk-border-subtle);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  /* narrow panes: shrink and ellipsize, never wrap into a second line */
  flex-wrap: nowrap;
  min-width: 0;
}
.gp__bar > * {
  flex-shrink: 0;
}
.gp__bar > .gp__name,
.gp__bar > .gp__stat,
.gp__bar > .gp__note {
  flex-shrink: 1;
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.gp__name {
  font-weight: 500;
  color: var(--pk-text-primary);
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  max-width: 160px;
}
.gp__stat {
  font-variant-numeric: tabular-nums;
}
.gp__note {
  color: var(--pk-text-muted);
}
.gp__spacer {
  flex: 1;
}
.gp__link {
  padding: 0;
  border: 0;
  background: none;
  color: var(--pk-accent, var(--pk-text-primary));
  font: inherit;
  cursor: pointer;
}
.gp__runs {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 5px 10px;
  border-bottom: 1px solid var(--pk-border-subtle);
  overflow-x: auto;
  white-space: nowrap;
}
.gp__chip {
  flex: none;
  padding: 2px 9px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-full, 999px);
  background: none;
  color: var(--pk-text-secondary);
  font: inherit;
  font-size: var(--pk-font-size-xs);
  font-family: var(--pk-font-mono);
  cursor: pointer;
}
.gp__chip--on {
  background: var(--pk-bg-base);
  color: var(--pk-text-primary);
  border-color: var(--pk-border-strong, var(--pk-border-default));
}
.gp__fold {
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
.gp__fold:hover {
  color: var(--pk-text-primary);
  background: var(--pk-bg-base);
}
.gp__ro {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--pk-text-muted);
  white-space: nowrap;
}
.gp__canvas {
  flex: 1;
  min-height: 200px;
  position: relative;
}
.gp__empty {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 100%;
  margin: 0;
  padding: 0 20px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  text-align: center;
}
.gp__empty--err {
  color: var(--pk-danger, #b91c1c);
}
.gp__query {
  border-top: 1px solid var(--pk-border-subtle);
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 40%;
  overflow-y: auto;
}
.gp__qerr {
  margin: 0;
  color: var(--pk-danger, #b91c1c);
  font-size: var(--pk-font-size-xs);
  overflow-wrap: anywhere;
}
.gp__table {
  display: flex;
  flex-direction: column;
  max-height: 220px;
  min-height: 0;
}
</style>
