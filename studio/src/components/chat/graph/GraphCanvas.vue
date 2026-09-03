<script setup lang="ts">
// The sigma mount and nothing else: takes a laid-out graphology graph, draws
// it, and owns hover/selection. Sigma wiring follows traverse studio's
// useGraphExplorer.initializeSigma (@ c1aaee4) trimmed to what this panel
// uses: circles, straight arrow edges, hover-fades-the-rest, click for a
// property card. A theme flip re-creates the renderer - the WebGL canvas
// bakes colors in, so a full re-init is the correct move, not a workaround.
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import Graph from 'graphology'
import Sigma from 'sigma'
import { EdgeArrowProgram } from 'sigma/rendering'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import FA2Layout from 'graphology-layout-forceatlas2/worker'
import noverlap from 'graphology-layout-noverlap'
import { downloadAsPNG } from '@sigma/export-image'
import { D3ForceLayoutController } from '@/lib/graph/force-layout'
import NodeHexagonProgram from '@/lib/graph/node-hexagon'
import NodePulseProgram from '@/lib/graph/node-pulse'
import { GRAPH_COLORS_LIGHT, fadeColor, graphColors } from '@/lib/graph/session'

const props = defineProps<{
  graph: Graph | null
  dark: boolean
  /** names the exported PNG; '' falls back to "graph". */
  exportName?: string
}>()
const emit = defineEmits<{
  /** double-click on a node: the caller expands its neighborhood. */
  expand: [fgId: number, nodeKey: string]
}>()

const host = ref<HTMLElement | null>(null)
let sigma: Sigma | null = null
const hovered = ref<string | null>(null)
let highlightNodes = new Set<string>()
let highlightEdges = new Set<string>()

interface Selected {
  label: string
  types: string[]
  properties: Record<string, unknown>
  /** set for an edge: "from -> to" by display name. */
  endpoints?: string
}
const selected = ref<Selected | null>(null)

// ── pulse ripple: rAF advances the shader clock only while any node uses
// the pulse program (query-result emphasis sets it), so an idle canvas
// costs zero frames. Same driver traverse studio runs.
let pulseRaf: number | null = null
/** The ripple animates long enough to draw the eye, then freezes into a
 *  static highlight: an unbounded 30fps WebGL loop held a Chrome renderer at
 *  ~15% of a core (+12% GPU process) for as long as a restored chat kept its
 * emphasis - which is forever. Re-emphasis restarts it. */
const PULSE_SECONDS = 8
function startPulse(): void {
  if (pulseRaf != null) return
  const start = performance.now() / 1000
  let odd = false
  const tick = (): void => {
    if (performance.now() / 1000 - start > PULSE_SECONDS) {
      pulseRaf = null
      return // frozen: the emphasis styling stays, the clock stops
    }
    pulseRaf = requestAnimationFrame(tick)
    // Half-rate, repaint-only: a bare refresh() re-runs the whole data
    // pipeline (indexation, label grid) and this loop runs for as long as
    // anything is emphasized - a full reprocess 60x/s pinned a core (seen in
    // Chrome's task manager). The ripple only needs the clock to
    // advance and a redraw.
    odd = !odd
    if (odd) return
    NodePulseProgram.currentTime = performance.now() / 1000 - start
    sigma?.refresh({ skipIndexation: true })
  }
  pulseRaf = requestAnimationFrame(tick)
}
function stopPulse(): void {
  if (pulseRaf != null) {
    cancelAnimationFrame(pulseRaf)
    pulseRaf = null
    NodePulseProgram.currentTime = 0
  }
}
function syncPulse(): void {
  let has = false
  props.graph?.forEachNode((_id, a) => {
    if (a.type === 'pulse') has = true
  })
  if (has) startPulse()
  else stopPulse()
}

// ── layout: traverse's default, ported faithfully - ForceAtlas2 in its own
// web worker, animating the scatter into place with convergence detection
// and a hard 5 s cap; labels hidden while it runs, noverlap cleanup after.
let fa2: InstanceType<typeof FA2Layout> | null = null
let fa2Check: ReturnType<typeof setInterval> | null = null
let d3ctl: D3ForceLayoutController | null = null

/** ForceAtlas2 by default, d3-force on the toggle - the same pair traverse
 *  studio switches between. */
const layoutKind = ref<'fa2' | 'd3'>('fa2')

function stopLayout(): void {
  if (fa2Check) {
    clearInterval(fa2Check)
    fa2Check = null
  }
  fa2?.kill()
  fa2 = null
  d3ctl?.kill()
  d3ctl = null
}

function layoutDone(graph: Graph, s: Sigma): void {
  animating = false
  noverlap.assign(graph, 120)
  s.setSetting('renderLabels', true)
  s.getCamera().animatedReset({ duration: 300 })
}

function animateLayout(graph: Graph, s: Sigma): void {
  stopLayout()
  animating = true
  s.setSetting('renderLabels', false)

  if (layoutKind.value === 'd3') {
    // Faithful port of upstream startD3ForceLayout: the animated controller
    // ticks against the live sigma until convergence or the 5 s cap.
    const ctl = new D3ForceLayoutController(graph, s)
    d3ctl = ctl
    ctl.start({
      maxRuntime: 5000,
      onEnd: () => {
        if (d3ctl === ctl) d3ctl = null
        layoutDone(graph, s)
      },
    })
    return
  }

  const inferred = forceAtlas2.inferSettings(graph)
  const layout = new FA2Layout(graph, {
    settings: {
      ...inferred,
      barnesHutOptimize: graph.order > 100,
      barnesHutTheta: 0.5,
      adjustSizes: true,
      gravity: 1,
      scalingRatio: (inferred.scalingRatio || 1) * 10,
      strongGravityMode: true,
      outboundAttractionDistribution: true,
    },
  })
  fa2 = layout
  layout.start()

  const startTime = performance.now()
  // Sample up to 100 nodes for the convergence check, like upstream - the
  // check must not cost more than the layout.
  const sample: string[] = []
  graph.forEachNode((n) => {
    if (sample.length < 100) sample.push(n)
  })
  const prev = new Map<string, { x: number; y: number }>()
  sample.forEach((n) => {
    const a = graph.getNodeAttributes(n)
    prev.set(n, { x: a.x as number, y: a.y as number })
  })
  let stable = 0

  const finish = (): void => {
    stopLayout()
    layoutDone(graph, s)
  }

  fa2Check = setInterval(() => {
    if (!fa2) return
    if (performance.now() - startTime > 5000) {
      finish()
      return
    }
    let movement = 0
    sample.forEach((n) => {
      const a = graph.getNodeAttributes(n)
      const p = prev.get(n)
      if (p) movement += Math.hypot((a.x as number) - p.x, (a.y as number) - p.y)
      prev.set(n, { x: a.x as number, y: a.y as number })
    })
    if (sample.length > 0 && movement / sample.length < 0.5) {
      stable++
      if (stable >= 3) finish()
    } else {
      stable = 0
    }
  }, 400)
}

/** True while a layout animation owns the positions - dragging against it
 *  fights the simulation (upstream guards the same way). */
let animating = false

function mount(): void {
  stopLayout()
  stopPulse()
  sigma?.kill()
  sigma = null
  selected.value = null
  hovered.value = null
  if (!host.value || !props.graph || props.graph.order === 0) return

  const colors = graphColors(props.dark)
  const s = new Sigma(props.graph, host.value, {
    renderEdgeLabels: false,
    labelRenderedSizeThreshold: 8,
    labelFont: 'Inter, system-ui, sans-serif',
    labelSize: 10,
    labelWeight: '500',
    labelColor: { attribute: 'labelColor', color: colors.label },
    defaultNodeColor: colors.defaultNode,
    defaultEdgeColor: colors.defaultEdge,
    defaultEdgeType: 'straight',
    nodeProgramClasses: { pulse: NodePulseProgram, hexagon: NodeHexagonProgram },
    edgeProgramClasses: { straight: EdgeArrowProgram },
    enableEdgeEvents: true,
    allowInvalidContainer: true,
    minCameraRatio: 0.1,
    maxCameraRatio: 10,
    nodeReducer: (node, data) => {
      const res = { ...data }
      if (hovered.value && hovered.value !== node && !highlightNodes.has(node)) {
        res.color = fadeColor(String(data.originalColor), props.dark)
        res.label = ''
      } else if (hovered.value) {
        res.forceLabel = true
        if (hovered.value === node) {
          // sigma's hover bubble is WHITE in every theme, so the hovered
          // label must take the LIGHT theme's dark text or dark mode reads
          // light-grey-on-white (upstream does the same).
          res.labelColor = GRAPH_COLORS_LIGHT.label
        }
      }
      return res
    },
    edgeReducer: (edge, data) => {
      const res = { ...data }
      if (hovered.value && !highlightEdges.has(edge)) {
        res.color = fadeColor(String(data.originalColor), props.dark)
      }
      if ((res.size as number) < 2) res.size = 2
      return res
    },
  })

  // Node dragging, ported from upstream setupEventHandlers: down arms, body
  // moves write positions straight into the graph, up releases. The custom
  // bbox pin stops sigma re-fitting the camera around the moving node, and
  // `hasDragged` keeps the click-card from opening at the end of a drag.
  let draggedNode: string | null = null
  let isDragging = false
  let hasDragged = false
  s.on('downNode', (e) => {
    if (animating) return
    isDragging = true
    hasDragged = false
    draggedNode = e.node
    if (!s.getCustomBBox()) s.setCustomBBox(s.getBBox())
  })
  s.on('moveBody', ({ event }) => {
    if (!isDragging || !draggedNode || !props.graph) return
    hasDragged = true
    const pos = s.viewportToGraph(event)
    props.graph.setNodeAttribute(draggedNode, 'x', pos.x)
    props.graph.setNodeAttribute(draggedNode, 'y', pos.y)
    event.preventSigmaDefault()
    event.original.preventDefault()
    event.original.stopPropagation()
  })
  const release = (): void => {
    isDragging = false
    draggedNode = null
  }
  s.on('upNode', release)
  s.on('upStage', release)

  s.on('enterNode', ({ node }) => {
    hovered.value = node
    const g = props.graph!
    highlightNodes = new Set([node, ...g.neighbors(node)])
    highlightEdges = new Set(g.edges(node))
    s.getContainer().style.cursor = 'pointer'
    s.refresh()
  })
  s.on('leaveNode', () => {
    hovered.value = null
    highlightNodes = new Set()
    highlightEdges = new Set()
    s.getContainer().style.cursor = 'default'
    s.refresh()
  })
  s.on('enterEdge', () => {
    s.getContainer().style.cursor = 'pointer'
  })
  s.on('leaveEdge', () => {
    s.getContainer().style.cursor = 'default'
  })
  s.on('clickNode', ({ node }) => {
    if (animating || hasDragged) {
      hasDragged = false
      return
    }
    const a = props.graph!.getNodeAttributes(node)
    selected.value = {
      label: String(a.label ?? node),
      types: (a.nodeLabels as string[]) ?? [],
      properties: (a.properties as Record<string, unknown>) ?? {},
    }
  })
  s.on('clickEdge', ({ edge }) => {
    const g = props.graph!
    const a = g.getEdgeAttributes(edge)
    const srcName = String(g.getNodeAttribute(g.source(edge), 'label') ?? g.source(edge))
    const tgtName = String(g.getNodeAttribute(g.target(edge), 'label') ?? g.target(edge))
    selected.value = {
      label: `[:${String(a.edgeType ?? a.label ?? 'edge')}]`,
      types: [],
      properties: (a.properties as Record<string, unknown>) ?? {},
      endpoints: `${srcName} \u2192 ${tgtName}`,
    }
  })
  s.on('clickStage', () => (selected.value = null))
  s.on('doubleClickNode', ({ node, event }) => {
    // The default dblclick zooms the camera; expanding is the useful action
    // on a graph you are exploring (upstream behavior).
    event.preventSigmaDefault()
    const fgId = props.graph!.getNodeAttribute(node, 'fgId')
    if (typeof fgId === 'number') emit('expand', fgId, node)
  })

  sigma = s

  animateLayout(props.graph, s)

  // Emphasis flips node types after mount, so the ripple clock follows the
  // graph's attribute events. Coalesced: position updates fire this per node
  // per layout tick, and a full-graph scan on each would be O(n^2) a frame.
  syncPulse()
  props.graph.on('nodeAttributesUpdated', scheduleSyncPulse)
}

let pulseSyncTimer: ReturnType<typeof setTimeout> | null = null
function scheduleSyncPulse(): void {
  if (pulseSyncTimer != null) return
  pulseSyncTimer = setTimeout(() => {
    pulseSyncTimer = null
    syncPulse()
  }, 60)
}

watch([() => props.graph, () => props.dark, host], mount, { immediate: true })

// Panel drags resize this container on every mousemove, and each canvas
// resize is a WebGL buffer reallocation - sixty of those a second is the
// sluggish drag. The ResizeHandle already marks drags
// globally (body.pk-resizing), so: pin the host to its pixel size for the
// duration, resize once at drop.
let dragWatch: MutationObserver | null = null
onMounted(() => {
  dragWatch = new MutationObserver(() => {
    const h = host.value
    if (!h) return
    if (document.body.classList.contains('pk-resizing')) {
      if (!h.style.width) {
        h.style.width = `${h.clientWidth}px`
        h.style.height = `${h.clientHeight}px`
      }
    } else if (h.style.width) {
      h.style.width = ''
      h.style.height = ''
      requestAnimationFrame(() => sigma?.refresh())
    }
  })
  dragWatch.observe(document.body, { attributes: true, attributeFilter: ['class'] })
})

onBeforeUnmount(() => {
  dragWatch?.disconnect()
  stopLayout()
  stopPulse()
  if (pulseSyncTimer != null) clearTimeout(pulseSyncTimer)
  sigma?.kill()
  sigma = null
})

/** Jiggle + re-animate: upstream applyLayout's randomize, so a re-run gives
 *  the forces something to work with instead of a no-op. */
function relayout(): void {
  const g = props.graph
  if (!g || !sigma) return
  g.forEachNode((n, a) => {
    g.setNodeAttribute(n, 'x', (a.x as number) + (Math.random() - 0.5) * 50)
    g.setNodeAttribute(n, 'y', (a.y as number) + (Math.random() - 0.5) * 50)
  })
  animateLayout(g, sigma)
}

function switchLayout(kind: 'fa2' | 'd3'): void {
  if (layoutKind.value === kind) return
  layoutKind.value = kind
  relayout()
}

/** Re-run the layout over the CURRENT graph - after a merge, new nodes need
 *  pulling out of their scatter. Exposed so the pane can call it. */
function relayoutMerged(): void {
  if (props.graph && sigma) animateLayout(props.graph, sigma)
}
defineExpose({ relayoutMerged })

/** PNG of the visible layers, 2x for crisp text - upstream exportAsImage. */
function exportImage(): void {
  if (!sigma) return
  const container = sigma.getContainer()
  const SKIPPED = new Set(['mouse', 'edgeLabels', 'hovers', 'hoverNodes'])
  const layers = [...container.querySelectorAll('canvas')]
    .map((c) => (c.className || '').replace('sigma-', ''))
    .filter((n) => n && !SKIPPED.has(n))
  void downloadAsPNG(sigma, {
    fileName: (props.exportName || 'graph').replace(/\.tvdb$/i, '').replace(/[^\w.-]+/g, '-'),
    layers,
    backgroundColor: props.dark ? '#12161a' : '#ffffff',
    width: container.offsetWidth * 2,
    height: container.offsetHeight * 2,
  }).catch((e) => console.error('graph image export failed', e))
}

function zoomFit(): void {
  sigma?.getCamera().animatedReset()
}
function zoomIn(): void {
  sigma?.getCamera().animatedZoom({ duration: 300 })
}
function zoomOut(): void {
  sigma?.getCamera().animatedUnzoom({ duration: 300 })
}

/** The card shows what the value is; long strings get cut so one biography
 *  property cannot swallow the card. */
function propText(v: unknown): string {
  const s = typeof v === 'object' ? JSON.stringify(v) : String(v)
  return s.length > 120 ? s.slice(0, 117) + '...' : s
}
</script>

<template>
  <div class="gc">
    <div ref="host" class="gc__canvas" />
    <div class="gc__zoom">
      <Tooltip label="Save as image">
        <button class="gc__btn" @click="exportImage"><Icon name="image" :size="13" /></button>
      </Tooltip>
      <div class="gc__seg">
        <Tooltip label="ForceAtlas2 layout">
          <button
            class="gc__segbtn"
            :class="{ 'gc__segbtn--on': layoutKind === 'fa2' }"
            @click="switchLayout('fa2')"
          >
            FA2
          </button>
        </Tooltip>
        <Tooltip label="D3 force layout">
          <button
            class="gc__segbtn"
            :class="{ 'gc__segbtn--on': layoutKind === 'd3' }"
            @click="switchLayout('d3')"
          >
            D3
          </button>
        </Tooltip>
      </div>
      <Tooltip label="Re-run layout">
        <button class="gc__btn" @click="relayout"><Icon name="regenerate" :size="13" /></button>
      </Tooltip>
      <Tooltip label="Fit">
        <button class="gc__btn" @click="zoomFit"><Icon name="square" :size="13" /></button>
      </Tooltip>
      <Tooltip label="Zoom in">
        <button class="gc__btn" @click="zoomIn"><Icon name="plus" :size="13" /></button>
      </Tooltip>
      <Tooltip label="Zoom out">
        <button class="gc__btn" @click="zoomOut"><Icon name="minus" :size="13" /></button>
      </Tooltip>
    </div>
    <div v-if="selected" class="gc__card">
      <div class="gc__cardhead">
        <span class="gc__cardname">{{ selected.label }}</span>
        <span v-for="t in selected.types" :key="t" class="gc__chip">{{ t }}</span>
      </div>
      <p v-if="selected.endpoints" class="gc__endpoints">{{ selected.endpoints }}</p>
      <dl v-if="Object.keys(selected.properties).length" class="gc__props">
        <template v-for="(v, k) in selected.properties" :key="k">
          <dt>{{ k }}</dt>
          <dd>{{ propText(v) }}</dd>
        </template>
      </dl>
    </div>
  </div>
</template>

<style scoped>
.gc {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 0;
  /* clips the pixel-frozen canvas while a panel drag shrinks the pane */
  overflow: hidden;
}
.gc__canvas {
  width: 100%;
  height: 100%;
}
.gc__zoom {
  position: absolute;
  right: 8px;
  bottom: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.gc__seg {
  display: flex;
  flex-direction: column;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-sm);
  overflow: hidden;
}
.gc__segbtn {
  padding: 3px 0;
  width: 26px;
  border: 0;
  background: var(--pk-bg-surface);
  color: var(--pk-text-muted);
  font-size: 9px;
  font-weight: 600;
  cursor: pointer;
}
.gc__segbtn--on {
  background: var(--pk-bg-base);
  color: var(--pk-text-primary);
}
.gc__btn {
  display: grid;
  place-items: center;
  width: 26px;
  height: 26px;
  padding: 0;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-surface);
  color: var(--pk-text-secondary);
  cursor: pointer;
}
.gc__btn:hover {
  color: var(--pk-text-primary);
}
.gc__card {
  position: absolute;
  left: 8px;
  bottom: 8px;
  max-width: 280px;
  max-height: 45%;
  overflow-y: auto;
  padding: 10px 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  box-shadow: var(--pk-shadow-md, 0 4px 12px rgba(0, 0, 0, 0.12));
}
.gc__cardhead {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 6px;
}
.gc__endpoints {
  margin: 0 0 6px;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-xs);
}
.gc__cardname {
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.gc__chip {
  padding: 1px 7px;
  border-radius: var(--pk-radius-full, 999px);
  background: var(--pk-bg-base);
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-xs);
}
.gc__props {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 2px 10px;
  margin: 0;
  font-size: var(--pk-font-size-xs);
}
.gc__props dt {
  color: var(--pk-text-muted);
  white-space: nowrap;
}
.gc__props dd {
  margin: 0;
  color: var(--pk-text-primary);
  overflow-wrap: anywhere;
}
</style>
