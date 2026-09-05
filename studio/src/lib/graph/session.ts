// The one file that touches @truespar/traverse-wasm.
//
// A GraphSession is a live in-browser Traverse engine (a Web Worker holding
// the single-threaded WASM build - D3)
// seeded from a graph artifact's Cypher script. The artifact text is the
// source of truth; the session is a playground - mutations live in the worker
// and die with it, exports are explicit (D1). No OPFS here: persisting a copy
// browser-side would just mint a stale second truth for the create case (D6).
import type { AlgorithmsResponse, GdsAlgorithm, ImportResult, QueryResponse, SchemaResponse } from '@truespar/traverse-wasm'
import { TraverseDb } from '@truespar/traverse-wasm'
import Graph from 'graphology'

export type { AlgorithmsResponse, GdsAlgorithm, ImportResult, QueryResponse, SchemaResponse }

/** Hydrated entity shapes from the wasm's QueryResponse.nodes / .edges. */
export interface TvNode {
  _type: 'node'
  id: number
  labels: string[]
  properties: Record<string, unknown>
}
export interface TvEdge {
  _type: 'edge'
  id: number
  type: string
  source: number
  target: number
  properties: Record<string, unknown>
}
interface TvPath {
  _type: 'path'
  nodes?: TvNode[]
  edges?: TvEdge[]
}

/** Everything the panel renders after a seed or a query. */
export interface GraphResult {
  graph: Graph
  nodeCount: number
  edgeCount: number
  truncated: boolean
}

export class GraphSession {
  private db: TraverseDb | null = null

  /** Spawns the worker and boots the WASM. One session per open panel. */
  async open(): Promise<void> {
    this.db = await TraverseDb.open()
  }

  private need(): TraverseDb {
    if (!this.db) throw new Error('graph session is closed')
    return this.db
  }

  /** Replace the graph with the artifact's script. Returns the per-statement
   *  import result - errors in it are shown, never swallowed. */
  async seed(cypher: string): Promise<ImportResult> {
    const db = this.need()
    await db.clear()
    return await db.importCypher(cypher)
  }

  /** The default view: every node, every relationship. The wasm caps the
   *  hydrated entity arrays at 1024, which is also our render bound (D7). */
  async renderAll(): Promise<QueryResponse> {
    return await this.need().query('MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m')
  }

  /** A user query from the strip. The deadline keeps a runaway pattern from
   *  wedging the worker; the executor checks it at iteration boundaries. */
  async run(cypher: string, timeoutMs = 10_000): Promise<QueryResponse> {
    return await this.need().query(cypher, null, { timeoutMs })
  }

  async schema(): Promise<SchemaResponse> {
    return await this.need().schema()
  }

  /** The 32-procedure GDS catalog, same shape as the native server's. */
  async algorithms(): Promise<AlgorithmsResponse> {
    return await this.need().algorithms()
  }

  async stats(): Promise<{ nodes: number; edges: number }> {
    return await this.need().stats()
  }

  /** The graduation path: the exported bytes round-trip with the native
   *  server's Database::open. */
  async exportTvdb(): Promise<Uint8Array> {
    return await this.need().exportTvdb()
  }

  // ── the attachment (read) case: tvdb bytes + OPFS as a parse cache ──
  // The manager's copy is the source of truth (D6); OPFS only saves
  // refetching + reparsing on reopen, keyed by content hash so a changed
  // file can never hit a stale cache. A failed load is a cache miss, not
  // an error - the caller falls back to fetching the bytes.

  /** Try to restore a previously-cached tvdb. */
  async loadCached(cacheKey: string): Promise<{ ok: boolean; nodes?: number; edges?: number }> {
    try {
      const r = await this.need().load(cacheKey)
      return r.ok ? r : { ok: false }
    } catch {
      return { ok: false }
    }
  }

  /** Load tvdb bytes into the engine and cache them under `cacheKey`.
   *  Cache write happens first (two distinct phases, like the upstream
   *  upload flow) and is best-effort - a full OPFS quota must not block
   *  the load itself. */
  async seedTvdb(
    bytes: Uint8Array,
    cacheKey: string,
  ): Promise<{ ok: boolean; nodes?: number; edges?: number }> {
    const db = this.need()
    try {
      await db.writeBytes(cacheKey, bytes)
      return await db.load(cacheKey)
    } catch {
      // OPFS unavailable or full - load directly, skip the cache.
      return await db.loadTvdb(bytes)
    }
  }

  /** The engine's own in-memory estimate (bytes) - shown after load so the
   *  will-it-fit refusal threshold can be tuned from real numbers. */
  async estimatedMemory(): Promise<number> {
    return await this.need().estimatedMemory()
  }

  close(): void {
    this.db?.close()
    this.db = null
  }
}

// ── entity extraction + graphology build ─────────────────────────────────
// Adapted from traverse studio's useGraphExplorer.buildGraphFromResults
// (@ c1aaee4), minus their style store: colors come from the fixed palettes
// below, sized flat, circles only.

/* Slate-forward palettes, one per theme (lifted with the builder). */
const NODE_PALETTE_LIGHT = [
  '#0369A1', '#2A6BB5', '#C53030', '#B8860B', '#7C3AED',
  '#0284C7', '#3B82C4', '#D44040', '#4A90D0', '#6366f1',
]
const NODE_PALETTE_DARK = [
  '#38BDF8', '#5B9BD5', '#E8716E', '#D4A030', '#a78bfa',
  '#7DD3FC', '#7BB4E0', '#f87171', '#60a5fa', '#818cf8',
]

export const GRAPH_COLORS_LIGHT = {
  edge: '#D0D5DA', label: '#1A1D20', edgeLabel: '#5A6370',
  defaultNode: '#9AA1AB', defaultEdge: '#D0D5DA',
}
export const GRAPH_COLORS_DARK = {
  edge: '#2A3A48', label: '#C9CDCF', edgeLabel: '#7E8B97',
  defaultNode: '#3A4A58', defaultEdge: '#2A3A48',
}

export function graphColors(isDark: boolean): typeof GRAPH_COLORS_LIGHT {
  return isDark ? GRAPH_COLORS_DARK : GRAPH_COLORS_LIGHT
}

/** Blend a hex color toward the theme ground - the de-emphasis fade. */
export function fadeColor(hex: string, isDark: boolean): string {
  const bg = isDark ? [18, 22, 26] : [255, 255, 255]
  const t = isDark ? 0.12 : 0.18
  const n = parseInt(hex.replace('#', ''), 16)
  const mix = (c: number, g: number): number => Math.round(c * t + g * (1 - t))
  const r = mix((n >> 16) & 255, bg[0])
  const g = mix((n >> 8) & 255, bg[1])
  const b = mix(n & 255, bg[2])
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`
}

/** Label -> palette slot, first-seen order, reset per build so colors depend
 *  only on the graph being shown. */
function makeColorPicker(isDark: boolean): (label: string) => string {
  const palette = isDark ? NODE_PALETTE_DARK : NODE_PALETTE_LIGHT
  const slots = new Map<string, number>()
  return (label) => {
    if (!slots.has(label)) slots.set(label, slots.size)
    return palette[slots.get(label)! % palette.length]
  }
}

/** Distinct entities of a response: the server-deduped arrays when present
 *  (computed over the full result set), else a scan of the row cells, which
 *  also surfaces path/list entities. */
export function extractEntities(res: QueryResponse): {
  nodes: Map<number, TvNode>
  edges: Map<number, TvEdge>
} {
  const nodes = new Map<number, TvNode>()
  const edges = new Map<number, TvEdge>()

  function extract(cell: unknown): void {
    if (!cell || typeof cell !== 'object') return
    if (Array.isArray(cell)) {
      cell.forEach(extract)
      return
    }
    const c = cell as Record<string, unknown>
    if (c._type === 'node') nodes.set((c as unknown as TvNode).id, c as unknown as TvNode)
    else if (c._type === 'edge') edges.set((c as unknown as TvEdge).id, c as unknown as TvEdge)
    else if (c._type === 'path') {
      const p = c as unknown as TvPath
      p.nodes?.forEach((n) => nodes.set(n.id, n))
      p.edges?.forEach((e) => edges.set(e.id, e))
    }
  }

  for (const n of res.nodes ?? []) extract(n)
  for (const e of res.edges ?? []) extract(e)
  // Provenance from the engine (traverse 0.8.3): entities the patterns BOUND
  // even when the projection returned only properties - RETURN t.amount now
  // lights up the transfer instead of painting nothing. Superset of the
  // returned arrays; the id-keyed maps dedupe.
  for (const n of res.bound_nodes ?? []) extract(n)
  for (const e of res.bound_edges ?? []) extract(e)
  if (nodes.size === 0 && edges.size === 0) {
    for (const row of res.rows ?? []) for (const cell of row) extract(cell)
  }
  return { nodes, edges }
}

/** Does this response carry anything the canvas can point at? */
export function hasEntities(res: QueryResponse): boolean {
  return (
    (res.nodes?.length ?? 0) > 0 ||
    (res.edges?.length ?? 0) > 0 ||
    (res.bound_nodes?.length ?? 0) > 0 ||
    (res.bound_edges?.length ?? 0) > 0
  )
}

/**
 * A query like `RETURN m, target` returns nodes and never mentions the
 * relationship, so the hydrated entities carry 0 edges and the canvas would
 * paint disconnected dots that READ as broken (seen live with
 * gemma4). Standard graph-browser behavior is to connect result nodes with
 * the edges that exist between them - one cheap induced-subgraph query.
 */
export async function withInducedEdges(
  session: GraphSession,
  res: QueryResponse,
): Promise<QueryResponse> {
  const { nodes, edges } = extractEntities(res)
  if (nodes.size < 2 || edges.size > 0) return res
  const ids = [...nodes.keys()].join(', ')
  try {
    const links = await session.run(
      `MATCH (a)-[r]->(b) WHERE id(a) IN [${ids}] AND id(b) IN [${ids}] RETURN r`,
      5_000,
    )
    if ((links.edges?.length ?? 0) > 0) return { ...res, edges: links.edges }
  } catch {
    // The plain result is still correct - connecting it is a nicety.
  }
  return res
}

/**
 * Point a query result out on the full graph instead of replacing the view:
 * result nodes pulse and grow, edges between them thicken, everything else
 * fades - and the camera never jumps, so the user keeps their bearings
 * ("change size and color to point out"). Returns how many
 * result nodes were actually present in the rendered graph; 0 means the
 * caller should fall back to painting the result standalone.
 */
export function applyEmphasis(graph: Graph, res: QueryResponse, isDark: boolean): number {
  const { nodes } = extractEntities(res)
  const hitIds = new Set([...nodes.keys()].map(String))
  let hits = 0
  graph.forEachNode((id, attrs) => {
    if (hitIds.has(id)) {
      hits++
      graph.mergeNodeAttributes(id, {
        type: 'pulse',
        size: 9,
        color: attrs.originalColor,
        forceLabel: true,
      })
    } else {
      graph.mergeNodeAttributes(id, {
        type: undefined,
        size: 5,
        color: fadeColor(String(attrs.originalColor), isDark),
        forceLabel: false,
      })
    }
  })
  graph.forEachEdge((id, attrs, src, tgt) => {
    // An edge with both endpoints in the result is part of the story even
    // when the query never returned it - this subsumes the induced-edges
    // pass for the emphasis path.
    const on = hitIds.has(src) && hitIds.has(tgt)
    graph.mergeEdgeAttributes(id, {
      size: on ? 3 : 2,
      color: on ? attrs.originalColor : fadeColor(String(attrs.originalColor), isDark),
    })
  })
  return hits
}

/**
 * Merge a query result into an existing rendered graph - the double-click
 * neighborhood expansion (upstream mergeResults). New nodes scatter around
 * `anchor` so the follow-up layout pulls them out from where the user
 * clicked; colors follow the labels already on screen, palette order for
 * labels never seen before.
 */
export function mergeIntoGraph(
  graph: Graph,
  res: QueryResponse,
  isDark: boolean,
  anchor?: string,
): number {
  const { nodes, edges } = extractEntities(res)
  const labelColor = graphColors(isDark).label
  const edgeColor = graphColors(isDark).edge
  const palette = isDark ? NODE_PALETTE_DARK : NODE_PALETTE_LIGHT

  // colors already on screen, by label - merged nodes must match them
  const byLabel = new Map<string, string>()
  graph.forEachNode((_id, a) => {
    if (a.nodeType && a.originalColor) byLabel.set(String(a.nodeType), String(a.originalColor))
  })
  const colorFor = (label: string): string => {
    const known = byLabel.get(label)
    if (known) return known
    const c = palette[byLabel.size % palette.length]
    byLabel.set(label, c)
    return c
  }

  const ax = anchor && graph.hasNode(anchor) ? Number(graph.getNodeAttribute(anchor, 'x')) : 0
  const ay = anchor && graph.hasNode(anchor) ? Number(graph.getNodeAttribute(anchor, 'y')) : 0
  let added = 0

  nodes.forEach((node) => {
    const id = String(node.id)
    if (graph.hasNode(id)) return
    added++
    const label = node.labels[0] || 'Node'
    const color = colorFor(label)
    graph.addNode(id, {
      label: node.properties?.name ? String(node.properties.name) : `${label} #${node.id}`,
      x: ax + (Math.random() - 0.5) * 60,
      y: ay + (Math.random() - 0.5) * 60,
      size: 6,
      color,
      originalColor: color,
      labelColor,
      nodeType: label,
      properties: node.properties,
      nodeLabels: node.labels,
      fgId: node.id,
    })
  })
  edges.forEach((edge) => {
    const srcId = String(edge.source)
    const tgtId = String(edge.target)
    if (!graph.hasNode(srcId) || !graph.hasNode(tgtId)) return
    const dup = graph
      .edges(srcId, tgtId)
      .some((e) => graph.getEdgeAttribute(e, 'fgId') === edge.id)
    if (dup) return
    graph.addEdge(srcId, tgtId, {
      label: edge.type,
      size: 2,
      color: edgeColor,
      originalColor: edgeColor,
      edgeType: edge.type,
      properties: edge.properties,
      fgId: edge.id,
    })
  })
  return added
}

/** Cypher literal for a GDS config map - lifted with buildAlgorithmCall. */
function cypherLiteral(value: unknown): string {
  if (value === null || value === undefined) return 'null'
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : 'null'
  if (typeof value === 'string') return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(cypherLiteral).join(', ')}]`
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => `${k}: ${cypherLiteral(v)}`)
    return `{${entries.join(', ')}}`
  }
  return JSON.stringify(String(value))
}

/** The CALL statement for a GDS run - lifted from upstream buildAlgorithmCall
 *  (both their backends reuse it, so the same Cypher hits the same procedure
 *  here). Stream mode only in this panel: it is the visualize mode. */
export function buildAlgorithmCall(
  algorithm: GdsAlgorithm,
  mode: 'stream' | 'stats',
  config: Record<string, unknown>,
): string {
  const configMap = cypherLiteral(config ?? {})
  const procName = `traverse.${algorithm.name}.${mode}`
  const cols = mode === 'stream' ? algorithm.outputColumns : algorithm.statsColumns
  if (cols.length > 0) {
    const cs = cols.join(', ')
    return `CALL ${procName}(${configMap}) YIELD ${cs} RETURN ${cs}`
  }
  return `CALL ${procName}(${configMap})`
}

/** Undo applyEmphasis - every element back to its built attributes. */
export function clearEmphasis(graph: Graph): void {
  graph.forEachNode((id, attrs) => {
    graph.mergeNodeAttributes(id, {
      type: undefined,
      size: 6,
      color: attrs.originalColor,
      forceLabel: false,
    })
  })
  graph.forEachEdge((id, attrs) => {
    graph.mergeEdgeAttributes(id, { size: 2, color: attrs.originalColor })
  })
}

/**
 * Build a laid-out graphology graph from a QueryResponse.
 */
export function buildGraph(res: QueryResponse, isDark: boolean): GraphResult {
  const { nodes, edges } = extractEntities(res)

  const colorFor = makeColorPicker(isDark)
  const labelColor = graphColors(isDark).label
  const graph = new Graph({ multi: true })
  const spread = Math.max(nodes.size * 5, 100)

  nodes.forEach((node) => {
    const label = node.labels[0] || 'Node'
    const color = colorFor(label)
    const displayLabel = node.properties?.name
      ? String(node.properties.name)
      : `${label} #${node.id}`
    graph.addNode(String(node.id), {
      label: displayLabel,
      x: (Math.random() - 0.5) * spread,
      y: (Math.random() - 0.5) * spread,
      size: 6,
      color,
      originalColor: color,
      labelColor,
      nodeType: label,
      properties: node.properties,
      nodeLabels: node.labels,
      fgId: node.id,
    })
  })

  const edgeColor = graphColors(isDark).edge
  edges.forEach((edge) => {
    const src = String(edge.source)
    const tgt = String(edge.target)
    if (!graph.hasNode(src) || !graph.hasNode(tgt)) return
    graph.addEdge(src, tgt, {
      label: edge.type,
      size: 2,
      color: edgeColor,
      originalColor: edgeColor,
      edgeType: edge.type,
      properties: edge.properties,
      fgId: edge.id,
    })
  })

  // Positions are only a scatter: GraphCanvas runs the animated ForceAtlas2
  // worker on mount (traverse's default look - the graph unfolds into place),
  // so a synchronous 300-tick pre-layout here would be paid twice.

  return {
    graph,
    nodeCount: graph.order,
    edgeCount: graph.size,
    truncated: res.entities_truncated === true,
  }
}

/** Compact display of a result cell for the rows table. Entities render as
 *  their display name rather than a JSON dump. */
export function cellText(cell: unknown): string {
  if (cell === null || cell === undefined) return ''
  if (typeof cell !== 'object') return String(cell)
  const c = cell as Record<string, unknown>
  if (c._type === 'node') {
    const n = c as unknown as TvNode
    const name = n.properties?.name
    return name ? String(name) : `${n.labels?.[0] ?? 'Node'} #${n.id}`
  }
  if (c._type === 'edge') {
    const e = c as unknown as TvEdge
    const ps = Object.entries(e.properties ?? {})
      .slice(0, 3)
      .map(([k, v]) => `${k}: ${String(v)}`)
      .join(', ')
    return ps ? `[:${e.type} {${ps}}]` : `[:${e.type}]`
  }
  if (c._type === 'path') {
    const p = c as unknown as TvPath
    return `path (${p.nodes?.length ?? 0} nodes)`
  }
  try {
    return JSON.stringify(cell)
  } catch {
    return String(cell)
  }
}
