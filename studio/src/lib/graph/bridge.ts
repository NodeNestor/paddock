// The tab side of graph_query (D4/D5).
//
// The manager's graph.rs forwards each model tool call here over a WebSocket,
// because the only Traverse engine in the product is this tab's WASM worker.
// This file owns both halves of the tab's job: the socket that stays
// registered for the conversation, and the policy that turns a model's cypher
// into a tool-result string - the read-only gate and the row/byte compaction
// that keeps results context-priced.
import type { GraphSession, QueryResponse, TvEdge, TvNode } from './session'

/** Rows a tool result may carry. The model is told total_rows and truncated,
 *  so a big match reads as "narrow this", never as "that's all there was". */
const ROW_CAP = 50
/** Ceiling on the serialized result. Halve rows until under it - one row of
 *  huge properties must not blow the model's context. */
const BYTE_CAP = 12_000

/** A result cell as the MODEL reads it. Unlike the pane's display cells,
 *  entities keep their properties - the live failure this fixes: the model
 *  ran the perfect query (RETURN t, a1, a2 ORDER by t.amount) and got bare
 *  "[:TRANSFERRED]" cells back, so it could not see the amounts it had just
 *  sorted by, and spiralled into re-querying. */
function modelCell(cell: unknown): string {
  if (cell === null || cell === undefined) return ''
  if (typeof cell !== 'object') return String(cell)
  const c = cell as Record<string, unknown>
  const props = (p: unknown): string => {
    const entries = Object.entries((p as Record<string, unknown>) ?? {})
    const shown = entries.slice(0, 5).map(([k, v]) => {
      const t = typeof v === 'object' ? JSON.stringify(v) : String(v)
      return `${k}: ${t.length > 40 ? t.slice(0, 37) + '...' : t}`
    })
    if (entries.length > 5) shown.push('...')
    return shown.join(', ')
  }
  if (c._type === 'node') {
    const n = c as unknown as TvNode
    return `(${n.labels?.[0] ?? 'Node'} {${props(n.properties)}})`
  }
  if (c._type === 'edge') {
    const e = c as unknown as TvEdge
    return `[:${e.type} {${props(e.properties)}}]`
  }
  if (c._type === 'path') {
    const p2 = c as { nodes?: unknown[] }
    return `path(${p2.nodes?.length ?? 0} nodes)`
  }
  try {
    return JSON.stringify(cell)
  } catch {
    return String(cell)
  }
}

export interface ModelAnswer {
  /** The finished tool-result text the manager relays verbatim. */
  body: string
  /** The full response, when one was produced - the pane paints it. */
  response: QueryResponse | null
}

/** Execute a MODEL's query under attached-graph policy: classify first via
 *  EXPLAIN (verified: returns the executor's query_type without
 *  running the statement), refuse anything but Read, then run with a deadline
 *  and compact the result. */
export async function answerModelQuery(
  session: GraphSession,
  cypher: string,
): Promise<ModelAnswer> {
  const q = cypher.trim()
  // PROFILE executes; EXPLAIN doesn't. On a read-only graph the former is a
  // write vector, the latter is fine as-is (and needs no second gate).
  if (/^profile\b/i.test(q)) {
    return {
      body: 'PROFILE executes the statement and this graph is read-only for you. Use EXPLAIN for the plan.',
      response: null,
    }
  }
  if (!/^explain\b/i.test(q)) {
    try {
      const gate = await session.run(`EXPLAIN ${q}`, 5_000)
      if (gate.query_type !== 'Read') {
        return {
          body: `This graph is read-only for you: that statement classifies as ${gate.query_type}. Use MATCH / RETURN / aggregation queries only.`,
          response: null,
        }
      }
    } catch {
      // The gate failing to parse means the real run fails identically -
      // let execution produce the one authoritative error message.
    }
  }
  let res: QueryResponse
  try {
    res = await session.run(q, 15_000)
  } catch (e) {
    // A raw parser message ("expected LParen, found LBracket") sent a 9B into
    // a 100-second rumination on first contact. Errors to a
    // model are prompts: restate the one syntax it needs and bound the retry.
    const msg = e instanceof Error ? e.message : String(e)
    const hint = /syntax|expected|parse/i.test(msg)
      ? ' Check the pattern syntax - one dash each side of the brackets: MATCH (a:Label)-[:REL_TYPE]->(b:Label) RETURN a.prop. Fix the statement and retry ONCE; if it fails again, report the error to the user.'
      : ''
    return { body: `Query failed: ${msg}.${hint}`, response: null }
  }

  // Zero rows on a pattern that names a label or relationship type the graph
  // does not have is almost always a token slip (live: a 9B wrote TRANSFER
  // for TRANSFERRED, got [], and went exploring). The engine knows the truth;
  // say it, instead of letting the model re-derive the schema by trial.
  let hint = ''
  if (res.total_rows === 0) {
    const referenced = new Set<string>()
    for (const m of q.matchAll(/[([]\s*\w*\s*:\s*([A-Za-z_]\w*)/g)) referenced.add(m[1])
    if (referenced.size > 0) {
      try {
        const schema = await session.schema()
        const known = new Set([...schema.labels, ...schema.relationship_types])
        const missing = [...referenced].filter((n) => !known.has(n))
        if (missing.length > 0) {
          hint =
            ` Note: ${missing.join(', ')} does not exist in this graph. ` +
            `Labels: ${schema.labels.join(', ')}. ` +
            `Relationship types: ${schema.relationship_types.join(', ')}.`
        }
      } catch {
        // schema unavailable - the plain empty result still stands
      }
    }
  }

  const out = {
    query_type: res.query_type,
    columns: res.columns,
    rows: (res.rows ?? []).slice(0, ROW_CAP).map((r) => r.map(modelCell)),
    total_rows: res.total_rows,
    truncated: res.truncated || (res.total_rows ?? 0) > ROW_CAP,
    time_ms: Math.round(res.time_ms * 10) / 10,
  }
  let body = JSON.stringify(out) + hint
  while (body.length > BYTE_CAP && out.rows.length > 5) {
    out.rows = out.rows.slice(0, Math.floor(out.rows.length / 2))
    out.truncated = true
    body = JSON.stringify(out)
  }
  return { body, response: res }
}

/** Answers one forwarded query; the returned string is the tool result.
 *  `model` is which lane asked ('' outside compare). */
export type BridgeHandler = (cypher: string, model: string) => Promise<string>

/**
 * The conversation's registration with the manager. Holds the socket open,
 * answers `{id, cypher}` frames with `{id, body}`, reconnects while alive -
 * the manager's timeout copes with the gap; a permanently closed tab is the
 * documented "session disconnected" failure, not something to mask.
 */
export class GraphBridge {
  private ws: WebSocket | null = null
  private closed = false
  private retry: number | undefined

  constructor(
    private readonly conversationId: string,
    private readonly handler: BridgeHandler,
  ) {}

  connect(): void {
    if (this.closed) return
    const base = import.meta.env.VITE_API_WS as string | undefined
    const url = base
      ? `${base}/api/graph/bridge`
      : `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/graph/bridge`
    const sock = new WebSocket(`${url}?conversation=${encodeURIComponent(this.conversationId)}`)
    this.ws = sock
    sock.onmessage = (ev) => {
      let frame: { id: number; cypher: string; model?: string }
      try {
        frame = JSON.parse(ev.data as string) as { id: number; cypher: string; model?: string }
      } catch {
        return
      }
      void this.handler(frame.cypher, frame.model ?? '')
        .catch((e) => `Query failed: ${e instanceof Error ? e.message : String(e)}`)
        .then((body) => {
          if (sock.readyState === WebSocket.OPEN) {
            sock.send(JSON.stringify({ id: frame.id, body }))
          }
        })
    }
    sock.onclose = () => {
      if (this.ws === sock) this.ws = null
      if (!this.closed) {
        this.retry = window.setTimeout(() => this.connect(), 3_000)
      }
    }
  }

  close(): void {
    this.closed = true
    clearTimeout(this.retry)
    this.ws?.close()
    this.ws = null
  }
}
