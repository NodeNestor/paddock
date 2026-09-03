// The conversation's attached graph (
// phase 2 - the read case).
//
// One live session at a time - the session belongs to the conversation on
// screen, and a 100 MB graph resident in a worker is not something to keep
// per background chat. Switching conversations releases and re-ensures; the
// OPFS cache (keyed by attachment id) makes the re-load cheap.
//
// The manager's copy of the tvdb is the source of truth. The session is
// read-only FOR the MODEL (the bridge handler classifies via EXPLAIN before
// executing); the user's own queries in the pane run unrestricted, because
// mutations live in the worker and die with it.
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { attachmentsApi } from '@/lib/api'
import { useChatStore } from '@/stores/chat'
// Type-only here - the implementation modules arrive via dynamic import in
// boot(), because this store is reached from the main chunk (useChatStream)
// and must not pull graphology/sigma/the wasm glue into it.
import type { GraphBridge } from '@/lib/graph/bridge'
import type { GraphSession, QueryResponse } from '@/lib/graph/session'

/** Ceiling on the grounding block - a thousand-label schema must not eat the
 *  system prompt. The model can still discover the rest by querying. */
const GROUNDING_CAP = 1_400

export const useGraphsStore = defineStore('graphs', () => {
  const conversationId = ref('')
  const attachmentId = ref('')
  const name = ref('')
  const status = ref<'idle' | 'loading' | 'ready' | 'error'>('idle')
  /** What loading is doing right now, for the pane ("fetching 96 MB..."). */
  const phase = ref('')
  const error = ref('')
  const counts = ref({ nodes: 0, edges: 0 })
  /** Engine's own in-memory estimate, for the pane's honesty line. */
  const memBytes = ref(0)
  /** Schema summary appended to request instructions while ready. */
  const grounding = ref('')
  /** Pane folded to a rail (same bargain as the document pane: the way back
   *  stays visible). Session-scoped - a fresh attach starts open. */
  const folded = ref(false)
  /** Every query the model has run this session, oldest first - the pane
   *  renders these as switchable chips, newest auto-selected. `enriched` is
   *  the induced-edges variant, cached on first paint. */
  /** `response: null` = restored from the conversation's stored tool calls -
   *  clicking the chip re-runs the query live (deterministic: same tvdb). */
  const modelRuns = ref<
    { cypher: string; model: string; response: QueryResponse | null; enriched?: QueryResponse }[]
  >([])

  let session: GraphSession | null = null
  let bridge: GraphBridge | null = null
  let booting: Promise<void> | null = null

  const active = computed(() => status.value !== 'idle')

  /** The live session, only when it belongs to `conv` and is usable. */
  function sessionFor(conv: string): GraphSession | null {
    return conv && conversationId.value === conv && status.value === 'ready' ? session : null
  }

  function groundingFor(conv: string): string {
    return sessionFor(conv) ? grounding.value : ''
  }

  /**
   * Make `attachment` the conversation's live graph. Idempotent for the same
   * pair; anything else is released first. `bytes` short-circuits the fetch
   * when the caller just uploaded the file and still holds it.
   */
  function ensure(conv: string, attId: string, attName: string, bytes?: Uint8Array): Promise<void> {
    if (conversationId.value === conv && attachmentId.value === attId && status.value !== 'error') {
      return booting ?? Promise.resolve()
    }
    release()
    conversationId.value = conv
    attachmentId.value = attId
    name.value = attName
    status.value = 'loading'
    booting = boot(conv, attId, bytes)
      .catch((e) => {
        // A load that dies here is overwhelmingly the will-it-fit wall - the
        // worker ran out of wasm memory parsing the image. Say what the
        // ceiling means and name the way out; never a bare stack line.
        status.value = 'error'
        error.value =
          `${e instanceof Error ? e.message : String(e)}. ` +
          'If this graph is large, it may exceed what the in-browser engine can hold (~2 GB) - a Traverse server has no such limit.'
      })
      .finally(() => {
        booting = null
      })
    return booting
  }

  async function boot(conv: string, attId: string, bytes?: Uint8Array): Promise<void> {
    const [{ GraphSession }, { GraphBridge, answerModelQuery }] = await Promise.all([
      import('@/lib/graph/session'),
      import('@/lib/graph/bridge'),
    ])
    const s = new GraphSession()
    session = s
    phase.value = 'starting the graph engine'
    await s.open()

    const cacheKey = `paddock-att-${attId}`
    phase.value = 'restoring from cache'
    let loaded = await s.loadCached(cacheKey)
    if (!loaded.ok) {
      let data = bytes
      if (!data) {
        phase.value = 'fetching the graph'
        const r = await fetch(attachmentsApi.url(attId))
        if (!r.ok) throw new Error(`could not fetch the stored graph (${r.status})`)
        data = new Uint8Array(await r.arrayBuffer())
      }
      phase.value = 'loading into memory'
      loaded = await s.seedTvdb(data, cacheKey)
      if (!loaded.ok) throw new Error('the engine could not read this .tvdb')
    }
    counts.value = { nodes: loaded.nodes ?? 0, edges: loaded.edges ?? 0 }
    memBytes.value = await s.estimatedMemory().catch(() => 0)

    phase.value = 'reading the schema'
    const schema = await s.schema()
    const labelProps = schema.labels_detail
      .map((l) => `${l.name}{${l.properties.map((p) => p.name).join(', ')}}`)
      .join('; ')
    // Relationship properties are not optional here: without them the model
    // does not know t.amount exists and explores blind - watched live:
    // "I need to check if there's an amount property on the
    // relationships" followed by 100 rows of guesswork.
    const relProps = schema.relationship_types_detail
      .filter((r) => r.properties.length > 0)
      .map((r) => `${r.name}{${r.properties.map((p) => p.name).join(', ')}}`)
      .join('; ')
    let g =
      `A graph database "${name.value}" is attached to this conversation ` +
      `(${counts.value.nodes.toLocaleString('en')} nodes, ${counts.value.edges.toLocaleString('en')} edges). ` +
      `Node labels: ${schema.labels.join(', ') || 'none'}. ` +
      `Node properties: ${labelProps || 'none'}. ` +
      `Relationship types: ${schema.relationship_types.join(', ') || 'none'}. ` +
      `Relationship properties: ${relProps || 'none'}.`
    if (g.length > GROUNDING_CAP) g = g.slice(0, GROUNDING_CAP - 1) + '...'
    grounding.value = g

    // Restore the model's earlier queries from the conversation's stored
    // tool calls, so returning to a chat keeps its chips - re-run on click,
    // which is exact: the graph is deterministic from the tvdb.
    const chat = useChatStore()
    if (chat.active?.id === conv) {
      const restored: typeof modelRuns.value = []
      for (const m of chat.active.messages) {
        for (const tc of m.toolCalls ?? []) {
          if (tc.serverLabel !== 'graph' || tc.name !== 'graph_query') continue
          try {
            const cypher = (JSON.parse(tc.arguments) as { cypher?: string }).cypher
            if (cypher) restored.push({ cypher, model: '', response: null })
          } catch {
            // unparseable arguments carry nothing to re-run
          }
        }
      }
      if (restored.length > 0) modelRuns.value = restored.slice(-20)
    }

    // Registered last: the model must never reach a session that is still
    // loading. The handler is the read-only gate + compaction in bridge.ts.
    const b = new GraphBridge(conv, async (cypher, model) => {
      const answer = await answerModelQuery(s, cypher)
      if (answer.response) {
        modelRuns.value.push({ cypher, model, response: answer.response })
        // Bounded: a long agentic session must not hold hundreds of result
        // sets. Twenty covers any conversation a human still follows.
        if (modelRuns.value.length > 20) modelRuns.value.shift()
      }
      return answer.body
    })
    bridge = b
    b.connect()
    phase.value = ''
    status.value = 'ready'
  }

  function release(): void {
    bridge?.close()
    bridge = null
    session?.close()
    session = null
    booting = null
    conversationId.value = ''
    attachmentId.value = ''
    name.value = ''
    status.value = 'idle'
    phase.value = ''
    error.value = ''
    counts.value = { nodes: 0, edges: 0 }
    memBytes.value = 0
    grounding.value = ''
    folded.value = false
    modelRuns.value = []
  }

  return {
    conversationId,
    attachmentId,
    name,
    status,
    phase,
    error,
    counts,
    memBytes,
    grounding,
    folded,
    modelRuns,
    active,
    sessionFor,
    groundingFor,
    ensure,
    release,
  }
})
