// Artifacts the model wrote for the current chat. The bodies live
// in the manager's SQLite, not in the conversation - that is the point of the
// feature, so this store holds only the LIST and hands out a fetcher; each
// visible pane owns the body it is showing.
//
// Shape: artifacts group by the model that wrote them, and a compare turn is
// exactly "two named groups". That grouping is the whole side-by-side feature -
// two panes, each with its own tabs, rather than one pane plus a special-cased
// "the other one" .
import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import { useChatStore } from '@/stores/chat'
import { activeMessages } from '@/lib/tree'

export interface ArtifactMeta {
  id: string
  kind: string
  title: string
  language: string
  createdAt: number
  updatedAt: number
  /** How many versions exist; 1 means it has never been edited. */
  versions: number
  /** Which model wrote it ('' for artifacts made before this was recorded). */
  model: string
}

export interface ArtifactVersion {
  seq: number
  /** How this version came about: create | update | rewrite. */
  op: string
  createdAt: number
  bytes: number
}

export interface ArtifactGroup {
  model: string
  items: ArtifactMeta[]
}

export const useArtifactsStore = defineStore('artifacts', () => {
  /** Artifacts of the conversation currently on screen, newest first. */
  const list = ref<ArtifactMeta[]>([])
  const conversationId = ref('')
  /** How many panes the panel has room for; it measures itself and says. */
  const paneCapacity = ref(1)
  /** Which artifact each model's pane is showing, keyed by model. */
  const picked = ref<Record<string, string>>({})
  /** Which artifact the single pane is showing, when there is only one. */
  const soleId = ref('')

  const any = computed(() => list.value.length > 0)

  /** By writer, in first-seen order. */
  const groups = computed<ArtifactGroup[]>(() => {
    const out: ArtifactGroup[] = []
    for (const a of list.value) {
      const model = a.model || ''
      const g = out.find((x) => x.model === model)
      if (g) g.items.push(a)
      else out.push({ model, items: [a] })
    }
    return out
  })

  /** The order the chat shows its compare lanes in, left to right: the order
   *  its assistant messages first name each model. */
  const laneOrder = computed<string[]>(() => {
    const seen: string[] = []
    const conv = useChatStore().active
    for (const m of conv ? activeMessages(conv) : []) {
      const id = m.model ?? m.run?.model
      if (m.role === 'assistant' && id && !seen.includes(id)) seen.push(id)
    }
    return seen
  })

  /** Only a NAMED group can own a pane. Artifacts written before the model was
   *  recorded ('') belong to nobody; they stay reachable through the single
   *  pane's tab strip, which lists everything.
   *
   *  Pane order FOLLOWS the lane order, not artifact recency. The list arrives
   *  newest-first, so ordering by it put whichever model happened to answer
   *  last on the left - the left lane's artifact has to be the left pane or
   * the comparison is a lie. Anything the conversation
   *  does not name (a deleted lane) sorts after what it does. */
  const panes = computed(() => {
    const order = laneOrder.value
    const rank = (model: string): number => {
      const i = order.indexOf(model)
      return i === -1 ? Number.MAX_SAFE_INTEGER : i
    }
    return groups.value.filter((g) => g.model).sort((a, b) => rank(a.model) - rank(b.model))
  })
  /** Split all the way or not at all: two lanes get two panes, four get four,
   *  and a panel with room for only three of four falls back to the single
   *  tabbed pane. A partial split would quietly hide a model's work, which is
   *  worse than making you widen the panel. */
  const split = computed(
    () => panes.value.length >= 2 && paneCapacity.value >= panes.value.length,
  )
  /** The panes actually on screen - every named group, or none. */
  const visible = computed<ArtifactGroup[]>(() => (split.value ? panes.value : []))
  /** What the single pane lists when not split: everything, legacy included. */
  const soleItems = computed(() => (split.value ? [] : list.value))

  /** Which artifact a given pane is showing. */
  const selectedIn = (model: string): string => picked.value[model] ?? ''

  // Keep every slot pointing at something that is actually in its pane. Runs
  // on any list change and on the narrow<->split flip, the only moments a slot
  // can go stale.
  watch(
    [visible, soleItems],
    ([panesNow, sole]) => {
      for (const g of panesNow) {
        if (!g.items.some((a) => a.id === picked.value[g.model])) {
          picked.value[g.model] = g.items[0]?.id ?? ''
        }
      }
      if (!sole.some((a) => a.id === soleId.value)) soleId.value = sole[0]?.id ?? ''
    },
    { immediate: true, deep: false },
  )

  /** Bring one artifact up - from a tab, or from the Show link on a tool card.
   *  It lands in whichever pane owns its writer, so clicking Show on one
   *  model's card during a compare never hijacks another model's pane. */
  function show(id: string): void {
    const a = list.value.find((x) => x.id === id)
    if (!a) return
    if (split.value && visible.value.some((g) => g.model === a.model)) picked.value[a.model] = id
    else soleId.value = id
  }

  async function refresh(convId: string): Promise<void> {
    conversationId.value = convId
    if (!convId) {
      list.value = []
      return
    }
    try {
      const res = await fetch(`/api/conversations/${encodeURIComponent(convId)}/artifacts`)
      if (!res.ok) return
      const data = (await res.json()) as { artifacts: ArtifactMeta[] }
      list.value = data.artifacts ?? []
    } catch (e) {
      console.error('failed to list artifacts', e)
    }
  }

  /** Hand edits in progress, keyed by artifact id. They live in the store, not
   *  in the pane, so switching tabs or remounting the panel never throws typed
   *  work away - which is why there is no "discard changes?" dialog anywhere:
   *  nothing is ever discarded without being asked for. */
  const drafts = ref<Record<string, string>>({})

  function setDraft(id: string, text: string, saved: string): void {
    if (text === saved) delete drafts.value[id]
    else drafts.value[id] = text
  }
  function clearDraft(id: string): void {
    delete drafts.value[id]
  }

  /** Write an edited body back as a new version. It lands under its own `edit`
   *  op, and the model sees it on its next read - the latest version is the
   *  one every artifact tool works against. */
  async function saveEdit(id: string, text: string): Promise<string> {
    try {
      const res = await fetch(`/api/artifacts/${encodeURIComponent(id)}/content`, {
        method: 'PUT',
        headers: { 'content-type': 'text/plain; charset=utf-8' },
        body: text,
      })
      if (!res.ok) return `save failed (${res.status})`
      clearDraft(id)
      await refresh(conversationId.value)
      return ''
    } catch (e) {
      return e instanceof Error ? e.message : 'save failed'
    }
  }

  /** One version's body plus that artifact's version list. `seq` 0 = latest. */
  async function fetchOne(
    id: string,
    seq = 0,
  ): Promise<{ body: string; versions: ArtifactVersion[] }> {
    const q = seq > 0 ? `?version=${seq}` : ''
    const [content, meta] = await Promise.all([
      fetch(`/api/artifacts/${encodeURIComponent(id)}/content${q}`),
      fetch(`/api/artifacts/${encodeURIComponent(id)}`),
    ])
    const versions =
      meta.ok ? (((await meta.json()) as { versions?: ArtifactVersion[] }).versions ?? []) : []
    return { body: content.ok ? await content.text() : '', versions }
  }

  // ── graph import auto-repair loop ──────────────────────────────────────
  // The model's artifact tools return success when the TEXT is stored; the
  // import happens browser-side, so three separate live runs
  // had the model announce a finished graph over a failed import. This is
  // the loop-closer: a failed seed queues one report here, ChatView sends it
  // into the conversation when the turn is over, and the model repairs with
  // artifact_update. Two repair attempts per artifact, then the panel's
  // error display is the (human) fallback - a model that failed twice is
  // looping, not converging.
  const graphImportFailure = ref<{ artifactId: string; version: number; summary: string } | null>(
    null,
  )
  /** Auto-reports allowed per CONVERSATION, total. Per-artifact budgets were
   *  a runaway: a model that answers every repair by CREATING a new artifact
   *  mints itself a fresh budget each time (seen live - the loop only
   *  ended when the user hit Stop), and the in-memory count reset on reload.
   *  The book now lives on the conversation, persisted. */
  const REPAIR_CAP = 3

  function repairBook(): { total: number; seen: Record<string, number> } | null {
    const conv = useChatStore().active
    if (!conv || conv.id !== conversationId.value) return null
    if (!conv.autoRepairs) conv.autoRepairs = { total: 0, seen: {} }
    return conv.autoRepairs
  }

  function reportGraphImport(
    artifactId: string,
    version: number,
    r: { errors: { statement: string; error: string }[]; executed: number },
  ): void {
    if (!r.errors.length) {
      // a clean import supersedes any queued failure for this artifact
      if (graphImportFailure.value?.artifactId === artifactId) graphImportFailure.value = null
      return
    }
    const book = repairBook()
    if (!book) return
    if ((book.seen[artifactId] ?? 0) >= version) return // this version was reported
    if (book.total >= REPAIR_CAP) return // budget spent: the panel is the fallback
    const first = r.errors[0]
    const excerpt = first.statement.length > 220 ? first.statement.slice(0, 217) + '...' : first.statement
    graphImportFailure.value = {
      artifactId,
      version,
      summary:
        `${r.errors.length} of ${r.errors.length + r.executed} statements failed. ` +
        `First error: ${first.error}\nFailing statement (excerpt):\n${excerpt}`,
    }
  }

  /** Hand the queued report to the sender once, writing it into the
   *  conversation's persisted book. */
  function consumeGraphImportFailure(): { artifactId: string; summary: string } | null {
    const f = graphImportFailure.value
    if (!f) return null
    graphImportFailure.value = null
    const book = repairBook()
    if (!book) return null
    if (book.total >= REPAIR_CAP) return null
    book.seen[f.artifactId] = Math.max(book.seen[f.artifactId] ?? 0, f.version)
    book.total += 1
    const chatStore = useChatStore()
    if (chatStore.active) chatStore.persist(chatStore.active)
    return f
  }

  return {
    list,
    conversationId,
    graphImportFailure,
    reportGraphImport,
    consumeGraphImportFailure,
    paneCapacity,
    picked,
    soleId,
    any,
    groups,
    panes,
    split,
    visible,
    soleItems,
    selectedIn,
    show,
    refresh,
    fetchOne,
    drafts,
    setDraft,
    clearDraft,
    saveEdit,
  }
})
