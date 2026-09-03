import { defineStore } from 'pinia'
import { uuid } from '@/lib/uuid'
import { computed, ref } from 'vue'
import type { AudioPart, Conversation, Message, SamplingParams } from '@/types/chat'
import { DEFAULT_PARAMS, messageText } from '@/types/chat'
import { activeMessages, deleteSubtree, migrate, stepSibling, tipId } from '@/lib/tree'
import { taskLabel } from '@/lib/tasks'
import { store, stubFromSummary } from '@/lib/api'

function uid(): string {
  return uuid()
}

export const useChatStore = defineStore('chat', () => {
  // Summaries hydrate the list; full messages load lazily on open.
  const conversations = ref<Conversation[]>([])
  const activeId = ref<string | null>(null)
  const loaded = ref(false)
  const loadedIds = ref<Set<string>>(new Set())

  // The start page's conversation-in-waiting. It is deliberately not in
  // `conversations` and not on the server: the sidebar lists committed chats
  // only, so `/` shows no row and nothing selected. It exists so the composer
  // has something real to write to - its model/system-prompt/thinking toggle
  // all land here - and gets committed by the first send. Abandon the start
  // page and it's simply dropped.
  const draft = ref<Conversation | null>(null)

  /** The conversation in view: the selected one, else the start-page draft. */
  const active = computed(
    () => conversations.value.find((c) => c.id === activeId.value) ?? draft.value,
  )

  function isDraft(c: Conversation): boolean {
    return draft.value?.id === c.id
  }

  // Both the shell (for the nav's chat count) and ChatView hydrate; share the
  // in-flight fetch so concurrent callers don't both list + both set activeId.
  let hydrating: Promise<void> | null = null

  async function hydrate(): Promise<void> {
    if (loaded.value) return
    if (hydrating) return hydrating
    hydrating = (async () => {
      try {
        const summaries = await store.listConversations()
        conversations.value = summaries.map(stubFromSummary)
      } catch (e) {
        console.error('failed to load conversations', e)
        conversations.value = []
      }
      loaded.value = true
      activeId.value = lastOpenId()
      if (activeId.value) await ensureLoaded(activeId.value)
    })()
    try {
      await hydrating
    } finally {
      hydrating = null
    }
  }

  /** The chat to resume when no id is given: the last one opened, else the
   *  newest, else null when there are none.
   *
   *  Reads localStorage rather than `activeId` deliberately - the start page
   *  clears activeId (nothing is selected there), so activeId is null exactly
   *  when someone arrives from `/`. */
  function lastOpenId(): string | null {
    const last = localStorage.getItem('pk_last_conversation')
    if (last && conversations.value.some((c) => c.id === last)) return last
    return conversations.value[0]?.id ?? null
  }

  /** Fetch a conversation's full document (messages) the first time it opens. */
  async function ensureLoaded(id: string): Promise<void> {
    if (loadedIds.value.has(id)) return
    try {
      const full = await store.getConversation(id)
      // Pre-popover chats carry the old hard-coded sampling the user never
      // chose (exactly 0.7/0.8/20). Those become "untouched" so nothing is
      // sent - the same values the user did set survive any other combination.
      const p = full.params
      if (p && p.temperature === 0.7 && p.topP === 0.8 && p.topK === 20) {
        p.temperature = null
        p.topP = null
        p.topK = null
      }
      // The penalty fields predate the popover (no UI ever wrote them, always
      // 0): 0 becomes "untouched" instead of claiming an explicit choice.
      if (p) {
        if (p.frequencyPenalty === 0) p.frequencyPenalty = null
        if (p.presencePenalty === 0) p.presencePenalty = null
      }
      // Give the thread its tree links. Every conversation written before
      // branching is a straight line, so this is exact rather than a guess -
      // and it is deliberately not saved back here: the inference is
      // deterministic, so re-deriving it on each open costs nothing and beats
      // rewriting every stored conversation the first time someone opens it.
      // The links persist the moment the chat is next written for a real
      // reason.
      migrate(full)
      const i = conversations.value.findIndex((c) => c.id === id)
      if (i >= 0) conversations.value.splice(i, 1, full)
      loadedIds.value.add(id)
    } catch (e) {
      console.error('failed to load conversation', id, e)
    }
  }

  function makeConversation(model: string, params?: Partial<SamplingParams>): Conversation {
    const now = Date.now()
    return {
      id: uid(),
      title: 'New chat',
      messages: [],
      model,
      systemPrompt: '',
      params: { ...DEFAULT_PARAMS, ...params },
      createdAt: now,
      updatedAt: now,
    }
  }

  /** Put a conversation into the list, select it, and save it - the moment it
   *  becomes real (shows in the sidebar, exists on the server). */
  function commit(c: Conversation): Conversation {
    draft.value = null
    conversations.value.unshift(c)
    activeId.value = c.id
    loadedIds.value.add(c.id)
    localStorage.setItem('pk_last_conversation', c.id)
    void save(c)
    return c
  }

  function newConversation(model: string, params?: Partial<SamplingParams>): Conversation {
    return commit(makeConversation(model, params))
  }

  /** Open the start page: an uncommitted conversation with nothing selected. */
  function startDraft(model: string): Conversation {
    draft.value = makeConversation(model)
    activeId.value = null // no sidebar row is current - there's no chat yet
    return draft.value
  }

  /** First send from the start page: the draft becomes a real conversation,
   *  keeping whatever the composer already put on it (model, system prompt,
   *  thinking). */
  function commitDraft(model: string): Conversation {
    return commit(draft.value ?? makeConversation(model))
  }

  function select(id: string): void {
    draft.value = null // leaving the start page discards its unsent draft
    activeId.value = id
    localStorage.setItem('pk_last_conversation', id)
    void ensureLoaded(id)
  }

  /** Bump updatedAt, re-sort to top, persist (debounced). */
  function touch(c: Conversation): void {
    c.updatedAt = Date.now()
    const i = conversations.value.findIndex((x) => x.id === c.id)
    if (i > 0) {
      conversations.value.splice(i, 1)
      conversations.value.unshift(c)
    }
    persist(c)
  }

  const timers = new Map<string, number>()
  function persist(c: Conversation): void {
    // An uncommitted draft stays in memory. The composer's toggles persist on
    // every change, which would otherwise PUT the start page's chat and leave
    // an orphan on the server that the sidebar never shows.
    if (isDraft(c)) return
    const prev = timers.get(c.id)
    if (prev) clearTimeout(prev)
    timers.set(
      c.id,
      window.setTimeout(() => {
        timers.delete(c.id)
        void save(c)
      }, 400),
    )
  }

  function persistNow(c: Conversation): void {
    if (isDraft(c)) return // see persist(): drafts are in-memory until sent
    const prev = timers.get(c.id)
    if (prev) {
      clearTimeout(prev)
      timers.delete(c.id)
    }
    void save(c)
  }

  async function save(c: Conversation): Promise<void> {
    try {
      await store.putConversation(c)
    } catch (e) {
      console.error('failed to save conversation', c.id, e)
    }
  }

  // Rename is metadata, not activity: don't bump updatedAt or re-sort (renaming
  // used to yank the chat to the top of the list - jarring). Persist in place.
  function rename(id: string, title: string): void {
    const c = conversations.value.find((x) => x.id === id)
    if (!c) return
    c.title = title.trim() || 'Untitled'
    persistNow(c)
  }

  /** Re-point activeId after the current chat is gone; keeps URL/localStorage in
   *  sync. Returns the new active id (or null when the list is now empty). */
  function repointActive(): string | null {
    activeId.value = conversations.value[0]?.id ?? null
    if (activeId.value) {
      localStorage.setItem('pk_last_conversation', activeId.value)
      void ensureLoaded(activeId.value)
    } else {
      localStorage.removeItem('pk_last_conversation')
    }
    return activeId.value
  }

  async function remove(id: string): Promise<void> {
    const i = conversations.value.findIndex((x) => x.id === id)
    if (i < 0) return
    conversations.value.splice(i, 1)
    loadedIds.value.delete(id)
    try {
      await store.deleteConversation(id)
    } catch (e) {
      console.error('failed to delete conversation', id, e)
    }
    if (activeId.value === id) repointActive()
  }

  /** Delete several chats at once (multi-select). Re-points the active chat once
   *  if it was among them. */
  async function removeMany(ids: string[]): Promise<void> {
    const doomed = new Set(ids)
    if (doomed.size === 0) return
    const hadActive = !!activeId.value && doomed.has(activeId.value)
    conversations.value = conversations.value.filter((c) => !doomed.has(c.id))
    for (const id of doomed) {
      loadedIds.value.delete(id)
      try {
        await store.deleteConversation(id)
      } catch (e) {
        console.error('failed to delete conversation', id, e)
      }
    }
    if (hadActive) repointActive()
  }

  // Pinning is metadata too - persist without a reorder/updatedAt bump; the
  // sidebar floats pinned chats to the top via its sort.
  function togglePin(id: string): void {
    const c = conversations.value.find((x) => x.id === id)
    if (!c) return
    c.pinned = !c.pinned
    persistNow(c)
  }

  /** Set the title from the first user message, once, if still default.
   *
   *  An AUDIO turn has no words of its own, so the title comes from what was
   *  SAID - which only exists once the transcript lands, hence this runs
   *  again when a transcription finishes. Until then it declines rather than
   *  locking in a placeholder. The file name is the fallback, and it is only
   *  a fallback: a microphone recording has no name at all, which is exactly
   *  why the transcript leads. */
  function maybeTitle(c: Conversation): void {
    if (c.title !== 'New chat') return
    // The branch on screen: if the opening question was edited before a
    // title existed, the title should come from the question being asked, not
    // from the one that was replaced.
    const path = activeMessages(c)
    const first = path.find((m) => m.role === 'user')
    if (!first) return
    let raw = messageText(first).replace(/\s+/g, ' ').trim()
    if (!raw) {
      const clip = first.content.find((p): p is AudioPart => p.type === 'audio')
      if (!clip) return
      // The answers to this turn: one per compare lane. Wait for them all to
      // settle, then take the first lane that actually produced words - a
      // lane that failed must not decide the title while a sibling succeeded.
      const answers: Message[] = []
      for (let i = path.indexOf(first) + 1; i < path.length; i++) {
        if (path[i].role !== 'assistant') break
        answers.push(path[i])
      }
      if (!answers.length || answers.some((m) => m.streaming)) return
      raw =
        answers.map((m) => messageText(m).replace(/\s+/g, ' ').trim()).find(Boolean) ?? clip.name
      if (!raw) return
    }
    // A chat opened with a task action is titled by the action's plain name -
    // the sidebar must not be the one place a markup token leaks into view.
    const t = (taskLabel(raw) ?? raw).slice(0, 48)
    if (t) {
      c.title = t
      persistNow(c)
    }
  }

  /** Add a turn to the tree; returns the REACTIVE element the array now holds
   *  (so a streaming caller mutates the proxy, not the raw object).
   *
   *  `parentId` is what makes a turn an ALTERNATIVE rather than a
   *  continuation: omit it and the turn extends the branch on screen; pass the
   *  parent of an existing turn and the new one becomes its sibling, which is
   *  all a regenerate or an edited question actually is. Pass `null` for a
   *  second root. */
  function addMessage(c: Conversation, m: Message, parentId?: string | null): Message {
    m.parentId = parentId !== undefined ? parentId : (tipId(c) ?? null)
    c.messages.push(m)
    const added = c.messages[c.messages.length - 1]
    // The cursor follows what was just written - except inside a compare
    // fan-out, where the lanes are one step and the anchor (the first lane to
    // land) is what later turns hang from. Without this the cursor would walk
    // to whichever lane was created last and the group's own children would
    // hang off the wrong message.
    if (m.group) {
      const anchor = c.messages.find((x) => x.group === m.group)
      c.leafId = anchor ? anchor.id : added.id
    } else {
      c.leafId = added.id
    }
    touch(c)
    return added
  }

  /** Show the sibling `delta` steps away from the one holding `id` - the
   *  `< 2/3 >` control under a branched turn. Returns false when there is
   *  nowhere to go, so the caller can disable rather than silently no-op. */
  function selectSibling(id: string, delta: number): boolean {
    const c = active.value
    if (!c) return false
    const moved = stepSibling(c, id, delta)
    if (moved) persist(c)
    return moved
  }

  /** Drop a turn and everything that followed from it. The whole step goes
   *  (half a compare block is not a thing), and the cursor lands on a
   *  surviving branch rather than nowhere. */
  function removeMessage(id: string): void {
    const c = active.value
    if (!c) return
    if (!deleteSubtree(c, id).length) return
    touch(c)
  }

  /** Put a document on screen: select it AND open the pane. The two ways in -
   *  a file chip in the thread and a tab in the pane itself - both land here,
   *  so "selection = target" (lib/docrun.ts) holds however you arrived. */
  function showDocument(sourceId: string): void {
    const c = active.value
    if (!c) return
    c.activeDocId = sourceId
    c.docPaneOpen = true
    persist(c)
  }

  /** Fold the document pane away, or bring it back. Writes the flag even when
   *  it matches what was implied, which is the point: from here on the
   *  conversation has an opinion and the model-derived default stops applying. */
  function setDocPane(open: boolean): void {
    const c = active.value
    if (!c) return
    c.docPaneOpen = open
    persist(c)
  }

  function setArtifactsPane(open: boolean): void {
    const c = active.value
    if (!c) return
    c.artifactsPaneOpen = open
    persist(c)
  }

  return {
    conversations,
    activeId,
    active,
    isDraft,
    loaded,
    showDocument,
    setDocPane,
    setArtifactsPane,
    hydrate,
    lastOpenId,
    ensureLoaded,
    newConversation,
    startDraft,
    commitDraft,
    select,
    touch,
    persist,
    persistNow,
    rename,
    remove,
    removeMany,
    togglePin,
    maybeTitle,
    addMessage,
    selectSibling,
    removeMessage,
  }
})
