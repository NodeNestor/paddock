<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useChatStore } from '@/stores/chat'
import { conversationBusy } from '@/composables/useChatStream'
import { useModelsStore } from '@/stores/models'
import { isDocParserConv } from '@/lib/docrun'
import { modelCapability } from '@/lib/model-name'
import type { Conversation } from '@/types/chat'
import { NEW_CHAT, SEARCH_CHATS, TOGGLE_CHATS } from '@/lib/shortcuts'
import Icon from '@/components/Icon.vue'
import Menu from '@/components/ui/Menu.vue'
import MenuTrigger from '@/components/ui/MenuTrigger.vue'
import MenuContent from '@/components/ui/MenuContent.vue'
import MenuItem from '@/components/ui/MenuItem.vue'
import MenuSeparator from '@/components/ui/MenuSeparator.vue'
import Checkbox from '@/components/ui/Checkbox.vue'
import Dialog from '@/components/ui/Dialog.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

const chat = useChatStore()
const models = useModelsStore()
const router = useRouter()

// ── what KIND of conversation each row is ───────────────────────────────────
// The list holds three quite different things and used to show them
// identically. Order matters: EVIDENCE beats capability,
// because plenty of chat models can also hear or read, and what the
// conversation actually did is the honest answer.
//
// The evidence lives in the MESSAGES, and the sidebar lists stubs, so the
// server does the reading now (store.rs conversation_kind, arriving on the
// summary as `kind`) and everything below it is fallback for a row that
// predates the column or a manager that does not send one.
type ConvKind = 'chat' | 'transcription' | 'document'
const KIND: Record<ConvKind, { icon: string; label: string }> = {
  chat: { icon: 'message-square', label: 'Chat' },
  transcription: { icon: 'microphone', label: 'Transcription' },
  document: { icon: 'file-text', label: 'Document' },
}
function convKind(c: Conversation): ConvKind {
  // 1. This conversation's own turns, once they are here. Both arms read
  //    `c.messages`, which a loaded conversation has and a stub does not, so
  //    on a stub they answer false and fall through instead of deciding. That
  //    silent fall-through was the bug: every unopened row read as a chat and
  //    corrected itself on click.
  if (isDocParserConv(c, models.caps)) return 'document'
  if (c.messages.some((m) => m.content.some((p) => p.type === 'audio'))) return 'transcription'
  // 2. What the SERVER decided when it last saved this conversation - the same
  //    evidence, read where the messages actually are. This is the arm that
  //    stops a row changing its mind when you click it.
  if (c.kind === 'document' || c.kind === 'transcription') return c.kind
  // 3. The CATALOG, for a row saved before the column existed. Transcription
  //    and alignment models are catalogued with no `chat` at all, so they say
  //    what they are on their own.
  //
  //    Deliberately no `documents` arm: that capability means "takes images,
  //    for structured extraction" - estimate.rs asserts a vision tower claims
  //    exactly one of vision|documents - so granite-vision wears it while
  //    chatting perfectly well. Reading it as "this is a document
  //    conversation" mislabels every ordinary granite-vision chat (measured:
  //    13 of them, not one with a document run).
  const cap = modelCapability(c.model)
  if (cap.length && !cap.includes('chat')) return 'transcription'
  // 4. Off-catalog (a hand-started GGUF): a model that cannot hold a text
  //    conversation can only be transcribing. `canChat` and not
  //    `canTranscribe` deliberately - the latter is true of every generative ASR
  //    model, which are ordinary chat models that happen to take audio.
  if (!models.canChat(c.model)) return 'transcription'
  return 'chat'
}
const emit = defineEmits<{ newChat: []; fold: [] }>()

/** Open a conversation by navigating - the route drives chat.select. */
function open(id: string): void {
  void router.push({ name: 'chat', params: { id } })
}

// ── sort ────────────────────────────────────────────────────────────────────
type SortMode = 'recent' | 'oldest'
const SORT_LABEL: Record<SortMode, string> = {
  recent: 'Newest first',
  oldest: 'Oldest first',
}
const sortMode = ref<SortMode>(
  (localStorage.getItem('pk_chat_sort') as SortMode) || 'recent',
)
// Controlled open state - reka's uncontrolled dropdown path doesn't reliably
// toggle in our wrappers (same reason the ⋯ menu is controlled).
const sortOpen = ref(false)
function setSort(m: SortMode): void {
  sortMode.value = m
  localStorage.setItem('pk_chat_sort', m)
  sortOpen.value = false
}

const query = ref('')
const searchInput = ref<HTMLInputElement | null>(null)
/** Focus + select, so the chord lands you ready to REPLACE an old query
 *  rather than appending to one you had forgotten was there. Exposed because
 *  the key handler lives in ChatView, which owns whether this panel is even
 *  mounted. */
function focusSearch(): void {
  searchInput.value?.focus()
  searchInput.value?.select()
}
defineExpose({ focusSearch })

// Pinned chats float to the top; within each group, sort by last activity in
// the chosen direction. Sorting lives here (not the store) so renaming/pinning
// never yanks a row around - position is a pure function of pinned + updatedAt.
const sorted = computed<Conversation[]>(() => {
  const dir = sortMode.value === 'recent' ? -1 : 1
  return [...chat.conversations].sort((a, b) => {
    if (!!a.pinned !== !!b.pinned) return a.pinned ? -1 : 1
    return (a.updatedAt - b.updatedAt) * dir
  })
})

// Plain case-insensitive substring match on the title - predictable (no fuzzy
// surprises) and keeps the chosen sort order.
const results = computed<Conversation[]>(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) return sorted.value
  return sorted.value.filter((c) => c.title.toLowerCase().includes(q))
})

// ── multi-select ──────────────────────────────────────────────────────────
const selectMode = ref(false)
const selected = ref<Set<string>>(new Set())
function toggleSelectMode(): void {
  selectMode.value = !selectMode.value
  if (!selectMode.value) selected.value = new Set()
}
function toggleSelected(id: string): void {
  const next = new Set(selected.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  selected.value = next
}
function onRowClick(c: Conversation): void {
  if (selectMode.value) toggleSelected(c.id)
  else open(c.id)
}
const allShownSelected = computed(
  () => results.value.length > 0 && results.value.every((c) => selected.value.has(c.id)),
)
function toggleSelectAll(): void {
  selected.value = allShownSelected.value
    ? new Set()
    : new Set(results.value.map((c) => c.id))
}
// "some but not all" is a real third state and Reka's checkbox spells it, so
// select-all stops lying about a half-picked list.
const selectAllState = computed<boolean | 'indeterminate'>(() => {
  if (allShownSelected.value) return true
  return results.value.some((c) => selected.value.has(c.id)) ? 'indeterminate' : false
})

// ── rename (inline) ─────────────────────────────────────────────────────────
const menuOpenId = ref<string | null>(null)
function setMenu(id: string, open: boolean): void {
  menuOpenId.value = open ? id : null
}
const renamingId = ref<string | null>(null)
const renameText = ref('')
const renameInput = ref<HTMLInputElement | HTMLInputElement[] | null>(null)

function startRename(id: string, title: string): void {
  menuOpenId.value = null
  renamingId.value = id
  renameText.value = title
  // The ref lives inside v-for (an array), and reka returns focus to the trigger
  // when the menu closes - focus on the next frame so we win, then select.
  requestAnimationFrame(() => {
    const r = renameInput.value
    const el = Array.isArray(r) ? r[0] : r
    el?.focus()
    el?.select()
  })
}
function commitRename(): void {
  if (renamingId.value) chat.rename(renamingId.value, renameText.value)
  renamingId.value = null
}

// ── delete (single + bulk), always behind a confirm ─────────────────────────
const pendingDelete = ref<{ id: string; title: string } | null>(null)
const pendingBulk = ref(false)

async function confirmDelete(): Promise<void> {
  const del = pendingDelete.value
  pendingDelete.value = null
  if (!del) return
  const wasActive = del.id === chat.activeId
  await chat.remove(del.id)
  if (wasActive) await followActive()
}

async function confirmBulkDelete(): Promise<void> {
  pendingBulk.value = false
  const ids = [...selected.value]
  if (!ids.length) return
  const hadActive = !!chat.activeId && selected.value.has(chat.activeId)
  await chat.removeMany(ids)
  selected.value = new Set()
  selectMode.value = false
  if (hadActive) await followActive()
}

// remove()/removeMany() re-point activeId; follow it in the URL.
async function followActive(): Promise<void> {
  await router.push(
    chat.activeId ? { name: 'chat', params: { id: chat.activeId } } : { name: 'chat' },
  )
}

// ── relative-time label (updated on each render; good enough for a list) ─────
const MIN = 60_000
const HOUR = 60 * MIN
const DAY = 24 * HOUR
function formatWhen(ts: number): string {
  const now = Date.now()
  const diff = now - ts
  if (diff < MIN) return 'now'
  if (diff < HOUR) return `${Math.floor(diff / MIN)}m`
  if (diff < DAY) return `${Math.floor(diff / HOUR)}h`
  if (diff < 7 * DAY) return `${Math.floor(diff / DAY)}d`
  const d = new Date(ts)
  const sameYear = d.getFullYear() === new Date(now).getFullYear()
  return d.toLocaleDateString(
    undefined,
    sameYear
      ? { month: 'short', day: 'numeric' }
      : { month: 'short', day: 'numeric', year: 'numeric' },
  )
}
function fullWhen(ts: number): string {
  return new Date(ts).toLocaleString()
}
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar__top">
      <Tooltip :label="`Hide chats (${TOGGLE_CHATS})`">
        <button
          class="pk-icon-btn sidebar__fold"
          type="button"
          aria-label="Hide chats"
          @click="emit('fold')"
        >
          <Icon name="chevron-left" :size="16" />
        </button>
      </Tooltip>
      <button class="pk-btn pk-btn--primary sidebar__new" @click="emit('newChat')">
        <Icon name="plus" :size="16" />
        <span class="sidebar__new-label">New chat</span>
        <kbd class="sidebar__kbd">{{ NEW_CHAT }}</kbd>
      </button>
    </div>

    <div class="sidebar__search">
      <Icon name="search" :size="14" class="sidebar__search-icon" />
      <input
        ref="searchInput"
        v-model="query"
        class="pk-input sidebar__search-input"
        placeholder="Search chats"
      />
      <!-- the chord that GETS you here, so it goes once you have arrived -->
      <kbd v-if="!query" class="sidebar__kbd sidebar__kbd--in">{{ SEARCH_CHATS }}</kbd>
    </div>

    <!-- sort + multi-select controls -->
    <div class="sidebar__tools">
      <Menu :open="sortOpen" @update:open="(v: boolean) => (sortOpen = v)">
        <MenuTrigger>
          <button class="sidebar__sort" type="button">
            <Icon name="arrow-up-down" :size="13" />
            <span>{{ SORT_LABEL[sortMode] }}</span>
            <Icon name="chevron-down" :size="13" />
          </button>
        </MenuTrigger>
        <MenuContent align="start" side="bottom" min-width="180px">
          <MenuItem @select="setSort('recent')">
            <Icon name="check" :size="14" :style="{ opacity: sortMode === 'recent' ? 1 : 0 }" />
            <span>Newest first</span>
          </MenuItem>
          <MenuItem @select="setSort('oldest')">
            <Icon name="check" :size="14" :style="{ opacity: sortMode === 'oldest' ? 1 : 0 }" />
            <span>Oldest first</span>
          </MenuItem>
        </MenuContent>
      </Menu>

      <button
        class="sidebar__tool"
        :class="{ 'sidebar__tool--on': selectMode }"
        type="button"
        @click="toggleSelectMode"
      >
        <Icon :name="selectMode ? 'x' : 'check-square'" :size="14" />
        {{ selectMode ? 'Cancel' : 'Select' }}
      </button>
    </div>

    <div v-if="selectMode" class="sidebar__selbar">
      <Checkbox
        :model-value="selectAllState"
        class="sidebar__selall"
        :size="15"
        @update:model-value="toggleSelectAll"
      >
        {{ allShownSelected ? 'Clear all' : 'Select all' }}
      </Checkbox>
      <span class="sidebar__selcount">{{ selected.size }} selected</span>
    </div>

    <div class="sidebar__list">
      <div v-if="results.length === 0" class="sidebar__empty">
        {{ query.trim() ? 'No matches' : 'No chats yet' }}
      </div>
      <div
        v-for="c in results"
        :key="c.id"
        class="conv"
        :class="{
          'conv--active': !selectMode && c.id === chat.activeId,
          'conv--selected': selectMode && selected.has(c.id),
        }"
        @click="onRowClick(c)"
      >
        <Checkbox
          v-if="selectMode"
          class="conv__check"
          :model-value="selected.has(c.id)"
          @click.stop
          @update:model-value="toggleSelected(c.id)"
        />

        <input
          v-if="renamingId === c.id"
          ref="renameInput"
          v-model="renameText"
          class="pk-input pk-input--sm conv__rename"
          @keydown.enter="commitRename"
          @keydown.esc="renamingId = null"
          @blur="commitRename"
          @click.stop
        />
        <template v-else>
          <Tooltip :label="KIND[convKind(c)].label">
            <Icon :name="KIND[convKind(c)].icon" :size="13" class="conv__kind" />
          </Tooltip>
          <Icon v-if="c.pinned" name="pin" :size="12" class="conv__pin" />
          <Tooltip v-if="conversationBusy(c.id)" label="Still answering">
            <span class="conv__live" />
          </Tooltip>
          <span class="conv__title">{{ c.title }}</span>
          <span class="conv__right" @click.stop>
            <Tooltip :label="fullWhen(c.updatedAt)">
              <span class="conv__when">{{ formatWhen(c.updatedAt) }}</span>
            </Tooltip>
            <span v-if="!selectMode" class="conv__actions">
              <Menu :open="menuOpenId === c.id" @update:open="(v: boolean) => setMenu(c.id, v)">
                <MenuTrigger>
                  <button class="pk-icon-btn conv__act" aria-label="Chat actions">
                    <Icon name="more-horizontal" :size="15" />
                  </button>
                </MenuTrigger>
                <MenuContent :label="`Actions for ${c.title}`">
                  <MenuItem @select="startRename(c.id, c.title)">
                    <Icon name="edit" :size="14" /> Rename
                  </MenuItem>
                  <MenuItem @select="chat.togglePin(c.id)">
                    <Icon name="pin" :size="14" /> {{ c.pinned ? 'Unpin' : 'Pin' }}
                  </MenuItem>
                  <MenuSeparator />
                  <MenuItem danger @select="pendingDelete = { id: c.id, title: c.title }">
                    <Icon name="trash" :size="14" /> Delete
                  </MenuItem>
                </MenuContent>
              </Menu>
            </span>
          </span>
        </template>
      </div>
    </div>

    <!-- bulk action bar -->
    <div v-if="selectMode" class="sidebar__bulk">
      <button
        class="pk-btn pk-btn--danger pk-btn--sm sidebar__bulk-del"
        :disabled="selected.size === 0"
        @click="pendingBulk = true"
      >
        <Icon name="trash" :size="14" /> Delete {{ selected.size || '' }}
      </button>
    </div>
  </aside>

  <Dialog
    :open="!!pendingDelete"
    role="alertdialog"
    danger
    icon="alert-triangle"
    title="Delete chat?"
    size="sm"
    @close="pendingDelete = null"
  >
    <p class="sidebar__confirm">
      <strong>{{ pendingDelete?.title }}</strong> and its messages will be permanently
      removed. This can't be undone.
    </p>
    <template #footer>
      <button class="pk-btn pk-btn--ghost" @click="pendingDelete = null">Cancel</button>
      <button class="pk-btn pk-btn--danger" @click="confirmDelete">Delete</button>
    </template>
  </Dialog>

  <Dialog
    :open="pendingBulk"
    role="alertdialog"
    danger
    icon="alert-triangle"
    title="Delete chats?"
    size="sm"
    @close="pendingBulk = false"
  >
    <p class="sidebar__confirm">
      <strong>{{ selected.size }}</strong> chat{{ selected.size === 1 ? '' : 's' }} and all their
      messages will be permanently removed. This can't be undone.
    </p>
    <template #footer>
      <button class="pk-btn pk-btn--ghost" @click="pendingBulk = false">Cancel</button>
      <button class="pk-btn pk-btn--danger" @click="confirmBulkDelete">Delete {{ selected.size }}</button>
    </template>
  </Dialog>
</template>

<style scoped>
.sidebar {
  width: var(--pk-sidebar-width);
  flex-shrink: 0;
  height: 100%;
  background: var(--pk-bg-surface);
  /* no border-right: the ResizeHandle draws the divider (Traverse pattern). */
  display: flex;
  flex-direction: column;
  padding: 12px;
  gap: 10px;
  /* so the New chat chord can drop out on a narrow panel - the sidebar is
     resizable, and a media query would ask about the WINDOW, not this column */
  container-type: inline-size;
}
/* Fold LEFTMOST, then New chat. Same shape as the document pane's own fold in
   lector's toolbar - leftmost, and the same chevron - because they are the
   same action on two panels, and a reader should not have to learn it twice.
   Folding used to be reachable only by dragging the
   divider past its collapse point, a gesture you have to already know about. */
.sidebar__top {
  display: flex;
  align-items: center;
  gap: 6px;
}
/* Kept as its own row rather than folded in beside the search and the fold
   control. New chat is the primary action AND now carries a visible shortcut,
   neither of which survives being reduced to a bare `+`; and a merged row puts
   an icon toggle, a text field and an icon button in 260px, leaving the search
   about 150 for a placeholder and a leading icon. The saving would be ~44px in
   a column that scrolls anyway - about a chat and a half (we asked which). */
.sidebar__new {
  flex: 1;
  min-width: 0;
  /* content left, chord right - pk-btn centres by default */
  justify-content: flex-start;
}
.sidebar__new-label {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
/* The chord, on the button rather than only in a tooltip: a shortcut you have
   to hover to discover is one most people never learn. */
.sidebar__kbd {
  flex: none;
  margin-left: auto;
  padding: 1px 5px;
  border-radius: var(--pk-radius-sm);
  /* a lighter/darker step of whatever the accent is, so it reads on the
     primary fill in both themes without a hard-coded colour */
  background: color-mix(in srgb, var(--pk-text-inverse) 20%, transparent);
  font-family: inherit;
  font-size: 10px;
  font-weight: 600;
  line-height: 1.6;
}
/* Below this the label and the chord fight over the row; the tooltip still
   carries the chord, so nothing is lost. */
@container (max-width: 215px) {
  .sidebar__kbd {
    display: none;
  }
  .sidebar__search-input {
    padding-right: 10px;
  }
}
/* square, and exactly as tall as the button it stands beside */
.sidebar__fold {
  flex: none;
  width: 34px;
  height: 34px;
}
.sidebar__search {
  position: relative;
}
.sidebar__search-icon {
  position: absolute;
  left: 9px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--pk-text-muted);
  pointer-events: none;
}
.sidebar__search-input {
  width: 100%;
  padding-left: 30px;
  /* room for the chord chip, so a long query never runs under it */
  padding-right: 52px;
}
/* Inside a field the chip means "press this to get here", so it steps aside
   the moment you are here - on focus, or as soon as there is a query. */
.sidebar__kbd--in {
  position: absolute;
  right: 7px;
  top: 50%;
  transform: translateY(-50%);
  margin: 0;
  pointer-events: none;
  background: var(--pk-bg-hover);
  color: var(--pk-text-muted);
}
.sidebar__search:focus-within .sidebar__kbd--in {
  display: none;
}

/* sort + select row */
.sidebar__tools {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
/* the sort button's LABEL gives way first on a narrow panel: the button may
   shrink (min-width:0 beats the flex-item auto floor, same trap as the row
   titles) and the text inside ellipsizes while both icons hold their size */
.sidebar__sort {
  min-width: 0;
  flex: 0 1 auto;
}
.sidebar__sort > span {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.sidebar__sort > svg {
  flex: none;
}
.sidebar__sort,
.sidebar__tool {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 26px;
  padding: 0 8px;
  border: 0;
  border-radius: var(--pk-radius-md);
  background: transparent;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-xs);
  font-weight: 500;
  cursor: pointer;
  transition: background 0.12s ease, color 0.12s ease;
}
.sidebar__sort:hover,
.sidebar__tool:hover,
.sidebar__sort[data-state='open'] {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
.sidebar__tool--on {
  color: var(--pk-accent-text);
  background: var(--pk-accent-subtle);
}

.sidebar__selbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 0 2px;
}
/* :deep(): Reka's CheckboxRoot is rendered through a clone that drops our
   scope attribute, so an unqualified rule would never reach it.
   Colour is Checkbox's own - it tracks checked/indeterminate, which this used
   to paint flat regardless of state. */
.sidebar__selbar :deep(.sidebar__selall) {
  font-size: var(--pk-font-size-xs);
  font-weight: 500;
  padding: 2px 0;
}
.sidebar__selcount {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}

.sidebar__list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  /* Break out of the aside's padding so the scrollbar sits at the edge. */
  margin: 0 -12px;
  padding: 0 12px;
}
.sidebar__empty {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  text-align: center;
  padding: 20px 0;
}
.conv {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 7px 9px;
  border-radius: var(--pk-radius-md);
  cursor: pointer;
  color: var(--pk-text-secondary);
  transition: background 0.12s ease, color 0.12s ease;
}
.conv:hover {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
.conv--active {
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text);
}
.conv--selected {
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text);
}
.conv :deep(.conv__check) {
  flex-shrink: 0;
}
/* what this conversation is - chat, transcription, document. Muted so the
   title still leads; the row's colour carries active/selected. */
.conv__kind {
  flex-shrink: 0;
  color: var(--pk-text-muted);
}
.conv--active .conv__kind,
.conv--selected .conv__kind {
  color: inherit;
}
.conv__pin {
  flex-shrink: 0;
  color: var(--pk-text-muted);
}
/* A chat you are not looking at is still answering. The row is the only place
   that can say so continuously - the completion toast is a moment, this is the
   state. */
.conv__live {
  flex: none;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--pk-accent);
  animation: conv-live 1.4s ease-in-out infinite;
}
@keyframes conv-live {
  50% {
    opacity: 0.25;
  }
}
.conv__title {
  flex: 1;
  /* a flex item's implicit min-width:auto refuses to shrink below the text,
     so the ellipsis never engaged and narrow panels overflowed - min-width:0
     is what lets overflow/ellipsis actually apply */
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: var(--pk-font-size-sm);
}
/* time by default; swap to the ⋯ button on hover (they share the slot) */
.conv__right {
  position: relative;
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: flex-end;
  min-width: 34px;
  height: 24px;
}
.conv__when {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  white-space: nowrap;
}
.conv__actions {
  position: absolute;
  right: 0;
  display: flex;
  opacity: 0;
}
.conv:hover .conv__when {
  opacity: 0;
}
/* show ⋯ on hover, and keep it shown while its menu is open (pointer may leave) */
.conv:hover .conv__actions,
.conv__actions:has([data-state='open']) {
  opacity: 1;
}
.conv:has(.conv__actions [data-state='open']) .conv__when {
  opacity: 0;
}
.conv__act {
  width: 24px;
  height: 24px;
}
.conv__rename {
  flex: 1;
  width: 100%;
}

.sidebar__bulk {
  display: flex;
  padding-top: 2px;
}
.sidebar__bulk-del {
  width: 100%;
  justify-content: center;
  gap: 6px;
}

.sidebar__confirm {
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
  line-height: 1.55;
}
.sidebar__confirm strong {
  color: var(--pk-text-primary);
  font-weight: 600;
}
</style>
