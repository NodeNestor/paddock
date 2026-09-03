<script setup lang="ts">
// The reusable system-prompt library - its own panel/route (activity-bar entry),
// not a Settings card. This is where presets are browsed, searched, created,
// edited and deleted; the per-chat SystemPromptPanel only LOADS from here.
import { computed, onMounted, ref } from 'vue'
import type { SavedPrompt } from '@/lib/api'
import { usePromptsStore } from '@/stores/prompts'
import Dialog from '@/components/ui/Dialog.vue'
import Icon from '@/components/Icon.vue'

const prompts = usePromptsStore()

onMounted(() => void prompts.refresh())

const query = ref('')
const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) return prompts.prompts
  return prompts.prompts.filter(
    (p) => p.name.toLowerCase().includes(q) || p.body.toLowerCase().includes(q),
  )
})

// ── editor dialog (create + edit share it) ──────────────────────────────────
const editId = ref<string | undefined>(undefined)
const editName = ref('')
const editBody = ref('')
const editorOpen = ref(false)
const editorTitle = computed(() => (editId.value ? 'Edit preset' : 'New preset'))
const canSaveEdit = computed(() => !!editName.value.trim() && !!editBody.value.trim())

function create(): void {
  editId.value = undefined
  editName.value = ''
  editBody.value = ''
  editorOpen.value = true
}
function edit(p: SavedPrompt): void {
  editId.value = p.id
  editName.value = p.name
  editBody.value = p.body
  editorOpen.value = true
}
async function saveEdit(): Promise<void> {
  if (!canSaveEdit.value) return
  await prompts.save(editName.value, editBody.value, editId.value)
  editorOpen.value = false
}

// ── delete confirm ──────────────────────────────────────────────────────────
const pendingDelete = ref<SavedPrompt | null>(null)
async function confirmDelete(): Promise<void> {
  const p = pendingDelete.value
  pendingDelete.value = null
  if (p) await prompts.remove(p.id)
}
</script>

<template>
  <div class="prompts">
    <header class="prompts__head">
      <div>
        <h1 class="prompts__title">Prompts</h1>
        <p class="prompts__sub">
          Reusable system prompts you can apply to any chat from the prompt icon in the composer.
        </p>
      </div>
      <button v-if="prompts.prompts.length" class="pk-btn pk-btn--primary" @click="create">
        <Icon name="plus" :size="15" /> New preset
      </button>
    </header>

    <div v-if="prompts.prompts.length > 4" class="prompts__search">
      <Icon name="search" :size="15" class="prompts__search-icon" />
      <input v-model="query" class="pk-input prompts__search-input" placeholder="Search presets" />
    </div>

    <div v-if="prompts.prompts.length === 0" class="prompts__empty">
      <span class="prompts__empty-mark"><Icon name="file-text" :size="26" /></span>
      <p>No presets yet.</p>
      <button class="pk-btn pk-btn--primary" @click="create">
        <Icon name="plus" :size="15" /> Create your first preset
      </button>
    </div>
    <div v-else-if="filtered.length === 0" class="prompts__none">
      No presets match "{{ query.trim() }}".
    </div>

    <ul v-else class="prompts__list">
      <li v-for="p in filtered" :key="p.id" class="pcard">
        <button class="pcard__main" type="button" @click="edit(p)">
          <span class="pcard__name">{{ p.name }}</span>
          <span class="pcard__body">{{ p.body }}</span>
        </button>
        <div class="pcard__acts">
          <button class="pk-icon-btn" aria-label="Edit preset" @click="edit(p)">
            <Icon name="edit" :size="16" />
          </button>
          <button class="pk-icon-btn pcard__del" aria-label="Delete preset" @click="pendingDelete = p">
            <Icon name="trash" :size="16" />
          </button>
        </div>
      </li>
    </ul>
  </div>

  <!-- create / edit -->
  <Dialog :open="editorOpen" :title="editorTitle" icon="file-text" size="lg" @close="editorOpen = false">
    <div class="pedit">
      <label class="pedit__field">
        <span class="pedit__label">Name</span>
        <input v-model="editName" class="pk-input" placeholder="e.g. Terse coder" />
      </label>
      <label class="pedit__field">
        <span class="pedit__label">Prompt</span>
        <textarea
          v-model="editBody"
          class="pk-input pedit__ta"
          rows="10"
          placeholder="Instructions the model should follow..."
        />
      </label>
    </div>
    <template #footer>
      <button class="pk-btn pk-btn--ghost" @click="editorOpen = false">Cancel</button>
      <button class="pk-btn pk-btn--primary" :disabled="!canSaveEdit" @click="saveEdit">Save</button>
    </template>
  </Dialog>

  <!-- delete confirm -->
  <Dialog
    :open="!!pendingDelete"
    role="alertdialog"
    danger
    icon="alert-triangle"
    title="Delete preset?"
    size="sm"
    @close="pendingDelete = null"
  >
    <p class="pedit__confirm">
      <strong>{{ pendingDelete?.name }}</strong> will be removed from your library. Chats already
      using it keep their copy.
    </p>
    <template #footer>
      <button class="pk-btn pk-btn--ghost" @click="pendingDelete = null">Cancel</button>
      <button class="pk-btn pk-btn--danger" @click="confirmDelete">Delete</button>
    </template>
  </Dialog>
</template>

<style scoped>
.prompts {
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
}
.prompts__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
}
.prompts__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
}
.prompts__sub {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
  margin-top: 4px;
  max-width: 46ch;
}
.prompts__head .pk-btn {
  flex-shrink: 0;
}

.prompts__search {
  position: relative;
  margin-bottom: 12px;
}
.prompts__search-icon {
  position: absolute;
  left: 11px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--pk-text-muted);
  pointer-events: none;
}
.prompts__search-input {
  width: 100%;
  padding-left: 34px;
}

.prompts__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  text-align: center;
  padding: 56px 0;
  color: var(--pk-text-muted);
}
.prompts__empty-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 56px;
  height: 56px;
  border-radius: var(--pk-radius-xl);
  background: var(--pk-bg-elevated);
  color: var(--pk-text-secondary);
}
.prompts__none {
  padding: 24px 2px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
}

.prompts__list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.pcard {
  display: flex;
  align-items: center;
  gap: 8px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  transition: border-color 0.12s ease, background 0.12s ease;
}
.pcard:hover {
  border-color: var(--pk-border-strong);
}
.pcard__main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 12px 14px;
  border: 0;
  background: transparent;
  text-align: left;
  cursor: pointer;
}
.pcard__name {
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.pcard__body {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  line-height: 1.45;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  white-space: normal;
}
.pcard__acts {
  display: flex;
  gap: 2px;
  padding-right: 10px;
  opacity: 0;
  transition: opacity 0.12s ease;
}
.pcard:hover .pcard__acts,
.pcard:focus-within .pcard__acts {
  opacity: 1;
}
.pcard__del:hover {
  color: var(--pk-status-error);
}

.pedit {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.pedit__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.pedit__label {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-secondary);
}
.pedit__ta {
  width: 100%;
  height: auto;
  resize: vertical;
  min-height: 200px;
  padding: 10px 12px;
  font-family: inherit;
  line-height: 1.5;
}
.pedit__confirm {
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
  line-height: 1.55;
}
.pedit__confirm strong {
  color: var(--pk-text-primary);
  font-weight: 600;
}
</style>
