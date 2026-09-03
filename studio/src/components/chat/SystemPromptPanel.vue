<script setup lang="ts">
// Per-chat system prompt - a fast, focused setting, not a library manager.
// Prompt-centric: the textarea is the truth. Empty = no system prompt (off);
// text that equals a saved preset shows "Using ..."; anything else is "Custom".
// Presets are just loaded in; managing the library (rename/edit/delete/browse)
// lives in the Prompts panel. "Deselect" is simply Clear.
//
// The preset picker is an INLINE reveal (not a reka menu): a dropdown menu
// nested inside a modal Dialog is unreliable - the two modal layers fight and
// the menu never opens. Inline conditional content has no such problem.
import { computed, nextTick, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import type { Conversation } from '@/types/chat'
import type { SavedPrompt } from '@/lib/api'
import { useChatStore } from '@/stores/chat'
import { usePromptsStore } from '@/stores/prompts'
import { useInjectedPrompt } from '@/composables/useInjectedPrompt'
import Collapsible from '@/components/ui/Collapsible.vue'
import Dialog from '@/components/ui/Dialog.vue'
import Icon from '@/components/Icon.vue'

const props = defineProps<{ conv: Conversation; open: boolean }>()
const emit = defineEmits<{ close: [] }>()

const router = useRouter()
const chat = useChatStore()
const prompts = usePromptsStore()

const bodyText = computed(() => props.conv.systemPrompt ?? '')
const hasBody = computed(() => !!bodyText.value.trim())
// The saved preset whose body equals this chat's prompt (if any) - informational.
const activePreset = computed(() => prompts.prompts.find((p) => p.body === bodyText.value) ?? null)
// What the RUNNER adds on top of the box above: each armed MCP server's own
// `instructions`. Shown read-only because an empty textarea otherwise reads as
// "nothing is being said on my behalf", which stopped being true the moment we
// started honoring server instructions.
const injected = useInjectedPrompt(computed(() => props.conv))

const status = computed(() => {
  if (!hasBody.value) return 'No system prompt'
  if (activePreset.value) return `Using "${activePreset.value.name}"`
  return 'Custom prompt'
})

// ── inline preset picker ────────────────────────────────────────────────────
const picking = ref(false)
const filter = ref('')
const filtered = computed(() => {
  const q = filter.value.trim().toLowerCase()
  if (!q) return prompts.prompts
  return prompts.prompts.filter((p) => p.name.toLowerCase().includes(q))
})

// ── save-as (inline in the footer) ──────────────────────────────────────────
const saving = ref(false)
const saveName = ref('')
const nameInput = ref<HTMLInputElement | null>(null)
const canSave = computed(() => hasBody.value && !!saveName.value.trim())

watch(
  () => props.open,
  async (isOpen) => {
    if (!isOpen) return
    picking.value = false
    saving.value = false
    saveName.value = ''
    filter.value = ''
    await prompts.refresh()
  },
  { immediate: true },
)

function persist(): void {
  chat.persist(props.conv)
}
function onEdit(e: Event): void {
  props.conv.systemPrompt = (e.target as HTMLTextAreaElement).value
  persist()
}
// Turn the system prompt off for this chat (the obvious deselect).
function clearPrompt(): void {
  props.conv.systemPrompt = ''
  persist()
}
function togglePicker(): void {
  picking.value = !picking.value
  filter.value = ''
}
// Load a preset's text into this chat (editable afterwards; applies next turn).
function usePreset(p: SavedPrompt): void {
  props.conv.systemPrompt = p.body
  persist()
  picking.value = false
}
function startSave(): void {
  saving.value = true
  picking.value = false
  saveName.value = activePreset.value?.name ?? ''
  void nextTick(() => {
    nameInput.value?.focus()
    nameInput.value?.select()
  })
}
async function confirmSave(): Promise<void> {
  if (!canSave.value) return
  // update a same-named preset, else create a new one.
  const id = prompts.prompts.find(
    (p) => p.name.trim().toLowerCase() === saveName.value.trim().toLowerCase(),
  )?.id
  await prompts.save(saveName.value, bodyText.value, id)
  saving.value = false
}
function manageLibrary(): void {
  emit('close')
  void router.push({ name: 'prompts' })
}
</script>

<template>
  <Dialog :open="open" title="System prompt" icon="sliders" @close="emit('close')">
    <div class="sp">
      <textarea
        class="pk-input sp__ta"
        :value="conv.systemPrompt"
        rows="7"
        placeholder="Instructions the model follows in this chat..."
        @input="onEdit"
      />

      <section v-if="injected.blocks.value.length" class="sp__inject">
        <h3 class="sp__injecth">Also sent, from the tools this chat has on</h3>
        <Collapsible v-for="b in injected.blocks.value" :key="b.label" :summary="b.label">
          <p class="sp__blockt">{{ b.text }}</p>
        </Collapsible>
      </section>

      <div class="sp__status">
        <span
          class="sp__badge"
          :class="{
            'sp__badge--off': !hasBody,
            'sp__badge--preset': hasBody && activePreset,
            'sp__badge--custom': hasBody && !activePreset,
          }"
        >
          {{ status }}
        </span>
        <button v-if="hasBody" class="sp__clear" type="button" @click="clearPrompt">Clear</button>
      </div>

      <div class="sp__preset">
        <button
          class="sp__pickbtn"
          :class="{ 'sp__pickbtn--open': picking }"
          type="button"
          @click="togglePicker"
        >
          <Icon name="file-text" :size="15" />
          <span>Use a saved preset</span>
          <Icon name="chevron-down" :size="15" class="sp__chev" />
        </button>

        <div v-if="picking" class="sp__panel">
          <input
            v-if="prompts.prompts.length > 7"
            v-model="filter"
            class="pk-input pk-input--sm sp__filter"
            placeholder="Filter presets"
          />
          <div class="sp__list">
            <button
              v-for="p in filtered"
              :key="p.id"
              type="button"
              class="sp__opt"
              :class="{ 'sp__opt--on': activePreset?.id === p.id }"
              @click="usePreset(p)"
            >
              <Icon name="check" :size="14" :style="{ opacity: activePreset?.id === p.id ? 1 : 0 }" />
              <span class="sp__opt-name">{{ p.name }}</span>
            </button>
            <div v-if="!filtered.length" class="sp__list-empty">
              {{ prompts.prompts.length ? 'No matches' : 'No saved presets yet' }}
            </div>
          </div>
          <button class="sp__manage" type="button" @click="manageLibrary">
            <Icon name="file-text" :size="13" /> Manage presets in Prompts ->
          </button>
        </div>
      </div>
    </div>

    <template #footer>
      <template v-if="!saving">
        <!-- Everything here applies live (typing + picking a preset persist on
             the spot), so the footer needs a positive "finish" affordance -
             the corner X reads as cancel/discard. "Done" just dismisses; saving
             to the library is the secondary action. -->
        <button class="pk-btn pk-btn--ghost" type="button" :disabled="!hasBody" @click="startSave">
          Save as preset
        </button>
        <button class="pk-btn pk-btn--primary" type="button" @click="emit('close')">
          Done
        </button>
      </template>
      <template v-else>
        <input
          ref="nameInput"
          v-model="saveName"
          class="pk-input sp__savename"
          placeholder="Preset name"
          @keydown.enter="confirmSave"
          @keydown.esc="saving = false"
        />
        <button class="pk-btn pk-btn--ghost" type="button" @click="saving = false">Cancel</button>
        <button class="pk-btn pk-btn--primary" type="button" :disabled="!canSave" @click="confirmSave">
          Save
        </button>
      </template>
    </template>
  </Dialog>
</template>

<style scoped>
.sp {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.sp__ta {
  width: 100%;
  height: auto;
  resize: vertical;
  min-height: 150px;
  padding: 10px 12px;
  font-family: inherit;
  line-height: 1.5;
}
.sp__inject {
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-base);
  padding: 10px 12px;
}
.sp__injecth {
  margin: 0 0 8px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
  font-weight: 500;
}
.sp__blockt {
  margin: 0;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-xs);
  line-height: 1.5;
  white-space: pre-wrap;
}
.sp__status {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.sp__badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 9px;
  border-radius: var(--pk-radius-full);
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
}
.sp__badge--off {
  background: var(--pk-bg-inset);
  color: var(--pk-text-muted);
}
.sp__badge--preset {
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text);
}
.sp__badge--custom {
  background: var(--pk-bg-inset);
  color: var(--pk-text-secondary);
}
.sp__clear {
  border: 0;
  background: transparent;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
  cursor: pointer;
  padding: 4px 6px;
  border-radius: var(--pk-radius-md);
}
.sp__clear:hover {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}

/* inline preset picker */
.sp__pickbtn {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  height: 34px;
  padding: 0 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
  color: var(--pk-text-primary);
  font-size: var(--pk-font-size-sm);
  cursor: pointer;
  transition: border-color 0.12s ease, background 0.12s ease;
}
.sp__pickbtn:hover {
  border-color: var(--pk-border-strong);
}
.sp__pickbtn .sp__chev {
  margin-left: auto;
  color: var(--pk-text-muted);
  transition: transform 0.15s ease;
}
.sp__pickbtn--open .sp__chev {
  transform: rotate(180deg);
}
.sp__panel {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 6px;
  padding: 6px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
}
.sp__filter {
  width: 100%;
}
.sp__list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 210px;
  overflow-y: auto;
}
.sp__opt {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 7px 8px;
  border: 0;
  border-radius: var(--pk-radius-md);
  background: transparent;
  color: var(--pk-text-primary);
  font: inherit;
  font-size: var(--pk-font-size-sm);
  text-align: left;
  cursor: pointer;
  transition: background 0.12s ease;
}
.sp__opt:hover {
  background: var(--pk-bg-hover);
}
.sp__opt--on {
  color: var(--pk-accent-text);
}
.sp__opt-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.sp__list-empty {
  padding: 10px 8px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.sp__manage {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  align-self: flex-start;
  border: 0;
  background: transparent;
  color: var(--pk-accent-text);
  font-size: var(--pk-font-size-xs);
  font-weight: 500;
  cursor: pointer;
  padding: 4px 6px;
  border-radius: var(--pk-radius-md);
}
.sp__manage:hover {
  background: var(--pk-bg-hover);
}
.sp__savename {
  flex: 1;
}
</style>
