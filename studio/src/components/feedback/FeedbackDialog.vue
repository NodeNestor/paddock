<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import Dialog from '@/components/ui/Dialog.vue'
import Switch from '@/components/ui/Switch.vue'
import ToggleGroup from '@/components/ui/ToggleGroup.vue'
import ToggleGroupItem from '@/components/ui/ToggleGroupItem.vue'
import Icon from '@/components/Icon.vue'
import { feedbackApi, type FeedbackCategory, type FeedbackContext } from '@/lib/api'
import { useModelsStore } from '@/stores/models'
import { useToastsStore } from '@/stores/toasts'

const props = defineProps<{ open: boolean }>()
const emit = defineEmits<{ (e: 'close'): void }>()

const models = useModelsStore()
const toasts = useToastsStore()

// The upstream cap. Mirrored from the manager (which mirrors the API) so the
// counter can go red before somebody loses a long report to a round trip.
const MAX = 10_000

const category = ref<FeedbackCategory>('bug')
const message = ref('')
const email = ref('')
const attach = ref(false)
const sending = ref(false)
const ctx = ref<FeedbackContext | null>(null)
const ctxError = ref('')

const PLACEHOLDER: Record<FeedbackCategory, string> = {
  bug: 'What happened, and what did you expect instead?',
  feature: 'What would you like paddock to do?',
  feedback: "What's working, and what could be better?",
}

const remaining = computed(() => MAX - message.value.length)
const canSend = computed(
  () => message.value.trim().length > 0 && remaining.value >= 0 && !sending.value,
)

// Fetch the diagnostics only when the switch goes on. Nothing is gathered - let
// alone sent - for somebody who never asks for it, and the panel shows the real
// answer rather than a promise about one.
watch(attach, async (on) => {
  if (!on || ctx.value) return
  ctxError.value = ''
  try {
    ctx.value = await feedbackApi.context()
  } catch (e) {
    ctxError.value = e instanceof Error ? e.message : 'could not read this machine'
    attach.value = false
  }
})

// A closed dialog is a discarded draft. Reset on the way in rather than out, so
// the fields do not visibly blank while the close animation is still running.
watch(
  () => props.open,
  (open) => {
    if (!open) return
    category.value = 'bug'
    message.value = ''
    email.value = ''
    attach.value = false
    ctx.value = null
    ctxError.value = ''
  },
)

/** The runner line: name, then whatever we actually know about how it runs. */
function runnerLine(r: NonNullable<FeedbackContext['runners']>[number]): string {
  const bits = [r.artifact, r.kv_cache_dtype ? `KV ${r.kv_cache_dtype}` : null]
    .filter(Boolean)
    .join(' · ')
  return bits ? `${r.model} - ${bits}` : r.model
}

async function send(): Promise<void> {
  if (!canSend.value) return
  sending.value = true
  try {
    await feedbackApi.submit({
      category: category.value,
      message: message.value.trim(),
      ...(email.value.trim() ? { email: email.value.trim() } : {}),
      include_context: attach.value,
    })
    toasts.push({
      tone: 'good',
      title: 'Feedback sent',
      description: 'Thank you - it reached the paddock team.',
    })
    emit('close')
  } catch (e) {
    // The server's own sentence, not ours: a rate limit says when to retry and
    // a transport failure says it did not send, and both are worth more than a
    // generic apology.
    toasts.push({
      tone: 'bad',
      title: 'Could not send',
      description: e instanceof Error ? e.message : 'unknown error',
      duration: 8000,
    })
  } finally {
    sending.value = false
  }
}
</script>

<template>
  <Dialog
    :open="open"
    title="Send feedback"
    icon="message-square"
    size="lg"
    :busy="sending"
    @close="emit('close')"
  >
    <div class="fb">
      <ToggleGroup v-model="category" class="fb__cats" label="Kind of feedback">
        <ToggleGroupItem value="bug" class="fb__cat">Something is broken</ToggleGroupItem>
        <ToggleGroupItem value="feature" class="fb__cat">I want a feature</ToggleGroupItem>
        <ToggleGroupItem value="feedback" class="fb__cat">General thoughts</ToggleGroupItem>
      </ToggleGroup>

      <label class="fb__field">
        <span class="fb__label">Message</span>
        <textarea
          v-model="message"
          class="pk-input fb__ta"
          rows="8"
          :placeholder="PLACEHOLDER[category]"
        />
        <span v-if="remaining < 500" class="fb__count" :class="{ 'fb__count--over': remaining < 0 }">
          {{ remaining < 0 ? `${-remaining} over the limit` : `${remaining} characters left` }}
        </span>
      </label>

      <label class="fb__field">
        <span class="fb__label">Email</span>
        <input
          v-model="email"
          class="pk-input"
          type="email"
          autocomplete="email"
          placeholder="Only if you want a reply"
        />
      </label>

      <section class="fb__card">
        <div class="fb__row">
          <div class="fb__rowtext">
            <span class="fb__rowhead">Include what this machine is running</span>
            <span class="fb__rowsub">
              Your graphics card, driver and the models you have started. No file paths, no
              keys, no chats.
            </span>
          </div>
          <Switch v-model="attach" label="Include what this machine is running" />
        </div>

        <p v-if="ctxError" class="fb__err">
          <Icon name="alert-triangle" :size="14" />
          {{ ctxError }}
        </p>

        <div v-else-if="attach && ctx" class="fb__ctx">
          <dl class="fb__facts">
            <dt>Paddock</dt>
            <dd>{{ ctx.manager.build }} · {{ ctx.manager.os }}-{{ ctx.manager.arch }}</dd>
            <dt>Graphics</dt>
            <dd>
              {{ ctx.gpu.card ?? 'no card found' }}
              <template v-if="ctx.gpu.driver">· driver {{ ctx.gpu.driver }}</template>
              <template v-if="ctx.gpu.cuda">· CUDA {{ ctx.gpu.cuda }}</template>
            </dd>
            <dt>Models</dt>
            <dd v-if="!ctx.runners.length">none running</dd>
            <dd v-else>
              <span v-for="r in ctx.runners" :key="r.model" class="fb__runner">
                {{ runnerLine(r) }}
              </span>
            </dd>
          </dl>
        </div>
      </section>
    </div>

    <template #footer>
      <span v-if="models.serverVersion" class="fb__ver">v{{ models.serverVersion }}</span>
      <button class="pk-btn pk-btn--ghost" :disabled="sending" @click="emit('close')">
        Cancel
      </button>
      <button class="pk-btn pk-btn--primary" :disabled="!canSend" @click="send">
        {{ sending ? 'Sending...' : 'Send' }}
      </button>
    </template>
  </Dialog>
</template>

<style scoped>
.fb {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.fb__cats {
  display: flex;
  gap: 6px;
}
/* :deep because Reka clones the item through roving-focus, keeping the class
   but dropping the scope attribute - a plain `.fb__cat {}` matches nothing and
   the button falls back to native chrome. Same shape as the header's area
   toggle; `npm run build` fails on this, which is how it was caught. */
.fb__cats :deep(.fb__cat) {
  flex: 1;
  padding: 7px 10px;
  font: inherit;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-secondary);
  background: var(--pk-bg-surface);
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  cursor: pointer;
  transition: color 0.12s ease, background 0.12s ease, border-color 0.12s ease;
}
.fb__cats :deep(.fb__cat:hover) {
  color: var(--pk-text-primary);
  border-color: var(--pk-border-strong);
}
.fb__cats :deep(.fb__cat[data-state='on']) {
  color: var(--pk-accent);
  background: var(--pk-accent-subtle);
  border-color: var(--pk-accent);
}

.fb__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.fb__label {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-secondary);
}
.fb__ta {
  width: 100%;
  height: auto;
  min-height: 150px;
  resize: vertical;
  padding: 10px 12px;
  font-family: inherit;
  line-height: 1.5;
}
.fb__count {
  align-self: flex-end;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.fb__count--over {
  color: var(--pk-status-error);
}

.fb__card {
  background: var(--pk-bg-surface);
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.fb__row {
  display: flex;
  align-items: flex-start;
  gap: 14px;
}
.fb__rowtext {
  display: flex;
  flex-direction: column;
  gap: 3px;
  flex: 1;
  min-width: 0;
}
.fb__rowhead {
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.fb__rowsub {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  line-height: 1.5;
}

.fb__ctx {
  background: var(--pk-bg-base);
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  padding: 10px 12px;
}
.fb__facts {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 4px 14px;
  margin: 0;
  font-size: var(--pk-font-size-xs);
  line-height: 1.6;
}
.fb__facts dt {
  color: var(--pk-text-muted);
  font-weight: 600;
}
.fb__facts dd {
  margin: 0;
  color: var(--pk-text-secondary);
  font-family: var(--pk-font-mono);
  overflow-wrap: anywhere;
}
.fb__runner {
  display: block;
}

.fb__err {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-status-error);
}

.fb__ver {
  margin-right: auto;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
</style>
