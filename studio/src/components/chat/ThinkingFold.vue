<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import Icon from '@/components/Icon.vue'
import Markdown from './Markdown.vue'
import { fmtDuration } from '@/lib/format'

const props = defineProps<{
  reasoning: string
  active?: boolean
  ms?: number
  tokens?: number
  tps?: number
}>()

// Live elapsed while thinking, so the wait shows progress instead of a static label.
const elapsed = ref(0)
let timer: number | undefined
watch(
  () => props.active,
  (a) => {
    clearInterval(timer)
    if (a) {
      const t0 = performance.now()
      elapsed.value = 0
      timer = window.setInterval(() => {
        elapsed.value = (performance.now() - t0) / 1000
      }, 250)
    }
  },
  { immediate: true },
)
onBeforeUnmount(() => clearInterval(timer))

const label = computed(() => {
  if (props.active) return elapsed.value >= 1 ? `Thinking... ${Math.floor(elapsed.value)}s` : 'Thinking...'
  if (props.ms != null) return `Thought for ${fmtDuration(props.ms)}`
  return 'Thought for a moment'
})
const meta = computed(() => {
  if (props.active || props.tokens == null) return ''
  const parts = [`${props.tokens} tokens`]
  if (props.tps) parts.push(`${Math.round(props.tps)} tok/s`)
  return parts.join(' · ')
})

// Open only WHILE actively thinking (a live fixed-height window scrolled to
// the newest lines), collapsed once done - the Ollama shape.
// Re-opening a finished thought shows it uncapped at natural
// height. A completed message mounts with active=false, so reloads start
// collapsed too.
const open = ref(false)
let userToggled = false
watch(
  () => props.active,
  (a) => {
    if (!userToggled) open.value = a
  },
  { immediate: true },
)

function toggle(): void {
  userToggled = true
  open.value = !open.value
}

// While live, pin the window to the newest lines - unless the reader
// scrolled back up, in which case leave them alone until they return to
// the bottom (within a small margin; the programmatic pin itself lands at
// distance 0, so it never unpins).
const body = ref<HTMLElement | null>(null)
let pinned = true
function onScroll(): void {
  const el = body.value
  if (!el) return
  pinned = el.scrollHeight - el.scrollTop - el.clientHeight < 24
}
watch(
  () => props.reasoning,
  async () => {
    if (!props.active || !pinned) return
    await nextTick()
    const el = body.value
    if (el) el.scrollTop = el.scrollHeight
  },
)
</script>

<template>
  <div class="think" :class="{ 'think--active': active }">
    <button class="think__head" type="button" @click="toggle">
      <Icon name="brain" :size="14" />
      <span class="think__label">{{ label }}</span>
      <span v-if="meta" class="think__meta">{{ meta }}</span>
      <Icon :name="open ? 'chevron-down' : 'chevron-right'" :size="14" class="think__chev" />
    </button>
    <div
      v-if="open"
      ref="body"
      class="think__body"
      :class="{ 'think__body--live': active }"
      @scroll="onScroll"
    >
      <Markdown :content="reasoning" :streaming="active" />
    </div>
  </div>
</template>

<style scoped>
.think {
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
  margin-bottom: 10px;
  overflow: hidden;
}
.think__head {
  display: flex;
  align-items: center;
  gap: 7px;
  width: 100%;
  padding: 7px 10px;
  border: none;
  background: transparent;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-xs);
  font-weight: 500;
  cursor: pointer;
}
.think__head:hover {
  color: var(--pk-text-primary);
}
.think--active .think__label {
  color: var(--pk-accent-text);
}
.think__meta {
  font-family: var(--pk-font-mono);
  font-size: 11px;
  color: var(--pk-text-muted);
}
.think__chev {
  margin-left: auto;
}
.think__body {
  padding: 2px 12px 12px 12px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  word-break: break-word;
}
/* live window: capped and scrolling while tokens stream; the cap comes off
   the moment thinking ends and the full thought stands at natural height */
.think__body--live {
  max-height: 220px;
  overflow-y: auto;
  overscroll-behavior: contain;
}
/* reasoning reads as a subdued version of the answer prose */
.think__body :deep(.pk-md) {
  font-size: var(--pk-font-size-sm);
}
</style>
