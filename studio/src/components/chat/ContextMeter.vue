<script setup lang="ts">
import { computed } from 'vue'
import Progress from '@/components/ui/Progress.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

const props = defineProps<{ used: number; max: number }>()

const ratio = computed(() => (props.max > 0 ? props.used / props.max : 0))
const level = computed(() => {
  if (ratio.value >= 1) return 'over'
  if (ratio.value >= 0.85) return 'high'
  return 'ok'
})

function fmt(n: number): string {
  // binary-K to match the "128K" context label in Settings
  if (n >= 1024) {
    const k = n / 1024
    return `${Number.isInteger(k) ? k : k.toFixed(1)}K`
  }
  return String(Math.round(n))
}

const label = computed(() => `${fmt(props.used)} / ${fmt(props.max)}`)
const tip = computed(
  () =>
    `Context used: ${Math.round(props.used).toLocaleString()} / ${props.max.toLocaleString()} tokens` +
    (level.value === 'over' ? ' - oldest messages are trimmed to fit' : ''),
)
</script>

<template>
  <Tooltip :label="tip">
    <div class="ctx" :class="`ctx--${level}`">
      <Progress
        class="ctx__bar"
        :value="used"
        :max="max"
        :tone="level === 'over' ? 'error' : level === 'high' ? 'warning' : 'accent'"
        :label="tip"
        size="xs"
      />
      <span class="ctx__label">{{ label }}</span>
    </div>
  </Tooltip>
</template>

<style scoped>
.ctx {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 30px;
  padding: 0 4px;
  cursor: default;
  user-select: none;
}
/* two classes deep so it outranks Progress's own `width: 100%` - both carry a
   scope attribute, so at equal specificity the winner would be bundle order */
.ctx .ctx__bar {
  width: 44px;
  flex: none;
}
.ctx__label {
  font-family: var(--pk-font-mono);
  font-size: 11px;
  color: var(--pk-text-muted);
  white-space: nowrap;
}
.ctx--high .ctx__label {
  color: var(--pk-status-warning);
}
.ctx--over .ctx__label {
  color: var(--pk-status-error);
}
</style>
