<script setup lang="ts">
// Reka-backed disclosure (the only collapsible in Studio - reka-ui reuse rule).
// Carries the aria-expanded/controls wiring and keyboard handling; we style the
// summary row and supply the chevron. `summary` is the always-visible line;
// the default slot is what unfolds.
import { CollapsibleContent, CollapsibleRoot, CollapsibleTrigger } from 'reka-ui'
import Icon from '@/components/Icon.vue'

defineProps<{ summary: string; hint?: string }>()
const open = defineModel<boolean>('open', { default: false })
</script>

<template>
  <CollapsibleRoot v-model:open="open" class="pk-coll">
    <CollapsibleTrigger class="pk-coll__row">
      <Icon :name="open ? 'chevron-down' : 'chevron-right'" :size="13" />
      <span class="pk-coll__sum">{{ summary }}</span>
      <span v-if="hint" class="pk-coll__hint">{{ hint }}</span>
    </CollapsibleTrigger>
    <CollapsibleContent class="pk-coll__body">
      <slot />
    </CollapsibleContent>
  </CollapsibleRoot>
</template>

<style scoped>
.pk-coll__row {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 4px 0;
  border: 0;
  background: none;
  color: var(--pk-text-secondary);
  font: inherit;
  font-size: var(--pk-font-size-xs);
  text-align: left;
  cursor: pointer;
}
.pk-coll__row:hover {
  color: var(--pk-text-primary);
}
.pk-coll__hint {
  color: var(--pk-text-muted);
}
.pk-coll__body {
  padding: 2px 0 6px 19px;
}
</style>
