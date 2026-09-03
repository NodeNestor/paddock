<script setup lang="ts">
// Reka-backed tab strip (the reuse rule: no hand-rolled tabs). Value-only
// wrapper: the caller switches its own content on the model - TabsContent
// stays unused so panels keep their own layout, and the strip carries the
// roving-focus/aria wiring we'd otherwise hand-roll.
import { TabsList, TabsRoot, TabsTrigger } from 'reka-ui'

export interface TabOption {
  value: string
  label: string
  disabled?: boolean
  /** muted trailing note, e.g. "not supported by this model". */
  hint?: string
}
const model = defineModel<string>({ required: true })
defineProps<{ tabs: TabOption[] }>()
</script>

<template>
  <TabsRoot v-model="model">
    <TabsList class="pk-tabs">
      <TabsTrigger
        v-for="t in tabs"
        :key="t.value"
        :value="t.value"
        :disabled="t.disabled"
        class="pk-tabs__tab"
      >
        <slot name="tab" :tab="t">
          {{ t.label }}<span v-if="t.hint" class="pk-tabs__hint">{{ t.hint }}</span>
        </slot>
      </TabsTrigger>
    </TabsList>
  </TabsRoot>
</template>

<style>
/* The app's tab look - the Instrument underline strip (ins__tabs), not an
   invented pill segment (match the app, don't redesign
   it). Instrument can adopt this wrapper and shed its local copy. */
.pk-tabs {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid var(--pk-border-default);
}
.pk-tabs__tab {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  border: none;
  border-bottom: 2px solid transparent;
  background: none;
  color: var(--pk-text-muted);
  font: inherit;
  font-size: var(--pk-font-size-sm);
  cursor: pointer;
}
.pk-tabs__tab:hover {
  color: var(--pk-text-primary);
}
.pk-tabs__tab[data-state='active'] {
  color: var(--pk-accent);
  border-bottom-color: var(--pk-accent);
  font-weight: 600;
}
.pk-tabs__tab[data-disabled] {
  opacity: 0.5;
  cursor: not-allowed;
}
.pk-tabs__hint {
  margin-left: 6px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
}
</style>
