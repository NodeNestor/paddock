<script setup lang="ts">
// Reka-backed select (the only dropdown-with-a-value in the app - reka-ui
// reuse rule; Menu is for actions, Select is for picking a value). Carries
// the listbox/aria/typeahead/keyboard wiring; we style trigger + content.
// v-model is string | number; options carry an optional muted hint.
import { computed } from 'vue'
import {
  SelectContent,
  SelectIcon,
  SelectItem,
  SelectItemIndicator,
  SelectItemText,
  SelectPortal,
  SelectRoot,
  SelectTrigger,
  SelectValue,
  SelectViewport,
} from 'reka-ui'
import Icon from '@/components/Icon.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

export interface SelectOption {
  value: string | number
  label: string
  /** muted trailing note, e.g. a size or "not installed". */
  hint?: string
  /** maker name ("Alibaba", "OpenAI") - renders the vendor mark before the
   *  label, in the trigger and in the list. */
  vendor?: string
  /** tooltip on the item - where the technical id lives. */
  title?: string
  disabled?: boolean
}

const model = defineModel<string | number>({ required: true })
const props = defineProps<{
  options: SelectOption[]
  placeholder?: string
  disabled?: boolean
  /** stretch to the container (default: size to content). */
  block?: boolean
  /** chrome-less trigger (header chips): no border/background until hover. */
  ghost?: boolean
}>()

// Reka round-trips option values as-is, but a v-model bound to a number must
// come back a number - map through the options so types survive the trip.
const current = computed(() => props.options.find((o) => o.value === model.value))
function onUpdate(v: unknown): void {
  const hit = props.options.find((o) => String(o.value) === String(v))
  if (hit) model.value = hit.value
}
</script>

<template>
  <SelectRoot
    :model-value="current !== undefined ? String(current.value) : undefined"
    :disabled="disabled"
    @update:model-value="onUpdate"
  >
    <SelectTrigger
      class="pk-select"
      :class="{ 'pk-select--block': block, 'pk-select--ghost': ghost }"
    >
      <!-- Reka's canonical trigger anatomy, nothing hand-rolled: a
           self-closed SelectValue (Reka renders the selected item's text)
           and the ONE caret inside SelectIcon. The vendor mark is a plain
           sibling; the hint lives in the open list, not the trigger. -->
      <VendorLogo
        v-if="current?.vendor"
        :vendor="current.vendor"
        :size="15"
        class="pk-select__logo"
      />
      <!-- the label comes from our options prop, not Reka's item registry:
           the registry lives in the portalled list, so dismissing the
           dropdown unregisters items and a self-closed SelectValue flickers
           while it recomputes -->
      <SelectValue class="pk-select__value" :placeholder="placeholder ?? 'Select...'">
        <template v-if="current">{{ current.label }}</template>
      </SelectValue>
      <SelectIcon as-child>
        <Icon name="chevron-down" :size="12" class="pk-select__caret" />
      </SelectIcon>
    </SelectTrigger>
    <SelectPortal>
      <SelectContent class="pk-select__content" position="popper" :side-offset="4">
        <SelectViewport class="pk-select__viewport">
          <SelectItem
            v-for="o in options"
            :key="String(o.value)"
            :value="String(o.value)"
            :disabled="o.disabled"
            class="pk-select__item"
          >
            <SelectItemIndicator class="pk-select__check">
              <Icon name="check" :size="13" />
            </SelectItemIndicator>
            <VendorLogo v-if="o.vendor" :vendor="o.vendor" :size="15" class="pk-select__logo" />
            <Tooltip :label="o.title" side="right">
              <span><SelectItemText>{{ o.label }}</SelectItemText></span>
            </Tooltip>
            <span v-if="o.hint" class="pk-select__hint">{{ o.hint }}</span>
          </SelectItem>
        </SelectViewport>
      </SelectContent>
    </SelectPortal>
  </SelectRoot>
</template>

<!-- Unscoped, like MenuContent: the listbox is portalled to <body> by Reka,
     so a scoped style hash would never reach it. Classes are pk-select-
     prefixed. -->
<style>
.pk-select {
  display: inline-flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 220px;
  padding: 7px 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  color: var(--pk-text-primary);
  font: inherit;
  font-size: var(--pk-font-size-sm);
  cursor: pointer;
  text-align: left;
}
/* Fill the container. The cap that used to live here (max-width: 440px) was a
   LAYOUT decision inside a shared primitive: `block` means "fill", and a
   consumer that wants a ceiling is the one that knows what it should be. It
   left the only `block` Select in the app - the model picker on a wide card -
   stopping at ~418px with empty space beside it. */
.pk-select--block {
  width: 100%;
}
/* ghost: a chip that reveals it's a control on hover (header model picker) */
.pk-select--ghost {
  min-width: 0;
  padding: 3px 8px;
  border-color: transparent;
  background: transparent;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
  font-weight: 500;
}
.pk-select--ghost:hover {
  border-color: var(--pk-border-default);
  background: var(--pk-bg-hover);
}
.pk-select:hover {
  border-color: var(--pk-accent);
}
.pk-select[data-disabled] {
  opacity: 0.6;
  cursor: not-allowed;
}
.pk-select[data-placeholder] {
  color: var(--pk-text-muted);
}
.pk-select__caret {
  color: var(--pk-text-muted);
  flex: none;
}
/* the trigger's value: Reka-rendered selected text - one line, truncating */
.pk-select__value {
  flex: 1;
  min-width: 0;
  text-align: left;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.pk-select__logo {
  flex: none;
  display: block;
}
.pk-select__hint {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
  margin-left: 8px;
  flex: none;
}

/* the portalled listbox - pk-menu's visual recipe */
.pk-select__content {
  /* above dialogs (9701), same tier as pk-menu */
  z-index: 9800;
  min-width: var(--reka-select-trigger-width);
  max-height: min(320px, var(--reka-select-content-available-height));
  background: var(--pk-bg-elevated);
  border: 1px solid var(--pk-border-strong);
  border-radius: var(--pk-radius-lg);
  box-shadow: var(--pk-shadow-lg);
  overflow: hidden;
}
.pk-select__viewport {
  padding: 4px;
}
.pk-select__item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px 6px 28px;
  position: relative;
  border-radius: var(--pk-radius-sm);
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
  cursor: pointer;
  user-select: none;
  outline: none;
}
.pk-select__item[data-highlighted] {
  background: var(--pk-bg-hover);
}
.pk-select__item[data-disabled] {
  opacity: 0.5;
  cursor: default;
}
.pk-select__check {
  position: absolute;
  left: 8px;
  display: inline-flex;
  color: var(--pk-accent);
}
</style>
