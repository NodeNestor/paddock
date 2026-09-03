<script setup lang="ts">
// One actionable row. Emits `select` on click / Enter / Space (Reka). The menu
// auto-closes after select unless the handler calls `event.preventDefault()`
// (use that for in-place toggles). `as-child` lets the row be another element
// while keeping the menuitem role.
import { DropdownMenuItem } from 'reka-ui'

defineProps<{
  disabled?: boolean | undefined
  danger?: boolean | undefined
  asChild?: boolean | undefined
}>()
defineEmits<{ (e: 'select', event: Event): void }>()
</script>

<template>
  <DropdownMenuItem
    class="pk-menu__item"
    :class="{ 'pk-menu__item--danger': danger }"
    :disabled="disabled"
    :as-child="asChild"
    @select="$emit('select', $event)"
  >
    <slot />
  </DropdownMenuItem>
</template>

<!-- Unscoped: portalled to <body> by Reka (scoped hashes don't reach it). -->
<style>
.pk-menu__item {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 7px 10px;
  border: 0;
  border-radius: var(--pk-radius-md);
  background: transparent;
  color: var(--pk-text-primary);
  font: inherit;
  font-size: var(--pk-font-size-sm);
  text-align: left;
  text-decoration: none;
  cursor: pointer;
  outline: none;
  transition: background 0.12s ease, color 0.12s ease;
}
/* Reka sets data-highlighted on the keyboard/pointer-focused row. */
.pk-menu__item[data-highlighted],
.pk-menu__item:hover {
  background: var(--pk-bg-hover);
}
.pk-menu__item[data-disabled] {
  opacity: 0.5;
  cursor: default;
  pointer-events: none;
}
.pk-menu__item svg {
  flex-shrink: 0;
  color: var(--pk-text-muted);
}

.pk-menu__item--danger {
  color: var(--pk-status-error);
}
.pk-menu__item--danger svg {
  color: var(--pk-status-error);
}
.pk-menu__item--danger[data-highlighted],
.pk-menu__item--danger:hover {
  background: var(--pk-status-error-subtle);
}
</style>
