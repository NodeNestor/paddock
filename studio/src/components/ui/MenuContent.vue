<script setup lang="ts">
// The menu panel: portalled to <body> and positioned by Reka's Floating-UI
// (collision-aware flip), so `side`/`align` replace any hand-rolled
// getBoundingClientRect math. Wrap actionable rows in <MenuItem> for keyboard
// nav. Tokens resolve off :root.
import { DropdownMenuPortal, DropdownMenuContent } from 'reka-ui'

withDefaults(
  defineProps<{
    side?: 'top' | 'right' | 'bottom' | 'left'
    align?: 'start' | 'center' | 'end'
    sideOffset?: number
    alignOffset?: number
    collisionPadding?: number
    /** aria-label when the menu has no visible labelling element. */
    label?: string | undefined
    /** Floor for the panel width (the trigger width is not inherited). */
    minWidth?: string | undefined
  }>(),
  {
    side: 'bottom',
    align: 'end',
    sideOffset: 6,
    alignOffset: 0,
    collisionPadding: 8,
  },
)
</script>

<template>
  <DropdownMenuPortal>
    <DropdownMenuContent
      class="pk-menu"
      :side="side"
      :align="align"
      :side-offset="sideOffset"
      :align-offset="alignOffset"
      :collision-padding="collisionPadding"
      :aria-label="label"
      :style="minWidth ? { minWidth } : undefined"
    >
      <slot />
    </DropdownMenuContent>
  </DropdownMenuPortal>
</template>

<!-- Unscoped: the content is portalled to <body> by Reka, so a scoped style
     hash would never reach it. Classes are `pk-menu`-prefixed. -->
<style>
.pk-menu {
  /* above dialogs (9701) so a menu opened inside a modal isn't hidden behind it */
  z-index: 9800;
  display: flex;
  flex-direction: column;
  min-width: 168px;
  max-width: min(360px, calc(100vw - 16px));
  padding: 4px;
  background: var(--pk-bg-elevated);
  border: 1px solid var(--pk-border-strong);
  border-radius: var(--pk-radius-lg);
  box-shadow: var(--pk-shadow-lg);
  /* Reka writes the transform-origin for the chosen side; animate off it. */
  transform-origin: var(--reka-dropdown-menu-content-transform-origin);
}
.pk-menu[data-state='open'] {
  animation: pk-menu-in 0.12s ease;
}
.pk-menu:focus-visible {
  outline: none;
}

@keyframes pk-menu-in {
  from {
    opacity: 0;
    transform: scale(0.97);
  }
}
@media (prefers-reduced-motion: reduce) {
  .pk-menu[data-state='open'] {
    animation: none;
  }
}
</style>
