<script setup lang="ts">
// Reka-backed popover - the reusable click-to-open overlay in Studio. Unlike
// Tooltip (hover, terse label, non-interactive), this holds content the user can
// select, copy, and click into:
//
//   <Popover>
//     <template #trigger><button>Models folder</button></template>
//     <p>...anything, including <a> and <button></p>
//   </Popover>
//
// `as-child` merges the trigger wiring onto the single slotted element, which
// must forward attrs (native elements do). Reka owns focus trapping, Esc/
// outside-click dismissal, and Floating-UI collision handling. This is the only
// way we do popovers - never a hand-rolled absolute-positioned div (see the
// reka-ui reuse rule).
import {
  PopoverRoot,
  PopoverTrigger,
  PopoverPortal,
  PopoverContent,
  PopoverArrow,
} from 'reka-ui'

withDefaults(
  defineProps<{
    side?: 'top' | 'right' | 'bottom' | 'left'
    align?: 'start' | 'center' | 'end'
    sideOffset?: number
  }>(),
  { side: 'bottom', align: 'start', sideOffset: 8 },
)

/** Optional two-way open state (v-model:open) - callers that must react to
 *  open/close bind it (the composer's tool picker kicks off tool listings);
 *  everyone else leaves it unbound and Reka behaves as before. */
const open = defineModel<boolean>('open', { default: false })
</script>

<template>
  <PopoverRoot v-model:open="open">
    <PopoverTrigger as-child>
      <slot name="trigger" />
    </PopoverTrigger>
    <PopoverPortal>
      <PopoverContent
        class="pk-pop"
        :side="side"
        :align="align"
        :side-offset="sideOffset"
        :collision-padding="10"
      >
        <slot />
        <PopoverArrow class="pk-pop__arrow" :width="10" :height="5" />
      </PopoverContent>
    </PopoverPortal>
  </PopoverRoot>
</template>

<!-- Unscoped: PopoverContent is portalled to <body> by Reka, so a scoped hash
     would never reach it. All classes are `pk-pop`-prefixed; tokens resolve off
     :root. -->
<style>
.pk-pop {
  z-index: 9500;
  max-width: 22rem;
  padding: 12px 14px;
  background: var(--pk-bg-elevated);
  color: var(--pk-text-primary);
  border: 1px solid var(--pk-border-strong);
  border-radius: var(--pk-radius-lg);
  box-shadow: var(--pk-shadow-lg);
  font-family: var(--pk-font-sans);
  font-size: var(--pk-font-size-xs);
  line-height: 1.5;
  transform-origin: var(--reka-popover-content-transform-origin);
}
.pk-pop:focus-visible {
  outline: none;
}
.pk-pop[data-state='open'] {
  animation: pk-pop-in 0.12s ease;
}
.pk-pop__arrow {
  fill: var(--pk-bg-elevated);
}

@keyframes pk-pop-in {
  from {
    opacity: 0;
    transform: scale(0.96);
  }
}
@media (prefers-reduced-motion: reduce) {
  .pk-pop[data-state] {
    animation: none;
  }
}
</style>
