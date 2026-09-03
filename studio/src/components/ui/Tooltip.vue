<script setup lang="ts">
// Reka-backed tooltip - the single reusable tooltip in Studio. Wrap the trigger
// element, pass its text via `label`:
//
//   <Tooltip label="Send"><button>...</button></Tooltip>
//   <Tooltip :label="busy ? undefined : 'Send'" side="top">...</Tooltip>
//
// A falsy `label` renders the slotted child bare (no tooltip), so callers can
// bind a maybe-empty expression. `as-child` merges the trigger wiring onto the
// single slotted element, which must forward attrs (native elements do). Shared
// delay + Floating-UI collision handling come from the one <TooltipProvider>
// mounted at the app root (App.vue). This is the only way we do tooltips -
// never a hand-rolled `title=` overlay (see the reka-ui reuse rule).
//
// Tooltip on a menu/popover trigger: this component must sit inside the other
// primitive's trigger, button innermost -
//
//   <MenuTrigger><Tooltip label="..."><button>...</button></Tooltip></MenuTrigger>
//
// never outside it. Reka's Floating-UI anchor context ("PopperRoot") is one
// unscoped injection key shared by every floating family, and TooltipRoot
// provides its own: wrap a MenuTrigger and the menu's anchor registers on the
// TOOLTIP's popper root, so the menu opens with no anchor - parked off-screen
// at the top-left. State toggles fine, which is exactly why it reads as "the
// button does nothing". Nesting inside works because we forward $attrs: the
// menu trigger's merged props (toggle handler, aria) arrive as our attrs and
// ride through TooltipTrigger's as-child chain onto the slotted button; the
// bare branch routes them through Reka's own Slot for the same merge.
import {
  Slot,
  TooltipRoot,
  TooltipTrigger,
  TooltipPortal,
  TooltipContent,
  TooltipArrow,
} from 'reka-ui'

defineOptions({ inheritAttrs: false })

withDefaults(
  defineProps<{
    label?: string | null | undefined
    side?: 'top' | 'right' | 'bottom' | 'left'
    sideOffset?: number
    disabled?: boolean
  }>(),
  { side: 'top', sideOffset: 8, disabled: false },
)
</script>

<template>
  <Slot v-if="!label || disabled" v-bind="$attrs">
    <slot />
  </Slot>
  <TooltipRoot v-else>
    <TooltipTrigger as-child v-bind="$attrs">
      <slot />
    </TooltipTrigger>
    <TooltipPortal>
      <TooltipContent
        class="pk-tip"
        :side="side"
        :side-offset="sideOffset"
        :collision-padding="6"
        role="tooltip"
      >
        {{ label }}
        <TooltipArrow class="pk-tip__arrow" :width="10" :height="5" />
      </TooltipContent>
    </TooltipPortal>
  </TooltipRoot>
</template>

<!-- Unscoped: TooltipContent is portalled to <body> by Reka, so a scoped hash
     would never reach it. All classes are `pk-tip`-prefixed; tokens resolve off
     :root. -->
<style>
.pk-tip {
  z-index: 9500;
  max-width: 16rem;
  padding: 5px 9px;
  background: var(--pk-bg-elevated);
  color: var(--pk-text-primary);
  border: 1px solid var(--pk-border-strong);
  border-radius: var(--pk-radius-md);
  box-shadow: var(--pk-shadow-lg);
  font-family: var(--pk-font-sans);
  font-size: var(--pk-font-size-xs);
  font-weight: 500;
  line-height: 1.4;
  white-space: normal;
  text-align: center;
  word-break: break-word;
  user-select: none;
  transform-origin: var(--reka-tooltip-content-transform-origin);
}
.pk-tip[data-state='delayed-open'],
.pk-tip[data-state='instant-open'] {
  animation: pk-tip-in 0.12s ease;
}
.pk-tip__arrow {
  fill: var(--pk-bg-elevated);
}

@keyframes pk-tip-in {
  from {
    opacity: 0;
    transform: scale(0.96);
  }
}
@media (prefers-reduced-motion: reduce) {
  .pk-tip[data-state] {
    animation: none;
  }
}
</style>
