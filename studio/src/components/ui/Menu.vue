<script setup lang="ts">
// Root of the shared dropdown menu (Reka DropdownMenu). Supports both controlled
// (`v-model:open`) and uncontrolled (no `open`) callers: we always drive reka in
// *controlled* mode and hold the open state locally when the caller doesn't.
// Reka still owns outside-click, Escape, roving focus, type-ahead and aria -
// never hand-roll document listeners.
//
//   <Menu>                          <!-- uncontrolled: state kept here -->
//     <MenuTrigger>...</MenuTrigger>
//     <MenuContent>...</MenuContent>
//   </Menu>
import { computed, ref } from 'vue'
import { DropdownMenuRoot } from 'reka-ui'

const props = withDefaults(
  defineProps<{
    open?: boolean
    /**
     * Reka's modal mode traps focus + locks pointer-events on the rest of the
     * page. That fights any other modal layer around it: a modal menu nested in
     * a modal Dialog opens and self-closes in the same tick (looks like the
     * trigger does nothing). Pass `:modal="false"` when this menu lives inside a
     * Dialog. Defaults true for standalone menus.
     */
    modal?: boolean
  }>(),
  // `open: undefined` is LOAD-BEARING, not decoration. Vue casts an absent
  // Boolean prop to `false` (never `undefined`), so without an explicit default
  // the uncontrolled check below can never see "caller passed nothing" - every
  // <Menu> would look permanently controlled-and-closed and the trigger would
  // do nothing. Declaring a default opts out of that cast. reka's own `open`
  // prop is declared `default: void 0` for exactly this reason.
  { open: undefined, modal: true },
)
const emit = defineEmits<{ (e: 'update:open', value: boolean): void }>()

const internal = ref(false)
const openState = computed<boolean>({
  get: () => (props.open !== undefined ? props.open : internal.value),
  set: (v) => {
    internal.value = v
    emit('update:open', v)
  },
})
</script>

<template>
  <DropdownMenuRoot v-model:open="openState" :modal="modal">
    <slot />
  </DropdownMenuRoot>
</template>
