<script setup lang="ts">
// Reka-backed radio group (reka-ui reuse rule: never hand-write role="radio").
// Container half of the pair - the caller supplies RadioItems as children and
// styles them, exactly like Menu/MenuItem. What this buys over the buttons it
// replaces: one tab stop for the whole group, arrow keys to move the choice,
// and aria-checked maintained by Reka rather than by hand.
import { RadioGroupRoot } from 'reka-ui'

const model = defineModel<string>({ required: true })
withDefaults(
  defineProps<{
    label: string
    disabled?: boolean
    orientation?: 'horizontal' | 'vertical'
  }>(),
  { orientation: 'horizontal' },
)
</script>

<template>
  <RadioGroupRoot
    v-model="model"
    :aria-label="label"
    :disabled="disabled"
    :orientation="orientation"
    loop
  >
    <slot />
  </RadioGroupRoot>
</template>
