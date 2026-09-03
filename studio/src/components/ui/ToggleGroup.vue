<script setup lang="ts">
// Reka-backed segmented control: a strip of buttons where exactly one is on.
// Container half of the pair (see RadioGroup for the same shape). Use this for
// a MODE the caller switches on - a workload preset, an area of the app - and
// RadioGroup when the thing being picked is a value in a form.
// Reka gives roving focus (one tab stop, arrows move), aria-pressed per item,
// and refuses to deselect the last item when `single`.
import { ToggleGroupRoot } from 'reka-ui'

const model = defineModel<string>({ required: true })
defineProps<{ label: string; disabled?: boolean }>()
</script>

<template>
  <ToggleGroupRoot
    type="single"
    :model-value="model"
    :aria-label="label"
    :disabled="disabled"
    loop
    @update:model-value="(v) => { if (typeof v === 'string' && v) model = v }"
  >
    <slot />
  </ToggleGroupRoot>
</template>
