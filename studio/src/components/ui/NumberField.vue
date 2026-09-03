<script setup lang="ts">
// Reka-backed number input (the only numeric stepper - reka-ui reuse rule).
// Carries the spinbutton aria, keyboard steps, and clamping; we style the
// box + steppers. v-model is a plain number.
import { NumberFieldDecrement, NumberFieldIncrement, NumberFieldInput, NumberFieldRoot } from 'reka-ui'
import Icon from '@/components/Icon.vue'

const model = defineModel<number>({ required: true })
defineProps<{
  min?: number
  max?: number
  step?: number
  disabled?: boolean
}>()
</script>

<template>
  <NumberFieldRoot
    class="pk-num"
    :model-value="model"
    :min="min"
    :max="max"
    :step="step"
    :disabled="disabled"
    :format-options="{ useGrouping: false }"
    @update:model-value="(v) => v !== undefined && (model = v)"
  >
    <NumberFieldDecrement class="pk-num__step">
      <Icon name="chevron-down" :size="12" />
    </NumberFieldDecrement>
    <NumberFieldInput class="pk-num__input" />
    <NumberFieldIncrement class="pk-num__step">
      <Icon name="chevron-up" :size="12" />
    </NumberFieldIncrement>
  </NumberFieldRoot>
</template>

<style scoped>
.pk-num {
  display: inline-flex;
  align-items: stretch;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  overflow: hidden;
}
.pk-num:focus-within {
  border-color: var(--pk-accent);
}
.pk-num[data-disabled] {
  opacity: 0.6;
}
.pk-num__input {
  width: 9ch;
  padding: 7px 4px;
  border: none;
  background: transparent;
  color: var(--pk-text-primary);
  font: inherit;
  font-size: var(--pk-font-size-sm);
  font-variant-numeric: tabular-nums;
  text-align: center;
  outline: none;
}
.pk-num__step {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  border: none;
  background: transparent;
  color: var(--pk-text-muted);
  cursor: pointer;
}
.pk-num__step:hover {
  color: var(--pk-text-primary);
  background: var(--pk-bg-hover);
}
.pk-num__step[data-disabled] {
  opacity: 0.4;
  cursor: default;
}
</style>
