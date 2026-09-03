<script setup lang="ts">
// Reka-backed single-thumb slider. reka's model is a number[]; this wraps it to
// a single number via defineModel. The only slider in Studio (reka-ui reuse
// rule). Not portalled, so scoped styles reach the primitives.
import { SliderRoot, SliderTrack, SliderRange, SliderThumb } from 'reka-ui'

const model = defineModel<number>({ required: true })
withDefaults(
  defineProps<{ min?: number; max?: number; step?: number; disabled?: boolean }>(),
  { min: 0, max: 100, step: 1, disabled: false },
)
</script>

<template>
  <SliderRoot
    class="pk-slider"
    :model-value="[model]"
    :min="min"
    :max="max"
    :step="step"
    :disabled="disabled"
    @update:model-value="(v) => { if (v && v.length) model = v[0] }"
  >
    <SliderTrack class="pk-slider__track">
      <SliderRange class="pk-slider__range" />
    </SliderTrack>
    <SliderThumb class="pk-slider__thumb" aria-label="Value" />
  </SliderRoot>
</template>

<style scoped>
.pk-slider {
  position: relative;
  display: flex;
  align-items: center;
  width: 100%;
  height: 20px;
  touch-action: none;
  user-select: none;
}
.pk-slider[data-disabled] {
  opacity: 0.45;
}
.pk-slider__track {
  position: relative;
  flex-grow: 1;
  height: 5px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-border-default);
}
.pk-slider__range {
  position: absolute;
  height: 100%;
  border-radius: var(--pk-radius-full);
  background: var(--pk-accent);
}
.pk-slider__thumb {
  display: block;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--pk-accent);
  box-shadow: var(--pk-shadow-sm);
  cursor: grab;
  transition: box-shadow 0.15s ease;
}
.pk-slider__thumb:hover {
  box-shadow: 0 0 0 4px var(--pk-accent-subtle);
}
.pk-slider__thumb:focus-visible {
  outline: none;
  box-shadow: 0 0 0 4px var(--pk-accent-subtle);
}
.pk-slider__thumb:active {
  cursor: grabbing;
}
</style>
