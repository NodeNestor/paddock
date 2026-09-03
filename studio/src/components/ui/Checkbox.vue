<script setup lang="ts">
// Reka-backed checkbox (the only checkbox in the app - reka-ui reuse rule).
// Carries role="checkbox" + aria-checked on a real focusable control, which
// none of the places this replaced had: they were <span>s and <button>s
// drawing a glyph, so nothing announced a state or took a keystroke.
// v-model accepts `'indeterminate'` - the honest state for a select-all over
// a partly-picked list, and Reka spells it natively.
//
// Two glyphs, same semantics: `square` is the standalone checkbox (a list
// row you tick), `check` is the menu idiom - a bare tick that holds its
// column when off, so rows don't shift as you pick.
import { CheckboxIndicator, CheckboxRoot } from 'reka-ui'
import Icon from '@/components/Icon.vue'

const model = defineModel<boolean | 'indeterminate'>({ required: true })
const props = withDefaults(
  defineProps<{ disabled?: boolean; size?: number; glyph?: 'square' | 'check' }>(),
  { size: 16, glyph: 'square' },
)

const SQUARE = { indeterminate: 'minus-square', on: 'check-square', off: 'square' }
function icon(): string {
  if (props.glyph === 'check') return 'check'
  return model.value === 'indeterminate' ? SQUARE.indeterminate
    : model.value ? SQUARE.on
    : SQUARE.off
}
// `check` conveys state by opacity so the tick column never collapses.
function opacity(): number {
  if (props.glyph !== 'check') return 1
  return model.value === 'indeterminate' ? 0.45 : model.value ? 1 : 0
}
</script>

<template>
  <CheckboxRoot
    class="pk-check"
    :model-value="model"
    :disabled="disabled"
    @update:model-value="(v) => (model = v)"
  >
    <CheckboxIndicator class="pk-check__on" force-mount>
      <Icon :name="icon()" :size="size" :style="{ opacity: opacity() }" />
    </CheckboxIndicator>
    <slot />
  </CheckboxRoot>
</template>

<style scoped>
.pk-check {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  font: inherit;
  padding: 0;
  border: none;
  background: none;
  color: var(--pk-text-muted);
  cursor: pointer;
  border-radius: var(--pk-radius-sm);
  transition: color 0.12s ease;
}
.pk-check:hover {
  color: var(--pk-text-primary);
}
.pk-check[data-state='checked'],
.pk-check[data-state='indeterminate'] {
  color: var(--pk-accent);
}
.pk-check[data-disabled] {
  opacity: 0.5;
  cursor: default;
}
/* line-height belongs to the glyph, not the root: the root can carry label
   text, and a zero line-height there collapses it */
.pk-check__on {
  display: inline-flex;
  flex: none;
  line-height: 0;
}
</style>
