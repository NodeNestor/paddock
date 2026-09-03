<script setup lang="ts">
// Reka-backed toggle switch (the only switch in Studio - reka-ui reuse rule).
// Carries the role/aria/keyboard wiring; we only style the track + thumb.
// v-model is a plain boolean.
//
// `label` is not optional in practice. SwitchRoot renders a <button
// role="switch"> with no text inside it, and a <button> is not a labelable
// element - so the surrounding <label class="..."> that every call site wraps it
// in contributes nothing to its accessible name, and Reka's own Label
// primitive can't help either (it just renders <label for>, which needs that
// same native association). Without this prop a screen reader reaches every
// toggle in the app as an anonymous "switch, off".
import { SwitchRoot, SwitchThumb } from 'reka-ui'

const model = defineModel<boolean>({ required: true })
defineProps<{ disabled?: boolean; label?: string | undefined }>()
</script>

<template>
  <SwitchRoot
    class="pk-switch"
    :model-value="model"
    :disabled="disabled"
    :aria-label="label"
    @update:model-value="(v) => (model = v)"
  >
    <SwitchThumb class="pk-switch__thumb" />
  </SwitchRoot>
</template>

<style scoped>
.pk-switch {
  position: relative;
  display: inline-flex;
  align-items: center;
  width: 34px;
  height: 20px;
  flex-shrink: 0;
  padding: 0;
  border: none;
  border-radius: var(--pk-radius-full);
  background: var(--pk-border-default);
  cursor: pointer;
  transition: background 0.18s ease;
}
.pk-switch[data-state='checked'] {
  background: var(--pk-accent);
}
.pk-switch[data-disabled] {
  opacity: 0.5;
  cursor: default;
}
.pk-switch__thumb {
  display: block;
  width: 16px;
  height: 16px;
  margin: 0 2px;
  border-radius: 50%;
  background: #fff;
  box-shadow: var(--pk-shadow-sm);
  transition: transform 0.18s ease;
  transform: translateX(0);
  will-change: transform;
}
.pk-switch__thumb[data-state='checked'] {
  transform: translateX(14px);
}
</style>
