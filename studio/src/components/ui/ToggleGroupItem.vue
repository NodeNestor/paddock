<script setup lang="ts">
// One segment inside ToggleGroup. Unstyled deliberately - the caller's class and
// scoped CSS land on this root element - and its on/off state is Reka's
// [data-state='on'], not a hand-maintained :class.
//
// `as` renders the item as something other than a <button>. Needed when a
// segment carries its own controls: a stepper inside a <button> is a button
// inside a button, which is invalid HTML and behaves unpredictably. Reka keeps
// the role and keyboard behaviour whatever element it renders.
import { ToggleGroupItem } from 'reka-ui'

defineProps<{ value: string; disabled?: boolean; as?: string }>()
</script>

<template>
  <ToggleGroupItem :value="value" :disabled="disabled" :as="as">
    <slot />
  </ToggleGroupItem>
</template>
