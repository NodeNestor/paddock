<script setup lang="ts">
// One choice inside RadioGroup. Deliberately unstyled: the caller passes its
// own class and content (the quality cards are whole cards, not dots), and
// Vue's scoped CSS reaches this root element because it is the child
// component's root. Selection state is read off Reka's [data-state='checked']
// so nobody recomputes "is this the current one" in a :class binding.
import { RadioGroupItem } from 'reka-ui'

defineProps<{ value: string; disabled?: boolean }>()
</script>

<template>
  <RadioGroupItem :value="value" :disabled="disabled">
    <slot />
  </RadioGroupItem>
</template>
