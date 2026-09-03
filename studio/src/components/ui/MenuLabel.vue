<script setup lang="ts">
// A group heading inside a menu (Reka DropdownMenuLabel - not focusable, so
// keyboard nav steps over it the way it steps over a separator). Pairs with
// MenuSeparator when one menu carries two kinds of choice.
import { DropdownMenuLabel } from 'reka-ui'
</script>

<template>
  <DropdownMenuLabel class="pk-menu__label">
    <slot />
  </DropdownMenuLabel>
</template>

<!-- Unscoped: portalled to <body> by Reka (scoped hashes don't reach it). -->
<style>
.pk-menu__label {
  padding: 6px 10px 4px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--pk-text-muted);
}
</style>
