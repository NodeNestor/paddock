<script setup lang="ts">
// A hairline rule between menu groups (Reka DropdownMenuSeparator - aria
// role="separator", skipped by keyboard nav).
import { DropdownMenuSeparator } from 'reka-ui'
</script>

<template>
  <DropdownMenuSeparator class="pk-menu__sep" />
</template>

<!-- Unscoped: portalled to <body> by Reka (scoped hashes don't reach it). -->
<style>
.pk-menu__sep {
  height: 1px;
  border: 0;
  background: var(--pk-border-default);
  margin: 4px -4px;
}
</style>
