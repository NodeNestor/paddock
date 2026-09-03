<script setup lang="ts">
// Wraps the caller's own trigger element (as-child): Reka merges the open
// state, aria-haspopup/expanded and the toggle handler onto it, so the app
// keeps its existing styled <button> markup unchanged.
import { DropdownMenuTrigger } from 'reka-ui'
</script>

<template>
  <DropdownMenuTrigger as-child>
    <slot />
  </DropdownMenuTrigger>
</template>
