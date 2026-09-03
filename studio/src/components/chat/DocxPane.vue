<script setup lang="ts">
// Word pane of the tabbed file preview: scriptor's read-mode ScriptorDoc -
// the document renders through the same OOXML engine that extracted its text
// for the model, so what you see is what the model read. Bytes arrive from
// the host (fetched once for all tabs).
import { ScriptorDoc } from '@truespar/scriptor-vue'

defineProps<{ bytes: Uint8Array | null }>()
</script>

<template>
  <div class="pv__pane">
    <div class="pv__canvas dv__scroll">
      <ScriptorDoc v-if="bytes" :docx="bytes" mode="read" :selectable="true" />
    </div>
  </div>
</template>

<style>
.dv__scroll {
  padding: 16px 0;
}
/* Center the document like a page stage. scriptor's .scriptor-sheet is an
   inline-block inside ScriptorDoc's root div; max-content + auto margins
   center it while it fits and fall back to left-edge + scroll when a page
   is wider than the dialog (flex centering would clip the left edge). */
.dv__scroll > div {
  width: max-content;
  margin: 0 auto;
}
</style>
