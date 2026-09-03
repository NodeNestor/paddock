<script setup lang="ts">
// The query editor under the graph: a compact Monaco with Cypher highlighting
// (monaco's own tokenizer) and the completions lib/graph/cypher-language
// layers on. The strip only edits and submits - what a result means is the
// parent's business.
import { ref } from 'vue'
import CodeEditor from '@/components/ui/CodeEditor.vue'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

defineProps<{
  running: boolean
}>()
const emit = defineEmits<{ run: [cypher: string] }>()

const text = ref('MATCH (n) RETURN n LIMIT 25')

function submit(): void {
  const q = text.value.trim()
  if (q) emit('run', q)
}
</script>

<template>
  <div class="qs">
    <div class="qs__editor">
      <CodeEditor v-model="text" language="cypher" runnable @run="submit" />
    </div>
    <Tooltip label="Run (Ctrl+Enter)">
      <button class="qs__run" :disabled="running" @click="submit">
        <Icon :name="running ? 'spinner' : 'play'" :size="14" />
      </button>
    </Tooltip>
  </div>
</template>

<style scoped>
.qs {
  display: flex;
  align-items: stretch;
  gap: 6px;
  min-height: 0;
}
.qs__editor {
  flex: 1;
  min-width: 0;
  height: 76px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-sm);
  overflow: hidden;
}
.qs__run {
  display: grid;
  place-items: center;
  width: 34px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-surface);
  color: var(--pk-text-secondary);
  cursor: pointer;
}
.qs__run:hover:not(:disabled) {
  color: var(--pk-text-primary);
}
.qs__run:disabled {
  opacity: 0.5;
  cursor: default;
}
</style>
