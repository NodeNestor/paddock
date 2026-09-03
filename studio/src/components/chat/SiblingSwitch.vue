<script setup lang="ts">
// The `< 2/3 >` control under a turn that has alternatives.
//
// It appears only where a branch actually exists, which is the point: a
// conversation you never edited or re-rolled looks exactly as it did before
// the tree, and the control is the only thing telling you the other versions
// are still there. Without it a branch is unreachable and therefore lost.
import { computed } from 'vue'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

const props = defineProps<{ index: number; count: number }>()
const emit = defineEmits<{ go: [delta: number] }>()

const atFirst = computed(() => props.index <= 0)
const atLast = computed(() => props.index >= props.count - 1)
</script>

<template>
  <nav v-if="count > 1" class="sib" aria-label="Other versions of this turn">
    <Tooltip label="Previous version">
      <button
        class="sib__btn"
        type="button"
        :disabled="atFirst"
        aria-label="Previous version"
        @click="emit('go', -1)"
      >
        <Icon name="chevron-left" :size="13" />
      </button>
    </Tooltip>
    <!-- aria-live so a screen reader hears the position change; tabular-nums
         so 9/10 does not shift the arrows sideways as you walk the list -->
    <span class="sib__count" aria-live="polite">{{ index + 1 }}/{{ count }}</span>
    <Tooltip label="Next version">
      <button
        class="sib__btn"
        type="button"
        :disabled="atLast"
        aria-label="Next version"
        @click="emit('go', 1)"
      >
        <Icon name="chevron-right" :size="13" />
      </button>
    </Tooltip>
  </nav>
</template>

<style scoped>
.sib {
  display: inline-flex;
  align-items: center;
  gap: 1px;
}
.sib__btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border: none;
  border-radius: var(--pk-radius-sm);
  background: none;
  color: var(--pk-text-muted);
  cursor: pointer;
}
.sib__btn:hover:not(:disabled) {
  background: var(--pk-bg-elevated);
  color: var(--pk-text-primary);
}
.sib__btn:disabled {
  opacity: 0.35;
  cursor: default;
}
.sib__count {
  min-width: 26px;
  text-align: center;
  font-family: var(--pk-font-mono);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  color: var(--pk-text-muted);
  user-select: none;
}
</style>
