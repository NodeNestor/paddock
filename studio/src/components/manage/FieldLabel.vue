<script setup lang="ts">
// A form label that can carry an explanation without spending a paragraph on
// it.
//
// The split that makes this work: the LABEL owns the concept ("what even is a
// KV cache"), and the control's own options own the trade-off (a Select's
// per-option `hint` - "exact" / "half the memory, slightly lossy"). Those are
// two different questions asked at two different moments, and they were
// previously mashed into one block of prose under every control, which is how
// the form ended up reading like documentation.
//
// Click to open, not hover: an explanation someone deliberately asked for
// should stay put while they read it, and hover-only content is unreachable on
// touch and awkward from the keyboard. Reka's Popover owns focus, Esc and
// outside-click.
//
// The icon only renders when there is something to say, so a label with no
// slot content looks exactly like a plain label - which is most of them.
import { useSlots } from 'vue'
import Icon from '@/components/Icon.vue'
import Popover from '@/components/ui/Popover.vue'

defineProps<{ label: string }>()
const slots = useSlots()
</script>

<template>
  <label class="fl">
    <span class="fl__text">{{ label }}</span>
    <Popover v-if="slots.default" side="right" align="start">
      <template #trigger>
        <button type="button" class="fl__btn" :aria-label="`What is ${label}?`">
          <Icon name="info" :size="13" />
        </button>
      </template>
      <div class="fl__body"><slot /></div>
    </Popover>
  </label>
</template>

<style scoped>
.fl {
  display: flex;
  align-items: center;
  gap: 5px;
  margin-top: 6px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-secondary);
}
/* Resting state is nearly invisible deliberately: it is an offer, not a warning,
   and a row of bright icons down the form would read as a list of problems. */
.fl__btn {
  display: inline-flex;
  padding: 0;
  border: 0;
  background: none;
  color: var(--pk-text-muted);
  opacity: 0.55;
  cursor: pointer;
}
.fl__btn:hover,
.fl__btn:focus-visible,
.fl:hover .fl__btn {
  color: var(--pk-accent);
  opacity: 1;
}
.fl__body {
  max-width: 40ch;
  font-size: var(--pk-font-size-xs);
  font-weight: 400;
  line-height: 1.5;
  color: var(--pk-text-secondary);
}
.fl__body :deep(p) {
  margin: 0 0 6px;
}
.fl__body :deep(p:last-child) {
  margin-bottom: 0;
}
</style>
