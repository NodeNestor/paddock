<script setup lang="ts">
// The one text input (there is no Reka primitive for plain text entry - this
// wrapper exists so every text field shares one look and the raw <input>
// never appears in feature components). v-model is a plain string.
//
// `reveal` (password fields only) adds an eye toggle: a pasted secret the
// user cannot re-read is a secret they cannot verify.
import { ref } from 'vue'
import Icon from '@/components/Icon.vue'

const model = defineModel<string>({ required: true })
const props = defineProps<{
  placeholder?: string
  disabled?: boolean
  /** stretch to the container (default: size to content). */
  block?: boolean
  type?: 'text' | 'password'
  /** show/hide toggle for password fields. */
  reveal?: boolean
}>()
const shown = ref(false)
</script>

<template>
  <div
    v-if="props.type === 'password' && props.reveal"
    class="pk-input-wrap"
    :class="{ 'pk-input-wrap--block': block }"
  >
    <input
      v-model="model"
      class="pk-input pk-input--eyed"
      :class="{ 'pk-input--block': block }"
      :type="shown ? 'text' : 'password'"
      :placeholder="placeholder"
      :disabled="disabled"
      spellcheck="false"
      autocomplete="off"
    />
    <button
      type="button"
      class="pk-input-eye"
      :aria-label="shown ? 'Hide value' : 'Show value'"
      :aria-pressed="shown"
      :disabled="disabled"
      @click="shown = !shown"
    >
      <Icon :name="shown ? 'eye-off' : 'eye'" :size="15" />
    </button>
  </div>
  <input
    v-else
    v-model="model"
    class="pk-input"
    :class="{ 'pk-input--block': block }"
    :type="type ?? 'text'"
    :placeholder="placeholder"
    :disabled="disabled"
    spellcheck="false"
    autocomplete="off"
  />
</template>

<style scoped>
.pk-input {
  min-width: 220px;
  padding: 7px 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  color: var(--pk-text-primary);
  font: inherit;
  font-size: var(--pk-font-size-sm);
  outline: none;
}
.pk-input--block {
  width: 100%;
  max-width: 440px;
  /* block = fill the CONTAINER: the base min-width must not overflow a
     narrow flex column (it made neighboring fields overlap) */
  min-width: 0;
}
.pk-input:focus {
  border-color: var(--pk-accent);
}
.pk-input:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.pk-input::placeholder {
  color: var(--pk-text-muted);
}

.pk-input-wrap {
  position: relative;
  display: inline-block;
}
.pk-input-wrap--block {
  display: block;
  width: 100%;
  max-width: 440px;
}
/* room for the eye so a long key never runs under it */
.pk-input--eyed {
  padding-right: 34px;
}
.pk-input-eye {
  position: absolute;
  top: 50%;
  right: 6px;
  transform: translateY(-50%);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border: none;
  border-radius: var(--pk-radius-sm);
  background: none;
  color: var(--pk-text-muted);
  cursor: pointer;
}
.pk-input-eye:hover {
  color: var(--pk-text-primary);
  background: var(--pk-bg-hover);
}
</style>
