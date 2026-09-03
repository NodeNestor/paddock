<script setup lang="ts">
// The shared centered-dialog shell, built on Reka UI's Dialog so it carries
// focus-trap, scroll-lock, Escape, return-focus and the aria wiring
// (labelledby/modal) for free - never hand-roll a modal (see the reka-ui reuse
// rule). Body content is the default slot; footer buttons go in #footer; a
// fully custom header goes in #header. The header/close icons come from the
// inline Icon registry (string names), keeping this icon-library-agnostic.
import { computed } from 'vue'
import {
  DialogRoot,
  DialogPortal,
  DialogOverlay,
  DialogContent,
  DialogTitle,
  DialogClose,
  VisuallyHidden,
} from 'reka-ui'
import Icon from '@/components/Icon.vue'

// Optionals are explicitly `| undefined` so callers can bind optional-chained
// values under exactOptionalPropertyTypes.
const props = withDefaults(
  defineProps<{
    open: boolean
    title?: string | undefined
    /** Optional leading header icon (an Icon-registry name). */
    icon?: string | undefined
    /** While busy, backdrop click / Escape / the close button are inert. */
    busy?: boolean | undefined
    size?: 'sm' | 'md' | 'lg' | undefined
    /** Caller controls dismissal: backdrop + Escape do not close. */
    persistent?: boolean | undefined
    /** Hide the header close (X). */
    hideClose?: boolean | undefined
    /** `alertdialog` for destructive / must-acknowledge dialogs. */
    role?: 'dialog' | 'alertdialog' | undefined
    /** Tone the header icon as destructive (red) instead of the accent. */
    danger?: boolean | undefined
  }>(),
  {
    busy: false,
    size: 'md',
    persistent: false,
    hideClose: false,
    role: 'dialog',
    danger: false,
  },
)

const emit = defineEmits<{ (e: 'close'): void }>()

// Dismissal is locked while busy or persistent. Reka requests close via
// `update:open(false)` (Escape, outside-click, the X); we relay it as `close`
// only when allowed, and `open` stays controlled by the parent prop - so a
// blocked request simply leaves the dialog up, no preventDefault gymnastics.
const locked = computed(() => props.busy || props.persistent)
function onUpdateOpen(next: boolean): void {
  if (!next && !locked.value) emit('close')
}
function guard(e: Event): void {
  if (locked.value) e.preventDefault()
}
// Reka's AlertDialogContent is literally DialogContent with role="alertdialog"
// and pointerDownOutside/interactOutside prevented - so rather than duplicate
// this whole template for the AlertDialog* aliases, reproduce the one
// behavioural difference. An alert must be answered, not dismissed by a stray
// click on the backdrop; Escape still works, exactly as Reka's own does.
function guardOutside(e: Event): void {
  if (locked.value || props.role === 'alertdialog') e.preventDefault()
}
</script>

<template>
  <DialogRoot :open="open" @update:open="onUpdateOpen">
    <DialogPortal>
      <DialogOverlay class="pk-dialog__overlay" />
      <DialogContent
        :class="['pk-dialog', `pk-dialog--${size}`]"
        :role="role"
        :aria-describedby="undefined"
        @escape-key-down="guard"
        @pointer-down-outside="guardOutside"
        @interact-outside="guardOutside"
      >
        <header v-if="$slots.header || title" class="pk-dialog__head">
          <slot name="header">
            <span
              v-if="icon"
              :class="['pk-dialog__icon', danger && 'pk-dialog__icon--danger']"
            >
              <Icon :name="icon" :size="20" />
            </span>
            <DialogTitle class="pk-dialog__title">{{ title }}</DialogTitle>
            <DialogClose
              v-if="!hideClose"
              class="pk-dialog__close"
              aria-label="Close"
              :disabled="busy"
            >
              <Icon name="x" :size="16" />
            </DialogClose>
          </slot>
        </header>
        <!-- Reka warns without a title; keep one for a11y even when the caller
             renders its own header chrome. -->
        <VisuallyHidden v-else as-child>
          <DialogTitle>Dialog</DialogTitle>
        </VisuallyHidden>

        <slot />

        <footer v-if="$slots.footer" class="pk-dialog__footer">
          <slot name="footer" />
        </footer>
      </DialogContent>
    </DialogPortal>
  </DialogRoot>
</template>

<!-- Unscoped: the overlay + content are portalled to <body> by Reka, so a
     scoped style hash would never reach them. Classes are `pk-dialog`-prefixed. -->
<style>
/* Portalled to <body>; tokens resolve off :root. The data-state hooks are
   Reka's (open/closed) - we drive the enter/leave fades off them so there's no
   JS transition wrapper. */
.pk-dialog__overlay {
  position: fixed;
  inset: 0;
  background: var(--pk-bg-overlay);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  z-index: 9700;
}
.pk-dialog__overlay[data-state='open'] {
  animation: pk-dialog-fade 0.18s ease;
}
/* Reka keeps the node mounted while a [data-state="closed"] animation runs. */
.pk-dialog__overlay[data-state='closed'] {
  animation: pk-dialog-fade 0.18s ease reverse;
}

.pk-dialog {
  position: fixed;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  z-index: 9701;
  box-sizing: border-box;
  background: var(--pk-bg-elevated);
  border: 1px solid var(--pk-border-strong);
  border-radius: var(--pk-radius-xl);
  box-shadow: var(--pk-shadow-xl);
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  width: min(460px, 92vw);
  max-height: calc(100dvh - 48px);
  overflow-y: auto;
}
.pk-dialog--sm {
  width: min(400px, 92vw);
}
.pk-dialog--md {
  width: min(460px, 92vw);
}
.pk-dialog--lg {
  width: min(640px, 94vw);
}
.pk-dialog[data-state='open'] {
  animation: pk-dialog-in 0.22s ease;
}
.pk-dialog[data-state='closed'] {
  animation: pk-dialog-in 0.18s ease reverse;
}

.pk-dialog__head {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 10px;
}
.pk-dialog__icon {
  display: inline-flex;
  flex-shrink: 0;
  color: var(--pk-accent);
}
.pk-dialog__icon--danger {
  color: var(--pk-status-error);
}
.pk-dialog__title {
  margin: 0;
  grid-column: 2;
  font-size: var(--pk-font-size-lg);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.pk-dialog__close {
  grid-column: 3;
  display: inline-flex;
  background: none;
  border: 0;
  color: var(--pk-text-muted);
  cursor: pointer;
  padding: 4px;
  border-radius: var(--pk-radius-md);
  line-height: 0;
  transition: background 0.12s ease, color 0.12s ease;
}
.pk-dialog__close:hover {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
.pk-dialog__close:disabled {
  opacity: 0.5;
  cursor: default;
}

.pk-dialog__footer {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  margin-top: 2px;
}

@keyframes pk-dialog-fade {
  from {
    opacity: 0;
  }
}
@keyframes pk-dialog-in {
  from {
    opacity: 0;
    transform: translate(-50%, calc(-50% + 10px)) scale(0.97);
  }
}
@media (prefers-reduced-motion: reduce) {
  .pk-dialog[data-state],
  .pk-dialog__overlay[data-state] {
    animation: none;
  }
}
</style>
