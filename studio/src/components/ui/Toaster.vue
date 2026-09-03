<script setup lang="ts">
// The app's one toast outlet (Reka Toast), mounted once in App.vue. Toasts
// announce outcomes - a deploy finishing, a deploy failing - in one line each;
// anything with more to say links to the page that owns the detail. This is
// the Manager's no-scrolling-logs rule: progress lives on
// rows, logs live in Instrument, toasts just tell you it's done.
import {
  ToastDescription,
  ToastProvider,
  ToastRoot,
  ToastTitle,
  ToastViewport,
} from 'reka-ui'
import { useRouter } from 'vue-router'
import { useToastsStore, type Toast } from '@/stores/toasts'
import Icon from '@/components/Icon.vue'

const toasts = useToastsStore()
const router = useRouter()

function onOpen(t: Toast, open: boolean): void {
  if (!open) toasts.dismiss(t.id)
}
function follow(t: Toast): void {
  if (!t.to) return
  void router.push(t.to)
  toasts.dismiss(t.id)
}
function iconFor(t: Toast): string {
  return t.tone === 'good' ? 'check' : t.tone === 'bad' ? 'alert-triangle' : 'server'
}
</script>

<template>
  <ToastProvider :duration="5000" swipe-direction="right">
    <ToastRoot
      v-for="t in toasts.items"
      :key="t.id"
      class="pk-toast"
      :class="[`pk-toast--${t.tone}`, { 'pk-toast--link': t.to }]"
      :duration="t.duration"
      @update:open="(o: boolean) => onOpen(t, o)"
      @click="follow(t)"
    >
      <Icon :name="iconFor(t)" :size="15" class="pk-toast__icon" />
      <div class="pk-toast__body">
        <ToastTitle class="pk-toast__title">{{ t.title }}</ToastTitle>
        <ToastDescription v-if="t.description" class="pk-toast__desc">
          {{ t.description }}
        </ToastDescription>
      </div>
    </ToastRoot>
    <ToastViewport class="pk-toast__viewport" />
  </ToastProvider>
</template>

<!-- Unscoped deliberately (the MenuContent rule): keep every popup surface on
     the same global pk-* recipe regardless of where Reka mounts it. -->
<style>
.pk-toast__viewport {
  position: fixed;
  right: 16px;
  bottom: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
  /* above selects/menus (9800) - a finished deploy outranks an open dropdown */
  z-index: 9900;
  max-width: 380px;
}
.pk-toast {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 11px 14px;
  border: 1px solid var(--pk-border-strong);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-elevated);
  box-shadow: var(--pk-shadow-lg);
  animation: pk-toast-in 0.16s ease-out;
}
.pk-toast--link {
  cursor: pointer;
}
.pk-toast--link:hover {
  border-color: var(--pk-accent);
}
.pk-toast[data-swipe='move'] {
  transform: translateX(var(--reka-toast-swipe-move-x));
}
.pk-toast[data-swipe='end'] {
  animation: pk-toast-out 0.12s ease-in forwards;
}
.pk-toast__icon {
  flex: none;
  margin-top: 1px;
}
.pk-toast--good .pk-toast__icon {
  color: var(--pk-status-success, #4a9);
}
.pk-toast--bad .pk-toast__icon {
  color: var(--pk-text-danger);
}
.pk-toast--info .pk-toast__icon {
  color: var(--pk-text-muted);
}
.pk-toast__title {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.pk-toast__desc {
  margin: 2px 0 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  line-height: 1.4;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}
/* A failure gets room to say WHY. Two lines is right for "11540 is running";
   it is not right for an engine refusal, whose whole value is the reason and
   the fix it names. Measured: a start that could not fit its KV
   returned a complete 600-character explanation, and the two lines that
   survived the clamp were the timestamp and the module path - indistinguishable
   from no message at all, which is exactly how it was reported. */
.pk-toast--bad .pk-toast__desc {
  -webkit-line-clamp: 7;
}
@keyframes pk-toast-in {
  from {
    opacity: 0;
    transform: translateX(12px);
  }
}
@keyframes pk-toast-out {
  to {
    opacity: 0;
    transform: translateX(100%);
  }
}
</style>
