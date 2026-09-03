import { defineStore } from 'pinia'
import { ref } from 'vue'

export type ToastTone = 'good' | 'bad' | 'info'

/** One transient notice. Toasts announce OUTCOMES ("11540 is running") -
 *  progress and logs belong to rows and the Instrument pages, never here. */
export interface Toast {
  id: number
  tone: ToastTone
  title: string
  description?: string
  /** where clicking the toast goes (e.g. a failed server's page). */
  to?: { name: string; params?: Record<string, string> }
  /** ms on screen; failures linger longer than good news. */
  duration?: number
}

let seq = 0

export const useToastsStore = defineStore('toasts', () => {
  const items = ref<Toast[]>([])

  function push(t: Omit<Toast, 'id'>): void {
    items.value = [...items.value, { ...t, id: ++seq }]
  }
  function dismiss(id: number): void {
    items.value = items.value.filter((t) => t.id !== id)
  }

  return { items, push, dismiss }
})
