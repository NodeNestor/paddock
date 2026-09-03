// Keep a scroll container pinned to the bottom while content grows (streaming),
// unless the user has scrolled up. ResizeObserver on the content re-pins within
// an animation frame; a scroll listener releases the pin when the user leaves
// the bottom and re-arms it when they return.

import { onBeforeUnmount, onMounted, ref, type Ref } from 'vue'

const BOTTOM_THRESHOLD = 64 // px from the bottom that still counts as "at bottom"

export function useStickyScroll(
  scrollEl: Ref<HTMLElement | null>,
  contentEl: Ref<HTMLElement | null>,
) {
  const stuck = ref(true)
  let ro: ResizeObserver | null = null
  let raf = 0

  function atBottom(): boolean {
    const el = scrollEl.value
    if (!el) return true
    return el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_THRESHOLD
  }

  function pin(): void {
    const el = scrollEl.value
    if (el) el.scrollTop = el.scrollHeight
  }

  function schedulePin(): void {
    if (!stuck.value || raf) return
    raf = requestAnimationFrame(() => {
      raf = 0
      pin()
    })
  }

  function onScroll(): void {
    stuck.value = atBottom()
  }

  /** Re-arm sticky mode and jump to the latest (the "scroll to latest" action). */
  function toBottom(): void {
    stuck.value = true
    pin()
  }

  onMounted(() => {
    const el = scrollEl.value
    if (!el) return
    el.addEventListener('scroll', onScroll, { passive: true })
    ro = new ResizeObserver(() => schedulePin())
    if (contentEl.value) ro.observe(contentEl.value)
    ro.observe(el)
    pin()
  })

  onBeforeUnmount(() => {
    scrollEl.value?.removeEventListener('scroll', onScroll)
    ro?.disconnect()
    if (raf) cancelAnimationFrame(raf)
  })

  return { stuck, pin, toBottom, schedulePin }
}
