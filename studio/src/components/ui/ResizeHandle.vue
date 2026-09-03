<script setup lang="ts">
// Vertical drag handle for resizing a side panel - a 1:1 port of Traverse's
// fg-resizer: a 7px hit area with a centered 1px divider line + a dotted grip,
// both turning accent on hover/drag. The resizer is the divider, so the panel
// it sits against must not draw its own border (else you get a double line).
// Drives a v-model width; `side` sets direction (left panel widens dragging
// right; a right panel is inverted). Raw drag -> we own the document listeners.
import { ref } from 'vue'

const props = withDefaults(
  defineProps<{
    min?: number
    max?: number
    side?: 'left' | 'right'
    /** Drag below this and the panel folds away entirely (width 0) instead of
     *  stopping at `min`. The handle stays where it is, so dragging back out
     *  is how it returns - no toggle button to hunt for. 0 disables it. */
    collapseAt?: number
  }>(),
  { min: 200, max: 560, side: 'left', collapseAt: 0 },
)
const width = defineModel<number>({ required: true })
const active = ref(false)

function start(e: MouseEvent): void {
  e.preventDefault()
  active.value = true
  // An iframe eats the mousemove stream the moment the pointer crosses it, so
  // a drag past a preview panel just stops. Mark the drag on <body> and let
  // CSS make every frame transparent to the pointer for its duration.
  document.body.classList.add('pk-resizing')
  const startX = e.clientX
  const startW = width.value

  const onMove = (ev: MouseEvent) => {
    const delta = props.side === 'right' ? startX - ev.clientX : ev.clientX - startX
    const raw = startW + delta
    width.value =
      props.collapseAt > 0 && raw < props.collapseAt ?
        0
      : Math.max(props.min, Math.min(props.max, raw))
  }
  const onUp = () => {
    active.value = false
    document.body.classList.remove('pk-resizing')
    document.removeEventListener('mousemove', onMove)
    document.removeEventListener('mouseup', onUp)
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
  }
  document.addEventListener('mousemove', onMove)
  document.addEventListener('mouseup', onUp)
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
}

// The window-splitter pattern (WAI-ARIA): a separator that can be moved is a
// focusable widget with arrow keys and a reported value. Not Reka's Splitter -
// that one sizes panels in PERCENT and owns their width, while these panels
// are px, conditionally unmounted, and hand their own width to their internal
// layout (ArtifactPanel reads it as a CSS var). This is the keyboard half of
// what SplitterResizeHandle would have supplied.
const STEP = 16
function nudge(delta: number): void {
  const signed = props.side === 'right' ? -delta : delta
  width.value = Math.max(props.min, Math.min(props.max, width.value + signed))
}
function onKey(e: KeyboardEvent): void {
  if (e.key === 'ArrowLeft') nudge(-STEP)
  else if (e.key === 'ArrowRight') nudge(STEP)
  else if (e.key === 'Home') width.value = props.side === 'right' ? props.max : props.min
  else if (e.key === 'End') width.value = props.side === 'right' ? props.min : props.max
  else if (e.key === 'Enter' && props.collapseAt > 0)
    width.value = width.value > 0 ? 0 : props.min
  else return
  e.preventDefault()
}
</script>

<template>
  <div
    class="pk-resizer"
    :class="{ 'pk-resizer--active': active }"
    role="separator"
    aria-orientation="vertical"
    aria-label="Resize panel"
    tabindex="0"
    :aria-valuenow="Math.round(width)"
    :aria-valuemin="collapseAt > 0 ? 0 : min"
    :aria-valuemax="max"
    :aria-valuetext="width > 0 ? `${Math.round(width)} pixels` : 'hidden'"
    @mousedown="start"
    @keydown="onKey"
  />
</template>

<style scoped>
.pk-resizer {
  flex-shrink: 0;
  width: 7px;
  height: 100%;
  position: relative;
  background: transparent;
  cursor: col-resize;
  display: flex;
  align-items: center;
  justify-content: center;
}
/* the 1px divider line, centered in the hit area */
.pk-resizer::before {
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  left: 3px;
  width: 1px;
  background: var(--pk-border-default);
  transition: background 0.15s ease;
}
.pk-resizer:hover::before,
.pk-resizer:focus-visible::before,
.pk-resizer--active::before {
  background: var(--pk-accent);
}
.pk-resizer:focus-visible {
  outline: none;
}
/* centered dotted grip */
.pk-resizer::after {
  content: '';
  position: relative;
  z-index: 1;
  width: 7px;
  height: 28px;
  border-radius: 3px;
  background-color: var(--pk-border-strong);
  background-image: radial-gradient(circle, var(--pk-text-muted) 0.8px, transparent 0.8px);
  background-size: 3px 5px;
  background-position: center;
  background-repeat: repeat-y;
  transition: background-color 0.15s ease;
}
.pk-resizer:hover::after,
.pk-resizer:focus-visible::after,
.pk-resizer--active::after {
  background-color: var(--pk-accent);
  background-image: radial-gradient(circle, var(--pk-text-inverse) 0.8px, transparent 0.8px);
}
</style>
