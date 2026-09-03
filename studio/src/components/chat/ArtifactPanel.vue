<script setup lang="ts">
// The artifact side panel. What the model wrote lives here, beside
// the conversation, not inside it - the messages carry the operations and this
// panel carries the thing itself, with every version still reachable.
//
// The panel is only the layout: one ArtifactPane per model that wrote
// something, side by side, in the order the chat shows its lanes - or a single
// tabbed pane when they will not all fit. Two, three and four all work; four
// is what the compare itself allows.
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import ArtifactPane from './ArtifactPane.vue'
import ResizeHandle from '@/components/ui/ResizeHandle.vue'
import { useArtifactsStore } from '@/stores/artifacts'

/** Narrower than this and a pane compares nothing, so it does not count. */
const PANE_MIN = 340

const artifacts = useArtifactsStore()
const root = ref<HTMLElement | null>(null)
const width = ref(0)
let ro: ResizeObserver | null = null
onMounted(() => {
  if (!root.value) return
  ro = new ResizeObserver(([e]) => {
    width.value = e.contentRect.width
    artifacts.paneCapacity = Math.max(1, Math.floor(width.value / PANE_MIN))
  })
  ro.observe(root.value)
})
onBeforeUnmount(() => {
  ro?.disconnect()
  artifacts.paneCapacity = 1
})

// Each pane's share of the width. Dividers move a RATIO, not a pixel count, so
// widening the panel keeps the balance you chose instead of feeding every new
// pixel to the last pane. Equal shares whenever the number of panes changes -
// there is no sensible way to carry a 2-way split into a 3-way one.
const weights = ref<number[]>([])
watch(
  () => artifacts.visible.length,
  (n) => {
    if (n > 0 && weights.value.length !== n) weights.value = Array(n).fill(1 / n)
  },
  { immediate: true },
)

const paneWidth = (i: number): number => Math.round((weights.value[i] ?? 0) * width.value)
/** A divider only ever moves space between the two panes it sits between. */
const pairMax = (i: number): number =>
  Math.max(PANE_MIN, Math.round(((weights.value[i] ?? 0) + (weights.value[i + 1] ?? 0)) * width.value) - PANE_MIN)

function setPaneWidth(i: number, px: number): void {
  const total = width.value
  const a = weights.value[i]
  const b = weights.value[i + 1]
  if (total <= 0 || a === undefined || b === undefined) return
  const pair = a + b
  const floor = PANE_MIN / total
  const next = Math.min(pair - floor, Math.max(floor, px / total))
  weights.value = weights.value.map((w, j) => (j === i ? next : j === i + 1 ? pair - next : w))
}
const paneStyle = (i: number): Record<string, string> => ({ flex: `0 0 ${paneWidth(i)}px` })
</script>

<template>
  <aside ref="root" class="ap" :class="{ 'ap--split': artifacts.split }">
    <template v-if="artifacts.split">
      <template v-for="(g, i) in artifacts.visible" :key="g.model">
        <ResizeHandle
          v-if="i > 0"
          :model-value="paneWidth(i - 1)"
          side="left"
          :min="PANE_MIN"
          :max="pairMax(i - 1)"
          @update:model-value="(px: number) => setPaneWidth(i - 1, px)"
        />
        <ArtifactPane
          class="ap__pane"
          :style="paneStyle(i)"
          :items="g.items"
          :selected="artifacts.selectedIn(g.model)"
          @select="artifacts.show"
        />
      </template>
    </template>
    <ArtifactPane
      v-else
      class="ap__pane"
      :items="artifacts.soleItems"
      :selected="artifacts.soleId"
      @select="artifacts.show"
    />
  </aside>
</template>

<style scoped>
.ap {
  display: flex;
  flex-direction: column;
  width: var(--pk-artifact-width, 420px);
  min-width: 0;
  background: var(--pk-bg-surface);
  overflow: hidden;
}
.ap__pane {
  flex: 1 1 0;
  min-width: 0;
  min-height: 0;
}
/* Side by side. The ResizeHandle between two panes is the divider, so neither
   draws a border - the same rule as the panel's own outer resizer. */
.ap--split {
  flex-direction: row;
}
</style>
