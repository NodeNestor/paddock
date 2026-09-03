<script setup lang="ts">
// PDF pane of the tabbed file preview: a headless lector LectorPane driven by
// the shared pdfium engine (lib/pdf.ts). Read-only. The host dialog's bar owns
// the controls, so page/zoom state and the zoom actions are EXPOSED rather
// than rendered here; bytes arrive from the host (fetched once for all tabs).
import { onBeforeUnmount, ref, watch } from 'vue'
import {
  LectorPane,
  type DocumentCapability,
  type DocumentId,
  type ZoomCapability,
} from '@truespar/lector-core'
import '@truespar/lector-core/css/tokens.css'
import '@truespar/lector-core/css/base.css'
import { pdfEngine } from '@/lib/pdf'
import Icon from '@/components/Icon.vue'

const props = defineProps<{ bytes: Uint8Array | null }>()

const containerEl = ref<HTMLElement | null>(null)
const loading = ref(true)
const error = ref<string | null>(null)
const pageCount = ref(0)
const currentPage = ref(1)
const zoomPct = ref(100)

let pane: LectorPane | null = null
let zoom: ZoomCapability | null = null
let docId: DocumentId | null = null
const unsubs: Array<() => void> = []
// Monotonic token so a slow load resolving after teardown can't clobber.
let gen = 0

async function load(buf: ArrayBuffer): Promise<void> {
  const mine = ++gen
  loading.value = true
  error.value = null
  pageCount.value = 0
  currentPage.value = 1
  try {
    const engine = await pdfEngine()
    if (mine !== gen) return
    const doc = engine.plugins.get<DocumentCapability>('document')
    const handle = await doc.load(buf)
    if (mine !== gen) {
      try {
        await doc.close(handle.id)
      } catch {
        /* ignore */
      }
      return
    }
    docId = handle.id
    pageCount.value = handle.pageCount
    if (containerEl.value) {
      pane = new LectorPane({ engine, container: containerEl.value })
      unsubs.push(
        pane.viewport.visiblePages.subscribe((pages) => {
          if (pages.length) currentPage.value = pages[0]! + 1
        }),
      )
    }
    zoom = engine.plugins.get<ZoomCapability>('zoom')
    unsubs.push(zoom.level.subscribe((v) => (zoomPct.value = Math.round(v * 100))))
    try {
      zoom.fitWidth()
    } catch {
      /* keep default scale */
    }
    loading.value = false
  } catch (e) {
    if (mine === gen) {
      error.value = e instanceof Error ? e.message : String(e)
      loading.value = false
    }
  }
}

async function teardown(): Promise<void> {
  gen++
  for (const u of unsubs) u()
  unsubs.length = 0
  pane?.destroy()
  pane = null
  zoom = null
  // Free the worker's copy of this document; the shared engine lives on.
  if (docId) {
    try {
      const engine = await pdfEngine()
      await engine.plugins.get<DocumentCapability>('document').close(docId)
    } catch {
      /* ignore */
    }
    docId = null
  }
}

// The pane lives exactly one dialog-open: bytes go null -> value once, and the
// close unmounts us (reka drops DialogContent), so unmount is the teardown.
watch(
  () => props.bytes,
  (b) => {
    if (b) void load(b.buffer as ArrayBuffer)
  },
  { immediate: true },
)
onBeforeUnmount(() => void teardown())

function zoomOut(): void {
  zoom?.zoomOut()
}
function zoomIn(): void {
  zoom?.zoomIn()
}
function fitWidth(): void {
  zoom?.fitWidth()
}
defineExpose({ zoomIn, zoomOut, fitWidth, pageCount, currentPage, zoomPct })
</script>

<template>
  <div class="pv__pane">
    <div ref="containerEl" class="pv__canvas"></div>
    <div v-if="loading" class="pv__overlay-msg">
      <Icon name="spinner" :size="22" class="pv__spin" />
      <span>Rendering...</span>
    </div>
    <div v-else-if="error" class="pv__overlay-msg pv__overlay-msg--err">
      <Icon name="file-text" :size="28" />
      <p>Could not render this PDF.</p>
    </div>
  </div>
</template>
