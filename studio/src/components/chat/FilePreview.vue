<script setup lang="ts">
// The file dialog: three ways to see an attachment, as tabs - "Document" (the
// human view: lector for PDFs, scriptor for Word), "Metadata" (everything
// the file says about itself) and "Model metadata" (the extraction the
// prompt actually carries). One chip action; the old detached eye button is
// gone (two adjacent controls could never share the chip's
// height, and the tab makes the honesty panel discoverable).
// This is now the FALLBACK surface, not the front door: a PDF or a .docx on a
// user turn opens in the document pane instead (MessageBubble.openFile), which
// is where both viewers get their full toolset. What still lands here is a
// document on an ASSISTANT turn - no entry in the conversation's document list
// to select - and every format with no in-app viewer at all.
// Types with no in-app viewer open straight on the model tab; their Document
// tab says so and offers the stored original for download.
// Bytes are fetched once here and feed every pane.
import { computed, ref, watch } from 'vue'
import {
  DialogRoot,
  DialogPortal,
  DialogOverlay,
  DialogContent,
  DialogTitle,
  VisuallyHidden,
} from 'reka-ui'
import type { FilePart } from '@/types/chat'
import { attachmentsApi } from '@/lib/api'
import Icon from '@/components/Icon.vue'
import Tabs from '@/components/ui/Tabs.vue'
import PdfPane from './PdfPane.vue'
import DocxPane from './DocxPane.vue'
import FileMetaPane from './FileMetaPane.vue'
import InsightPane from './InsightPane.vue'

const props = defineProps<{
  /** Attachment to open; null closes the dialog. */
  file: FilePart | null
  /** Mirror of the chat's file-details toggle (the model tab shows what a
   *  send from this conversation would actually carry). */
  withMeta?: boolean
  /** Model whose server runs the extraction for the model tab. */
  model?: string
}>()
const emit = defineEmits<{ close: [] }>()

const open = computed(() => !!props.file)
const src = computed(() =>
  props.file?.attachmentId ? attachmentsApi.url(props.file.attachmentId) : '',
)
const kind = computed<'pdf' | 'docx' | 'none'>(() => {
  const n = props.file?.name.toLowerCase() ?? ''
  return n.endsWith('.pdf') ? 'pdf' : n.endsWith('.docx') ? 'docx' : 'none'
})

// The same three the document pane offers, in the same order. This
// dialog is where the formats with no in-app viewer land - a spreadsheet, a
// presentation, a .csv - and those are exactly the files whose details are the
// most of what there is to see.
const TABS = [
  { value: 'doc', label: 'Document' },
  { value: 'meta', label: 'Metadata' },
  { value: 'model', label: 'Model metadata' },
]
const tab = ref('doc')
// The details tab reads the stored blob through the manager, so it takes the
// attachment id rather than the bytes this dialog fetches for the other two.
const metaParts = computed(() =>
  props.file?.attachmentId
    ? [{ attachmentId: props.file.attachmentId, name: props.file.name }]
    : [],
)

const bytes = ref<Uint8Array | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
// Monotonic token so a slow fetch resolving after a close/reopen can't clobber.
let gen = 0

watch(
  () => props.file,
  async (f) => {
    gen++
    bytes.value = null
    error.value = null
    loading.value = false
    if (!f) return
    tab.value = kind.value === 'none' ? 'model' : 'doc'
    if (!src.value) {
      error.value = 'No stored copy of this file. It was sent before originals were kept.'
      return
    }
    const mine = gen
    loading.value = true
    try {
      const resp = await fetch(src.value)
      if (!resp.ok) throw new Error(`fetch failed (${resp.status})`)
      const buf = await resp.arrayBuffer()
      if (mine !== gen) return
      bytes.value = new Uint8Array(buf)
      loading.value = false
    } catch (e) {
      if (mine === gen) {
        error.value = e instanceof Error ? e.message : String(e)
        loading.value = false
      }
    }
  },
)

const insightFile = computed(() =>
  props.file && bytes.value
    ? { name: props.file.name, bytes: bytes.value, mime: props.file.mime || undefined }
    : null,
)
const pdfPane = ref<InstanceType<typeof PdfPane> | null>(null)

function onOpenChange(v: boolean): void {
  if (!v) emit('close')
}
</script>

<template>
  <DialogRoot :open="open" @update:open="onOpenChange">
    <DialogPortal>
      <DialogOverlay class="pv__overlay" />
      <DialogContent
        class="pv__content"
        :class="{ 'pv__content--doc': kind === 'docx' }"
        @escape-key-down="emit('close')"
      >
        <VisuallyHidden as-child>
          <DialogTitle>{{ file?.name || 'File' }}</DialogTitle>
        </VisuallyHidden>
        <div class="pv__bar pv__bar--tabbed">
          <span class="pv__name">{{ file?.name }}</span>
          <span v-if="kind === 'pdf' && tab === 'doc' && pdfPane?.pageCount" class="pv__page">
            {{ pdfPane.currentPage }} / {{ pdfPane.pageCount }}
          </span>
          <span class="pv__spacer" />
          <template v-if="kind === 'pdf' && tab === 'doc'">
            <button class="pv__btn" type="button" aria-label="Zoom out" @click="pdfPane?.zoomOut()">-</button>
            <span class="pv__zoom">{{ pdfPane?.zoomPct ?? 100 }}%</span>
            <button class="pv__btn" type="button" aria-label="Zoom in" @click="pdfPane?.zoomIn()">+</button>
            <button class="pv__btn pv__btn--text" type="button" @click="pdfPane?.fitWidth()">Fit</button>
          </template>
          <a
            v-if="src"
            class="pv__btn"
            :href="src"
            :download="file?.name || 'file'"
            aria-label="Download"
          >
            <Icon name="arrow-down" :size="15" />
          </a>
          <button class="pv__btn" type="button" aria-label="Close" @click="emit('close')">
            <Icon name="x" :size="16" />
          </button>
        </div>
        <div class="pv__tabs">
          <Tabs v-model="tab" :tabs="TABS" />
        </div>
        <div class="pv__body">
          <PdfPane v-if="kind === 'pdf'" v-show="tab === 'doc'" ref="pdfPane" :bytes="bytes" />
          <DocxPane v-else-if="kind === 'docx'" v-show="tab === 'doc'" :bytes="bytes" />
          <div v-else v-show="tab === 'doc'" class="pv__pane">
            <div class="pv__overlay-msg">
              <Icon name="file-text" :size="28" />
              <p>No in-app viewer for this file type.</p>
              <a v-if="src" class="pv__dl" :href="src" :download="file?.name">
                <Icon name="arrow-down" :size="14" /> Download
              </a>
            </div>
          </div>
          <FileMetaPane v-show="tab === 'meta'" :parts="metaParts" />
          <InsightPane
            v-show="tab === 'model'"
            :file="insightFile"
            :with-meta="withMeta"
            :model="model"
          />
          <div v-if="loading" class="pv__overlay-msg">
            <Icon name="spinner" :size="22" class="pv__spin" />
            <span>Opening...</span>
          </div>
          <div v-else-if="error" class="pv__overlay-msg pv__overlay-msg--err">
            <Icon name="file-text" :size="28" />
            <p>{{ error }}</p>
            <a v-if="src" class="pv__dl" :href="src" :download="file?.name">
              <Icon name="arrow-down" :size="14" /> Download instead
            </a>
          </div>
        </div>
      </DialogContent>
    </DialogPortal>
  </DialogRoot>
</template>

<!-- Portalled to <body> by reka; unscoped `pv__`-prefixed classes. This file
     owns the shared pv shell - FileInsight and the panes reuse it. -->
<style>
.pv__overlay {
  position: fixed;
  inset: 0;
  z-index: 9700;
  background: rgba(0, 0, 0, 0.82);
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
  animation: pv-fade 0.15s ease;
}
.pv__content {
  position: fixed;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  z-index: 9701;
  display: flex;
  flex-direction: column;
  width: min(1000px, 94vw);
  height: 90vh;
  background: var(--pk-bg-surface);
  border-radius: var(--pk-radius-lg);
  box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5);
  overflow: hidden;
  outline: none;
}
/* The shared shell caps at 1000px - narrower than scriptor's page stage
   (~1123px for a letter document), which would pin every page to the left
   edge with a permanent horizontal scroll. Word gets a shell wide enough. */
.pv__content--doc {
  width: min(1240px, 94vw);
}
.pv__bar {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 8px;
  height: 46px;
  padding: 0 12px;
  border-bottom: 1px solid var(--pk-border-default);
  background: var(--pk-bg-elevated);
}
/* With the tab strip below, the chrome's one rule is the strip's underline. */
.pv__bar--tabbed {
  border-bottom: none;
}
.pv__tabs {
  flex: 0 0 auto;
  padding: 0 12px;
  background: var(--pk-bg-elevated);
}
.pv__name {
  font-size: var(--pk-font-size-sm);
  font-weight: 500;
  color: var(--pk-text-primary);
  max-width: 40%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.pv__page {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.pv__spacer {
  flex: 1 1 auto;
}
.pv__zoom {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  min-width: 40px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}
.pv__btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 28px;
  height: 28px;
  padding: 0 6px;
  border: none;
  border-radius: var(--pk-radius-md);
  background: transparent;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-base);
  text-decoration: none;
  cursor: pointer;
  transition: background 0.12s ease, color 0.12s ease;
}
.pv__btn:hover {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
.pv__btn--text {
  font-size: var(--pk-font-size-xs);
}
.pv__body {
  position: relative;
  flex: 1 1 0;
  min-height: 0;
  background: var(--pk-bg-inset);
}
.pv__pane {
  position: absolute;
  inset: 0;
}
.pv__canvas {
  position: absolute;
  inset: 0;
  overflow: auto;
}
.pv__overlay-msg {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  background: var(--pk-bg-inset);
}
.pv__spin {
  animation: pv-spin 0.8s linear infinite;
}
.pv__dl {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-primary);
  background: var(--pk-bg-surface);
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  text-decoration: none;
}
@keyframes pv-fade {
  from {
    opacity: 0;
  }
}
@keyframes pv-spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
