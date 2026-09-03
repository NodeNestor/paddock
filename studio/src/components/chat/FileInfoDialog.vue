<script setup lang="ts">
// A document's two NON-VIEWER answers in one dialog: what the file says about
// itself, and what the model reads from it.
//
// It exists because lector cannot host them. Every other document surface puts
// these on a tab strip under the pane's own bar, but a PDF renders inside
// lector's viewer and a host row above that toolbar is precisely the second
// row of chrome removed. So lector's toolbar carries one extra button and
// it opens this - one entry with tabs, not two buttons with two windows, which
// is the shape we settled on when the file dialog's detached eye went.
// The tabs are the pane's own, in the pane's order.
//
// `open` is its own prop rather than "file is not null" : the
// details tab needs an attachment id and nothing else, so the dialog opens at
// once and the extraction's bytes arrive behind it. Blocking a window on a
// 22 MB fetch its first tab never reads is a stall for nothing.
import { ref, watch } from 'vue'
import {
  DialogRoot,
  DialogPortal,
  DialogOverlay,
  DialogContent,
  DialogTitle,
  VisuallyHidden,
} from 'reka-ui'
import Icon from '@/components/Icon.vue'
import Tabs from '@/components/ui/Tabs.vue'
import FileMetaPane from './FileMetaPane.vue'
import InsightPane from './InsightPane.vue'

const props = defineProps<{
  open: boolean
  /** Every file the document is made of; the details tab answers per file. */
  parts: { attachmentId?: string; name?: string }[]
  /** Bytes for the extraction preview - null while they are still loading. */
  file: { name: string; bytes: Uint8Array; mime?: string } | null
  /** The byte fetch is not coming (no stored copy). The details tab is
   *  unaffected: it reads the manager, not these bytes. */
  fileError?: string | null
  /** What to call the document in the bar. */
  title?: string
  /** Mirror of the chat's file-details toggle so the preview matches what a
   *  send from this conversation would actually carry. */
  withMeta?: boolean
  /** Model whose server runs the extraction (any running server gives the
   *  same answer; this picks the port). */
  model?: string
}>()
const emit = defineEmits<{ close: [] }>()

const TABS = [
  { value: 'meta', label: 'Metadata' },
  { value: 'model', label: 'Model metadata' },
]
const tab = ref('meta')
// Reopening starts on the details tab again: the button that opens this leads
// with them, and a tab remembered from a different document is a surprise.
watch(
  () => props.open,
  (v) => {
    if (v) tab.value = 'meta'
  },
)

function onOpenChange(v: boolean): void {
  if (!v) emit('close')
}
</script>

<template>
  <DialogRoot :open="open" @update:open="onOpenChange">
    <DialogPortal>
      <DialogOverlay class="pv__overlay" />
      <DialogContent class="pv__content" @escape-key-down="emit('close')">
        <VisuallyHidden as-child>
          <DialogTitle>{{ title || 'Metadata' }}</DialogTitle>
        </VisuallyHidden>
        <div class="pv__bar pv__bar--tabbed">
          <span class="pv__name">{{ title }}</span>
          <span class="pv__spacer" />
          <button class="pv__btn" type="button" aria-label="Close" @click="emit('close')">
            <Icon name="x" :size="16" />
          </button>
        </div>
        <div class="pv__tabs">
          <Tabs v-model="tab" :tabs="TABS" />
        </div>
        <div class="pv__body">
          <FileMetaPane v-show="tab === 'meta'" :parts="parts" />
          <div v-show="tab === 'model'" class="pv__pane">
            <InsightPane v-if="file" :file="file" :with-meta="withMeta" :model="model" />
            <div v-else-if="fileError" class="pv__overlay-msg pv__overlay-msg--err">
              <Icon name="file-text" :size="28" />
              <p>{{ fileError }}</p>
            </div>
            <div v-else class="pv__overlay-msg">
              <Icon name="spinner" :size="22" class="pv__spin" />
              <span>Opening...</span>
            </div>
          </div>
        </div>
      </DialogContent>
    </DialogPortal>
  </DialogRoot>
</template>
