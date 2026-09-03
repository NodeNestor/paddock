<script setup lang="ts">
// "Model metadata": the runner's extraction preview for one attachment,
// shown verbatim - header note, metadata block (Title/Author/dates, photo
// time/camera/GPS) and the extracted content, exactly as the prompt carries
// it. Pane only, and one of two - FileMetaPane is the other half of the same
// question, everything the FILE says rather than everything the PROMPT carries
// Three hosts, all of them tabbing the pair: the document pane,
// the file dialog, and the dialog lector's toolbar opens. Lane refusals
// (encrypted PDF, binary bytes) show as-is: they are what a send would do.
import { computed, ref, watch } from 'vue'
import { useModelsStore } from '@/stores/models'
import Icon from '@/components/Icon.vue'

const props = defineProps<{
  /** Attachment to preview; null shows nothing (the fetch waits for bytes). */
  file: { name: string; bytes: Uint8Array; mime?: string } | null
  /** Mirror of the chat's file-details toggle so the preview matches what a
   *  send from this conversation would actually carry. */
  withMeta?: boolean
  /** Model whose server runs the extraction (any running server gives the
   *  same answer; this picks the port). */
  model?: string
}>()

const models = useModelsStore()
const text = ref('')
const kind = ref('')
/** What this file adds to the SYSTEM turn - today only the map capability a
 *  geotagged photo earns. Shown separately because it lands in a different
 *  turn: the pane is called "what the model reads", and it was reading half
 *  ("but why isn't what we inject seen under what the model
 *  reads?"). */
const systemNote = ref('')
const loading = ref(false)
const error = ref<string | null>(null)
let gen = 0

function toB64(u8: Uint8Array): string {
  let s = ''
  const CHUNK = 0x8000
  for (let i = 0; i < u8.length; i += CHUNK) {
    s += String.fromCharCode(...u8.subarray(i, i + CHUNK))
  }
  return btoa(s)
}

// The url is part of the watch source: the runner list revalidates in the
// background, so a server started after this pane opened flips the "no
// running server" notice into the real extraction on its own.
watch(
  [() => props.file, () => models.extractUrl(props.model)],
  async ([f, url]) => {
    if (!f) {
      gen++
      return
    }
    const mine = ++gen
    text.value = ''
    kind.value = ''
    systemNote.value = ''
    error.value = null
    if (!url) {
      // a cloud model has no extraction preview to show: the provider reads
      // the file itself, and this pane describes the LOCAL injection only
      error.value = models.models.find((m) => m.id === props.model)?.cloud
        ? 'This model runs in the cloud. The provider reads the file itself, so there is no local extraction to preview.'
        : 'No running server to ask. Start this model to preview its extraction.'
      return
    }
    loading.value = true
    try {
      const body: Record<string, unknown> = {
        filename: f.name,
        data: `data:${f.mime ?? ''};base64,${toB64(f.bytes)}`,
      }
      if (props.withMeta === false) body.file_metadata = 'off'
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      })
      const j = await res.json()
      if (mine !== gen) return
      if (!res.ok) {
        // the lane's honest refusal is the answer: show it as the content
        text.value = j?.error?.message ?? `extraction failed (${res.status})`
        kind.value = 'refusal'
      } else {
        text.value = j.text ?? ''
        kind.value = j.kind ?? ''
        systemNote.value = j.system ?? ''
      }
      loading.value = false
    } catch (e) {
      if (mine === gen) {
        error.value = e instanceof Error ? e.message : String(e)
        loading.value = false
      }
    }
  },
  { immediate: true },
)

const note = computed(() => {
  switch (kind.value) {
    case 'photo':
      return text.value
        ? 'This line is injected next to the image; the pixels go to the model separately.'
        : // Not "this image has no metadata" - that was false on any file whose
          // fields the prompt line does not curate, and the Metadata tab sits
          // one across proving it (a GIMP export with 38 fields, none of them a
          // capture time, camera or location, read exactly that way).
          'No capture time, camera or location - the three things the prompt carries. The photo goes to the model as pixels only; the Metadata tab lists everything the file does carry.'
    case 'refusal':
      return 'Sending this file would be refused with this message.'
    case 'pdf':
      return 'On a vision model with PDF rendering, pages go as images instead; the metadata block is the same.'
    default:
      return 'This text replaces the attachment in the prompt.'
  }
})
</script>

<template>
  <div class="pv__pane">
    <div class="pv__canvas fi__scroll">
      <p v-if="!loading && !error && file" class="fi__note">{{ note }}</p>
      <pre v-if="text" class="fi__text">{{ text }}</pre>
      <template v-if="systemNote">
        <p class="fi__note fi__note--sys">
          This file also adds to the system prompt, which is a different turn:
        </p>
        <pre class="fi__text">{{ systemNote }}</pre>
      </template>
    </div>
    <div v-if="loading" class="pv__overlay-msg">
      <Icon name="spinner" :size="22" class="pv__spin" />
      <span>Extracting...</span>
    </div>
    <div v-else-if="error" class="pv__overlay-msg pv__overlay-msg--err">
      <Icon name="file-text" :size="28" />
      <p>{{ error }}</p>
    </div>
  </div>
</template>

<style>
.fi__scroll {
  padding: 14px 18px;
}
.fi__note--sys {
  margin-top: 14px;
}
.fi__note {
  margin: 0 0 10px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.fi__text {
  margin: 0;
  padding: 12px 14px;
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  color: var(--pk-text-primary);
  font-family: var(--pk-font-mono, ui-monospace, monospace);
  font-size: 12px;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
