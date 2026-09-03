<script setup lang="ts">
// A Monaco editor as a v-model'd text box. Monaco itself is fetched on first
// use (see lib/monaco.ts) so nothing but a source view pays for it - but that
// is 3.8 MB to parse, which is a visible wait on the click that asked for it
// Two things follow: callers should warm the chunk while
// the browser is idle, and the `placeholder` slot covers the editor until it
// is live, so the caller can show the same text read-only instead of a spinner.
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type * as Monaco from 'monaco-editor'
import { useSettingsStore } from '@/stores/settings'

const props = withDefaults(
  defineProps<{
    language?: string
    readonly?: boolean
    /** Shown in place when someone types into a read-only editor - otherwise
     *  the keystrokes just vanish and it reads as a broken box. */
    readonlyMessage?: string
    minimap?: boolean
    /** Wire Ctrl/Cmd+Enter to a `run` emit. Opt-in because it shadows
     *  monaco's own insert-line-below, which a plain source editor keeps. */
    runnable?: boolean
  }>(),
  { language: '', readonly: false, readonlyMessage: '', minimap: false, runnable: false },
)
const emit = defineEmits<{ save: []; run: [] }>()
const text = defineModel<string>({ required: true })

const settings = useSettingsStore()
const host = ref<HTMLElement | null>(null)
const loading = ref(true)
const error = ref('')
let editor: Monaco.editor.IStandaloneCodeEditor | null = null
let lib: typeof import('@/lib/monaco') | null = null
// Set while we push a prop change into the editor, so the resulting change
// event does not bounce straight back out as a user edit.
let echoing = false

onMounted(async () => {
  try {
    await build()
  } catch (e) {
    // Without this the box sat on "loading" for good - the one failure mode a
    // lazily-loaded editor must never have.
    console.error('monaco failed to load', e)
    error.value = e instanceof Error ? e.message : String(e)
  }
})

async function build(): Promise<void> {
  lib = await import('@/lib/monaco')
  const monaco = await lib.loadMonaco(settings.theme === 'dark')
  if (!host.value) return
  editor = monaco.editor.create(host.value, {
    value: text.value,
    language: lib.resolveLanguage(props.language),
    readOnly: props.readonly,
    ...(props.readonlyMessage ? { readOnlyMessage: { value: props.readonlyMessage } } : {}),
    // Monaco watches its own container, which is what keeps it correct through
    // a panel resize without us wiring a second ResizeObserver.
    automaticLayout: true,
    minimap: { enabled: props.minimap },
    fontFamily: getComputedStyle(document.documentElement)
      .getPropertyValue('--pk-font-mono')
      .trim(),
    fontSize: 12,
    lineHeight: 18,
    scrollBeyondLastLine: false,
    wordWrap: 'on',
    tabSize: 2,
    padding: { top: 8, bottom: 8 },
    lineNumbersMinChars: 3,
    overviewRulerLanes: 0,
    renderLineHighlight: 'none',
    scrollbar: { verticalScrollbarSize: 10, horizontalScrollbarSize: 10 },
  })
  editor.onDidChangeModelContent(() => {
    if (echoing || !editor) return
    text.value = editor.getValue()
  })
  // Ctrl/Cmd+S is what anyone editing text will reach for; without it the
  // browser's own save dialog answers instead.
  editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => emit('save'))
  if (props.runnable) {
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => emit('run'))
  }
  loading.value = false
}

onBeforeUnmount(() => {
  editor?.getModel()?.dispose()
  editor?.dispose()
  editor = null
})

watch(text, (v) => {
  if (!editor || v === editor.getValue()) return
  echoing = true
  editor.setValue(v)
  echoing = false
})
watch(
  () => props.language,
  (l) => {
    const model = editor?.getModel()
    if (model && lib) lib.monaco.editor.setModelLanguage(model, lib.resolveLanguage(l))
  },
)
watch(
  () => props.readonly,
  (r) => editor?.updateOptions({ readOnly: r }),
)
watch(
  () => props.readonlyMessage,
  (m) => editor?.updateOptions({ readOnlyMessage: m ? { value: m } : undefined }),
)
watch(
  () => settings.theme,
  (t) => lib?.applyTheme(t === 'dark'),
)
</script>

<template>
  <div class="ce">
    <div ref="host" class="ce__host" />
    <!-- Covers the editor until it is live. The caller passes the same text
         read-only, so opening a source view shows the source immediately and
         the editor swaps in underneath when its chunk lands. -->
    <div v-if="loading" class="ce__wait">
      <p v-if="error" class="ce__err">The editor did not load: {{ error }}</p>
      <slot name="placeholder">Loading the editor...</slot>
    </div>
  </div>
</template>

<style scoped>
.ce {
  position: relative;
  height: 100%;
  min-height: 0;
}
.ce__host {
  height: 100%;
}
.ce__wait {
  position: absolute;
  inset: 0;
  overflow: auto;
  background: var(--pk-bg-surface);
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.ce__err {
  margin: 0;
  padding: 6px 10px;
  background: var(--pk-status-error-subtle);
  color: var(--pk-status-error);
  font-size: var(--pk-font-size-xs);
}
</style>
