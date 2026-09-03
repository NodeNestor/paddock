<script setup lang="ts">
// The document COLUMN: in document mode the main area is
// divided into columns and the first one is the document. PDFs render in
// lector's full viewer  - its toolbar, search, thumbnail
// sidebar and DOCUMENT TABS: every PDF in the conversation opens in the one
// viewer through its document manager, and lector's own tab bar switches
// them. The pane's strip survives only for image documents, which the PDF
// viewer cannot show. The viewer chrome is trimmed for the chat column: no
// file menu (open/print/security), no annotation modes - thumbnails,
// toolbar and tabs are what matter here.
//
// WORD documents render here too, through scriptor. They
// used to be the one format still opening in a modal, so "open the document"
// meant two different things depending on the file you clicked. Same bargain
// as lector's: the pane is the entry point, the dialog is only for formats
// nothing in-app can render. scriptor ships no toolbar of its own (its
// `Viewer` is bare pages), so the pane draws one - the same fold / zoom /
// download set lector gets, in the same place.
//
// PHOTOS finish the migration. A picture in any chat now
// opens here too, which was the last format still answering a click with a
// dialog - two of them, in fact, stacked on one tile: a lightbox for the
// pixels and a second window behind an eye button for "Model metadata".
// Both are gone. The pane already had the zoom the lightbox never did, and the
// honesty panel becomes a TAB beside the picture rather than a window over it.
//
// FILE DETAILS is the second tab of that kind:
// everything the file says about itself, which on a photo is 121 fields
// against the 3 the prompt takes. It also settles where the strip goes. The
// strip used to go only to documents with pictures, because a lone Word document
// had exactly one secondary view and the bar's eye button covered it; with two
// there is nothing left of that argument, so every document the pane draws its
// own bar for now carries a strip. lector stays the exception it always was -
// a host row above its toolbar is a second row of chrome - so
// its documents keep one toolbar button that opens both views in a dialog.
//
// SELECTION = TARGET, synced both ways with lector: a lector tab click moves
// the conversation's activeDocId (the composer chips follow), and a newly
// sent document activates its lector tab. Never by scroll.
//
// The OCR overlay rides lector's host-overlay contract (ui:page-mounted /
// ui:page-unmounted engine events): lector hands us one layer per live page
// element, covering the page's content box exactly, and the boxes teleport
// into it - percentage geometry, so lector's zoom holds for free.
import { copyText } from '@/lib/clipboard'
import { computed, nextTick, onBeforeUnmount, reactive, ref, shallowRef, watch } from 'vue'
import { LectorPdfViewer } from '@truespar/lector-vue'
import { ScriptorDoc, type TrackDisplay } from '@truespar/scriptor-vue'
import { DEFAULT_UI_SCHEMA } from '@truespar/lector-core'
import type {
  DocumentCapability,
  DocumentManagerCapability,
  LectorEngine,
  PageMountedEvent,
} from '@truespar/lector-core'
import '@truespar/lector-core/css/tokens.css'
import '@truespar/lector-core/css/base.css'
import { useChatStore } from '@/stores/chat'
import { useSettingsStore } from '@/stores/settings'
import { docContext, docContexts, pageRangeBounds } from '@/lib/docrun'
import { pageRegionBoxes } from '@/lib/ocr'
import { attachmentsApi } from '@/lib/api'
import { pdfViewerEngine } from '@/lib/pdf'
import DocumentPages from './DocumentPages.vue'
import FileInfoDialog from './FileInfoDialog.vue'
import FileMetaPane from './FileMetaPane.vue'
import ForensicsPane from './ForensicsPane.vue'
import InsightPane from './InsightPane.vue'
import Icon from '@/components/Icon.vue'
import Menu from '@/components/ui/Menu.vue'
import MenuContent from '@/components/ui/MenuContent.vue'
import MenuItem from '@/components/ui/MenuItem.vue'
import MenuTrigger from '@/components/ui/MenuTrigger.vue'
import Popover from '@/components/ui/Popover.vue'
import Tabs from '@/components/ui/Tabs.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

const chat = useChatStore()
const settings = useSettingsStore()
const emit = defineEmits<{ fold: [] }>()
const all = computed(() => docContexts(chat.active))
const ctx = computed(() => docContext(chat.active))
const runPages = computed(() => ctx.value?.run?.docRun?.pages)
const regions = computed(() => ctx.value?.run?.ocr?.regions)
// The pane's document switcher - one strip for every document the
// conversation holds, PDFs and images alike.
//
// lector's own tab bar used to cover PDF↔PDF while this strip covered images,
// so which control you got - and what it looked like - depended on the file
// type. Worse, lector's tabs carry a close button, and "close" has no meaning
// here: the document rode in on a turn that was already sent, so it stays in
// `all` either way. Closing one just left `lectorDocBySource` pointing at a
// document the viewer no longer held, and `openMissingDocs` skips anything
// already mapped - so the pane went blank for that document until you left the
// conversation and came back.
//
// The viewer is told `:document-tabs="false"`, and switching still runs the
// way it always did: select() -> activeDocId -> syncActiveToLector() ->
// setActive(). The stranding is not fixed so much as made unrepresentable -
// there is no close gesture left anywhere.

/// A tab's label, and with one document open the only place the pane says
/// which file you are looking at - which is why the strip now shows even for a
/// single one ("we don't have the filename readily available
/// in the sidepanel"). A turn that carried several files says so: the lead
/// name plus how many more, because "holiday.jpg" alone would be a lie about a
/// tab holding three photos.
function docName(
  c: { pdf?: { name?: string }; docx?: { name?: string }; images: { name?: string }[] },
  i: number,
): string {
  const lead = c.pdf?.name ?? c.docx?.name ?? c.images[0]?.name ?? `Document ${i + 1}`
  const more = (c.pdf ? 1 : 0) + (c.docx ? 1 : 0) + c.images.length - 1
  return more > 0 ? `${lead} +${more}` : lead
}

/** The switcher's glyph. A photo is a document here now, but it is not a
 *  page-shaped one, and a strip of identical file marks says nothing about
 *  which tab holds the picture. */
function docIcon(c: { pdf?: unknown; docx?: unknown }): string {
  return c.pdf || c.docx ? 'file-text' : 'image'
}

// The selected document's name, for the dialog's bar. `ctx` and `all` come
// from two separate docContexts() builds, so they share no object identity -
// the index has to be found by source id, never by indexOf.
const ctxName = computed(() => {
  const c = ctx.value
  if (!c) return ''
  return docName(c, Math.max(0, all.value.findIndex((x) => x.source.id === c.source.id)))
})

function select(id: string): void {
  const c = chat.active
  if (!c || c.activeDocId === id) return
  c.activeDocId = id
  chat.persist(c)
}

// ── the lector viewer (PDF documents) ───────────────────────────────────────
// `hasPdf` MOUNTS the viewer, `showViewer` SHOWS it. Those have to be two
// different questions once a conversation can hold a Word document beside a
// PDF: selecting the .docx would otherwise unmount LectorPdfViewer, and on the
// way back `@ready` would fire a second time - re-subscribing every listener
// against a `lectorDocBySource` map that still says every document is open, so
// `openMissingDocs` would skip them all and the viewer would come back blank.
//
// A PDF that rode in alone is lector's. One that arrived with a picture beside
// it in the same turn is not: a turn is one document to
// `docContexts`, so the pane can show either the PDF or the pictures, never
// both - and before images had a pane at all, the pictures simply vanished and
// the lightbox in the thread covered for them. With the lightbox gone that
// would be a photo with nowhere to be seen, so the whole turn goes to the page
// stack instead, which renders the pictures AND the rasterized PDF pages in one
// scroll. The cost is lector's toolbar/search/thumbnails for that one turn; the
// alternative is losing an attachment.
function usesLector(c: { pdf?: unknown; images: unknown[] }): boolean {
  return !!c.pdf && c.images.length === 0
}
const hasPdf = computed(() => all.value.some(usesLector))
const showViewer = computed(() => !!ctx.value && usesLector(ctx.value))
const viewerEngine = shallowRef<LectorEngine>()
const viewerReady = ref(false)

// ── the views a NON-lector document offers, as tabs ─────────────────────────
// A photo has no viewer chrome of its own, so its honesty panel became a tab
// beside the pixels rather than a dialog over them. Metadata
// is the second tab of that kind, and it is what settles where the
// STRIP goes: it used to go only to documents with pictures, on the argument
// that a lone Word document had exactly one secondary view and the bar's eye
// button covered it. With two there is no such argument left - two eyes in a
// row would be worse than a strip - so every document the pane draws its own
// bar for gets one. lector is the exception it always was: its documents keep
// a single toolbar button that opens both views in a dialog, because a host
// row above that toolbar is a second row of chrome.
//   'docx'  scriptor's pages
//   'stack' DocumentPages - the pictures, plus rasterized PDF pages when a
//           turn carried both
//   'meta'  Metadata - what the FILE says about itself
//   'model' Model - the extraction the model is handed (labelled just
//           "Model" in the tab row: every sibling is one word, and the panel
//           itself says what it is)
const views = computed(() => {
  const c = ctx.value
  if (!c) return []
  const out: { value: string; label: string }[] = []
  if (c.docx) out.push({ value: 'docx', label: 'Document' })
  if (c.pdf || c.images.length) {
    out.push({
      value: 'stack',
      label: c.pdf ? 'Document' : c.images.length > 1 ? 'Images' : 'Image',
    })
  }
  out.push({ value: 'meta', label: 'Metadata' })
  // Forensics only applies to images and PDFs (the pass reads raster/document
  // bytes) - a docx-only document has no forensic view.
  if (c.pdf || c.images.length) out.push({ value: 'forensics', label: 'Forensics' })
  out.push({ value: 'model', label: 'Model' })
  return out
})
const view = ref('stack')
// Zoom belongs to whatever is being rendered; the two panels have nothing to
// zoom, and a control that does nothing is worse than one that is not there.
const showZoom = computed(() => view.value === 'docx' || view.value === 'stack')
// The details panel asks the manager the moment it mounts, so it mounts when
// the tab is first opened and not before - the same rule the honesty panel
// follows, for the same reason: a pane nobody asks reads nothing. Once opened
// it STAYS mounted, so flipping back to the pictures and returning does not
// re-ask; a different document starts over, because it is a different file.
const metaSeen = ref(false)
// Same lazy-mount rule for the forensics pane: it fetches on mount, so it mounts
// when the tab is first opened and stays mounted until a different document.
const forensicsSeen = ref(false)
watch(view, (v) => {
  if (v === 'meta') metaSeen.value = true
  if (v === 'forensics') forensicsSeen.value = true
})
// A different document starts on its own first view.
watch(
  () => ctx.value?.source.id,
  () => {
    view.value = views.value[0]?.value ?? 'stack'
  },
  { immediate: true },
)
watch(
  hasPdf,
  (want) => {
    if (want && !viewerEngine.value) {
      void pdfViewerEngine().then((e) => (viewerEngine.value = e))
    }
  },
  { immediate: true },
)

// Chat-column chrome: keep the sidebar toggle, page nav, zoom, fits, the two
// reading interaction modes and search; drop the file menu (open/print/
// security/export), panning and the annotation/comment tools - this is a
// reading surface.
//
// Pointer + text-select are here because a reading surface people READ from
// has to let them take the words with them. The
// interaction plugin starts in 'pointer' and text only becomes selectable in
// 'text-select', so without the toggle the pane could not select or copy a
// single character at any window size. They come as a PAIR - exposing
// text-select alone would be a one-way door with no way back to pointer.
// Copy and Search then arrive for free: lector floats its own toolbar over a
// live selection, and under READER_PLUGINS (no annotation plugin) that
// toolbar is exactly those two buttons.
const KEEP_TOOLBAR = new Set([
  'tb-sidebar-toggle',
  'tb-sep-1',
  'tb-prev-page',
  'tb-page-input',
  'tb-next-page',
  'tb-zoom',
  'tb-sep-2',
  'tb-fit-page',
  'tb-fit-width',
  'tb-pointer-mode',
  'tb-text-select-mode',
  'tb-search',
])
// Re-skin, not just trim. lector's builder used to hardcode every glyph and
// ignore the `icon` its own schema carries; now it honours it, so the button
// and the panel it opens can finally agree - the thumbnails PANEL declares
// `icon: 'grid'`, and its sidebar tab has always been a grid, while the
// toolbar button that opens it was a generic 'sidebar'. That is also the
// obvious glyph for "hide this pane", which is how two identical icons ended
// up side by side. A grid says "thumbnails" anyway.
const ICON_OVERRIDES: Record<string, string> = {
  'tb-sidebar-toggle': 'grid',
}
const studioSchema = {
  toolbar: {
    items: (DEFAULT_UI_SCHEMA.toolbar?.items ?? [])
      .filter((i: { id: string }) => KEEP_TOOLBAR.has(i.id))
      .map((i: { id: string }) => (ICON_OVERRIDES[i.id] ? { ...i, icon: ICON_OVERRIDES[i.id] } : i)),
  },
}

// The pane's own three actions, rendered by lector inside its toolbar rather
// than by us in a bar above it. A second row of chrome
// meant two sets of button metrics and a host action that visibly did not
// belong; lector builds these with its own `.lector-btn`, so they are the
// same size, spacing and hover as everything beside them.
// Icons are lector's, not paddock's, for the same reason - its glyphs are
// drawn on a 24-box at stroke 2 and ours are Phosphor at a different weight.
const toolbarExtras = [
  {
    // Leftmost, before the thumbnails toggle: this one collapses the whole
    // pane, so it belongs outside the controls it collapses rather than after
    // them. And a chevron, not `sidebar` - that is the
    // glyph lector's own thumbnails toggle already wears, and two identical
    // icons one after the other say nothing.
    id: 'pk-fold',
    icon: 'chevron-left',
    tooltip: 'Hide the document',
    section: 'left' as const,
    placement: 'start' as const,
    onSelect: () => emit('fold'),
  },
  {
    // The two views lector leaves no room for, behind one button.
    // It was the eye alone while "what the model reads" was the only one; file
    // details made it two, and two buttons opening two windows over one
    // document is the shape we took out of the file dialog.
    // Still the eye - "look inside this file" covers both answers, and it is
    // lector's own glyph set, which has no `info`.
    id: 'pk-insight',
    icon: 'eye',
    tooltip: 'Metadata and model metadata',
    section: 'right' as const,
    disabled: () => !ctx.value,
    onSelect: () => openInfo(),
  },
  {
    id: 'pk-download',
    icon: 'download',
    tooltip: 'Download the original document',
    section: 'right' as const,
    disabled: () => !ctx.value,
    onSelect: () => download(),
  },
]

// ── multi-doc: every PDF opens as a lector tab; selection syncs both ways ──
const lectorDocBySource = reactive(new Map<string, string>())
const sourceByLectorDoc = reactive(new Map<string, string>())
const pageCountBySource = reactive(new Map<string, number>())
const opening = new Set<string>()
const offs: (() => void)[] = []

function docManager(): DocumentManagerCapability | undefined {
  try {
    return viewerEngine.value?.plugins.get<DocumentManagerCapability>('document-manager')
  } catch {
    return undefined
  }
}
function docCap(): DocumentCapability | undefined {
  try {
    return viewerEngine.value?.plugins.get<DocumentCapability>('document')
  } catch {
    return undefined
  }
}

async function openMissingDocs(): Promise<void> {
  const dm = docManager()
  if (!dm || !viewerReady.value) return
  for (const [i, c] of all.value.entries()) {
    const att = usesLector(c) ? c.pdf?.attachmentId : undefined
    const sid = c.source.id
    if (!att || lectorDocBySource.has(sid) || opening.has(sid)) continue
    opening.add(sid)
    try {
      const od = await dm.openFromUrl(attachmentsApi.url(att), {
        name: docName(c, i),
        skipRecent: true,
      })
      lectorDocBySource.set(sid, String(od.id))
      sourceByLectorDoc.set(String(od.id), sid)
      pageCountBySource.set(sid, od.handle.pageCount)
    } catch {
      /* a doc that fails to open simply has no tab; its run still shows */
    } finally {
      opening.delete(sid)
    }
  }
  syncActiveToLector()
}

// conv -> lector: the selected document's tab activates
function syncActiveToLector(): void {
  const dc = docCap()
  const sid = ctx.value?.source.id
  if (!dc || !sid) return
  const want = lectorDocBySource.get(sid)
  if (!want) return
  const current = dc.activeDocument.peek()
  if (current && String(current.id) === want) return
  dc.setActive(want as never)
}

function onViewerReady(engine: LectorEngine): void {
  viewerEngine.value = engine
  viewerReady.value = true
  const events = engine.plugins.events
  // the host-overlay contract: lector announces each live page layer
  offs.push(
    events.on('ui:page-mounted', (...args: unknown[]) => {
      const ev = args[0] as PageMountedEvent
      pageHosts.set(ev.pageIndex, ev.overlayEl)
    }),
    events.on('ui:page-unmounted', (...args: unknown[]) => {
      const ev = args[0] as { pageIndex: number }
      pageHosts.delete(ev.pageIndex)
    }),
  )
  // lector -> conv: a tab click moves the selection
  const dc = docCap()
  if (dc) {
    offs.push(
      dc.activeDocument.subscribe((h) => {
        const sid = h ? sourceByLectorDoc.get(String(h.id)) : undefined
        if (sid) select(sid)
      }),
    )
  }
  void openMissingDocs()
}

watch(all, () => void openMissingDocs())
watch(
  () => ctx.value?.source.id,
  () => syncActiveToLector(),
)
// a conversation switch resets the viewer's document set
watch(
  () => chat.active?.id,
  () => {
    lectorDocBySource.clear()
    sourceByLectorDoc.clear()
    pageCountBySource.clear()
    pageHosts.clear()
    viewerReady.value = false
    void docManager()?.closeAll()
  },
)
onBeforeUnmount(() => {
  for (const off of offs) off()
  void docManager()?.closeAll()
})

// ── the OCR overlay on lector's pages ───────────────────────────────────────
const pageHosts = reactive(new Map<number, HTMLElement>())

// fan-out page i is DOCUMENT page rangeStart+i when a page range rode the file
const rangeStart = computed(() => {
  const c = ctx.value
  if (!c?.pdf) return 0
  const count = pageCountBySource.get(c.source.id) ?? 1
  return pageRangeBounds(c.pdf.pageRange, count)[0]
})

// boxes for a DOCUMENT page index of the SELECTED doc (live parse while a
// page reads - lib/ocr.ts pageRegionBoxes)
function boxesFor(docPage: number) {
  const pages = runPages.value
  if (!pages) return []
  const p = pages[docPage - rangeStart.value]
  return p ? pageRegionBoxes(p) : []
}

// auto-follow: the viewer tracks the page being read
const readingIdx = computed(() => runPages.value?.findIndex((p) => p.state === 'reading') ?? -1)
watch(readingIdx, (i) => {
  if (i < 0 || !viewerEngine.value || !showViewer.value) return
  try {
    const vp = viewerEngine.value.plugins.get<{ scrollToPage(n: number, s?: boolean): void }>(
      'viewport',
    )
    vp.scrollToPage(rangeStart.value + i, true)
  } catch {
    /* viewport not ready yet - the next page flip tries again */
  }
})

// per-box copy feedback for the click popover
const copiedBox = ref('')
async function copyBox(key: string, text: string): Promise<void> {
  try {
    await copyText(text)
    copiedBox.value = key
    setTimeout(() => (copiedBox.value = ''), 1500)
  } catch {
    /* clipboard denied - the button just doesn't confirm */
  }
}

// ── the scriptor lane (Word documents) ──────────────────────────────────────
// lector opens a PDF from a URL on its own worker; scriptor takes the bytes,
// so the pane fetches them for the SELECTED document only. A monotonic token
// guards the swap: a slow fetch that lands after you've moved to another tab
// must not paint the wrong document.
const docxBytes = shallowRef<Uint8Array | null>(null)
const docxError = ref<string | null>(null)
const scriptor = ref<InstanceType<typeof ScriptorDoc> | null>(null)
const docxPages = ref(0)
// Declared here rather than down in the tracked-changes section because the
// fetch watch below is `immediate` - its callback runs during setup, so
// anything it clears has to already exist or it reads a const in its TDZ.
const reviewers = ref<string[]>([])
const trackView = ref<TrackDisplay>('all')
let docxGen = 0
watch(
  () => ctx.value?.docx?.attachmentId,
  async (id) => {
    const mine = ++docxGen
    docxBytes.value = null
    docxError.value = null
    docxPages.value = 0
    // Reset the readouts too, or the previous document's page count and review
    // state sit in the toolbar until the new view's first notify lands.
    reviewers.value = []
    trackView.value = 'all'
    if (!id) return
    try {
      const resp = await fetch(attachmentsApi.url(id))
      if (!resp.ok) throw new Error(`fetch failed (${resp.status})`)
      const buf = await resp.arrayBuffer()
      if (mine !== docxGen) return
      docxBytes.value = new Uint8Array(buf)
    } catch (e) {
      // Loud, not blank: a document that cannot be read says so and offers the
      // original, the same as the dialog always did.
      if (mine === docxGen) docxError.value = e instanceof Error ? e.message : String(e)
    }
  },
  { immediate: true },
)

// ── zoom: one control, two engines behind it ────────────────────────────────
// The header's -/%/+ drives whichever surface is showing. For images that is
// the page stack's own width; for Word it is scriptor's `setZoom`, which
// re-renders the canvas at the new scale (a CSS transform would just blur it).
// scriptor also binds Ctrl/Cmd +/-/0 and Ctrl+wheel itself, so the readout
// follows `state-change` rather than assuming this header is the only writer.
// Which engine it is follows the VIEW, not the document: a turn can carry a
// Word document and a picture at once, and then the same control has to mean
// scriptor on one tab and the stack's width on the other.
//
// The percentage is a SCALE of the PICTURE, not a fraction of the pane.
// It used to be the latter - the stack was simply `width: N%` of
// its scroll box - which made the number mean nothing on its own: dragging the
// divider resized the photo while the readout still said 100%, and "100%" of a
// 4000 px photo in a 451 px column was an 11% downscale, so a 25-point step
// moved almost nothing. Now 100% is one image pixel per CSS pixel, the way
// every image viewer means it, the pane's width is independent of it, and
// filling the pane is a separate action (the readout button) that re-runs
// itself on resize only while nobody has chosen a zoom.
const bodyEl = ref<HTMLElement>()
const docxEl = ref<HTMLElement>()
const zoom = ref(100)
/** The widest page's own pixel width, reported by the stack. 0 until the first
 *  page decodes, and while it is 0 the old pane-relative width stands in. */
const stackNatural = ref(0)
// A picture can need a far smaller number than a Word page ever does: fitting
// a phone photo to a narrow column lands near 10%.
const zoomMin = computed(() => (view.value === 'docx' ? 25 : 1))
const zoomMax = computed(() => (view.value === 'docx' ? 400 : 300))
// True while the zoom is still the fit we computed - nobody has touched it. It
// is what lets the fit re-run when the pane is resized without ever overriding
// a zoom the user chose.
const zoomIsFit = ref(true)
// A step is MULTIPLICATIVE on the picture stack and additive on Word. Both
// follow from what the number means there: a photo sitting at 11% and one at
// 250% each want "a quarter bigger", where +25 points would be a fourfold jump
// for the first and barely visible for the second. A Word page is always near
// 100%, so points are the natural unit and scriptor's own bindings use them.
const ZOOM_STEP = 1.25
function zoomBy(d: number): void {
  if (view.value === 'stack') {
    setZoom(d > 0 ? zoom.value * ZOOM_STEP : zoom.value / ZOOM_STEP)
    return
  }
  setZoom(zoom.value + d)
}
function setZoom(pct: number, fit = false): void {
  zoom.value = Math.round(Math.min(zoomMax.value, Math.max(zoomMin.value, pct)))
  zoomIsFit.value = fit
}
/** The stack's width in CSS pixels - the zoom applied to the picture's own
 *  size. Falls back to the pane-relative form until a page reports a width,
 *  which is also what an empty stack wants. */
const stackStyle = computed(() =>
  stackNatural.value > 0
    ? { width: `${Math.round((stackNatural.value * zoom.value) / 100)}px` }
    : { width: `${zoom.value}%` },
)
watch(zoom, (v) => {
  if (view.value === 'docx') scriptor.value?.setZoom(v / 100)
})

/** Fit the Word page to the pane, which is what a 560-to-1000px column wants
 *  on open - a letter page lays out at ~1123 CSS px, so at 100% most of it is
 * simply off the right edge and the pane looks broken.
 *
 *  Measured, not computed from paper size: ask the rendered sheet how wide it
 *  is at the CURRENT zoom and scale from there. That works for any page size,
 *  any margin, and any zoom we happen to be at, without the pane needing to
 *  know a thing about the document. */
function fitDocx(): void {
  const host = docxEl.value
  const sheet = host?.querySelector<HTMLElement>('.scriptor-sheet')
  if (!host || !sheet) return
  const sheetW = sheet.getBoundingClientRect().width
  if (sheetW <= 0) return
  // clientWidth excludes the scrollbar, so the fit does not oscillate with it
  const avail = host.clientWidth - 24
  if (avail <= 0) return
  setZoom(Math.round(((zoom.value * avail) / sheetW) / 5) * 5, true)
}
// Fit once the first layout exists (pageCount goes 0 -> n on the first render).
watch(docxPages, (n, prev) => {
  if (n > 0 && !prev) void nextTick(fitDocx)
})
// ...and again when the column is dragged, but only while the zoom is still the
// one we computed. The pane's width arrives as a CSS var, so its own box is the
// thing to watch rather than a number this component never receives.
let docxRo: ResizeObserver | null = null
watch(docxEl, (el) => {
  docxRo?.disconnect()
  docxRo = null
  if (!el) return
  docxRo = new ResizeObserver(() => {
    if (zoomIsFit.value) fitDocx()
  })
  docxRo.observe(el)
})
onBeforeUnmount(() => docxRo?.disconnect())

/** The picture stack's own fit - the same contract as fitDocx, measured the
 *  same way. Reading the padding off the box rather than restating 12px keeps
 *  it true if the CSS moves; clientWidth excludes the scrollbar, so a fit that
 *  removes one cannot oscillate against it. */
function fitStack(): void {
  const host = bodyEl.value
  if (!host || stackNatural.value <= 0) {
    // no page has reported a size yet: the pane-relative fallback is in force,
    // and there 100% is the fit
    setZoom(100, true)
    return
  }
  const cs = getComputedStyle(host)
  const avail = host.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)
  if (avail <= 0) return
  setZoom((avail / stackNatural.value) * 100, true)
}
// Fit the moment the first page reports its size, and again when the column is
// dragged - but only while the zoom is still the one we computed, so a chosen
// zoom survives a resize instead of being overwritten by it.
watch(stackNatural, (n) => {
  if (n > 0 && zoomIsFit.value) void nextTick(fitStack)
})
let bodyRo: ResizeObserver | null = null
watch(bodyEl, (el) => {
  bodyRo?.disconnect()
  bodyRo = null
  if (!el) return
  bodyRo = new ResizeObserver(() => {
    if (zoomIsFit.value && view.value === 'stack') fitStack()
  })
  bodyRo.observe(el)
})
onBeforeUnmount(() => bodyRo?.disconnect())

// ── looking at a picture, not a web page ("the image viewer
//    feels 1990") ────────────────────────────────────────────────────────────
//
// Why not A LIBRARY. OpenSeadragon (BSD, still shipping - 6.1.0 this month) is
// the reference deep-zoom viewer, and @panzoom/panzoom is the small MIT one;
// both are TRANSFORM-VIEWPORT designs, a fixed box with the content translated
// and scaled inside it. Our stack is a SCROLLED DOM: real <img> elements the
// browser lays out, with the OCR grounding overlay positioned over them in
// percentages and the PDF page stack scrolling as one column. A canvas viewer
// deletes the overlay outright; a transform viewport replaces the scrolling
// the multi-page case is built on. So the gestures land here, on the model we
// already have - drag to pan, ctrl+wheel and pinch to zoom at the pointer,
// rotate, double-click to fit - which is about 60 lines and no dependency.

/** Quarter turns, and only the picture stack has them: a Word page or a PDF
 *  arrives the way up its author chose, but a photo off a camera does not. */
const rotation = ref(0)
function rotateBy(deg: number): void {
  rotation.value = (((rotation.value + deg) % 360) + 360) % 360
}
const rotStyle = computed(() =>
  rotation.value ? { transform: `rotate(${rotation.value}deg)` } : undefined,
)

/** Zoom so the point under the pointer STAYS under the pointer. Without this
 *  every zoom jumps to wherever the scroll happened to be, which is the single
 *  thing that makes a viewer feel old. */
function zoomAt(pct: number, clientX: number, clientY: number): void {
  const host = bodyEl.value
  if (!host) {
    setZoom(pct)
    return
  }
  const before = zoom.value
  const box = host.getBoundingClientRect()
  // where the pointer is in CONTENT coordinates, at the old scale
  const cx = host.scrollLeft + (clientX - box.left)
  const cy = host.scrollTop + (clientY - box.top)
  setZoom(pct)
  const k = zoom.value / before
  void nextTick(() => {
    host.scrollLeft = cx * k - (clientX - box.left)
    host.scrollTop = cy * k - (clientY - box.top)
  })
}

/** Drag anywhere on the page to move it, the way every picture viewer works.
 *  Only when there is something to move: below the fit, a drag would rubber-
 *  band against nothing and read as a broken control. */
let drag: { x: number; y: number; left: number; top: number; id: number } | null = null
function canPan(host: HTMLElement): boolean {
  return host.scrollWidth > host.clientWidth + 1 || host.scrollHeight > host.clientHeight + 1
}
function onPointerDown(e: PointerEvent): void {
  const host = bodyEl.value
  if (!host || view.value !== 'stack' || e.button !== 0 || !canPan(host)) return
  // never steal a click meant for an OCR box, a link or a text selection
  if ((e.target as HTMLElement).closest('button, a, [role="button"]')) return
  // An <img> is natively DRAGGABLE, so without this the pan gesture starts a
  // file drag instead: the picture follows the cursor across the page and
  // drops into the composer as an attachment (and it does
  // attach). The img carries draggable="false" too - belt and braces, since
  // preventDefault here also stops the text selection a drag would paint.
  e.preventDefault()
  drag = { x: e.clientX, y: e.clientY, left: host.scrollLeft, top: host.scrollTop, id: e.pointerId }
  host.setPointerCapture(e.pointerId)
  host.style.cursor = 'grabbing'
}
function onPointerMove(e: PointerEvent): void {
  const host = bodyEl.value
  if (!host || !drag || e.pointerId !== drag.id) return
  host.scrollLeft = drag.left - (e.clientX - drag.x)
  host.scrollTop = drag.top - (e.clientY - drag.y)
}
function onPointerUp(e: PointerEvent): void {
  const host = bodyEl.value
  if (!host || !drag || e.pointerId !== drag.id) return
  host.releasePointerCapture(e.pointerId)
  host.style.cursor = ''
  drag = null
}

/** Double-click toggles between filling the pane and life size - the gesture
 *  every viewer has, and the fastest way back from a zoom that went too far. */
function onDoubleClick(e: MouseEvent): void {
  if (view.value !== 'stack') return
  if (zoomIsFit.value) zoomAt(100, e.clientX, e.clientY)
  else fitStack()
}

/** scriptor moved something we display - its own zoom, the page count arriving
 *  after the first layout, or the review state. */
function onScriptorState(): void {
  const s = scriptor.value
  if (!s) return
  docxPages.value = s.pageCount()
  const pct = Math.round(s.getZoom() * 100)
  if (pct !== zoom.value) {
    // A value we did not send: scriptor binds Ctrl+wheel and Ctrl +/-/0 itself,
    // so this is the user zooming. Reaching for that is a choice, and the
    // auto-fit stops following the pane from here on. (Our own writes read back
    // equal and never get here.)
    zoomIsFit.value = false
    zoom.value = pct
  }
  reviewers.value = s.reviewers().map((r) => r.name)
  trackView.value = s.trackDisplay()
}

// ── tracked changes ─────────────────────────────────────────────────────────
// Nothing here turns them on: scriptor's engine defaults to All Markup, so a
// document with revisions arrives with insertions underlined and deletions
// struck, author-coloured, in read mode, without the host asking. What was
// missing is any way to know that - a clean-looking document and a document
// whose markup you can't reach look identical.
//
// So the control appears only when the document actually carries revisions
// (`reviewers()` is empty otherwise), which makes the toolbar itself the
// answer to "does this have tracked changes?", and offers Word's four Display
// for Review views behind it. `reviewers` / `trackView` are declared up with
// the rest of the docx state, for the TDZ reason noted there.
const TRACK_VIEWS: { value: TrackDisplay; label: string }[] = [
  { value: 'all', label: 'All markup' },
  { value: 'simple', label: 'Simple markup' },
  { value: 'none', label: 'No markup' },
  { value: 'original', label: 'Original' },
]
function setTrackView(mode: TrackDisplay): void {
  scriptor.value?.setTrackDisplay(mode)
  trackView.value = mode
}
function onWheel(e: WheelEvent): void {
  // ctrl+wheel zooms (and a trackpad pinch arrives as exactly that); a bare
  // wheel keeps scrolling the column, because a PDF stack is read by scrolling
  // and hijacking that would be worse than any zoom is good.
  if (!e.ctrlKey) return
  e.preventDefault()
  if (view.value === 'stack') {
    // multiplicative, like the buttons: a photo at 11% and one at 250% both
    // want "a bit bigger", not "+10 points"
    zoomAt(e.deltaY < 0 ? zoom.value * 1.12 : zoom.value / 1.12, e.clientX, e.clientY)
    return
  }
  zoomBy(e.deltaY < 0 ? 10 : -10)
}
// The pane's own toolbar (`docpane__bar`) stands in for lector's on every
// surface lector does not draw - Word and image documents - in the same order:
// fold on the left, zoom in the middle, download on the right, at lector's own
// metrics (see the CSS). Before this the image path had no fold control at
// all, so a photo document could open a column with no way to shut it; the
// parity that Word needed fixed that too. The two panels lector reaches
// through a toolbar button are tabs here instead, which is why the bar's set
// is one shorter than the viewer's.

// A fresh document starts over: carrying the last one's zoom across is a
// surprise, and scriptor's view is new anyway. Word documents then fit
// themselves once the first layout reports (see fitDocx).
watch(
  () => ctx.value?.source.id,
  () => {
    zoom.value = 100
    zoomIsFit.value = true
    // the stack re-reports it as the new document's pages decode; carrying the
    // old one's width across would scale the new picture by a stranger
    stackNatural.value = 0
  },
)

// ── "Model metadata" ──────────────────────────────────────────────────────
// The honesty panel, which used to live as a tab in the file-preview dialog.
// That dialog is no longer the entry point for a PDF or a Word document (the
// chip opens this pane now), so the panel moves here or it becomes
// unreachable - and in the document-parser lane it already was, because the
// chip was inert there.
// Bytes are fetched on demand: the pane renders through lector's own worker
// and never needs them otherwise.
//
// One loader, two ways to reach it: lector's toolbar opens it in a dialog, and
// every other document shows the same panel as a tab (see `views`).
//
// It describes one file, and a turn that carried several has to pick: the
// document first, its pictures second. A photo that rode in beside a .docx
// therefore reads the .docx's extraction - the panel is about what the prompt
// carries, and the document is the bulk of it. The per-file answer is the File
// details tab, which asks about each file the document is made of.
const insightFile = ref<{ name: string; bytes: Uint8Array; mime?: string } | null>(null)
const insightError = ref<string | null>(null)
const insightOpen = ref(false)
const insightBusy = ref(false)
// Which document the loaded bytes belong to. Without it, selecting another
// document and opening the panel would show the previous one's extraction -
// the fetch would be skipped as already done.
let insightSid = ''
async function loadInsight(): Promise<void> {
  const c = ctx.value
  const p = c?.pdf ?? c?.docx ?? c?.images[0]
  if (!c || insightBusy.value) return
  if (insightFile.value && insightSid === c.source.id) return
  if (!p?.attachmentId) {
    insightError.value = 'No stored copy of this file. It was sent before originals were kept.'
    return
  }
  insightBusy.value = true
  try {
    const resp = await fetch(attachmentsApi.url(p.attachmentId))
    if (!resp.ok) throw new Error(`fetch failed (${resp.status})`)
    insightFile.value = {
      name: p.name || 'document',
      bytes: new Uint8Array(await resp.arrayBuffer()),
      mime: p.mime || undefined,
    }
    insightSid = c.source.id
  } catch (e) {
    // Loud, not blank: the panel says why rather than spinning forever.
    insightError.value = e instanceof Error ? e.message : String(e)
  } finally {
    insightBusy.value = false
  }
}
/** lector's toolbar button. The dialog opens at once and the bytes arrive
 *  behind it - its first tab is the file's own metadata, which the manager
 *  answers off the blob and which never needed these bytes at all. */
function openInfo(): void {
  insightOpen.value = true
  void loadInsight()
}

// The panel tab fetches on arrival rather than on mount, so a pane that is
// never asked the question never reads the file.
watch(view, (v) => {
  if (v === 'model') void loadInsight()
})
// A different document's extraction is a different file's. The dialog closes
// with it: leaving `insightOpen` set would make it reappear by itself the next
// time any load put bytes back.
watch(
  () => ctx.value?.source.id,
  () => {
    insightFile.value = null
    insightError.value = null
    insightSid = ''
    insightOpen.value = false
    metaSeen.value = false
    forensicsSeen.value = false
  },
)

// The ORIGINAL file(s) this document is made of - the stored bytes, never a
// re-render. Everything the turn carried, because they are one document here:
// a picture that rode in beside a PDF has no other way out now that the
// thread's tile is a link into this pane. The Metadata tab reads the same
// list, one answer per file, which is the whole reason it is a list.
const docFiles = computed(() => {
  const c = ctx.value
  if (!c) return []
  const parts: { attachmentId: string; name: string }[] = []
  if (c.pdf) parts.push(c.pdf)
  if (c.docx) parts.push(c.docx)
  parts.push(...c.images)
  return parts.filter((p) => !!p.attachmentId)
})
const downloadLabel = computed(() => {
  const n = docFiles.value.length
  if (n > 1) return 'Download the originals'
  return ctx.value?.pdf || ctx.value?.docx
    ? 'Download the original document'
    : 'Download the original image'
})
function download(): void {
  for (const p of docFiles.value) {
    const a = document.createElement('a')
    a.href = attachmentsApi.url(p.attachmentId)
    a.download = p.name || 'document'
    a.click()
  }
}
</script>

<template>
  <aside class="docpane">
    <nav v-if="all.length" class="docpane__tabs">
      <Tooltip v-for="(c, i) in all" :key="c.source.id" :label="docName(c, i)">
        <button
          type="button"
          class="docpane__tab"
          :class="{ 'docpane__tab--on': c.source.id === ctx?.source.id }"
          @click="select(c.source.id)"
        >
          <Icon :name="docIcon(c)" :size="12" />
          <span class="docpane__tabname">{{ docName(c, i) }}</span>
        </button>
      </Tooltip>
    </nav>

    <div v-if="hasPdf" v-show="ctx && showViewer" class="docpane__lector">
      <LectorPdfViewer
        v-if="viewerEngine"
        :key="chat.active?.id"
        :engine="viewerEngine"
        :theme="settings.theme"
        :panels="['thumbnails']"
        initial-panel="thumbnails"
        :document-tabs="false"
        :toolbar-extras="toolbarExtras"
        :ui-schema="studioSchema"
        @ready="onViewerReady"
      />
      <template v-for="[idx, host] of pageHosts" :key="idx">
        <Teleport :to="host">
          <Popover v-for="(b, bi) in boxesFor(idx)" :key="bi" side="bottom" align="start">
            <template #trigger>
              <button
                type="button"
                class="docpane-ov__box"
                :class="{ 'docpane-ov__box--unsure': b.unsure }"
                :style="{ ...b.style, '--hue': String(b.hue) }"
                :aria-label="b.label"
              />
            </template>
            <div class="dpgpop">
              <div class="dpgpop__hd">
                <span class="dpgpop__label" :style="{ '--hue': String(b.hue) }">{{ b.label }}</span>
                <span v-if="b.unsure" class="dpgpop__flag">has unsure words</span>
                <button
                  v-if="b.text"
                  class="dpgpop__copy"
                  type="button"
                  @click="copyBox(`${idx}:${bi}`, b.text)"
                >
                  <Icon :name="copiedBox === `${idx}:${bi}` ? 'check' : 'copy'" :size="12" />
                  {{ copiedBox === `${idx}:${bi}` ? 'Copied' : 'Copy text' }}
                </button>
              </div>
              <p v-if="b.text" class="dpgpop__text">{{ b.text }}</p>
            </div>
          </Popover>
        </Teleport>
      </template>
    </div>

    <template v-if="ctx && !showViewer">
      <header class="docpane__bar">
        <div class="docpane__grp">
          <Tooltip label="Hide the document">
            <button
              class="docpane__btn"
              type="button"
              aria-label="Hide the document"
              @click="emit('fold')"
            >
              <Icon name="chevron-left" />
            </button>
          </Tooltip>
        </div>
        <template v-if="showZoom">
          <span class="docpane__div" />
          <div class="docpane__grp">
            <Tooltip label="Zoom out">
              <button class="docpane__btn" type="button" aria-label="Zoom out" @click="zoomBy(-25)">
                <Icon name="minus" />
              </button>
            </Tooltip>
            <Tooltip :label="view === 'docx' ? 'Fit the page to the pane' : 'Fit the picture to the pane'">
              <button
                class="docpane__btn docpane__btn--num"
                type="button"
                :aria-label="view === 'docx' ? 'Fit the page to the pane' : 'Fit the picture to the pane'"
                @click="view === 'docx' ? fitDocx() : fitStack()"
              >
                {{ zoom }}%
              </button>
            </Tooltip>
            <Tooltip label="Zoom in">
              <button class="docpane__btn" type="button" aria-label="Zoom in" @click="zoomBy(25)">
                <Icon name="plus" />
              </button>
            </Tooltip>
          </div>
          <div v-if="view === 'stack'" class="docpane__grp">
            <span class="docpane__div" />
            <Tooltip label="Rotate left">
              <button
                class="docpane__btn"
                type="button"
                aria-label="Rotate left"
                @click="rotateBy(-90)"
              >
                <Icon name="rotate-left" />
              </button>
            </Tooltip>
            <Tooltip label="Rotate right">
              <button
                class="docpane__btn"
                type="button"
                aria-label="Rotate right"
                @click="rotateBy(90)"
              >
                <Icon name="rotate-right" />
              </button>
            </Tooltip>
          </div>
        </template>
        <span v-if="docxPages && view === 'docx'" class="docpane__pages">
          {{ docxPages }} {{ docxPages === 1 ? 'page' : 'pages' }}
        </span>
        <span class="docpane__spacer" />
        <div v-if="reviewers.length && view === 'docx'" class="docpane__grp">
          <Menu>
            <MenuTrigger>
              <Tooltip
                :label="`Tracked changes by ${reviewers.join(', ')} - choose what to show`"
              >
                <button class="docpane__btn docpane__btn--label" aria-label="Display for review">
                  <Icon name="edit" />
                  {{ TRACK_VIEWS.find((v) => v.value === trackView)?.label }}
                </button>
              </Tooltip>
            </MenuTrigger>
            <MenuContent align="end">
              <MenuItem
                v-for="v in TRACK_VIEWS"
                :key="v.value"
                @select="setTrackView(v.value)"
              >
                {{ v.label }}
              </MenuItem>
            </MenuContent>
          </Menu>
          <span class="docpane__div" />
        </div>
        <div class="docpane__grp">
          <Tooltip :label="downloadLabel">
            <button
              class="docpane__btn"
              type="button"
              :aria-label="downloadLabel"
              @click="download()"
            >
              <Icon name="download" />
            </button>
          </Tooltip>
        </div>
      </header>

      <div class="docpane__views">
        <Tabs v-model="view" :tabs="views" />
      </div>

      <div
        v-if="ctx.docx"
        v-show="view === 'docx'"
        ref="docxEl"
        class="docpane__body docpane__body--docx"
      >
        <div v-if="docxError" class="docpane__msg">
          <Icon name="file-text" :size="22" />
          <p>{{ docxError }}</p>
        </div>
        <ScriptorDoc
          v-else-if="docxBytes"
          ref="scriptor"
          :key="ctx.source.id"
          :docx="docxBytes"
          mode="read"
          :selectable="true"
          @state-change="onScriptorState"
        />
        <div v-else class="docpane__msg">
          <Icon name="spinner" :size="20" class="docpane__spin" />
          <span>Opening...</span>
        </div>
      </div>

      <div
        v-if="ctx.images.length || ctx.pdf"
        v-show="view === 'stack'"
        ref="bodyEl"
        class="docpane__body"
        :class="{ 'docpane__body--grab': view === 'stack' }"
        @wheel="onWheel"
        @pointerdown="onPointerDown"
        @pointermove="onPointerMove"
        @pointerup="onPointerUp"
        @pointercancel="onPointerUp"
        @dblclick="onDoubleClick"
      >
        <div class="docpane__zoom" :style="stackStyle">
          <div class="docpane__rot" :style="rotStyle">
          <DocumentPages
            :key="ctx.source.id"
            :doc-id="ctx.source.id"
            :images="ctx.images"
            :pdf="ctx.pdf"
            :regions="regions"
            :run-pages="runPages"
            @natural="stackNatural = $event"
          />
          </div>
        </div>
      </div>

      <div v-show="view === 'meta'" class="docpane__insight">
        <FileMetaPane v-if="metaSeen" :parts="docFiles" />
      </div>

      <div v-show="view === 'forensics'" class="docpane__insight">
        <ForensicsPane v-if="forensicsSeen" :parts="docFiles" />
      </div>

      <div v-show="view === 'model'" class="docpane__insight">
        <InsightPane
          v-if="insightFile"
          :file="insightFile"
          :with-meta="chat.active?.fileMetadataEnabled ?? true"
          :model="chat.active?.model"
        />
        <div v-else-if="insightError" class="pv__overlay-msg pv__overlay-msg--err">
          <Icon name="file-text" :size="28" />
          <p>{{ insightError }}</p>
        </div>
        <div v-else class="pv__overlay-msg">
          <Icon name="spinner" :size="22" class="pv__spin" />
          <span>Opening...</span>
        </div>
      </div>
    </template>

    <div v-if="!ctx" class="docpane__empty">
      <Icon name="file-text" :size="22" />
      <span>Drop a document to read it here.</span>
    </div>

    <FileInfoDialog
      :open="insightOpen"
      :parts="docFiles"
      :file="insightFile"
      :file-error="insightError"
      :title="ctxName"
      :with-meta="chat.active?.fileMetadataEnabled ?? true"
      :model="chat.active?.model"
      @close="insightOpen = false"
    />
  </aside>
</template>

<style scoped>
.docpane {
  display: flex;
  flex-direction: column;
  width: var(--pk-docpane-width, 560px);
  min-width: 0;
  height: 100%;
  flex-shrink: 0;
  /* no border-right: the ResizeHandle draws the divider (Traverse pattern).
     Drawing one here put a second line 3px from the handle's own, with the
     grip sitting on only one of them. */
  background: var(--pk-bg-surface);
}
.docpane__tabs {
  display: flex;
  gap: 4px;
  padding: 8px 10px;
  border-bottom: 1px solid var(--pk-border-default);
  overflow-x: auto;
  flex-shrink: 0;
}
.docpane__tab {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  max-width: 180px;
  padding: 3px 9px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-sm);
  background: none;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-xs);
  cursor: pointer;
  white-space: nowrap;
}
.docpane__tab:hover {
  color: var(--pk-text-primary);
  border-color: var(--pk-border-strong);
}
.docpane__tab--on {
  color: var(--pk-accent);
  background: var(--pk-accent-subtle);
  border-color: var(--pk-accent);
}
.docpane__tabname {
  overflow: hidden;
  text-overflow: ellipsis;
}
/* the lector viewer owns its scroll, toolbar, tabs and sidebar in this box */
.docpane__lector {
  flex: 1;
  min-height: 0;
  position: relative;
}
/* The toolbar for every document lector does not draw, built to lector's own
   metrics rather than near them ("avoid too many varieties").
   Flip between a PDF tab and a Word tab and the row must not move: lector's
   .lector-toolbar is 8px/16px padding around 32px buttons, its groups sit at
   gap 4 with gap 8 between them, and its divider is 1x24. Every number below is
   that one, expressed in paddock's tokens - which line up exactly, since
   --lector-radius is 6px (= --pk-radius-md) and its font-size-sm is 14px
   (= --pk-font-size-sm).
   Glyphs stay Phosphor at the app's own 18: lector draws a different icon set
   on a 24-box at stroke 2, and no size makes those two families identical. The
   BUTTON is what makes the rows read as one toolbar. */
/* lector's toolbar, to the pixel - because on a PDF this pane is lector's
   toolbar and on an image it is ours, and the two sat at different heights
   with only one of them ruled off ("32+2x6 with border vs
   32+2x8 without"). The numbers come from base.css:
   `.lector-workspace--compact .lector-toolbar { padding: 6px 8px }` over the
   base rule's border-bottom, and the pane hosts lector in compact. */
.docpane__bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-bottom: 1px solid var(--pk-border-default);
  flex-shrink: 0;
  user-select: none;
}
.docpane__grp {
  display: flex;
  align-items: center;
  gap: 4px;
}
.docpane__div {
  width: 1px;
  height: 24px;
  background: var(--pk-border-default);
  flex-shrink: 0;
}
.docpane__spacer {
  flex: 1;
}
.docpane__btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 32px;
  min-width: 32px;
  padding: 5px;
  border: none;
  border-radius: var(--pk-radius-md);
  background: transparent;
  color: var(--pk-text-muted);
  font-family: inherit;
  font-size: var(--pk-font-size-sm);
  font-weight: 500;
  line-height: 1;
  cursor: pointer;
  white-space: nowrap;
  flex-shrink: 0;
  transition: background 0.15s cubic-bezier(0.4, 0, 0.2, 1),
    color 0.15s cubic-bezier(0.4, 0, 0.2, 1),
    box-shadow 0.15s cubic-bezier(0.4, 0, 0.2, 1);
}
.docpane__btn:hover {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
  box-shadow: inset 0 0 0 1px var(--pk-border-default);
}
.docpane__btn:active {
  background: var(--pk-bg-hover);
}
/* lector's own buttons carry a label the same way (gap 6, padding 5/8) when
   they show one - .lector-btn is width:auto with a __label span. */
.docpane__btn--label {
  padding: 5px 8px;
  gap: 6px;
}
/* The percentage is the reset control, the way lector's zoom button is both
   readout and control - one button instead of a readout plus a "Fit" pill. */
.docpane__btn--num {
  padding: 5px 8px;
  font-variant-numeric: tabular-nums;
}
.docpane__pages {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
/* The tab strip, sitting under the bar the way the file dialog's does under
   its own (pv__bar--tabbed) - one recipe for "a surface with tabs", not a
   second invention. The chrome has ONE rule, the strip's own underline, which
   is why the bar above draws no border of its own; the 16px inset is the bar's
   padding, so the two rows share a left edge. */
.docpane__views {
  flex-shrink: 0;
  padding: 0 16px;
}
/* The scroll container of the image-document path */
.docpane__body {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 12px;
}
/* Both panels position themselves absolutely inside their host (the pv shell
   they share with the file dialog), so the host has to be the relative box
   with a real height of its own. */
.docpane__insight {
  position: relative;
  flex: 1;
  min-height: 0;
  background: var(--pk-bg-inset);
}
/* The zoom is this box's width, so nothing may clamp it. It used to carry
   min-width:100%, which floored every value under 100% straight back to the
   pane width - zoom OUT was inert while zoom in worked, since nothing clamps
   the other direction. Auto margins take over the job that
   was really wanted: a stack narrower than the pane sits centred rather than
   against the left edge, and one wider than the pane overflows to the right
   only, so the scroll container can still reach its left edge. */
/* A picture you can take hold of. Only while something overflows the box -
   below the fit there is nothing to drag and a grab cursor would be a promise
   the pane does not keep. */
.docpane__body--grab {
  cursor: grab;
}
.docpane__rot {
  transition: transform 0.18s ease;
  transform-origin: center center;
}
.docpane__zoom {
  margin: 0 auto;
}
/* scriptor's page stage. It draws white sheets with their own drop shadows and
   leaves the backdrop to the host, so the pane supplies the recessed ground -
   the same one the file dialog used, because moving a document into the pane
   should not change how it looks. */
.docpane__body--docx {
  padding: 16px 0;
  background: var(--pk-bg-inset);
}
/* ScriptorDoc's root is a plain div holding an inline-block .scriptor-sheet;
   max-content + auto margins centre it while it fits and fall back to the left
   edge + scroll when a page is wider than the pane (flex centring would clip
   the left edge instead). */
.docpane__body--docx > div {
  width: max-content;
  margin: 0 auto;
}
/* the loading/error notes are children of the same box and are not sheets */
.docpane__body--docx > .docpane__msg {
  width: auto;
}
.docpane__msg {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  height: 100%;
  padding: 0 24px;
  text-align: center;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.docpane__spin {
  animation: docpane-spin 0.8s linear infinite;
}
@keyframes docpane-spin {
  to {
    transform: rotate(360deg);
  }
}
.docpane__empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
</style>

<!-- Unscoped deliberately: the box buttons teleport into lector's host-overlay
     layers, which carry no Vue scope hash. All classes are docpane-ov
     prefixed. -->
<style>
/* lector, wearing paddock's theme.
   The viewer ships its own ~40-token palette, and in DARK it is a different
   world: ground #1f2937 against our #121C26, a cool #f9fafb text ramp against
   our warm #ECE8E0, blue-500 against our sky, and a system-ui stack against
   Inter. In LIGHT the two agree almost everywhere (both grounds are #ffffff),
   which is why this only ever showed up in dark.
   Bridging the ground alone was worse than bridging nothing: lector picked its
   hover (#374151) to sit on ITS ground, so on ours the same hover became a
   bigger jump than lector ever intended. Moving the floor means moving the
   furniture.
   Tokens are lector's documented theming API - "override on .lector-workspace"
   - and since lector wraps its own defaults in :where() (specificity 0) this
   single-class rule wins outright, in both themes, with no doubled-class or
   !important escalation.
   Not bridged, deliberately: --lector-canvas-bg, the ground BEHIND the pages.
   That is a reading decision, not a consistency one - our --pk-bg-inset is
   near-black, which makes a white page pop and also glare - so it stays
   lector's until someone decides it on its own merits. */
.docpane__lector .lector-workspace {
  /* surfaces */
  --lector-bg: var(--pk-bg-surface);
  /* the doctabs strip: recessed against the toolbar in both themes, which is
     paddock's convention - lector raises it in dark and recesses it in light */
  --lector-bg-alt: var(--pk-bg-base);
  --lector-bg-hover: var(--pk-bg-hover);
  --lector-bg-active: var(--pk-bg-hover);
  --lector-bg-selected: var(--pk-accent-subtle);
  --lector-bg-input: var(--pk-bg-base);
  --lector-bg-overlay: var(--pk-bg-overlay);

  /* text */
  --lector-fg: var(--pk-text-primary);
  --lector-fg-secondary: var(--pk-text-secondary);
  --lector-fg-muted: var(--pk-text-muted);
  --lector-fg-disabled: var(--pk-text-muted);
  --lector-fg-on-accent: var(--pk-text-inverse);

  /* one accent per screen */
  --lector-accent: var(--pk-accent);
  --lector-accent-hover: var(--pk-accent-hover);
  --lector-accent-active: var(--pk-accent-active);
  --lector-accent-light: var(--pk-accent-subtle);
  --lector-accent-fg: var(--pk-text-inverse);

  /* borders */
  --lector-border: var(--pk-border-default);
  --lector-border-subtle: var(--pk-border-subtle);
  --lector-border-strong: var(--pk-border-strong);

  /* state */
  --lector-danger: var(--pk-status-error);
  --lector-danger-hover: var(--pk-status-error-hover);
  --lector-success: var(--pk-status-success);
  --lector-warning: var(--pk-status-warning);

  /* type: the viewer's chrome is part of the Studio, so it reads in the
     Studio's face rather than the OS default */
  --lector-font-family: var(--pk-font-sans);
  --lector-font-mono: var(--pk-font-mono);

  /* shadows + scrollbars, so a floating menu inside the viewer lands with the
     same weight as one outside it */
  --lector-shadow-lg: var(--pk-shadow-lg);
  --lector-scrollbar-track: var(--pk-bg-base);
  --lector-scrollbar-thumb: var(--pk-border-default);
  --lector-scrollbar-thumb-hover: var(--pk-border-strong);

  /* a tooltip is inverted in both products; keep lector's own inversion but in
     our colours */
  --lector-tooltip-bg: var(--pk-text-primary);
  --lector-tooltip-fg: var(--pk-bg-base);
}

.docpane-ov__box {
  position: absolute;
  padding: 0;
  appearance: none;
  pointer-events: auto;
  border: 1.5px solid hsl(var(--hue) 75% 45%);
  box-shadow: inset 0 0 0 1px rgb(255 255 255 / 0.65);
  background: hsl(var(--hue) 75% 50% / 0.08);
  border-radius: 2px;
  cursor: pointer;
}
.docpane-ov__box:hover {
  background: hsl(var(--hue) 75% 50% / 0.18);
}
.docpane-ov__box--unsure {
  border-color: var(--pk-status-error);
  border-style: dashed;
  background: color-mix(in srgb, var(--pk-status-error) 16%, transparent);
}
.docpane-ov__box--unsure:hover {
  background: color-mix(in srgb, var(--pk-status-error) 28%, transparent);
}
</style>
