<script setup lang="ts">
// "Metadata": everything the FILE says about itself, as a table.
//
// The other half of a two-audiences split. "Model metadata" is a
// CURATION - six PDF fields and three photo bits - because a tag zoo in a
// prompt is noise a model does not answer questions from. This pane is the
// person's half and discards nothing: a Nikon JPEG carries 121 fields against
// the 3 the prompt takes, and the difference is not more of the same. It is
// the whole exposure set, full GPS down to satellites and map datum, the
// provenance line, the entire MakerNote, and the values sift DERIVES - field
// of view, hyperfocal distance, 35 mm equivalent - which nothing else on the
// box computes. On a PDF it is Producer, page size, linearization, the XMP
// document/instance ids, and `JavaScript: Yes`, which we already read and
// used to bin.
//
// Answered by the MANAGER off the stored blob, so it works with no server
// running and in a cloud-model chat - reading your own file's EXIF must not
// need a GPU. Nothing is cached at either end: the parse is faster than the
// round trip, and a stale answer after an extractor fix is worse than none.
//
// A document can be more than one file - a turn that carried a PDF and a photo
// is one document to the pane - so this takes a list and answers per file.
// That is the sharpening the honesty panel could not do: it describes
// one extraction, this describes every file the extraction came from.
//
// The file's NAME leads every section, including when there is only one of
// them. It used to appear only in the multi-file case, on the theory that a
// single file's name was already known from the thread - but a document pane
// opened from a tile has no other label, and "which of my photos is this?" is
// the first question the details answer.
//
// One field gets drawn rather than tabled: a photo's coordinates go to
// PhotoLocation, above the rows and outside the filter, because they describe
// the FILE the way the summary line does and because two decimals are the one
// value in the table that nobody can read.
import { computed, ref, watch } from 'vue'
import { attachmentsApi, type AttachmentMetadata, type FileMetaTag } from '@/lib/api'
import { fmtFileSize } from '@/lib/format'
import Icon from '@/components/Icon.vue'
import PhotoLocation from '@/components/chat/PhotoLocation.vue'

const props = defineProps<{
  /** The file(s) the document is made of. Parts with no stored blob are
   *  skipped - there is nothing to read them from. */
  parts: { attachmentId?: string; name?: string }[]
}>()

interface Entry {
  id: string
  name: string
  meta: AttachmentMetadata | null
  error: string | null
}

const entries = ref<Entry[]>([])
const loading = ref(false)
// Monotonic token: selecting another document mid-flight must not let the
// previous file's answer paint over the new one.
let gen = 0

const stored = computed(() => props.parts.filter((p) => !!p.attachmentId))
// The identity of the REQUEST, not of the array - the parent recomputes its
// parts list on every render and an array watch would refetch each time.
const key = computed(() => stored.value.map((p) => p.attachmentId).join('|'))

watch(
  key,
  async () => {
    const mine = ++gen
    const want = stored.value
    entries.value = []
    if (!want.length) {
      loading.value = false
      return
    }
    loading.value = true
    const out = await Promise.all(
      want.map(async (p): Promise<Entry> => {
        const id = p.attachmentId as string
        try {
          return { id, name: p.name || '', meta: await attachmentsApi.metadata(id), error: null }
        } catch (e) {
          // One unreadable file is not the whole answer: the others still show.
          return { id, name: p.name || '', meta: null, error: e instanceof Error ? e.message : String(e) }
        }
      }),
    )
    if (mine !== gen) return
    entries.value = out
    loading.value = false
  },
  { immediate: true },
)

const total = computed(() =>
  entries.value.reduce((n, e) => n + (e.meta?.groups.reduce((g, x) => g + x.tags.length, 0) ?? 0), 0),
)

// A list you can take in at a glance does not need a filter; 121 fields across
// five groups very much does.
const FILTER_FROM = 20
const q = ref('')
const needle = computed(() => q.value.trim().toLowerCase())
watch(key, () => (q.value = ''))

/** Entries with their groups narrowed to what matches, empty groups dropped.
 *  Both halves of a row are searchable: "nikon" finds it by value, "gps" by
 *  name, and neither is more natural than the other. */
const shown = computed(() =>
  entries.value.map((e) => ({
    entry: e,
    groups: (e.meta?.groups ?? [])
      .map((g) => ({
        name: g.name,
        tags: needle.value
          ? g.tags.filter(
              (t) =>
                t.name.toLowerCase().includes(needle.value) ||
                t.value.toLowerCase().includes(needle.value),
            )
          : g.tags,
      }))
      .filter((g) => g.tags.length > 0),
  })),
)
const matches = computed(() => shown.value.reduce((n, s) => n + s.groups.reduce((g, x) => g + x.tags.length, 0), 0))

/** The browser's guess from the extension, beside what the bytes turned out to
 *  be. Shown only when the two do not obviously agree: a .png that is really a
 *  JPEG is worth noticing, "image/jpeg" next to "JPEG" is not. The test errs
 *  toward showing - an extra fact costs a line, a hidden one costs the point. */
function mimeDisagrees(format: string | null, mime: string): boolean {
  if (!mime) return false
  if (!format) return true
  const sub = (mime.split('/').pop() ?? '').toLowerCase().replace(/[^a-z0-9]/g, '')
  const f = format.toLowerCase().replace(/[^a-z0-9]/g, '')
  // Every OOXML mime is a long vnd.openxmlformats-... string that shares no
  // substring with the format's own name, so it would disagree on every single
  // Word document without this.
  if (f === 'officeopenxml' && sub.includes('openxmlformats')) return false
  return !sub.includes(f) && !f.includes(sub)
}

/** The lead line: what it is, how big, how much it says, and who read it -
 *  the upstream-attribution principle applies to what a user sees, not only to
 *  licence files. */
function summary(m: AttachmentMetadata): string {
  const bits: string[] = [m.format ?? 'Not a format we read']
  if (mimeDisagrees(m.format, m.mime)) bits.push(m.mime)
  if (m.size) bits.push(fmtFileSize(m.size))
  const n = m.groups.reduce((a, g) => a + g.tags.length, 0)
  if (n) bits.push(n === 1 ? '1 field' : `${n} fields`)
  if (m.reader !== 'none') bits.push(`read by ${m.reader}`)
  return bits.join(' · ')
}

// Group names come from the reader and the file, so the set is open and most
// of them say what they are. Composite is the one that does not: those values
// are not in the file at all, which changes what they mean.
const GROUP_NOTES: Record<string, string> = {
  Composite: 'Derived from the fields above - not stored in the file.',
}

// Rows are keyed by POSITION, not by tag name. sift guards against duplicate
// names within a group and its full-corpus scan is green, but 22 files in that
// corpus are DOCUMENTED exceptions where the file itself says a name twice - a
// second conflicting EXIF IFD, one Padding per IFD - and a duplicate Vue key
// is a render bug waiting for one of them to arrive as an attachment.

/** A fact about the file that is a SECURITY fact, not a descriptive one. A PDF
 *  that carries script is something a local-first product should say out loud
 * rather than leave as row 24 of 27. sift only emits the tag when it
 *  finds /Names->/JavaScript or a JavaScript /OpenAction, so its presence is
 *  the whole message. */
function flagged(group: string, t: FileMetaTag): boolean {
  return group === 'PDF' && t.name === 'JavaScript'
}
</script>

<template>
  <div class="pv__pane fmp">
    <div v-if="total >= FILTER_FROM" class="fmp__filter">
      <input
        v-model="q"
        class="pk-input pk-input--sm fmp__q"
        type="search"
        placeholder="Filter fields"
        aria-label="Filter fields"
      />
      <span v-if="needle" class="fmp__count">{{ matches }} of {{ total }}</span>
    </div>

    <div class="fmp__scroll">
      <section v-for="s in shown" :key="s.entry.id" class="fmp__file">
        <h3 class="fmp__name">{{ s.entry.name || 'File' }}</h3>
        <p v-if="s.entry.meta" class="fmp__summary">{{ summary(s.entry.meta) }}</p>

        <PhotoLocation v-if="s.entry.meta?.location" :location="s.entry.meta.location" />

        <p v-if="s.entry.error" class="fmp__note">{{ s.entry.error }}</p>
        <p v-else-if="!s.entry.meta?.groups.length" class="fmp__note">
          This file carries nothing about itself. Screenshots and re-encoded exports usually don't.
        </p>
        <p v-else-if="!s.groups.length" class="fmp__note">No field matches "{{ q.trim() }}".</p>

        <div v-for="g in s.groups" :key="g.name" class="fmp__group">
          <h4 class="fmp__gname">{{ g.name }}</h4>
          <p v-if="GROUP_NOTES[g.name]" class="fmp__gnote">{{ GROUP_NOTES[g.name] }}</p>
          <dl class="fmp__rows">
            <template v-for="(t, ti) in g.tags" :key="ti">
              <dt class="fmp__k">{{ t.name }}</dt>
              <dd class="fmp__v" :class="{ 'fmp__v--flag': flagged(g.name, t) }">
                <Icon v-if="flagged(g.name, t)" name="alert-triangle" :size="13" />
                <span>{{ t.value }}</span>
                <span v-if="t.truncated" class="fmp__cut">...value continues</span>
              </dd>
            </template>
          </dl>
        </div>
      </section>

      <p v-if="!loading && !stored.length" class="fmp__note fmp__note--only">
        No stored copy of this file. It was sent before originals were kept.
      </p>
    </div>

    <div v-if="loading" class="pv__overlay-msg">
      <Icon name="spinner" :size="22" class="pv__spin" />
      <span>Reading...</span>
    </div>
  </div>
</template>

<style scoped>
.fmp {
  display: flex;
  flex-direction: column;
}
/* The filter sits outside the scroll so it stays put through 121 rows. Its own
   band is a surface so the field is never bare on the pane's recessed ground
   (the sf__card recipe: card on surface, field steps to base). */
.fmp__filter {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
  padding: 10px 18px;
  border-bottom: 1px solid var(--pk-border-subtle);
  background: var(--pk-bg-surface);
}
.fmp__q {
  flex: 1;
  min-width: 0;
}
.fmp__count {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.fmp__scroll {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 14px 18px 24px;
}
.fmp__file + .fmp__file {
  margin-top: 22px;
  padding-top: 18px;
  border-top: 1px solid var(--pk-border-default);
}
.fmp__name {
  margin: 0 0 2px;
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
  overflow-wrap: anywhere;
}
.fmp__summary {
  margin: 0 0 14px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.fmp__note {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
}
.fmp__note--only {
  padding: 24px 0;
  text-align: center;
}
.fmp__group + .fmp__group {
  margin-top: 16px;
}
.fmp__gname {
  margin: 0 0 6px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--pk-text-secondary);
}
.fmp__gnote {
  margin: -2px 0 6px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
/* Two columns that both keep their share: a run-together tag name
   ("RedToneReproductionCurve") must wrap inside its own column rather than
   push the values off the pane, which at 560px it otherwise would. */
.fmp__rows {
  display: grid;
  grid-template-columns: minmax(0, 34%) minmax(0, 1fr);
  gap: 0;
  margin: 0;
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  overflow: hidden;
}
.fmp__k,
.fmp__v {
  margin: 0;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}
.fmp__k {
  color: var(--pk-text-muted);
  border-right: 1px solid var(--pk-border-subtle);
}
.fmp__v {
  color: var(--pk-text-primary);
  font-family: var(--pk-font-mono, ui-monospace, monospace);
}
/* Zebra by ROW, which in a dt/dd grid means every other PAIR - nth-child counts
   CELLS, so one striped row is the 4n+1 key next to the 4n+2 value. */
.fmp__k:nth-child(4n + 1),
.fmp__v:nth-child(4n + 2) {
  background: var(--pk-bg-base);
}
.fmp__v--flag {
  display: flex;
  align-items: baseline;
  gap: 6px;
  color: var(--pk-status-warning);
  background: color-mix(in srgb, var(--pk-status-warning) 12%, transparent);
}
/* the icon is the only child that must not sit on the text baseline */
.fmp__v--flag svg {
  align-self: center;
  flex-shrink: 0;
}
.fmp__cut {
  margin-left: 6px;
  color: var(--pk-text-muted);
  font-family: var(--pk-font-sans);
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
}
</style>
