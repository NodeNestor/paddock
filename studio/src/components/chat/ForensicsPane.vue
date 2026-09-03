<script setup lang="ts">
// "Forensics": the forensic pass's verdict on the FILE's own bytes - the other panel in
// the document side view, next to "Metadata" and "Model metadata".
//
// Where Metadata is descriptive (what the file says about itself), this is
// evidentiary: signal-level analysis of the ORIGINAL upload - Error Level
// Analysis, resampling and splice traces, and (for PDFs) a render-vs-scan
// comparison - collapsed into a risk verdict, headline findings and a
// plain-language explanation. Forensics is VLM-coupled: these same findings were
// injected into the chat so the vision model examined the flagged pixels, and
// this panel is the human's view of that evidence.
//
// Read from the MANAGER's stored row (GET /api/attachments/{id}/forensics), so
// it works with the runner stopped - same as the metadata pane. Unlike
// metadata, the report is PERSISTED (it is GPU-expensive to derive), so a `null`
// answer means "never analyzed", distinct from an error. A CLEAN report (zero
// findings, risk "info") is a real, stored verdict, not an empty state.
//
// A document can be more than one file (a turn that carried a PDF and a photo is
// one document), so this takes a list and answers per file - same contract as
// FileMetaPane.
import { computed, ref, watch } from 'vue'
import { forensicsApi, type ForensicReport } from '@/lib/api'
import Icon from '@/components/Icon.vue'

const props = defineProps<{
  /** The file(s) the document is made of. Parts with no stored blob are
   *  skipped - there is nothing to look up. */
  parts: { attachmentId?: string; name?: string }[]
}>()

interface Entry {
  id: string
  name: string
  report: ForensicReport | null
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
          return { id, name: p.name || '', report: await forensicsApi.forAttachment(id), error: null }
        } catch (e) {
          // One unreadable report is not the whole answer: the others still show.
          return {
            id,
            name: p.name || '',
            report: null,
            error: e instanceof Error ? e.message : String(e),
          }
        }
      }),
    )
    if (mine !== gen) return
    entries.value = out
    loading.value = false
  },
  { immediate: true },
)

// Severity -> a visual band. Findings grade info < low < medium < high <
// critical; anything unrecognized reads as info (the calmest, never alarmist).
const SEV_RANK: Record<string, number> = { info: 0, low: 1, medium: 2, high: 3, critical: 4 }
function sevClass(sev: string): string {
  const s = (sev || 'info').toLowerCase()
  if (SEV_RANK[s] === undefined) return 'fx__sev--info'
  return `fx__sev--${s}`
}
function sevLabel(sev: string): string {
  const s = (sev || 'info').toLowerCase()
  return s.charAt(0).toUpperCase() + s.slice(1)
}
/** A confidence in [0,1] as a whole-percent string. */
function pct(c: number): string {
  return `${Math.round(c * 100)}%`
}
/** Worst first, then most-confident first.
 *
 *  The report lists key findings in analyzer order, which is an implementation
 *  detail - read top-down it buries a critical splice under three info-level
 *  notes. Copied before sorting: this is a prop straight off the stored report
 *  and sorting in place would mutate it. */
function bySeverity<T extends { severity: string; confidence: number }>(list: T[]): T[] {
  const rank = (s: string) => SEV_RANK[(s || 'info').toLowerCase()] ?? 0
  return [...list].sort(
    (a, b) => rank(b.severity) - rank(a.severity) || b.confidence - a.confidence,
  )
}

/** The lead line under a file's name: what it is, its raster format/size, and
 *  how the analysis ran. */
function summary(r: ForensicReport): string {
  const bits: string[] = []
  if (r.content_type) bits.push(r.content_type)
  if (r.format) bits.push(r.format.toUpperCase())
  if (r.width && r.height) bits.push(`${r.width}×${r.height}`)
  bits.push(r.gpu ? 'GPU' : 'CPU')
  if (r.elapsed_ms) bits.push(`${r.elapsed_ms} ms`)
  return bits.join(' · ')
}
</script>

<template>
  <div class="pv__pane fx">
    <div class="fx__scroll">
      <section v-for="e in entries" :key="e.id" class="fx__file">
        <h3 class="fx__name">{{ e.name || 'File' }}</h3>

        <p v-if="e.error" class="fx__note">{{ e.error }}</p>
        <p v-else-if="!e.report" class="fx__note">
          Not analyzed. Forensics didn't run on this attachment - turn it on for the
          endpoint (Intelligence) or per request in the composer, then resend.
        </p>

        <template v-else>
          <p class="fx__summary">{{ summary(e.report) }}</p>

          <!-- The verdict band: risk level + the one-line verdict. -->
          <div class="fx__verdict" :class="sevClass(e.report.risk_level)">
            <span class="fx__badge">{{ sevLabel(e.report.risk_level) }}</span>
            <span class="fx__vtext">{{ e.report.verdict }}</span>
          </div>

          <p v-if="e.report.explanation.summary" class="fx__expl">
            {{ e.report.explanation.summary }}
          </p>
          <p v-if="e.report.explanation.anti_forensics_warning" class="fx__expl fx__expl--warn">
            <Icon name="alert-triangle" :size="13" />
            {{ e.report.explanation.anti_forensics_warning }}
          </p>
          <p v-if="e.report.explanation.cross_corroboration" class="fx__expl">
            {{ e.report.explanation.cross_corroboration }}
          </p>
          <p v-if="e.report.explanation.visual_review" class="fx__expl">
            {{ e.report.explanation.visual_review }}
          </p>

          <!-- A clean report is a real verdict, not an empty state. -->
          <p v-if="!e.report.key_findings.length && !e.report.findings.length" class="fx__note">
            No forensic signals - analyzed, nothing found.
          </p>

          <!-- Headline findings: the deduplicated, human-titled signals.
               Worst first, and each one a card carrying its severity as a
               coloured edge - the report emits them in analyzer order, which
               reads as a pile rather than a ranking, and an 8px dot was the
               only thing separating "critical" from "info". -->
          <div v-if="e.report.key_findings.length" class="fx__group">
            <h4 class="fx__gname">Key findings</h4>
            <div class="fx__cards">
              <div
                v-for="(k, ki) in bySeverity(e.report.key_findings)"
                :key="ki"
                class="fx__card"
                :class="sevClass(k.severity)"
              >
                <div class="fx__crow">
                  <span class="fx__sevtag">{{ sevLabel(k.severity) }}</span>
                  <span class="fx__ctitle">{{ k.title }}</span>
                </div>
                <p v-if="k.description" class="fx__cdesc">{{ k.description }}</p>
                <div class="fx__cmeta">
                  <span>{{ pct(k.confidence) }} confidence</span>
                  <span v-if="k.count > 1">· {{ k.count }} occurrences</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Grouped explanation by family. -->
          <div v-if="e.report.explanation.categories.length" class="fx__group">
            <h4 class="fx__gname">Explanation</h4>
            <div v-for="(c, ci) in e.report.explanation.categories" :key="ci" class="fx__finding">
              <div class="fx__frow">
                <span class="fx__dot" :class="sevClass(c.max_severity)" />
                <span class="fx__ftitle">{{ c.name }}</span>
                <span class="fx__fconf">{{ c.finding_count }} signals</span>
              </div>
              <p v-if="c.explanation" class="fx__fdesc">{{ c.explanation }}</p>
            </div>
          </div>

          <!-- Every raw signal, for the reader who wants the whole picture. -->
          <div v-if="e.report.findings.length" class="fx__group">
            <h4 class="fx__gname">All signals ({{ e.report.findings.length }})</h4>
            <dl class="fx__rows">
              <template v-for="(f, fi) in e.report.findings" :key="fi">
                <dt class="fx__k">
                  <span class="fx__dot" :class="sevClass(f.severity)" />
                  {{ f.analyzer }} · {{ f.code }}
                </dt>
                <dd class="fx__v">
                  <span>{{ f.description }}</span>
                  <span class="fx__fconf">{{ pct(f.confidence) }}</span>
                </dd>
              </template>
            </dl>
          </div>
        </template>
      </section>

      <p v-if="!loading && !stored.length" class="fx__note fx__note--only">
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
.fx {
  display: flex;
  flex-direction: column;
}
.fx__scroll {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 14px 18px 24px;
}
.fx__file + .fx__file {
  margin-top: 22px;
  padding-top: 18px;
  border-top: 1px solid var(--pk-border-default);
}
.fx__name {
  margin: 0 0 2px;
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
  overflow-wrap: anywhere;
}
.fx__summary {
  margin: 0 0 12px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.fx__note {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
}
.fx__note--only {
  padding: 24px 0;
  text-align: center;
}
/* The verdict band, tinted by the risk level. */
.fx__verdict {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--pk-radius-md);
  border: 1px solid var(--pk-border-subtle);
  background: color-mix(in srgb, var(--fx-tone, var(--pk-text-muted)) 12%, transparent);
  border-color: color-mix(in srgb, var(--fx-tone, var(--pk-border-subtle)) 40%, transparent);
  margin-bottom: 12px;
}
.fx__badge {
  flex-shrink: 0;
  font-size: var(--pk-font-size-xs);
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--fx-tone, var(--pk-text-secondary));
}
.fx__vtext {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
  overflow-wrap: anywhere;
}
.fx__expl {
  display: flex;
  align-items: baseline;
  gap: 6px;
  margin: 0 0 10px;
  font-size: var(--pk-font-size-sm);
  line-height: 1.5;
  color: var(--pk-text-secondary);
}
.fx__expl--warn {
  color: var(--pk-status-warning);
}
.fx__expl--warn svg {
  align-self: center;
  flex-shrink: 0;
}
.fx__group {
  margin-top: 16px;
}
.fx__gname {
  margin: 0 0 8px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--pk-text-secondary);
}
.fx__finding + .fx__finding {
  margin-top: 10px;
}
/* Key findings as cards. Three loose lines per finding - title row, indented
   description, indented count - ran together with the next finding at a 10px
   gap and no edge, so a list of five read as one block of text. A card with a
   severity-toned left edge gives each finding a boundary AND puts severity in
   the layout instead of only in an 8px dot. */
.fx__cards {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.fx__card {
  padding: 9px 11px 9px 10px;
  border: 1px solid var(--pk-border-default);
  border-left: 3px solid var(--fx-tone, var(--pk-text-muted));
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
}
.fx__crow {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 4px 8px;
}
/* The severity word, not just a hue: colour alone fails for the ~8% of men with
   a red/green deficiency, and "Critical" vs "Info" is the single most important
   thing on the row. */
.fx__sevtag {
  flex-shrink: 0;
  padding: 0 6px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--fx-tone, var(--pk-text-muted)) 16%, transparent);
  color: var(--fx-tone, var(--pk-text-muted));
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.fx__ctitle {
  flex: 1;
  min-width: 0;
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
  overflow-wrap: anywhere;
}
.fx__cdesc {
  margin: 5px 0 0;
  font-size: var(--pk-font-size-xs);
  line-height: 1.5;
  color: var(--pk-text-secondary);
}
/* Confidence and count were a stray right-aligned number and a whole line of
   their own; together on one muted line they read as what they are - footnotes
   about the finding above. */
.fx__cmeta {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: 5px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.fx__frow {
  display: flex;
  align-items: center;
  gap: 8px;
}
.fx__dot {
  flex-shrink: 0;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--fx-tone, var(--pk-text-muted));
}
.fx__ftitle {
  flex: 1;
  min-width: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
  overflow-wrap: anywhere;
}
.fx__fconf {
  flex-shrink: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.fx__fdesc {
  margin: 3px 0 0 16px;
  font-size: var(--pk-font-size-xs);
  line-height: 1.5;
  color: var(--pk-text-secondary);
}
.fx__rows {
  display: grid;
  grid-template-columns: minmax(0, 40%) minmax(0, 1fr);
  gap: 0;
  margin: 0;
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  overflow: hidden;
}
.fx__k,
.fx__v {
  margin: 0;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}
.fx__k {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--pk-text-muted);
  border-right: 1px solid var(--pk-border-subtle);
  font-family: var(--pk-font-mono, ui-monospace, monospace);
}
.fx__v {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  color: var(--pk-text-primary);
}
.fx__k:nth-child(4n + 1),
.fx__v:nth-child(4n + 2) {
  background: var(--pk-bg-base);
}
/* Severity tones - set the local --fx-tone the band, badge and dots read. */
.fx__sev--info {
  --fx-tone: var(--pk-text-muted);
}
.fx__sev--low {
  --fx-tone: var(--pk-accent, #4a90d9);
}
.fx__sev--medium {
  --fx-tone: var(--pk-status-warning);
}
.fx__sev--high {
  --fx-tone: var(--pk-status-danger, #d9534f);
}
.fx__sev--critical {
  --fx-tone: var(--pk-status-danger, #d9534f);
}
</style>
