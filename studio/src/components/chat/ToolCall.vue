<script setup lang="ts">
// One MCP tool call in the assistant turn: the model's arguments in, the tool's
// result (or an approve/deny gate) out. Approval-gated calls pause the stream
// server-side until the user decides here.
import { computed, ref } from 'vue'
import type { McpCall } from '@/types/chat'
import { approvalsApi } from '@/lib/api'
import Icon from '@/components/Icon.vue'
import { useArtifactsStore } from '@/stores/artifacts'

const props = defineProps<{ call: McpCall }>()

const open = ref(false)
const deciding = ref(false)
const decideError = ref('')

// An artifact call names a thing the panel can show, so the card doubles as a
// link to it - clicking any of them brings that artifact up. Without this the
// only way between several artifacts was the tab strip, which is easy to miss
// and only appears once there is more than one. The id is in
// the arguments for edits and in the result line for a create.
const artifacts = useArtifactsStore()
const artifactId = computed(() => {
  if (!/^artifacts__artifact_|^artifact_/.test(props.call.name ?? '')) return ''
  const hay = `${props.call.arguments ?? ''} ${props.call.output ?? ''}`
  return hay.match(/art_[0-9a-f]{12}/)?.[0] ?? ''
})
function openArtifact(): void {
  if (artifactId.value) artifacts.show(artifactId.value)
}

const isPending = computed(() => props.call.status === 'pending')
const isRunning = computed(() => props.call.status === 'in_progress')

// The wire name is a raw tool IDENTIFIER - `mcp_search_tools`,
// `artifacts__artifact_create`, sometimes camelCase - and reading "Ran
// mcp_search_tools" in a chat is being handed somebody else's source code
// Say it in words. The exact id is not lost: it sits in
// the expanded card and in the approval gate, where debugging and deciding
// actually happen.
//
//   mcp_search_tools            ->  Search tools
//   artifacts__artifact_create  ->  Artifact create
//   getUserProfile              ->  Get User Profile
//
// Case is SPLIT, never folded: lowercasing to get a tidy "get user profile"
// also turns readCSV into "Read csv", and a mangled acronym is worse than a
// capital letter in the middle of a sentence.
function humanize(raw: string, server: string): string {
  let s = raw
  // `server__tool` is MCP's namespacing and the chip beside this already
  // names the server, so the prefix is pure repetition - but only drop it
  // when there is a chip to carry it. Non-greedy: strip exactly one level.
  if (server) s = s.replace(/^[A-Za-z0-9][\w-]*?__/, '')
  s = s.replace(/^mcp_/, '') // our own protocol noise, meaningless to a reader
  s = s.replace(/([a-z0-9])([A-Z])/g, '$1 $2') // camelCase word breaks
  s = s.replace(/[_-]+/g, ' ').trim()
  return s ? s[0].toUpperCase() + s.slice(1) : raw
}
const prettyName = computed(() => humanize(props.call.name ?? '', props.call.serverLabel ?? ''))

// The verb earns its place only when something is in FLIGHT or is being ASKED
// of you. On a settled call it is narration: the status icon already says what
// happened (check / triangle / x), the tag says "error" or "denied", and three
// finished calls in a row just stack "Ran / Ran / Ran" down the gutter.
//
// Pending keeps its words absolutely: an approval gate must never describe the
// thing you are approving with an icon alone. Running keeps them because a wait
// deserves a word, and only one call is ever in flight, so it cannot repeat.
const verb = computed(() => (isPending.value ? 'Wants to run' : isRunning.value ? 'Running' : ''))

const statusIcon = computed(() => {
  switch (props.call.status) {
    case 'completed':
      return 'check'
    case 'failed':
      return 'alert-triangle'
    case 'denied':
      return 'x'
    case 'pending':
      return 'shield'
    default:
      return 'wrench'
  }
})

/** Pretty-print a JSON string; fall back to the raw text if it isn't JSON. */
function pretty(s: string): string {
  try {
    return JSON.stringify(JSON.parse(s), null, 2)
  } catch {
    return s
  }
}

/** Arguments as a person reads them, not as the wire spells them: each key on
 *  its own line, multi-line string values (a cypher query, an artifact body)
 *  as indented BLOCKS - never 
-escaped one-liners. */
function prettyArgs(s: string | undefined): string {
  if (!s) return ''
  try {
    const v: unknown = JSON.parse(s)
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      return Object.entries(v as Record<string, unknown>)
        .map(([k, val]) => {
          if (typeof val === 'string') {
            return val.includes('\n')
              ? `${k}:\n  ${val.split('\n').join('\n  ')}`
              : `${k}: ${val}`
          }
          return `${k}: ${JSON.stringify(val, null, 2)}`
        })
        .join('\n')
    }
    return JSON.stringify(v, null, 2)
  } catch {
    return s
  }
}

/** The result as CONTENT: MCP's array-of-text wrapper is unwrapped, and a
 *  text part that is itself JSON gets pretty-printed instead of arriving
 *  as one long escaped string. */
function prettyOutput(s: string | undefined): string {
  if (!s) return ''
  try {
    const v: unknown = JSON.parse(s)
    if (
      Array.isArray(v) &&
      v.length > 0 &&
      v.every((p) => p && typeof p === 'object' && typeof (p as { text?: unknown }).text === 'string')
    ) {
      return v.map((p) => pretty((p as { text: string }).text)).join('\n')
    }
    return JSON.stringify(v, null, 2)
  } catch {
    return s
  }
}
const argsPretty = computed(() => prettyArgs(props.call.arguments))
const outputPretty = computed(() => prettyOutput(props.call.output))

async function decide(approve: boolean): Promise<void> {
  const id = props.call.approvalId
  if (!id || deciding.value) return
  deciding.value = true
  decideError.value = ''
  try {
    await approvalsApi.approve(id, approve)
    // The stream resumes and drives the card's status from here.
  } catch (e) {
    decideError.value = e instanceof Error ? e.message : String(e)
  } finally {
    deciding.value = false
  }
}
</script>

<template>
  <div class="tc" :class="`tc--${call.status}`">
    <div class="tc__headrow">
      <button class="tc__head" type="button" @click="open = !open">
        <span class="tc__icon">
          <Icon v-if="isRunning" name="spinner" :size="14" class="tc__spin" />
          <Icon v-else :name="statusIcon" :size="14" />
        </span>
        <span class="tc__title">
          <span v-if="verb" class="tc__verb">{{ verb }}</span>
          <span class="tc__name">{{ prettyName }}</span>
          <span class="tc__server">{{ call.serverLabel }}</span>
        </span>
        <span v-if="call.status === 'denied'" class="tc__tag tc__tag--denied">denied</span>
        <span v-else-if="call.status === 'failed'" class="tc__tag tc__tag--failed">error</span>
        <Icon name="chevron-down" :size="14" class="tc__chev" :class="{ 'tc__chev--open': open }" />
      </button>
      <button v-if="artifactId" type="button" class="tc__open" @click="openArtifact">Show</button>
    </div>

    <!-- approval gate -->
    <div v-if="isPending" class="tc__approval">
      <div class="tc__section">
        <span class="tc__section-label">Tool</span>
        <code class="tc__id">{{ call.name }}</code>
      </div>
      <pre class="tc__code tc__code--args">{{ argsPretty }}</pre>
      <div class="tc__approve-row">
        <span class="tc__approve-q">Run this tool?</span>
        <div class="tc__approve-btns">
          <button class="pk-btn pk-btn--sm" :disabled="deciding" @click="decide(false)">Deny</button>
          <button class="pk-btn pk-btn--sm pk-btn--primary" :disabled="deciding" @click="decide(true)">
            Approve
          </button>
        </div>
      </div>
      <div v-if="decideError" class="tc__err">{{ decideError }}</div>
    </div>

    <!-- expanded details (args + result) -->
    <div v-else-if="open" class="tc__body">
      <div class="tc__section">
        <span class="tc__section-label">Tool</span>
        <code class="tc__id">{{ call.name }}</code>
      </div>
      <div class="tc__section">
        <span class="tc__section-label">Arguments</span>
        <pre class="tc__code">{{ argsPretty || '-' }}</pre>
      </div>
      <div v-if="call.output != null" class="tc__section">
        <span class="tc__section-label">{{ call.error ? 'Error' : 'Result' }}</span>
        <pre class="tc__code" :class="{ 'tc__code--err': call.error }">{{ outputPretty }}</pre>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tc {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  overflow: hidden;
}
.tc--pending {
  border-color: var(--pk-status-warning);
}
.tc--failed,
.tc--denied {
  border-color: var(--pk-border-subtle);
}
/* "Show" is a sibling of the disclosure button, not a child of it: an
   interactive element inside a <button> is invalid and leaves the inner
   control unreachable to a screen reader (it was a span+role="button" before,
   which has the same problem). The row carries the padding and the hover so
   the pair still reads as one strip. */
/* Gutter scale, shared with ThinkingFold and WebSearchCall: 7px/10px padding
   around a 16px content row (13px text). These three stack in the same strip
   above an answer, so they have to read as one quiet family - the tool cards
   were at 9px/12px with 14px text, which measured 22.5px of content against
   thinking's 16px and made the tool strip the loudest thing in the turn.
   Anything that could out-grow 16px - the server chip,
   the status tag - carries an explicit line-height below. */
.tc__headrow {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 7px 10px;
}
.tc__headrow:hover {
  background: var(--pk-bg-hover);
}
.tc__head {
  display: flex;
  align-items: center;
  gap: 7px;
  flex: 1;
  min-width: 0;
  padding: 0;
  border: none;
  background: transparent;
  color: var(--pk-text-primary);
  cursor: pointer;
  text-align: left;
  font: inherit;
}
.tc__icon {
  display: inline-flex;
  color: var(--pk-text-muted);
  flex-shrink: 0;
}
.tc--completed .tc__icon {
  color: var(--pk-status-success);
}
.tc--pending .tc__icon {
  color: var(--pk-status-warning);
}
.tc--failed .tc__icon,
.tc--denied .tc__icon {
  color: var(--pk-text-danger);
}
/* center, not baseline: baseline lets a padded chip drag the whole line box
   taller than its own text, which is where the extra height came from */
.tc__title {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
  flex: 1;
}
.tc__verb {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
}
/* a real <button>: it was a span+role=button+tabindex, so Space did nothing */
.tc__open {
  padding: 0 8px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-sm);
  background: transparent;
  color: var(--pk-text-muted);
  font: inherit;
  font-size: 11px;
  /* 14 + 2 borders = the row's 16px: this button only appears on artifact
     calls, and at its old 2px padding it made THOSE rows taller than the
     rest of the strip */
  line-height: 14px;
  cursor: pointer;
}
.tc__open:hover {
  color: var(--pk-text-primary);
}
/* sans, not mono: the name is words now, and a monospace face was half of
   what made it read as somebody's source code */
.tc__name {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.tc__server {
  font-size: 11px;
  line-height: 16px;
  color: var(--pk-text-muted);
  background: var(--pk-bg-inset);
  padding: 0 6px;
  border-radius: var(--pk-radius-sm);
  flex-shrink: 0;
}
.tc__tag {
  font-size: 10px;
  line-height: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  padding: 2px 6px;
  border-radius: var(--pk-radius-sm);
  flex-shrink: 0;
}
.tc__tag--denied,
.tc__tag--failed {
  background: var(--pk-bg-danger-subtle);
  color: var(--pk-text-danger);
}
.tc__chev {
  color: var(--pk-text-muted);
  flex-shrink: 0;
  transition: transform 0.15s ease;
}
.tc__chev--open {
  transform: rotate(180deg);
}
.tc__spin {
  animation: tc-spin 0.8s linear infinite;
}
.tc__approval {
  padding: 0 10px 10px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.tc__approve-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.tc__approve-q {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
.tc__approve-btns {
  display: flex;
  gap: 8px;
}
.tc__body {
  padding: 0 10px 10px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.tc__section {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.tc__section-label {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
/* the exact wire identifier, which the head no longer shows. This is what a
   bug report quotes and what you check before approving a call, so it stays
   plain text you can select - not a heading, not a chip. */
.tc__id {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  word-break: break-all;
}
.tc__code {
  margin: 0;
  padding: 9px 11px;
  background: var(--pk-bg-inset);
  border-radius: var(--pk-radius-sm);
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  line-height: 1.5;
  color: var(--pk-text-secondary);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 260px;
  overflow: auto;
}
.tc__code--err {
  color: var(--pk-text-danger);
}
.tc__err {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-danger);
}
@keyframes tc-spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
