<script setup lang="ts">
import { computed, ref } from 'vue'
import type { Message } from '@/types/chat'
import { messageText } from '@/types/chat'
import { disagreements, transcriptWords } from '@/lib/transcript-diff'
import { useChatStore } from '@/stores/chat'
import { useStickyScroll } from '@/composables/useStickyScroll'
import { previewPlan } from '@/composables/useChatStream'
import { fleetLabel, fleetVendor } from '@/lib/model-name'
import MessageBubble from './MessageBubble.vue'
import SiblingSwitch from './SiblingSwitch.vue'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import { activeSteps, siblingInfo, stepMessages } from '@/lib/tree'
import type { ContentPart } from '@/types/chat'

const chat = useChatStore()
const emit = defineEmits<{
  regenerate: []
  continueReply: []
  edit: [id: string, parts: ContentPart[]]
}>()

// Lane header labels come from lib/model-name so the artifact panel's pane
// badges - which sit directly under these in a compare - cannot drift.
const laneName = fleetLabel
const laneVendor = fleetVendor

// What the send path will do with the oldest messages - surfaced here as a
// divider so nothing disappears (or gets summarized) silently. Mirrors the
// exact plan the next send uses - including server-side compaction, whose
// stored summary shows here the same way the client one always has.
const plan = computed(() => previewPlan(chat.active))
const trimFrom = computed(() => plan.value.from)
const showSummary = ref(false)

// The thread as BLOCKS, taken from the ACTIVE PATH rather than the raw array -
// a conversation holds every branch and this renders exactly one of them. A
// compare fan-out is one step, so it still arrives here as a single block and
// the split view below is unchanged.
//
// `i` is the block's first message index in the PATH, which is the same space
// `previewPlan`'s `from` counts in (lib/tokens.ts measures the identical
// array) - the trim divider would sit at the wrong turn if either side
// counted the other.
type Block = { kind: 'msg'; m: Message; i: number } | { kind: 'group'; ms: Message[]; i: number }
const blocks = computed<Block[]>(() => {
  const conv = chat.active
  if (!conv) return []
  const out: Block[] = []
  let i = 0
  for (const s of activeSteps(conv)) {
    out.push(s.kind === 'msg' ? { kind: 'msg', m: s.m, i } : { kind: 'group', ms: s.ms, i })
    i += stepMessages(s).length
  }
  return out
})
const hasGroups = computed(() => blocks.value.some((b) => b.kind === 'group'))

/** Which blocks have alternatives, and where you are among them. Keyed by the
 *  step's anchor id - the same id `selectSibling` walks from. Absent means one
 *  version, which is the overwhelmingly common case and draws no control. */
const sibs = computed(() => {
  const out = new Map<string, { index: number; count: number }>()
  const conv = chat.active
  if (!conv) return out
  for (const b of blocks.value) {
    const anchor = b.kind === 'msg' ? b.m : b.ms[0]
    const info = siblingInfo(conv, anchor.id)
    if (info && info.steps.length > 1) {
      out.set(anchor.id, { index: info.index, count: info.steps.length })
    }
  }
  return out
})

// Where the LANES DISAGREE, per compare block. Computed here because it is the
// only place that can see more than one lane - a MessageBubble knows its own
// turn and nothing else.
//
// For a transcription compare this is the answer to the question the panel
// asks. Each model's own confidence is scaled differently, so counting marks
// across lanes compares their logprob distributions; two models DIFFERING on a
// word is symmetric evidence that one of them is wrong, and the listener
// settles it. Keyed by message id so a lane finds its own set.
const laneDiffs = computed(() => {
  const out = new Map<string, Set<number>>()
  for (const b of blocks.value) {
    if (b.kind !== 'group' || b.ms.length < 2) continue
    // only transcription lanes: two chat models answering the same question
    // in different words is the POINT, not a disagreement about facts
    const lanes = b.ms.filter((m) => m.transcript)
    if (lanes.length < 2) continue
    const words = lanes.map((m) => transcriptWords(m.transcript, messageText(m)).map((w) => w.word))
    disagreements(words).forEach((set, i) => out.set(lanes[i].id, set))
  }
  return out
})
const EMPTY: Set<number> = new Set()

// Which LANE WON, when there was a race to win. Same reason as `laneDiffs` for
// living here: only the block sees more than one lane.
//
// Three conditions, and all three are about whether the comparison is sound
// rather than whether it is available:
//
//  · TRANSCRIPTION lanes only. The same clip is the same work, so finishing
//    first means something. Two chat models answering the same question is not
//    the same work - one writes three lines and the other thirty, and a
//    wall-clock winner would just be rewarding brevity.
//  · nothing CONTENDED. Local lanes fired together share the card, so their
//    times measure how they crowded each other (already stamped and shown
//    per-lane as "shared GPU"). A cloud lane in that same block shares nothing,
//    so a badge would systematically flatter it.
//  · FINISHED and TIMED. A lane still streaming has not run its race.
//
// When any of them fails there is no badge and no substitute - the per-lane
// "× realtime" still stands on its own, which is the whole reason for
// preferring it to a ratio.
const fastestLane = computed(() => {
  const out = new Set<string>()
  for (const b of blocks.value) {
    if (b.kind !== 'group' || b.ms.length < 2) continue
    if (b.ms.some((m) => m.run?.contended)) continue
    const raced = b.ms.filter((m) => m.transcript && !m.streaming && (m.usage?.ms ?? 0) > 0)
    if (raced.length < 2) continue
    const best = raced.reduce((a, m) => (m.usage!.ms! < a.usage!.ms! ? m : a))
    // A dead heat is not a win. Two lanes within a few percent of each other
    // would otherwise hand the badge to whichever the network jittered in
    // front, and it would flip between runs of the same pair.
    const rest = raced.filter((m) => m !== best)
    if (rest.every((m) => m.usage!.ms! > best.usage!.ms! * 1.1)) out.add(best.id)
  }
  return out
})

const scrollEl = ref<HTMLElement | null>(null)
const contentEl = ref<HTMLElement | null>(null)
const { stuck, toBottom } = useStickyScroll(scrollEl, contentEl)
</script>

<template>
  <div class="threadwrap">
    <div ref="scrollEl" class="thread">
      <div ref="contentEl" class="thread__inner" :class="{ 'thread__inner--wide': hasGroups }">
        <template v-for="(b, bi) in blocks" :key="b.kind === 'msg' ? b.m.id : b.ms[0].id">
          <div v-if="b.i === trimFrom && trimFrom > 0" class="thread__trim">
            <!-- summarized: the older messages ride along as a summary -->
            <template v-if="plan.summary">
              <button class="thread__trim-btn" type="button" @click="showSummary = !showSummary">
                {{ trimFrom }} earlier message{{ trimFrom > 1 ? 's' : '' }} summarized for the model
                <Icon :name="showSummary ? 'chevron-up' : 'chevron-down'" :size="12" />
              </button>
              <div v-if="showSummary" class="thread__summary">
                <p class="thread__summary-body">{{ plan.summary }}</p>
                <p v-if="plan.by" class="thread__summary-by">Summarized by {{ plan.by }}</p>
              </div>
            </template>
            <!-- summaries off (or none yet): the old messages are simply dropped -->
            <span v-else>
              {{ trimFrom }} earlier message{{ trimFrom > 1 ? 's' : '' }} not sent to the model -
              trimmed to fit the context window
            </span>
          </div>
          <template v-if="b.kind === 'msg'">
            <MessageBubble
              :message="b.m"
              :is-last="bi === blocks.length - 1"
              @regenerate="emit('regenerate')"
              @continue="emit('continueReply')"
              @edit="(parts) => emit('edit', b.m.id, parts)"
            />
            <!-- Sits with the turn it belongs to and takes its side, so the
                 control for "other versions of THIS question" reads as part of
                 the question rather than of the answer under it. -->
            <div
              v-if="sibs.get(b.m.id)"
              class="thread__sibs"
              :class="{ 'thread__sibs--user': b.m.role === 'user' }"
            >
              <SiblingSwitch
                :index="sibs.get(b.m.id)!.index"
                :count="sibs.get(b.m.id)!.count"
                @go="(d) => chat.selectSibling(b.m.id, d)"
              />
            </div>
          </template>
          <!-- compare block: one lane per model, side by side (2-up; 3-4 wrap
               into a 2x2 grid). Lanes never share context - each model only
               ever saw its own column. -->
          <!-- One re-rolled compare run switches as a whole: every lane of run
               2 replaces every lane of run 1, because a half-swapped block
               would compare two different questions. -->
          <div v-else-if="sibs.get(b.ms[0].id)" class="thread__sibs thread__sibs--cmp">
            <SiblingSwitch
              :index="sibs.get(b.ms[0].id)!.index"
              :count="sibs.get(b.ms[0].id)!.count"
              @go="(d) => chat.selectSibling(b.ms[0].id, d)"
            />
          </div>
          <div v-if="b.kind === 'group'" class="thread__cmp">
            <section v-for="lm in b.ms" :key="lm.id" class="thread__lane">
              <header class="thread__lane-hd">
                <Tooltip :label="lm.model ?? ''">
                  <span class="thread__lane-model">
                    <VendorLogo v-if="laneVendor(lm.model)" :vendor="laneVendor(lm.model)!" :size="14" />
                    {{ laneName(lm.model) }}
                  </span>
                </Tooltip>
                <Tooltip
                  v-if="lm.run?.tools?.length"
                  :label="`Ran with MCP tools: ${lm.run.tools.join(', ')}`"
                >
                  <span class="thread__lane-note"><Icon name="plug" :size="11" /> tools</span>
                </Tooltip>
                <Tooltip
                  v-if="fastestLane.has(lm.id)"
                  label="Finished first on this clip - every model had the same audio to transcribe and nothing else was using the GPU"
                >
                  <span class="thread__lane-note thread__lane-note--won">fastest</span>
                </Tooltip>
                <Tooltip
                  v-if="lm.run?.contended"
                  label="These models ran at the same time on one GPU, so the speed here shows how much they slowed each other down, not how fast the model is"
                >
                  <span class="thread__lane-note">shared GPU</span>
                </Tooltip>
              </header>
              <MessageBubble
                :message="lm"
                :is-last="false"
                :stamp="false"
                :in-lane="true"
                :differs="laneDiffs.get(lm.id) ?? EMPTY"
              />
            </section>
          </div>
        </template>
      </div>

      </div>

    <Transition name="fade">
      <div v-if="!stuck" class="thread__jump">
        <Tooltip label="Scroll to latest" side="left">
          <button class="thread__jump-btn" type="button" @click="toBottom">
            <Icon name="arrow-down" :size="18" />
          </button>
        </Tooltip>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
/* Wrapper so the jump button can be a SIBLING of the scroller rather than a
   child of it. A scroll container scrolls its absolutely positioned
   descendants, so a button inside .thread drifts up the page as you scroll
   instead of staying put - which it has always done, just invisibly while the
   thread was short. Anchored here, it holds. */
.threadwrap {
  position: relative;
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}
.thread {
  position: relative;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
}
.thread__inner {
  max-width: var(--pk-chat-width);
  margin: 0 auto;
  /* The composer floats on this surface, so the last message needs its height
     clear underneath - --pk-composer-h is measured in ChatView, because the
     composer grows with multi-line input, the file tray and the hint line.
     The fallback only covers the frame before the first measurement. */
  padding: 28px 24px calc(var(--pk-composer-h, 96px) + 12px);
}
/* a chat with compare blocks earns more width - two real columns need it */
.thread__inner--wide {
  max-width: 1240px;
}

/* compare block: split lanes */
/* Lanes are COLUMNS. Always, at every width, however many there are. A fixed
   two-column grid put lane 3 underneath lane 1 and lane 4 underneath lane 2,
   which is not a comparison - it is two comparisons stacked, and nothing lines
   up across the fold. When the columns would drop below a
   readable floor the block scrolls sideways; it never folds. */
.thread__cmp {
  display: grid;
  grid-auto-flow: column;
  grid-auto-columns: minmax(240px, 1fr);
  gap: 12px;
  align-items: start;
  margin-bottom: 24px;
  overflow-x: auto;
}
.thread__lane {
  min-width: 0;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 10px 14px 4px;
}
.thread__lane-hd {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--pk-border-subtle);
}
.thread__lane-model {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.thread__lane-note {
  flex: none;
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 0 7px;
  border: 1px solid var(--pk-border-subtle);
  border-radius: 999px;
  font-size: 10px;
  color: var(--pk-text-muted);
}
.thread__lane-note--won {
  border-color: var(--pk-accent);
  color: var(--pk-accent-text);
}
/* Pulled up under the turn it belongs to: the bubble's own bottom margin
   would otherwise leave the control floating between two turns, reading as if
   it belonged to the next one. */
.thread__sibs {
  display: flex;
  margin: -14px 0 14px;
}
.thread__sibs--user {
  justify-content: flex-end;
}
.thread__sibs--cmp {
  justify-content: center;
  margin: 0 0 6px;
}
.thread__trim {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  margin: 2px 0 24px;
}
.thread__trim span,
.thread__trim-btn {
  padding: 4px 12px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-elevated);
  border: 1px solid var(--pk-border-subtle);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  text-align: center;
}
.thread__trim-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}
.thread__trim-btn:hover {
  color: var(--pk-text-secondary);
  border-color: var(--pk-border-default);
}
.thread__summary {
  max-width: 560px;
  padding: 12px 16px;
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-elevated);
  border: 1px solid var(--pk-border-subtle);
}
.thread__summary-body {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.55;
  white-space: pre-wrap;
}
.thread__summary-by {
  margin-top: 8px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
/* Clears the floating composer - the thread runs the full height of the pane
   now, so a plain 16px would park this button behind the input. */
/* Centred on the SCROLLBAR-EXCLUDED width, so it lines up with the composer
   and the message column rather than with the raw pane. */
.thread__jump {
  position: absolute;
  bottom: calc(var(--pk-composer-h, 96px) + 16px);
  left: calc((100% - var(--pk-sbw, 0px)) / 2);
  transform: translateX(-50%);
  z-index: 20;
}
.thread__jump-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: 1px solid var(--pk-border-default);
  background: var(--pk-bg-elevated);
  color: var(--pk-text-secondary);
  cursor: pointer;
  box-shadow: var(--pk-shadow-md);
}
.thread__jump-btn:hover {
  color: var(--pk-text-primary);
}
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
