<script setup lang="ts">
// Per-turn "run" record: exactly what produced this answer (provenance) + the
// GPU environment it ran under + metrics. The Studio's lab-notebook view.
import { computed } from 'vue'
import type { Message } from '@/types/chat'
import { fmtCost, fmtDuration, fmtVram } from '@/lib/format'
import Collapsible from '@/components/ui/Collapsible.vue'

const props = defineProps<{ message: Message }>()

const run = computed(() => props.message.run)
const usage = computed(() => props.message.usage)
const gpu = computed(() => props.message.run?.gpu)

function gb(b?: number): string {
  return b ? fmtVram(b) : '-'
}
// Sampler dials read as tenths (1.0, 0.8) - keep the record in that idiom.
function dial(n: number | null | undefined): string {
  if (n == null) return 'default'
  return Number.isInteger(n) ? n.toFixed(1) : String(n)
}
const promptLabel = computed(() => {
  const r = run.value
  if (!r) return '-'
  if (r.systemPromptName) return r.systemPromptName
  return r.systemPrompt?.trim() ? 'Custom' : 'None'
})
</script>

<template>
  <div v-if="run || usage" class="run">
    <div v-if="run" class="run__sec">
      <span class="run__title">Provenance</span>
      <dl class="run__grid">
        <dt>Model</dt>
        <dd>{{ run.model }}</dd>
        <template v-if="run.spec">
          <dt>Speculation</dt>
          <dd>{{ run.spec }}</dd>
        </template>
        <dt>System prompt</dt>
        <dd>{{ promptLabel }}</dd>
        <dt>Sampling</dt>
        <dd>
          temp {{ dial(run.params.temperature) }} · top-p {{ dial(run.params.topP) }} · top-k
          {{ run.params.topK === 0 ? 'off' : (run.params.topK ?? 'default') }}<span
            v-if="run.params.minP"
          >
            · min-p {{ run.params.minP }}</span
          ><span v-if="run.params.presencePenalty">
            · pres {{ dial(run.params.presencePenalty) }}</span
          ><span v-if="run.params.frequencyPenalty">
            · freq {{ dial(run.params.frequencyPenalty) }}</span
          ><span v-if="run.params.repeatPenalty && run.params.repeatPenalty !== 1">
            · repeat {{ dial(run.params.repeatPenalty) }}</span
          >
        </dd>
        <dt>Reasoning</dt>
        <dd>{{ run.params.thinking ? run.params.reasoningEffort : 'off' }}</dd>
        <dt>Max tokens</dt>
        <dd>{{ run.params.maxTokens ?? '-' }}<span v-if="run.params.seed != null"> · seed {{ run.params.seed }}</span></dd>
        <template v-if="run.tools.length">
          <dt>Tools</dt>
          <dd>{{ run.tools.join(', ') }}</dd>
        </template>
      </dl>
      <Collapsible v-if="run.systemPrompt" class="run__prompt" summary="Prompt text">
        <pre>{{ run.systemPrompt }}</pre>
      </Collapsible>
    </div>

    <div v-if="usage" class="run__sec">
      <span class="run__title">Metrics</span>
      <dl class="run__grid">
        <dt>Tokens</dt>
        <dd>
          {{ usage.promptTokens ?? 0 }} in · {{ usage.completionTokens ?? 0 }} out<span
            v-if="usage.reasoningTokens"
          >
            · {{ usage.reasoningTokens }} reasoning</span
          >
        </dd>
        <dt>Speed</dt>
        <dd>
          <span v-if="usage.tps">{{ Math.round(usage.tps) }} tok/s</span><span v-if="usage.ttftMs">
            · TTFT {{ fmtDuration(usage.ttftMs) }}</span
          ><span v-if="usage.ms"> · {{ fmtDuration(usage.ms) }} total</span>
        </dd>
        <dt v-if="usage.costUsd !== undefined">Cost</dt>
        <dd v-if="usage.costUsd !== undefined">{{ fmtCost(usage.costUsd) }}</dd>
      </dl>
    </div>

    <div v-if="gpu" class="run__sec">
      <span class="run__title">GPU environment (peak)</span>
      <dl class="run__grid">
        <dt v-if="gpu.device">Device</dt>
        <dd v-if="gpu.device">{{ gpu.device }}</dd>
        <dt>Load</dt>
        <dd>
          <span v-if="gpu.utilPeak != null">{{ gpu.utilPeak }}% util</span><span
            v-if="gpu.memUsedPeak"
          >
            · {{ gb(gpu.memUsedPeak) }} VRAM</span
          ><span v-if="gpu.powerPeakW"> · {{ Math.round(gpu.powerPeakW) }} W</span><span
            v-if="gpu.tempPeakC"
          >
            · {{ gpu.tempPeakC }}°C</span
          >
        </dd>
        <template v-if="gpu.batchPeak || gpu.kvTotal || gpu.tokSPeak">
          <dt>Engine</dt>
          <dd>
            <span v-if="gpu.batchPeak">batch {{ gpu.batchPeak }}</span><span v-if="gpu.kvTotal">
              · KV {{ gpu.kvPeak }}/{{ gpu.kvTotal }}</span
            ><span v-if="gpu.tokSPeak"> · {{ Math.round(gpu.tokSPeak) }} tok/s peak</span>
          </dd>
        </template>
      </dl>
    </div>
  </div>
</template>

<style scoped>
.run {
  margin-top: 8px;
  padding: 12px 14px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-base);
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.run__sec {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.run__title {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--pk-text-muted);
}
.run__grid {
  display: grid;
  grid-template-columns: 96px 1fr;
  gap: 3px 12px;
  margin: 0;
}
.run__grid dt {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
}
.run__grid dd {
  margin: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-primary);
  font-family: var(--pk-font-mono);
  word-break: break-word;
}
.run__prompt {
  margin-top: 2px;
}
.run__prompt pre {
  margin: 6px 0 0;
  padding: 8px 10px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  font-size: var(--pk-font-size-xs);
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--pk-text-secondary);
  max-height: 220px;
  overflow-y: auto;
}
</style>
