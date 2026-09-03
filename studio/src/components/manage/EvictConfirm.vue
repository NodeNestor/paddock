<script setup lang="ts">
// The eviction offer (vram-budget plan, layer 2): a start the card can't fit
// arrives as a 507 with a concrete plan - "stopping these would make room" -
// and this dialog turns it into the user's explicit yes. Mounted once in
// App.vue; the fleet store owns the pending ask (any start path can raise it:
// a stopped row's Start, the Start page's deploy, an edit's redeploy).
import { computed } from 'vue'
import { useFleetStore } from '@/stores/fleet'
import { fmtVram } from '@/lib/format'
import Dialog from '@/components/ui/Dialog.vue'

const fleet = useFleetStore()

/** The plan's stops, resolved to their candidate rows (name + bytes freed). */
const stops = computed(() => {
  const ask = fleet.evictAsk
  if (!ask) return []
  return ask.offer.plan.map((port) => {
    const c = ask.offer.candidates.find((x) => x.port === port)
    return {
      port,
      name: c?.display ?? `port ${port}`,
      frees: c?.frees ?? 0,
    }
  })
})

const confirmLabel = computed(() =>
  stops.value.length === 1
    ? `Stop ${stops.value[0]?.name} and start`
    : `Stop ${stops.value.length} models and start`,
)
</script>

<template>
  <Dialog
    :open="!!fleet.evictAsk"
    role="alertdialog"
    icon="alert-triangle"
    title="Not enough GPU memory"
    @close="fleet.cancelEvict()"
  >
    <div v-if="fleet.evictAsk" class="evict">
      <p class="evict__lead">
        <strong>{{ fleet.evictAsk.label }}</strong> needs
        {{ fmtVram(fleet.evictAsk.offer.need) }} of GPU memory, but only
        {{ fmtVram(fleet.evictAsk.offer.residual) }} is unclaimed. Stopping
        {{ stops.length === 1 ? 'this model' : 'these models' }} makes room -
        {{ stops.length === 1 ? 'it stays' : 'they stay' }} configured and can
        be started again with one click:
      </p>
      <ul class="evict__list">
        <li v-for="s in stops" :key="s.port" class="evict__row">
          <span class="evict__name">{{ s.name }}</span>
          <span class="evict__meta">port {{ s.port }} · frees {{ fmtVram(s.frees) }}</span>
        </li>
      </ul>
    </div>
    <template #footer>
      <button class="pk-btn pk-btn--ghost" @click="fleet.cancelEvict()">Cancel</button>
      <button class="pk-btn pk-btn--primary" @click="fleet.confirmEvict()">
        {{ confirmLabel }}
      </button>
    </template>
  </Dialog>
</template>

<style scoped>
.evict {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.evict__lead {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
}
.evict__lead strong {
  color: var(--pk-text-primary);
}
.evict__list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.evict__row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 10px;
  border: 1px solid var(--pk-border);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
}
.evict__name {
  font-weight: 600;
  color: var(--pk-text-primary);
  font-size: var(--pk-font-size-sm);
}
.evict__meta {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  white-space: nowrap;
}
</style>
