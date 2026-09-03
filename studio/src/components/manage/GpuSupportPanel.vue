<script setup lang="ts">
// Which graphics cards run models - one row per CARD, because the question
// being asked is "will mine work" and nobody knows which generation theirs
// belongs to. Its own page rather than a fold-out inside a notice: 47 rows do
// not fit beside anything, and this is a link people will want to send.
import { computed, onMounted, ref } from 'vue'
import Icon from '@/components/Icon.vue'

interface CardRow {
  card: string
  generation: string
  kind: 'workstation' | 'datacenter' | 'jetson'
  status: 'supported' | 'testing' | 'planned' | 'too-old'
}

const rows = ref<CardRow[]>([])
const yours = ref<string | null>(null)
const q = ref('')

onMounted(async () => {
  const r = await fetch('/api/gpus')
  if (!r.ok) return
  const d = (await r.json()) as { cards: CardRow[]; yours?: string | null }
  rows.value = d.cards
  yours.value = d.yours ?? null
})

const KIND: Record<CardRow['kind'], string> = {
  workstation: 'Workstation & desktop',
  datacenter: 'Data centre',
  jetson: 'Jetson',
}
const STATUS: Record<CardRow['status'], string> = {
  supported: 'Runs models',
  testing: 'Being tested',
  planned: 'Not yet',
  'too-old': 'Too old',
}

const shown = computed(() => {
  const needle = q.value.trim().toLowerCase()
  if (!needle) return rows.value
  return rows.value.filter(
    (r) =>
      r.card.toLowerCase().includes(needle) || r.generation.toLowerCase().includes(needle),
  )
})
const supportedCount = computed(() => rows.value.filter((r) => r.status === 'supported').length)
function isYours(r: CardRow): boolean {
  return !!yours.value && r.card.toLowerCase() === yours.value.toLowerCase()
}
</script>

<template>
  <div class="gpu">
    <div class="gpu__head">
      <div>
        <h1 class="gpu__title">Graphics cards</h1>
        <p class="gpu__lead">
          Paddock runs models on a card only after measuring it on real hardware, so this list is
          what has actually been tested - {{ supportedCount }} cards today.
        </p>
      </div>
    </div>

    <div class="gpu__card">
      <input
        v-model="q"
        class="pk-input gpu__search"
        placeholder="Find your card"
        spellcheck="false"
        aria-label="Find your card"
      />
      <div class="gpu__tablewrap">
        <table class="gpu__table">
        <colgroup>
          <col />
          <col class="col-gen" />
          <col class="col-kind" />
          <col class="col-status" />
        </colgroup>
        <thead>
          <tr>
            <th>Graphics card</th>
            <th>Generation</th>
            <th>Type</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in shown" :key="r.card" :class="{ 'gpu__row--yours': isYours(r) }">
            <td class="gpu__name">
              {{ r.card }}
              <span v-if="isYours(r)" class="gpu__yours">yours</span>
            </td>
            <td class="gpu__dim">{{ r.generation }}</td>
            <td class="gpu__dim">{{ KIND[r.kind] }}</td>
            <td>
              <span class="gpu__pill" :class="`gpu__pill--${r.status}`">
                <Icon
                  :name="r.status === 'supported' ? 'check-circle' : 'x-circle'"
                  :size="12"
                />
                {{ STATUS[r.status] }}
              </span>
            </td>
          </tr>
          <tr v-if="!shown.length">
            <td colspan="4" class="gpu__none">
              No card here matches "{{ q }}". If it isn't listed, Paddock can't run models on it.
            </td>
          </tr>
        </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<style scoped>
.gpu {
  width: 100%;
  max-width: 1000px;
}
.gpu__head {
  margin-bottom: 18px;
}
.gpu__title {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
}
.gpu__lead {
  margin: 6px 0 0;
  max-width: 70ch;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
}
/* The cloud page's recipe, because it is the same job: a big search over a
   long table. Fields NEVER sit bare on the grey content background - they live
   on a bg-surface card, and the rows inside step to bg-base (otherwise you
   are dropping inputs into the grey main container). */
.gpu__card {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 14px;
}
.gpu__search {
  width: 100%;
  max-width: none;
  height: 52px;
  margin-bottom: 12px;
  padding: 0 18px;
  font-size: var(--pk-font-size-lg);
  font-weight: 600;
  border-radius: var(--pk-radius-lg);
}
.gpu__search::placeholder {
  font-weight: 500;
}
/* No inner scroller and no vh: a full PAGE scrolls with the content area, and
   a table pinned to 60vh made the card fill the viewport however few rows it
   held. The sticky head still works - the scroll container is the page. */
.gpu__tablewrap {
  overflow-x: auto;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-base);
}
.gpu__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-sm);
}
.col-gen {
  width: 190px;
}
.col-kind {
  width: 190px;
}
.col-status {
  width: 150px;
}
.gpu__table thead th {
  text-align: left;
  font-weight: 600;
  font-size: var(--pk-font-size-xs);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
  padding: 10px 14px;
  border-bottom: 1px solid var(--pk-border-default);
  white-space: nowrap;
  background: var(--pk-bg-base);
  position: sticky;
  top: 0;
}
.gpu__table td {
  padding: 9px 14px;
  border-bottom: 1px solid var(--pk-border-subtle, var(--pk-border-default));
  vertical-align: middle;
}
.gpu__table tbody tr:last-child td {
  border-bottom: 0;
}
.gpu__name {
  color: var(--pk-text-primary);
}
.gpu__dim {
  color: var(--pk-text-muted);
  white-space: nowrap;
}
.gpu__row--yours {
  background: var(--pk-accent-subtle);
}
.gpu__yours {
  margin-left: 8px;
  padding: 1px 7px;
  border-radius: 999px;
  background: var(--pk-accent);
  color: var(--pk-bg-base);
  font-size: var(--pk-font-size-xs);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.gpu__pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: var(--pk-font-size-xs);
  font-weight: 650;
  white-space: nowrap;
}
.gpu__pill--supported {
  background: var(--pk-status-success-subtle);
  color: var(--pk-status-success);
}
.gpu__pill--testing {
  background: var(--pk-status-warning-subtle);
  color: var(--pk-status-warning);
}
.gpu__pill--planned,
.gpu__pill--too-old {
  background: var(--pk-bg-inset);
  color: var(--pk-text-muted);
}
.gpu__none {
  color: var(--pk-text-muted);
  text-align: center;
  padding: 28px 14px;
}
</style>
