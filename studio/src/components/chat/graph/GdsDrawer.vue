<script setup lang="ts">
// The GDS drawer, slimmed from traverse studio's (theirs is 1,300 lines of
// store-coupled UI; this one is catalog -> params -> run). The engine does
// the real work: a run is ordinary Cypher (`CALL traverse.<name>.stream`)
// through the same session as every other query - buildAlgorithmCall is
// lifted from upstream so identical Cypher hits the identical procedure.
import { computed, onMounted, ref } from 'vue'
import Icon from '@/components/Icon.vue'
import NumberField from '@/components/ui/NumberField.vue'
import Select from '@/components/ui/Select.vue'
import Switch from '@/components/ui/Switch.vue'
import TextInput from '@/components/ui/TextInput.vue'
import {
  buildAlgorithmCall,
  type GdsAlgorithm,
  type GraphSession,
  type QueryResponse,
} from '@/lib/graph/session'

const props = defineProps<{
  session: GraphSession
}>()
const emit = defineEmits<{
  ran: [algorithm: GdsAlgorithm, response: QueryResponse]
  close: []
}>()

const catalog = ref<GdsAlgorithm[]>([])
const picked = ref('')
const config = ref<Record<string, unknown>>({})
const running = ref(false)
const error = ref('')

onMounted(async () => {
  try {
    const all = (await props.session.algorithms()).algorithms
    catalog.value = all.filter((a) => a.modes.includes('stream'))
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
})

const options = computed(() => {
  const byCat = new Map<string, GdsAlgorithm[]>()
  for (const a of catalog.value) {
    const list = byCat.get(a.category) ?? []
    list.push(a)
    byCat.set(a.category, list)
  }
  const out: { value: string; label: string }[] = []
  for (const [cat, algs] of byCat) {
    for (const a of algs) {
      out.push({
        value: a.name,
        label: `${cat} · ${a.displayName}${a.stability !== 'stable' ? ` (${a.stability})` : ''}`,
      })
    }
  }
  return out
})

const algorithm = computed(() => catalog.value.find((a) => a.name === picked.value) ?? null)
/** The knobs worth showing - advanced ones keep their defaults silently. */
const params = computed(() => algorithm.value?.configSchema.filter((p) => !p.advanced) ?? [])

function pick(name: string): void {
  picked.value = name
  const alg = catalog.value.find((a) => a.name === name)
  const c: Record<string, unknown> = {}
  for (const p of alg?.configSchema ?? []) {
    if (!p.advanced && p.default !== null && p.default !== undefined) c[p.key] = p.default
  }
  config.value = c
}

async function run(): Promise<void> {
  const alg = algorithm.value
  if (!alg || running.value) return
  running.value = true
  error.value = ''
  try {
    const call = buildAlgorithmCall(alg, 'stream', config.value)
    const res = await props.session.run(call, 60_000)
    emit('ran', alg, res)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    running.value = false
  }
}
</script>

<template>
  <div class="gd">
    <div class="gd__head">
      <span class="gd__title">Graph algorithms</span>
      <button class="gd__x" type="button" aria-label="Close" @click="emit('close')">
        <Icon name="x-circle" :size="14" />
      </button>
    </div>
    <Select
      :model-value="picked"
      :options="options"
      placeholder="Pick an algorithm"
      @update:model-value="pick(String($event))"
    />
    <p v-if="algorithm" class="gd__desc">{{ algorithm.description }}</p>
    <div v-for="p in params" :key="p.key" class="gd__param" :class="{ 'gd__param--row': p.type === 'bool' }">
      <label class="gd__key">{{ p.key }}</label>
      <Switch
        v-if="p.type === 'bool'"
        :model-value="Boolean(config[p.key])"
        @update:model-value="config[p.key] = $event"
      />
      <NumberField
        v-else-if="p.type === 'integer' || p.type === 'float'"
        :model-value="Number(config[p.key] ?? 0)"
        :step="p.type === 'float' ? 0.05 : 1"
        @update:model-value="config[p.key] = $event"
      />
      <Select
        v-else-if="p.type === 'enum'"
        :model-value="String(config[p.key] ?? '')"
        :options="String(p.description).match(/\[(.*)\]/)?.[1]?.split('|').map(v => ({ value: v.trim(), label: v.trim() })) ?? []"
        @update:model-value="config[p.key] = $event"
      />
      <TextInput
        v-else
        :model-value="String(config[p.key] ?? '')"
        @update:model-value="config[p.key] = $event"
      />
    </div>
    <button class="pk-btn pk-btn--sm pk-btn--primary gd__run" :disabled="!algorithm || running" @click="run">
      {{ running ? 'Running...' : 'Run' }}
    </button>
    <p v-if="error" class="gd__err">{{ error }}</p>
  </div>
</template>

<style scoped>
.gd {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 300px;
  max-height: calc(100% - 16px);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  box-shadow: var(--pk-shadow-md, 0 4px 12px rgba(0, 0, 0, 0.12));
}
.gd__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.gd__title {
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
}
.gd__x {
  display: grid;
  place-items: center;
  padding: 0;
  border: 0;
  background: none;
  color: var(--pk-text-secondary);
  cursor: pointer;
}
.gd__desc {
  margin: 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
}
/* Label over a full-width control: number fields carry spinners and text
   inputs need room, so side-by-side starves them. A lone switch is the
   exception - it reads as one line. */
.gd__param {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.gd__param--row {
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
}
.gd__param > :not(label) {
  width: 100%;
}
.gd__param--row > :not(label) {
  width: auto;
}
.gd__key {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  font-family: var(--pk-font-mono);
}
.gd__run {
  align-self: flex-end;
}
.gd__err {
  margin: 0;
  color: var(--pk-danger, #b91c1c);
  font-size: var(--pk-font-size-xs);
  overflow-wrap: anywhere;
}
</style>
