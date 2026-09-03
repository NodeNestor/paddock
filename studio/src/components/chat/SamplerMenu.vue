<script setup lang="ts">
// The sampler popover: temperature / top-p / top-k / seed for this chat.
// The trigger is the PARENT's slot content, MenuTrigger included:
// composer__tool styling is scoped to Composer.vue, and the Reka as-child
// chain (MenuTrigger outside, Tooltip inside, button innermost - see the
// popper-context note in ui/Tooltip.vue) only composes when built next to
// the button itself.
// The load-bearing rule is untouched-means-absent - a dial the user never
// moved sends nothing, so the model's own serving defaults apply
// (honor-the-defaults). Values persist per conversation, and Run details
// records exactly what rode. Native OpenAI/Anthropic lanes drop sampling
// entirely (Sonnet 5 rejects explicit temperature) - said here, never
// silently leveled.
import { computed } from 'vue'
import { useChatStore } from '@/stores/chat'
import { useModelsStore } from '@/stores/models'
import Menu from '@/components/ui/Menu.vue'
import MenuContent from '@/components/ui/MenuContent.vue'
import MenuTrigger from '@/components/ui/MenuTrigger.vue'
import Slider from '@/components/ui/Slider.vue'
import TextInput from '@/components/ui/TextInput.vue'

const chat = useChatStore()
const models = useModelsStore()

type DialKey = 'temperature' | 'topP' | 'topK' | 'minP' | 'presencePenalty' | 'repeatPenalty'
const WIRE_KEY = {
  temperature: 'temperature',
  topP: 'top_p',
  topK: 'top_k',
  minP: 'min_p',
  repeatPenalty: 'repeat_penalty',
} as const
// The value at which each dial does nothing: top-k/min-p 0 = no cutoff,
// presence 0 = no penalty, repeat 1 = no penalty. Those dials' scales START
// at their off value, so "off" is a real slider position, never a lie about
// some other number. Temperature and top-p have no off.
const OFF_VALUE: Partial<Record<DialKey, number>> = {
  topK: 0,
  minP: 0,
  presencePenalty: 0,
  repeatPenalty: 1,
}

// Slider models: while the param is unset the thumb parks at the endpoint's
// ADVERTISED default (so it agrees with the "default (x)" label), else at a
// sensible start; any interaction writes, flipping it from "default" to set.
function advertised(key: DialKey): number | undefined {
  // penalties are per-request only - no server carries a default for them
  if (key === 'presencePenalty') return undefined
  const id = chat.active?.model || models.currentId
  return models.caps[id]?.sampling?.[WIRE_KEY[key]]
}
function dial(key: DialKey, start: number) {
  return computed<number>({
    get: () => {
      const v = chat.active?.params[key]
      if (v != null) return v
      const d = advertised(key)
      if (d == null) return start
      if (key === 'topK') return Math.min(d, 200)
      return d
    },
    set: (v) => {
      const c = chat.active
      if (!c) return
      // step arithmetic can wobble in binary floats (0.8500000000000001)
      c.params[key] = Math.round(v * 100) / 100
      chat.persist(c)
    },
  })
}
const temperature = dial('temperature', 0.8)
const topP = dial('topP', 0.95)
const topK = dial('topK', 0)
const minP = dial('minP', 0)
const presencePenalty = dial('presencePenalty', 0)
const repeatPenalty = dial('repeatPenalty', 1)

const seedText = computed<string>({
  get: () => {
    const s = chat.active?.params.seed
    return s == null ? '' : String(s)
  },
  set: (v) => {
    const c = chat.active
    if (!c) return
    const n = Number.parseInt(v, 10)
    c.params.seed = Number.isFinite(n) ? n : null
    chat.persist(c)
  },
})

// "default" shows the NUMBER when the endpoint advertises its own defaults
// (local runners do via /server). Dials with an off state fall back to
// "default (off)" - that is the universal wire default for them - while
// temperature/top-p on a lane that advertises nothing stay a plain word
// rather than an invented figure.
// These dials read as tenths by convention (1.0, 0.8, 0.95) - print them that
// way even when the value is whole; top-k is a plain count.
function num(key: DialKey, n: number): string {
  if (key === 'topK') return String(n)
  return Number.isInteger(n) ? n.toFixed(1) : String(n)
}
function fmt(key: DialKey): string {
  const v = chat.active?.params[key]
  if (v != null) return v === OFF_VALUE[key] ? 'off' : num(key, v)
  const d = advertised(key) ?? OFF_VALUE[key]
  if (d == null) return 'default'
  return d === OFF_VALUE[key] ? 'default (off)' : `default (${num(key, d)})`
}
const anySet = computed(() => {
  const p = chat.active?.params
  return (
    !!p &&
    (p.temperature != null ||
      p.topP != null ||
      p.topK != null ||
      p.minP != null ||
      p.presencePenalty != null ||
      p.repeatPenalty != null ||
      p.seed != null)
  )
})
// Where the "default (x)" numbers come from. Worth one line because they are
// not a house setting any more: each model is served at what
// its own authors published, so two models here legitimately read different.
const defaultsSource = computed<string | undefined>(() =>
  anySet.value ? undefined : models.caps[chat.active?.model || models.currentId]?.sampling?.source,
)
function reset(): void {
  const c = chat.active
  if (!c) return
  c.params.temperature = null
  c.params.topP = null
  c.params.topK = null
  c.params.minP = null
  c.params.presencePenalty = null
  c.params.frequencyPenalty = null
  c.params.repeatPenalty = null
  c.params.seed = null
  chat.persist(c)
}

// The exceptions, stated exactly where they hold and no wider: OpenAI's
// always-thinking families reject explicit temperature/top_p; Anthropic
// documents that extended thinking requires sampler params unset. Everything
// else - local, OpenRouter, custom, plain OpenAI chat models, Claude with
// thinking off - honors the dials.
function laneIds(): string[] {
  const c = chat.active
  return c?.compareModels?.length ? c.compareModels : [c?.model || models.currentId]
}
function laneKind(id: string): string | undefined {
  const cl = models.models.find((m) => m.id === id)?.cloud
  return cl ? models.cloudEndpoints.find((e) => e.id === cl.endpoint)?.kind : undefined
}
const oaiReasoningLane = computed(() =>
  laneIds().some((id) => {
    if (laneKind(id) !== 'openai') return false
    const bare = id.replace(/^cloud:[^:]+:/, '')
    return bare.startsWith('gpt-5') || /^o[134]/.test(bare)
  }),
)
const claudeThinkingLane = computed(
  () =>
    (chat.active?.params.thinking ?? true) && laneIds().some((id) => laneKind(id) === 'anthropic'),
)
// The extension dials don't travel to the native cloud APIs: OpenAI's
// Responses wire takes temperature/top-p only, Anthropic adds top-k but has
// no penalty or min-p knobs. Said when a set dial would stay home.
const oaiInertSet = computed(() => {
  const p = chat.active?.params
  const oai = laneIds().some((id) => laneKind(id) === 'openai')
  return (
    oai &&
    !!p &&
    (p.topK != null || p.minP != null || p.presencePenalty != null || p.repeatPenalty != null)
  )
})
const claudeInertSet = computed(() => {
  const p = chat.active?.params
  const claude = laneIds().some((id) => laneKind(id) === 'anthropic')
  return !!p && claude && (p.minP != null || p.presencePenalty != null || p.repeatPenalty != null)
})
</script>

<template>
  <Menu>
    <slot />
    <MenuContent side="top" align="start" min-width="280px">
      <div class="sm__body">
        <div class="sm__row">
          <span class="sm__label">Temperature</span>
          <span class="sm__val">{{ fmt('temperature') }}</span>
        </div>
        <Slider v-model="temperature" :min="0" :max="2" :step="0.05" />
        <div class="sm__row">
          <span class="sm__label">Top-p</span>
          <span class="sm__val">{{ fmt('topP') }}</span>
        </div>
        <Slider v-model="topP" :min="0.01" :max="1" :step="0.01" />
        <div class="sm__row">
          <span class="sm__label">Top-k</span>
          <span class="sm__val">{{ fmt('topK') }}</span>
        </div>
        <Slider v-model="topK" :min="0" :max="200" :step="1" />
        <div class="sm__row">
          <span class="sm__label">Min-p</span>
          <span class="sm__val">{{ fmt('minP') }}</span>
        </div>
        <Slider v-model="minP" :min="0" :max="1" :step="0.01" />
        <div class="sm__row">
          <span class="sm__label">Presence penalty</span>
          <span class="sm__val">{{ fmt('presencePenalty') }}</span>
        </div>
        <Slider v-model="presencePenalty" :min="-2" :max="2" :step="0.1" />
        <div class="sm__row">
          <span class="sm__label">Repeat penalty</span>
          <span class="sm__val">{{ fmt('repeatPenalty') }}</span>
        </div>
        <Slider v-model="repeatPenalty" :min="1" :max="2" :step="0.01" />
        <div class="sm__row">
          <span class="sm__label">Seed</span>
          <span class="sm__seed">
            <TextInput v-model="seedText" placeholder="random" />
          </span>
        </div>
        <p v-if="defaultsSource" class="sm__note">Defaults are {{ defaultsSource }}.</p>
        <p v-if="oaiReasoningLane" class="sm__note">
          gpt-5 and o-series models set their own sampling.
        </p>
        <p v-if="claudeThinkingLane" class="sm__note">
          Claude models ignore these while thinking is on.
        </p>
        <p v-if="oaiInertSet" class="sm__note">
          OpenAI models take temperature and top-p from here - not the other settings.
        </p>
        <p v-if="claudeInertSet" class="sm__note">
          Claude models take temperature, top-p and top-k - not the other settings.
        </p>
        <button v-if="anySet" class="pk-btn pk-btn--sm sm__reset" type="button" @click="reset">
          Model defaults
        </button>
      </div>
    </MenuContent>
  </Menu>
</template>

<style scoped>
.sm__body {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px;
  width: 268px;
}
.sm__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.sm__label {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
.sm__val {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.sm__seed {
  width: 120px;
  min-width: 0;
}
.sm__seed :deep(input) {
  width: 100%;
  min-width: 0;
  text-align: right;
  font-family: var(--pk-font-mono);
}
.sm__note {
  margin: 2px 0 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  max-width: 100%;
}
.sm__reset {
  align-self: flex-start;
}
</style>
