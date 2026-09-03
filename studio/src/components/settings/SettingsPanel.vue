<script setup lang="ts">
// Studio settings = only what changes how conversations behave.
// Everything else was evicted: tools/search are model config (the Manager's
// Start/Edit page), the DB export is manager admin (/manage/settings), theme
// lives in the header, and model facts live on the model's page.
import { computed, onMounted, ref } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import { useModelsStore } from '@/stores/models'
import { useAudioDevices } from '@/composables/useAudioDevices'
import { OSM_TILES, tileHost, tileTemplate } from '@/lib/maptiles'
import Slider from '@/components/ui/Slider.vue'
import Switch from '@/components/ui/Switch.vue'
import Select, { type SelectOption } from '@/components/ui/Select.vue'
import TextInput from '@/components/ui/TextInput.vue'

const settings = useSettingsStore()
const models = useModelsStore()

onMounted(() => {
  if (!models.maxCtx) void models.fetchLimits()
})

// Reply-length stops: powers of two from 512, then a final MODEL MAXIMUM stop
// (null) that resolves per send to whatever the window has left after the
// prompt. The old top stop was maxCtx itself, which is not a usable setting -
// reserving the whole window for the reply leaves nothing for the prompt, and
// the cap doubled as that reservation. Model maximum is
// the default now; a numbered stop is an explicit ceiling the user chose.
const stops = computed<(number | null)[]>(() => {
  const cap = models.maxCtx || 8192
  const arr: (number | null)[] = []
  for (let v = 512; v < cap; v *= 2) arr.push(v)
  arr.push(null)
  return arr
})

const idx = computed<number>({
  get: () => {
    if (settings.maxTokens == null) return stops.value.length - 1
    const want = settings.maxTokens
    const i = stops.value.findIndex((s) => s != null && s >= want)
    return i < 0 ? stops.value.length - 1 : i
  },
  set: (v) => {
    settings.maxTokens = stops.value[Math.min(Math.max(0, v), stops.value.length - 1)]
  },
})

// The stored value is never rewritten to fit the current model: this watch
// used to clamp it destructively, so one small-context model being current
// silently turned a 32K setting into 4096 - which then capped every send,
// on every model, until noticed. A model whose context
// is smaller than the setting bounds the OUTPUT at use time (the runner
// clamps server-side; providers clamp themselves); the slider just shows
// its top stop while such a model is current.

function fmtStop(n: number | null): string {
  return n == null ? 'Model maximum' : `${fmtTokens(n)} tokens`
}

// How many tools one reply may run. "Server default" (0 here, null on the
// wire) sends nothing and leaves the server's own budget alone - the
// honour-the-defaults stance. A number rides as the Responses API's
// `max_tool_calls`, which caps the calls AND lifts the server's round ceiling
// to match, so raising it actually buys more work rather than only less.
const TOOL_CALL_STOPS: SelectOption[] = [
  { value: 0, label: 'Server default' },
  { value: 5, label: '5 tool calls' },
  { value: 10, label: '10 tool calls' },
  { value: 25, label: '25 tool calls' },
  { value: 50, label: '50 tool calls' },
  { value: 100, label: '100 tool calls' },
]

const toolCalls = computed<number>({
  get: () => settings.maxToolCalls ?? 0,
  set: (v) => {
    settings.maxToolCalls = v > 0 ? v : null
  },
})

function fmtTokens(n: number): string {
  if (n >= 1024) {
    const k = n / 1024
    return `${Number.isInteger(k) ? k : k.toFixed(1)}K`
  }
  return String(n)
}

// ── Microphone ──────────────────────────────────────────────────────────────
// The durable default for every mic path in the Studio. The COMPOSER's mic
// menu holds the same preference, because that is where the decision is
// actually made - you notice the wrong device with the mic about to open, not
// while reading a settings page. This is the other half: where you set the box
// up once, and the only place that can offer the permission step below.
//
// Output is deliberately absent. Routing playback needs `setSinkId`, which
// only Chromium implements, so a picker here would be a control that looks
// like it works and silently does not for everyone else - and the OS already
// routes output perfectly well.
const audio = useAudioDevices()
/** Reka reads '' as "nothing selected", so the system default needs a value of
 *  its own. Stored as an empty id, which is what sends no device constraint. */
const SYSTEM = 'system'
const micChoice = computed<string>({
  get: () => settings.micDeviceId || SYSTEM,
  set: (v) => {
    const hit = audio.devices.value.find((d) => d.id === v)
    settings.micDeviceId = v === SYSTEM ? '' : v
    // The label is stored so an UNPLUGGED device can still be named - a
    // disconnected one is not in `enumerateDevices` at all, and "your Jabra
    // headset isn't here" is actionable where "the microphone you chose isn't
    // here" is a riddle.
    settings.micDeviceLabel = v === SYSTEM ? '' : (hit?.label ?? '')
  },
})
const micOptions = computed<SelectOption[]>(() => {
  const opts: SelectOption[] = [{ value: SYSTEM, label: 'System default' }]
  for (const d of audio.devices.value) opts.push({ value: d.id, label: d.label })
  // The chosen device, when it is not plugged in right now. Listed rather than
  // dropped: removing it would move the selection onto a device nobody picked.
  if (settings.micDeviceId && audio.missing(settings.micDeviceId)) {
    opts.push({
      value: settings.micDeviceId,
      label: settings.micDeviceLabel || 'Chosen microphone',
      hint: 'not connected',
    })
  }
  return opts
})
const revealing = ref(false)
/** The permission prompt was dismissed or the microphone is blocked for this
 *  page. Said out loud: a button that quietly does nothing when clicked is the
 *  worst answer available, and "allow it and try again" is actionable. */
const revealDenied = ref(false)
// ── Map tiles ───────────────────────────────────────────────────────────────
// It belongs on this page by the rule above: it changes what a
// conversation does - whether opening a photo you attached makes a network
// request, and to whom. The head names the host rather than echoing the
// template, because the host is the part that matters here.
const mapHost = computed(() => tileHost(tileTemplate(settings.mapTiles, settings.theme)))

async function revealMics(): Promise<void> {
  revealing.value = true
  revealDenied.value = false
  try {
    revealDenied.value = !(await audio.reveal())
  } finally {
    revealing.value = false
  }
}

</script>

<template>
  <div class="settings">
    <h1 class="settings__title">Settings</h1>

    <section class="settings__card">
      <div class="settings__head">
        <h2>Max reply length</h2>
        <span class="settings__val">{{ fmtStop(stops[idx]) }}</span>
      </div>
      <p class="settings__sub">
        The longest a single reply can be. Thinking and answer share this budget, and a reply
        can't exceed the context window.
      </p>
      <Slider v-model="idx" :min="0" :max="stops.length - 1" :step="1" class="settings__slider" />
      <div class="settings__stops">
        <span
          v-for="(s, i) in stops"
          :key="s ?? 'max'"
          class="settings__stop"
          :class="{ 'settings__stop--cur': i === idx }"
          >{{ s == null ? 'Max' : fmtTokens(s) }}</span
        >
      </div>
    </section>

    <section class="settings__card">
      <div class="settings__head">
        <h2>Tools per reply</h2>
        <Select v-model="toolCalls" :options="TOOL_CALL_STOPS" class="settings__pick" />
      </div>
      <p class="settings__sub">
        How many tools one reply may run before it answers with what it found. Reaching the limit
        is not an error - the reply says so and finishes.
      </p>
    </section>

    <section class="settings__card">
      <div class="settings__head">
        <h2>Summarize older messages</h2>
        <Switch v-model="settings.summarize" label="Summarize older messages" />
      </div>
      <p class="settings__sub">
        When a chat outgrows the context window, older messages are summarized in the background
        so the model keeps the thread of the conversation. Turn off to drop the oldest messages
        instead.
      </p>
    </section>

    <section class="settings__card">
      <div class="settings__head">
        <h2>Microphone</h2>
        <Select
          v-if="audio.named()"
          v-model="micChoice"
          :options="micOptions"
          class="settings__pick"
        />
        <button
          v-else-if="audio.supported()"
          class="pk-btn pk-btn--sm"
          type="button"
          :disabled="revealing"
          @click="revealMics"
        >
          Show my microphones
        </button>
      </div>
      <p v-if="!audio.supported()" class="settings__warn">
        The browser blocks the microphone on this address. See
        <RouterLink :to="{ name: 'trust' }">Trust this computer</RouterLink> in the Manager.
      </p>
      <p v-else-if="revealDenied" class="settings__warn">
        The microphone was blocked. Allow it for this page and try again.
      </p>
      <p v-else-if="!audio.named()" class="settings__sub">
        The browser only names your microphones once this page has been allowed to use one.
      </p>
      <p v-else-if="settings.micDeviceId && audio.missing(settings.micDeviceId)" class="settings__warn">
        {{ settings.micDeviceLabel || 'The microphone you chose' }} isn't connected. Recording uses
        the system default until it is back.
      </p>
    </section>

    <section class="settings__card">
      <div class="settings__head">
        <h2>Map tiles</h2>
        <span class="settings__val">{{ mapHost }}</span>
      </div>
      <p class="settings__sub">
        A photo with GPS shows a map drawn from an outline inside Paddock, which contacts nobody.
        Opening the interactive map fetches tiles from this address, which tells that host where
        the photo was taken. Leave it empty for a basemap that follows your theme, or name a
        server - your own, or OpenStreetMap's at {{ OSM_TILES }}.
      </p>
      <TextInput v-model="settings.mapTiles" block placeholder="Follow the theme" />
    </section>
  </div>
</template>

<style scoped>
.settings {
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
}
.settings__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin-bottom: 20px;
}
.settings__card {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 18px 20px;
  margin-bottom: 14px;
}
.settings__head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}
.settings__head h2 {
  font-size: var(--pk-font-size-base);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.settings__val {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-sm);
  color: var(--pk-accent-text);
}
.settings__sub {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
  margin: 6px 0 16px;
}
.settings__warn {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-status-warning);
  line-height: 1.5;
  margin: 6px 0 0;
}
.settings__sub code {
  font-family: var(--pk-font-mono);
  font-size: 0.9em;
  background: var(--pk-bg-inset);
  padding: 1px 5px;
  border-radius: var(--pk-radius-sm);
}
.settings__slider {
  margin: 4px 0 10px;
}
.settings__pick {
  min-width: 168px;
}
.settings__stops {
  display: flex;
  justify-content: space-between;
  gap: 4px;
}
.settings__stop {
  font-family: var(--pk-font-mono);
  font-size: 11px;
  color: var(--pk-text-muted);
}
.settings__stop--cur {
  color: var(--pk-accent-text);
  font-weight: 600;
}
.seg {
  display: inline-flex;
  gap: 4px;
  padding: 3px;
  background: var(--pk-bg-inset);
  border-radius: var(--pk-radius-md);
  width: fit-content;
}
.seg__btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border: none;
  border-radius: var(--pk-radius-sm);
  background: transparent;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
  font-weight: 500;
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
}
.seg__btn--on {
  background: var(--pk-bg-elevated);
  color: var(--pk-text-primary);
  box-shadow: var(--pk-shadow-sm);
}
.settings__searchrow {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
}
.settings__searchkey {
  flex: 1;
  min-width: 0;
  font-family: var(--pk-font-mono);
}
.settings__testres {
  margin-top: 10px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-status-success);
}
.settings__testres--bad {
  color: var(--pk-text-danger);
}
.settings__stats {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 4px;
}
.settings__stats > div {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}
.settings__stats dt {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
.settings__stats dd {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
