<script setup lang="ts">
// Cloud models, two clear zones ("add models -> provide
// key -> added models are available in a separate section on the same page"):
//
//   1. "In your pickers" - every added model, grouped by provider, with the
//      provider's key state and explicit Replace key / Remove buttons. One
//      key per PROVIDER; all of its models share it.
//   2. The browse card - provider tabs over a pure search surface. Adding a
//      model without a working key pops the key dialog; the model lands only
//      once the key checks out. OpenRouter rows expand into the per-provider
//      breakdown, where Add pins the pick to that provider.
//
// Keys never come back to the browser (rows carry hasKey only).
import { computed, onMounted, reactive, ref, watch } from 'vue'
import {
  cloudVendor,
  useModelsStore,
  type CloudEndpoint,
  type CloudModelPick,
} from '@/stores/models'
import Dialog from '@/components/ui/Dialog.vue'
import Icon from '@/components/Icon.vue'
import Menu from '@/components/ui/Menu.vue'
import MenuTrigger from '@/components/ui/MenuTrigger.vue'
import MenuContent from '@/components/ui/MenuContent.vue'
import MenuItem from '@/components/ui/MenuItem.vue'
import Tabs from '@/components/ui/Tabs.vue'
import TextInput from '@/components/ui/TextInput.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import ModelBrowser from './ModelBrowser.vue'

const models = useModelsStore()
const endpoints = computed(() => models.cloudEndpoints)

// ── tabs ────────────────────────────────────────────────────────────────────
const TAB_DEFS = {
  openrouter: {
    name: 'OpenRouter',
    kind: 'openai-compat',
    base: 'https://openrouter.ai/api/v1',
    keyHint: 'sk-or-v1-...',
  },
  openai: { name: 'OpenAI', kind: 'openai', base: 'https://api.openai.com/v1', keyHint: 'sk-...' },
  anthropic: {
    name: 'Anthropic',
    kind: 'anthropic',
    base: 'https://api.anthropic.com/v1',
    keyHint: 'sk-ant-...',
  },
} as const
type KnownTab = keyof typeof TAB_DEFS
/** Maker implied by the tab when the id itself names none - native
 *  OpenAI/Anthropic lists are bare ids (o3-mini, claude-sonnet-5). */
const TAB_VENDOR: Record<KnownTab, string | undefined> = {
  openrouter: undefined,
  openai: 'OpenAI',
  anthropic: 'Anthropic',
}
function pickVendor(m: CloudModelPick, t: KnownTab | null): string | undefined {
  return cloudVendor(m.id) ?? (t ? TAB_VENDOR[t] : undefined)
}

const tab = ref<KnownTab | 'custom'>('openrouter')
const TAB_OPTIONS = [
  { value: 'openrouter', label: 'OpenRouter' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'custom', label: 'Custom' },
]

function epForTab(t: KnownTab): CloudEndpoint | undefined {
  if (t === 'openrouter') {
    return endpoints.value.find((e) => e.baseUrl.startsWith('https://openrouter.ai'))
  }
  return endpoints.value.find((e) => e.kind === TAB_DEFS[t].kind)
}
/** The known tab this endpoint is, if any (the key dialog reuses the tab
 *  path for known providers). */
function tabForEp(ep: CloudEndpoint): KnownTab | null {
  for (const t of ['openrouter', 'openai', 'anthropic'] as const) {
    if (epForTab(t) === ep) return t
  }
  return null
}
const customEndpoints = computed(() =>
  endpoints.value.filter(
    (e) => e.kind === 'openai-compat' && !e.baseUrl.startsWith('https://openrouter.ai'),
  ),
)

const knownTab = computed<KnownTab | null>(() => (tab.value === 'custom' ? null : tab.value))
const curEp = computed(() => (knownTab.value ? epForTab(knownTab.value) : undefined))
/** OpenRouter's catalog is public; OpenAI/Anthropic share theirs only with a
 *  key, so those tabs must gate on it (a provider constraint, not ours). */
const needsKeyToBrowse = computed(
  () => knownTab.value !== null && knownTab.value !== 'openrouter' && !curEp.value?.hasKey,
)

/** One pick's identity: bare id for auto-routing, id@Provider when pinned. */
function pickKey(m: CloudModelPick): string {
  return m.provider ? `${m.id}@${m.provider}` : m.id
}

// ── per-tab browse state ────────────────────────────────────────────────────
interface BrowseState {
  list: CloudModelPick[]
  ranked: boolean
  loading: boolean
  error: string | null
  loaded: boolean
}
const blank = (): BrowseState => ({ list: [], ranked: false, loading: false, error: null, loaded: false })
const browses = reactive<Record<KnownTab, BrowseState>>({
  openrouter: blank(),
  openai: blank(),
  anthropic: blank(),
})
const curBrowse = computed(() => (knownTab.value ? browses[knownTab.value] : null))

async function loadTab(t: KnownTab, force = false): Promise<void> {
  const b = browses[t]
  if (b.loaded && !force) return
  const ep = epForTab(t)
  if (t !== 'openrouter' && !ep?.hasKey) return
  b.loading = true
  b.error = null
  try {
    const url = ep ? `/api/cloud/${ep.id}/models` : '/api/cloud/browse'
    const res = await fetch(url)
    const j = (await res.json()) as {
      models?: CloudModelPick[]
      ranked?: boolean
      error?: { message?: string }
    }
    if (!res.ok) {
      b.error = j?.error?.message ?? `The provider didn't answer (HTTP ${res.status}).`
      return
    }
    b.list = j.models ?? []
    b.ranked = j.ranked ?? false
    b.loaded = true
    void backfillReasoning(t, b.list)
  } catch (e) {
    b.error = e instanceof Error ? e.message : String(e)
  } finally {
    b.loading = false
  }
}

/** Picks stored before the reasoning flag was persisted don't know they can
 *  think, so their thinking toggle never appears. When the live list is in
 *  hand anyway, stamp the missing flag onto stored picks once. */
async function backfillReasoning(t: KnownTab, list: CloudModelPick[]): Promise<void> {
  const ep = epForTab(t)
  if (!ep) return
  const canThink = new Set(list.filter((m) => m.reasoning).map((m) => m.id))
  const patched = ep.models.map((m) =>
    m.reasoning === undefined && canThink.has(m.id) ? { ...m, reasoning: true } : m,
  )
  if (patched.some((m, i) => m !== ep.models[i])) await patchEp(ep.id, { models: patched })
}

watch(tab, (t) => {
  armRemove.value = null
  if (t !== 'custom') void loadTab(t)
})
onMounted(async () => {
  await models.refresh()
  void loadTab('openrouter')
})

// ── shared endpoint edits ───────────────────────────────────────────────────
async function patchEp(id: string, body: Record<string, unknown>): Promise<void> {
  await fetch(`/api/cloud/${id}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  await models.refresh()
}

/** The tab's endpoint row, created the moment it's first needed - the user's
 *  first act is a pick or a key, never a form for a known provider. */
async function ensureEp(t: KnownTab): Promise<string> {
  const ep = epForTab(t)
  if (ep) return ep.id
  const d = TAB_DEFS[t]
  const res = await fetch('/api/cloud', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: d.name, kind: d.kind, baseUrl: d.base, apiKey: '' }),
  })
  const j = (await res.json()) as { id: string }
  await models.refresh()
  return j.id
}

/** Persist only the stable facts - prices and blurbs drift with the
 *  provider's list and belong to the live browser, not the database. */
function stablePick(m: CloudModelPick): CloudModelPick {
  const keep: CloudModelPick = { id: m.id }
  if (m.display) keep.display = m.display
  if (m.ctx) keep.ctx = m.ctx
  // The reply cap is computed from this, so it is a stable fact, not picker
  // decoration - without it "Model maximum" falls back to the window and can
  // overshoot what the model will emit.
  if (m.maxOut) keep.maxOut = m.maxOut
  // Decides the model's KIND, so it is a stable fact and not picker decoration.
  if (m.asr) keep.asr = true
  if (m.vision !== undefined) keep.vision = m.vision
  if (m.reasoning !== undefined) keep.reasoning = m.reasoning
  if (m.provider) keep.provider = m.provider
  return keep
}

async function addForTab(t: KnownTab, m: CloudModelPick): Promise<void> {
  // No key yet, or the stored one failed its last check: the key dialog
  // carries the add; the model lands only once the key checks out.
  const cur = epForTab(t)
  if (!cur?.hasKey || keyCheck[cur.id]?.ok === false) {
    openKeyModal(t, m)
    return
  }
  const id = await ensureEp(t)
  const ep = models.cloudEndpoints.find((e) => e.id === id)
  if (!ep || ep.models.some((x) => pickKey(x) === pickKey(m))) return
  await patchEp(id, { models: [...ep.models, stablePick(m)] })
}

async function addModel(ep: CloudEndpoint, m: CloudModelPick): Promise<void> {
  if (ep.models.some((x) => pickKey(x) === pickKey(m))) return
  await patchEp(ep.id, { models: [...ep.models, stablePick(m)] })
}
async function removeModel(ep: CloudEndpoint, key: string): Promise<void> {
  await patchEp(ep.id, { models: ep.models.filter((x) => pickKey(x) !== key) })
}

// ── key checks ──────────────────────────────────────────────────────────────
// Every saved key is TESTED against the provider (does it authenticate, does
// an OpenAI-style API answer). Session state, keyed by endpoint id.
const keyCheck = reactive<Record<string, { ok: boolean; message?: string }>>({})
const checking = ref<string | null>(null)
async function checkKey(epId: string): Promise<void> {
  checking.value = epId
  try {
    const res = await fetch(`/api/cloud/${epId}/check`, { method: 'POST' })
    keyCheck[epId] = (await res.json()) as { ok: boolean; message?: string }
  } catch (e) {
    keyCheck[epId] = { ok: false, message: e instanceof Error ? e.message : String(e) }
  } finally {
    if (checking.value === epId) checking.value = null
  }
}

// ── the one key dialog ──────────────────────────────────────────────────────
// Key and search are different things: the key lives in a focused dialog that
// opens when actually needed - Add without a working key, the OpenAI/
// Anthropic browse gate, or Replace key in the pickers section. The dialog
// validates the key before the pending model lands.
const keyModal = reactive<{
  open: boolean
  tab: KnownTab | null
  epId: string | null
  pending: CloudModelPick | null
  error: string | null
  busy: boolean
}>({ open: false, tab: null, epId: null, pending: null, error: null, busy: false })
const modalKey = ref('')

function openKeyModal(t: KnownTab, pending: CloudModelPick | null = null): void {
  keyModal.open = true
  keyModal.tab = t
  keyModal.epId = null
  keyModal.pending = pending
  keyModal.error = null
  keyModal.busy = false
  modalKey.value = ''
}
/** Replace key from the pickers section - known providers reuse the tab
 *  path, a custom server targets its endpoint directly. */
function openKeyModalFor(ep: CloudEndpoint): void {
  const t = tabForEp(ep)
  if (t) {
    openKeyModal(t)
    return
  }
  keyModal.open = true
  keyModal.tab = null
  keyModal.epId = ep.id
  keyModal.pending = null
  keyModal.error = null
  keyModal.busy = false
  modalKey.value = ''
}
function closeKeyModal(): void {
  keyModal.open = false
  keyModal.pending = null
}
async function submitKeyModal(): Promise<void> {
  const k = modalKey.value.trim()
  if (!k || keyModal.busy || (!keyModal.tab && !keyModal.epId)) return
  keyModal.busy = true
  keyModal.error = null
  try {
    const id = keyModal.tab ? await ensureEp(keyModal.tab) : keyModal.epId!
    await patchEp(id, { apiKey: k })
    await checkKey(id)
    const verdict = keyCheck[id]
    if (!verdict?.ok) {
      keyModal.error = verdict?.message ?? 'The key check failed.'
      return
    }
    if (keyModal.pending) {
      const ep = models.cloudEndpoints.find((e) => e.id === id)
      if (ep && !ep.models.some((x) => pickKey(x) === pickKey(keyModal.pending!))) {
        await patchEp(id, { models: [...ep.models, stablePick(keyModal.pending)] })
      }
    }
    modalKey.value = ''
    closeKeyModal()
    if (keyModal.tab) void loadTab(keyModal.tab, true)
    else if (picker.id === id) {
      const fresh = endpoints.value.find((e) => e.id === id)
      if (fresh) void openPicker(fresh)
    }
  } finally {
    keyModal.busy = false
  }
}
const modalProvider = computed(() => {
  if (keyModal.tab) return TAB_DEFS[keyModal.tab].name
  return endpoints.value.find((e) => e.id === keyModal.epId)?.name ?? ''
})
const modalKeyHint = computed(() => (keyModal.tab ? TAB_DEFS[keyModal.tab].keyHint : 'API key'))

// Removing a provider drops its key and picks - inline confirm bar in the
// pickers section, never a modal.
const armRemove = ref<string | null>(null)
async function removeEndpoint(id: string): Promise<void> {
  armRemove.value = null
  await fetch(`/api/cloud/${id}`, { method: 'DELETE' })
  await models.refresh()
  // a tab whose endpoint just vanished must not keep serving its stale list
  for (const k of ['openrouter', 'openai', 'anthropic'] as const) {
    if (!epForTab(k) && browses[k].loaded) {
      Object.assign(browses[k], blank())
      if (k === 'openrouter') void loadTab(k)
    }
  }
}

// ── custom servers (any OpenAI-compatible endpoint) ─────────────────────────
const addName = ref('')
const addBase = ref('')
const addKey = ref('')
const addBusy = ref(false)
const addError = ref<string | null>(null)
const canAdd = computed(() => !addBusy.value && !!addName.value.trim() && !!addBase.value.trim())

async function addCustom(): Promise<void> {
  addBusy.value = true
  addError.value = null
  try {
    const res = await fetch('/api/cloud', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        name: addName.value.trim(),
        kind: 'openai-compat',
        baseUrl: addBase.value.trim(),
        apiKey: addKey.value.trim(),
      }),
    })
    const j = (await res.json()) as CloudEndpoint & { error?: { message?: string } }
    if (!res.ok) {
      addError.value = j?.error?.message ?? `Could not add (HTTP ${res.status})`
      return
    }
    addName.value = ''
    addBase.value = ''
    addKey.value = ''
    await models.refresh()
    const ep = endpoints.value.find((e) => e.id === j.id)
    if (ep) void openPicker(ep)
  } catch (e) {
    addError.value = e instanceof Error ? e.message : String(e)
  } finally {
    addBusy.value = false
  }
}

// custom endpoints keep the open-on-demand picker (fetch only; the browsing
// UI is ModelBrowser)
const picker = reactive<{
  id: string | null
  loading: boolean
  error: string | null
  list: CloudModelPick[]
  ranked: boolean
}>({ id: null, loading: false, error: null, list: [], ranked: false })

async function openPicker(ep: CloudEndpoint): Promise<void> {
  picker.id = ep.id
  picker.list = []
  picker.ranked = false
  picker.error = null
  picker.loading = true
  try {
    const res = await fetch(`/api/cloud/${ep.id}/models`)
    const j = (await res.json()) as {
      models?: CloudModelPick[]
      ranked?: boolean
      error?: { message?: string }
    }
    if (!res.ok) {
      picker.error = j?.error?.message ?? `The server didn't answer (HTTP ${res.status}).`
      return
    }
    picker.list = j.models ?? []
    picker.ranked = j.ranked ?? false
  } catch (e) {
    picker.error = e instanceof Error ? e.message : String(e)
  } finally {
    picker.loading = false
  }
}
function closePicker(): void {
  picker.id = null
}

function fmtCtx(n: number): string {
  return n >= 1_000_000 ? `${Number((n / 1e6).toFixed(1))}M` : `${Math.round(n / 1000)}K`
}
/** Same rule as the browser rows: the vendor mark names the maker, so
 *  OpenRouter's "DeepSeek: ..." display prefix is noise here too. */
function pickName(m: CloudModelPick): string {
  const d = m.display ?? m.id
  return cloudVendor(m.id) ? d.replace(/^[^:]{2,24}:\s+/, '') : d
}
</script>

<template>
  <div class="cl">
    <h1 class="cl__title">Cloud models</h1>
    <p class="cl__sub">
      API keys are stored locally and never with us.
    </p>

    <section class="cl__card">
      <div class="cl__tabs">
        <Tabs v-model="tab" :tabs="TAB_OPTIONS">
          <template #tab="{ tab: t }">
            <VendorLogo
              v-if="t.value !== 'custom'"
              :vendor="TAB_DEFS[t.value as KnownTab].name"
              :size="14"
            />
            <Icon v-else name="plug" :size="14" />
            {{ t.label }}
          </template>
        </Tabs>
        <span class="cl__spacer" />
        <template v-if="knownTab && curEp">
          <span v-if="checking === curEp.id" class="cl__keyok">
            <Icon name="spinner" :size="13" class="cl__spin" /> testing key...
          </span>
          <span v-else-if="curEp.hasKey && keyCheck[curEp.id]?.ok" class="cl__keyok">
            <Icon name="check" :size="13" /> key works
          </span>
          <Tooltip
            v-else-if="curEp.hasKey && keyCheck[curEp.id] && !keyCheck[curEp.id].ok && !keyModal.open"
            :label="keyCheck[curEp.id].message"
          >
            <span class="cl__keybad">
              <Icon name="alert-triangle" :size="13" /> key check failed
            </span>
          </Tooltip>
          <Menu>
            <MenuTrigger>
              <button
                class="pk-icon-btn"
                type="button"
                :aria-label="`${TAB_DEFS[knownTab].name} settings`"
              >
                <Icon name="more-horizontal" :size="16" />
              </button>
            </MenuTrigger>
            <MenuContent align="end" min-width="180px">
              <MenuItem v-if="curEp.hasKey" @select="checkKey(curEp!.id)">Test key</MenuItem>
              <MenuItem @select="openKeyModal(knownTab!)">
                {{ curEp.hasKey ? 'Replace key...' : 'Add key...' }}
              </MenuItem>
              <MenuItem @select="armRemove = curEp!.id">Remove provider...</MenuItem>
            </MenuContent>
          </Menu>
        </template>
      </div>
      <div class="cl__body">
        <template v-if="knownTab">
          <div v-if="curEp && (curEp.models.length || armRemove === curEp.id)" class="cl__own">
            <h3 v-if="curEp.models.length" class="cl__h3 cl__own-title">In your pickers</h3>
            <div v-if="armRemove === curEp.id" class="cl__confirm">
              <span>Remove {{ TAB_DEFS[knownTab].name }}? The key and model picks go with it.</span>
              <button
                class="pk-btn pk-btn--sm pk-btn--danger"
                type="button"
                @click="removeEndpoint(curEp.id)"
              >
                Remove
              </button>
              <button class="pk-btn pk-btn--sm" type="button" @click="armRemove = null">
                Keep
              </button>
            </div>
            <ul v-if="curEp.models.length" class="cl__models">
              <li v-for="m in curEp.models" :key="pickKey(m)" class="cl__model">
                <span class="cl__model-logo">
                  <VendorLogo v-if="pickVendor(m, knownTab)" :vendor="pickVendor(m, knownTab)!" :size="15" />
                </span>
                <span class="cl__model-cell">
                  <Tooltip :label="m.id">
                    <span class="cl__model-name">{{ pickName(m) }}</span>
                  </Tooltip>
                </span>
                <span class="cl__model-prov">
                  <template v-if="knownTab === 'openrouter'">
                    <Tooltip v-if="m.provider" label="Always routed to this provider">
                      <span class="cl__tag">{{ m.provider }}</span>
                    </Tooltip>
                    <Tooltip v-else label="OpenRouter picks the provider for each request">
                      <span class="cl__tag cl__tag--auto">auto</span>
                    </Tooltip>
                  </template>
                </span>
                <span class="cl__model-ctx">
                  <Tooltip v-if="m.ctx" label="Context window">
                    <span class="cl__num">{{ fmtCtx(m.ctx) }}</span>
                  </Tooltip>
                </span>
                <span class="cl__model-marks">
                  <Tooltip v-if="m.asr" label="Turns speech into text - takes a sound file, not a chat message">
                    <span class="cl__mark"><Icon name="microphone" :size="12" /></span>
                  </Tooltip>
                  <Tooltip v-if="m.vision" label="Reads images">
                    <span class="cl__mark"><Icon name="eye" :size="12" /></span>
                  </Tooltip>
                </span>
                <button
                  class="pk-icon-btn cl__model-x"
                  type="button"
                  :aria-label="`Remove ${m.id}`"
                  @click="removeModel(curEp!, pickKey(m))"
                >
                  <Icon name="x" :size="13" />
                </button>
              </li>
            </ul>
          </div>
          <template v-if="needsKeyToBrowse">
            <p class="cl__gate">
              {{ TAB_DEFS[knownTab].name }} shares its model list only with a key.
            </p>
            <div>
              <button class="pk-btn pk-btn--primary" type="button" @click="openKeyModal(knownTab)">
                Add API key...
              </button>
            </div>
          </template>
          <template v-else>
            <ModelBrowser
              class="cl__browser"
              :models="curBrowse!.list"
              :ranked="curBrowse!.ranked"
              :loading="curBrowse!.loading"
              :error="curBrowse!.error"
              :enabled="(curEp?.models ?? []).map(pickKey)"
              :name="TAB_DEFS[knownTab].name"
              :vendor="TAB_VENDOR[knownTab]"
              :expandable="knownTab === 'openrouter'"
              fill
              @add="(m) => addForTab(knownTab!, m)"
            />
          </template>
        </template>

        <template v-else>
          <p class="cl__gate">
            Any server that speaks the OpenAI chat API: another Paddock machine, vLLM, a gateway.
            Include the /v1 in the URL.
          </p>
          <div class="cl__form">
            <label class="cl__field">
              <span>Name</span>
              <TextInput v-model="addName" placeholder="e.g. Workstation vLLM" block />
            </label>
            <label class="cl__field">
              <span>Base URL</span>
              <TextInput v-model="addBase" placeholder="https://host/v1" block />
            </label>
            <label class="cl__field">
              <span>API key</span>
              <TextInput v-model="addKey" type="password" reveal placeholder="optional" block />
            </label>
          </div>
          <p v-if="addError" class="cl__error">{{ addError }}</p>
          <button class="pk-btn pk-btn--primary" type="button" :disabled="!canAdd" @click="addCustom">
            Add server
          </button>

          <div v-for="ep in customEndpoints" :key="ep.id" class="cl__cust">
            <div class="cl__status">
              <span class="cl__cust-name">{{ ep.name }}</span>
              <span class="cl__note">{{ ep.baseUrl }}</span>
              <span v-if="checking === ep.id" class="cl__keyok">
                <Icon name="spinner" :size="13" class="cl__spin" /> testing...
              </span>
              <span v-else-if="ep.hasKey && keyCheck[ep.id]?.ok" class="cl__keyok">
                <Icon name="check" :size="13" /> key works
              </span>
              <Tooltip
                v-else-if="ep.hasKey && keyCheck[ep.id] && !keyCheck[ep.id].ok && !keyModal.open"
                :label="keyCheck[ep.id].message"
              >
                <span class="cl__keybad"><Icon name="alert-triangle" :size="13" /> check failed</span>
              </Tooltip>
              <span class="cl__spacer" />
              <Menu>
                <MenuTrigger>
                  <button class="pk-icon-btn" type="button" :aria-label="`${ep.name} settings`">
                    <Icon name="more-horizontal" :size="16" />
                  </button>
                </MenuTrigger>
                <MenuContent align="end" min-width="180px">
                  <MenuItem v-if="ep.hasKey" @select="checkKey(ep.id)">Test key</MenuItem>
                  <MenuItem @select="openKeyModalFor(ep)">
                    {{ ep.hasKey ? 'Replace key...' : 'Add key...' }}
                  </MenuItem>
                  <MenuItem @select="armRemove = ep.id">Remove server...</MenuItem>
                </MenuContent>
              </Menu>
            </div>
            <div v-if="armRemove === ep.id" class="cl__confirm">
              <span>Remove {{ ep.name }}? The key and model picks go with it.</span>
              <button
                class="pk-btn pk-btn--sm pk-btn--danger"
                type="button"
                @click="removeEndpoint(ep.id)"
              >
                Remove
              </button>
              <button class="pk-btn pk-btn--sm" type="button" @click="armRemove = null">Keep</button>
            </div>
            <template v-if="ep.models.length">
              <h3 class="cl__h3">In your pickers</h3>
              <ul class="cl__models">
                <li v-for="m in ep.models" :key="pickKey(m)" class="cl__model">
                  <span class="cl__model-logo">
                    <VendorLogo v-if="cloudVendor(m.id)" :vendor="cloudVendor(m.id)!" :size="15" />
                  </span>
                  <span class="cl__model-cell">
                    <Tooltip :label="m.id">
                      <span class="cl__model-name">{{ pickName(m) }}</span>
                    </Tooltip>
                  </span>
                  <span class="cl__model-prov" />
                  <span class="cl__model-ctx">
                    <Tooltip v-if="m.ctx" label="Context window">
                      <span class="cl__num">{{ fmtCtx(m.ctx) }}</span>
                    </Tooltip>
                  </span>
                  <span class="cl__model-marks">
                    <Tooltip v-if="m.vision" label="Reads images">
                      <span class="cl__mark"><Icon name="eye" :size="12" /></span>
                    </Tooltip>
                  </span>
                  <button
                    class="pk-icon-btn cl__model-x"
                    type="button"
                    :aria-label="`Remove ${m.id}`"
                    @click="removeModel(ep, pickKey(m))"
                  >
                    <Icon name="x" :size="13" />
                  </button>
                </li>
              </ul>
            </template>
            <template v-if="picker.id !== ep.id">
              <button class="pk-btn" type="button" @click="openPicker(ep)">Choose models...</button>
            </template>
            <div v-else>
              <div class="cl__done">
                <button class="pk-btn pk-btn--sm" type="button" @click="closePicker">Done</button>
              </div>
              <ModelBrowser
                :models="picker.list"
                :ranked="picker.ranked"
                :loading="picker.loading"
                :error="picker.error"
                :enabled="ep.models.map(pickKey)"
                :name="ep.name"
                @add="(m) => addModel(ep, m)"
              />
            </div>
          </div>
        </template>
      </div>
    </section>

    <Dialog
      :open="keyModal.open"
      :busy="keyModal.busy"
      :title="`${modalProvider} API key`"
      icon="cloud"
      size="sm"
      @close="closeKeyModal"
    >
      <p class="cl__modal-note">
        <template v-if="keyModal.pending">
          Chatting with "{{ pickName(keyModal.pending) }}" needs your {{ modalProvider }} API key.
        </template>
        <template v-else-if="keyModal.tab && keyModal.tab !== 'openrouter'">
          {{ modalProvider }} needs your API key to list and chat with its models.
        </template>
        <template v-else> Your {{ modalProvider }} API key unlocks chatting. </template>
        It stays on this machine and is only ever sent to {{ modalProvider }}.
      </p>
      <TextInput
        v-model="modalKey"
        type="password"
        reveal
        :placeholder="modalKeyHint"
        block
        :disabled="keyModal.busy"
        @keydown.enter="submitKeyModal"
      />
      <p v-if="keyModal.error" class="cl__error cl__modal-err">{{ keyModal.error }}</p>
      <template #footer>
        <button class="pk-btn" type="button" :disabled="keyModal.busy" @click="closeKeyModal">
          Cancel
        </button>
        <button
          class="pk-btn pk-btn--primary"
          type="button"
          :disabled="keyModal.busy || !modalKey.trim()"
          @click="submitKeyModal"
        >
          <Icon v-if="keyModal.busy" name="spinner" :size="13" class="cl__spin" />
          {{ keyModal.busy ? 'Checking...' : keyModal.pending ? 'Save & add' : 'Save key' }}
        </button>
      </template>
    </Dialog>
  </div>
</template>

<style scoped>
/* the page is a COLUMN that claims the whole content height, so the browse
   card (and its model list) can fill the viewport */
.cl {
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
  align-self: stretch;
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.cl__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin-bottom: 8px;
}
.cl__sub {
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
  margin: 0 0 20px;
  max-width: 70ch;
}
/* the tab's own block: what you added on this provider. Rendered only when
   picks exist - key state + the endpoint menu live at the far right of the
   tab strip, so an empty keyed provider adds NOTHING to
   the body. */
.cl__own {
  border-bottom: 1px solid var(--pk-border-subtle);
  margin-bottom: 14px;
  padding-bottom: 8px;
}
.cl__own-title {
  margin-bottom: 4px;
}
.cl__h3 {
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
  margin: 0;
}
.cl__spacer {
  flex: 1;
}
.cl__keyok {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--pk-status-ok, var(--pk-accent));
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
}
.cl__keybad {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--pk-status-warning, var(--pk-text-secondary));
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
}
.cl__spin {
  animation: cl-spin 0.9s linear infinite;
}
@keyframes cl-spin {
  to {
    transform: rotate(360deg);
  }
}
.cl__confirm {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin: 6px 0;
  padding: 8px 12px;
  border: 1px solid var(--pk-status-warning, var(--pk-border-default));
  border-radius: var(--pk-radius-md);
  color: var(--pk-text-primary);
  font-size: var(--pk-font-size-sm);
}
.cl__models {
  list-style: none;
  margin: 0 0 6px;
  padding: 0;
  display: flex;
  flex-direction: column;
}
/* pick rows share the browser's column discipline: logo · name · provider ·
   ctx · marks · remove */
.cl__model {
  display: grid;
  grid-template-columns: 18px minmax(0, 1fr) 110px 44px 24px 28px;
  align-items: center;
  gap: 8px;
  padding: 5px 8px;
  border-radius: var(--pk-radius-sm);
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
}
.cl__model:hover {
  background: var(--pk-bg-hover);
}
.cl__model-logo {
  display: inline-flex;
  justify-content: center;
}
.cl__model-cell {
  min-width: 0;
  display: flex;
}
.cl__model-name {
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  cursor: default;
  min-width: 0;
}
.cl__model-prov {
  text-align: right;
  white-space: nowrap;
  justify-self: end;
}
.cl__model-ctx {
  text-align: right;
  white-space: nowrap;
}
.cl__num {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  cursor: default;
}
.cl__tag {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-full);
  padding: 0 7px;
  white-space: nowrap;
  cursor: default;
  display: inline-block;
}
.cl__tag--auto {
  border-style: dashed;
}
.cl__model-marks {
  display: inline-flex;
  justify-content: flex-end;
}
.cl__mark {
  display: inline-flex;
  color: var(--pk-text-muted);
}
.cl__model-x {
  flex: none;
}

/* the browse card - the tab strip carries the active provider's key state
   and endpoint menu at its far right; the underline moves onto the full
   strip so it spans past the chrome */
.cl__card {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  overflow: hidden;
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.cl__tabs {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 12px;
  background: var(--pk-bg-elevated);
  border-bottom: 1px solid var(--pk-border-default);
}
.cl__tabs :deep(.pk-tabs) {
  border-bottom: none;
}
.cl__body {
  padding: 16px 20px 18px;
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: auto;
}
.cl__gate {
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
  margin: 0 0 10px;
  max-width: 60ch;
}
.cl__browser {
  min-height: 0;
}
.cl__done {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 6px;
}
.cl__error {
  color: var(--pk-status-error, #d33);
  font-size: var(--pk-font-size-sm);
  margin: 0 0 10px;
}
.cl__modal-note {
  margin: 0;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
}
.cl__modal-err {
  margin: 0;
}
.cl__form {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 12px;
}
.cl__field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1 1 220px;
  min-width: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.cl__status {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.cl__note {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}
/* one custom server block; a rule between servers, never nested cards */
.cl__cust {
  border-top: 1px solid var(--pk-border-subtle);
  margin-top: 16px;
  padding-top: 14px;
}
.cl__cust-name {
  font-weight: 600;
  color: var(--pk-text-primary);
}
.cl__cust .cl__h3 {
  margin: 10px 0 6px;
}
</style>
