<script setup lang="ts">
// The connector library, built like the Cloud models page: two tabs on one
// surface card - "Find connectors" is a viewport-filling TABLE of the public
// registry (search + filter chips + click-to-sort headers, the ModelBrowser
// idiom), "Your connectors" is the installed list, no scroll-past-the-catalog
// to reach it. registry.truespar.com is queried straight from the browser
// (keyless, CORS-open, the manager never in the path); with no query it
// serves the ranked catalog, so the table is browsable before a keystroke.
// The catalog is DISCOVERED, not VETTED (its own words): the note rides into
// the UI and liveness is a check result, never an endorsement. Only the
// stable registry `key` is persisted - rank/liveness churn every sync.
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useConnectorsStore, type Connector } from '@/stores/connectors'
import { useFleetStore } from '@/stores/fleet'
import Dialog from '@/components/ui/Dialog.vue'
import Switch from '@/components/ui/Switch.vue'
import Tabs from '@/components/ui/Tabs.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import Icon from '@/components/Icon.vue'

const connectors = useConnectorsStore()

// ── registry data ───────────────────────────────────────────────────────────
interface RegistryHit {
  key: string
  name: string
  description: string
  domain: string
  authorityTier: string
  liveness: string
  githubStars: number | null
  toolCount: number | null
  remoteEndpoints?: Array<{ transport: string; url: string }>
}
const hits = ref<RegistryHit[]>([])
const busy = ref(false)
const fetchError = ref('')
let searchTimer = 0
let queryGen = 0

async function fetchHits(q: string): Promise<void> {
  const gen = ++queryGen
  busy.value = true
  fetchError.value = ''
  try {
    const qs = q.length >= 2 ? `q=${encodeURIComponent(q)}&` : ''
    const res = await fetch(`https://registry.truespar.com/v1/servers?${qs}hostable=true&limit=50`)
    if (!res.ok) throw new Error(`registry answered ${res.status}`)
    const body = (await res.json()) as { results?: RegistryHit[] }
    if (gen !== queryGen) return // a newer query superseded this one
    hits.value = body.results ?? []
  } catch (e) {
    if (gen === queryGen) fetchError.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (gen === queryGen) busy.value = false
  }
}
const fleet = useFleetStore()
onMounted(() => {
  void connectors.refresh()
  void fleet.refresh()
  void fetchHits('')
})
/** The configured models a connector can be scoped to (port + name). */
const scopeModels = computed(() =>
  fleet.configured.map((c) => ({ port: c.port, name: c.display || String(c.port) })),
)

// ── view state: tab, search, filters, sort (the ModelBrowser idiom) ─────────
const tab = ref<'search' | 'installed'>('search')
const tabOptions = computed(() => [
  { value: 'search', label: 'Find connectors' },
  { value: 'installed', label: `Your connectors (${connectors.list.length})` },
])
type SortKey = 'rank' | 'name' | 'tier' | 'tools' | 'stars' | 'status'
const view = reactive<{
  query: string
  sort: SortKey
  dir: 1 | -1
  fReachable: boolean
  tiers: Set<string>
}>({ query: '', sort: 'rank', dir: 1, fReachable: false, tiers: new Set() })

function onSearchInput(): void {
  clearTimeout(searchTimer)
  searchTimer = window.setTimeout(() => void fetchHits(view.query.trim()), 300)
}
function setSort(k: SortKey): void {
  if (view.sort === k) {
    view.dir = view.dir === 1 ? -1 : 1
  } else {
    view.sort = k
    // sensible first direction per column: most tools / most stars first
    view.dir = k === 'tools' || k === 'stars' ? -1 : 1
  }
}
function sortState(k: SortKey): 'ascending' | 'descending' | undefined {
  if (view.sort !== k) return undefined
  return view.dir === 1 ? 'ascending' : 'descending'
}
function toggleTier(bucket: string): void {
  if (view.tiers.has(bucket)) view.tiers.delete(bucket)
  else view.tiers.add(bucket)
}

// Nobody reads "S" - they read "First-party" (provenance signal, not a
// safety guarantee; the bucketing hq's Discover tab settled on).
function tierBucket(t: string): string {
  if (t === 'S') return 'First-party'
  if (t === 'A') return 'Trusted'
  return 'Community'
}
const TIER_ORD: Record<string, number> = { S: 0, A: 1, B: 2, C: 3, D: 4 }
// Reachability: one icon per state, the honest words living in its tooltip -
// a check RESULT, never an endorsement.
const LIVENESS: Record<string, { icon: string; cls: string; label: string; ord: number }> = {
  'ok-tools': { icon: 'check-circle', cls: 'ok', label: 'Reachable at last check', ord: 0 },
  ok: { icon: 'check-circle', cls: 'ok', label: 'Reachable at last check', ord: 1 },
  'auth-required': {
    icon: 'lock',
    cls: 'auth',
    label: 'Reachable - needs credentials',
    ord: 2,
  },
  dead: { icon: 'x-circle', cls: 'dead', label: 'Unreachable at last check', ord: 3 },
}
function reachable(h: RegistryHit): boolean {
  return h.liveness !== 'dead'
}
/** Catalog names often stuff a tagline behind an em-dash ("xmp4 - Semantic
 *  code knowledge for..."). The column shows the NAME; the pitch lives in the
 *  fold's description. Stray long dashes become plain hyphens. */
function displayName(h: RegistryHit): string {
  const raw = h.name || h.domain
  return raw.split(/\s+[--|]\s+/)[0].replace(/[--]/g, '-').trim() || h.domain
}

const hasTools = computed(() => hits.value.some((h) => h.toolCount))
const hasStars = computed(() => hits.value.some((h) => h.githubStars))
const colCount = computed(() => 5 + (hasTools.value ? 1 : 0) + (hasStars.value ? 1 : 0))

// ── foldable rows (the ModelBrowser expandable idiom): the registry's detail
// call has more to say than a table row - full description, tool names,
// endpoint, license, links, and its own per-server connection note.
interface HitDetail {
  description?: string
  categories?: string[]
  spdxLicense?: string | null
  tools?: string[]
  repoUrl?: string | null
  homepage?: string | null
  lastHandshakeAt?: string | null
  connection?: {
    recommended?: { url?: string } | null
    authRequired?: boolean | null
    note?: string
  } | null
}
const expanded = reactive<Record<string, boolean>>({})
const details = reactive<Record<string, { loading: boolean; error: string; d: HitDetail | null }>>(
  {},
)
async function toggleRow(h: RegistryHit): Promise<void> {
  expanded[h.key] = !expanded[h.key]
  if (!expanded[h.key] || details[h.key]) return
  details[h.key] = { loading: true, error: '', d: null }
  try {
    const res = await fetch(`https://registry.truespar.com/v1/servers/${h.key}`)
    if (!res.ok) throw new Error(`registry answered ${res.status}`)
    details[h.key].d = ((await res.json()) as { server?: HitDetail }).server ?? null
  } catch (e) {
    details[h.key].error = e instanceof Error ? e.message : String(e)
  } finally {
    details[h.key].loading = false
  }
}

const shown = computed<RegistryHit[]>(() => {
  let list = hits.value
  if (view.fReachable) list = list.filter(reachable)
  if (view.tiers.size) list = list.filter((h) => view.tiers.has(tierBucket(h.authorityTier)))
  if (view.sort === 'rank') return list
  const d = view.dir
  return [...list].sort((a, b) => {
    switch (view.sort) {
      case 'name':
        return d * (a.name || a.domain).localeCompare(b.name || b.domain)
      case 'tier':
        return d * ((TIER_ORD[a.authorityTier] ?? 9) - (TIER_ORD[b.authorityTier] ?? 9))
      case 'tools':
        return d * ((a.toolCount ?? 0) - (b.toolCount ?? 0))
      case 'stars':
        return d * ((a.githubStars ?? 0) - (b.githubStars ?? 0))
      case 'status':
        return d * ((LIVENESS[a.liveness]?.ord ?? 9) - (LIVENESS[b.liveness]?.ord ?? 9))
      default:
        return 0
    }
  })
})

// ── adding ──────────────────────────────────────────────────────────────────
/** Already in the library? Matched on the stable registry key, or on the
 *  endpoint URL for hand-entered twins. */
function added(hit: RegistryHit): boolean {
  const url = hit.remoteEndpoints?.[0]?.url
  return connectors.list.some((c) => c.registryKey === hit.key || (!!url && c.url === url))
}
/** A usable slug for the wire label: the name's tail when it says something,
 *  else the meaningful part of the domain. */
function slugFrom(hit: RegistryHit): string {
  const generic = new Set(['mcp', 'api', 'www', 'app', 'gateway', 'server'])
  const nameTail = (hit.name.split('/').pop() || '').toLowerCase()
  const fromDomain = hit.domain.split('.').find((p) => p && !generic.has(p.toLowerCase())) || ''
  const raw = (!generic.has(nameTail) && nameTail) || fromDomain || 'connector'
  const s = raw
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return (s || 'connector').slice(0, 64)
}
/** The detail call carries the recommended endpoint + a definite auth flag
 *  (under `connection`); an already-expanded row's cached detail is reused,
 *  and the search row's endpoint stands if the call fails. */
async function resolveEndpoint(hit: RegistryHit): Promise<{ url: string; auth: boolean }> {
  let url = hit.remoteEndpoints?.[0]?.url ?? ''
  let auth = hit.liveness === 'auth-required'
  let d = details[hit.key]?.d
  if (!d) {
    try {
      const res = await fetch(`https://registry.truespar.com/v1/servers/${hit.key}`)
      if (res.ok) d = ((await res.json()) as { server?: HitDetail }).server ?? null
    } catch {
      // search-row data stands
    }
  }
  if (d?.connection?.recommended?.url) url = d.connection.recommended.url
  if (d?.connection?.authRequired) auth = true
  return { url, auth }
}
// One click adds a reachable no-auth server outright; servers that want
// credentials (and label collisions) land in the form instead, prefilled.
const addingKey = ref('')
async function addHit(hit: RegistryHit): Promise<void> {
  if (!reachable(hit) || added(hit) || addingKey.value) return
  addingKey.value = hit.key
  try {
    const { url, auth } = await resolveEndpoint(hit)
    if (!url) return
    if (auth) {
      openForm({ label: slugFrom(hit), url, registryKey: hit.key, authHint: true })
      return
    }
    try {
      await connectors.save({ label: slugFrom(hit), url, headers: {}, registryKey: hit.key })
    } catch (e) {
      openForm({
        label: slugFrom(hit),
        url,
        registryKey: hit.key,
        error: e instanceof Error ? e.message : String(e),
      })
    }
  } finally {
    addingKey.value = ''
  }
}

// ── form dialog (edit + add-by-URL + credential-needed adds) ────────────────
const editId = ref<string | undefined>(undefined)
const editLabel = ref('')
const editUrl = ref('')
const editHeaders = ref<Array<{ k: string; v: string }>>([])
const editRegistryKey = ref('')
const editError = ref('')
const editorOpen = ref(false)
const authHint = ref(false)
// The scope choice lives here, with room for its explanation - the list row
// only shows the resulting badge (a cramped in-row switch was the wrong UI).
// One scope, two views: this picker and each model's edit page write the
// same per-model membership.
const editAll = ref(false)
const editPorts = ref<number[]>([])
const editAllWas = ref(false)
const editPortsWas = ref<number[]>([])
function togglePort(p: number): void {
  editPorts.value = editPorts.value.includes(p)
    ? editPorts.value.filter((x) => x !== p)
    : [...editPorts.value, p]
}
const editorTitle = computed(() => (editId.value ? 'Edit connector' : 'Add connector'))
const canSave = computed(() => !!editLabel.value.trim() && !!editUrl.value.trim())

function openForm(pre?: {
  label?: string
  url?: string
  registryKey?: string
  authHint?: boolean
  error?: string
}): void {
  editId.value = undefined
  editLabel.value = pre?.label ?? ''
  editUrl.value = pre?.url ?? ''
  editHeaders.value = []
  editRegistryKey.value = pre?.registryKey ?? ''
  editError.value = pre?.error ?? ''
  authHint.value = pre?.authHint ?? false
  editAll.value = false
  editPorts.value = []
  editAllWas.value = false
  editPortsWas.value = []
  editorOpen.value = true
}
function edit(c: Connector): void {
  editId.value = c.id
  editLabel.value = c.label
  editUrl.value = c.url
  editHeaders.value = Object.entries(c.headers).map(([k, v]) => ({ k, v }))
  editRegistryKey.value = c.registryKey
  editError.value = ''
  authHint.value = false
  editAll.value = c.system
  editPorts.value = [...c.ports]
  editAllWas.value = c.system
  editPortsWas.value = [...c.ports]
  editorOpen.value = true
}
// Save is check-then-save: the URL must answer an MCP handshake first. A 401
// is reachable (credentials come next - the auth warning shows); anything
// else blocks once and the button turns into "Save anyway" for servers that
// are merely down right now.
const checking = ref(false)
const forceSave = ref(false)
async function saveEdit(): Promise<void> {
  if (!canSave.value || checking.value) return
  const label = editLabel.value.trim()
  const headers: Record<string, string> = {}
  for (const h of editHeaders.value) {
    if (h.k.trim()) headers[h.k.trim()] = h.v
  }
  if (!forceSave.value) {
    checking.value = true
    try {
      const res = await connectors.check(editUrl.value.trim(), headers)
      if (res.authRequired) {
        authHint.value = true
      } else if (!res.ok) {
        editError.value = `No MCP handshake at this URL - ${res.error ?? 'no answer'}`
        forceSave.value = true
        return
      }
    } finally {
      checking.value = false
    }
  }
  try {
    await connectors.save(
      { label, url: editUrl.value.trim(), headers, registryKey: editRegistryKey.value || undefined },
      editId.value,
    )
    const scopeChanged =
      editAll.value !== editAllWas.value ||
      [...editPorts.value].sort().join() !== [...editPortsWas.value].sort().join()
    if (scopeChanged) {
      const row = connectors.list.find((x) => (editId.value ? x.id === editId.value : x.label === label))
      if (row) await connectors.setScope(row.id, editAll.value, editPorts.value)
    }
    editorOpen.value = false
  } catch (e) {
    editError.value = e instanceof Error ? e.message : String(e)
  }
}

// ── OAuth sign-in (existing connectors) ─────────────────────────────────────
// The manager runs the whole flow (discovery, registration, PKCE); this side
// only opens the consent tab and watches for the row to turn `connected`.
const oauthBusy = ref(false)
const oauthClientId = ref('')
const needClientId = ref(false)
let oauthPoll = 0
function stopOauthPoll(): void {
  clearInterval(oauthPoll)
  oauthBusy.value = false
}
async function connect(): Promise<void> {
  if (!editId.value || oauthBusy.value) return
  editError.value = ''
  try {
    const url = await connectors.oauthStart(editId.value, oauthClientId.value.trim() || undefined)
    window.open(url, '_blank', 'noopener')
    oauthBusy.value = true
    const started = Date.now()
    oauthPoll = window.setInterval(() => {
      void connectors.refresh().then(() => {
        const c = editId.value ? connectors.byId(editId.value) : undefined
        if (c?.connected || Date.now() - started > 180_000) stopOauthPoll()
      })
    }, 2000)
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e)
    editError.value = msg
    if (msg.includes('client id')) needClientId.value = true
  }
}
async function disconnect(): Promise<void> {
  if (!editId.value) return
  await connectors.oauthDisconnect(editId.value)
}
const editConnected = computed(
  () => !!editId.value && connectors.byId(editId.value)?.connected === true,
)
watch(editorOpen, (open) => {
  if (!open) {
    stopOauthPoll()
    needClientId.value = false
    oauthClientId.value = ''
  }
  forceSave.value = false
})
// a corrected URL earns a fresh check, not a lingering "Save anyway"
watch(editUrl, () => {
  forceSave.value = false
})

// ── delete confirm ──────────────────────────────────────────────────────────
const pendingDelete = ref<Connector | null>(null)
async function confirmDelete(): Promise<void> {
  const c = pendingDelete.value
  pendingDelete.value = null
  if (c) await connectors.remove(c.id)
}
</script>

<template>
  <div class="conn">
    <header class="conn__head">
      <div>
        <h1 class="conn__title">Connectors</h1>
      </div>
    </header>

    <div class="conn__card">
      <div class="conn__tabs">
        <Tabs v-model="tab" :tabs="tabOptions" />
      </div>

      <div v-if="tab === 'search'" class="conn__body">
        <input
          v-model="view.query"
          class="pk-input conn__search"
          placeholder="Search the MCP directory - stripe, github, weather..."
          spellcheck="false"
          @input="onSearchInput"
        />
        <div class="conn__tools">
          <button
            type="button"
            class="conn__chip"
            :class="{ 'conn__chip--on': view.fReachable }"
            :aria-pressed="view.fReachable"
            @click="view.fReachable = !view.fReachable"
          >
            Reachable
          </button>
          <span class="conn__tools-sep" />
          <button
            v-for="b in ['First-party', 'Trusted', 'Community']"
            :key="b"
            type="button"
            class="conn__chip"
            :class="{ 'conn__chip--on': view.tiers.has(b) }"
            :aria-pressed="view.tiers.has(b)"
            @click="toggleTier(b)"
          >
            {{ b }}
          </button>
        </div>
        <p v-if="fetchError" class="conn__error">Directory not answering: {{ fetchError }}</p>
        <p v-else-if="busy && !hits.length" class="conn__hint">
          <Icon name="spinner" :size="13" class="conn__spin" /> Asking the directory...
        </p>
        <p v-else-if="!shown.length" class="conn__hint">
          Nothing matches{{ view.query.trim() ? ` "${view.query.trim()}"` : '' }} with these
          filters.
        </p>
        <div v-if="shown.length" class="conn__tablewrap">
          <table class="conn__table">
            <thead>
              <tr>
                <th class="th-sort conn__th-name" :aria-sort="sortState('name')" @click="setSort('name')">
                  Server
                  <Icon v-if="view.sort === 'name'" :name="view.dir === 1 ? 'chevron-up' : 'chevron-down'" :size="11" />
                </th>
                <th class="conn__th-domain">Domain</th>
                <th class="th-sort" :aria-sort="sortState('tier')" @click="setSort('tier')">
                  Provenance
                  <Icon v-if="view.sort === 'tier'" :name="view.dir === 1 ? 'chevron-up' : 'chevron-down'" :size="11" />
                </th>
                <th v-if="hasTools" class="th-sort conn__th-r" :aria-sort="sortState('tools')" @click="setSort('tools')">
                  Tools
                  <Icon v-if="view.sort === 'tools'" :name="view.dir === 1 ? 'chevron-up' : 'chevron-down'" :size="11" />
                </th>
                <th v-if="hasStars" class="th-sort conn__th-r" :aria-sort="sortState('stars')" @click="setSort('stars')">
                  ★
                  <Icon v-if="view.sort === 'stars'" :name="view.dir === 1 ? 'chevron-up' : 'chevron-down'" :size="11" />
                </th>
                <th class="th-sort" :aria-sort="sortState('status')" @click="setSort('status')">
                  Status
                  <Icon v-if="view.sort === 'status'" :name="view.dir === 1 ? 'chevron-up' : 'chevron-down'" :size="11" />
                </th>
                <th class="conn__th-act" />
              </tr>
            </thead>
            <tbody>
              <template v-for="h in shown" :key="h.key">
                <tr
                  class="conn__row-x"
                  :class="{ 'conn__row-dead': !reachable(h) }"
                  @click="toggleRow(h)"
                >
                  <td class="conn__td-name">
                    <Icon
                      :name="expanded[h.key] ? 'chevron-down' : 'chevron-right'"
                      :size="12"
                      class="conn__caret"
                    />
                    <span class="conn__name">{{ displayName(h) }}</span>
                  </td>
                  <td class="conn__td-domain">{{ h.domain }}</td>
                  <td>
                    <span class="conn__badge">{{ tierBucket(h.authorityTier) }}</span>
                  </td>
                  <td v-if="hasTools" class="conn__td-r">
                    <span v-if="h.toolCount" class="conn__num">{{ h.toolCount }}</span>
                  </td>
                  <td v-if="hasStars" class="conn__td-r">
                    <span v-if="h.githubStars" class="conn__num">{{ h.githubStars }}</span>
                  </td>
                  <td class="conn__td-c">
                    <Tooltip v-if="LIVENESS[h.liveness]" :label="LIVENESS[h.liveness].label">
                      <span class="conn__live" :class="`conn__live--${LIVENESS[h.liveness].cls}`">
                        <Icon :name="LIVENESS[h.liveness].icon" :size="15" />
                      </span>
                    </Tooltip>
                  </td>
                  <td class="conn__td-act">
                    <span v-if="added(h)" class="conn__in"
                      ><Icon name="check" :size="13" /> added</span
                    >
                    <button
                      v-else-if="reachable(h)"
                      class="pk-btn pk-btn--sm"
                      type="button"
                      :disabled="addingKey === h.key"
                      @click.stop="addHit(h)"
                    >
                      {{ addingKey === h.key ? 'Adding...' : 'Add' }}
                    </button>
                  </td>
                </tr>
                <tr v-if="expanded[h.key]" class="conn__detail">
                  <td :colspan="colCount">
                    <p v-if="details[h.key]?.loading" class="conn__hint">
                      <Icon name="spinner" :size="13" class="conn__spin" /> Asking the directory...
                    </p>
                    <p v-else-if="details[h.key]?.error" class="conn__error">
                      {{ details[h.key].error }}
                    </p>
                    <div v-else-if="details[h.key]?.d" class="conn__dd">
                      <p v-if="details[h.key].d!.description" class="conn__dd-desc">
                        {{ details[h.key].d!.description }}
                      </p>
                      <dl class="conn__dd-facts">
                        <div v-if="details[h.key].d!.connection?.recommended?.url" class="conn__dd-fact">
                          <dt>Endpoint</dt>
                          <dd>
                            <code>{{ details[h.key].d!.connection!.recommended!.url }}</code>
                          </dd>
                        </div>
                        <div v-if="details[h.key].d!.categories?.length" class="conn__dd-fact">
                          <dt>Categories</dt>
                          <dd>{{ details[h.key].d!.categories!.join(', ') }}</dd>
                        </div>
                        <div v-if="details[h.key].d!.spdxLicense" class="conn__dd-fact">
                          <dt>License</dt>
                          <dd>{{ details[h.key].d!.spdxLicense }}</dd>
                        </div>
                      </dl>
                      <div v-if="details[h.key].d!.tools?.length" class="conn__dd-tools">
                        <code v-for="t in details[h.key].d!.tools!.slice(0, 24)" :key="t">{{
                          t
                        }}</code>
                        <span v-if="details[h.key].d!.tools!.length > 24" class="conn__dd-more">
                          +{{ details[h.key].d!.tools!.length - 24 }} more
                        </span>
                      </div>
                      <div class="conn__dd-links">
                        <a
                          v-if="details[h.key].d!.repoUrl"
                          :href="details[h.key].d!.repoUrl!"
                          target="_blank"
                          rel="noopener noreferrer"
                          @click.stop
                        >
                          <Icon name="external-link" :size="13" /> Repository
                        </a>
                        <a
                          v-if="details[h.key].d!.homepage"
                          :href="details[h.key].d!.homepage!"
                          target="_blank"
                          rel="noopener noreferrer"
                          @click.stop
                        >
                          <Icon name="globe" :size="13" /> Homepage
                        </a>
                      </div>
                    </div>
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
        <p v-if="shown.length" class="conn__note">
          Public servers from the open MCP ecosystem - not vetted here.
        </p>
      </div>

      <div v-else class="conn__body conn__body--scroll">
        <div v-if="connectors.loaded && connectors.list.length === 0" class="conn__empty">
          <span class="conn__empty-mark"><Icon name="plug" :size="26" /></span>
          <p>Nothing added yet.</p>
          <button class="pk-btn pk-btn--primary" @click="openForm()">
            <Icon name="plus" :size="15" /> Add by URL
          </button>
        </div>
        <div v-else class="conn__mine-bar">
          <button class="pk-btn pk-btn--ghost" type="button" @click="openForm()">
            <Icon name="plus" :size="15" /> Add by URL
          </button>
        </div>
        <ul v-if="connectors.list.length" class="conn__list">
          <li v-for="c in connectors.list" :key="c.id" class="ccard">
            <button class="ccard__main" type="button" @click="edit(c)">
              <span class="ccard__label">{{ c.label }}</span>
              <span class="ccard__url">{{ c.url }}</span>
            </button>
            <span v-if="Object.keys(c.headers).length" class="ccard__badge">
              {{ Object.keys(c.headers).length }} header{{ Object.keys(c.headers).length > 1 ? 's' : '' }}
            </span>
            <Tooltip
              v-if="c.system || c.ports.length"
              label="These models serve this tool as their own - apps calling their APIs see it too"
            >
              <span class="ccard__badge ccard__badge--sys">
                {{ c.system ? 'Every model' : `${c.ports.length} model${c.ports.length > 1 ? 's' : ''}` }}
              </span>
            </Tooltip>
            <div class="ccard__acts">
              <button class="pk-icon-btn" aria-label="Edit connector" @click="edit(c)">
                <Icon name="edit" :size="16" />
              </button>
              <button
                class="pk-icon-btn ccard__del"
                aria-label="Remove connector"
                @click="pendingDelete = c"
              >
                <Icon name="trash" :size="16" />
              </button>
            </div>
          </li>
        </ul>
      </div>
    </div>
  </div>

  <Dialog :open="editorOpen" :title="editorTitle" icon="plug" size="lg" @close="editorOpen = false">
    <div class="cedit">
      <p v-if="authHint" class="cedit__auth">
        This server reported it needs credentials - add its auth header below.
      </p>
      <label class="cedit__field">
        <span class="cedit__label">Name</span>
        <input v-model="editLabel" class="pk-input" placeholder="e.g. mcp-registry" />
      </label>
      <label class="cedit__field">
        <span class="cedit__label">Server URL</span>
        <input v-model="editUrl" class="pk-input" placeholder="https://registry.truespar.com/mcp" />
      </label>
      <div class="cedit__field">
        <span class="cedit__label">Headers</span>
        <div v-for="(h, i) in editHeaders" :key="i" class="cedit__hrow">
          <input v-model="h.k" class="pk-input cedit__hkey" placeholder="Authorization" />
          <input v-model="h.v" class="pk-input cedit__hval" placeholder="Bearer ..." />
          <button
            class="pk-icon-btn"
            type="button"
            aria-label="Remove header"
            @click="editHeaders.splice(i, 1)"
          >
            <Icon name="x" :size="14" />
          </button>
        </div>
        <button
          class="pk-btn pk-btn--ghost pk-btn--sm cedit__hadd"
          type="button"
          @click="editHeaders.push({ k: '', v: '' })"
        >
          <Icon name="plus" :size="13" /> Add header
        </button>
      </div>
      <div v-if="editId" class="cedit__field">
        <span class="cedit__label">Sign in</span>
        <div v-if="editConnected" class="cedit__oauth">
          <span class="cedit__connected"><Icon name="check-circle" :size="15" /> Connected</span>
          <button class="pk-btn pk-btn--ghost pk-btn--sm" type="button" @click="disconnect">
            Disconnect
          </button>
        </div>
        <template v-else>
          <div class="cedit__oauth">
            <button class="pk-btn pk-btn--sm" type="button" :disabled="oauthBusy" @click="connect">
              {{ oauthBusy ? 'Waiting for the sign-in tab...' : 'Connect...' }}
            </button>
            <input
              v-if="needClientId"
              v-model="oauthClientId"
              class="pk-input cedit__cid"
              placeholder="client id from the provider's app settings"
            />
          </div>
        </template>
      </div>
      <div class="cedit__field">
        <span class="cedit__label">Where it works</span>
        <label class="cedit__sysrow">
          <Switch v-model="editAll" label="Every model on this machine" />
          <span class="cedit__systext">
            <span>Every model on this machine, including ones you start later</span>
          </span>
        </label>
        <template v-if="!editAll">
          <label v-for="m in scopeModels" :key="m.port" class="cedit__sysrow">
            <Switch
              :model-value="editPorts.includes(m.port)"
              :label="m.name"
              @update:model-value="() => togglePort(m.port)"
            />
            <span class="cedit__systext">
              <span>{{ m.name }}</span>
            </span>
          </label>
        </template>
      </div>
      <p v-if="editError" class="cedit__error">{{ editError }}</p>
    </div>
    <template #footer>
      <button class="pk-btn pk-btn--ghost" @click="editorOpen = false">Cancel</button>
      <button class="pk-btn pk-btn--primary" :disabled="!canSave || checking" @click="saveEdit">
        <Icon v-if="checking" name="spinner" :size="13" class="conn__spin" />
        {{ checking ? 'Checking...' : forceSave ? 'Save anyway' : 'Save' }}
      </button>
    </template>
  </Dialog>

  <Dialog
    :open="!!pendingDelete"
    role="alertdialog"
    danger
    icon="alert-triangle"
    title="Remove connector?"
    size="sm"
    @close="pendingDelete = null"
  >
    <p class="cedit__confirm">
      <strong>{{ pendingDelete?.label }}</strong> will be removed from your library and from any
      chat where it is switched on.
    </p>
    <template #footer>
      <button class="pk-btn pk-btn--ghost" @click="pendingDelete = null">Cancel</button>
      <button class="pk-btn pk-btn--danger" @click="confirmDelete">Remove</button>
    </template>
  </Dialog>
</template>

<style scoped>
/* fills the view like the Cloud page: the card takes the remaining height,
   the TABLE scrolls inside it - the page itself never scrolls past */
.conn {
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
  align-self: stretch;
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.conn__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
}
.conn__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
}
.conn__sub {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
  margin-top: 4px;
  max-width: 52ch;
}
.conn__head .pk-btn {
  flex-shrink: 0;
}
/* one surface card (the Cloud page's cl__card recipe): inputs sit on
   bg-surface, never bare on the content background */
.conn__card {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  overflow: hidden;
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.conn__tabs {
  display: flex;
  align-items: center;
  padding: 0 12px;
  background: var(--pk-bg-elevated);
  border-bottom: 1px solid var(--pk-border-default);
}
.conn__tabs :deep(.pk-tabs) {
  border-bottom: none;
}
/* One scroller: the tablewrap. The body itself never scrolls in the finder
   tab (nested scrollbars under a sticky header read as broken); the installed
   tab scrolls its list normally. */
.conn__body {
  padding: 16px 20px 18px;
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.conn__body--scroll {
  overflow: auto;
}
.conn__search {
  width: 100%;
  max-width: none;
  height: 52px;
  margin-bottom: 12px;
  padding: 0 18px;
  font-size: var(--pk-font-size-lg);
  font-weight: 600;
  border-radius: var(--pk-radius-lg);
  flex-shrink: 0;
}
.conn__search::placeholder {
  font-weight: 500;
}
.conn__tools {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-bottom: 10px;
  flex-shrink: 0;
}
.conn__chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 11px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-full);
  background: transparent;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  cursor: pointer;
  transition: border-color 0.12s, color 0.12s, background 0.12s;
}
.conn__chip:hover {
  border-color: var(--pk-border-strong);
  color: var(--pk-text-primary);
}
.conn__chip--on {
  background: var(--pk-accent-subtle);
  border-color: var(--pk-accent);
  color: var(--pk-accent);
}
.conn__tools-sep {
  width: 1px;
  height: 16px;
  background: var(--pk-border-default);
  margin: 0 4px;
}
.conn__hint {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  padding: 2px 2px 8px;
}
.conn__spin {
  animation: conn-spin 0.9s linear infinite;
}
@keyframes conn-spin {
  to {
    transform: rotate(360deg);
  }
}
.conn__error {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-danger);
  padding: 2px 2px 8px;
}
/* the table takes the remaining card height and scrolls inside (mb--fill) */
.conn__tablewrap {
  flex: 1;
  min-height: 0;
  overflow: auto;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-base);
}
.conn__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-sm);
}
.conn__table thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--pk-bg-elevated);
  text-align: left;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-secondary);
  padding: 8px 12px;
  border-bottom: 1px solid var(--pk-border-default);
  white-space: nowrap;
}
.conn__table thead th.th-sort {
  cursor: pointer;
  user-select: none;
}
.conn__table thead th.th-sort:hover {
  color: var(--pk-text-primary);
}
.conn__table tbody td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--pk-border-default);
  vertical-align: middle;
}
.conn__table tbody tr:last-child td {
  border-bottom: none;
}
.conn__table tbody tr:hover {
  background: var(--pk-bg-hover);
}
.conn__row-dead {
  opacity: 0.55;
}
.conn__row-x {
  cursor: pointer;
}
.conn__td-name {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.conn__caret {
  flex-shrink: 0;
  color: var(--pk-text-muted);
}
.conn__th-r,
.conn__td-r {
  text-align: right;
}
.conn__td-c {
  text-align: center;
}
.conn__th-act {
  width: 84px;
}
.conn__td-act {
  text-align: right;
  white-space: nowrap;
}
.conn__name {
  font-weight: 600;
  color: var(--pk-text-primary);
}
.conn__td-domain {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 210px;
}
.conn__badge {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  padding: 1px 8px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-elevated);
  border: 1px solid var(--pk-border-default);
  color: var(--pk-text-secondary);
  white-space: nowrap;
}
.conn__num {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
}
/* the status is one icon; the honest wording lives in its tooltip */
.conn__live {
  display: inline-flex;
  align-items: center;
  color: var(--pk-text-muted);
}
.conn__live--ok {
  color: var(--pk-success, #2e7d32);
}
.conn__live--auth {
  color: var(--pk-warning, #b78103);
}
.conn__live--dead {
  color: var(--pk-danger);
}
/* fold-out detail (the ModelBrowser expandable idiom) */
.conn__detail > td {
  background: var(--pk-bg-elevated);
  padding: 12px 16px 14px 30px;
}
.conn__dd {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.conn__dd-desc {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
  max-width: 90ch;
}
.conn__dd-facts {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 28px;
}
.conn__dd-fact {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.conn__dd-fact dt {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.conn__dd-fact dd {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
.conn__dd-fact code {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  word-break: break-all;
}
.conn__dd-tools {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.conn__dd-tools code {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  padding: 2px 7px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-surface);
  border: 1px solid var(--pk-border-default);
  color: var(--pk-text-secondary);
}
.conn__dd-more {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.conn__dd-links {
  display: flex;
  gap: 16px;
}
.conn__dd-links a {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  text-decoration: none;
}
.conn__dd-links a:hover {
  color: var(--pk-text-primary);
  text-decoration: underline;
}
.conn__in {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-accent);
}
.conn__note {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  line-height: 1.45;
  padding: 8px 2px 0;
  flex-shrink: 0;
}
.conn__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  text-align: center;
  flex: 1;
  color: var(--pk-text-muted);
}
.conn__empty-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 56px;
  height: 56px;
  border-radius: var(--pk-radius-xl);
  background: var(--pk-bg-elevated);
  color: var(--pk-text-secondary);
}
.conn__empty-sub {
  font-size: var(--pk-font-size-sm);
  max-width: 46ch;
  line-height: 1.5;
}
.conn__mine-bar {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 6px;
}
.conn__list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.ccard {
  display: flex;
  align-items: center;
  gap: 8px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-base);
  transition: border-color 0.12s ease;
}
.ccard:hover {
  border-color: var(--pk-border-strong);
}
.ccard__main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 12px 14px;
  border: 0;
  background: transparent;
  text-align: left;
  cursor: pointer;
}
.ccard__label {
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.ccard__url {
  font-size: var(--pk-font-size-xs);
  font-family: var(--pk-font-mono);
  color: var(--pk-text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ccard__badge {
  flex-shrink: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-full);
  padding: 2px 8px;
}
.ccard__badge--sys {
  color: var(--pk-accent);
  border-color: var(--pk-accent);
  background: var(--pk-accent-subtle);
  font-weight: 600;
}
.cedit__oauth {
  display: flex;
  align-items: center;
  gap: 10px;
}
.cedit__connected {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-success, #2e7d32);
}
.cedit__cid {
  flex: 1;
  min-width: 0;
  font-family: var(--pk-font-mono);
}
.cedit__sysrow {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  cursor: pointer;
}
/* :deep() - SwitchRoot is rendered through a clone that drops our scope
   attribute, so this nudge has silently never applied (found by an
   out-of-tree tool) */
.cedit__sysrow :deep(.pk-switch) {
  margin-top: 2px;
}
.cedit__systext {
  display: flex;
  flex-direction: column;
  gap: 3px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
}
.ccard__acts {
  display: flex;
  gap: 2px;
  padding-right: 10px;
  opacity: 0;
  transition: opacity 0.12s ease;
}
.ccard:hover .ccard__acts,
.ccard:focus-within .ccard__acts {
  opacity: 1;
}
.ccard__del:hover {
  color: var(--pk-danger);
}
.cedit {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.cedit__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.cedit__label {
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.cedit__hint {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.cedit__auth {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  padding: 8px 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-elevated);
}
.cedit__hrow {
  display: flex;
  gap: 6px;
  align-items: center;
}
.cedit__hkey {
  width: 180px;
  flex-shrink: 0;
}
.cedit__hval {
  flex: 1;
  min-width: 0;
  font-family: var(--pk-font-mono);
}
.cedit__hadd {
  align-self: flex-start;
}
.cedit__error {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-danger);
}
.cedit__confirm {
  font-size: var(--pk-font-size-sm);
  line-height: 1.5;
  color: var(--pk-text-secondary);
}
</style>
