// The personal MCP connector library - hosted MCP servers the user tries per
// chat. Lives in the manager's SQLite (survives browsers and machines);
// distinct from a served endpoint's own tools, which are endpoint contract in
// servers/<port>.toml. A selected connector rides per REQUEST as the OpenAI
// inline `mcp` tool (server_url + headers) - the runner dials it, the browser
// never does.
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useModelsStore } from '@/stores/models'

export interface Connector {
  id: string
  label: string
  url: string
  headers: Record<string, string>
  registryKey: string
  /** Scope: `system` = every model incl. future ones; `ports` = exactly those
   *  endpoints; both empty = per-chat only. Materialized into the servers'
   *  TOMLs by the manager; runners re-read their tool registry live, so scope
   *  changes apply on the next request - no model restart. */
  system: boolean
  ports: number[]
  /** Signed in via OAuth: the manager holds the tokens, refreshes them, and
   *  merges the bearer into `headers` on the way out. */
  connected?: boolean
  createdAt: number
}

export const useConnectorsStore = defineStore('connectors', () => {
  const list = ref<Connector[]>([])
  const loaded = ref(false)

  async function refresh(): Promise<void> {
    try {
      const res = await fetch('/api/connectors')
      if (res.ok) list.value = (await res.json()) as Connector[]
    } catch (e) {
      console.error('failed to load connectors', e)
    }
    loaded.value = true
  }
  async function ensure(): Promise<void> {
    if (!loaded.value) await refresh()
  }

  /** Throws Error(message) on a 400 so forms can show the server's words. */
  async function save(
    doc: { label: string; url: string; headers: Record<string, string>; registryKey?: string },
    id?: string,
  ): Promise<void> {
    const res = await fetch(id ? `/api/connectors/${id}` : '/api/connectors', {
      method: id ? 'PUT' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(doc),
    })
    if (!res.ok) {
      const body = (await res.json().catch(() => null)) as {
        error?: { message?: string }
      } | null
      throw new Error(body?.error?.message ?? `save failed (${res.status})`)
    }
    await refresh()
    // library mutations rematerialize endpoint configs, and runners re-read
    // them live - what the models advertise just changed, so the cached
    // /api/server answers are stale (a connector scoped to every model was
    // invisible in the tool picker until a hard refresh)
    useModelsStore().invalidateCaps()
  }

  async function remove(id: string): Promise<void> {
    await fetch(`/api/connectors/${id}`, { method: 'DELETE' })
    await refresh()
    useModelsStore().invalidateCaps()
  }

  async function setScope(id: string, all: boolean, ports: number[]): Promise<void> {
    const res = await fetch(`/api/connectors/${id}/scope`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ all, ports }),
    })
    if (!res.ok) {
      const body = (await res.json().catch(() => null)) as {
        error?: { message?: string }
      } | null
      throw new Error(body?.error?.message ?? `change failed (${res.status})`)
    }
    await refresh()
    useModelsStore().invalidateCaps()
  }

  function byId(id: string): Connector | undefined {
    return list.value.find((c) => c.id === id)
  }

  /** Begin the OAuth flow; returns the authorize URL to open in a new tab.
   *  Throws with the manager's words (e.g. "no dynamic registration - enter
   *  a client id"). */
  async function oauthStart(id: string, clientId?: string): Promise<string> {
    const res = await fetch(`/api/connectors/${id}/oauth/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(clientId ? { client_id: clientId } : {}),
    })
    const body = (await res.json().catch(() => null)) as {
      url?: string
      error?: { message?: string }
    } | null
    if (!res.ok || !body?.url) {
      throw new Error(body?.error?.message ?? `sign-in could not start (${res.status})`)
    }
    return body.url
  }

  /** Handshake probe before saving: does this URL answer an MCP initialize? */
  async function check(
    url: string,
    headers: Record<string, string>,
  ): Promise<{ ok: boolean; authRequired?: boolean; server?: string; error?: string }> {
    const res = await fetch('/api/connectors/check', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, headers }),
    })
    const body = (await res.json().catch(() => null)) as {
      ok?: boolean
      auth_required?: boolean
      server?: string
      error?: { message?: string } | string
    } | null
    if (!res.ok) {
      const msg = typeof body?.error === 'object' ? body.error?.message : body?.error
      return { ok: false, error: msg ?? `check failed (${res.status})` }
    }
    return {
      ok: body?.ok === true,
      authRequired: body?.auth_required === true,
      server: body?.server,
      error: typeof body?.error === 'string' ? body.error : undefined,
    }
  }

  async function oauthDisconnect(id: string): Promise<void> {
    await fetch(`/api/connectors/${id}/oauth/disconnect`, { method: 'POST' })
    await refresh()
  }

  return { list, loaded, refresh, ensure, save, remove, setScope, check, oauthStart, oauthDisconnect, byId }
})
