// Is there a newer paddock, and is one being fetched.
//
// The manager does the asking and caches the answer for an hour, so this store
// can poll /api/updates freely - a render never becomes an outbound request to
// the release server.
//
// The ENDPOINT never fails, deliberately. "Could not reach the release server"
// comes back as state 'unknown' with a 200, because a Manager that shows an
// error because a laptop is on a train is worse than one that quietly says it
// does not know. So there is no error branch here to speak of: `unknown` is the
// offline case, and the UI shows nothing for it.

import { defineStore } from 'pinia'
import { ref } from 'vue'

export type UpdatePhase = 'idle' | 'running' | 'ready' | 'failed'

export interface UpdateDownload {
  version: string
  phase: UpdatePhase
  received: number
  /** 0 when the published row predates the API's fileSize column - the UI then
   *  shows bytes rather than inventing a percentage. */
  total: number
  path: string | null
  error: string | null
}

export interface UpdateState {
  state: 'current' | 'available' | 'unknown'
  /** What we are running. Present in every state. */
  current?: string
  version?: string
  latest?: string
  notes?: string | null
  publishedAt?: string | null
  size?: number | null
  downloadable?: boolean
  /** False when the release carries no sha256, so the download cannot be
   *  checked beyond TLS. Surfaced rather than hidden - see the card. */
  verifiable?: boolean
  why?: string
  download: UpdateDownload | null
}

export const useUpdatesStore = defineStore('updates', () => {
  const info = ref<UpdateState | null>(null)
  const busy = ref(false)

  /** Server-pushed update state (SSE 'update') - the same body /api/updates
   *  answers, so consumers cannot tell push from poll. */
  function apply(state: UpdateState): void {
    info.value = state
  }

  async function refresh(): Promise<void> {
    try {
      const r = await fetch('/api/updates')
      if (r.ok) info.value = (await r.json()) as UpdateState
    } catch {
      /* the manager is momentarily away; the next poll asks again */
    }
  }

  /** Start the download. The manager refuses cleanly if there is nothing to
   *  fetch, so a failure here is worth showing. */
  async function download(): Promise<string | null> {
    busy.value = true
    try {
      const r = await fetch('/api/updates/download', { method: 'POST' })
      const body = await r.json().catch(() => null)
      await refresh()
      if (!r.ok) return (body?.error as string) ?? `HTTP ${r.status}`
      return null
    } catch (e) {
      return e instanceof Error ? e.message : 'could not start the download'
    } finally {
      busy.value = false
    }
  }

  async function cancel(): Promise<void> {
    try {
      await fetch('/api/updates/download/cancel', { method: 'POST' })
    } finally {
      await refresh()
    }
  }

  return {
    apply, info, busy, refresh, download, cancel }
})
