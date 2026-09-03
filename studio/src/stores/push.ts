// Server-push consumer (`/api/events`, SSE): the manager tells this tab when
// fleet or update state CHANGES, and the poll timers collapse to slow
// reconciles while the stream is live. Polling stays fully intact as the
// FALLBACK - an older manager 404s the endpoint, EventSource closes, `live`
// stays false, and every store keeps its original cadence. Degrade, never
// blank.
//
// SSE and not a second WebSocket deliberately: one-directional state push,
// native auto-reconnect, cookie auth for keyed managers with zero upgrade
// ceremony. (The graph bridge stays WS - it is bidirectional RPC.)
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useModelsStore } from '@/stores/models'
import { useFleetStore } from '@/stores/fleet'
import { useUpdatesStore } from '@/stores/updates'

export const usePushStore = defineStore('push', () => {
  /** The stream is delivering: polls may relax to slow reconciles. */
  const live = ref(false)
  let es: EventSource | null = null

  function connect(): void {
    if (es) return
    try {
      es = new EventSource('/api/events')
    } catch {
      return // no EventSource (ancient embed) - polling carries everything
    }
    es.addEventListener('fleet', (e) => {
      live.value = true
      try {
        const rows = JSON.parse((e as MessageEvent).data)
        useModelsStore().integrateRunnerRows(rows)
        useFleetStore().applyRunnerRows(rows)
      } catch {
        /* one bad frame never kills the stream */
      }
    })
    es.addEventListener('update', (e) => {
      try {
        useUpdatesStore().apply(JSON.parse((e as MessageEvent).data))
      } catch {
        /* ignore */
      }
    })
    es.onerror = () => {
      // EventSource retries transient drops itself; closed means the endpoint
      // refused (older manager) - give up for this page load, polls carry on
      live.value = false
      if (es && es.readyState === EventSource.CLOSED) {
        es = null
      }
    }
  }

  return { live, connect }
})
