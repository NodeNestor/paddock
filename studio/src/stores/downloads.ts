import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { useToastsStore } from '@/stores/toasts'
import { useModelsStore } from '@/stores/models'
import { useRegistryStore } from '@/stores/registry'

/** One manager pull job as /api/models/pulls reports it. `start` is present
 *  when a "start when the bytes land" plan rides the job - the MANAGER runs
 *  that plan, so it survives this tab closing. */
export interface DownloadJob {
  id: string
  model: string
  /** the catalog's human name ("Qwen 3.5 9B") */
  display: string
  artifacts?: string[] | null
  downloaded: number
  total: number
  created_ms: number
  status: { state: 'running' | 'done' | 'cancelled' | 'error'; message?: string }
  start?: {
    port?: number | null
    action?: 'spawn' | 'switch' | null
    state?: { state: 'queued' | 'starting' | 'ok' | 'error'; message?: string; port?: number } | null
  } | null
}

/** Is this job still doing (or about to do) something? Running download, or a
 *  finished download whose queued start hasn't settled yet. */
export function jobActive(j: DownloadJob): boolean {
  if (j.status.state === 'running') return true
  const s = j.start?.state?.state
  return j.status.state === 'done' && (s === 'queued' || s === 'starting')
}

/**
 * Model downloads, manager-owned: the job list is server state (survives
 * reloads and closed tabs), streamed over SSE while anything is active. The
 * header indicator, the fleet's downloading rows, and cancel/resume all read
 * from here.
 */
export const useDownloadsStore = defineStore('downloads', () => {
  const jobs = ref<DownloadJob[]>([])
  const loaded = ref(false)
  // client-side dismissals of settled jobs (the manager keeps its history)
  const dismissed = ref<Set<string>>(new Set())

  /** Jobs the UI still cares about. Clean successes (download done, start ok
   *  or none queued) leave on their own - the toast and the Models page carry
   *  the news; the chip lingers only for things needing attention (running,
   *  paused, failed). */
  const visible = computed(() =>
    jobs.value.filter((j) => {
      if (dismissed.value.has(j.id)) return false
      const st = j.start?.state?.state
      if (j.status.state === 'done' && (st === undefined || st === 'ok')) return false
      return true
    }),
  )
  const active = computed(() => visible.value.filter(jobActive))
  /** Aggregate progress across active downloads, for the header chip. */
  const aggregate = computed(() => {
    const running = active.value.filter((j) => j.status.state === 'running')
    const total = running.reduce((s, j) => s + j.total, 0)
    const done = running.reduce((s, j) => s + j.downloaded, 0)
    return { count: running.length, done, total }
  })

  // ── outcome watching: a settled job gets exactly one toast ────────────────
  const settled = new Set<string>()
  function settleKey(j: DownloadJob): string | null {
    const startState = j.start?.state?.state
    return j.status.state === 'running' || startState === 'queued' || startState === 'starting'
      ? null
      : `${j.id}:${j.status.state}:${startState ?? ''}`
  }
  function noteOutcomes(next: DownloadJob[]): void {
    if (!loaded.value) {
      // first sight of the manager's job history (page load): anything that
      // settled before this tab existed is old news, not a fresh toast
      for (const j of next) {
        const k = settleKey(j)
        if (k) settled.add(k)
      }
      return
    }
    const toasts = useToastsStore()
    for (const j of next) {
      const startState = j.start?.state?.state
      const key = settleKey(j)
      if (!key || settled.has(key)) continue
      settled.add(key)
      if (j.status.state === 'done') {
        // fresh bytes on disk: install state + fit estimates can move from
        // "not downloaded" to real measurements
        const reg = useRegistryStore()
        void reg.refresh().then(() => reg.estimate())
      }
      if (j.status.state === 'error') {
        toasts.push({
          tone: 'bad',
          title: `${j.display} - download failed`,
          description: (j.status.message ?? '').split('\n')[0],
          duration: 10000,
        })
      } else if (j.status.state === 'done' && startState === 'ok') {
        const port = j.start?.state?.port ?? j.start?.port
        toasts.push({
          tone: 'good',
          title: `${j.display} is running`,
          description: `port ${port}`,
          to: port
            ? { name: 'server-detail', params: { port: String(port) } }
            : undefined,
        })
        void useModelsStore().refresh()
        useModelsStore().invalidateCaps()
        // The SSE removes this job's fleet row INSTANTLY, but the fleet's own
        // poll can lag 3s behind the new live runner - that gap flashed the
        // first-run hero between the two states. Dynamic
        // import: fleet.ts imports this store, a static cycle would bite.
        void import('@/stores/fleet').then((m) => m.useFleetStore().refresh())
      } else if (j.status.state === 'done' && startState === 'error') {
        toasts.push({
          tone: 'bad',
          title: `${j.display} failed to start`,
          description: (j.start?.state?.message ?? '').split('\n')[0],
          duration: 10000,
        })
      }
      // plain done (no queued start) and cancelled settle silently - the
      // popover row says it, a toast would be noise
    }
  }

  // ── transfer rate, client-side: the manager reports BYTES per SSE frame;
  // speed and ETA derive here from successive frames with a light EMA so
  // the number reads steady instead of jittering with every frame ─────────
  const rates = ref(new Map<string, { t: number; bytes: number; bps: number }>())
  function trackRates(next: DownloadJob[]): void {
    const now = performance.now()
    const m = new Map(rates.value)
    for (const j of next) {
      if (j.status.state !== 'running') {
        m.delete(j.id)
        continue
      }
      const prev = m.get(j.id)
      if (!prev) {
        m.set(j.id, { t: now, bytes: j.downloaded, bps: 0 })
        continue
      }
      const dt = (now - prev.t) / 1000
      if (dt < 0.4) continue // sub-frame jitter
      const inst = Math.max(0, (j.downloaded - prev.bytes) / dt)
      m.set(j.id, { t: now, bytes: j.downloaded, bps: prev.bps === 0 ? inst : prev.bps * 0.7 + inst * 0.3 })
    }
    rates.value = m
  }
  /** Live rate + ETA for a running job; null until two frames have landed. */
  function rateOf(id: string): { bps: number; etaS: number | null } | null {
    const r = rates.value.get(id)
    const j = jobs.value.find((x) => x.id === id)
    if (!r || !j || r.bps <= 0) return null
    const left = Math.max(0, j.total - j.downloaded)
    return { bps: r.bps, etaS: j.total ? left / r.bps : null }
  }

  function apply(next: DownloadJob[]): void {
    noteOutcomes(next)
    trackRates(next)
    jobs.value = next
    loaded.value = true
    if (next.some(jobActive)) ensureStream()
    else stopStream()
  }

  async function load(): Promise<void> {
    try {
      const r = await fetch('/api/models/pulls')
      if (!r.ok) return
      apply((await r.json()) as DownloadJob[])
    } catch {
      /* manager unreachable - the header just shows nothing */
    }
  }

  // ── the SSE live channel (manager pushes the jobs array while active) ─────
  let es: EventSource | null = null
  function ensureStream(): void {
    if (es) return
    es = new EventSource('/api/models/pulls/events')
    es.onmessage = (ev) => {
      try {
        apply(JSON.parse(ev.data) as DownloadJob[])
      } catch {
        /* one bad frame - the next is 600ms away */
      }
    }
    es.onerror = () => {
      // the manager went away mid-stream: close and re-probe on a timer
      stopStream()
      setTimeout(() => void load(), 3000)
    }
  }
  function stopStream(): void {
    es?.close()
    es = null
  }

  // ── actions ───────────────────────────────────────────────────────────────
  /** Start a pull; with `thenStart` the MANAGER starts the endpoint when the
   *  bytes land (spec = the user's configured variables, verbatim). */
  async function pull(
    modelId: string,
    artifacts?: string[],
    thenStart?: { spec: unknown; action: 'spawn' | 'switch' },
  ): Promise<string> {
    const body: Record<string, unknown> = { id: modelId }
    if (artifacts?.length) body.artifacts = artifacts
    if (thenStart) {
      body.then_start = thenStart.spec
      body.then_action = thenStart.action
    }
    const r = await fetch('/api/models/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const data = (await r.json().catch(() => null)) as
      | { job?: string; error?: { message?: string } }
      | null
    if (!r.ok || !data?.job) throw new Error(data?.error?.message ?? `HTTP ${r.status}`)
    await load()
    ensureStream()
    return data.job
  }

  /** Stop a running download. Partial bytes stay on disk; Resume continues. */
  async function cancel(id: string): Promise<void> {
    await fetch(`/api/models/pull/${id}/cancel`, { method: 'POST' })
    await load()
  }

  /** Resume a cancelled/failed download (fresh job, same selection; a queued
   *  start carries over). The old job is dismissed in place of the new one. */
  async function resume(id: string): Promise<void> {
    const r = await fetch(`/api/models/pull/${id}/resume`, { method: 'POST' })
    if (r.ok) dismissed.value = new Set([...dismissed.value, id])
    await load()
  }

  function dismiss(id: string): void {
    dismissed.value = new Set([...dismissed.value, id])
  }

  return { jobs, visible, active, aggregate, loaded, load, pull, cancel, resume, dismiss, rateOf }
})
