// What the RUNNER adds to the system prompt on this chat's behalf.
//
// The runner honors each connected MCP server's own
// `instructions` (the MCP spec means them as system-prompt material) and adds
// a note when progressive tool disclosure is on. That is real steering the
// user never typed, so an empty prompt box would otherwise read as "nothing is
// being said for me" - untrue, and exactly the kind of silent behaviour the
// product principles forbid. This composable resolves the same text the runner
// will send, so the panel can show it read-only before the turn goes out.
//
// Source of truth is the manager's own tool probe (`/api/mcp/tools` returns
// each server's `instructions` alongside its tools), so this cannot drift from
// what the runner actually folds in.
import { computed, ref, watch, type ComputedRef, type Ref } from 'vue'
import type { Conversation } from '@/types/chat'
import { ARTIFACTS_LABEL, toolSelection } from '@/composables/useChatStream'
import { useConnectorsStore } from '@/stores/connectors'
import { builtinKey, connectorKey, useMcpToolsStore } from '@/stores/mcpTools'

export interface InjectedBlock {
  /** Which server contributed it. */
  label: string
  text: string
}

export function useInjectedPrompt(conv: Ref<Conversation> | ComputedRef<Conversation>): {
  blocks: ComputedRef<InjectedBlock[]>
} {
  const mcpTools = useMcpToolsStore()
  const connectors = useConnectorsStore()

  // The progressive-disclosure rule lives in the runner; the manager publishes
  // its threshold and the exact note so this panel never keeps a second copy.
  const searchThreshold = ref<number | null>(null)
  const searchHidden = ref('')
  const searchAvailable = ref('')
  const searchPartial = ref('')
  void fetch('/api/server')
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      if (!d?.tool_search) return
      searchThreshold.value = d.tool_search.threshold ?? null
      searchHidden.value = d.tool_search.hidden ?? ''
      searchAvailable.value = d.tool_search.available ?? ''
      searchPartial.value = d.tool_search.partial ?? ''
    })
    .catch(() => {})

  /** The servers armed for this chat, as (label, cache key) pairs. */
  const armed = computed<Array<{ label: string; key: string }>>(() => {
    const sel = toolSelection(conv.value)
    const out = [{ label: ARTIFACTS_LABEL, key: builtinKey(ARTIFACTS_LABEL) }]
    const ids = conv.value.connectorIds ?? []
    for (const c of connectors.list) {
      if (!ids.includes(c.id)) continue
      out.push({ label: c.label, key: connectorKey(c.id) })
    }
    if (sel.mode === 'all') return out
    const picked = new Set(sel.picks.map((p) => p.label))
    return out.filter((g) => picked.has(g.label))
  })

  // Probe anything armed but not yet listed, so opening the panel without
  // having opened the tool picker still shows the truth.
  watch(
    armed,
    (groups) => {
      for (const g of groups) {
        if (g.label === ARTIFACTS_LABEL) mcpTools.ensureBuiltin(ARTIFACTS_LABEL)
        else {
          const c = connectors.list.find((x) => x.label === g.label)
          if (c) mcpTools.ensureConnector(c.id)
        }
      }
    },
    { immediate: true },
  )

  const blocks = computed<InjectedBlock[]>(() => {
    const out = armed.value
      .map((g) => ({ label: g.label, text: mcpTools.get(g.key)?.instructions ?? '' }))
      .filter((b) => !!b.text.trim())
    // The search pair is declared in every mode, so a mention always rides too;
    // only which one depends on which servers kept their schemas.
    // Deliberately no tool count in the label: it read as "only 5 tools are on"
    // when the point is the opposite - these are the ones listed, and the rest
    // stay reachable through search.
    //
    // The split mirrors tool_search::disclose_servers - smallest server first,
    // admit while the total fits the threshold. Count only: the runner also
    // weighs schema bytes against the model's context, which this panel cannot
    // know, so a very small context can hide more than shown here.
    const groups = armed.value.map((g) => ({
      label: g.label,
      n: mcpTools.get(g.key)?.tools.length ?? 0,
    }))
    const limit = searchThreshold.value
    if (groups.some((g) => g.n > 0) && limit != null) {
      const shown = new Set<string>()
      let used = 0
      for (const g of [...groups].sort((a, b) => a.n - b.n || a.label.localeCompare(b.label))) {
        if (used + g.n > limit) continue
        used += g.n
        shown.add(g.label)
      }
      const gone = groups.filter((g) => !shown.has(g.label))
      const note =
        used === 0
          ? searchHidden.value
          : gone.length === 0
            ? searchAvailable.value
            : searchPartial.value
                .replace('{tools}', String(gone.reduce((n, g) => n + g.n, 0)))
                .replace('{servers}', gone.map((g) => g.label).join(', '))
      if (note) out.push({ label: 'tool search', text: note })
    }
    return out
  })

  return { blocks }
}
