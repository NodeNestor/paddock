// One rule for what a model is CALLED in the UI: the catalog's human display
// name ("Qwen 3.5 9B") wherever the registry knows the model, a cleaned-up id
// otherwise. The technical id/path never headlines - it rides tooltips.

import { cloudVendor, useModelsStore } from '@/stores/models'
import { useRegistryStore } from '@/stores/registry'
import { friendlyModelName } from '@/lib/model-caps'

const baseName = (p: string): string => p.split(/[\\/]/).pop() ?? p

/** Match a name against a catalog file's dest by file name OR stem - a
 *  runner's file-derived model id drops the `.gguf` ("Qwen3.5-9B-Q8_0"). */
function matchesDest(dest: string, base: string): boolean {
  const d = baseName(dest).toLowerCase()
  return d === base || d.replace(/\.gguf$/, '') === base
}

function catalogHit(idOrPath: string) {
  const reg = useRegistryStore()
  const byId = reg.models.find((m) => m.id === idOrPath)
  if (byId) return byId
  const base = baseName(idOrPath).toLowerCase()
  return reg.models.find((m) =>
    m.artifacts.some((a) => a.files.some((f) => matchesDest(f.dest, base))),
  )
}

/** Human label for a model id, name, or weights path. */
export function modelLabel(idOrPath: string | null | undefined): string {
  if (!idOrPath) return ''
  const hit = catalogHit(idOrPath)
  if (hit) return hit.display
  return friendlyModelName(baseName(idOrPath).replace(/\.gguf$/i, ''))
}

/** The maker ("Alibaba", "OpenAI") for the vendor logo, when known. */
export function modelVendor(idOrPath: string | null | undefined): string | undefined {
  if (!idOrPath) return undefined
  return catalogHit(idOrPath)?.vendor
}

/** What the catalog says this model is FOR: "chat", "documents",
 *  "transcription", "alignment", "embeddings", "rerank", ... Empty when the
 *  catalog does not know it - a hand-started GGUF, or a cloud pick.
 *
 *  Durable in a way the runtime caps are not: caps come off a RUNNING
 *  endpoint, so anything asking "what kind of thing is this" about a stopped
 *  model has to come here instead. Same reason the vendor mark does. */
export function modelCapability(idOrPath: string | null | undefined): string[] {
  if (!idOrPath) return []
  return catalogHit(idOrPath)?.capability ?? []
}

/** A cloud pick's id is `cloud:<endpoint>:<model>[@provider]`. Only the model
 *  part means anything once the endpoint is out of the picture - nobody wants
 *  to read a UUID. */
function bareId(id: string): string {
  return /^cloud:[^:]+:(.+)$/.exec(id)?.[1]?.split('@')[0] ?? id
}

/** How a model is named and marked wherever a reply is ATTRIBUTED to it - the
 *  chat's lane headers and the artifact panel's header badge. Both must agree:
 *  in a compare the pane badge sits directly under the lane badge.
 *
 *  The running fleet answers first, because it is the only thing that knows a
 *  cloud pick. But attribution has to outlive the run: an artifact written
 *  yesterday still says who wrote it after that model is stopped, so a miss
 *  falls through to the catalog and only then to a cleaned-up id. Asking the
 *  fleet alone is what made a stopped Qwen lose its mark and its name in both
 * places at once. */
export function fleetLabel(id: string | null | undefined): string {
  if (!id) return ''
  const hit = useModelsStore().models.find((m) => m.id === id)
  if (hit?.display) return hit.display
  return modelLabel(bareId(id))
}

export function fleetVendor(id: string | null | undefined): string | undefined {
  if (!id) return undefined
  const hit = useModelsStore().models.find((m) => m.id === id)
  if (hit?.vendor) return hit.vendor
  const bare = bareId(id)
  return modelVendor(bare) ?? cloudVendor(bare)
}
