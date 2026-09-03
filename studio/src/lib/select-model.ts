// The one way the Studio gets pointed at a model - the header dropdown and
// the Manager's "Open in Studio" both land here. Flipping the model is not
// comparing: it also retargets the active chat (or draft) and disarms any
// compare set, so the next send goes to exactly the model the user picked.

import { useChatStore } from '@/stores/chat'
import { useModelsStore } from '@/stores/models'

/** The model the next send will use, and therefore the only thing any "current
 *  model" display may show.
 *
 *  Two values are in play and they are not the same: `models.currentId` is the
 *  fleet-wide SEAT (what a new chat would start on), while a conversation
 *  carries its own `model` from the moment it exists. The send path has always
 *  read the conversation first - so anything reading the seat alone will lie
 *  the moment they diverge, and they diverge routinely:
 *
 *   - the seat auto-moves when a runner comes up (models.ts `refresh`), which
 *     is not a user choice and must not silently retarget a conversation;
 *   - opening an older chat does not move the seat at all.
 *
 *  Found live: with only a cloud model up, the Studio opens
 *  a draft on it; starting a local model moved the seat, the header dropdown
 *  switched to the local model - and the send still went to the cloud one. */
export function effectiveModelId(): string {
  const models = useModelsStore()
  const chat = useChatStore()
  return chat.active?.model || models.currentId || ''
}

export function selectStudioModel(id: string): void {
  if (!id) return
  const models = useModelsStore()
  const chat = useChatStore()
  models.currentId = id
  const c = chat.active
  if (c) {
    c.model = id
    c.compareModels = undefined
    chat.persist(c)
  }
  // the ctx window / reasoning style / vision flags are per-runner - refresh
  void models.fetchLimits()
}
