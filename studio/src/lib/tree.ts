// The conversation as a TREE.
//
// A chat is not a list. Editing a question, regenerating an answer or trying
// the same prompt on another model all produce an ALTERNATIVE to something
// that already exists, and a list can only represent that by destroying what
// was there. So every message carries `parentId`, the conversation carries
// `leafId` (the tip of the branch currently on screen), and what you see is
// the path from a root down to that leaf.
//
// Three decisions worth knowing before reading the code:
//
// 1. The ARRAY STAYS. `Conversation.messages` remains a flat array of every
//    node in creation order - it is not replaced by an id map. The manager
//    stores a conversation as one opaque JSON doc and classifies it by
//    iterating `doc.messages` (store.rs `conversation_kind`), so keeping the
//    array keeps the server, its migrations and every existing reader working
//    untouched. Adjacency rides on the nodes instead of replacing them.
//
// 2. The UNIT is A STEP, not A MESSAGE. A compare fan-out is N assistant
//    turns answering one question, and they belong on screen together. So the
//    path is a list of steps, where a step is either one message or a whole
//    compare group, and the tree links hang off the group's ANCHOR (its first
//    lane). This is what lets branching and compare coexist instead of
//    fighting: regenerating a compare block makes a second group under the
//    same question, and the sibling switch flips between whole runs.
//
// 3. BRANCH MEMORY is PART of the MODEL. `branchMemory` records, for each
//    branch point, which child step you last took. Coming back to a branch
//    restores the exact path you were on inside it, at every depth, rather
//    than dumping you on its newest or first child. LibreChat tracks a
//    sibling index per level (which loses depth) and open-webui walks to the
//    last child (which loses your position); remembering the CHOICE at each
//    branch point reconstructs the whole path for free and is bounded by the
//    number of branch points rather than the number of messages.
//
// Everything here is a pure function over the conversation document, with no
// Vue and no store, so it is directly testable - see tree.test.ts.

import type { Conversation, Message } from '@/types/chat'

/** Key used for "has no parent". Real ids are uuids, so '' cannot collide. */
export const ROOT = ''

/** One rendered step of a conversation: a single turn, or a compare fan-out
 *  whose lanes render side by side. */
export type Step = { kind: 'msg'; m: Message } | { kind: 'group'; ms: Message[] }

/** The message a step hangs from: the turn itself, or a group's first lane.
 *  Tree links always point at anchors, so this is the id children use. */
export function stepAnchor(s: Step): Message {
  return s.kind === 'msg' ? s.m : s.ms[0]
}

export function stepId(s: Step): string {
  return stepAnchor(s).id
}

/** Every message in a step, in render order. */
export function stepMessages(s: Step): Message[] {
  return s.kind === 'msg' ? [s.m] : s.ms
}

/** parent id -> its children, in array (creation) order. Roots land under ROOT. */
export function childIndex(msgs: Message[]): Map<string, Message[]> {
  const idx = new Map<string, Message[]>()
  for (const m of msgs) {
    const key = m.parentId ?? ROOT
    const at = idx.get(key)
    if (at) at.push(m)
    else idx.set(key, [m])
  }
  return idx
}

function byIdMap(msgs: Message[]): Map<string, Message> {
  const m = new Map<string, Message>()
  for (const x of msgs) m.set(x.id, x)
  return m
}

/** Bucket one parent's children into steps: each compare group collapses into
 *  a single step (in the order its first lane appears), everything else is its
 *  own step. This is the list a sibling switch walks. */
export function stepsOf(children: Message[]): Step[] {
  const out: Step[] = []
  const groupAt = new Map<string, number>()
  for (const c of children) {
    if (c.role === 'assistant' && c.group) {
      const at = groupAt.get(c.group)
      if (at !== undefined) {
        const s = out[at]
        if (s.kind === 'group') s.ms.push(c)
        continue
      }
      groupAt.set(c.group, out.length)
      out.push({ kind: 'group', ms: [c] })
    } else {
      out.push({ kind: 'msg', m: c })
    }
  }
  return out
}

/** The steps on screen, root first. Walks up from the leaf (one parent each,
 *  so the path is unambiguous) then expands each anchor to its full step.
 *
 *  The `seen` guard is not paranoia about our own writes - a conversation doc
 *  is user-editable JSON on disk, and a cycle there would otherwise hang the
 *  renderer rather than degrade. */
export function activeSteps(conv: Conversation): Step[] {
  const msgs = conv.messages
  if (!msgs.length) return []
  const byId = byIdMap(msgs)
  const idx = childIndex(msgs)

  const chain: Message[] = []
  const seen = new Set<string>()
  let cur = conv.leafId ? byId.get(conv.leafId) : undefined
  while (cur && !seen.has(cur.id)) {
    seen.add(cur.id)
    chain.push(cur)
    cur = cur.parentId ? byId.get(cur.parentId) : undefined
  }
  chain.reverse()

  return chain.map((a) => {
    if (a.role !== 'assistant' || !a.group) return { kind: 'msg', m: a } as Step
    const sibs = (idx.get(a.parentId ?? ROOT) ?? []).filter((m) => m.group === a.group)
    return { kind: 'group', ms: sibs } as Step
  })
}

/** Every message on screen, compare lanes included, in render order. This is
 *  what the send path reads - Not `conv.messages`, which also holds every
 *  branch you are not looking at. */
export function activeMessages(conv: Conversation): Message[] {
  const out: Message[] = []
  for (const s of activeSteps(conv)) out.push(...stepMessages(s))
  return out
}

/** The id a new step should hang from: the anchor of the last step on screen,
 *  or undefined in an empty conversation (the new step becomes a root). */
export function tipId(conv: Conversation): string | undefined {
  const steps = activeSteps(conv)
  return steps.length ? stepId(steps[steps.length - 1]) : undefined
}

/** The step-siblings of whatever step `id` belongs to, plus which one it is.
 *  `steps.length > 1` is exactly the condition for showing a switcher. */
export function siblingInfo(
  conv: Conversation,
  id: string,
): { steps: Step[]; index: number } | null {
  const byId = byIdMap(conv.messages)
  const m = byId.get(id)
  if (!m) return null
  const kids = childIndex(conv.messages).get(m.parentId ?? ROOT) ?? []
  const steps = stepsOf(kids)
  const index = steps.findIndex((s) => stepMessages(s).some((x) => x.id === id))
  return index < 0 ? null : { steps, index }
}

/** Walk down from an anchor to a leaf, taking the remembered turn at every
 *  branch point and the NEWEST step where nothing is remembered.
 *
 *  Newest rather than first is deliberate: an unremembered branch point is one
 *  you have never chosen at, and the last child is the one just generated -
 *  which is where a fresh regenerate should leave you. */
export function descendLeaf(conv: Conversation, anchorId: string): string {
  const idx = childIndex(conv.messages)
  const mem = conv.branchMemory ?? {}
  const seen = new Set<string>()
  let cur = anchorId
  for (;;) {
    if (seen.has(cur)) return cur
    seen.add(cur)
    const kids = idx.get(cur)
    if (!kids || !kids.length) return cur
    const steps = stepsOf(kids)
    const want = mem[cur]
    const next = steps.find((s) => stepId(s) === want) ?? steps[steps.length - 1]
    cur = stepId(next)
  }
}

/** Record the path currently on screen as the remembered choice at every
 *  branch point it passes through, so leaving and returning is lossless. */
export function rememberPath(conv: Conversation): void {
  const steps = activeSteps(conv)
  if (!steps.length) return
  const mem = conv.branchMemory ?? (conv.branchMemory = {})
  for (const s of steps) {
    const a = stepAnchor(s)
    mem[a.parentId ?? ROOT] = a.id
  }
}

/** Put `anchorId`'s step on screen and descend to the branch tip under it.
 *  Remembers the outgoing path first - that is what makes flipping back and
 *  forth between two branches return you to the same place each time. */
export function focusStep(conv: Conversation, anchorId: string): void {
  const byId = byIdMap(conv.messages)
  const a = byId.get(anchorId)
  if (!a) return
  rememberPath(conv)
  const mem = conv.branchMemory ?? (conv.branchMemory = {})
  mem[a.parentId ?? ROOT] = anchorId
  conv.leafId = descendLeaf(conv, anchorId)
}

/** Move to the sibling `delta` steps away from the step holding `id`.
 *  Returns false when there is nowhere to go (no siblings, or already at an
 *  end) so a caller can leave the control disabled rather than no-op silently. */
export function stepSibling(conv: Conversation, id: string, delta: number): boolean {
  const info = siblingInfo(conv, id)
  if (!info || info.steps.length < 2) return false
  const next = info.index + delta
  if (next < 0 || next >= info.steps.length) return false
  focusStep(conv, stepId(info.steps[next]))
  return true
}

/** Delete a step and everything under it. The cursor lands on the parent's
 *  newest surviving branch, or on the parent itself when that was the only
 *  child - never nowhere. Returns the ids removed. */
export function deleteSubtree(conv: Conversation, id: string): string[] {
  const byId = byIdMap(conv.messages)
  const target = byId.get(id)
  if (!target) return []

  // The whole STEP goes, not one lane of it: half a compare block is not a
  // thing the rest of the model (or the reader) can make sense of.
  const info = siblingInfo(conv, id)
  const step = info?.steps[info.index]
  const roots = step ? stepMessages(step).map((m) => m.id) : [id]

  const idx = childIndex(conv.messages)
  const doomed = new Set<string>()
  const stack = [...roots]
  while (stack.length) {
    const cur = stack.pop() as string
    if (doomed.has(cur)) continue
    doomed.add(cur)
    for (const k of idx.get(cur) ?? []) stack.push(k.id)
  }

  const parentId = target.parentId ?? null
  conv.messages = conv.messages.filter((m) => !doomed.has(m.id))

  // Memory entries pointing into the removed subtree would send a later
  // descend somewhere that no longer exists.
  if (conv.branchMemory) {
    const alive = new Set(conv.messages.map((m) => m.id))
    for (const [k, v] of Object.entries(conv.branchMemory)) {
      if ((k !== ROOT && !alive.has(k)) || !alive.has(v)) delete conv.branchMemory[k]
    }
  }

  const survivors = childIndex(conv.messages).get(parentId ?? ROOT) ?? []
  if (survivors.length) {
    const steps = stepsOf(survivors)
    conv.leafId = descendLeaf(conv, stepId(steps[steps.length - 1]))
  } else if (parentId && conv.messages.some((m) => m.id === parentId)) {
    conv.leafId = parentId
  } else {
    const last = conv.messages[conv.messages.length - 1]
    conv.leafId = last ? last.id : undefined
  }
  return [...doomed]
}

/** Give a conversation tree links if it does not have them, and repair them if
 *  they are damaged. Idempotent: a second call changes nothing.
 *
 *  Every conversation written before the tree is a straight line, so the array
 *  order is the path and inference is exact - a message hangs from the step
 *  before it, and a compare run (consecutive assistant turns sharing a group)
 *  is one step whose lanes all hang from the same place.
 *
 *  The same loop self-heals a partially written or hand-edited doc: a message
 *  is relinked when it has no parent OR when its parent has vanished, which
 *  is what keeps an orphan visible in the thread instead of silently dropping
 *  out of every path. Returns true when anything changed, so the caller can
 *  decide whether a save is owed. */
export function migrate(conv: Conversation): boolean {
  const msgs = conv.messages
  if (!msgs.length) {
    if (conv.leafId !== undefined) {
      conv.leafId = undefined
      return true
    }
    return false
  }
  const present = new Set(msgs.map((m) => m.id))
  let changed = false
  let prevAnchor: string | undefined

  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i]
    // A compare run in array order is one step; every lane shares the parent
    // and the first lane becomes the anchor the next step hangs from.
    const members = [m]
    if (m.role === 'assistant' && m.group) {
      while (
        i + 1 < msgs.length &&
        msgs[i + 1].role === 'assistant' &&
        msgs[i + 1].group === m.group
      ) {
        members.push(msgs[++i])
      }
    }
    for (const x of members) {
      const broken = x.parentId != null && !present.has(x.parentId)
      if (x.parentId === undefined || broken) {
        x.parentId = prevAnchor ?? null
        changed = true
      }
    }
    prevAnchor = members[0].id
  }

  // A cursor that resolves to nothing renders an empty thread over a full
  // conversation - the one failure here that would look like data loss.
  if (!conv.leafId || !present.has(conv.leafId)) {
    conv.leafId = prevAnchor
    changed = true
  }
  if (conv.branchMemory) {
    for (const [k, v] of Object.entries(conv.branchMemory)) {
      if ((k !== ROOT && !present.has(k)) || !present.has(v)) {
        delete conv.branchMemory[k]
        changed = true
      }
    }
  }
  return changed
}
