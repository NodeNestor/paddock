// Tests for the conversation tree (lib/tree.ts).
//
// This structure holds the user's entire chat history and every one of its
// operations is destructive if it is wrong: a bad `migrate` hides turns, a bad
// `deleteSubtree` takes the wrong branch, a bad `activeMessages` sends the
// model a branch the user abandoned. It is pure and cheap to exercise, so it
// gets exercised.

import { describe, expect, it } from 'vitest'
import type { Conversation, Message } from '@/types/chat'
import { DEFAULT_PARAMS } from '@/types/chat'
import {
  activeMessages,
  activeSteps,
  deleteSubtree,
  descendLeaf,
  migrate,
  siblingInfo,
  stepSibling,
  tipId,
} from './tree'

// ── builders ────────────────────────────────────────────────────────────────

function conv(messages: Message[] = [], leafId?: string): Conversation {
  return {
    id: 'c1',
    title: 't',
    messages,
    leafId,
    model: 'm',
    systemPrompt: '',
    params: { ...DEFAULT_PARAMS },
    createdAt: 0,
    updatedAt: 0,
  }
}

let seq = 0
function msg(
  role: 'user' | 'assistant',
  text: string,
  parentId?: string | null,
  extra: Partial<Message> = {},
): Message {
  return {
    id: `${role[0]}${++seq}-${text}`,
    role,
    parentId,
    content: [{ type: 'text', text }],
    createdAt: seq,
    ...extra,
  }
}

/** Text of every turn on screen, in order - the readable form of a path. */
function pathText(c: Conversation): string[] {
  return activeMessages(c).map((m) => (m.content[0] as { text: string }).text)
}

// ── migration ───────────────────────────────────────────────────────────────

describe('migrate', () => {
  it('infers a straight chain from array order and points the cursor at the tail', () => {
    const a = msg('user', 'q1')
    const b = msg('assistant', 'a1')
    const c = msg('user', 'q2')
    const d = msg('assistant', 'a2')
    // a pre-tree document: nothing carries a parent
    for (const m of [a, b, c, d]) delete (m as { parentId?: unknown }).parentId
    const cv = conv([a, b, c, d])

    expect(migrate(cv)).toBe(true)
    expect(a.parentId).toBe(null)
    expect(b.parentId).toBe(a.id)
    expect(c.parentId).toBe(b.id)
    expect(d.parentId).toBe(c.id)
    expect(cv.leafId).toBe(d.id)
    expect(pathText(cv)).toEqual(['q1', 'a1', 'q2', 'a2'])
  })

  it('is idempotent - a second pass changes nothing', () => {
    const a = msg('user', 'q1')
    const b = msg('assistant', 'a1')
    for (const m of [a, b]) delete (m as { parentId?: unknown }).parentId
    const cv = conv([a, b])
    expect(migrate(cv)).toBe(true)
    const snapshot = JSON.stringify(cv)
    expect(migrate(cv)).toBe(false)
    expect(JSON.stringify(cv)).toBe(snapshot)
  })

  it('treats a compare run as ONE step: lanes share a parent, the next turn hangs off lane 1', () => {
    const q = msg('user', 'q')
    const l1 = msg('assistant', 'lane1', undefined, { group: 'g', model: 'A' })
    const l2 = msg('assistant', 'lane2', undefined, { group: 'g', model: 'B' })
    const q2 = msg('user', 'q2')
    for (const m of [q, l1, l2, q2]) delete (m as { parentId?: unknown }).parentId
    const cv = conv([q, l1, l2, q2])
    migrate(cv)

    expect(l1.parentId).toBe(q.id)
    expect(l2.parentId).toBe(q.id)
    // the anchor, not the last lane - otherwise the follow-up would sit under
    // one column of a comparison
    expect(q2.parentId).toBe(l1.id)
    // both lanes are on screen even though only the anchor is on the chain
    expect(pathText(cv)).toEqual(['q', 'lane1', 'lane2', 'q2'])
  })

  it('re-points a cursor that resolves to nothing', () => {
    const a = msg('user', 'q1', null)
    const cv = conv([a], 'ghost-id')
    expect(migrate(cv)).toBe(true)
    expect(cv.leafId).toBe(a.id)
    expect(pathText(cv)).toEqual(['q1'])
  })

  it('re-links an orphan rather than letting it fall out of every path', () => {
    const a = msg('user', 'q1', null)
    const b = msg('assistant', 'a1', 'deleted-parent')
    const cv = conv([a, b], b.id)
    expect(migrate(cv)).toBe(true)
    expect(b.parentId).toBe(a.id)
    expect(pathText(cv)).toEqual(['q1', 'a1'])
  })

  it('clears the cursor of an empty conversation', () => {
    const cv = conv([], 'stale')
    expect(migrate(cv)).toBe(true)
    expect(cv.leafId).toBeUndefined()
    expect(activeSteps(cv)).toEqual([])
  })
})

// ── paths and siblings ──────────────────────────────────────────────────────

describe('the active path', () => {
  it('shows only the branch under the cursor', () => {
    const q = msg('user', 'q', null)
    const a1 = msg('assistant', 'first', q.id)
    const a2 = msg('assistant', 'second', q.id)
    const cv = conv([q, a1, a2], a1.id)

    expect(pathText(cv)).toEqual(['q', 'first'])
    cv.leafId = a2.id
    expect(pathText(cv)).toEqual(['q', 'second'])
  })

  it('counts a compare group as one sibling step, not N', () => {
    const q = msg('user', 'q', null)
    const r1a = msg('assistant', 'run1-A', q.id, { group: 'g1', model: 'A' })
    const r1b = msg('assistant', 'run1-B', q.id, { group: 'g1', model: 'B' })
    const r2a = msg('assistant', 'run2-A', q.id, { group: 'g2', model: 'A' })
    const r2b = msg('assistant', 'run2-B', q.id, { group: 'g2', model: 'B' })
    const cv = conv([q, r1a, r1b, r2a, r2b], r1a.id)

    const info = siblingInfo(cv, r1a.id)
    expect(info?.steps).toHaveLength(2) // two runs, four messages
    expect(info?.index).toBe(0)
    // every lane of the shown run is on screen
    expect(pathText(cv)).toEqual(['q', 'run1-A', 'run1-B'])
  })

  it('tipId is the anchor of the last step, so a follow-up hangs off the group', () => {
    const q = msg('user', 'q', null)
    const l1 = msg('assistant', 'lane1', q.id, { group: 'g', model: 'A' })
    const l2 = msg('assistant', 'lane2', q.id, { group: 'g', model: 'B' })
    const cv = conv([q, l1, l2], l1.id)
    expect(tipId(cv)).toBe(l1.id)
  })

  it('survives a cycle instead of hanging', () => {
    const a = msg('user', 'a', null)
    const b = msg('assistant', 'b', a.id)
    a.parentId = b.id // corrupt: a <-> b
    const cv = conv([a, b], b.id)
    expect(activeSteps(cv).length).toBeLessThanOrEqual(2)
  })
})

// ── switching, and the branch memory that makes it lossless ─────────────────

describe('stepSibling', () => {
  it('moves between alternatives and refuses to walk off either end', () => {
    const q = msg('user', 'q', null)
    const a1 = msg('assistant', 'first', q.id)
    const a2 = msg('assistant', 'second', q.id)
    const cv = conv([q, a1, a2], a1.id)

    expect(stepSibling(cv, a1.id, -1)).toBe(false) // already leftmost
    expect(stepSibling(cv, a1.id, 1)).toBe(true)
    expect(pathText(cv)).toEqual(['q', 'second'])
    expect(stepSibling(cv, a2.id, 1)).toBe(false) // already rightmost
  })

  it('returns false where there is nothing to switch', () => {
    const q = msg('user', 'q', null)
    const cv = conv([q], q.id)
    expect(stepSibling(cv, q.id, 1)).toBe(false)
  })

  it('RESTORES THE WHOLE BRANCH you were on, including a nested choice', () => {
    // The property other implementations get wrong: LibreChat remembers a sibling index
    // per level (losing depth) and open-webui walks to the last child (losing
    // your position). Remembering the CHOICE at each branch point rebuilds the
    // entire path.
    //
    // Branch A must itself FORK for this to test anything: if every branch
    // were a single chain, "descend to the newest child" would reconstruct it
    // without any memory at all. So we sit on A's OLDER sub-branch - the one
    // the newest-child default would never pick.
    const q = msg('user', 'q', null)
    const bA = msg('assistant', 'A', q.id)
    const bB = msg('assistant', 'B', q.id)
    const aOld = msg('user', 'A-old', bA.id)
    const aOldDeep = msg('assistant', 'A-old-deep', aOld.id)
    const aNew = msg('user', 'A-new', bA.id) // newer sibling under A
    const b2 = msg('user', 'B-q2', bB.id)
    const cv = conv([q, bA, bB, aOld, aOldDeep, aNew, b2], aOldDeep.id)

    expect(pathText(cv)).toEqual(['q', 'A', 'A-old', 'A-old-deep'])

    stepSibling(cv, bA.id, 1) // over to B
    expect(pathText(cv)).toEqual(['q', 'B', 'B-q2'])

    // Back to A: the nested choice must come back too. Without memory this
    // lands on 'A-new', the newest child.
    stepSibling(cv, bB.id, -1)
    expect(pathText(cv)).toEqual(['q', 'A', 'A-old', 'A-old-deep'])
  })

  it('descends to the NEWEST branch where nothing is remembered', () => {
    // An unvisited branch point should land on the freshest turn, which is
    // where a just-finished regenerate leaves you.
    const q = msg('user', 'q', null)
    const a = msg('assistant', 'a', q.id)
    const old = msg('user', 'older', a.id)
    const recent = msg('user', 'newer', a.id)
    const cv = conv([q, a, old, recent], q.id)
    expect(descendLeaf(cv, q.id)).toBe(recent.id)
  })
})

// ── deletion ────────────────────────────────────────────────────────────────

describe('deleteSubtree', () => {
  it('takes the turn and everything under it, and lands the cursor on a survivor', () => {
    const q = msg('user', 'q', null)
    const a1 = msg('assistant', 'keep', q.id)
    const a2 = msg('assistant', 'drop', q.id)
    const a2b = msg('user', 'drop-child', a2.id)
    const cv = conv([q, a1, a2, a2b], a2b.id)

    const gone = deleteSubtree(cv, a2.id)
    expect(gone.sort()).toEqual([a2.id, a2b.id].sort())
    expect(cv.messages.map((m) => m.id)).toEqual([q.id, a1.id])
    expect(pathText(cv)).toEqual(['q', 'keep'])
  })

  it('removes a compare group whole - never one lane of it', () => {
    const q = msg('user', 'q', null)
    const l1 = msg('assistant', 'lane1', q.id, { group: 'g', model: 'A' })
    const l2 = msg('assistant', 'lane2', q.id, { group: 'g', model: 'B' })
    const cv = conv([q, l1, l2], l1.id)

    deleteSubtree(cv, l2.id) // asked about one lane
    expect(cv.messages.map((m) => m.id)).toEqual([q.id]) // both went
    expect(cv.leafId).toBe(q.id)
  })

  it('falls back to the parent when its last child goes', () => {
    const q = msg('user', 'q', null)
    const a = msg('assistant', 'a', q.id)
    const cv = conv([q, a], a.id)
    deleteSubtree(cv, a.id)
    expect(cv.leafId).toBe(q.id)
    expect(pathText(cv)).toEqual(['q'])
  })

  it('prunes branch memory that would point into the removed subtree', () => {
    const q = msg('user', 'q', null)
    const a1 = msg('assistant', 'keep', q.id)
    const a2 = msg('assistant', 'drop', q.id)
    const cv = conv([q, a1, a2], a2.id)
    cv.branchMemory = { [q.id]: a2.id }

    deleteSubtree(cv, a2.id)
    expect(cv.branchMemory?.[q.id]).toBeUndefined()
    // and the path still resolves rather than descending into a ghost
    expect(pathText(cv)).toEqual(['q', 'keep'])
  })
})
