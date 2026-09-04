// Tests for the context ladder (lib/ctx-ladder.ts).
//
// The ladder decides what a person can choose for the one setting that
// most directly costs VRAM. A wrong cap refuses a context the card could
// hold; a missing fit rung hides half the card behind powers of two.

import { describe, expect, it } from 'vitest'
import { CTX_CUSTOM, CTX_MAX, CTX_STEPS, ctxCapOf, ctxFits, ctxLadder } from './ctx-ladder'

const fmt = (n: number) => `${n}`
const gb = (b: number) => `${(b / 1e9).toFixed(1)} GB`

// a 32 GB card that backs ~224K for one conversation of a 256K model
const card = { vramCap: 229_400, modelCap: 262_144, batch: 1, bytesPerToken: 98_304 }

describe('ctxCapOf', () => {
  it('is the lower of card and model, aligned down to a KV page', () => {
    expect(ctxCapOf(card)).toBe(229_392)
    expect(ctxCapOf({ vramCap: 500_000, modelCap: 32_768 })).toBe(32_768)
  })
  it('is 0 until there is an estimate', () => {
    expect(ctxCapOf({ vramCap: 0, modelCap: 262_144 })).toBe(0)
  })
  it('ignores an unknown model cap', () => {
    expect(ctxCapOf({ vramCap: 100_010, modelCap: 0 })).toBe(100_000)
  })
})

describe('ctxFits', () => {
  it('is every rung when nothing is estimated', () => {
    expect(ctxFits({ vramCap: 0, modelCap: 0 })).toEqual([...CTX_STEPS])
  })
  it('keeps the rungs under the cap', () => {
    expect(ctxFits(card)).toEqual([4096, 8192, 16384, 32768, 65536, 131072])
  })
  it('falls back to the exact cap when no rung fits', () => {
    expect(ctxFits({ vramCap: 3000, modelCap: 0 })).toEqual([2992])
  })
})

describe('ctxLadder', () => {
  it('leads with the fit rung, greys and prices what the card cannot back, ends with Custom', () => {
    const l = ctxLadder(card, fmt, gb)
    expect(l[0]).toEqual({ value: CTX_MAX, label: 'Everything that fits · 229392', hint: '1 at once' })
    expect(l[l.length - 1]).toEqual({ value: CTX_CUSTOM, label: 'Custom, pick a number' })
    const rungs = l.slice(1, -1)
    expect(rungs.map((o) => o.value)).toEqual([...CTX_STEPS])
    expect(rungs.filter((o) => o.disabled).map((o) => o.value)).toEqual([262_144])
    // (262144 - 229400) tokens x 1 x 98304 B = 3.2 GB
    expect(rungs[rungs.length - 1]?.hint).toBe('needs 3.2 GB more')
  })
  it('prices the shortfall per conversation at once', () => {
    const l = ctxLadder({ ...card, batch: 4 }, fmt, gb)
    expect(l[0]?.hint).toBe('4 at once')
    expect(l[l.length - 2]?.hint).toBe('needs 12.9 GB more')
  })
  it('says will-not-fit when it cannot price it', () => {
    const l = ctxLadder({ ...card, bytesPerToken: 0 }, fmt, gb)
    expect(l[l.length - 2]?.hint).toBe('will not fit VRAM')
  })
  it('has no fit rung and nothing greyed before an estimate', () => {
    const l = ctxLadder({ vramCap: 0, modelCap: 0, batch: 1, bytesPerToken: 0 }, fmt, gb)
    expect(l[0]?.value).toBe(4096)
    expect(l.some((o) => o.disabled)).toBe(false)
    expect(l[l.length - 1]?.value).toBe(CTX_CUSTOM)
  })
  it('leaves out rungs past the model itself', () => {
    const l = ctxLadder({ vramCap: 500_000, modelCap: 32_768, batch: 1, bytesPerToken: 1 }, fmt, gb)
    expect(l.map((o) => o.value)).toEqual([CTX_MAX, 4096, 8192, 16384, 32768, CTX_CUSTOM])
  })
})
