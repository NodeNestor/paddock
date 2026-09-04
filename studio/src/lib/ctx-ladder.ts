// The context ladder on the Start page: what the "Context per conversation"
// select offers, computed from two facts that must never be conflated - what
// the CARD can back at this concurrency (VRAM) and what the MODEL itself
// supports. Pure, so the ladder can be exercised without a GPU or a store.
//
// Why the top rung exists: the ladder is powers of two, and the card's real
// ceiling almost never is. On a 32 GB card that backs 224K, a ladder that only
// offers what fits shows 128K as its top and quietly loses almost half the
// headroom. The fit number itself is the rung people actually want.

/** The fixed rungs, 4K to 256K. */
export const CTX_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144] as const

/** Select value for "everything that fits" - the cap itself, tracked live. */
export const CTX_MAX = '__max'
/** Select value for "custom" - the number field takes over. */
export const CTX_CUSTOM = '__custom'

/** KV pages hold 16 tokens (kv_pool.rs BLOCK_TOKENS); a context the pool can
 *  back is a whole number of pages, so the fit rung is aligned down to one. */
export const CTX_BLOCK = 16

export interface CtxLadderInput {
  /** Tokens per conversation VRAM backs at this concurrency; 0 = not estimated. */
  vramCap: number
  /** The model's own maximum context; 0 = unknown. */
  modelCap: number
  /** Conversations at once - the multiplier on every KV byte. */
  batch: number
  /** KV bytes per token per conversation; 0 = unknown (no shortfall figure). */
  bytesPerToken: number
}

export interface CtxOption {
  value: number | string
  label: string
  hint?: string
  disabled?: boolean
}

/** The cap that applies - the lower of the card and the model - aligned down
 *  to a KV page. 0 while there is no estimate. */
export function ctxCapOf(i: Pick<CtxLadderInput, 'vramCap' | 'modelCap'>): number {
  if (!i.vramCap) return 0
  const cap = i.modelCap ? Math.min(i.vramCap, i.modelCap) : i.vramCap
  return Math.max(CTX_BLOCK, cap - (cap % CTX_BLOCK))
}

/** The rungs that FIT - what auto-selection and clamping may choose from.
 *  Falls back to the exact cap when nothing on the ladder does. */
export function ctxFits(i: Pick<CtxLadderInput, 'vramCap' | 'modelCap'>): number[] {
  const cap = ctxCapOf(i)
  if (!cap) return [...CTX_STEPS]
  const opts = CTX_STEPS.filter((c) => c <= cap)
  return opts.length ? opts : [cap]
}

/** Everything the select shows, top to bottom: the fit rung (when estimated),
 *  the model's rungs with the ones VRAM cannot back greyed and priced, and
 *  Custom. Rungs past the model's own maximum stay out - those do not exist
 *  at any VRAM.
 *
 *  `fmt` renders a token count ("224K"), `gb` a byte count ("6.1 GB"). */
export function ctxLadder(
  i: CtxLadderInput,
  fmt: (tokens: number) => string,
  gb: (bytes: number) => string,
): CtxOption[] {
  const cap = ctxCapOf(i)
  const steps = CTX_STEPS.filter((c) => !i.modelCap || c <= i.modelCap)
  const rungs: CtxOption[] = steps.map((c) => {
    if (!cap || c <= cap) return { value: c, label: fmt(c) }
    // Priced, not just refused: the number that tells someone what lowering
    // concurrency or KV precision would have to buy back.
    const short = i.bytesPerToken > 0 ? (c - i.vramCap) * Math.max(1, i.batch) * i.bytesPerToken : 0
    return {
      value: c,
      label: fmt(c),
      disabled: true,
      hint: short > 0 ? `needs ${gb(short)} more` : 'will not fit VRAM',
    }
  })
  const top: CtxOption[] = cap
    ? [{ value: CTX_MAX, label: `Everything that fits · ${fmt(cap)}`, hint: `${Math.max(1, i.batch)} at once` }]
    : []
  return [...top, ...rungs, { value: CTX_CUSTOM, label: 'Custom, pick a number' }]
}
