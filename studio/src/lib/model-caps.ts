// Model capability inference. The server's /v1/models only reports id + owner,
// so until it advertises capabilities richly we infer them from the id. One
// place, so the composer's image gating and the Models panel agree.

export type Capability = 'chat' | 'vision' | 'embeddings' | 'rerank'

/** Best-effort primary capability from a model id. */
export function capability(id: string): Capability {
  const s = id.toLowerCase()
  if (s.includes('embedding')) return 'embeddings'
  if (s.includes('reranker') || s.includes('rerank')) return 'rerank'
  if (isVisionModel(id)) return 'vision'
  return 'chat'
}

/** A human-friendly name from a techy model id: drop the trailing quant/format
 *  tokens and space the rest. "Qwen3.6-27B-Q8_0" -> "Qwen3.6 27B". The full id
 *  stays available (shown as a tooltip). */
export function friendlyModelName(id: string): string {
  if (!id) return ''
  // A cloud composite id (`cloud:{endpoint-uuid}:{provider/model}`) whose
  // endpoint is gone can still reach a label fallback (stale compare sets,
  // old conversations) - the provider-model part is the readable piece, and
  // the quant-stripping below has nothing to say about it.
  if (id.startsWith('cloud:')) {
    const rest = id.slice('cloud:'.length)
    const colon = rest.indexOf(':')
    return colon >= 0 ? rest.slice(colon + 1) : rest
  }
  const parts = id.split('-')
  // Trailing quant/format tokens: Q8_0, Q4_K_M, F16, BF16, FP8, IQ4_XS, GGUF, ...
  const quant = /^(iq?\d|q\d|f16|bf16|fp\d|gguf|gptq|awq|int\d)/i
  while (parts.length > 1 && quant.test(parts[parts.length - 1])) parts.pop()
  return parts.join(' ')
}

/** gpt-oss / Harmony dialect - uses a Low/Med/High reasoning_effort knob rather
 *  than qwen3.5's binary enable_thinking. */
export function isHarmony(id: string): boolean {
  return /gpt-oss/i.test(id)
}

/** Does this model read images? (id heuristic: -vl / vision / mmproj / multimodal) */
export function isVisionModel(id: string): boolean {
  const s = id.toLowerCase()
  return (
    s.includes('-vl') ||
    s.includes('vision') ||
    s.includes('mmproj') ||
    s.includes('multimodal') ||
    s.includes('llava') ||
    s.includes('moondream') ||
    // a document reader is a vision model (unlimited-ocr); word-bounded so
    // an unlucky substring never turns a text model's composer into one
    /(^|[-_])ocr($|[-_])/.test(s)
  )
}
