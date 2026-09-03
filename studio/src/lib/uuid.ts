// crypto.randomUUID is SECURE-CONTEXT-ONLY: it exists on https and on
// http://localhost, and is simply absent on a LAN origin like
// http://10.10.0.189 - where every chat action threw and the Studio broke
// getRandomValues is available in all contexts, so the
// fallback builds the same RFC 4122 v4 uuid from it.
export function uuid(): string {
  // typed as optional: the lib's Crypto type claims randomUUID always
  // exists, and the `in` guard then narrows the fallback branch to `never`
  const c = crypto as Crypto & { randomUUID?: () => string }
  if (c.randomUUID) return c.randomUUID()
  const b = c.getRandomValues(new Uint8Array(16))
  b[6] = (b[6] & 0x0f) | 0x40
  b[8] = (b[8] & 0x3f) | 0x80
  const h = [...b].map((x) => x.toString(16).padStart(2, '0')).join('')
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`
}
