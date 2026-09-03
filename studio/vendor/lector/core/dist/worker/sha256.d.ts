/**
 * SHA-256 (FIPS 180-4) in plain JS — the fallback digest for insecure
 * contexts.
 *
 * `crypto.subtle` exists only on https and localhost origins; an embedder
 * served over plain http on a LAN address (a self-hosted tool opened as
 * http://192.168.x.x) has no WebCrypto at all, and the open-time document
 * fingerprint used to throw there and take the whole open down with it.
 * This produces byte-identical digests to `subtle.digest('SHA-256', …)`,
 * so a fingerprint minted on an insecure origin still matches one minted
 * on https for the same bytes.
 *
 * Speed is not the point — WebCrypto stays the primary path — but this
 * runs at tens of MB/s in the worker, which is fine for open-time hashing.
 */
/**
 * Digest `data` and return a 32-byte ArrayBuffer — the same shape
 * `crypto.subtle.digest('SHA-256', data)` resolves to.
 */
export declare function sha256(data: ArrayBuffer): ArrayBuffer;
//# sourceMappingURL=sha256.d.ts.map