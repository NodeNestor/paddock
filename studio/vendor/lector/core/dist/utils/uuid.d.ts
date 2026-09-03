/**
 * `crypto.randomUUID` exists only in secure contexts (https, localhost). An
 * embedder served over plain http on a LAN address has `crypto` without it,
 * and every id mint used to throw there. `getRandomValues` exists in ALL
 * contexts, so the fallback assembles the same RFC 4122 v4 uuid by hand.
 * Works in both window and worker scopes.
 */
export declare function uuid(): string;
//# sourceMappingURL=uuid.d.ts.map