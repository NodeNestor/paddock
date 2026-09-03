/**
 * `navigator.clipboard` is secure-context-only (https, localhost). A viewer
 * embedded on a plain-http LAN origin has no async clipboard at all, so
 * text copy falls back to the deprecated-but-universal execCommand path.
 * Rich copies (ClipboardItem images) have no fallback — callers keep their
 * own try/catch and surface an honest failure there.
 */
export declare function copyText(text: string): Promise<void>;
//# sourceMappingURL=clipboard.d.ts.map