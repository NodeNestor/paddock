import type { DocumentId } from '../types/handle-id.js';
import type { SignatureValidationResult } from '../worker/crypto-ops.js';
export type { SignatureValidationResult, CertificateInfo, SignatureValidationStatus } from '../worker/crypto-ops.js';
export interface SignatureValidationCapability {
    /**
     * Validate a specific signature cryptographically. The worker uses
     * the document bytes already loaded on its WASM heap — no need to
     * pass anything from the main thread.
     */
    validate(docId: DocumentId, sigIndex: number): Promise<SignatureValidationResult>;
    /**
     * Validate all signatures in a document. Cheap on the main thread
     * because the bytes never cross the worker boundary.
     */
    validateAll(docId: DocumentId): Promise<SignatureValidationResult[]>;
}
export declare const signatureValidationPlugin: import("../index.js").PluginDefinition<SignatureValidationCapability, Record<string, never>>;
//# sourceMappingURL=signature-validation-plugin.d.ts.map