import type { DocumentId } from '../types/handle-id.js';
import type { SigningOptions, SigningResult } from '../worker/signing-ops.js';
export type { SigningOptions, SigningResult, SignaturePlaceholderOptions, PreparedSignature, } from '../worker/signing-ops.js';
export interface SignatureSigningCapability {
    /**
     * Digitally sign the document with a PFX/P12 certificate.
     *
     * @returns The signed PDF bytes and signer information.
     */
    sign(docId: DocumentId, options: SigningOptions): Promise<SigningResult>;
}
export declare const signatureSigningPlugin: import("../index.js").PluginDefinition<SignatureSigningCapability, Record<string, never>>;
//# sourceMappingURL=signature-signing-plugin.d.ts.map