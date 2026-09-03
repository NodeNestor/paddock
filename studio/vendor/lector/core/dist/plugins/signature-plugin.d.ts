import type { DocumentId } from '../types/handle-id.js';
import type { SignatureInfo } from '../worker/signature-ops.js';
/**
 * Signature reading capability.
 *
 * Reads digital signature data from PDF documents. pdfium extracts signature
 * contents, byte ranges, sub-filters, and metadata. Actual cryptographic
 * validation (PKCS#7/CMS) requires a separate library — this plugin only
 * reads the signature data as-is.
 */
export interface SignatureCapability {
    /** Get the number of digital signatures in a document. */
    getCount(docId: DocumentId): Promise<number>;
    /** Read detailed info for a specific signature by index. */
    getInfo(docId: DocumentId, sigIndex: number): Promise<SignatureInfo>;
    /** Read all signatures in the document. */
    getAllInfo(docId: DocumentId): Promise<SignatureInfo[]>;
}
export declare const signaturePlugin: import("../index.js").PluginDefinition<SignatureCapability, Record<string, never>>;
//# sourceMappingURL=signature-plugin.d.ts.map