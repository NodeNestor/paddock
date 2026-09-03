import type { DocumentId } from '../types/handle-id.js';
import type { AttachmentInfo } from '../worker/attachment-ops.js';
/**
 * Attachment management capability.
 *
 * Reads, adds, and deletes file attachments embedded in PDF documents.
 * Attachments are stored inside the PDF file — they persist across save/export.
 */
export interface AttachmentCapability {
    /** Get the number of embedded attachments. */
    getCount(docId: DocumentId): Promise<number>;
    /** List metadata for all attachments. */
    list(docId: DocumentId): Promise<AttachmentInfo[]>;
    /** Download an attachment's file content. */
    download(docId: DocumentId, index: number): Promise<{
        name: string;
        data: ArrayBuffer;
    }>;
    /** Add a new file attachment to the document. */
    add(docId: DocumentId, name: string, data: ArrayBuffer): Promise<void>;
    /** Delete an attachment by index. */
    delete(docId: DocumentId, index: number): Promise<void>;
}
export declare const attachmentPlugin: import("../index.js").PluginDefinition<AttachmentCapability, Record<string, never>>;
//# sourceMappingURL=attachment-plugin.d.ts.map