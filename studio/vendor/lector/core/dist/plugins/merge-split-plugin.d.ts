import type { DocumentId } from '../types/handle-id.js';
/**
 * Capability provided by the merge-split plugin.
 *
 * All methods return raw PDF bytes (ArrayBuffer) that the consumer can
 * download, open as a new document, or send to a server.
 */
export interface MergeSplitCapability {
    /**
     * Merge multiple open documents into a single new PDF.
     * Pages are concatenated in the order of the provided document IDs.
     */
    mergeDocuments(docIds: DocumentId[]): Promise<ArrayBuffer>;
    /**
     * Split a document into multiple PDFs by page ranges.
     * Each range is `{ start, end }` with 0-based inclusive page indices.
     * Returns one ArrayBuffer per range.
     */
    splitDocument(docId: DocumentId, ranges: Array<{
        start: number;
        end: number;
    }>): Promise<ArrayBuffer[]>;
    /**
     * Extract specific pages from a document into a new PDF.
     * @param pageIndices 0-based page indices to extract.
     */
    extractPages(docId: DocumentId, pageIndices: number[]): Promise<ArrayBuffer>;
}
export declare const mergeSplitPlugin: import("../index.js").PluginDefinition<MergeSplitCapability, Record<string, never>>;
//# sourceMappingURL=merge-split-plugin.d.ts.map