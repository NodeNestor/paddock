import type { DocumentId } from '../types/handle-id.js';
import type { AnnotationData, RgbaColor, TrackedObject } from '../data/types.js';
/**
 * Redaction capability.
 *
 * Two-phase workflow:
 * 1. **Mark** — create REDACT annotations on areas to be redacted.
 *    These appear as red boxes indicating what will be removed.
 * 2. **Apply** — permanently remove content under redaction annotations
 *    by flattening the page. This is irreversible.
 */
export interface RedactionCapability {
    /** Create a redaction annotation marking an area for removal. */
    markForRedaction(docId: DocumentId, pageIndex: number, rect: AnnotationData['rect'], options?: {
        reason?: string;
        overlayText?: string;
        overlayColor?: RgbaColor;
    }): Promise<TrackedObject<AnnotationData>>;
    /**
     * Apply all redactions on a page — permanently removes content.
     * This operation is irreversible.
     */
    applyRedactions(docId: DocumentId, pageIndex: number): Promise<void>;
    /** Get all pending (unapplied) redaction annotations for a document. */
    getMarkedRedactions(docId: DocumentId): TrackedObject<AnnotationData>[];
    /** Get pending redaction count for a specific page. */
    getPageRedactionCount(docId: DocumentId, pageIndex: number): number;
}
export declare const redactionPlugin: import("../index.js").PluginDefinition<RedactionCapability, Record<string, never>>;
//# sourceMappingURL=redaction-plugin.d.ts.map