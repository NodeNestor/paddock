import type { ReadonlySignal } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
import type { DocumentHandle } from '../engine/lector-engine.js';
/**
 * Capability provided by the document plugin.
 *
 * Manages loading, closing, and tracking open PDF documents.
 */
export interface DocumentCapability {
    load(source: ArrayBuffer | File | URL | string, password?: string): Promise<DocumentHandle>;
    close(docId: DocumentId): Promise<void>;
    getHandle(docId: DocumentId): DocumentHandle | undefined;
    activeDocument: ReadonlySignal<DocumentHandle | null>;
    setActive(docId: DocumentId): void;
}
/**
 * Document loading plugin.
 *
 * Provides the `'document'` capability for opening, closing, and tracking
 * PDF documents. Emits `'document:loaded'` and `'document:closed'` events.
 */
export declare const documentPlugin: import("../index.js").PluginDefinition<DocumentCapability, Record<string, never>>;
//# sourceMappingURL=document-plugin.d.ts.map