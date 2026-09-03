import type { ReadonlySignal } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
export interface ProgressiveLoadProgress {
    readonly phase: 'probing' | 'header' | 'first-page' | 'loading' | 'complete' | 'fallback';
    readonly bytesReceived: number;
    readonly totalBytes: number;
    readonly pagesAvailable: number;
    readonly totalPages: number;
}
export interface LinearizationCapability {
    /**
     * Load a PDF progressively from a URL using HTTP Range requests.
     *
     * For linearized PDFs, the first page renders immediately while
     * remaining pages stream in the background. Non-linearized PDFs
     * fall back to a full download.
     *
     * @param url       URL to fetch the PDF from.
     * @param options   Optional password and abort signal.
     * @returns The document ID once the document structure is available.
     */
    loadProgressive(url: string, options?: {
        password?: string;
        signal?: AbortSignal;
    }): Promise<DocumentId>;
    /** Current progress of the progressive load operation. */
    readonly progress: ReadonlySignal<ProgressiveLoadProgress | null>;
}
export declare const linearizationPlugin: import("../index.js").PluginDefinition<LinearizationCapability, Record<string, never>>;
//# sourceMappingURL=linearization-plugin.d.ts.map