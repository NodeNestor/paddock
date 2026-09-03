import type { ReadonlySignal } from '@truespar/lector-utils';
import type { DocumentHandle } from '../engine/lector-engine.js';
import type { DocumentId } from '../types/handle-id.js';
/**
 * Provenance of an open document.
 *
 * Used both as live metadata on `OpenDocument` and as the persistent
 * record in the recent-files store.
 */
export type DocumentSource = {
    readonly type: 'file';
    readonly fileName: string;
} | {
    readonly type: 'url';
    readonly url: string;
} | {
    readonly type: 'buffer';
    readonly name?: string;
};
/** Per-document metadata tracked by the manager. */
export interface OpenDocument {
    readonly id: DocumentId;
    readonly handle: DocumentHandle;
    /** Display name. Falls back to the source's intrinsic name. */
    readonly name: string;
    /** Original size in bytes, or 0 if unknown (e.g. opened from a URL). */
    readonly size: number;
    readonly source: DocumentSource;
    readonly openedAt: Date;
}
/** A persisted recent-files entry. */
export interface RecentFileEntry {
    readonly name: string;
    readonly source: DocumentSource;
    readonly size?: number;
    readonly lastOpenedAt: Date;
}
/**
 * Pluggable storage backend for the recent-files list.
 *
 * The default implementation uses `window.localStorage`. Apps can supply
 * their own (IndexedDB, server-side, custom encryption, etc.) by setting
 * `LectorEngineOptions.recentFilesStore`.
 */
export interface RecentFilesStore {
    list(): Promise<RecentFileEntry[]> | RecentFileEntry[];
    add(entry: RecentFileEntry): Promise<void> | void;
    remove(source: DocumentSource): Promise<void> | void;
    clear(): Promise<void> | void;
}
/** Options for `openFromBuffer` / `openFromFile` / `openFromUrl`. */
export interface OpenDocumentOptions {
    /** Password for encrypted documents. */
    readonly password?: string;
    /** Override the displayed name (default: derived from source). */
    readonly name?: string;
    /** Skip recording into the recent-files list (e.g. for ephemeral previews). */
    readonly skipRecent?: boolean;
}
/** Options for `openFileDialog`. */
export interface OpenFileDialogOptions {
    /** `accept` attribute for the file input. Default `.pdf,application/pdf`. */
    readonly accept?: string;
    /** Allow multiple selection. Default false. */
    readonly multiple?: boolean;
    /** Common open options applied to each picked file. */
    readonly openOptions?: OpenDocumentOptions;
}
/** Options for `registerDropZone`. */
export interface DropZoneOptions {
    /** Accept multiple dropped files. Default true. */
    readonly multiple?: boolean;
    /** CSS class to toggle on the element while a valid drag is over it. */
    readonly hoverClass?: string;
    /** Common open options applied to each dropped file. */
    readonly openOptions?: OpenDocumentOptions;
    /**
     * Localized text shown in the centered overlay while a file is being
     * dragged over the drop zone (e.g. `"Release to load PDF"`). When
     * provided, an overlay element is appended on dragenter and removed on
     * dragleave/drop. Pass the already-translated string — the doc manager
     * is i18n-agnostic.
     */
    readonly promptText?: string;
    /**
     * Whether to render the overlay at all. Defaults to `true` whenever
     * `promptText` is provided. Set to `false` to keep the hover-class
     * styling without an overlay element.
     */
    readonly showOverlay?: boolean;
}
/**
 * Document manager capability — high-level wrapper around the lower-level
 * `document` capability.
 *
 * Adds:
 *  - rich per-document metadata (name, size, source, openedAt)
 *  - a reactive list of open documents
 *  - a recent-files history with pluggable persistence
 *  - high-level convenience methods for opening from File / URL / Buffer
 *  - a file dialog helper
 *  - a drag-and-drop registration helper
 *
 * Designed to be the canonical entry point for app code that wants to
 * open PDFs and track which ones are open. The lower-level `document`
 * capability stays available for plugin code that needs raw load access.
 */
export interface DocumentManagerCapability {
    /** Open a PDF from a raw `ArrayBuffer`. */
    openFromBuffer(buffer: ArrayBuffer, options?: OpenDocumentOptions): Promise<OpenDocument>;
    /** Open a PDF from a `File` (e.g. from a file input or drag-drop). */
    openFromFile(file: File, options?: OpenDocumentOptions): Promise<OpenDocument>;
    /**
     * Open a PDF from a URL. The URL is fetched with the Fetch API; CORS
     * rules apply. The recent-files entry stores the URL so the doc can
     * be re-opened later via `openRecentFile`.
     */
    openFromUrl(url: string | URL, options?: OpenDocumentOptions): Promise<OpenDocument>;
    /**
     * Show a native file dialog and open the selected files. Resolves with
     * the list of opened documents (empty if the user cancelled).
     */
    openFileDialog(options?: OpenFileDialogOptions): Promise<OpenDocument[]>;
    /** Close one open document. */
    close(docId: DocumentId): Promise<void>;
    /** Close every open document. */
    closeAll(): Promise<void>;
    /** Get the metadata for a single open document. */
    getInfo(docId: DocumentId): OpenDocument | undefined;
    /** Reactive list of currently-open documents, in open-order. */
    readonly openDocuments: ReadonlySignal<readonly OpenDocument[]>;
    /** Reactive recent-files list, sorted most-recent-first. */
    readonly recentFiles: ReadonlySignal<readonly RecentFileEntry[]>;
    /** Re-open a recent file. URL entries refetch; File entries cannot be replayed. */
    openRecentFile(entry: RecentFileEntry, options?: OpenDocumentOptions): Promise<OpenDocument>;
    /** Remove a single recent-files entry. */
    removeRecentFile(source: DocumentSource): Promise<void>;
    /** Wipe the recent-files list. */
    clearRecentFiles(): Promise<void>;
    /**
     * Register a DOM element as a drop zone for PDF files. Adds dragover /
     * drop listeners that filter for PDF MIME or `.pdf` extension. Returns
     * an unsubscribe function.
     */
    registerDropZone(el: HTMLElement, options?: DropZoneOptions): () => void;
}
export declare const documentManagerPlugin: import("../index.js").PluginDefinition<DocumentManagerCapability, Record<string, never>>;
//# sourceMappingURL=document-manager-plugin.d.ts.map