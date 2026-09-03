import type { ReadonlySignal } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
import type { TextCharInfo, TextRect } from '../worker/text-ops.js';
/** A contiguous range of selected characters on a page. */
export interface TextSelection {
    /**
     * Document the selection belongs to. Required so consumers (overlay
     * managers, markup tools, etc.) can scope rendering to the correct
     * pane in a split-view configuration. Without this, a selection
     * created in one pane would also render on the other pane at the
     * same page index.
     */
    readonly docId: DocumentId;
    readonly pageIndex: number;
    readonly startCharIndex: number;
    readonly endCharIndex: number;
    readonly text: string;
    readonly rects: TextRect[];
}
/**
 * Capability provided by the text layer plugin.
 *
 * Extracts text and character positions from PDF pages, supports text
 * selection with copy-to-clipboard, and provides an accessible text layer.
 */
export interface TextLayerCapability {
    /**
     * Extract all text from a page.
     * Results are cached per document/page.
     */
    getPageText(docId: DocumentId, pageIndex: number): Promise<string>;
    /**
     * Extract per-character position and bounding box info for a page.
     * Results are cached per document/page.
     */
    getPageCharInfo(docId: DocumentId, pageIndex: number): Promise<TextCharInfo[]>;
    /** Get bounding rectangles for a range of characters. */
    getTextRects(docId: DocumentId, pageIndex: number, charIndex: number, count: number): Promise<TextRect[]>;
    /** Get the character index at a viewport position. */
    getCharIndexAtPos(docId: DocumentId, pageIndex: number, x: number, y: number, tolerance?: number): Promise<number>;
    /** Current text selection, or null if nothing is selected. */
    selection: ReadonlySignal<TextSelection | null>;
    /** Set the text selection programmatically. Pass null to clear. */
    setSelection(selection: TextSelection | null): void;
    /** Copy the current selection text to the clipboard. */
    copySelection(): Promise<void>;
    /** Clear any cached text data for a document. */
    clearCache(docId: DocumentId): void;
}
/**
 * Text layer plugin.
 *
 * Provides text extraction, character-level position data, and text
 * selection with clipboard support. Caches extracted text per document/page
 * to avoid redundant worker round-trips.
 */
export declare const textLayerPlugin: import("../index.js").PluginDefinition<TextLayerCapability, Record<string, never>>;
//# sourceMappingURL=text-layer-plugin.d.ts.map