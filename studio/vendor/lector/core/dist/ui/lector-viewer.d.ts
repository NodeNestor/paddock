import type { LectorEngine } from '../engine/lector-engine.js';
import type { UICapability } from '../plugins/ui-plugin.js';
import type { DocumentId } from '../types/handle-id.js';
/**
 * Options for constructing a {@link LectorViewer} instance.
 *
 * The viewer reads plugin capabilities from the engine, so most
 * feature configuration lives on {@link LectorEngineOptions}. These
 * options control the viewer's initial visual state and which UI
 * elements are shown.
 */
export interface LectorViewerOptions {
    /** DOM element that will host the viewer. Must already be in the document. */
    container: HTMLElement;
    /** The initialized {@link LectorEngine} instance powering this viewer. */
    engine: LectorEngine;
    /**
     * Whether the "Open file" button and drag-and-drop are enabled so
     * users can load local PDF files from their device. Default `false`.
     */
    allowLocalOpen?: boolean;
    /** Initial theme. Defaults to `'system'` (follows OS preference). */
    theme?: 'light' | 'dark' | 'system';
    /** Whether the sidebar starts open. Defaults to `true` on desktop, `false` on mobile. */
    sidebarOpen?: boolean;
    /** Which sidebar panel is initially active (e.g. `'thumbnails'`, `'bookmarks'`). */
    initialPanel?: string;
    /** Initial page layout mode. Defaults to `'continuous'`. */
    layoutMode?: 'single' | 'continuous' | 'double' | 'book';
    /**
     * Initial zoom level applied when the first document loads.
     * Pass a number (`1.0` = 100%) or a named fit mode. Defaults to `'fit-width'`.
     */
    initialZoom?: number | 'fit-width' | 'fit-page';
    /**
     * Which sidebar panels to show. When provided, only these panel IDs
     * are included in the tab strip. Omit to show all registered panels.
     *
     * @example ['thumbnails', 'bookmarks', 'annotations']
     */
    panels?: string[];
    /**
     * Partial UI schema override. Merged on top of the default schema
     * at construction time, allowing consumers to reorder toolbar items,
     * hide specific panels, or inject custom entries without replacing
     * the entire schema.
     */
    uiSchema?: Record<string, unknown>;
    /**
     * Whether the document tab bar is shown. Defaults to `true`.
     *
     * Turn it off when the HOST already owns document switching — an app with
     * its own file list, where the viewer shows whichever document that list
     * has selected. Two controls for one choice is confusing on its own, and
     * the tab's close button is worse: in a host like that, "close" has no
     * meaning the host can honour (the document is still in its list) and
     * closing one leaves the host pointing at a document the viewer no longer
     * holds. Switching keeps working through `document.setActive()` either way.
     */
    documentTabs?: boolean;
    /**
     * Buttons the HOST contributes to the toolbar, rendered by lector so they
     * are the same size, spacing and hover treatment as its own — which is the
     * whole point. An embedder that draws its own bar above or below the
     * viewer gets two rows of chrome with two different button metrics; there
     * is nowhere for a host action to sit that looks like it belongs.
     *
     * `icon` is a built-in icon name (see icons.ts). Unknown names render an
     * empty button rather than throwing, so a typo is visible but not fatal.
     */
    toolbarExtras?: ReadonlyArray<{
        /** Stable id, used as `data-action` so hosts can find the button. */
        id: string;
        /** Built-in icon name. */
        icon: string;
        /** Tooltip text — already localized by the host. */
        tooltip: string;
        /** Which group it joins. Defaults to `'right'`. */
        section?: 'left' | 'center' | 'right';
        /**
         * Where inside that group. `'end'` (the default) puts it after the
         * viewer's own controls; `'start'` puts it before them, which is what a
         * host action that OWNS the pane wants — collapsing the whole viewer
         * belongs outside the controls it collapses, not tacked on after them.
         * The divider follows: it lands between the extras and lector's buttons
         * either way.
         */
        placement?: 'start' | 'end';
        /** Runs on click. */
        onSelect: () => void;
        /** Greys the button out and blocks the click. Re-read on every rebuild. */
        disabled?: () => boolean;
    }>;
}
/**
 * One entry in the viewer's tab bar. A tab is either a single document
 * or a side-by-side split of two documents (orientation determines the
 * divider direction). Splits are ephemeral per session — they are not
 * persisted across reloads.
 */
type DocTab = {
    kind: 'single';
    docId: DocumentId;
    name: string;
} | {
    kind: 'split';
    orientation: 'horizontal' | 'vertical';
    left: {
        docId: DocumentId;
        name: string;
    };
    /**
     * The right pane's content. `null` while the user has split the tab
     * but not yet chosen a second document — the pane is rendered as an
     * "Open PDF to compare" placeholder. Filled in by the file picker
     * or a per-pane file drop.
     */
    right: {
        docId: DocumentId;
        name: string;
    } | null;
};
/**
 * LectorViewer — vanilla DOM renderer for the Lector PDF viewer.
 */
export declare class LectorViewer implements Disposable {
    #private;
    constructor(options: LectorViewerOptions);
    /** Toggle the comments sidebar (toolbar button + keyboard handler). */
    toggleCommentsSidebar(): void;
    /** Predefined PDF stamp names (standard + custom). */
    static readonly STAMPS: {
        name: string;
        label: string;
        color: string;
    }[];
    /** Get the currently selected stamp name for placement. */
    get activeStampName(): string;
    /** True if the active tab is a split. */
    get isSplit(): boolean;
    /** Read-only view of the current tab list. */
    get tabs(): readonly DocTab[];
    /**
     * Load a PDF document from raw bytes into a new tab.
     *
     * @param data   - The complete PDF file as an ArrayBuffer.
     * @param password - Optional document-open password for encrypted PDFs.
     * @param name   - Display name shown in the tab bar (defaults to "Document N").
     */
    loadDocument(data: ArrayBuffer, password?: string, name?: string): Promise<void>;
    /**
     * Load a PDF from a URL with optional custom headers.
     *
     * @example
     * ```ts
     * await viewer.loadDocumentFromUrl('/api/docs/report.pdf', {
     *   headers: { Authorization: `Bearer ${token}` },
     *   credentials: 'include',
     * });
     * ```
     */
    loadDocumentFromUrl(url: string, options?: RequestInit & {
        password?: string;
        name?: string;
    }): Promise<void>;
    /**
     * Switch to whichever tab contains the given doc id. Backward-compat
     * shim around the new tab model — prefer `switchTab(index)` for
     * direct tab navigation.
     */
    switchDocument(docId: string): void;
    /**
     * Switch to a tab by index. For split tabs, optionally focus a
     * specific side. For same-tab side switches inside a split, the
     * extra pane is reused (no rebuild) — only the active-viewport
     * pointer is updated, which lets the effect downstream update the
     * active document, the doctab highlight, and the chrome.
     */
    switchTab(index: number, side?: 'left' | 'right'): void;
    /**
     * Close the document with the given id. Backward-compat shim — for
     * split tabs this closes only the matching side and demotes the tab
     * back to a single. To close the whole tab use `closeTab(index)`.
     */
    closeDocument(docId: string): Promise<void>;
    /** Close an entire tab. For split tabs this closes both documents. */
    closeTab(index: number): Promise<void>;
    /**
     * Close one side of a split tab, demoting it to a single tab. The
     * remaining doc continues to be the tab's content. No-op if the
     * tab is not a split or the index is out of bounds.
     *
     * Special cases:
     *  - Closing the empty placeholder side (right === null on the side
     *    being closed): nothing to free in the engine; just demote.
     *  - Closing the loaded side while the other side is still empty: the
     *    tab would have no content left, so close the whole tab.
     */
    closeTabSide(index: number, side: 'left' | 'right'): Promise<void>;
    /** The {@link LectorEngine} instance powering this viewer. */
    get engine(): LectorEngine;
    /** The UI capability for programmatic sidebar / theme / schema control. */
    get ui(): UICapability;
    /**
     * Tear down the viewer: remove DOM, disconnect observers, and free
     * all resources. The engine is NOT destroyed — the consumer owns it
     * and may attach another viewer or call `engine.destroy()` separately.
     */
    destroy(): void;
    [Symbol.dispose](): void;
}
export {};
//# sourceMappingURL=lector-viewer.d.ts.map