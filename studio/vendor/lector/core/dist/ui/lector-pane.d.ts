/**
 * One canvas / scroll area / viewport-instance pair within a multi-pane
 * Lector layout. A `LectorPane` is the v2 split-view counterpart to the
 * canvas portion of `LectorViewer`: it owns no chrome (no toolbar, no
 * sidebar, no doctabs) — just the page rendering surface and the page
 * lifecycle that goes with it.
 *
 * `LectorShell` constructs N `LectorPane`s and provides the shared
 * chrome around them. In single-pane mode (one pane in the shell) the
 * effect is identical to a regular `LectorViewer` minus the chrome.
 *
 * **Architecture intent.** Each pane:
 *  - Creates its own `ViewportInstance` from the shared engine's viewport
 *    plugin (multi-instance API from phase 1).
 *  - Owns its own canvas + scroll area + page DOM elements.
 *  - Owns its own `PageOverlayManager` bound to its viewport instance.
 *  - Drives its own render loop reactively from `viewport.visiblePages`
 *    and `viewport.pagePositions` of its instance — not the singleton
 *    facade.
 *
 * The shared engine, document, annotation store, search results, etc.
 * mean that opening a doc once shows it in every pane that points at
 * that doc, and edits in one pane reflect in another immediately.
 */
import type { LectorEngine } from '../engine/lector-engine.js';
import type { ViewportInstance } from '../plugins/viewport-plugin.js';
import type { DocumentId } from '../types/handle-id.js';
import { PageOverlayManager } from './page-overlays.js';
export interface LectorPaneOptions {
    /** Shared Lector engine. */
    readonly engine: LectorEngine;
    /** DOM element the pane will mount inside. */
    readonly container: HTMLElement;
    /**
     * Optional document id to pin to this pane. If omitted the pane
     * follows the engine's active document. Useful in split-view to show
     * different documents in different panes.
     */
    readonly docId?: DocumentId;
}
/**
 * One viewport's worth of canvas + page lifecycle + render loop, with
 * no chrome. Constructed by `LectorShell`; not generally used directly.
 */
export declare class LectorPane implements Disposable {
    #private;
    constructor(options: LectorPaneOptions);
    /** The pane's viewport instance. */
    get viewport(): ViewportInstance;
    /** The pane's canvas element (the scrollable host). */
    get canvas(): HTMLElement;
    /** The pane's outer container element. */
    get container(): HTMLElement;
    /**
     * The pane's PageOverlayManager. Exposed so chrome (LectorViewer) can
     * push comparison overlays onto the pane in compare mode without
     * having to mirror the overlay state inside the pane itself.
     */
    get overlays(): PageOverlayManager;
    /**
     * Pin a document to this pane (or null to follow the active document).
     * Use this in split-view to show different docs in different panes.
     */
    setDocument(docId: DocumentId | null): void;
    /** Force a re-render of currently visible pages. */
    rerenderVisible(): void;
    /** Tear down the pane: viewport, overlays, DOM, listeners. */
    destroy(): void;
    [Symbol.dispose](): void;
}
//# sourceMappingURL=lector-pane.d.ts.map