import type { LectorEngine } from '../engine/lector-engine.js';
import type { ViewportInstance } from '../plugins/viewport-plugin.js';
import type { FormattingCapability } from '../plugins/formatting-plugin.js';
import type { ComparisonChange, PageDiff } from '../worker/comparison-ops.js';
import type { DocumentId } from '../types/handle-id.js';
/**
 * Payload of the `ui:page-mounted` engine event: the public contract for
 * embedding applications that draw their OWN page-aligned content (OCR
 * region boxes, ML highlights, review marks) on top of the viewer.
 *
 * `overlayEl` is a lector-managed layer covering the page's content box
 * exactly — the host renders into it (fractional/percentage coordinates of
 * the rendered page hold under every zoom level) and never positions
 * against the viewer's internals. It is emitted when a page element starts
 * showing a `(document, pageIndex)` pair: on first mount, and again when a
 * tab switch reuses the element for a different document. The element is
 * removed with its page; `ui:page-unmounted` `{ pageIndex }` says when.
 *
 * The layer itself is `pointer-events: none`; interactive host content
 * re-enables pointer events on its own nodes.
 */
export interface PageMountedEvent {
    pageIndex: number;
    overlayEl: HTMLElement;
    docId: DocumentId | null;
}
/**
 * Which side of a comparison this overlay manager renders. The
 * comparison plugin produces one `ComparisonResult` containing rects on
 * both sides; each pane's overlay manager only renders its own side's
 * geometry. `'A'` shows deletes + replaces' before-rect; `'B'` shows
 * inserts + replaces' after-rect. Region-mode changes carry rects on
 * both sides and render in both panes.
 */
export type ComparisonSide = 'A' | 'B';
/**
 * Manages per-page overlay layers for text selection, search highlights,
 * links, annotations, and form fields.
 */
export declare class PageOverlayManager implements Disposable {
    #private;
    constructor(engine: LectorEngine, viewport: ViewportInstance, formatting?: FormattingCapability | null);
    /**
     * Called by the viewer when a page element is created.
     * Adds an overlay div to the page element.
     */
    attachPage(pageIndex: number, pageEl: HTMLElement, pageWidthPts: number, pageHeightPts: number): void;
    /** Called by the viewer when a page element is removed. */
    detachPage(pageIndex: number): void;
    /**
     * Apply a comparison result to this pane. Pass `null` to clear.
     *
     * The same `ComparisonResult` is shared across both panes — each pane's
     * manager only renders the rects that belong to its own side
     * (`'A'` = left/deletions, `'B'` = right/insertions). The third
     * argument is the change index to highlight as "active" (for prev/next
     * navigation).
     */
    setComparison(side: ComparisonSide, pageDiffs: readonly PageDiff[] | null, activeFlatIndex?: number): void;
    /** Update only the active change index without rebuilding the geometry. */
    setComparisonActiveIndex(activeFlatIndex: number): void;
    /**
     * Look up the page index that contains a specific change for this
     * side. Returns -1 if the change has no rect on this side (e.g. an
     * insertion-only change has no `pageA` for the A pane).
     */
    comparisonPageForChange(flatIndex: number): number;
    /** The flat change list snapshot — used by the sidebar panel. */
    get comparisonFlatChanges(): readonly {
        diff: PageDiff;
        change: ComparisonChange;
    }[];
    rebuildOverlays(): void;
    destroy(): void;
    [Symbol.dispose](): void;
}
//# sourceMappingURL=page-overlays.d.ts.map