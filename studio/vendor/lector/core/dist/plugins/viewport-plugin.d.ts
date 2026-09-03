import type { ReadonlySignal } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
/** Layout mode for page arrangement in the viewport. */
export type LayoutMode = 'single' | 'continuous' | 'double' | 'book';
/** Computed position and size of a page within the viewport's virtual canvas. */
export interface PagePosition {
    readonly pageIndex: number;
    readonly x: number;
    readonly y: number;
    readonly width: number;
    readonly height: number;
}
/** Stable identifier for a viewport instance within an engine. */
export type ViewportId = string;
/**
 * One scrollable view onto a (possibly shared) document. The split-pane
 * design allows multiple `ViewportInstance`s on the same engine — each
 * with its own scroll position, scale, layout, container, and optionally
 * its own document.
 *
 * Existing single-pane code uses the `ViewportCapability`'s singleton
 * facade methods (`attach`, `scale`, `pagePositions`, etc.), which
 * delegate to a "primary" instance created on first `attach()`. New code
 * uses `viewport.createViewport()` to obtain explicit instances.
 */
export interface ViewportInstance {
    /** Stable id assigned at creation time. */
    readonly id: ViewportId;
    /** The DOM container this viewport is attached to (or null if unattached). */
    readonly container: HTMLElement | null;
    /**
     * Document currently displayed in this viewport. If null, the
     * viewport will follow `documentCapability.activeDocument` (legacy
     * single-pane behavior). Set explicitly for multi-pane to pin a
     * specific document to a specific pane.
     */
    readonly docId: ReadonlySignal<DocumentId | null>;
    /** Reactive container size. */
    readonly containerSize: ReadonlySignal<{
        width: number;
        height: number;
    }>;
    /** Reactive scroll offset (pixels). */
    readonly scrollOffset: ReadonlySignal<{
        x: number;
        y: number;
    }>;
    /** Reactive scale factor (1 = 100%). */
    readonly scale: ReadonlySignal<number>;
    /** Reactive layout mode. */
    readonly layoutMode: ReadonlySignal<LayoutMode>;
    /** Reactive computed page positions for the viewport's document. */
    readonly pagePositions: ReadonlySignal<PagePosition[]>;
    /** Reactive total content height (pixels). */
    readonly totalHeight: ReadonlySignal<number>;
    /** Reactive list of visible page indices. */
    readonly visiblePages: ReadonlySignal<number[]>;
    /** Attach this viewport to a DOM container. May only be attached to one container at a time. */
    attach(container: HTMLElement): void;
    /** Detach from its container; cleans up the ResizeObserver and scroll listener. */
    detach(): void;
    /** Pin a specific document to this viewport. Pass null to follow the active document. */
    setDocument(docId: DocumentId | null): void;
    /** Set scale factor (1 = 100%). */
    setScale(scale: number): void;
    /** Set the layout mode. */
    setLayoutMode(mode: LayoutMode): void;
    /** Scroll so the given page index is in view. Pass `false` for instant jump. */
    scrollToPage(pageIndex: number, smooth?: boolean): void;
    /**
     * Pause/resume container size updates from the ResizeObserver. Used
     * during interactions like sidebar drag where the user resizes the
     * surrounding layout via CSS only and we don't want every pointermove
     * to flush a full reactive layout pass.
     */
    setResizeObserverPaused(paused: boolean): void;
    /** Destroy this viewport, freeing all observers and listeners. */
    destroy(): void;
}
/** Options for creating a new viewport instance. */
export interface CreateViewportOptions {
    /** Optional explicit id. If omitted, one is generated. */
    readonly id?: ViewportId;
    /**
     * Initial document to display. If omitted, the viewport follows
     * `documentCapability.activeDocument`.
     */
    readonly docId?: DocumentId | null;
    /** Initial layout mode. Default `'continuous'`. */
    readonly layoutMode?: LayoutMode;
    /** Initial scale factor. Default `1`. */
    readonly scale?: number;
    /** Gap between pages in px. Defaults to engine option or 8. */
    readonly pageGap?: number;
    /** Padding around the viewport in px. Defaults to engine option or 12. */
    readonly viewportPadding?: number;
}
/**
 * Capability provided by the viewport plugin.
 *
 * Two-tier API:
 *
 * 1. **Multi-instance API** (preferred for new code) — `createViewport`,
 *    `destroyViewport`, `getViewport`, `viewports`, `activeViewport`.
 *    Lets the host create multiple independent viewports on the same
 *    engine for split-pane layouts.
 *
 * 2. **Singleton facade** (kept for backward compatibility) — `attach`,
 *    `detach`, `scale`, `scrollOffset`, `pagePositions`, etc. These
 *    methods delegate to a "primary" viewport instance, which is
 *    created lazily on the first `attach()` call. Single-pane code that
 *    has not been migrated to the multi-instance API continues to work.
 */
export interface ViewportCapability {
    /** Create a new viewport instance. */
    createViewport(options?: CreateViewportOptions): ViewportInstance;
    /** Destroy a viewport instance by id. */
    destroyViewport(id: ViewportId): void;
    /** Look up a viewport by id. */
    getViewport(id: ViewportId): ViewportInstance | null;
    /** All currently-active viewport instances (reactive). */
    readonly viewports: ReadonlySignal<readonly ViewportInstance[]>;
    /** The active viewport (the one the user is interacting with). */
    readonly activeViewport: ReadonlySignal<ViewportInstance | null>;
    /** Mark a viewport as active. */
    setActiveViewport(id: ViewportId): void;
    /**
     * Attach the singleton viewport to a container. If no primary viewport
     * exists yet, one is created. Identical behavior to single-pane code
     * before the multi-instance refactor.
     */
    attach(container: HTMLElement): void;
    /** Detach the singleton viewport's container. */
    detach(): void;
    /** Scroll the singleton viewport to a page. Pass `false` for instant jump. */
    scrollToPage(pageIndex: number, smooth?: boolean): void;
    /** Set the singleton viewport's layout mode. */
    setLayoutMode(mode: LayoutMode): void;
    /** Set the buffer size (pages above/below visible to keep rendered). */
    setBufferSize(pages: number): void;
    /** Set the singleton viewport's scale factor. */
    setScale(scale: number): void;
    /** Pause/resume the singleton viewport's ResizeObserver. */
    setResizeObserverPaused(paused: boolean): void;
    /** Reactive signals — delegate to the primary viewport. */
    readonly visiblePages: ReadonlySignal<number[]>;
    readonly containerSize: ReadonlySignal<{
        width: number;
        height: number;
    }>;
    readonly scrollOffset: ReadonlySignal<{
        x: number;
        y: number;
    }>;
    readonly layoutMode: ReadonlySignal<LayoutMode>;
    readonly pagePositions: ReadonlySignal<PagePosition[]>;
    readonly totalHeight: ReadonlySignal<number>;
    readonly scale: ReadonlySignal<number>;
}
/**
 * Viewport & virtual scroll plugin.
 *
 * Manages one or more `ViewportInstance`s per engine. Single-pane code
 * uses the singleton facade methods (`attach`, `scale`, etc.) which
 * transparently delegate to a "primary" viewport instance. Split-pane
 * code uses `createViewport()` to obtain explicit instances.
 *
 * The primary viewport is created lazily on the first `attach()` call.
 * Detaching it leaves the instance in place but disconnected; a
 * subsequent `attach()` reuses it.
 */
export declare const viewportPlugin: import("../index.js").PluginDefinition<ViewportCapability, Record<string, never>>;
//# sourceMappingURL=viewport-plugin.d.ts.map