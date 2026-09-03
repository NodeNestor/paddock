import type { ReadonlySignal } from '@truespar/lector-utils';
import type { ViewportInstance } from './viewport-plugin.js';
/** The current interaction mode. */
export type InteractionMode = 'pointer' | 'draw' | 'text-select' | 'marquee' | 'pan';
/**
 * A point in page coordinate space.
 *
 * Coordinates are in PDF point units (1/72 inch), relative to the page's
 * top-left corner in DOM layout (Y increases downward). To convert to
 * true PDF coordinates (Y up from bottom), use: `pdfY = pageHeight - y`.
 */
export interface PagePoint {
    readonly pageIndex: number;
    readonly x: number;
    readonly y: number;
}
/** Event data passed to mode handlers. */
export interface PageInteractionEvent {
    /** The page point where the event occurred, or null if outside any page. */
    readonly pagePoint: PagePoint | null;
    /** The viewport-relative client coordinates. */
    readonly clientX: number;
    readonly clientY: number;
    /** Original DOM event. */
    readonly domEvent: PointerEvent | MouseEvent | KeyboardEvent;
    /**
     * The container element the event was raised on (i.e. the canvas of
     * one specific viewport / pane). Used by handlers that need to operate
     * on the right pane in a split-view configuration. May be null for
     * keyboard events not bound to a specific container.
     */
    readonly container: HTMLElement | null;
    /**
     * The viewport instance that owns `container`, if any. Handlers should
     * read viewport state (scale, pagePositions, scrollOffset) from this
     * instance rather than the singleton facade so split-view scenarios
     * see the right pane's state.
     */
    readonly viewport: ViewportInstance | null;
}
/**
 * Handler for a specific interaction mode.
 *
 * Implement the methods you need; all are optional. The interaction manager
 * calls the active handler's methods in response to DOM events.
 */
export interface ModeHandler {
    readonly cursor?: string;
    onPointerDown?(event: PageInteractionEvent): void;
    onPointerMove?(event: PageInteractionEvent): void;
    onPointerUp?(event: PageInteractionEvent): void;
    onClick?(event: PageInteractionEvent): void;
    onDoubleClick?(event: PageInteractionEvent): void;
    onKeyDown?(event: KeyboardEvent): void;
    onKeyUp?(event: KeyboardEvent): void;
    onActivate?(): void;
    onDeactivate?(): void;
}
/**
 * Capability provided by the interaction manager plugin.
 *
 * Manages the current interaction mode, dispatches pointer/keyboard events
 * to the active mode handler, and converts viewport coordinates to page
 * coordinate space.
 */
export interface InteractionCapability {
    /** Current interaction mode. */
    mode: ReadonlySignal<InteractionMode>;
    /** Current cursor CSS value. */
    cursor: ReadonlySignal<string>;
    /** Set the active interaction mode. */
    setMode(mode: InteractionMode): void;
    /** Register a handler for a specific mode. Replaces any existing handler. */
    registerHandler(mode: InteractionMode, handler: ModeHandler): void;
    /** Unregister a handler for a mode. */
    unregisterHandler(mode: InteractionMode): void;
    /**
     * Convert viewport-relative client coordinates to PDF page coordinates.
     *
     * The optional `container` argument disambiguates which pane to query
     * in a split-view configuration. When omitted, the function hit-tests
     * against every attached container's bounding rect.
     */
    viewportToPage(clientX: number, clientY: number, container?: HTMLElement | null): PagePoint | null;
    /** Override the cursor temporarily (e.g., during a drag). Call with null to reset. */
    setCursorOverride(cursor: string | null): void;
}
/**
 * Interaction manager plugin.
 *
 * Provides a mode system for coordinating pointer/keyboard interactions
 * across plugins. Each mode (pointer, draw, text-select, marquee, pan)
 * has a registered handler that receives translated page-coordinate events.
 */
export declare const interactionPlugin: import("../index.js").PluginDefinition<InteractionCapability, Record<string, never>>;
//# sourceMappingURL=interaction-plugin.d.ts.map