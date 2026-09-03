import type { ReadonlySignal } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
/** A rectangle in PDF-point coordinates with DOM-style top-left origin. */
export interface CaptureRect {
    readonly pageIndex: number;
    readonly x: number;
    readonly y: number;
    readonly width: number;
    readonly height: number;
}
/** Options accepted by `captureRegion`. */
export interface CaptureOptions {
    /** Target DPI for the rendered image. Default 300. */
    readonly dpi?: number;
    /**
     * Optional page rotation override (0/1/2/3 = 0/90/180/270° CW). Defaults
     * to 0; the marquee handler always uses the visual page coordinates.
     */
    readonly rotation?: 0 | 1 | 2 | 3;
}
/** Result of a capture operation. */
export interface CaptureResult {
    readonly bitmap: ImageBitmap;
    readonly blob: Blob;
    readonly rect: CaptureRect;
    readonly dpi: number;
}
/**
 * Capture capability — marquee snapshot tool.
 *
 * Lets the user drag a rectangle on a page to grab a high-DPI PNG of that
 * region, suitable for clipboard copy or download. Equivalent to the
 * "Snapshot" / "Capture" tool found in Adobe, Foxit, and Bluebeam.
 */
export interface CaptureCapability {
    /** True while the marquee tool is active. */
    readonly isMarqueeActive: ReadonlySignal<boolean>;
    /** Enter marquee-capture mode (sets interaction mode to 'marquee'). */
    enableMarquee(): void;
    /** Leave marquee-capture mode (sets interaction mode back to 'pointer'). */
    disableMarquee(): void;
    /** Toggle marquee mode. */
    toggleMarquee(): void;
    /**
     * Capture an arbitrary region of any page programmatically. Useful for
     * scripted exports without going through the marquee UI.
     */
    captureRegion(docId: DocumentId, rect: CaptureRect, options?: CaptureOptions): Promise<CaptureResult>;
}
/**
 * Marquee capture plugin.
 *
 * Registers a handler for the existing `marquee` interaction mode. The
 * handler draws an SVG rectangle on the canvas scroll area while the user
 * drags, then on release fires a `capture:region-selected` event with the
 * captured rect. The UI shell is responsible for showing the action
 * popover (Copy / Save / Cancel) — the plugin only deals with capturing
 * pixels and updating state.
 */
export declare const capturePlugin: import("../index.js").PluginDefinition<CaptureCapability, Record<string, never>>;
/**
 * Convert an `ImageBitmap` to a PNG `Blob` via OffscreenCanvas. Used both
 * by the capability's programmatic capture and by the UI shell when the
 * user picks "Save" or "Copy" from the action popover.
 */
export declare function imageBitmapToPngBlob(bitmap: ImageBitmap): Promise<Blob>;
//# sourceMappingURL=capture-plugin.d.ts.map