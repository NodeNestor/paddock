/**
 * Marquee capture: render a sub-region of a PDF page to an `ImageBitmap`
 * at a chosen DPI.
 *
 * Implementation note: instead of rendering the full page and cropping in
 * JavaScript, we exploit `FPDF_RenderPageBitmap`'s ability to take negative
 * `start_x` / `start_y` arguments. The bitmap we allocate is exactly the
 * size of the requested region; the page is rendered into it with the
 * top-left offset that places the requested region inside the bitmap. Only
 * the visible region is actually drawn, so the cost scales with the
 * captured area, not the page size — important for high-DPI captures of
 * small regions on large pages.
 */
import type { FpdfDocument, PdfiumInstance } from '@truespar/lector-pdfium-wasm';
export interface CaptureRegionOptions {
    readonly pageIndex: number;
    /**
     * Region to capture, in PDF points relative to the page's top-left
     * corner with Y growing downward (the same coordinate space used by the
     * interaction plugin's `viewportToPage`). PDF's native bottom-left
     * coordinate system is not used here so that callers can pass marquee
     * rects directly without conversion.
     */
    readonly rect: {
        readonly x: number;
        readonly y: number;
        readonly width: number;
        readonly height: number;
    };
    /** Target DPI. Default 300. */
    readonly dpi?: number;
    /**
     * Page rotation to apply on top of the document's intrinsic rotation.
     * 0 = none, 1 = 90° CW, 2 = 180°, 3 = 270° CW. Default 0.
     */
    readonly rotation?: 0 | 1 | 2 | 3;
    /** Background fill colour as 0xAARRGGBB (pdfium format). Default opaque white. */
    readonly backgroundColor?: number;
}
export declare function captureRegion(pdfium: PdfiumInstance, docHandle: FpdfDocument, options: CaptureRegionOptions): Promise<ImageBitmap>;
//# sourceMappingURL=capture-ops.d.ts.map