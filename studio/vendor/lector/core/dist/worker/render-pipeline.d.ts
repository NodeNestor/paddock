import type { FpdfDocument, FpdfFormHandle, PdfiumInstance } from '@truespar/lector-pdfium-wasm';
import type { RenderOptions } from '../types/render.js';
/**
 * Render a single PDF page to an ImageBitmap via the pdfium WASM module.
 *
 * Pipeline:
 * 1. Load the page from the document
 * 2. Create a BGRA bitmap in pdfium's WASM heap
 * 3. Fill with the background color
 * 4. Render the page into the bitmap
 * 5. Copy pixels out of WASM heap (heap may be invalidated by later allocations)
 * 6. Swap BGRA to RGBA in-place on the copy
 * 7. Create ImageData, then ImageBitmap
 * 8. Clean up pdfium bitmap and page in a finally block
 *
 * CRITICAL NOTES:
 * - REVERSE_BYTE_ORDER (0x10) is NOT used. It causes ghost rendering on
 *   certain manipulated PDFs (e.g. Transportstyrelsen-forged.pdf page 1).
 *   Instead, we do an explicit BGRA-to-RGBA swap after rendering.
 * - Pixels MUST be copied out of the WASM heap before creating ImageData,
 *   because any subsequent WASM allocation could grow the heap and invalidate
 *   the underlying ArrayBuffer.
 * - The `0` arguments to FPDFBitmap_CreateEx for buffer/stride mean pdfium
 *   allocates and owns the bitmap buffer internally.
 */
/**
 * Render a single PDF page into a raw RGBA pixel buffer. Same pipeline
 * as `renderPageToImageBitmap` but stops one step short — it returns
 * the bytes instead of wrapping them in an ImageBitmap. Used by
 * worker-side consumers (e.g. comparison ops) that need to inspect
 * pixels directly without going through the canvas / ImageBitmap dance.
 *
 * The returned buffer is RGBA (8 bits per channel, premultiplied alpha
 * left as pdfium emits it) and is owned by the caller.
 */
export declare function renderPageToRgba(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, width: number, height: number, options?: Required<RenderOptions>, formHandle?: FpdfFormHandle): {
    width: number;
    height: number;
    rgba: Uint8Array;
};
/**
 * Render a rectangular tile of a PDF page to an ImageBitmap.
 *
 * Uses the same pdfium `FPDF_RenderPageBitmap` API but with negative
 * `start_x` / `start_y` offsets so the page is rendered at full
 * resolution but shifted so that only the tile region lands in the
 * (smaller) bitmap. This means pdfium only rasterises the visible
 * tile — it does NOT render the full page and crop.
 *
 * @param tileX - Left edge of the tile in full-page pixel space
 * @param tileY - Top edge of the tile in full-page pixel space
 * @param tileW - Width of the tile in pixels
 * @param tileH - Height of the tile in pixels
 * @param fullW - Full page width in pixels (at the target scale)
 * @param fullH - Full page height in pixels (at the target scale)
 */
export declare function renderPageTileToImageBitmap(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, tileX: number, tileY: number, tileW: number, tileH: number, fullW: number, fullH: number, options?: Required<RenderOptions>, formHandle?: FpdfFormHandle): Promise<ImageBitmap>;
export declare function renderPageToImageBitmap(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, width: number, height: number, options?: Required<RenderOptions>, formHandle?: FpdfFormHandle): Promise<ImageBitmap>;
//# sourceMappingURL=render-pipeline.d.ts.map