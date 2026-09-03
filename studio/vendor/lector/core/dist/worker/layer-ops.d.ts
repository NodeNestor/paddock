/**
 * Layer (OCG — Optional Content Group) operations.
 *
 * Wraps custom pdfium C API extensions that read and toggle PDF layer
 * visibility. These functions were added to our pdfium WASM build via
 * build/custom/fpdf_ocg.cpp.
 */
import type { PdfiumInstance, FpdfDocument } from '@truespar/lector-pdfium-wasm';
/** Information about a single PDF layer (Optional Content Group). */
export interface LayerInfo {
    /** Zero-based index in the document's OCG list. */
    readonly index: number;
    /** Display name of the layer. */
    readonly name: string;
    /** Intent (e.g., "View", "Design", or empty). */
    readonly intent: string;
    /** Whether the layer is currently visible in the default configuration. */
    readonly visible: boolean;
}
/** Get the number of layers (OCGs) in a document. Returns 0 if none. */
export declare function getLayerCount(pdfium: PdfiumInstance, docHandle: FpdfDocument): number;
/** Read information about all layers in the document. */
export declare function getAllLayers(pdfium: PdfiumInstance, docHandle: FpdfDocument): LayerInfo[];
/** Read information about a specific layer by index. */
export declare function getLayerInfo(pdfium: PdfiumInstance, docHandle: FpdfDocument, index: number): LayerInfo;
/** Set the visibility of a layer. */
export declare function setLayerVisible(pdfium: PdfiumInstance, docHandle: FpdfDocument, index: number, visible: boolean): void;
//# sourceMappingURL=layer-ops.d.ts.map