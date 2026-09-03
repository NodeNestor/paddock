import type { FpdfDocument, FpdfFormHandle, PdfiumInstance } from '@truespar/lector-pdfium-wasm';
import type { AnnotationData } from '../data/types.js';
/** Read all annotations from a page, returning serializable AnnotationData[]. */
export declare function readPageAnnotations(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, formHandle?: FpdfFormHandle): AnnotationData[];
/** Create a new annotation on a page. Returns the created annotation data. */
export declare function createAnnotation(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, data: Partial<AnnotationData>): AnnotationData;
/** Update an existing annotation. Returns the updated annotation data. */
export declare function updateAnnotation(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, annotIndex: number, patch: Partial<AnnotationData>): AnnotationData;
/** Delete an annotation from a page. */
export declare function deleteAnnotation(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, annotIndex: number): void;
//# sourceMappingURL=annotation-ops.d.ts.map