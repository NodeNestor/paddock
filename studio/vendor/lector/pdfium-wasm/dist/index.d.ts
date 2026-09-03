export { createPdfiumInstance } from './lifecycle.js';
export type { PdfiumInstance, PdfiumMemory } from './lifecycle.js';
export type { PdfiumExports } from './generated/bindings.js';
export type { EmscriptenValueType, PdfiumModule, PdfiumModuleConfig, } from './types/module.js';
export type { FpdfAction, FpdfAnnotation, FpdfAttachment, FpdfAvail, FpdfBitmap, FpdfBookmark, FpdfClipPath, FpdfDest, FpdfDocument, FpdfFont, FpdfFormHandle, FpdfGlyphPath, FpdfJavaScriptAction, FpdfLink, FpdfPage, FpdfPageLink, FpdfPageObject, FpdfPageObjectMark, FpdfPageRange, FpdfPathSegment, FpdfSchHandle, FpdfSignature, FpdfStructElement, FpdfStructElementAttr, FpdfStructElementAttrValue, FpdfStructTree, FpdfSystemFontInfo, FpdfTextPage, FpdfWidget, FpdfXObject, WasmPointer, } from './types/handles.js';
export { FpdfActionType, FpdfAnnotAAction, FpdfAnnotAppearanceMode, FpdfAnnotColorType, FpdfAnnotFlag, FpdfAnnotSubtype, FpdfBitmapFormat, FpdfDuplexType, FpdfError, FpdfFillMode, FpdfFlattenResult, FpdfFormFieldType, FpdfFormType, FpdfLineCap, FpdfLineJoin, FpdfObjectType, FpdfPageMode, FpdfPageObjectType, FpdfRenderFlag, FpdfSegmentType, FpdfTextRenderMode, FpdfUnsupportedType, } from './types/enums.js';
export type { FsMatrix, FsPointF, FsQuadPointsF, FsRectF, FsSizeF, } from './types/structs.js';
export { FS_MATRIX_SIZE, FS_POINTF_SIZE, FS_QUADPOINTSF_SIZE, FS_RECTF_SIZE, FS_SIZEF_SIZE, } from './types/structs.js';
export { checkBool, checkHandle, PdfiumError } from './error.js';
export { WasmAlloc, wasmAlloc, wasmFree, wasmMalloc } from './memory/allocator.js';
export { fromHeap, heapView, toHeap } from './memory/buffer.js';
export { fromByteString, fromWideString, toByteString, toWideString, } from './memory/strings.js';
export { readMatrix, readPointF, readQuadPointsF, readRectF, readSizeF, writeMatrix, writePointF, writeQuadPointsF, writeRectF, writeSizeF, } from './memory/structs.js';
//# sourceMappingURL=index.d.ts.map