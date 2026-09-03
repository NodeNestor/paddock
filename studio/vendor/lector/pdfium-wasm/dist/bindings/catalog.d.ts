import type { FpdfDocument, WasmPointer } from '../types/handles.js';
export declare const catalogDescriptor: {
    readonly FPDFCatalog_IsTagged: readonly ["number", readonly ["number"]];
    readonly FPDFCatalog_GetLanguage: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFCatalog_SetLanguage: readonly ["number", readonly ["number", "number"]];
};
export interface CatalogBindings {
    FPDFCatalog_IsTagged(document: FpdfDocument): number;
    FPDFCatalog_GetLanguage(document: FpdfDocument, buffer: WasmPointer, buflen: number): number;
    FPDFCatalog_SetLanguage(document: FpdfDocument, language: WasmPointer): number;
}
//# sourceMappingURL=catalog.d.ts.map