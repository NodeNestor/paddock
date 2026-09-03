import type { FpdfDocument, FpdfJavaScriptAction, WasmPointer } from '../types/handles.js';
export declare const javascriptDescriptor: {
    readonly FPDFDoc_GetJavaScriptActionCount: readonly ["number", readonly ["number"]];
    readonly FPDFDoc_GetJavaScriptAction: readonly ["number", readonly ["number", "number"]];
    readonly FPDFDoc_CloseJavaScriptAction: readonly [null, readonly ["number"]];
    readonly FPDFJavaScriptAction_GetName: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFJavaScriptAction_GetScript: readonly ["number", readonly ["number", "number", "number"]];
};
export interface JavaScriptBindings {
    FPDFDoc_GetJavaScriptActionCount(document: FpdfDocument): number;
    FPDFDoc_GetJavaScriptAction(document: FpdfDocument, index: number): FpdfJavaScriptAction;
    FPDFDoc_CloseJavaScriptAction(javascript: FpdfJavaScriptAction): void;
    FPDFJavaScriptAction_GetName(javascript: FpdfJavaScriptAction, buffer: WasmPointer, buflen: number): number;
    FPDFJavaScriptAction_GetScript(javascript: FpdfJavaScriptAction, buffer: WasmPointer, buflen: number): number;
}
//# sourceMappingURL=javascript.d.ts.map