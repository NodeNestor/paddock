import type { FpdfSignature, WasmPointer } from '../types/handles.js';
export declare const signatureDescriptor: {
    readonly FPDFSignatureObj_GetContents: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFSignatureObj_GetByteRange: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFSignatureObj_GetSubFilter: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFSignatureObj_GetReason: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFSignatureObj_GetTime: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFSignatureObj_GetDocMDPPermission: readonly ["number", readonly ["number"]];
};
export interface SignatureBindings {
    FPDFSignatureObj_GetContents(signature: FpdfSignature, buffer: WasmPointer, length: number): number;
    FPDFSignatureObj_GetByteRange(signature: FpdfSignature, buffer: WasmPointer, length: number): number;
    FPDFSignatureObj_GetSubFilter(signature: FpdfSignature, buffer: WasmPointer, length: number): number;
    FPDFSignatureObj_GetReason(signature: FpdfSignature, buffer: WasmPointer, length: number): number;
    FPDFSignatureObj_GetTime(signature: FpdfSignature, buffer: WasmPointer, length: number): number;
    FPDFSignatureObj_GetDocMDPPermission(signature: FpdfSignature): number;
}
//# sourceMappingURL=signature.d.ts.map