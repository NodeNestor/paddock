import type { FpdfAttachment, FpdfDocument, WasmPointer } from '../types/handles.js';
export declare const attachmentDescriptor: {
    readonly FPDFDoc_GetAttachmentCount: readonly ["number", readonly ["number"]];
    readonly FPDFDoc_AddAttachment: readonly ["number", readonly ["number", "number"]];
    readonly FPDFDoc_GetAttachment: readonly ["number", readonly ["number", "number"]];
    readonly FPDFDoc_DeleteAttachment: readonly ["number", readonly ["number", "number"]];
    readonly FPDFAttachment_GetName: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFAttachment_HasKey: readonly ["number", readonly ["number", "number"]];
    readonly FPDFAttachment_GetValueType: readonly ["number", readonly ["number", "number"]];
    readonly FPDFAttachment_SetStringValue: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFAttachment_GetStringValue: readonly ["number", readonly ["number", "number", "number", "number"]];
    readonly FPDFAttachment_SetFile: readonly ["number", readonly ["number", "number", "number", "number"]];
    readonly FPDFAttachment_GetFile: readonly ["number", readonly ["number", "number", "number", "number"]];
    readonly FPDFAttachment_GetSubtype: readonly ["number", readonly ["number", "number", "number"]];
};
export interface AttachmentBindings {
    FPDFDoc_GetAttachmentCount(document: FpdfDocument): number;
    FPDFDoc_AddAttachment(document: FpdfDocument, name: WasmPointer): FpdfAttachment;
    FPDFDoc_GetAttachment(document: FpdfDocument, index: number): FpdfAttachment;
    FPDFDoc_DeleteAttachment(document: FpdfDocument, index: number): number;
    FPDFAttachment_GetName(attachment: FpdfAttachment, buffer: WasmPointer, buflen: number): number;
    FPDFAttachment_HasKey(attachment: FpdfAttachment, key: WasmPointer): number;
    FPDFAttachment_GetValueType(attachment: FpdfAttachment, key: WasmPointer): number;
    FPDFAttachment_SetStringValue(attachment: FpdfAttachment, key: WasmPointer, value: WasmPointer): number;
    FPDFAttachment_GetStringValue(attachment: FpdfAttachment, key: WasmPointer, buffer: WasmPointer, buflen: number): number;
    FPDFAttachment_SetFile(attachment: FpdfAttachment, document: FpdfDocument, contents: WasmPointer, len: number): number;
    FPDFAttachment_GetFile(attachment: FpdfAttachment, buffer: WasmPointer, buflen: number, outBuflen: WasmPointer): number;
    FPDFAttachment_GetSubtype(attachment: FpdfAttachment, buffer: WasmPointer, buflen: number): number;
}
//# sourceMappingURL=attachment.d.ts.map