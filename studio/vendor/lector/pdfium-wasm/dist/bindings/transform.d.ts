import type { FpdfPage, WasmPointer } from '../types/handles.js';
export declare const transformDescriptor: {
    readonly FPDF_DeviceToPage: readonly ["number", readonly ["number", "number", "number", "number", "number", "number", "number", "number", "number", "number"]];
    readonly FPDF_PageToDevice: readonly ["number", readonly ["number", "number", "number", "number", "number", "number", "number", "number", "number", "number"]];
};
export interface TransformBindings {
    FPDF_DeviceToPage(page: FpdfPage, startX: number, startY: number, sizeX: number, sizeY: number, rotate: number, deviceX: number, deviceY: number, pageX: WasmPointer, pageY: WasmPointer): number;
    FPDF_PageToDevice(page: FpdfPage, startX: number, startY: number, sizeX: number, sizeY: number, rotate: number, pageX: number, pageY: number, deviceX: WasmPointer, deviceY: WasmPointer): number;
}
//# sourceMappingURL=transform.d.ts.map