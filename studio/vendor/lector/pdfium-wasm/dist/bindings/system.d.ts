import type { FpdfSystemFontInfo, WasmPointer } from '../types/handles.js';
export declare const systemDescriptor: {
    readonly FSDK_SetTimeFunction: readonly [null, readonly ["number"]];
    readonly FSDK_SetLocaltimeFunction: readonly [null, readonly ["number"]];
    readonly FSDK_SetUnSpObjProcessHandler: readonly ["number", readonly ["number"]];
    readonly FPDF_SetSystemFontInfo: readonly [null, readonly ["number"]];
    readonly FPDF_GetDefaultSystemFontInfo: readonly ["number", readonly []];
    readonly FPDF_FreeDefaultSystemFontInfo: readonly [null, readonly ["number"]];
    readonly FPDF_AddInstalledFont: readonly [null, readonly ["number", "number", "number"]];
    readonly FPDF_GetDefaultTTFMap: readonly ["number", readonly []];
    readonly FPDF_GetDefaultTTFMapCount: readonly ["number", readonly []];
    readonly FPDF_GetDefaultTTFMapEntry: readonly ["number", readonly ["number"]];
};
export interface SystemBindings {
    FSDK_SetTimeFunction(func: WasmPointer): void;
    FSDK_SetLocaltimeFunction(func: WasmPointer): void;
    FSDK_SetUnSpObjProcessHandler(unspInfo: WasmPointer): number;
    FPDF_SetSystemFontInfo(fontInfo: FpdfSystemFontInfo): void;
    FPDF_GetDefaultSystemFontInfo(): FpdfSystemFontInfo;
    FPDF_FreeDefaultSystemFontInfo(fontInfo: FpdfSystemFontInfo): void;
    FPDF_AddInstalledFont(mapper: WasmPointer, face: WasmPointer, charset: number): void;
    FPDF_GetDefaultTTFMap(): WasmPointer;
    FPDF_GetDefaultTTFMapCount(): number;
    FPDF_GetDefaultTTFMapEntry(index: number): WasmPointer;
}
//# sourceMappingURL=system.d.ts.map