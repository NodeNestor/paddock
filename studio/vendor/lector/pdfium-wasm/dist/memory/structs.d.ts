import type { WasmPointer } from '../types/handles.js';
import type { PdfiumModule } from '../types/module.js';
import type { FsMatrix, FsPointF, FsQuadPointsF, FsRectF, FsSizeF } from '../types/structs.js';
export declare function readRectF(module: PdfiumModule, ptr: WasmPointer): FsRectF;
export declare function writeRectF(module: PdfiumModule, ptr: WasmPointer, rect: FsRectF): void;
export declare function readMatrix(module: PdfiumModule, ptr: WasmPointer): FsMatrix;
export declare function writeMatrix(module: PdfiumModule, ptr: WasmPointer, matrix: FsMatrix): void;
export declare function readSizeF(module: PdfiumModule, ptr: WasmPointer): FsSizeF;
export declare function writeSizeF(module: PdfiumModule, ptr: WasmPointer, size: FsSizeF): void;
export declare function readPointF(module: PdfiumModule, ptr: WasmPointer): FsPointF;
export declare function writePointF(module: PdfiumModule, ptr: WasmPointer, point: FsPointF): void;
export declare function readQuadPointsF(module: PdfiumModule, ptr: WasmPointer): FsQuadPointsF;
export declare function writeQuadPointsF(module: PdfiumModule, ptr: WasmPointer, qp: FsQuadPointsF): void;
//# sourceMappingURL=structs.d.ts.map