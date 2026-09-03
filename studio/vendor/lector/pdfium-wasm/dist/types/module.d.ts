import type { WasmPointer } from './handles.js';
import type { PdfiumExports } from '../generated/bindings.js';
/** Emscripten getValue/setValue type string. */
export type EmscriptenValueType = 'i8' | 'i16' | 'i32' | 'i64' | 'float' | 'double' | '*';
/** The Emscripten module object returned by createPdfiumModule(). */
export interface PdfiumModule extends PdfiumExports {
    getValue(ptr: number, type: EmscriptenValueType): number;
    setValue(ptr: number, value: number, type: EmscriptenValueType): void;
    UTF8ToString(ptr: number, maxBytesToRead?: number): string;
    UTF16ToString(ptr: number, maxBytesToRead?: number): string;
    stringToUTF8(str: string, outPtr: number, maxBytesToWrite: number): void;
    stringToUTF16(str: string, outPtr: number, maxBytesToWrite: number): void;
    HEAP8: Int8Array;
    HEAP16: Int16Array;
    HEAP32: Int32Array;
    HEAPU8: Uint8Array;
    HEAPU16: Uint16Array;
    HEAPU32: Uint32Array;
    HEAPF32: Float32Array;
    HEAPF64: Float64Array;
    _malloc: (size: number) => WasmPointer;
    _free: (ptr: number) => void;
}
/** Options for createPdfiumModule(). */
export interface PdfiumModuleConfig {
    locateFile?: (path: string, scriptDirectory: string) => string;
    onRuntimeInitialized?: () => void;
    print?: (text: string) => void;
    printErr?: (text: string) => void;
    /**
     * Emscripten hook to take over WASM instantiation.
     * When provided, Emscripten calls this instead of its default fetch+compile.
     * The callback must call `receiveInstance(instance, module?)` to hand the
     * instantiated module back to Emscripten, and return the exports object.
     */
    instantiateWasm?: (imports: WebAssembly.Imports, receiveInstance: (instance: WebAssembly.Instance, module?: WebAssembly.Module) => void) => WebAssembly.Exports;
    /** URL of the main script (used by Emscripten for worker spawning). */
    mainScriptUrlOrBlob?: string;
}
//# sourceMappingURL=module.d.ts.map