import type { EmscriptenArgType, EmscriptenReturnType, PdfiumModule } from '../types/module.js';
/**
 * A binding descriptor maps C function names to their cwrap type signatures.
 * Key: function name without underscore prefix (e.g. 'FPDF_InitLibrary')
 * Value: [returnType, argTypes]
 */
export type BindingDescriptor = Record<string, readonly [EmscriptenReturnType, readonly EmscriptenArgType[]]>;
/**
 * Create typed function bindings by calling cwrap for every entry in the descriptor.
 * Returns an object mapping function names to their bound JS functions.
 *
 * The returned object is typed generically at runtime. Each domain file provides
 * a concrete TypeScript interface that is applied via the PdfiumBindings intersection
 * type in lifecycle.ts.
 */
export declare function createBindings(module: PdfiumModule, descriptors: BindingDescriptor): Record<string, (...args: never[]) => unknown>;
//# sourceMappingURL=registry.d.ts.map