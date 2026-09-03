import type { InjectionKey } from 'vue';
import type { LectorEngine } from '@truespar/lector-core';
/**
 * Vue injection key for the `LectorEngine` instance.
 *
 * Provided by `<LectorPdfViewer>` or manually via `provide(LECTOR_KEY, engine)`.
 * Consumed by composables like `useLector()`, `useDocument()`, etc.
 */
export declare const LECTOR_KEY: InjectionKey<LectorEngine>;
//# sourceMappingURL=context.d.ts.map