import type { ReadonlySignal } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
import type { PdfJavaScriptAction } from '../worker/acroform-js-ops.js';
export interface AcroFormJSCapability {
    /** Whether the AcroForm JS engine is enabled. */
    readonly enabled: ReadonlySignal<boolean>;
    /** Enable or disable script execution. */
    setEnabled(enabled: boolean): void;
    /** Load document-level scripts into the sandbox. */
    loadDocumentScripts(docId: DocumentId): Promise<void>;
    /** Execute a format/calculate/validate script for a field. */
    executeFieldScript(script: string, fieldName: string, currentValue: string): Promise<string | undefined>;
    /** Get document-level JavaScript actions. */
    getActions(docId: DocumentId): Promise<PdfJavaScriptAction[]>;
    /** Dispose the JS runtime. */
    dispose(): Promise<void>;
}
export declare const acroformJSPlugin: import("../index.js").PluginDefinition<AcroFormJSCapability, Record<string, never>>;
//# sourceMappingURL=acroform-js-plugin.d.ts.map