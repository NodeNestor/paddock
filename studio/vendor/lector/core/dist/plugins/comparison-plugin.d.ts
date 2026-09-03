import type { ReadonlySignal } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
import type { ComparisonResult } from '../worker/comparison-ops.js';
export type { ComparisonResult, PageDiff, ComparisonChange, PageComparisonMode, } from '../worker/comparison-ops.js';
/** Top-level state of the comparison feature. */
export type ComparisonState = 'inactive' | 'computing' | 'active' | 'error';
/**
 * Capability provided by the document comparison plugin.
 *
 * Comparison is a *mode* the user enters from the existing split-view:
 * once two distinct documents are loaded into the split, the user can
 * trigger `enter()` to compute a structured diff and activate the
 * comparison overlays. Calling `exit()` clears the overlays without
 * touching either document or the split state.
 */
export interface ComparisonCapability {
    /** Reactive top-level state. */
    readonly state: ReadonlySignal<ComparisonState>;
    /** The latest comparison result, or null when inactive. */
    readonly result: ReadonlySignal<ComparisonResult | null>;
    /** Last error message, if state === 'error'. */
    readonly error: ReadonlySignal<string | null>;
    /**
     * Document IDs the active comparison is bound to. Both null when
     * the plugin is in 'inactive'.
     */
    readonly activePair: ReadonlySignal<{
        docA: DocumentId;
        docB: DocumentId;
    } | null>;
    /**
     * Compute a comparison without entering compare mode. Useful for
     * background pre-computation, headless consumers, or tests. Both
     * documents must already be open.
     */
    compare(docA: DocumentId, docB: DocumentId): Promise<ComparisonResult>;
    /**
     * Compute a comparison and activate compare mode. Sets `state` to
     * 'computing' during the diff and 'active' when the result is in.
     * If a previous comparison was active it is replaced.
     */
    enter(docA: DocumentId, docB: DocumentId): Promise<void>;
    /**
     * Exit compare mode. Clears `result`, resets `state` to 'inactive',
     * and emits `comparison:exited`. The two documents themselves are
     * not closed.
     */
    exit(): void;
}
/**
 * Document comparison plugin.
 *
 * Wraps the worker-side `compareDocuments` op behind a small reactive
 * facade. Carries no UI of its own — the viewer subscribes to the
 * exposed signals to render comparison overlays, the change sidebar,
 * and synchronised scroll between split panes.
 *
 * Lifecycle is event-driven so other plugins can react:
 *   - `comparison:computing` — emitted when `enter()` starts
 *   - `comparison:entered`  — emitted with the result when active
 *   - `comparison:exited`   — emitted when `exit()` is called
 *   - `comparison:error`    — emitted with the error message
 */
export declare const comparisonPlugin: import("../index.js").PluginDefinition<ComparisonCapability, Record<string, never>>;
//# sourceMappingURL=comparison-plugin.d.ts.map