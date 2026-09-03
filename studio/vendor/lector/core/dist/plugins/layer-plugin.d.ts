import type { ReadonlySignal } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
import type { LayerInfo } from '../worker/layer-ops.js';
/**
 * Layer (OCG — Optional Content Group) capability.
 *
 * Reads and toggles the visibility of PDF layers. Layer changes take effect
 * on the next page render. The `layers` signal is reactive — UI components
 * can subscribe to it for automatic updates.
 */
export interface LayerCapability {
    /** All layers in the active document as a reactive signal. */
    layers: ReadonlySignal<readonly LayerInfo[]>;
    /** Whether the active document has layers. */
    hasLayers: ReadonlySignal<boolean>;
    /** Load layers from the worker for a document. */
    loadLayers(docId: DocumentId): Promise<void>;
    /** Toggle the visibility of a layer. Triggers re-render event. */
    setVisible(docId: DocumentId, layerIndex: number, visible: boolean): Promise<void>;
    /** Toggle all layers on or off. */
    setAllVisible(docId: DocumentId, visible: boolean): Promise<void>;
}
export declare const layerPlugin: import("../index.js").PluginDefinition<LayerCapability, Record<string, never>>;
//# sourceMappingURL=layer-plugin.d.ts.map