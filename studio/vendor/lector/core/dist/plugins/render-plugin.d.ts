import type { DocumentId } from '../types/handle-id.js';
import type { RenderPriority } from '../types/render.js';
/**
 * Capability provided by the render plugin.
 *
 * Renders individual PDF pages to `ImageBitmap` via the engine's worker,
 * with priority scheduling and abort support.
 */
export interface RenderCapability {
    renderPage(docId: DocumentId, pageIndex: number, widthPx: number, heightPx: number, options?: {
        priority?: RenderPriority;
        signal?: AbortSignal;
        flags?: number;
        rotation?: 0 | 1 | 2 | 3;
    }): Promise<ImageBitmap>;
    reprioritize(docId: DocumentId, pageIndex: number, priority: RenderPriority): void;
}
/**
 * Page rendering plugin.
 *
 * Thin wrapper over `LectorEngine.renderPage()` that adds DPI-aware
 * convenience and delegates priority management to the engine's scheduler.
 */
export declare const renderPlugin: import("../index.js").PluginDefinition<RenderCapability, Record<string, never>>;
//# sourceMappingURL=render-plugin.d.ts.map