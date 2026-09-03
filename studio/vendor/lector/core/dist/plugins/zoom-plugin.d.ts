import type { ReadonlySignal } from '@truespar/lector-utils';
/** How the zoom level is determined. */
export type FitMode = 'none' | 'page' | 'width';
/**
 * Capability provided by the zoom plugin.
 *
 * Controls the zoom level and fit mode, integrating with the viewport
 * plugin to keep page scale in sync.
 */
export interface ZoomCapability {
    level: ReadonlySignal<number>;
    fitMode: ReadonlySignal<FitMode>;
    setLevel(factor: number): void;
    zoomIn(): void;
    zoomOut(): void;
    fitPage(): void;
    fitWidth(): void;
    resetZoom(): void;
}
/**
 * Zoom plugin.
 *
 * Manages zoom level and fit modes. Integrates with the viewport plugin
 * by updating its scale whenever the zoom level changes, and recalculates
 * fit-based zoom levels when the container size changes.
 *
 * Registers four keyboard commands: zoom.in, zoom.out, zoom.fit-page, zoom.fit-width.
 */
export declare const zoomPlugin: import("../index.js").PluginDefinition<ZoomCapability, Record<string, never>>;
//# sourceMappingURL=zoom-plugin.d.ts.map