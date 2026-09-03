import type { ReadonlySignal } from '@truespar/lector-utils';
import type { BreakpointConfig, BreakpointTier } from './types.js';
/** Default breakpoints (container-width based). */
export declare const DEFAULT_BREAKPOINTS: BreakpointConfig;
/**
 * Observes a container's width and emits the active breakpoint tier.
 *
 * Uses ResizeObserver on the container element — NOT window width.
 * This means embedded viewers in narrow panels get correct responsive behavior.
 */
export declare class BreakpointObserver implements Disposable {
    #private;
    readonly breakpoint: ReadonlySignal<BreakpointTier>;
    constructor(config?: BreakpointConfig);
    /** Start observing a container element's width. */
    observe(container: HTMLElement): void;
    /** Stop observing. */
    disconnect(): void;
    /** Update breakpoint thresholds at runtime. */
    setConfig(config: BreakpointConfig): void;
    [Symbol.dispose](): void;
}
/**
 * Determines whether a toolbar item's category is visible at the given breakpoint.
 *
 * - `compact` → only `'essential'`
 * - `medium`  → `'essential'` + `'standard'`
 * - `wide`    → all categories
 */
export declare function isCategoryVisible(category: 'essential' | 'standard' | 'extended' | undefined, tier: BreakpointTier): boolean;
//# sourceMappingURL=responsive.d.ts.map