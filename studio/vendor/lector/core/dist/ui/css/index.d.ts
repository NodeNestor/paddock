/**
 * CSS utilities for Lector's design system.
 *
 * Framework bindings import these to inject Lector's styles.
 * The CSS is authored as plain `.css` files using `@layer lector`
 * so consumer styles always win without `!important`.
 *
 * Usage in framework bindings:
 * ```ts
 * import '@truespar/lector-core/ui/css/tokens.css';
 * import '@truespar/lector-core/ui/css/base.css';
 * ```
 *
 * Or use the helper to inject at runtime:
 * ```ts
 * import { injectLectorStyles } from '@truespar/lector-core';
 * injectLectorStyles();
 * ```
 */
import type { ThemeMode } from '../types.js';
/**
 * Inject Lector's CSS stylesheets into the document head at runtime.
 * Safe to call multiple times — only injects once.
 *
 * For build-time inclusion, import the CSS files directly instead:
 * ```ts
 * import '@truespar/lector-core/ui/css/tokens.css';
 * import '@truespar/lector-core/ui/css/base.css';
 * ```
 */
export declare function injectLectorStyles(): void;
/**
 * Build the CSS class list for the viewer root element.
 */
export declare function buildViewerClass(theme: ThemeMode, customClass?: string): string;
//# sourceMappingURL=index.d.ts.map