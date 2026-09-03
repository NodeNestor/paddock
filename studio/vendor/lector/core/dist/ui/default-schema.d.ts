import type { LectorUISchema } from './types.js';
/**
 * Default UI schema for the Lector PDF viewer.
 *
 * Toolbar order follows PDF viewer conventions (Adobe, Chrome, etc.):
 * Left:   sidebar toggle | page navigation
 * Center: zoom controls | fit modes
 * Right:  search | tools | more menu
 *
 * All user-visible strings reference i18n keys via `labelKey` /
 * `tooltipKey` so the schema can be rendered in any locale the host
 * application has wired up. Literal `label` / `tooltip` fields are
 * kept only where the i18n plugin types require a non-optional
 * literal (SidebarPanel.label), in which case they serve as the
 * English fallback for consumers that don't register the i18n plugin.
 */
export declare const DEFAULT_UI_SCHEMA: LectorUISchema;
/**
 * Deep-merge a partial schema over the default, producing a complete schema.
 *
 * Only top-level sections are merged — items arrays are replaced entirely
 * when provided in the override.
 */
export declare function mergeSchema(base: LectorUISchema, override: Partial<LectorUISchema>): LectorUISchema;
//# sourceMappingURL=default-schema.d.ts.map