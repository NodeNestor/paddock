import type { PluginDefinition } from '@truespar/lector-core';
type AnyPlugin = PluginDefinition<any, any>;
/** All built-in Lector plugins in the correct dependency order. */
export declare const ALL_PLUGINS: readonly AnyPlugin[];
/** Minimum viable plugin set for basic PDF viewing. */
export declare const CORE_PLUGINS: readonly AnyPlugin[];
/**
 * Read-only viewer preset: navigation, search, text selection, thumbnails
 * and document tabs — none of the editing machinery. Without the annotation,
 * form and signature plugins registered, their page layers never render and
 * their tools never appear, which no toolbar filtering can guarantee. For
 * embeddings where the viewer is a reading surface (a chat pane, a preview),
 * this is the honest preset.
 */
export declare const READER_PLUGINS: readonly AnyPlugin[];
export {};
//# sourceMappingURL=plugins.d.ts.map