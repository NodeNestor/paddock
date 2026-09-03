import type { ReadonlySignal } from '@truespar/lector-utils';
import type { RgbaColor } from '../data/types.js';
/**
 * A reusable annotation style template. Presets capture a subset of the
 * fields on `AnnotationStyleState` plus optional display metadata. Any
 * field left undefined is *not* applied when the preset is activated, so
 * a preset can target only color, only border width, etc.
 */
export interface AnnotationPreset {
    /** Stable identifier — must be unique across both builtin and user presets. */
    readonly name: string;
    /** Human-readable label. Falls back to `name` if absent. */
    readonly label?: string;
    /** Optional icon name (resolved through the icon catalog). */
    readonly icon?: string;
    /** Stroke / text colour. */
    readonly color?: RgbaColor;
    /** Fill colour for shape annotations. `null` means "no fill". */
    readonly interiorColor?: RgbaColor | null;
    /** Border width in PDF points. */
    readonly borderWidth?: number;
    /** Font size for free-text annotations. */
    readonly fontSize?: number;
    /** Per-annotation opacity (0–1). */
    readonly opacity?: number;
    /**
     * Whether this preset was supplied via engine config and is therefore
     * not user-deletable. User-saved presets are never builtin.
     */
    readonly builtin?: boolean;
}
/**
 * Capability provided by the annotation presets plugin.
 *
 * Manages a list of named style templates layered on top of the
 * annotation plugin's `toolStyle` signal. Activating a preset patches
 * the tool style and re-applies it on every tool change so the per-tool
 * defaults don't clobber the user's choice.
 */
export interface AnnotationPresetsCapability {
    /** Reactive list of all presets (builtin + user-saved). */
    readonly presets: ReadonlySignal<readonly AnnotationPreset[]>;
    /** Currently active preset name, or null. */
    readonly activePreset: ReadonlySignal<string | null>;
    /** Look up a preset by name. */
    getPreset(name: string): AnnotationPreset | undefined;
    /** Activate a preset by name (or clear with `null`). */
    setActivePreset(name: string | null): void;
    /** Save or replace a preset. User-supplied presets are persisted to localStorage. */
    savePreset(preset: AnnotationPreset): void;
    /** Delete a non-builtin preset by name. Returns false for builtin/missing. */
    deletePreset(name: string): boolean;
    /** Snapshot the current `toolStyle` into a new user preset. */
    saveCurrentAsPreset(name: string): AnnotationPreset | null;
}
/**
 * Annotation presets plugin.
 *
 * Sits above the annotation plugin and patches its `toolStyle` signal
 * whenever a preset is activated. Re-applies the active preset on every
 * `annotation:tool-changed` event so the per-tool default colours don't
 * silently overwrite a preset that's still in effect.
 *
 * Presets are sourced from two places:
 *  1. `engine.annotationPresets` config — flagged `builtin: true` and
 *     therefore non-deletable from the UI.
 *  2. The user's localStorage entry under `lector.annotationPresets.user`
 *     — created via `saveCurrentAsPreset()`.
 *
 * If both sources define the same name, the user entry wins (the user
 * intentionally edited the builtin).
 */
export declare const annotationPresetsPlugin: import("../index.js").PluginDefinition<AnnotationPresetsCapability, Record<string, never>>;
//# sourceMappingURL=annotation-presets-plugin.d.ts.map