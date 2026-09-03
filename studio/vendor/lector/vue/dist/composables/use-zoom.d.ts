/**
 * Composable for zoom controls.
 *
 * @example
 * ```vue
 * <script setup>
 * const { level, zoomIn, zoomOut, fitWidth } = useZoom();
 * </script>
 * <template>
 *   <button @click="zoomOut">−</button>
 *   <span>{{ Math.round(level * 100) }}%</span>
 *   <button @click="zoomIn">+</button>
 * </template>
 * ```
 */
export declare function useZoom(): {
    level: Readonly<import("vue").Ref<number, number>>;
    fitMode: Readonly<import("vue").Ref<import("@truespar/lector-core").FitMode, import("@truespar/lector-core").FitMode>>;
    setLevel: (factor: number) => void;
    zoomIn: () => void;
    zoomOut: () => void;
    fitPage: () => void;
    fitWidth: () => void;
    resetZoom: () => void;
};
//# sourceMappingURL=use-zoom.d.ts.map