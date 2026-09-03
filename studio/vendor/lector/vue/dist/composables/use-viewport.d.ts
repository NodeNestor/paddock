/**
 * Composable for viewport management.
 *
 * Use the returned `attach`/`detach` methods to connect a container element,
 * or use a template ref callback.
 *
 * @example
 * ```vue
 * <script setup>
 * const containerRef = ref<HTMLElement>();
 * const { visiblePages, attach, detach } = useViewport();
 *
 * onMounted(() => { if (containerRef.value) attach(containerRef.value); });
 * onBeforeUnmount(() => { detach(); });
 * </script>
 * <template>
 *   <div ref="containerRef" style="height: 100%; overflow: auto" />
 * </template>
 * ```
 */
export declare function useViewport(): {
    visiblePages: Readonly<import("vue").Ref<readonly number[], readonly number[]>>;
    scale: Readonly<import("vue").Ref<number, number>>;
    layoutMode: Readonly<import("vue").Ref<import("@truespar/lector-core").LayoutMode, import("@truespar/lector-core").LayoutMode>>;
    pagePositions: Readonly<import("vue").Ref<readonly {
        readonly pageIndex: number;
        readonly x: number;
        readonly y: number;
        readonly width: number;
        readonly height: number;
    }[], readonly {
        readonly pageIndex: number;
        readonly x: number;
        readonly y: number;
        readonly width: number;
        readonly height: number;
    }[]>>;
    totalHeight: Readonly<import("vue").Ref<number, number>>;
    containerSize: Readonly<import("vue").Ref<{
        readonly width: number;
        readonly height: number;
    }, {
        readonly width: number;
        readonly height: number;
    }>>;
    attach: (container: HTMLElement) => void;
    detach: () => void;
    scrollToPage: (pageIndex: number, smooth?: boolean) => void;
    setLayoutMode: (mode: import("@truespar/lector-core").LayoutMode) => void;
    setScale: (scale: number) => void;
    setBufferSize: (pages: number) => void;
};
//# sourceMappingURL=use-viewport.d.ts.map