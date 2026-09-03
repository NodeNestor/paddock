import { type PropType } from 'vue';
import { LectorEngine, type LectorEngineOptions, type LectorViewerOptions, type LayoutMode, type ThemeMode, type PluginDefinition } from '@truespar/lector-core';
type AnyPlugin = PluginDefinition<any, any>;
/**
 * Drop-in PDF viewer component for Vue 3.
 *
 * Wraps `LectorEngine` and `LectorViewer` into a single Vue component.
 * Includes the full toolbar, sidebar, annotation tools, search, and all
 * other built-in UI. For a custom UI, use `provide(LECTOR_KEY, engine)`
 * with headless composables instead.
 *
 * CSS must be imported separately:
 * ```ts
 * import '\@truespar/lector-core/css/tokens.css';
 * import '\@truespar/lector-core/css/base.css';
 * ```
 *
 * @example
 * ```vue
 * <script setup>
 * import { LectorPdfViewer } from '@truespar/lector-vue';
 * const engineOptions = {
 *   wasmUrl: '/pdfium/pdfium.wasm',
 *   wasmJsUrl: '/pdfium/pdfium.js',
 *   workerUrl: new URL('./pdfium-worker.ts', import.meta.url),
 * };
 * </script>
 * <template>
 *   <LectorPdfViewer
 *     :engine-options="engineOptions"
 *     src="/document.pdf"
 *     theme="system"
 *     style="height: 100vh"
 *   />
 * </template>
 * ```
 */
export declare const LectorPdfViewer: import("vue").DefineComponent<import("vue").ExtractPropTypes<{
    /** Pre-created and initialized `LectorEngine`. */
    engine: {
        type: PropType<LectorEngine>;
        default: undefined;
    };
    /** Options to create a new engine internally. */
    engineOptions: {
        type: PropType<LectorEngineOptions>;
        default: undefined;
    };
    /** Plugins to register. Defaults to all built-in plugins. */
    plugins: {
        type: PropType<readonly AnyPlugin[]>;
        default: undefined;
    };
    /** PDF source: URL string, ArrayBuffer, or File. */
    src: {
        type: PropType<string | ArrayBuffer | File>;
        default: undefined;
    };
    /** Password for encrypted PDFs. */
    password: {
        type: StringConstructor;
        default: undefined;
    };
    /** Theme mode. */
    theme: {
        type: PropType<ThemeMode>;
        default: undefined;
    };
    /** Whether the sidebar starts open. */
    sidebarOpen: {
        type: BooleanConstructor;
        default: undefined;
    };
    /** Initial sidebar panel. */
    initialPanel: {
        type: StringConstructor;
        default: undefined;
    };
    /** Page layout mode. */
    layoutMode: {
        type: PropType<LayoutMode>;
        default: undefined;
    };
    /** Initial zoom level or fit mode. */
    initialZoom: {
        type: PropType<number | "fit-width" | "fit-page">;
        default: undefined;
    };
    /** Visible sidebar panels. */
    panels: {
        type: PropType<string[]>;
        default: undefined;
    };
    /** Allow opening local files from UI. */
    allowLocalOpen: {
        type: BooleanConstructor;
        default: boolean;
    };
    /**
     * Show the document tab bar. Defaults to true; pass false when the host
     * app has its own document switcher (see `LectorViewerOptions`).
     * `default: undefined` rather than `true` on purpose - Vue casts an ABSENT
     * Boolean prop to false, which would silently turn the bar off for every
     * embedder that never mentions it.
     */
    documentTabs: {
        type: BooleanConstructor;
        default: undefined;
    };
    /**
     * Host-contributed toolbar buttons, rendered by lector so they match its
     * own metrics (see `LectorViewerOptions.toolbarExtras`).
     */
    toolbarExtras: {
        type: PropType<NonNullable<LectorViewerOptions["toolbarExtras"]>>;
        default: undefined;
    };
    /**
     * Partial UI schema override, merged over `DEFAULT_UI_SCHEMA` — the same
     * option `LectorViewer` takes. Lets an embedding app trim or reorder the
     * toolbar without giving up the drop-in component.
     */
    uiSchema: {
        type: PropType<Record<string, unknown>>;
        default: undefined;
    };
}>, () => import("vue").VNode<import("vue").RendererNode, import("vue").RendererElement, {
    [key: string]: any;
}>, {}, {}, {}, import("vue").ComponentOptionsMixin, import("vue").ComponentOptionsMixin, {
    /** Fired once the engine is ready and viewer mounted. */
    ready: (_engine: LectorEngine) => true;
    /** Fired when a document is loaded. */
    'document-loaded': (_handle: unknown) => true;
    /** Fired on initialization or loading errors. */
    error: (_err: Error) => true;
}, string, import("vue").PublicProps, Readonly<import("vue").ExtractPropTypes<{
    /** Pre-created and initialized `LectorEngine`. */
    engine: {
        type: PropType<LectorEngine>;
        default: undefined;
    };
    /** Options to create a new engine internally. */
    engineOptions: {
        type: PropType<LectorEngineOptions>;
        default: undefined;
    };
    /** Plugins to register. Defaults to all built-in plugins. */
    plugins: {
        type: PropType<readonly AnyPlugin[]>;
        default: undefined;
    };
    /** PDF source: URL string, ArrayBuffer, or File. */
    src: {
        type: PropType<string | ArrayBuffer | File>;
        default: undefined;
    };
    /** Password for encrypted PDFs. */
    password: {
        type: StringConstructor;
        default: undefined;
    };
    /** Theme mode. */
    theme: {
        type: PropType<ThemeMode>;
        default: undefined;
    };
    /** Whether the sidebar starts open. */
    sidebarOpen: {
        type: BooleanConstructor;
        default: undefined;
    };
    /** Initial sidebar panel. */
    initialPanel: {
        type: StringConstructor;
        default: undefined;
    };
    /** Page layout mode. */
    layoutMode: {
        type: PropType<LayoutMode>;
        default: undefined;
    };
    /** Initial zoom level or fit mode. */
    initialZoom: {
        type: PropType<number | "fit-width" | "fit-page">;
        default: undefined;
    };
    /** Visible sidebar panels. */
    panels: {
        type: PropType<string[]>;
        default: undefined;
    };
    /** Allow opening local files from UI. */
    allowLocalOpen: {
        type: BooleanConstructor;
        default: boolean;
    };
    /**
     * Show the document tab bar. Defaults to true; pass false when the host
     * app has its own document switcher (see `LectorViewerOptions`).
     * `default: undefined` rather than `true` on purpose - Vue casts an ABSENT
     * Boolean prop to false, which would silently turn the bar off for every
     * embedder that never mentions it.
     */
    documentTabs: {
        type: BooleanConstructor;
        default: undefined;
    };
    /**
     * Host-contributed toolbar buttons, rendered by lector so they match its
     * own metrics (see `LectorViewerOptions.toolbarExtras`).
     */
    toolbarExtras: {
        type: PropType<NonNullable<LectorViewerOptions["toolbarExtras"]>>;
        default: undefined;
    };
    /**
     * Partial UI schema override, merged over `DEFAULT_UI_SCHEMA` — the same
     * option `LectorViewer` takes. Lets an embedding app trim or reorder the
     * toolbar without giving up the drop-in component.
     */
    uiSchema: {
        type: PropType<Record<string, unknown>>;
        default: undefined;
    };
}>> & Readonly<{
    onReady?: ((_engine: LectorEngine) => any) | undefined;
    "onDocument-loaded"?: ((_handle: unknown) => any) | undefined;
    onError?: ((_err: Error) => any) | undefined;
}>, {
    toolbarExtras: readonly {
        id: string;
        icon: string;
        tooltip: string;
        section?: "left" | "center" | "right";
        placement?: "start" | "end";
        onSelect: () => void;
        disabled?: () => boolean;
    }[];
    engine: LectorEngine;
    engineOptions: LectorEngineOptions;
    plugins: readonly AnyPlugin[];
    src: string | ArrayBuffer | File;
    password: string;
    theme: ThemeMode;
    sidebarOpen: boolean;
    initialPanel: string;
    layoutMode: LayoutMode;
    initialZoom: number | "fit-width" | "fit-page";
    panels: string[];
    allowLocalOpen: boolean;
    documentTabs: boolean;
    uiSchema: Record<string, unknown>;
}, {}, {}, {}, string, import("vue").ComponentProvideOptions, true, {}, any>;
export {};
//# sourceMappingURL=LectorPdfViewer.d.ts.map