/**
 * Composable for document lifecycle management.
 *
 * @example
 * ```vue
 * <script setup>
 * const { activeDocument, open, close } = useDocument();
 * </script>
 * ```
 */
export declare function useDocument(): {
    activeDocument: Readonly<import("vue").Ref<{
        readonly id: import("@truespar/lector-core").DocumentId;
        readonly pageCount: number;
        readonly pageSizes: readonly {
            readonly width: number;
            readonly height: number;
        }[];
        readonly sha256: string;
        readonly close: () => Promise<void>;
        readonly [Symbol.dispose]: () => void;
    } | null, {
        readonly id: import("@truespar/lector-core").DocumentId;
        readonly pageCount: number;
        readonly pageSizes: readonly {
            readonly width: number;
            readonly height: number;
        }[];
        readonly sha256: string;
        readonly close: () => Promise<void>;
        readonly [Symbol.dispose]: () => void;
    } | null>>;
    open: (source: ArrayBuffer | File | URL | string, password?: string) => Promise<import("@truespar/lector-core").DocumentHandle>;
    close: (docId: import("@truespar/lector-core").DocumentId) => Promise<void>;
    getHandle: (docId: import("@truespar/lector-core").DocumentId) => import("@truespar/lector-core").DocumentHandle | undefined;
    setActive: (docId: import("@truespar/lector-core").DocumentId) => void;
};
//# sourceMappingURL=use-document.d.ts.map