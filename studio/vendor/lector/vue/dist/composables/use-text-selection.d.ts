/** Composable for text selection and extraction. */
export declare function useTextSelection(): {
    selection: Readonly<import("vue").Ref<{
        readonly docId: import("@truespar/lector-core").DocumentId;
        readonly pageIndex: number;
        readonly startCharIndex: number;
        readonly endCharIndex: number;
        readonly text: string;
        readonly rects: readonly {
            readonly left: number;
            readonly top: number;
            readonly right: number;
            readonly bottom: number;
        }[];
    } | null, {
        readonly docId: import("@truespar/lector-core").DocumentId;
        readonly pageIndex: number;
        readonly startCharIndex: number;
        readonly endCharIndex: number;
        readonly text: string;
        readonly rects: readonly {
            readonly left: number;
            readonly top: number;
            readonly right: number;
            readonly bottom: number;
        }[];
    } | null>>;
    setSelection: (selection: import("@truespar/lector-core").TextSelection | null) => void;
    copySelection: () => Promise<void>;
    getPageText: (docId: import("@truespar/lector-core").DocumentId, pageIndex: number) => Promise<string>;
};
//# sourceMappingURL=use-text-selection.d.ts.map