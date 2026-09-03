/** Composable for undo/redo history. */
export declare function useHistory(): {
    canUndo: Readonly<import("vue").Ref<boolean, boolean>>;
    canRedo: Readonly<import("vue").Ref<boolean, boolean>>;
    undoLabel: Readonly<import("vue").Ref<string | null, string | null>>;
    redoLabel: Readonly<import("vue").Ref<string | null, string | null>>;
    undo: (docId: import("@truespar/lector-core").DocumentId) => Promise<void>;
    redo: (docId: import("@truespar/lector-core").DocumentId) => Promise<void>;
    clear: (docId: import("@truespar/lector-core").DocumentId) => void;
};
//# sourceMappingURL=use-history.d.ts.map