/** Composable for interactive PDF form management. */
export declare function useForm(): {
    readOnly: Readonly<import("vue").Ref<boolean, boolean>>;
    focusedField: Readonly<import("vue").Ref<string | null, string | null>>;
    store: import("@truespar/lector-core").FormStore;
    loadPage: (docId: import("@truespar/lector-core").DocumentId, pageIndex: number) => Promise<void>;
    getPageFields: (docId: import("@truespar/lector-core").DocumentId, pageIndex: number) => import("@truespar/lector-core").TrackedObject<import("@truespar/lector-core").WidgetData>[];
    getDocumentFields: (docId: import("@truespar/lector-core").DocumentId) => import("@truespar/lector-core").TrackedObject<import("@truespar/lector-core").WidgetData>[];
    getFieldValue: (docId: import("@truespar/lector-core").DocumentId, fieldName: string) => string | undefined;
    isPageLoaded: (docId: import("@truespar/lector-core").DocumentId, pageIndex: number) => boolean;
    setFieldValue: (docId: import("@truespar/lector-core").DocumentId, pageIndex: number, fieldName: string, value: string) => Promise<void>;
    populateFields: (docId: import("@truespar/lector-core").DocumentId, fields: ReadonlyArray<{
        pageIndex: number;
        fieldName: string;
        value: string;
    }>) => Promise<void>;
    extractFormData: (docId: import("@truespar/lector-core").DocumentId) => Record<string, string>;
    clickWidget: (docId: import("@truespar/lector-core").DocumentId, pageIndex: number, pageX: number, pageY: number) => Promise<import("@truespar/lector-core").WidgetData[]>;
    setReadOnly: (readOnly: boolean) => void;
    focusField: (fieldName: string | null) => void;
    hasDirty: (docId: import("@truespar/lector-core").DocumentId) => import("@truespar/lector-utils").ReadonlySignal<boolean>;
    markAllSynced: (docId: import("@truespar/lector-core").DocumentId) => void;
    subscribe: (fn: (event: import("@truespar/lector-core").DataEvent<import("@truespar/lector-core").WidgetData>) => void) => import("@truespar/lector-utils").Unsubscribe;
};
//# sourceMappingURL=use-form.d.ts.map