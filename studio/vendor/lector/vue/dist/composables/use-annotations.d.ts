/** Composable for annotation management. */
export declare function useAnnotations(): {
    selectedAnnotation: Readonly<import("vue").Ref<string | null, string | null>>;
    selectedAnnotations: Readonly<import("vue").Ref<readonly string[], readonly string[]>>;
    activeTool: Readonly<import("vue").Ref<import("@truespar/lector-core").AnnotationTool | null, import("@truespar/lector-core").AnnotationTool | null>>;
    toolStyle: Readonly<import("vue").Ref<{
        readonly color: {
            readonly r: number;
            readonly g: number;
            readonly b: number;
            readonly a: number;
        };
        readonly interiorColor: {
            readonly r: number;
            readonly g: number;
            readonly b: number;
            readonly a: number;
        } | null;
        readonly borderWidth: number;
        readonly fontSize: number;
        readonly opacity: number;
    }, {
        readonly color: {
            readonly r: number;
            readonly g: number;
            readonly b: number;
            readonly a: number;
        };
        readonly interiorColor: {
            readonly r: number;
            readonly g: number;
            readonly b: number;
            readonly a: number;
        } | null;
        readonly borderWidth: number;
        readonly fontSize: number;
        readonly opacity: number;
    }>>;
    lockMode: Readonly<import("vue").Ref<import("@truespar/lector-core").AnnotationLockMode, import("@truespar/lector-core").AnnotationLockMode>>;
    store: import("@truespar/lector-core").AnnotationStore;
    create: (docId: import("@truespar/lector-core").DocumentId, pageIndex: number, data: Partial<import("@truespar/lector-core").AnnotationData>) => Promise<import("@truespar/lector-core").TrackedObject<import("@truespar/lector-core").AnnotationData>>;
    update: (docId: import("@truespar/lector-core").DocumentId, annotationId: string, patch: Partial<import("@truespar/lector-core").AnnotationData>) => Promise<void>;
    delete: (docId: import("@truespar/lector-core").DocumentId, annotationId: string) => Promise<void>;
    loadPage: (docId: import("@truespar/lector-core").DocumentId, pageIndex: number) => Promise<void>;
    getForPage: (docId: import("@truespar/lector-core").DocumentId, pageIndex: number) => import("@truespar/lector-core").TrackedObject<import("@truespar/lector-core").AnnotationData>[];
    getForDocument: (docId: import("@truespar/lector-core").DocumentId) => import("@truespar/lector-core").TrackedObject<import("@truespar/lector-core").AnnotationData>[];
    selectAnnotation: (annotationId: string | null) => void;
    toggleAnnotationSelection: (annotationId: string) => void;
    clearAnnotationSelection: () => void;
    setActiveTool: (tool: import("@truespar/lector-core").AnnotationTool | null) => void;
    setToolStyle: (patch: Partial<import("@truespar/lector-core").AnnotationStyleState>) => void;
    setLockMode: (mode: import("@truespar/lector-core").AnnotationLockMode) => void;
    getDirty: (docId: import("@truespar/lector-core").DocumentId) => import("@truespar/lector-core").TrackedObject<import("@truespar/lector-core").AnnotationData>[];
    markSynced: (docId: import("@truespar/lector-core").DocumentId, annotationId: string) => void;
    markAllSynced: (docId: import("@truespar/lector-core").DocumentId) => void;
    subscribe: (fn: (event: import("@truespar/lector-core").DataEvent<import("@truespar/lector-core").AnnotationData>) => void) => import("@truespar/lector-utils").Unsubscribe;
    setCommentStatus: (docId: import("@truespar/lector-core").DocumentId, annotationId: string, status: import("@truespar/lector-core").CommentStatus) => Promise<void>;
    toggleResolved: (docId: import("@truespar/lector-core").DocumentId, annotationId: string) => Promise<void>;
    editComment: (docId: import("@truespar/lector-core").DocumentId, annotationId: string, commentId: string, newText: string) => Promise<void>;
    deleteComment: (docId: import("@truespar/lector-core").DocumentId, annotationId: string, commentId: string) => Promise<void>;
    bringToFront: (docId: import("@truespar/lector-core").DocumentId, annotationId: string) => Promise<void>;
    sendToBack: (docId: import("@truespar/lector-core").DocumentId, annotationId: string) => Promise<void>;
    canEdit: (docId: import("@truespar/lector-core").DocumentId, annotationId: string) => boolean;
    canDelete: (docId: import("@truespar/lector-core").DocumentId, annotationId: string) => boolean;
};
//# sourceMappingURL=use-annotations.d.ts.map