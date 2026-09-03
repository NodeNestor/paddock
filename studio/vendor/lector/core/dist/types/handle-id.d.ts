/** Branded string identifying an open PDF document in the worker. */
export type DocumentId = string & {
    readonly __brand: 'DocumentId';
};
/** Branded string identifying a pending async operation. */
export type TaskId = string & {
    readonly __brand: 'TaskId';
};
//# sourceMappingURL=handle-id.d.ts.map