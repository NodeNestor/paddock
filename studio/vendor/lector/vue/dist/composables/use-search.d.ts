/** Composable for full-text search. */
export declare function useSearch(): {
    result: Readonly<import("vue").Ref<{
        readonly docId: import("@truespar/lector-core").DocumentId;
        readonly query: string;
        readonly matches: readonly {
            readonly pageIndex: number;
            readonly charIndex: number;
            readonly length: number;
            readonly rects: readonly {
                readonly left: number;
                readonly top: number;
                readonly right: number;
                readonly bottom: number;
            }[];
        }[];
        readonly totalCount: number;
    } | null, {
        readonly docId: import("@truespar/lector-core").DocumentId;
        readonly query: string;
        readonly matches: readonly {
            readonly pageIndex: number;
            readonly charIndex: number;
            readonly length: number;
            readonly rects: readonly {
                readonly left: number;
                readonly top: number;
                readonly right: number;
                readonly bottom: number;
            }[];
        }[];
        readonly totalCount: number;
    } | null>>;
    activeMatchIndex: Readonly<import("vue").Ref<number, number>>;
    searching: Readonly<import("vue").Ref<boolean, boolean>>;
    progress: Readonly<import("vue").Ref<{
        readonly pagesSearched: number;
        readonly totalPages: number;
        readonly matchesSoFar: number;
    } | null, {
        readonly pagesSearched: number;
        readonly totalPages: number;
        readonly matchesSoFar: number;
    } | null>>;
    search: (docId: import("@truespar/lector-core").DocumentId, query: string, options?: import("@truespar/lector-core").SearchOptions) => Promise<import("@truespar/lector-core").SearchResult>;
    nextMatch: () => void;
    previousMatch: () => void;
    goToMatch: (index: number) => void;
    clear: () => void;
};
//# sourceMappingURL=use-search.d.ts.map