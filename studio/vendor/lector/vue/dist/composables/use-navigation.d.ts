/** Composable for page navigation. */
export declare function useNavigation(): {
    currentPage: Readonly<import("vue").Ref<number, number>>;
    canGoBack: Readonly<import("vue").Ref<boolean, boolean>>;
    canGoForward: Readonly<import("vue").Ref<boolean, boolean>>;
    goToPage: (pageIndex: number) => void;
    goBack: () => void;
    goForward: () => void;
    getBookmarks: (docId: import("@truespar/lector-core").DocumentId) => Promise<import("@truespar/lector-core").BookmarkNode[]>;
    navigateToTarget: (target: import("@truespar/lector-core").LinkTarget) => void;
};
//# sourceMappingURL=use-navigation.d.ts.map