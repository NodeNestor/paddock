import type { ReadonlySignal } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
import type { BookmarkNode, PageLink, WebLink, LinkTarget } from '../worker/navigation-ops.js';
/**
 * Capability provided by the navigation plugin.
 *
 * Manages bookmarks (table of contents), link annotations, web links,
 * and page-level navigation with history.
 */
export interface NavigationCapability {
    /**
     * Get the bookmark (outline / TOC) tree for a document.
     * Results are cached per document.
     */
    getBookmarks(docId: DocumentId): Promise<BookmarkNode[]>;
    /** Get all link annotations from a page. */
    getPageLinks(docId: DocumentId, pageIndex: number): Promise<PageLink[]>;
    /** Get auto-detected web links from a page. */
    getPageWebLinks(docId: DocumentId, pageIndex: number): Promise<WebLink[]>;
    /**
     * Navigate to a link target. Handles goto (internal), URI (external),
     * and other link types.
     */
    navigateToTarget(target: LinkTarget): void;
    /** Navigate to a specific page index. */
    goToPage(pageIndex: number): void;
    /** Navigate forward in the page history stack. */
    goForward(): void;
    /** Navigate backward in the page history stack. */
    goBack(): void;
    /** Whether back navigation is available. */
    canGoBack: ReadonlySignal<boolean>;
    /** Whether forward navigation is available. */
    canGoForward: ReadonlySignal<boolean>;
    /** Current page index (the most-visible page). */
    currentPage: ReadonlySignal<number>;
    /** Clear cached bookmark/link data for a document. */
    clearCache(docId: DocumentId): void;
}
/**
 * Navigation plugin.
 *
 * Provides bookmark tree access, link resolution, page navigation with
 * back/forward history, and web link detection.
 */
export declare const navigationPlugin: import("../index.js").PluginDefinition<NavigationCapability, Record<string, never>>;
//# sourceMappingURL=navigation-plugin.d.ts.map