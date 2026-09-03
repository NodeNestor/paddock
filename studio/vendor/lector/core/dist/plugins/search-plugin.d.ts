import type { ReadonlySignal } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
import type { TextSearchMatch } from '../worker/text-ops.js';
/** Options for a search operation. */
export interface SearchOptions {
    /** Case-sensitive matching. Default: false. */
    readonly matchCase?: boolean;
    /** Match whole words only. Default: false. */
    readonly matchWholeWord?: boolean;
}
/** Result of a search across a document. */
export interface SearchResult {
    /**
     * Document the result belongs to. Required so consumers (overlay
     * managers in split-view, etc.) can scope match highlighting to the
     * pane that owns this doc — without it, matches from doc A would
     * also paint into doc B's pages at the same page index.
     */
    readonly docId: DocumentId;
    /** The search query that produced these results. */
    readonly query: string;
    /** All matches across all pages, ordered by page index. */
    readonly matches: TextSearchMatch[];
    /** Total number of matches found. */
    readonly totalCount: number;
}
/** Progress of an in-flight search operation. */
export interface SearchProgress {
    /** Pages searched so far. */
    readonly pagesSearched: number;
    /** Total pages in the document. */
    readonly totalPages: number;
    /** Matches found so far. */
    readonly matchesSoFar: number;
}
/**
 * Capability provided by the search plugin.
 *
 * Provides async full-text search across all pages with progress events,
 * match highlighting, and next/previous navigation.
 */
export interface SearchCapability {
    /** The current search result, or null if no search has been performed. */
    result: ReadonlySignal<SearchResult | null>;
    /** Index of the currently focused match, or -1 if none. */
    activeMatchIndex: ReadonlySignal<number>;
    /** Whether a search is currently in progress. */
    searching: ReadonlySignal<boolean>;
    /** Progress of the current search operation. */
    progress: ReadonlySignal<SearchProgress | null>;
    /** Start a new search. Cancels any in-progress search. */
    search(docId: DocumentId, query: string, options?: SearchOptions): Promise<SearchResult>;
    /** Navigate to the next match. Wraps around. */
    nextMatch(): void;
    /** Navigate to the previous match. Wraps around. */
    previousMatch(): void;
    /** Jump to a specific match by index. */
    goToMatch(index: number): void;
    /** Clear the current search results. */
    clear(): void;
}
/**
 * Search plugin.
 *
 * Searches across all pages of a document asynchronously, emitting progress
 * events and supporting cancellation via AbortController. Provides
 * next/previous match navigation for the UI layer.
 */
export declare const searchPlugin: import("../index.js").PluginDefinition<SearchCapability, Record<string, never>>;
//# sourceMappingURL=search-plugin.d.ts.map