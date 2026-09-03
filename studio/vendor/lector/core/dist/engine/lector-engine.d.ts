import * as Comlink from 'comlink';
import type { DocumentId } from '../types/handle-id.js';
import type { PageSize, RenderOptions, RenderPriority } from '../types/render.js';
import type { PdfiumWorkerApi } from '../types/worker-api.js';
import { PluginRegistry } from '../plugin/registry.js';
import { RenderPool } from './render-pool.js';
import { PageRotationCache } from './page-rotation-cache.js';
/** User identity injected by the consumer application. */
export interface LectorUser {
    /** Unique user identifier (e.g., UUID, email). */
    readonly id: string;
    /** Display name shown on annotations and comments. */
    readonly name: string;
    /** Optional avatar URL. */
    readonly avatarUrl?: string;
}
/**
 * Resolves mentionable users for @mention autocomplete in comments.
 *
 * Called when the user types `@` in a comment textarea. The `query` is the
 * text after the `@` (may be empty). Return matching users.
 *
 * Can be a static array or an async function for backend lookups.
 */
export type MentionUserResolver = readonly LectorUser[] | ((query: string) => readonly LectorUser[] | Promise<readonly LectorUser[]>);
/** Document-level permission overrides. */
export interface DocumentPermissions {
    readonly canEdit?: boolean;
    readonly canDelete?: boolean;
    readonly canCreate?: boolean;
    readonly canSign?: boolean;
    readonly canRedact?: boolean;
    readonly canPrint?: boolean;
    readonly canExport?: boolean;
}
/** Custom stamp template for the stamp annotation tool. */
export interface StampTemplate {
    /** Internal stamp name (e.g., "Approved"). */
    readonly name: string;
    /** Display label (localized). */
    readonly label: string;
    /** Optional icon reference or SVG string. */
    readonly icon?: string;
    /** Custom appearance stream or data URI. */
    readonly appearance?: string;
}
/** Configuration for creating a LectorEngine instance. */
export interface LectorEngineOptions {
    /** URL of the pdfium.wasm binary (multi-threaded, requires COOP+COEP headers). */
    readonly wasmUrl: string;
    /** URL of the Emscripten JS loader (pdfium.js). */
    readonly wasmJsUrl: string;
    /**
     * Fallback URLs for environments without `crossOriginIsolated`
     * (no SharedArrayBuffer / pthreads). If provided and the page is not
     * cross-origin isolated, the engine automatically uses these instead.
     * Point to the single-threaded build: `pdfium-st.wasm` + `pdfium-st.js`.
     */
    readonly wasmUrlFallback?: string;
    /** Fallback JS loader for the single-threaded build (`pdfium-st.js`). */
    readonly wasmJsUrlFallback?: string;
    /** Optional URL for the worker script. Defaults to the bundled worker entry point. */
    readonly workerUrl?: string | URL;
    /** Current user identity. Used for annotation authorship and comments. */
    readonly user?: LectorUser;
    /**
     * Users available for @mention in comments.
     * Either a static list or an async resolver function.
     * If not provided, @mention autocomplete is disabled.
     */
    readonly mentionUsers?: MentionUserResolver;
    /** Initial UI translation locale for i18n. Defaults to 'en'. */
    readonly locale?: string;
    /** Custom translation overrides, keyed by locale ID. */
    readonly translations?: Readonly<Record<string, Readonly<Record<string, string>>>>;
    /**
     * BCP 47 locale tag used for date, number, file-size, and unit
     * formatting (e.g., `'sv-SE'`, `'de-DE'`). Independent of `locale` —
     * a user can read the UI in English while seeing dates and numbers in
     * Swedish format. Defaults to `navigator.language`, then `'en-US'`.
     */
    readonly formatLocale?: string;
    /**
     * Measurement system for length/area display.
     * - `'metric'` — mm, cm, m
     * - `'imperial'` — in, ft, yd
     *
     * Defaults to derivation from `formatLocale`'s region (US/LR/MM →
     * imperial, everything else → metric).
     */
    readonly measurementSystem?: 'metric' | 'imperial';
    /**
     * Force a specific hour-cycle for time display, overriding the locale
     * default. Use `'h23'` for a 24-hour clock or `'h12'` for a 12-hour
     * clock with AM/PM. Defaults to the locale's natural cycle.
     */
    readonly hourCycle?: 'h11' | 'h12' | 'h23' | 'h24';
    /**
     * Custom storage backend for the document manager's recent-files list.
     * Defaults to a `localStorage`-backed implementation. Apps that need
     * IndexedDB, server-side persistence, or custom encryption can supply
     * their own. Type is loose to avoid a circular import; the actual shape
     * is `RecentFilesStore` from `plugins/document-manager-plugin`.
     */
    readonly recentFilesStore?: unknown;
    /** Maximum entries kept in the recent-files list. Default 20. */
    readonly recentFilesMax?: number;
    /** localStorage key under which the recent-files list is persisted. */
    readonly recentFilesStorageKey?: string;
    /**
     * Whether the document manager should automatically register the viewer
     * container as a drop zone for PDF files. When enabled, the user can
     * drag any PDF onto the viewer to open it without app code wiring up
     * `registerDropZone` manually.
     *
     * Default: `true`. Set to `false` for embedded contexts where the host
     * app already handles drag-and-drop on the surrounding chrome.
     */
    readonly enableViewerDropZone?: boolean;
    /** Document-level permission overrides. */
    readonly permissions?: DocumentPermissions;
    /**
     * Named annotation style presets.
     * Keys are preset names; values are partial style objects.
     * Consumers can switch between presets via the annotation plugin.
     */
    readonly annotationPresets?: Readonly<Record<string, Record<string, unknown>>>;
    /**
     * Whether annotation tools stay active after creating one annotation.
     * When false (default), tools deactivate after each creation (one-shot UX).
     * When true, the tool stays active for rapid sequential annotation.
     */
    readonly keepSelectedTool?: boolean;
    /** Custom stamp templates for the stamp annotation tool. */
    readonly customStamps?: readonly StampTemplate[];
    /** Minimum zoom level (default 0.1 = 10%). */
    readonly zoomMin?: number;
    /** Maximum zoom level (default 10.0 = 1000%). */
    readonly zoomMax?: number;
    /** Zoom step multiplier for zoom-in/out (default 1.1 = 10% increments). */
    readonly zoomStep?: number;
    /** Gap in pixels between pages in continuous/multi-page layouts (default 8). */
    readonly pageGap?: number;
    /** Padding around the viewport edge in pixels (default 12). */
    readonly viewportPadding?: number;
    /**
     * Default annotation style applied to new annotations when no preset
     * is active. Partial — omitted fields use the built-in defaults
     * (red color, 2px border, 14pt font, 100% opacity).
     */
    readonly annotationDefaults?: {
        readonly color?: {
            readonly r: number;
            readonly g: number;
            readonly b: number;
            readonly a: number;
        };
        readonly borderWidth?: number;
        readonly fontSize?: number;
        readonly opacity?: number;
    };
    /** Device pixel ratio for rendering (default: window.devicePixelRatio). */
    readonly renderDpi?: number;
    /** Default DPI for page-to-image capture/export (default 300). */
    readonly captureDpi?: number;
    /**
     * Number of additional render workers for parallel page rendering.
     * Each worker loads its own pdfium WASM instance (~4MB extra memory).
     * Set to 0 (default) for single-worker mode, 2-4 for parallel rendering.
     * Higher values give faster concurrent render but use more memory.
     */
    readonly renderPoolSize?: number;
}
/**
 * A handle to an open PDF document.
 *
 * Implements Disposable for automatic cleanup via `using`.
 * Calling `close()` or disposing releases all worker-side resources.
 */
export interface DocumentHandle extends Disposable {
    /** Opaque identifier for this document. */
    readonly id: DocumentId;
    /** Total number of pages in the document. */
    readonly pageCount: number;
    /** Dimensions of every page in PDF points, indexed by page number. */
    readonly pageSizes: ReadonlyArray<PageSize>;
    /**
     * Hex-encoded SHA-256 of the original PDF bytes, computed once at
     * open time. Empty string for linearised documents that stream in
     * without a single contiguous buffer. Used by features like
     * comparison to short-circuit on byte-identical inputs.
     */
    readonly sha256: string;
    /** Close the document and free worker-side resources. */
    close(): Promise<void>;
    /** Synchronous dispose — schedules async cleanup. */
    [Symbol.dispose](): void;
}
/**
 * Main-thread entry point for the Lector PDF rendering engine.
 *
 * Creates a Web Worker running pdfium WASM, manages document lifecycle,
 * and dispatches render requests through a priority scheduler.
 *
 * Implements Disposable for automatic cleanup via `using`.
 *
 * ```ts
 * const engine = new LectorEngine({
 *   wasmUrl: '/pdfium.wasm',
 *   wasmJsUrl: '/pdfium.js',
 *   workerUrl: '/worker/pdfium-worker.js',
 * });
 * await engine.init();
 *
 * const doc = await engine.openDocument(pdfBytes);
 * const bitmap = await engine.renderPage(doc.id, 0, 800, 600);
 * // ... draw bitmap to canvas ...
 * bitmap.close();
 * doc.close();
 * engine[Symbol.dispose]();
 * ```
 */
export declare class LectorEngine implements Disposable {
    #private;
    /** Plugin registry — register plugins before calling init(). */
    readonly plugins: PluginRegistry;
    /** Current user identity, if provided by the consumer. */
    readonly user: LectorUser | undefined;
    /** Mention user resolver for @mention autocomplete. */
    readonly mentionUsers: MentionUserResolver | undefined;
    /** Initial UI translation locale for i18n. */
    readonly locale: string;
    /** Custom translation overrides. */
    readonly translations: LectorEngineOptions['translations'];
    /** BCP 47 format locale (or undefined to auto-detect from `navigator.language`). */
    readonly formatLocale: string | undefined;
    /** Measurement system override (or undefined to derive from `formatLocale`). */
    readonly measurementSystem: 'metric' | 'imperial' | undefined;
    /** Hour-cycle override (or undefined to use the locale default). */
    readonly hourCycle: 'h11' | 'h12' | 'h23' | 'h24' | undefined;
    /** Custom recent-files store, if provided. */
    readonly recentFilesStore: unknown;
    /** Max number of recent-files entries. */
    readonly recentFilesMax: number | undefined;
    /** localStorage key for recent files. */
    readonly recentFilesStorageKey: string | undefined;
    /** Auto-register the viewer container as a drop zone. */
    readonly enableViewerDropZone: boolean;
    /** Document-level permission overrides. */
    readonly permissions: DocumentPermissions | undefined;
    /** Whether annotation tools stay active after creation. */
    readonly keepSelectedTool: boolean;
    /** Custom stamp templates. */
    readonly customStamps: readonly StampTemplate[] | undefined;
    /** Named annotation style presets. */
    readonly annotationPresets: LectorEngineOptions['annotationPresets'];
    /** Zoom limits and step — consumed by the zoom plugin. */
    readonly zoomMin: number | undefined;
    readonly zoomMax: number | undefined;
    readonly zoomStep: number | undefined;
    /** Viewport spacing — consumed by the viewport plugin. */
    readonly pageGap: number | undefined;
    readonly viewportPadding: number | undefined;
    /** Annotation tool defaults — consumed by the annotation plugin. */
    readonly annotationDefaults: LectorEngineOptions['annotationDefaults'];
    /** Render quality — consumed by render and capture plugins. */
    readonly renderDpi: number | undefined;
    readonly captureDpi: number | undefined;
    constructor(options: LectorEngineOptions);
    /**
     * Initialize the engine: create the worker, load the WASM module.
     *
     * Must be called exactly once before any other method.
     * @throws {EngineError} with code NOT_INITIALIZED if init fails.
     */
    init(): Promise<void>;
    /**
     * Open a PDF document from a URL, fetching it with optional custom headers.
     *
     * @param url URL to fetch the PDF from.
     * @param options Fetch options (headers, credentials, etc.) and optional password.
     * @returns A DocumentHandle with metadata and a close() method.
     *
     * @example
     * ```ts
     * const doc = await engine.openDocumentFromUrl('/api/documents/123.pdf', {
     *   headers: { Authorization: `Bearer ${token}`, 'X-Team-Id': teamId },
     *   credentials: 'include',
     *   password: 'secret',
     * });
     * ```
     */
    openDocumentFromUrl(url: string, options?: RequestInit & {
        password?: string;
    }): Promise<DocumentHandle>;
    /**
     * Open a PDF document from raw bytes.
     *
     * The ArrayBuffer is transferred to the worker (zero-copy). After this call,
     * the provided ArrayBuffer is detached and cannot be used.
     *
     * @param data PDF file contents as an ArrayBuffer.
     * @param password Optional password for encrypted PDFs.
     * @returns A DocumentHandle with metadata and a close() method.
     */
    openDocument(data: ArrayBuffer, password?: string): Promise<DocumentHandle>;
    /**
     * Render a page to an ImageBitmap at the specified pixel dimensions.
     *
     * The returned ImageBitmap is transferred from the worker (zero-copy).
     * The caller owns the bitmap and must call `.close()` when done.
     *
     * @param docId Document identifier from a DocumentHandle.
     * @param pageIndex Zero-based page index.
     * @param width Target width in pixels.
     * @param height Target height in pixels.
     * @param options Render options including flags, rotation, DPI, and scheduling hints.
     * @returns The rendered page as an ImageBitmap.
     */
    renderPage(docId: DocumentId, pageIndex: number, width: number, height: number, options?: RenderOptions & {
        signal?: AbortSignal;
        priority?: RenderPriority;
    }): Promise<ImageBitmap>;
    /**
     * Render a rectangular tile of a PDF page. Used by the tile-based
     * rendering system for large pages at high zoom where allocating a
     * full-page bitmap would exceed memory limits.
     *
     * Bypasses the render scheduler (tiles have their own dedup/cancel
     * logic in TileManager) and calls the worker directly.
     */
    renderPageTile(docId: DocumentId, pageIndex: number, tileX: number, tileY: number, tileW: number, tileH: number, fullW: number, fullH: number, options?: RenderOptions): Promise<ImageBitmap>;
    /**
     * Access the Comlink proxy to the pdfium worker.
     *
     * This is intended for internal use by plugins that need direct access
     * to worker APIs beyond rendering (text extraction, navigation, annotations, etc.).
     *
     * @throws {EngineError} if the engine is not initialized.
     */
    get workerProxy(): Comlink.Remote<PdfiumWorkerApi>;
    /**
     * Render pool, if one is configured (via `renderPoolSize`). Plugins
     * that perform destructive page mutations on the primary worker must
     * also propagate the same op to the pool so the pool workers' pdfium
     * copies stay in sync; otherwise renders served from the pool will
     * show pre-mutation content.
     */
    get renderPool(): RenderPool | null;
    /**
     * Session-wide cache of per-page rotation. Used by the overlay, text, and
     * annotation layers to map coordinates correctly on rotated pages. The
     * overlay layer warms it for visible pages via `resolve()`; synchronous
     * consumers (annotation drawing) read it via `get()`.
     */
    get pageRotation(): PageRotationCache;
    /**
     * Change the priority of pending render tasks for a specific page.
     *
     * Tasks already actively rendering are not affected. This is typically
     * called by the viewport/render plugins when visible pages change.
     *
     * @param docId Document identifier.
     * @param pageIndex Zero-based page index.
     * @param priority New priority level.
     */
    reprioritize(docId: DocumentId, pageIndex: number, priority: RenderPriority): void;
    /**
     * Synchronous dispose — schedules async cleanup.
     *
     * Use this with the `using` keyword for automatic cleanup.
     * For awaitable cleanup, use `destroy()` instead.
     */
    [Symbol.dispose](): void;
    /**
     * Async dispose — waits for all cleanup to complete.
     *
     * Closes all documents, tears down pdfium, and terminates the worker.
     */
    destroy(): Promise<void>;
}
//# sourceMappingURL=lector-engine.d.ts.map