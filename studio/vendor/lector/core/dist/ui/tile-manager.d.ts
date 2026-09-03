/**
 * Tile-based rendering manager for high-zoom PDF pages.
 *
 * When a page is rendered at high zoom / DPI the full-page bitmap can exceed
 * GPU texture limits or consume hundreds of megabytes of memory. The
 * {@link TileManager} splits the page into a fixed-size grid of tiles, tracks
 * which tiles are visible in the current viewport, requests rendering only for
 * those tiles, and maintains an LRU cache of rendered {@link ImageBitmap}s.
 *
 * Tiles are keyed by `docId:pageIndex:scale:col:row` so a zoom change
 * automatically invalidates stale tiles without an explicit flush.
 *
 * @module
 */
/** Configuration for the tile grid and cache. */
export interface TileConfig {
    /** Tile size in CSS pixels. Default 512. */
    readonly tileSize?: number;
    /**
     * Overlap in pixels between adjacent tiles to prevent sub-pixel seams at
     * tile boundaries. Default 1.
     */
    readonly overlapPx?: number;
    /** Maximum number of tiles kept in the LRU cache. Default 256. */
    readonly maxCachedTiles?: number;
    /**
     * Total pixel-area threshold below which tile mode is NOT used (full-page
     * rendering is cheaper). Default `4096 * 4096 = 16_777_216`.
     */
    readonly tileThreshold?: number;
}
/**
 * A request issued to the consumer-provided render callback describing the
 * exact region of the page that should be rendered.
 */
export interface TileRequest {
    /** Document identifier. */
    readonly docId: string;
    /** Zero-based page index. */
    readonly pageIndex: number;
    /** Tile left edge in full-page pixel space. */
    readonly tileX: number;
    /** Tile top edge in full-page pixel space. */
    readonly tileY: number;
    /** Tile width in pixels. */
    readonly tileW: number;
    /** Tile height in pixels. */
    readonly tileH: number;
    /** Full page width in pixels (needed by pdfium to compute the crop matrix). */
    readonly fullW: number;
    /** Full page height in pixels. */
    readonly fullH: number;
    /** Scale at which this tile was requested. */
    readonly scale: number;
}
/**
 * Async function the consumer provides to actually render a tile region via
 * the worker pipeline. The returned {@link ImageBitmap} is cached by the
 * {@link TileManager}.
 */
export type RenderTileFn = (req: TileRequest) => Promise<ImageBitmap>;
/** Render state of a single tile. */
export type TileStatus = 'queued' | 'rendering' | 'ready';
/** Descriptor for one tile in the grid, returned by the manager. */
export interface TileDescriptor {
    /** Cache key (`docId:pageIndex:scale:col:row`). */
    readonly key: string;
    /** Column index in the tile grid. */
    readonly col: number;
    /** Row index in the tile grid. */
    readonly row: number;
    /** Left offset of this tile within the page element (CSS pixels). */
    readonly x: number;
    /** Top offset of this tile within the page element (CSS pixels). */
    readonly y: number;
    /** Width of this tile (CSS pixels). */
    readonly w: number;
    /** Height of this tile (CSS pixels). */
    readonly h: number;
    /** Current render state. */
    readonly status: TileStatus;
    /** Rendered bitmap. Only present when {@link status} is `'ready'`. */
    readonly bitmap?: ImageBitmap;
}
/** A simple axis-aligned rectangle used for viewport intersection tests. */
interface Rect {
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
}
/**
 * Manages a tile grid for rendering large PDF pages in smaller chunks.
 *
 * ## Typical usage
 *
 * ```ts
 * const tm = new TileManager({ tileSize: 512, maxCachedTiles: 256 });
 *
 * // On every scroll / zoom frame:
 * if (tm.shouldTile(fullW, fullH)) {
 *   const tiles = tm.updateVisibleTiles(
 *     docId, pageIndex, fullW, fullH, viewportRect, scale, renderTileFn,
 *   );
 *   for (const t of tiles) {
 *     if (t.status === 'ready') {
 *       ctx.drawImage(t.bitmap!, t.x, t.y, t.w, t.h);
 *     }
 *   }
 * }
 * ```
 */
export declare class TileManager {
    #private;
    /**
     * Optional callback invoked whenever an async tile render completes
     * successfully. The consumer (viewer) uses this to trigger a re-paint
     * so the newly-ready tile is drawn onto the canvas.
     */
    onTileReady: (() => void) | null;
    constructor(config?: TileConfig);
    /**
     * Determine whether a page at the given pixel dimensions should use tile
     * mode. Returns `false` when the total pixel area is below
     * {@link TileConfig.tileThreshold} (full-page rendering is cheaper).
     *
     * @param fullW - Full page width in pixels at the current scale and DPR.
     * @param fullH - Full page height in pixels.
     */
    shouldTile(fullW: number, fullH: number): boolean;
    /**
     * Compute visible tiles for a page and kick off rendering for any that are
     * not yet cached.
     *
     * Call this on every scroll or zoom event. The method is synchronous and
     * returns immediately with the current state of each visible tile. Render
     * callbacks are fired asynchronously; call this method again on the next
     * frame to pick up newly-ready tiles.
     *
     * @param docId      - Document identifier.
     * @param pageIndex  - Zero-based page index.
     * @param fullW      - Full page width in pixels (at current scale x DPR).
     * @param fullH      - Full page height in pixels.
     * @param viewportRect - Visible area within the page, in page-pixel coords.
     * @param scale      - Current zoom scale. Tiles are invalidated when this
     *                     changes because different scales produce different
     *                     bitmaps.
     * @param renderFn   - Async callback the consumer provides to render a tile
     *                     region via the worker pipeline.
     * @returns An array of {@link TileDescriptor}s for every tile that
     *          intersects the viewport.
     */
    updateVisibleTiles(docId: string, pageIndex: number, fullW: number, fullH: number, viewportRect: Rect, scale: number, renderFn: RenderTileFn): TileDescriptor[];
    /**
     * Return all tiles for a page at a specific scale whose bitmaps are ready
     * for drawing. Unlike {@link updateVisibleTiles} this does not issue any
     * new render requests — it only returns what is already cached.
     *
     * @param pageIndex - Zero-based page index.
     * @param scale     - Zoom scale to match.
     */
    getReadyTiles(pageIndex: number, scale: number): TileDescriptor[];
    /**
     * Remove all cached tiles for a specific page (all scales). Use this when
     * a page is scrolled far out of view and removed from the DOM.
     *
     * @param pageIndex - Zero-based page index.
     */
    clearPage(pageIndex: number): void;
    /**
     * Clear the entire tile cache. Call this when the document changes or the
     * engine is disposed.
     */
    clearAll(): void;
    /**
     * Dispose the tile manager, closing all cached {@link ImageBitmap}s and
     * releasing memory. The instance should not be used after calling this.
     */
    destroy(): void;
}
export {};
//# sourceMappingURL=tile-manager.d.ts.map