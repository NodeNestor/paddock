/**
 * Page coordinate transform — the single source of truth for mapping between
 * PDF user space and the CSS pixel space of a page's overlay layer.
 *
 * ## Why this exists
 *
 * PDF user space has its origin at the bottom-left with Y pointing up, measured
 * in points. The DOM overlay has its origin at the top-left with Y pointing
 * down, measured in CSS pixels. On top of that, a page can be rotated 0/90/180/
 * 270° — either intrinsically (the PDF `/Rotate` entry, common in scans) or by
 * the user. pdfium renders the rotated page, so the canvas is already correct;
 * the overlays (text selection, links, annotations, form widgets, search
 * highlights, ink) must use the *same* transform to stay aligned.
 *
 * Rather than scatter rotation branches across ~100 coordinate sites, the whole
 * mapping is encoded in one affine matrix per (page, rotation, scale). Every
 * forward mapping goes through {@link pointToCss}/{@link rectToCss}; every
 * inverse (drag, resize, hit-test) goes through {@link cssPointToPdf}/
 * {@link cssDeltaToPdf}. The inverse is derived from the forward matrix, so the
 * two can never disagree.
 *
 * The per-rotation forms below were verified empirically against real pdfium in
 * `__tests__/rotation-coords.test.ts` (the rendered glyph centroid matches the
 * predicted device position to 2 decimals at every rotation).
 */
/** Page rotation in 90° clockwise steps: 0=0°, 1=90°, 2=180°, 3=270°. */
export type PageRotation = 0 | 1 | 2 | 3;
/** A 2×3 affine matrix `[a, b, c, d, e, f]` (CSS/SVG `matrix()` convention). */
export type AffineMatrix = readonly [number, number, number, number, number, number];
/** A rectangle in PDF user space. Corners may be in any order. */
export interface PdfRect {
    readonly left: number;
    readonly top: number;
    readonly right: number;
    readonly bottom: number;
}
/** A point or delta. */
export interface Vec2 {
    readonly x: number;
    readonly y: number;
}
/** An axis-aligned box in CSS pixel space. */
export interface CssBox {
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
}
/**
 * Maps a page's PDF user space to its overlay CSS-pixel space, accounting for
 * scale, rotation, and the Y-axis flip.
 *
 * Construct from the page's *unrotated* dimensions plus rotation, or from the
 * *rotated* dimensions (what pdfium reports via `FPDF_GetPageSizeByIndexF`)
 * via {@link fromRotatedSize} — the latter is what overlay code has on hand.
 */
export declare class PageViewport {
    #private;
    /** Page width in unrotated PDF points. */
    readonly unrotatedWidthPts: number;
    /** Page height in unrotated PDF points. */
    readonly unrotatedHeightPts: number;
    readonly rotation: PageRotation;
    readonly scale: number;
    /** Overlay width in CSS px (rotated, scaled). */
    readonly width: number;
    /** Overlay height in CSS px (rotated, scaled). */
    readonly height: number;
    /** PDF user space → CSS px. `px = a·x + c·y + e`, `py = b·x + d·y + f`. */
    readonly matrix: AffineMatrix;
    /**
     * @param unrotatedWidthPts  Page width in PDF points, before rotation.
     * @param unrotatedHeightPts Page height in PDF points, before rotation.
     * @param rotation           Page rotation (0/1/2/3 = 0/90/180/270°).
     * @param scale              CSS px per PDF point.
     */
    constructor(unrotatedWidthPts: number, unrotatedHeightPts: number, rotation: PageRotation, scale: number);
    /**
     * Build a viewport from the page's *rotated* dimensions — i.e. the size
     * pdfium reports for the page in its current rotation. The unrotated size is
     * recovered by un-swapping width/height at 90°/270°.
     */
    static fromRotatedSize(rotatedWidthPts: number, rotatedHeightPts: number, rotation: PageRotation, scale: number): PageViewport;
    /** Map a PDF user-space point to CSS px within the overlay. */
    pointToCss(x: number, y: number): Vec2;
    /**
     * Map a PDF user-space rect to its CSS axis-aligned bounding box. All four
     * corners are transformed and min/max'd, so the result is correct regardless
     * of rotation or corner ordering. For orthogonal rotations an axis-aligned
     * PDF rect maps to an axis-aligned CSS rect with no distortion.
     */
    rectToCss(rect: PdfRect): CssBox;
    /**
     * Map a CSS-pixel point (relative to the page overlay's top-left) back to
     * PDF user space. Used for hit-testing — e.g. converting a click to the PDF
     * coordinate where a new annotation should be created.
     */
    cssPointToPdf(px: number, py: number): Vec2;
    /**
     * Map a CSS-pixel delta (no translation) back to a PDF user-space delta.
     * Used for drag/resize: a pointer movement of (dx, dy) px becomes the
     * corresponding shift in PDF points, with rotation and Y-flip applied.
     */
    cssDeltaToPdf(dx: number, dy: number): Vec2;
}
//# sourceMappingURL=page-viewport.d.ts.map