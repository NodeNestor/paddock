/** Page dimensions in PDF points (1 point = 1/72 inch). */
export interface PageSize {
    readonly width: number;
    readonly height: number;
}
/** Configuration for rendering a single page bitmap. */
export interface RenderOptions {
    /** Pdfium render flags (bitwise OR of FpdfRenderFlag values). */
    readonly flags?: number;
    /** Page rotation: 0 = 0deg, 1 = 90deg CW, 2 = 180deg, 3 = 270deg CW. */
    readonly rotation?: 0 | 1 | 2 | 3;
    /** Device pixel ratio for HiDPI rendering (e.g. 2 for Retina). */
    readonly devicePixelRatio?: number;
    /** Background fill color as 0xAARRGGBB (pdfium format). */
    readonly backgroundColor?: number;
}
/** Priority levels for the background render queue. */
export declare const RenderPriority: {
    /** Pages currently visible in the viewport. */
    readonly VISIBLE: 0;
    /** Pages in the pre-render buffer zone adjacent to visible pages. */
    readonly BUFFER: 1;
    /** Low-priority renders (thumbnails, prefetch). */
    readonly LOW: 2;
};
export type RenderPriority = (typeof RenderPriority)[keyof typeof RenderPriority];
/**
 * Sensible default render options.
 *
 * Flags: LCD_TEXT (0x02)
 * - LCD_TEXT: optimize text for LCD displays
 *
 * ANNOT (0x01) is intentionally NOT included. Lector renders all
 * annotations as DOM/SVG overlays via PageOverlayManager — including
 * highlights, underlines, ink, sticky notes, shapes, free text, and
 * stamps. If we also asked pdfium to render annotations into the
 * page bitmap, we'd get a baked-in pixel layer underneath the JS
 * overlays. The two are pixel-perfect on creation, so the user sees
 * one marker — but on delete, the overlay vanishes and the
 * baked-in pixel remains as a ghost. Disabling ANNOT here makes
 * the JS overlay layer the single source of truth for annotation
 * display, which matches the documented architecture intent.
 *
 * NOTE: REVERSE_BYTE_ORDER (0x10) is intentionally NOT included.
 * It causes ghost rendering artifacts on certain manipulated PDFs.
 * Lector handles BGRA-to-RGBA conversion explicitly in the render pipeline.
 */
export declare const DEFAULT_RENDER_OPTIONS: Readonly<Required<RenderOptions>>;
//# sourceMappingURL=render.d.ts.map