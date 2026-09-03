/**
 * Annotation creation tools — drawing handlers for each annotation type.
 *
 * Each tool processes pointer events in the interaction plugin's 'draw' mode,
 * collects geometry, and creates an annotation via the annotation plugin.
 */
import type { DocumentId } from '../types/handle-id.js';
import type { RgbaColor } from '../data/types.js';
import type { InteractionCapability } from './interaction-plugin.js';
import type { ViewportCapability } from './viewport-plugin.js';
import type { AnnotationCapability } from './annotation-plugin.js';
import type { TextLayerCapability } from './text-layer-plugin.js';
import type { HistoryCapability } from './history-plugin.js';
import type { DocumentCapability } from './document-plugin.js';
import type { FormattingCapability } from './formatting-plugin.js';
import { type PageRotation } from '../ui/page-viewport.js';
/** All available annotation creation tools. */
export type AnnotationTool = 'highlight' | 'underline' | 'strikeout' | 'squiggly' | 'ink' | 'ink-highlighter' | 'eraser' | 'freetext' | 'sticky-note' | 'insert-text' | 'callout' | 'rectangle' | 'circle' | 'line' | 'arrow' | 'polygon' | 'polyline' | 'stamp' | 'image' | 'measure-distance' | 'measure-area' | 'measure-perimeter' | 'redaction';
/** Default colors for annotation tools. */
export declare const ANNOTATION_TOOL_DEFAULTS: Record<AnnotationTool, {
    color: RgbaColor;
    opacity?: number;
}>;
/** Map tool names to pdfium subtypes. */
export declare const TOOL_TO_SUBTYPE: Record<AnnotationTool, number>;
/**
 * Per-tool behavior after annotation creation.
 * Configurable instead of hardcoded if/else.
 */
export interface ToolBehavior {
    /** Auto-select the annotation after creation. Default true. */
    selectAfterCreate: boolean;
    /** Deactivate the tool after creation (one-shot mode). Default true. */
    deactivateAfterCreate: boolean;
}
/** Default tool behaviors. Tool outputs (measurements, redactions) don't auto-select. */
export declare const TOOL_BEHAVIORS: Record<AnnotationTool, ToolBehavior>;
/** Whether a tool is a markup tool that requires text selection. */
export declare function isMarkupTool(tool: AnnotationTool): boolean;
/** Whether a tool draws with continuous strokes (ink-family). */
export declare function isInkTool(tool: AnnotationTool): boolean;
/** Whether this is the eraser tool. */
export declare function isEraserTool(tool: AnnotationTool): boolean;
/** Whether a tool creates shapes via drag (start → end). */
export declare function isShapeTool(tool: AnnotationTool): boolean;
/** Whether a tool creates a multi-vertex shape via click-to-add, dblclick-to-finish. */
export declare function isPolygonTool(tool: AnnotationTool): boolean;
/** Whether a tool places an annotation on click. */
export declare function isPlacementTool(tool: AnnotationTool): boolean;
/**
 * Whether a tool is the callout drawing tool.
 *
 * Callout uses a bespoke two-click UX:
 *   1. First click + drag = place the text rectangle
 *   2. Second click       = place the leader endpoint (where the arrow points)
 *
 * It is intentionally NOT a placement tool (which is single-click) and not
 * a shape tool (which uses click-and-drag for a single rect).
 */
export declare function isCalloutTool(tool: AnnotationTool): boolean;
/**
 * Whether a tool is the image-placement tool.
 *
 * Image has its own activation flow: when activated, the engine pops a file
 * picker so the user can choose an image. After the file is selected and
 * decoded into a data URI (held in `getStagedImage`), the next click on a
 * page places the image at that point. Until a file is selected, clicks on
 * the page do nothing — there is nothing to place.
 */
export declare function isImageTool(tool: AnnotationTool): boolean;
/** Whether a tool is a measurement tool. */
export declare function isMeasurementTool(tool: AnnotationTool): boolean;
/** Whether a tool is the stamp tool. */
export declare function isStampTool(tool: AnnotationTool): boolean;
/** Whether a tool is the redaction tool. */
export declare function isRedactionTool(tool: AnnotationTool): boolean;
/**
 * Whether a tool produces "tool output" annotations — ephemeral results
 * that are NOT user communication. Tool outputs:
 * - Don't auto-select after creation
 * - Show popover on click (for delete/properties) but don't open the comments sidebar
 * - Are hidden from the comments sidebar unless the user explicitly added a comment
 *
 * This covers measurement tools and redaction. Stamps/images are NOT tool
 * outputs — they're user-placed content that can carry comments.
 */
export declare function isToolOutputTool(tool: AnnotationTool): boolean;
/**
 * Whether an annotation (by its stored data) is a "tool output" — a result
 * from a tool that is not user communication. Used by the sidebar filter
 * and the selection effect to decide popover vs sidebar behavior.
 */
export declare function isToolOutputAnnotation(tag: string | undefined, subtype: number): boolean;
/**
 * Whether an annotation subtype represents user-relevant content that
 * should be rendered and shown in panels. This is a whitelist — only
 * explicitly listed subtypes pass. Everything else (links, widgets,
 * multimedia, production artifacts) is excluded.
 */
export declare function isUserAnnotation(subtype: number): boolean;
/** Mutable style configuration for the active tool. */
export interface AnnotationStyleState {
    color: RgbaColor;
    interiorColor: RgbaColor | null;
    borderWidth: number;
    fontSize: number;
    opacity: number;
}
/**
 * Context passed to the drawing handler from the annotation plugin.
 */
export interface DrawingContext {
    document: DocumentCapability;
    viewport: ViewportCapability;
    annotation: AnnotationCapability;
    textLayer: TextLayerCapability | null;
    history: HistoryCapability | null;
    interaction: InteractionCapability;
    /** Optional formatting capability — used for live measurement labels. */
    formatting: FormattingCapability | null;
    /**
     * Optional measurement capability getter. Called at draw time so the
     * lookup resolves *after* the measurement plugin has been set up
     * (annotation plugin runs first because measurement depends on it).
     * Returns null when the consumer hasn't registered the plugin.
     */
    getMeasurement: () => import('./measurement-plugin.js').MeasurementCapability | null;
    /**
     * Synchronously read a page's current rotation (0/1/2/3). Used to map
     * pointer positions to PDF coordinates correctly on rotated pages. Returns
     * the cached value (warmed by the overlay layer for visible pages); an
     * unresolved page reads as 0.
     */
    getPageRotation: (docId: DocumentId, pageIndex: number) => PageRotation;
    activeTool: () => AnnotationTool | null;
    style: () => AnnotationStyleState;
    emit: (event: string, ...args: unknown[]) => void;
    /** Returns the canvas scroll container for drawing previews. */
    getContainer: () => HTMLElement | null;
    /** Returns the currently selected stamp name (for stamp tool). */
    getStampName?: () => string;
    /**
     * Returns the data URI of the image staged for placement by the image
     * tool, plus its natural pixel dimensions, or null if no image has been
     * picked yet. The viewer's image-tool button is responsible for showing
     * the file picker, decoding the file, and stashing the result somewhere
     * this getter can read.
     */
    getStagedImage?: () => {
        dataUri: string;
        naturalWidth: number;
        naturalHeight: number;
    } | null;
    /**
     * Called when the image tool successfully consumes its staged image
     * (i.e. when the user clicks to place it). Lets the viewer clear the
     * staged image so a subsequent click does NOT place the same image
     * again — the user must pick a fresh file for each placement.
     */
    onImagePlaced?: () => void;
}
/**
 * Create and register the draw mode handler with the interaction plugin.
 *
 * Returns start/stop functions for the annotation tool system.
 */
export declare function createDrawModeHandler(ctx: DrawingContext): {
    activate: () => void;
    deactivate: () => void;
};
//# sourceMappingURL=annotation-tools.d.ts.map