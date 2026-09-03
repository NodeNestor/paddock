import type { ReadonlySignal } from '@truespar/lector-utils';
/** Commit state for dirty tracking. */
export type CommitState = 'new' | 'dirty' | 'synced' | 'deleted';
/** Color (0-255 per channel). */
export interface RgbaColor {
    readonly r: number;
    readonly g: number;
    readonly b: number;
    readonly a: number;
}
/** Border style for annotations. */
export interface BorderStyle {
    readonly horizontalRadius: number;
    readonly verticalRadius: number;
    readonly width: number;
}
/** Line ending styles for line/polyline/callout annotations. */
export declare const LineCap: {
    readonly NONE: 0;
    readonly SQUARE: 1;
    readonly CIRCLE: 2;
    readonly DIAMOND: 3;
    readonly OPEN_ARROW: 4;
    readonly CLOSED_ARROW: 5;
    readonly BUTT: 6;
    readonly REVERSE_OPEN_ARROW: 7;
    readonly REVERSE_CLOSED_ARROW: 8;
    readonly SLASH: 9;
};
export type LineCap = (typeof LineCap)[keyof typeof LineCap];
/** PDF blend modes for annotation rendering. */
export declare const BlendMode: {
    readonly NORMAL: "Normal";
    readonly MULTIPLY: "Multiply";
    readonly SCREEN: "Screen";
    readonly OVERLAY: "Overlay";
    readonly DARKEN: "Darken";
    readonly LIGHTEN: "Lighten";
    readonly COLOR_DODGE: "ColorDodge";
    readonly COLOR_BURN: "ColorBurn";
    readonly HARD_LIGHT: "HardLight";
    readonly SOFT_LIGHT: "SoftLight";
    readonly DIFFERENCE: "Difference";
    readonly EXCLUSION: "Exclusion";
};
export type BlendMode = (typeof BlendMode)[keyof typeof BlendMode];
/** Note icon types for sticky-note (TEXT subtype) annotations. */
export declare const NoteIcon: {
    readonly COMMENT: "Comment";
    readonly RIGHT_POINTER: "RightPointer";
    readonly RIGHT_ARROW: "RightArrow";
    readonly CHECK: "Check";
    readonly CIRCLE: "Circle";
    readonly CROSS: "Cross";
    readonly INSERT: "Insert";
    readonly NEW_PARAGRAPH: "NewParagraph";
    readonly NOTE: "Note";
    readonly PARAGRAPH: "Paragraph";
    readonly HELP: "Help";
    readonly STAR: "Star";
    readonly KEY: "Key";
};
/** NoteIcon is extensible — consumers can pass any string for custom icons. */
export type NoteIcon = (typeof NoteIcon)[keyof typeof NoteIcon] | string;
/** Measurement units for distance/area/perimeter annotations. */
export declare const MeasurementUnit: {
    readonly PT: "pt";
    readonly MM: "mm";
    readonly IN: "in";
    readonly CM: "cm";
    readonly M: "m";
    readonly FT: "ft";
    readonly YD: "yd";
};
export type MeasurementUnit = (typeof MeasurementUnit)[keyof typeof MeasurementUnit];
/** Markup text annotation data (highlight, underline, squiggly, strikeout). */
export interface MarkupData {
    readonly quadPoints: ReadonlyArray<{
        readonly x1: number;
        readonly y1: number;
        readonly x2: number;
        readonly y2: number;
        readonly x3: number;
        readonly y3: number;
        readonly x4: number;
        readonly y4: number;
    }>;
}
/** Ink (freehand drawing) annotation data. */
export interface InkData {
    /**
     * Strokes are flat arrays of points. Each point has PDF coordinates
     * and an optional `pressure` value in [0, 1]. Pressure is only set
     * when the stroke was captured with `pointerType === 'pen'` (real
     * stylus / pen tablet); mouse and touch strokes leave it undefined,
     * which the renderer treats as a constant-width stroke.
     */
    readonly strokes: ReadonlyArray<ReadonlyArray<{
        readonly x: number;
        readonly y: number;
        readonly pressure?: number;
    }>>;
}
/** Line annotation data. */
export interface LineData {
    readonly start: {
        readonly x: number;
        readonly y: number;
    };
    readonly end: {
        readonly x: number;
        readonly y: number;
    };
}
/** Free text annotation data. */
export interface FreeTextData {
    readonly text: string;
    readonly fontSize: number;
    readonly fontColor?: {
        readonly r: number;
        readonly g: number;
        readonly b: number;
    };
    readonly textAlign?: 'left' | 'center' | 'right';
}
/** Stamp annotation data. */
export interface StampData {
    /** Standard stamp name (e.g., "Approved", "Draft", "Confidential", "AsIs"). */
    readonly name: string;
    /** Custom appearance stream reference or data URI for non-standard stamps. */
    readonly customAppearance?: string;
}
/** Redaction annotation data. */
export interface RedactionData {
    /** Reason for the redaction (shown in redaction log). */
    readonly reason?: string;
    /** Text displayed over the redacted area after applying. */
    readonly overlayText?: string;
    /** Font size for overlay text. */
    readonly overlayFontSize?: number;
    /** Color of the overlay text. */
    readonly overlayColor?: RgbaColor;
    /** Whether the redaction has been permanently applied. */
    readonly applied?: boolean;
}
/** Image annotation data. */
export interface ImageAnnotationData {
    /** Reference ID or data URI for the image content. */
    readonly imageRef: string;
    /** Display width in PDF points. */
    readonly width: number;
    /** Display height in PDF points. */
    readonly height: number;
    /** Natural image width in pixels (before scaling). */
    readonly naturalWidth?: number;
    /** Natural image height in pixels (before scaling). */
    readonly naturalHeight?: number;
}
/** Signature annotation data (for signature form fields and visual signatures). */
export interface SignatureData {
    readonly signerName?: string;
    readonly reason?: string;
    readonly location?: string;
    readonly contactInfo?: string;
    readonly signedAt?: string;
    readonly certificateSubject?: string;
    readonly certificateIssuer?: string;
    /** Whether the field has been signed. */
    readonly isSigned: boolean;
}
/** Measurement annotation data (distance, area, perimeter). */
export interface MeasurementData {
    readonly type: 'distance' | 'area' | 'perimeter';
    /** Calculated value in the target unit. */
    readonly value: number;
    /** Display unit. */
    readonly unit: MeasurementUnit;
    /** Scale mapping from PDF points to real-world units. */
    readonly scale?: {
        readonly source: number;
        readonly sourceUnit: MeasurementUnit;
        readonly target: number;
        readonly targetUnit: MeasurementUnit;
    };
    /** Decimal precision for display. */
    readonly precision?: number;
}
/** Callout annotation data (text with leader line). */
export interface CalloutData {
    /**
     * Knee point where the leader line bends.
     * Optional — when absent, the leader is a straight segment from
     * the text-box edge to the endpoint.
     */
    readonly knee?: {
        readonly x: number;
        readonly y: number;
    };
    /** Endpoint of the leader line (arrow tip). */
    readonly endpoint: {
        readonly x: number;
        readonly y: number;
    };
    /**
     * Style of the leader line's terminating arrowhead at the endpoint.
     * Defaults to 'OpenArrow' to match the PDF /LE convention for /IT
     * /FreeTextCallout. Currently only 'OpenArrow' and 'None' are rendered.
     */
    readonly lineEnding?: 'OpenArrow' | 'ClosedArrow' | 'None';
}
/** Rich text content for comments (plain text or XHTML). */
export interface RichTextContent {
    readonly format: 'plain' | 'xhtml';
    readonly value: string;
}
/** A user mention inside a comment. */
export interface CommentMention {
    readonly userId: string;
    readonly userName: string;
    /** Character offset of the @mention in the comment text. */
    readonly offset: number;
    /** Length of the mention string including the @. */
    readonly length: number;
}
/** Review status for an annotation's comment thread. */
export type CommentStatus = 'open' | 'accepted' | 'rejected' | 'completed' | 'cancelled';
/** A single comment in an annotation's comment thread. */
export interface AnnotationComment {
    readonly id: string;
    readonly authorId: string;
    readonly authorName: string;
    /** Plain text content. Always present for backwards compatibility. */
    readonly text: string;
    /** Rich text alternative. When present, consumers that support it should prefer this over `text`. */
    readonly richText?: RichTextContent;
    readonly timestamp: string;
    readonly edited?: boolean;
    /** Timestamp of last edit, if edited. */
    readonly editedAt?: string;
    /** @mentions in this comment. */
    readonly mentions?: readonly CommentMention[];
    /** Timestamp when the current user last read this comment (client-side). */
    readonly readAt?: string;
    /** Arbitrary consumer-defined metadata. Lector stores/emits it but never interprets it. */
    readonly customData?: Readonly<Record<string, unknown>>;
}
/** Widget (form field) data embedded in an annotation. */
export interface WidgetData {
    readonly fieldType: number;
    readonly fieldName: string;
    readonly fieldValue: string;
    readonly exportValue?: string;
    readonly isChecked?: boolean;
    readonly options?: ReadonlyArray<{
        readonly label: string;
        readonly selected: boolean;
        readonly index: number;
    }>;
    /** Annotation index on the page — used to match widget data to its annotation rect. */
    readonly annotIndex: number;
    /** PDF form field flags (multiline, password, read-only, required, etc.). */
    readonly fieldFlags: number;
}
/**
 * Annotation data — serializable, crosses the worker boundary.
 *
 * Every annotation loaded from pdfium or created via the API is represented
 * by this structure. Subtype-specific payloads are stored in optional fields.
 * All fields except `id`, `pageIndex`, `subtype`, `rect`, and `flags` are optional
 * to ensure backwards compatibility as the model evolves.
 */
export interface AnnotationData {
    readonly id: string;
    readonly pageIndex: number;
    readonly subtype: number;
    readonly rect: {
        readonly left: number;
        readonly top: number;
        readonly right: number;
        readonly bottom: number;
    };
    readonly flags: number;
    readonly color?: RgbaColor;
    readonly interiorColor?: RgbaColor;
    readonly border?: BorderStyle;
    /** Per-annotation opacity (0.0–1.0). Separate from color alpha. Maps to PDF /CA key. */
    readonly opacity?: number;
    /** Line ending styles for line/polyline annotations. */
    readonly lineCaps?: {
        readonly start: LineCap;
        readonly end: LineCap;
    };
    /** PDF blend mode for compositing. */
    readonly blendMode?: BlendMode;
    /** Custom appearance stream reference. */
    readonly appearance?: string;
    /** Note icon type for TEXT (sticky note) annotations. */
    readonly noteIcon?: NoteIcon;
    readonly contents?: string;
    /** Human-readable author name (PDF /T entry). */
    readonly author?: string;
    /** Author's user ID (for programmatic identity, not stored in PDF). */
    readonly authorId?: string;
    readonly modifiedDate?: string;
    readonly createdDate?: string;
    /** User ID of the last person to modify this annotation. */
    readonly modifiedBy?: string;
    /**
     * Internal tag for annotation subtype disambiguation.
     * Used to distinguish e.g. 'line' vs 'arrow' vs 'polygon' when stored
     * as INK subtype due to pdfium limitations. NOT the human author.
     */
    readonly tag?: string;
    /**
     * Threaded comments attached to this annotation.
     * The first entry is the initial comment; subsequent entries are replies.
     * Stored client-side (not in the PDF — use XFDF or backend for persistence).
     */
    readonly comments?: readonly AnnotationComment[];
    /** Review status of the annotation's comment thread. Defaults to 'open'. */
    readonly commentStatus?: CommentStatus;
    /** Whether the comment thread is resolved (collapsed in sidebar). */
    readonly resolved?: boolean;
    /**
     * User IDs allowed to edit this annotation. If undefined, only the
     * creator (authorId) and users with admin role can edit.
     * An empty array means no one can edit (locked).
     */
    readonly editableBy?: readonly string[];
    /** User IDs allowed to delete this annotation. Same semantics as editableBy. */
    readonly deletableBy?: readonly string[];
    /** Z-order index for rendering order. Higher values render on top. */
    readonly zIndex?: number;
    /** Group ID. Annotations with the same groupId move/delete/change as a unit. */
    readonly groupId?: string;
    /** Timestamp when the current user last viewed this annotation (client-side). */
    readonly readAt?: string;
    readonly markup?: MarkupData;
    readonly ink?: InkData;
    readonly line?: LineData;
    readonly freeText?: FreeTextData;
    readonly widget?: WidgetData;
    readonly stamp?: StampData;
    readonly redaction?: RedactionData;
    readonly image?: ImageAnnotationData;
    readonly signature?: SignatureData;
    readonly measurement?: MeasurementData;
    readonly callout?: CalloutData;
    /** Arbitrary consumer-defined metadata. Lector stores/emits it but never interprets it. */
    readonly customData?: Readonly<Record<string, unknown>>;
}
/**
 * Event emitted for every data mutation (create, update, delete).
 *
 * These events form an append-only operation log that can be replayed for
 * undo/redo, synced to a server, or used for collaborative editing.
 */
export interface DataEvent<T> {
    readonly type: 'created' | 'updated' | 'deleted';
    readonly documentId: string;
    readonly pageIndex: number;
    readonly objectId: string;
    readonly data: T;
    readonly patch?: Partial<T>;
    readonly timestamp: number;
    readonly operationId: string;
    readonly userId?: string;
}
/** A tracked object with reactive commit state for dirty tracking. */
export interface TrackedObject<T> {
    readonly id: string;
    readonly data: T;
    readonly commitState: ReadonlySignal<CommitState>;
    markSynced(): void;
}
//# sourceMappingURL=types.d.ts.map