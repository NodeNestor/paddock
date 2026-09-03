export declare const FpdfError: {
    readonly SUCCESS: 0;
    readonly UNKNOWN: 1;
    readonly FILE: 2;
    readonly FORMAT: 3;
    readonly PASSWORD: 4;
    readonly SECURITY: 5;
    readonly PAGE: 6;
    readonly XFALOAD: 7;
    readonly XFALAYOUT: 8;
};
export type FpdfError = (typeof FpdfError)[keyof typeof FpdfError];
export declare const FpdfRenderFlag: {
    readonly ANNOT: 1;
    readonly LCD_TEXT: 2;
    readonly NO_NATIVETEXT: 4;
    readonly GRAYSCALE: 8;
    readonly REVERSE_BYTE_ORDER: 16;
    readonly CONVERT_FILL_TO_STROKE: 32;
    readonly NO_CATCH: 256;
    readonly LIMITED_IMAGE_CACHE: 512;
    readonly FORCE_HALFTONE: 1024;
    readonly PRINTING: 2048;
    readonly NO_SMOOTHTEXT: 4096;
    readonly NO_SMOOTHIMAGE: 8192;
    readonly NO_SMOOTHPATH: 16384;
};
export type FpdfRenderFlag = (typeof FpdfRenderFlag)[keyof typeof FpdfRenderFlag];
export declare const FpdfBitmapFormat: {
    readonly UNKNOWN: 0;
    readonly GRAY: 1;
    readonly BGR: 2;
    readonly BGRX: 3;
    readonly BGRA: 4;
    readonly BGRA_PREMUL: 5;
};
export type FpdfBitmapFormat = (typeof FpdfBitmapFormat)[keyof typeof FpdfBitmapFormat];
export declare const FpdfAnnotSubtype: {
    readonly UNKNOWN: 0;
    readonly TEXT: 1;
    readonly LINK: 2;
    readonly FREETEXT: 3;
    readonly LINE: 4;
    readonly SQUARE: 5;
    readonly CIRCLE: 6;
    readonly POLYGON: 7;
    readonly POLYLINE: 8;
    readonly HIGHLIGHT: 9;
    readonly UNDERLINE: 10;
    readonly SQUIGGLY: 11;
    readonly STRIKEOUT: 12;
    readonly STAMP: 13;
    readonly CARET: 14;
    readonly INK: 15;
    readonly POPUP: 16;
    readonly FILEATTACHMENT: 17;
    readonly SOUND: 18;
    readonly MOVIE: 19;
    readonly WIDGET: 20;
    readonly SCREEN: 21;
    readonly PRINTERMARK: 22;
    readonly TRAPNET: 23;
    readonly WATERMARK: 24;
    readonly THREED: 25;
    readonly RICHMEDIA: 26;
    readonly XFAWIDGET: 27;
    readonly REDACT: 28;
};
export type FpdfAnnotSubtype = (typeof FpdfAnnotSubtype)[keyof typeof FpdfAnnotSubtype];
export declare const FpdfAnnotFlag: {
    readonly NONE: 0;
    readonly INVISIBLE: number;
    readonly HIDDEN: number;
    readonly PRINT: number;
    readonly NOZOOM: number;
    readonly NOROTATE: number;
    readonly NOVIEW: number;
    readonly READONLY: number;
    readonly LOCKED: number;
    readonly TOGGLENOVIEW: number;
};
export type FpdfAnnotFlag = (typeof FpdfAnnotFlag)[keyof typeof FpdfAnnotFlag];
export declare const FpdfAnnotAppearanceMode: {
    readonly NORMAL: 0;
    readonly ROLLOVER: 1;
    readonly DOWN: 2;
};
export type FpdfAnnotAppearanceMode = (typeof FpdfAnnotAppearanceMode)[keyof typeof FpdfAnnotAppearanceMode];
export declare const FpdfAnnotAAction: {
    readonly KEY_STROKE: 12;
    readonly FORMAT: 13;
    readonly VALIDATE: 14;
    readonly CALCULATE: 15;
};
export type FpdfAnnotAAction = (typeof FpdfAnnotAAction)[keyof typeof FpdfAnnotAAction];
export declare const FpdfAnnotColorType: {
    readonly COLOR: 0;
    readonly INTERIOR_COLOR: 1;
};
export type FpdfAnnotColorType = (typeof FpdfAnnotColorType)[keyof typeof FpdfAnnotColorType];
export declare const FpdfFormFieldType: {
    readonly UNKNOWN: 0;
    readonly PUSHBUTTON: 1;
    readonly CHECKBOX: 2;
    readonly RADIOBUTTON: 3;
    readonly COMBOBOX: 4;
    readonly LISTBOX: 5;
    readonly TEXTFIELD: 6;
    readonly SIGNATURE: 7;
};
export type FpdfFormFieldType = (typeof FpdfFormFieldType)[keyof typeof FpdfFormFieldType];
export declare const FpdfFormType: {
    readonly NONE: 0;
    readonly ACRO_FORM: 1;
    readonly XFA_FULL: 2;
    readonly XFA_FOREGROUND: 3;
};
export type FpdfFormType = (typeof FpdfFormType)[keyof typeof FpdfFormType];
export declare const FpdfActionType: {
    readonly UNSUPPORTED: 0;
    readonly GOTO: 1;
    readonly REMOTEGOTO: 2;
    readonly URI: 3;
    readonly LAUNCH: 4;
    readonly EMBEDDEDGOTO: 5;
};
export type FpdfActionType = (typeof FpdfActionType)[keyof typeof FpdfActionType];
export declare const FpdfPageObjectType: {
    readonly UNKNOWN: 0;
    readonly TEXT: 1;
    readonly PATH: 2;
    readonly IMAGE: 3;
    readonly SHADING: 4;
    readonly FORM: 5;
};
export type FpdfPageObjectType = (typeof FpdfPageObjectType)[keyof typeof FpdfPageObjectType];
export declare const FpdfSegmentType: {
    readonly UNKNOWN: -1;
    readonly LINETO: 0;
    readonly BEZIERTO: 1;
    readonly MOVETO: 2;
};
export type FpdfSegmentType = (typeof FpdfSegmentType)[keyof typeof FpdfSegmentType];
export declare const FpdfFillMode: {
    readonly NONE: 0;
    readonly ALTERNATE: 1;
    readonly WINDING: 2;
};
export type FpdfFillMode = (typeof FpdfFillMode)[keyof typeof FpdfFillMode];
export declare const FpdfLineCap: {
    readonly BUTT: 0;
    readonly ROUND: 1;
    readonly PROJECTING_SQUARE: 2;
};
export type FpdfLineCap = (typeof FpdfLineCap)[keyof typeof FpdfLineCap];
export declare const FpdfLineJoin: {
    readonly MITER: 0;
    readonly ROUND: 1;
    readonly BEVEL: 2;
};
export type FpdfLineJoin = (typeof FpdfLineJoin)[keyof typeof FpdfLineJoin];
export declare const FpdfTextRenderMode: {
    readonly UNKNOWN: -1;
    readonly FILL: 0;
    readonly STROKE: 1;
    readonly FILL_STROKE: 2;
    readonly INVISIBLE: 3;
    readonly FILL_CLIP: 4;
    readonly STROKE_CLIP: 5;
    readonly FILL_STROKE_CLIP: 6;
    readonly CLIP: 7;
};
export type FpdfTextRenderMode = (typeof FpdfTextRenderMode)[keyof typeof FpdfTextRenderMode];
export declare const FpdfDuplexType: {
    readonly UNDEFINED: 0;
    readonly SIMPLEX: 1;
    readonly DUPLEX_FLIP_SHORT_EDGE: 2;
    readonly DUPLEX_FLIP_LONG_EDGE: 3;
};
export type FpdfDuplexType = (typeof FpdfDuplexType)[keyof typeof FpdfDuplexType];
export declare const FpdfFlattenResult: {
    readonly FAIL: 0;
    readonly SUCCESS: 1;
    readonly NOTHING_TO_DO: 2;
};
export type FpdfFlattenResult = (typeof FpdfFlattenResult)[keyof typeof FpdfFlattenResult];
export declare const FpdfPageMode: {
    readonly UNKNOWN: -1;
    readonly USE_NONE: 0;
    readonly USE_OUTLINES: 1;
    readonly USE_THUMBS: 2;
    readonly FULL_SCREEN: 3;
    readonly USE_OC: 4;
    readonly USE_ATTACHMENTS: 5;
};
export type FpdfPageMode = (typeof FpdfPageMode)[keyof typeof FpdfPageMode];
export declare const FpdfObjectType: {
    readonly UNKNOWN: 0;
    readonly BOOLEAN: 1;
    readonly NUMBER: 2;
    readonly STRING: 3;
    readonly NAME: 4;
    readonly ARRAY: 5;
    readonly DICTIONARY: 6;
    readonly STREAM: 7;
    readonly NULLOBJ: 8;
    readonly REFERENCE: 9;
};
export type FpdfObjectType = (typeof FpdfObjectType)[keyof typeof FpdfObjectType];
export declare const FpdfUnsupportedType: {
    readonly DOC_XFAFORM: 1;
    readonly DOC_PORTABLECOLLECTION: 2;
    readonly DOC_ATTACHMENT: 3;
    readonly DOC_SECURITY: 4;
    readonly DOC_SHAREDREVIEW: 5;
    readonly DOC_SHAREDFORM_ACROBAT: 6;
    readonly DOC_SHAREDFORM_FILESYSTEM: 7;
    readonly DOC_SHAREDFORM_EMAIL: 8;
    readonly ANNOT_3DANNOT: 11;
    readonly ANNOT_MOVIE: 12;
    readonly ANNOT_SOUND: 13;
    readonly ANNOT_SCREEN_MEDIA: 14;
    readonly ANNOT_SCREEN_RICHMEDIA: 15;
    readonly ANNOT_ATTACHMENT: 16;
    readonly ANNOT_SIG: 17;
};
export type FpdfUnsupportedType = (typeof FpdfUnsupportedType)[keyof typeof FpdfUnsupportedType];
//# sourceMappingURL=enums.d.ts.map