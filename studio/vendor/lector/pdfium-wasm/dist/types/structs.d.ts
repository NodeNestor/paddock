/** Rectangle in page or device coordinates. */
export interface FsRectF {
    readonly left: number;
    readonly top: number;
    readonly right: number;
    readonly bottom: number;
}
/** Transformation matrix [a b c d e f]. */
export interface FsMatrix {
    readonly a: number;
    readonly b: number;
    readonly c: number;
    readonly d: number;
    readonly e: number;
    readonly f: number;
}
/** Size with width and height. */
export interface FsSizeF {
    readonly width: number;
    readonly height: number;
}
/** 2D point. */
export interface FsPointF {
    readonly x: number;
    readonly y: number;
}
/** Quadrilateral defined by four points (used for link/annotation regions). */
export interface FsQuadPointsF {
    readonly x1: number;
    readonly y1: number;
    readonly x2: number;
    readonly y2: number;
    readonly x3: number;
    readonly y3: number;
    readonly x4: number;
    readonly y4: number;
}
/** Byte sizes for heap allocation of each struct. */
export declare const FS_RECTF_SIZE = 16;
export declare const FS_MATRIX_SIZE = 24;
export declare const FS_SIZEF_SIZE = 8;
export declare const FS_POINTF_SIZE = 8;
export declare const FS_QUADPOINTSF_SIZE = 32;
//# sourceMappingURL=structs.d.ts.map