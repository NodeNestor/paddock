import type { ReadonlySignal } from '@truespar/lector-utils';
import type { MeasurementUnit } from '../data/types.js';
/** Scale mapping from PDF coordinates to real-world measurements. */
export interface MeasurementScale {
    readonly source: number;
    readonly sourceUnit: MeasurementUnit;
    readonly target: number;
    readonly targetUnit: MeasurementUnit;
}
/** Convert a raw PDF-point distance into the given unit (no scale applied). */
export declare function convertPointsToUnit(valuePt: number, unit: MeasurementUnit): number;
/**
 * Convert a length stored in raw PDF points into a real-world value
 * using a calibration scale snapshot.
 *
 * Semantic: `source` units in `sourceUnit` correspond to `target` units
 * in `targetUnit`. So we first express valuePt in sourceUnit, then
 * scale by `target / source`. The resulting number is in targetUnit.
 *
 * When `scale` is null (uncalibrated), we just convert valuePt into
 * the requested `unit` directly.
 */
export declare function convertLengthWithScale(valuePt: number, scale: MeasurementScale | null | undefined, fallbackUnit: MeasurementUnit): {
    value: number;
    unit: MeasurementUnit;
};
/**
 * Same as `convertLengthWithScale` but for area values stored in raw
 * square PDF points. Area scales quadratically with the linear ratio.
 */
export declare function convertAreaWithScale(valuePt2: number, scale: MeasurementScale | null | undefined, fallbackUnit: MeasurementUnit): {
    value: number;
    unit: MeasurementUnit;
};
/**
 * Measurement tool capability.
 *
 * Provides distance, area, and perimeter calculations with real-world
 * units. Works with the annotation plugin to create measurement annotations
 * that display their computed values.
 */
export interface MeasurementCapability {
    /** Set the measurement scale (PDF points → real-world units). */
    setScale(scale: MeasurementScale): void;
    /** Get the current scale, or null if not configured. */
    getScale(): MeasurementScale | null;
    /** Active display unit as a reactive signal. */
    activeUnit: ReadonlySignal<MeasurementUnit>;
    /** Set the display unit. */
    setActiveUnit(unit: MeasurementUnit): void;
    /** Set decimal precision for display. */
    setPrecision(precision: number): void;
    /** Get current precision. */
    precision: ReadonlySignal<number>;
    /** Calculate distance between two PDF-coordinate points. */
    calculateDistance(p1: {
        x: number;
        y: number;
    }, p2: {
        x: number;
        y: number;
    }): number;
    /** Calculate area of a polygon defined by vertices (PDF coordinates). */
    calculateArea(vertices: ReadonlyArray<{
        x: number;
        y: number;
    }>): number;
    /** Calculate perimeter of a polyline/polygon (PDF coordinates). */
    calculatePerimeter(vertices: ReadonlyArray<{
        x: number;
        y: number;
    }>, closed?: boolean): number;
    /** Convert a PDF-point value to the current display unit. */
    convert(valuePt: number): number;
    /** Format a value with unit suffix (e.g., "12.5 cm"). */
    format(valuePt: number): string;
}
export declare const measurementPlugin: import("../index.js").PluginDefinition<MeasurementCapability, Record<string, never>>;
//# sourceMappingURL=measurement-plugin.d.ts.map