import { type HourCycle, type MeasurementSystem } from '../i18n/formatting-manager.js';
import type { ReadonlySignal } from '@truespar/lector-utils';
/**
 * Locale-aware formatting capability.
 *
 * Exposes the engine's `FormattingManager` to other plugins and the UI
 * shell. Use this for any user-visible date, number, file size, length,
 * or area string.
 *
 * **Decoupled from `i18n`:** the i18n plugin handles UI translation
 * strings; this plugin handles format conventions. A user can read the
 * UI in English while seeing dates in Swedish format.
 */
export interface FormattingCapability {
    /** Active BCP 47 format locale (reactive). */
    readonly locale: ReadonlySignal<string>;
    /** Active measurement system (reactive). */
    readonly measurementSystem: ReadonlySignal<MeasurementSystem>;
    /** Active hour-cycle override, or `undefined` for locale default (reactive). */
    readonly hourCycle: ReadonlySignal<HourCycle | undefined>;
    /** Change the format locale at runtime. Emits `formatting:locale-changed`. */
    setFormatLocale(locale: string): void;
    /** Change the measurement system at runtime. Emits `formatting:system-changed`. */
    setMeasurementSystem(system: MeasurementSystem): void;
    /** Change the hour-cycle override at runtime. Emits `formatting:hour-cycle-changed`. */
    setHourCycle(cycle: HourCycle | undefined): void;
    /** Parse a PDF date string into a `Date`, or `null` if malformed. */
    parsePdfDate(input: string): Date | null;
    /** Format as localized medium date (no time). */
    formatDate(input: Date | number | string): string;
    /** Format as localized short time (no date). */
    formatTime(input: Date | number | string): string;
    /** Format as localized medium date + short time. */
    formatDateTime(input: Date | number | string): string;
    /** Locale-aware number formatting. */
    formatNumber(value: number, options?: Intl.NumberFormatOptions): string;
    /** Format a byte count with locale-correct decimal separator. */
    formatFileSize(bytes: number): string;
    /** Format a length given in PDF points using the active measurement system. */
    formatLengthFromPoints(points: number, precision?: number): string;
    /** Format an area given in square PDF points using the active measurement system. */
    formatAreaFromSquarePoints(squarePoints: number, precision?: number): string;
}
export declare const formattingPlugin: import("../index.js").PluginDefinition<FormattingCapability, Record<string, never>>;
//# sourceMappingURL=formatting-plugin.d.ts.map