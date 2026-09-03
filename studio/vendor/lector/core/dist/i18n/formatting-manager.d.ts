import type { ReadonlySignal } from '@truespar/lector-utils';
/**
 * Measurement system for length and area display.
 *
 * Determines which units are used when formatting raw PDF measurements
 * (which are always in points) for the user. Independent of the
 * measurement plugin's per-annotation `MeasurementUnit` — that one is
 * an explicit author choice; this one is a viewer-side display preference.
 */
export type MeasurementSystem = 'metric' | 'imperial';
/**
 * Hour-cycle override for time formatting.
 *
 * Mirrors the BCP 47 `hourCycle` field accepted by `Intl.DateTimeFormat`.
 * - `'h11'` — 0..11 (Japanese style)
 * - `'h12'` — 1..12 (US-style 12-hour clock)
 * - `'h23'` — 0..23 (24-hour clock starting at 0)
 * - `'h24'` — 1..24 (rare; midnight is 24)
 */
export type HourCycle = 'h11' | 'h12' | 'h23' | 'h24';
/**
 * Options accepted by `FormattingManager`. All are optional — anything
 * left unset is auto-detected from the browser locale (or sensible defaults
 * if `navigator` is unavailable, e.g., during SSR).
 */
export interface FormattingManagerOptions {
    /**
     * BCP 47 locale tag used for date, number, and unit formatting
     * (e.g., `'en-US'`, `'sv-SE'`, `'de-DE'`).
     *
     * Defaults to `navigator.language`, then `'en-US'`.
     */
    readonly formatLocale?: string;
    /**
     * Measurement system for length/area display.
     *
     * Defaults to derivation from `formatLocale`'s region:
     * `US`, `LR`, `MM` → `'imperial'`; everything else → `'metric'`.
     */
    readonly measurementSystem?: MeasurementSystem;
    /**
     * Hour-cycle override. When unset, the locale's default is used
     * (which gives 24-hour for most of the world and 12-hour for en-US).
     */
    readonly hourCycle?: HourCycle;
}
/**
 * Manages locale-aware formatting for dates, numbers, file sizes, and
 * measurements.
 *
 * Three independent inputs:
 * 1. **Format locale** (BCP 47 tag) — drives `Intl.*` formatters
 * 2. **Measurement system** — chooses metric vs imperial units
 * 3. **Hour cycle** — overrides 12/24-hour clock independent of locale
 *
 * All three are reactive signals; UI consumers can subscribe and re-format
 * when the user (or developer) changes them at runtime.
 *
 * **Why this exists separately from `I18nManager`:** UI translation language
 * and number/date format conventions are unrelated. A Swedish user reading
 * an English UI still expects `2026-04-06 14:30` and `12,5 MB`, not
 * `Apr 6, 2026, 2:30 PM` and `12.5 MB`. Conflating the two is a bug.
 */
export declare class FormattingManager implements Disposable {
    #private;
    constructor(options?: FormattingManagerOptions);
    /** Active BCP 47 format locale as a reactive signal. */
    get locale(): ReadonlySignal<string>;
    /** Active measurement system as a reactive signal. */
    get measurementSystem(): ReadonlySignal<MeasurementSystem>;
    /** Active hour-cycle override (or `undefined` to use locale default). */
    get hourCycle(): ReadonlySignal<HourCycle | undefined>;
    /**
     * Change the active format locale at runtime.
     *
     * If the measurement system was auto-derived (not explicitly provided
     * at construction), it is re-derived from the new locale's region.
     */
    setFormatLocale(locale: string): void;
    /** Change the measurement system. */
    setMeasurementSystem(system: MeasurementSystem): void;
    /** Change the hour-cycle override (or pass `undefined` to clear it). */
    setHourCycle(cycle: HourCycle | undefined): void;
    /**
     * Parse a PDF date string (`D:YYYYMMDDHHmmSSOHH'mm'`) into a `Date`,
     * or fall back to `Date` constructor for ISO/RFC strings.
     *
     * Returns `null` if parsing fails.
     */
    parsePdfDate(input: string): Date | null;
    /** Format a date as a localized medium-style date (no time). */
    formatDate(input: Date | number | string): string;
    /** Format a date as a localized short-style time (no date). */
    formatTime(input: Date | number | string): string;
    /** Format a date as combined medium date + short time. */
    formatDateTime(input: Date | number | string): string;
    /** Locale-aware number formatting. */
    formatNumber(value: number, options?: Intl.NumberFormatOptions): string;
    /**
     * Format a byte count as a human-readable string with locale-correct
     * decimal separator. Uses binary prefixes (1 KB = 1024 B) consistent
     * with how the rest of the codebase reports sizes.
     */
    formatFileSize(bytes: number): string;
    /**
     * Format a length given in PDF points (1 pt = 1/72 inch) using the
     * active measurement system. Picks an appropriate sub-unit based on
     * magnitude.
     *
     * **Metric** uses CAD/engineering conventions: millimetres for sub-metre
     * lengths, metres for ≥1 m. Centimetres are intentionally skipped — they
     * are an everyday unit and are not used in technical drawings, which is
     * the dominant use case for a PDF measurement tool.
     *
     * **Imperial** uses inches up to 12, feet up to 3, yards beyond.
     */
    formatLengthFromPoints(points: number, precision?: number): string;
    /**
     * Format an area given in square PDF points using the active
     * measurement system.
     *
     * **Metric** mirrors the length convention: square millimetres for
     * sub-square-metre areas, square metres for ≥1 m². No cm² for the same
     * reason length skips cm — it isn't a technical drawing convention.
     *
     * **Imperial** uses square inches up to 1 ft² (144 in²), then square feet.
     */
    formatAreaFromSquarePoints(squarePoints: number, precision?: number): string;
    [Symbol.dispose](): void;
}
//# sourceMappingURL=formatting-manager.d.ts.map