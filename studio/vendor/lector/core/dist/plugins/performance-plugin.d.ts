import type { ReadonlySignal } from '@truespar/lector-utils';
/** A single timing sample with context. */
export interface TimingSample {
    /** Unique label (e.g. `'render:doc-1:page-3'`). */
    readonly label: string;
    /** Duration in milliseconds. */
    readonly durationMs: number;
    /** Timestamp when the sample was recorded. */
    readonly timestamp: number;
}
/** Aggregated statistics for a metric category. */
export interface MetricStats {
    /** Number of samples collected. */
    readonly count: number;
    /** Mean duration in ms. */
    readonly meanMs: number;
    /** Median duration in ms (p50). */
    readonly medianMs: number;
    /** 95th percentile in ms. */
    readonly p95Ms: number;
    /** 99th percentile in ms. */
    readonly p99Ms: number;
    /** Minimum duration in ms. */
    readonly minMs: number;
    /** Maximum duration in ms. */
    readonly maxMs: number;
}
/** Memory usage estimate in bytes. */
export interface MemoryEstimate {
    /** Estimated total memory used by Lector caches (bytes). */
    readonly totalBytes: number;
    /** Tile cache memory estimate (bytes). */
    readonly tileCacheBytes: number;
    /** Number of tiles in cache. */
    readonly tileCacheCount: number;
    /** Number of rendered page canvases in DOM. */
    readonly renderedPageCount: number;
    /** WASM heap size if available (bytes). */
    readonly wasmHeapBytes: number | null;
}
/** Full performance report. */
export interface PerformanceReport {
    /** Engine initialization time in ms. */
    readonly initTimeMs: number | null;
    /** Document load statistics. */
    readonly documentLoad: MetricStats;
    /** Page render statistics. */
    readonly pageRender: MetricStats;
    /** Current memory estimate. */
    readonly memory: MemoryEstimate;
    /** All raw timing samples (up to the configured ring buffer size). */
    readonly samples: readonly TimingSample[];
    /** Report generation timestamp. */
    readonly generatedAt: number;
}
/**
 * Capability provided by the performance plugin.
 *
 * Instruments the engine and exposes metrics as reactive signals.
 * Useful for developer tools, dashboards, and automated benchmarks.
 */
export interface PerformanceCapability {
    /** Engine init time in ms (null until init completes). */
    readonly initTimeMs: ReadonlySignal<number | null>;
    /** Total number of pages rendered since engine init. */
    readonly totalRenders: ReadonlySignal<number>;
    /** Mean page render time (rolling window). */
    readonly meanRenderMs: ReadonlySignal<number>;
    /** Total number of documents loaded. */
    readonly totalDocumentsLoaded: ReadonlySignal<number>;
    /** Mean document load time (rolling window). */
    readonly meanDocLoadMs: ReadonlySignal<number>;
    /** Current estimated memory usage. */
    readonly memory: ReadonlySignal<MemoryEstimate>;
    /**
     * Record a custom timing sample.
     * Useful for application-level metrics (e.g. total time to interactive).
     */
    mark(label: string, durationMs: number): void;
    /**
     * Start a named timer. Returns a function that stops it and records the sample.
     *
     * @example
     * ```ts
     * const stop = perf.startTimer('my-operation');
     * await doSomething();
     * stop(); // Records the duration
     * ```
     */
    startTimer(label: string): () => void;
    /** Generate a full performance report with aggregated statistics. */
    getReport(): PerformanceReport;
    /** Clear all collected samples and reset counters. */
    reset(): void;
}
/**
 * Performance monitoring plugin.
 *
 * Instruments engine initialization, document loading, and page rendering.
 * Exposes metrics via reactive signals and a `getReport()` method for
 * benchmarking and developer tools.
 *
 * Register this plugin alongside your other plugins. It has minimal overhead
 * — timing uses `performance.now()` and samples are stored in a fixed-size
 * ring buffer.
 */
export declare const performancePlugin: import("../index.js").PluginDefinition<PerformanceCapability, Record<string, never>>;
//# sourceMappingURL=performance-plugin.d.ts.map