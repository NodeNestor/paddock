/**
 * Describes a plugin's dependency graph node for topological sorting.
 */
export interface PluginNode {
    /** Unique plugin identifier. */
    readonly id: string;
    /** Capabilities this plugin provides. */
    readonly provides: readonly string[];
    /** Capabilities this plugin requires (must be present). */
    readonly requires: readonly string[];
    /** Capabilities this plugin optionally depends on (ignored if absent). */
    readonly optional: readonly string[];
}
/**
 * Resolve plugin initialization order via topological sort (Kahn's algorithm).
 *
 * @param plugins - Plugin nodes describing the dependency graph.
 * @returns Plugin IDs in valid initialization order (dependencies first).
 * @throws If a required capability is not provided by any plugin.
 * @throws If a circular dependency is detected (includes cycle path).
 */
export declare function resolveDependencies(plugins: readonly PluginNode[]): string[];
//# sourceMappingURL=dependency-resolver.d.ts.map