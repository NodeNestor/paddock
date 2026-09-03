import type { PluginContext } from './context.js';
/**
 * A complete plugin definition.
 *
 * Plugins declare what capabilities they provide, what they require from
 * other plugins, and a `setup()` function that receives a context and
 * returns the plugin's capability value.
 *
 * @typeParam TCapability - The type of the capability this plugin provides.
 * @typeParam TState - The type of the plugin's reactive state.
 */
export interface PluginDefinition<TCapability = unknown, TState = Record<string, never>> {
    /** Unique plugin identifier. */
    readonly id: string;
    /** Capabilities this plugin provides to others. */
    readonly provides: readonly string[];
    /** Capabilities this plugin requires (must be available at init time). */
    readonly requires: readonly string[];
    /** Capabilities this plugin optionally depends on. */
    readonly optional: readonly string[];
    /** Factory for the plugin's reactive state. Called once before setup(). */
    readonly state?: () => TState;
    /**
     * Initialize the plugin. Receives a context with access to other plugins'
     * capabilities, the event bus, engine, and commands. Returns the plugin's
     * capability value (or a promise for async initialization).
     */
    setup(ctx: PluginContext<TState>): TCapability | Promise<TCapability>;
    /** Teardown hook called when the plugin registry is disposed. */
    dispose?: () => void | Promise<void>;
}
/**
 * Define a plugin with full type inference.
 *
 * This is an identity function — it performs no runtime transformation.
 * Its purpose is to enable TypeScript to infer the generic parameters
 * from the definition object, providing full type safety for `setup()`.
 *
 * @example
 * ```ts
 * const myPlugin = definePlugin({
 *   id: 'my-plugin',
 *   provides: ['myCapability'],
 *   requires: [],
 *   optional: [],
 *   setup(ctx) {
 *     return { doSomething() { ... } };
 *   },
 * });
 * ```
 */
export declare function definePlugin<TCapability, TState = Record<string, never>>(definition: PluginDefinition<TCapability, TState>): PluginDefinition<TCapability, TState>;
//# sourceMappingURL=define-plugin.d.ts.map