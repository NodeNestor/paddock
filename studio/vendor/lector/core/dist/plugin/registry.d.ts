import type { LectorEngine } from '../engine/lector-engine.js';
import type { PluginDefinition } from './define-plugin.js';
import { EventBus } from './event-bus.js';
import { CommandRegistry } from './commands.js';
/**
 * Plugin registry — the main orchestrator for the plugin system.
 *
 * Manages plugin registration, dependency resolution, initialization
 * in topological order, capability lookup, and disposal.
 *
 * @example
 * ```ts
 * const registry = new PluginRegistry(engine);
 * registry.register(zoomPlugin);
 * registry.register(searchPlugin);
 * await registry.init();
 *
 * const zoom = registry.get<ZoomCapability>('zoom');
 * zoom.setLevel(1.5);
 *
 * registry[Symbol.dispose]();
 * ```
 */
export declare class PluginRegistry implements Disposable {
    #private;
    /** Shared event bus for inter-plugin communication. */
    readonly events: EventBus;
    /** Shared command registry for keyboard shortcuts and actions. */
    readonly commands: CommandRegistry;
    constructor(engine: LectorEngine);
    /**
     * Register a plugin definition. Must be called before `init()`.
     *
     * @param definition - The plugin definition to register.
     * @throws If the registry has already been initialized.
     */
    register(definition: PluginDefinition<unknown, unknown>): void;
    /**
     * Initialize all registered plugins in dependency order.
     *
     * 1. Validates no duplicate capabilities across plugins.
     * 2. Resolves topological initialization order via dependency graph.
     * 3. For each plugin (in order): creates state, builds context, calls setup().
     * 4. Stores returned capabilities and dispose callbacks.
     *
     * @throws If already initialized, if dependencies cannot be resolved,
     *         or if a plugin's setup() throws.
     */
    init(): Promise<void>;
    /**
     * Get a capability by name. Throws if not found.
     *
     * @typeParam T - The expected capability type.
     * @param capability - The capability name to look up.
     * @returns The capability value.
     * @throws If the capability is not registered.
     */
    get<T>(capability: string): T;
    /**
     * Get a capability by name, or null if not found.
     *
     * @typeParam T - The expected capability type.
     * @param capability - The capability name to look up.
     * @returns The capability value, or null.
     */
    tryGet<T>(capability: string): T | null;
    /**
     * Dispose: run plugin dispose callbacks in reverse order,
     * then dispose the event bus and command registry.
     */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=registry.d.ts.map