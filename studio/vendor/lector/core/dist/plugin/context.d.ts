import type { Unsubscribe } from '@truespar/lector-utils';
import type { LectorEngine } from '../engine/lector-engine.js';
import type { EventBus } from './event-bus.js';
import type { Command, CommandRegistry } from './commands.js';
/**
 * Context provided to a plugin's `setup()` function.
 *
 * Gives plugins access to capabilities from other plugins, the shared event
 * bus, the engine instance, reactive effects, and the command registry.
 */
export interface PluginContext<TState = Record<string, never>> {
    /** Get a required capability from another plugin. Throws if not found. */
    require<T>(capability: string): T;
    /** Get an optional capability. Returns null if not found. */
    optional<T>(capability: string): T | null;
    /** The plugin's reactive state. */
    readonly state: TState;
    /** Subscribe to lifecycle events. */
    on(event: string, handler: (...args: unknown[]) => void): Unsubscribe;
    /** Emit an event to other plugins. */
    emit(event: string, ...args: unknown[]): void;
    /** Create a reactive effect that auto-tracks signal dependencies. */
    effect(fn: () => void | (() => void)): Unsubscribe;
    /** The Lector engine instance. */
    readonly engine: LectorEngine;
    /** Register a command (keyboard shortcut + action). */
    registerCommand(command: Command): void;
}
/**
 * Options for creating a plugin context.
 */
export interface CreatePluginContextOptions<TState> {
    /** Map of capability name to resolved value. */
    readonly capabilities: ReadonlyMap<string, unknown>;
    /** Shared event bus. */
    readonly events: EventBus;
    /** The Lector engine instance. */
    readonly engine: LectorEngine;
    /** Shared command registry. */
    readonly commands: CommandRegistry;
    /** The plugin's reactive state instance. */
    readonly state: TState;
    /** The plugin's ID (for error messages). */
    readonly pluginId: string;
}
/**
 * Create a PluginContext instance for a specific plugin.
 *
 * @param options - Dependencies and state for the context.
 * @returns A fully wired PluginContext.
 */
export declare function createPluginContext<TState>(options: CreatePluginContextOptions<TState>): PluginContext<TState>;
//# sourceMappingURL=context.d.ts.map