import type { ReadonlySignal } from '@truespar/lector-utils';
/**
 * A registered command — an executable action with metadata.
 *
 * Commands can be bound to keyboard shortcuts and organized by category
 * for display in command palettes or toolbars.
 */
export interface Command {
    /** Unique command identifier (e.g. `'zoom.in'`, `'page.next'`). */
    readonly id: string;
    /** Human-readable label for UI display. */
    readonly label: string;
    /** Optional icon identifier. */
    readonly icon?: string;
    /** Optional keyboard shortcut (e.g. `'Ctrl+='`). */
    readonly shortcut?: string;
    /** Optional category for grouping (e.g. `'Navigation'`, `'Zoom'`). */
    readonly category?: string;
    /** Reactive signal indicating whether the command is currently enabled. */
    readonly enabled?: ReadonlySignal<boolean>;
    /** Execute the command. */
    execute(): void | Promise<void>;
}
/**
 * Registry of executable commands.
 *
 * Plugins register commands during setup. The registry provides lookup,
 * execution, and category-based filtering for UI integration.
 */
export declare class CommandRegistry implements Disposable {
    #private;
    /**
     * Register a command. Throws if a command with the same ID already exists.
     *
     * @param command - The command to register.
     */
    register(command: Command): void;
    /**
     * Unregister a command by ID.
     *
     * @param id - The command ID to remove.
     */
    unregister(id: string): void;
    /**
     * Execute a command by ID.
     *
     * @param id - The command ID to execute.
     * @throws If the command is not found or is disabled.
     */
    execute(id: string): Promise<void>;
    /**
     * Get a command by ID.
     *
     * @param id - The command ID to look up.
     * @returns The command, or undefined if not found.
     */
    get(id: string): Command | undefined;
    /**
     * Get all registered commands as a read-only map.
     */
    getAll(): ReadonlyMap<string, Command>;
    /**
     * Get all commands in a given category.
     *
     * @param category - The category to filter by.
     * @returns Commands whose category matches.
     */
    getByCategory(category: string): Command[];
    /** Dispose: clear all commands. */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=commands.d.ts.map