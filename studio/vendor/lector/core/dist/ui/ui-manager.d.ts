import type { ReadonlySignal } from '@truespar/lector-utils';
import type { CommandRegistry } from '../plugin/commands.js';
import type { LectorUISchema, ThemeMode, UIState } from './types.js';
/**
 * Optional translator function passed into UIManager. When provided,
 * schema items carrying `labelKey` / `tooltipKey` fields have those
 * keys resolved through this function at item-resolution time.
 * Without a translator, keys are ignored and the literal `label` /
 * `tooltip` fields are used as-is (or left empty).
 */
export type SchemaTranslator = (key: string) => string;
/**
 * UIManager — resolves a LectorUISchema into reactive UIState.
 *
 * This is the framework-agnostic core of the UI system. It:
 * - Resolves toolbar items against the CommandRegistry
 * - Tracks breakpoint tier and filters by category visibility
 * - Manages sidebar panel state (active panel, collapsed)
 * - Manages theme mode (light/dark/system)
 * - Provides the reactive UIState that framework bindings consume
 *
 * Framework bindings (React, Vue, Svelte) subscribe to UIState signals
 * and render the appropriate DOM elements.
 */
export declare class UIManager implements Disposable {
    #private;
    readonly state: UIState;
    constructor(schema: LectorUISchema, commands: CommandRegistry, translator?: SchemaTranslator);
    /** Start observing a container for responsive breakpoints. */
    observe(container: HTMLElement): void;
    /** Stop observing. */
    disconnect(): void;
    /** Replace the UI schema at runtime. */
    updateSchema(schema: LectorUISchema): void;
    /** Get the current schema. */
    get schema(): Readonly<LectorUISchema>;
    /** Set the theme mode. */
    setTheme(mode: ThemeMode): void;
    /** Get resolved (effective) theme considering 'system'. */
    get effectiveTheme(): ReadonlySignal<'light' | 'dark'>;
    /** Toggle sidebar collapsed state. */
    toggleSidebar(): void;
    /** Set sidebar collapsed state explicitly. */
    setSidebarCollapsed(collapsed: boolean): void;
    /** Set the active sidebar panel by ID. */
    setActivePanel(panelId: string | null): void;
    [Symbol.dispose](): void;
}
//# sourceMappingURL=ui-manager.d.ts.map