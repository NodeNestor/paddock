import type { ReadonlySignal } from '@truespar/lector-utils';
import type { LectorUISchema, ThemeMode, UIState } from '../ui/types.js';
import { UIManager } from '../ui/ui-manager.js';
/**
 * Capability provided by the UI plugin.
 *
 * Exposes the full reactive UI state and methods to control
 * theme, sidebar, schema overrides, and breakpoint queries.
 */
export interface UICapability {
    /** Full reactive UI state — consumed by framework bindings. */
    readonly state: UIState;
    /** The resolved CSS class string for the viewer root element. */
    readonly viewerClass: ReadonlySignal<string>;
    /** Set theme mode (light/dark/system). */
    setTheme(mode: ThemeMode): void;
    /** Get effective resolved theme (never 'system'). */
    readonly effectiveTheme: ReadonlySignal<'light' | 'dark'>;
    /** Toggle sidebar collapsed state. */
    toggleSidebar(): void;
    /** Set sidebar collapsed state. */
    setSidebarCollapsed(collapsed: boolean): void;
    /** Set the active sidebar panel by ID. */
    setActivePanel(panelId: string | null): void;
    /** Replace the UI schema at runtime. */
    updateSchema(override: Partial<LectorUISchema>): void;
    /** Get the current schema. */
    readonly schema: Readonly<LectorUISchema>;
    /** Access the UIManager directly (advanced usage). */
    readonly manager: UIManager;
}
/**
 * UI plugin.
 *
 * Manages the viewer's UI chrome: toolbar, sidebar, status bar,
 * theme, and responsive breakpoints. Uses the UIManager internally
 * and exposes a clean capability interface for framework bindings.
 *
 * Registers commands for sidebar toggle, theme switching, and
 * layout mode changes.
 */
export declare const uiPlugin: import("../index.js").PluginDefinition<UICapability, Record<string, never>>;
//# sourceMappingURL=ui-plugin.d.ts.map