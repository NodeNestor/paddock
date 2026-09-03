import type { ReadonlySignal } from '@truespar/lector-utils';
/**
 * Icon reference — either a built-in name or a custom SVG string.
 *
 * Built-in icons are resolved by the framework binding.
 * Custom SVG strings start with `<svg`.
 */
export type IconRef = string;
/** Where a toolbar item sits horizontally. */
export type ToolbarSection = 'left' | 'center' | 'right';
/** Base properties shared by all toolbar items. */
interface ToolbarItemBase {
    /** Unique item identifier. */
    readonly id: string;
    /** Horizontal section within the toolbar. */
    readonly section?: ToolbarSection;
    /**
     * Visibility category — controls responsive hiding.
     *
     * - `'essential'` — always visible, even at compact breakpoint
     * - `'standard'` — visible at medium and wide breakpoints
     * - `'extended'` — visible only at wide breakpoint
     */
    readonly category?: 'essential' | 'standard' | 'extended';
    /**
     * Sort priority within section. Higher values appear first.
     * Items with equal priority maintain declaration order.
     */
    readonly priority?: number;
    /** Override visibility. When false, the item is never rendered. */
    readonly visible?: boolean;
}
/**
 * Label / tooltip fields on schema items can be provided two ways:
 *
 *  - As a literal string (`label`, `tooltip`) — the value is rendered
 *    verbatim, regardless of locale. Use when you want hardcoded text.
 *  - As an i18n key reference (`labelKey`, `tooltipKey`) — the value
 *    is resolved through the i18n plugin at render time, so locale
 *    changes automatically re-translate the UI.
 *
 * When both are present, the key wins. The literal form is retained
 * for consumers that don't wire up i18n or for one-off custom items.
 */
/** A clickable button bound to a command. */
export interface ToolbarButton extends ToolbarItemBase {
    readonly type: 'button';
    /** Command ID from CommandRegistry. */
    readonly commandId: string;
    /** Display label. Falls back to command label if omitted. */
    readonly label?: string;
    /** i18n key for the display label (takes precedence over `label`). */
    readonly labelKey?: string;
    /** Icon reference. Falls back to command icon if omitted. */
    readonly icon?: IconRef;
    /** Tooltip text. Falls back to label if omitted. */
    readonly tooltip?: string;
    /** i18n key for the tooltip (takes precedence over `tooltip`). */
    readonly tooltipKey?: string;
}
/** A toggle button with on/off state bound to a signal. */
export interface ToolbarToggle extends ToolbarItemBase {
    readonly type: 'toggle';
    /** Command ID executed on click. */
    readonly commandId: string;
    /** Signal that determines whether the toggle is "on". */
    readonly active?: ReadonlySignal<boolean>;
    readonly label?: string;
    /** i18n key for the display label (takes precedence over `label`). */
    readonly labelKey?: string;
    readonly icon?: IconRef;
    readonly tooltip?: string;
    /** i18n key for the tooltip (takes precedence over `tooltip`). */
    readonly tooltipKey?: string;
}
/** A dropdown that opens a list of sub-items. */
export interface ToolbarDropdown extends ToolbarItemBase {
    readonly type: 'dropdown';
    readonly label?: string;
    /** i18n key for the display label (takes precedence over `label`). */
    readonly labelKey?: string;
    readonly icon?: IconRef;
    readonly tooltip?: string;
    /** i18n key for the tooltip (takes precedence over `tooltip`). */
    readonly tooltipKey?: string;
    /** Items shown in the dropdown menu. */
    readonly items: readonly ToolbarMenuItem[];
}
/** An item inside a toolbar dropdown. */
export interface ToolbarMenuItem {
    readonly id: string;
    readonly type: 'item' | 'separator';
    /** Command ID. Ignored for separators. */
    readonly commandId?: string;
    readonly label?: string;
    /** i18n key for the display label (takes precedence over `label`). */
    readonly labelKey?: string;
    readonly icon?: IconRef;
    /** When true, shows a checkmark. */
    readonly checked?: ReadonlySignal<boolean>;
}
/** A visual separator between toolbar items. */
export interface ToolbarSeparator extends ToolbarItemBase {
    readonly type: 'separator';
}
/** A group of items displayed together (e.g., zoom in/out pair). */
export interface ToolbarGroup extends ToolbarItemBase {
    readonly type: 'group';
    readonly items: readonly ToolbarItem[];
}
/**
 * A slot for custom content — framework bindings render this
 * via a user-provided render function or component.
 */
export interface ToolbarCustom extends ToolbarItemBase {
    readonly type: 'custom';
    /** Arbitrary key the framework binding uses to resolve the renderer. */
    readonly component: string;
    /** Props passed to the custom renderer. */
    readonly props?: Readonly<Record<string, unknown>>;
}
/** Any toolbar item. */
export type ToolbarItem = ToolbarButton | ToolbarToggle | ToolbarDropdown | ToolbarSeparator | ToolbarGroup | ToolbarCustom;
/** Toolbar schema. */
export interface ToolbarSchema {
    /** Show or hide the entire toolbar. */
    readonly visible?: boolean;
    /** Position relative to the viewer. */
    readonly position?: 'top' | 'bottom';
    /** Toolbar items. */
    readonly items: readonly ToolbarItem[];
}
/** A panel displayed in the sidebar. */
export interface SidebarPanel {
    /** Unique panel identifier. */
    readonly id: string;
    /** Display label shown in the panel tab. */
    readonly label: string;
    /** i18n key for the label (takes precedence over `label`). */
    readonly labelKey?: string;
    /** Icon for the tab. */
    readonly icon?: IconRef;
    /**
     * Component key — framework bindings resolve this to a concrete
     * component/element for the panel body.
     */
    readonly component: string;
    /** Props passed to the panel body renderer. */
    readonly props?: Readonly<Record<string, unknown>>;
    /** Default width in pixels. */
    readonly defaultWidth?: number;
    /** Whether the panel is resizable by dragging. */
    readonly resizable?: boolean;
    /** Minimum width in pixels when resizable. */
    readonly minWidth?: number;
    /** Maximum width in pixels when resizable. */
    readonly maxWidth?: number;
}
/** Sidebar schema. */
export interface SidebarSchema {
    /** Which side the sidebar appears on. */
    readonly position?: 'left' | 'right';
    /** Default collapsed state. */
    readonly collapsed?: boolean;
    /** Panels available in the sidebar. */
    readonly panels: readonly SidebarPanel[];
}
/** A single status bar item. */
export interface StatusBarItem {
    readonly id: string;
    /** Section: left-aligned or right-aligned. */
    readonly section?: 'left' | 'right';
    /**
     * Component key for framework bindings, OR one of the built-in
     * types that the default renderer handles.
     */
    readonly component: string;
    /** Props for the renderer. */
    readonly props?: Readonly<Record<string, unknown>>;
    /** Sort priority within section. */
    readonly priority?: number;
}
/** Status bar schema. */
export interface StatusBarSchema {
    readonly visible?: boolean;
    readonly items: readonly StatusBarItem[];
}
/** A context menu entry. */
export interface ContextMenuItem {
    readonly id: string;
    readonly type: 'item' | 'separator' | 'submenu';
    readonly commandId?: string;
    readonly label?: string;
    /** i18n key for the label (takes precedence over `label`). */
    readonly labelKey?: string;
    readonly icon?: IconRef;
    readonly shortcutHint?: string;
    /** Sub-items for type 'submenu'. */
    readonly children?: readonly ContextMenuItem[];
    /** When to show this item (e.g., only when text is selected). */
    readonly when?: string;
}
/** Context menu schema. */
export interface ContextMenuSchema {
    readonly items: readonly ContextMenuItem[];
}
/** An overlay panel (e.g., search bar, find & replace). */
export interface OverlaySchema {
    readonly id: string;
    readonly component: string;
    readonly props?: Readonly<Record<string, unknown>>;
    /** Position hint for the framework binding. */
    readonly position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'center';
    /** Whether clicking outside dismisses the overlay. */
    readonly dismissOnClickOutside?: boolean;
}
/**
 * Breakpoint thresholds in pixels (container width, not window).
 *
 * - Below `compact`: only `essential` category items visible
 * - Between `compact` and `wide`: `essential` + `standard` visible
 * - Above `wide`: all items visible
 */
export interface BreakpointConfig {
    /** Compact threshold — below this, only essentials. */
    readonly compact: number;
    /** Wide threshold — above this, everything shows. */
    readonly wide: number;
}
/** The active breakpoint tier. */
export type BreakpointTier = 'compact' | 'medium' | 'wide';
/** Complete UI schema describing the viewer's chrome. */
export interface LectorUISchema {
    readonly toolbar?: ToolbarSchema;
    readonly sidebar?: SidebarSchema;
    readonly statusBar?: StatusBarSchema;
    readonly contextMenu?: ContextMenuSchema;
    readonly overlays?: readonly OverlaySchema[];
    readonly breakpoints?: BreakpointConfig;
}
/** Named theme — resolved to a CSS class on the root element. */
export type ThemeMode = 'light' | 'dark' | 'system';
/** Theme configuration. */
export interface ThemeConfig {
    /** Active theme. `'system'` follows `prefers-color-scheme`. */
    readonly mode: ThemeMode;
    /** Custom CSS class applied to the viewer root. */
    readonly customClass?: string;
}
/** Reactive state for a resolved toolbar item. */
export interface ResolvedToolbarItem {
    readonly schema: ToolbarItem;
    readonly enabled: ReadonlySignal<boolean>;
    readonly visible: ReadonlySignal<boolean>;
    readonly label: string;
    readonly icon?: IconRef;
    readonly tooltip?: string;
}
/** Reactive state of the full UI. */
export interface UIState {
    readonly toolbar: {
        readonly visible: ReadonlySignal<boolean>;
        readonly items: ReadonlySignal<readonly ResolvedToolbarItem[]>;
    };
    readonly sidebar: {
        readonly visible: ReadonlySignal<boolean>;
        readonly collapsed: ReadonlySignal<boolean>;
        readonly activePanel: ReadonlySignal<string | null>;
        readonly panels: readonly SidebarPanel[];
    };
    readonly statusBar: {
        readonly visible: ReadonlySignal<boolean>;
        readonly items: readonly StatusBarItem[];
    };
    readonly breakpoint: ReadonlySignal<BreakpointTier>;
    readonly theme: ReadonlySignal<ThemeMode>;
}
export {};
//# sourceMappingURL=types.d.ts.map