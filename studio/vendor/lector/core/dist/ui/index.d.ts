/**
 * Lector UI System — framework-agnostic UI infrastructure.
 *
 * This module exports the schema types, default schema, UI manager,
 * responsive breakpoint observer, and CSS utilities that framework
 * bindings consume to render the viewer chrome.
 */
export type { IconRef, ToolbarSection, ToolbarButton, ToolbarToggle, ToolbarDropdown, ToolbarMenuItem, ToolbarSeparator, ToolbarGroup, ToolbarCustom, ToolbarItem, ToolbarSchema, SidebarPanel, SidebarSchema, StatusBarItem, StatusBarSchema, ContextMenuItem, ContextMenuSchema, OverlaySchema, BreakpointConfig, BreakpointTier, LectorUISchema, ThemeMode, ThemeConfig, ResolvedToolbarItem, UIState, } from './types.js';
export { DEFAULT_UI_SCHEMA, mergeSchema } from './default-schema.js';
export { UIManager } from './ui-manager.js';
export { BreakpointObserver, DEFAULT_BREAKPOINTS, isCategoryVisible, } from './responsive.js';
export { injectLectorStyles, buildViewerClass } from './css/index.js';
export { LectorViewer } from './lector-viewer.js';
export type { LectorViewerOptions } from './lector-viewer.js';
export { resolveIcon, getIcon, isInlineSvg } from './icons.js';
//# sourceMappingURL=index.d.ts.map