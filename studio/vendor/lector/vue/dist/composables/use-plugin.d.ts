/**
 * Access a plugin capability by name.
 *
 * @typeParam T - The capability interface.
 * @param capability - The capability identifier (e.g. `'zoom'`).
 * @throws If the capability is not registered.
 */
export declare function usePlugin<T>(capability: string): T;
/**
 * Access an optional plugin capability by name. Returns `null` if not registered.
 */
export declare function useOptionalPlugin<T>(capability: string): T | null;
//# sourceMappingURL=use-plugin.d.ts.map