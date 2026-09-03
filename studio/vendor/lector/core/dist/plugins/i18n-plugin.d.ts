import type { LocaleId, TranslationMap, InterpolationParams } from '../i18n/types.js';
import type { ReadonlySignal } from '@truespar/lector-utils';
/**
 * Internationalization capability.
 *
 * Provides translation resolution (`t()`), locale switching, and
 * dynamic translation loading. Other plugins optionally depend on this
 * to localize their UI strings.
 */
export interface I18nCapability {
    /** Resolve a translation key, with optional interpolation parameters. */
    t(key: string, params?: InterpolationParams): string;
    /** Current locale as a reactive signal. */
    locale: ReadonlySignal<LocaleId>;
    /** Switch the active locale. */
    setLocale(locale: LocaleId): void;
    /** Add or merge translations for a locale. */
    addTranslations(locale: LocaleId, translations: TranslationMap): void;
    /** Check if a locale has translations. */
    hasLocale(locale: LocaleId): boolean;
    /** Get all registered locale IDs. */
    getLocales(): LocaleId[];
}
export declare const i18nPlugin: import("../index.js").PluginDefinition<I18nCapability, Record<string, never>>;
//# sourceMappingURL=i18n-plugin.d.ts.map