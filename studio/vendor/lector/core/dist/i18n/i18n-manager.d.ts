import type { ReadonlySignal } from '@truespar/lector-utils';
import type { LocaleId, TranslationMap, InterpolationParams } from './types.js';
/**
 * Manages translations and locale state.
 *
 * English is always available as the fallback locale. Consumers add
 * additional locales via `addTranslations()`. The `t()` method resolves
 * a key to its translated string, falling back to English, then to the
 * key itself if no translation is found.
 *
 * Interpolation: `t('pagination.pageOf', { current: 3, total: 10 })`
 * resolves `"Page {current} of {total}"` → `"Page 3 of 10"`.
 */
export declare class I18nManager implements Disposable {
    #private;
    constructor(initialLocale?: LocaleId);
    /** Current locale as a reactive signal. */
    get locale(): ReadonlySignal<LocaleId>;
    /** Switch the active locale. */
    setLocale(locale: LocaleId): void;
    /**
     * Add or merge translations for a locale.
     * Existing keys for the same locale are overwritten.
     */
    addTranslations(locale: LocaleId, translations: TranslationMap): void;
    /** Check if a locale has been registered. */
    hasLocale(locale: LocaleId): boolean;
    /** Get all registered locale IDs. */
    getLocales(): LocaleId[];
    /**
     * Resolve a translation key to a string.
     *
     * Resolution order:
     * 1. Current locale's translations
     * 2. Fallback locale (English)
     * 3. The key itself (for development — missing keys are visible)
     *
     * @param key The translation key (e.g., 'toolbar.save')
     * @param params Optional interpolation values (e.g., `{ value: 42 }`)
     */
    t(key: string, params?: InterpolationParams): string;
    [Symbol.dispose](): void;
}
//# sourceMappingURL=i18n-manager.d.ts.map