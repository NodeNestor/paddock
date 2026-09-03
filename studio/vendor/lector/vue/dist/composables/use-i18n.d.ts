/** Composable for internationalization. */
export declare function useI18n(): {
    locale: Readonly<import("vue").Ref<string, string>>;
    t: (key: string, params?: import("@truespar/lector-core").InterpolationParams) => string;
    setLocale: (locale: import("@truespar/lector-core").LocaleId) => void;
    addTranslations: (locale: import("@truespar/lector-core").LocaleId, translations: import("@truespar/lector-core").TranslationMap) => void;
    hasLocale: (locale: import("@truespar/lector-core").LocaleId) => boolean;
    getLocales: () => import("@truespar/lector-core").LocaleId[];
};
//# sourceMappingURL=use-i18n.d.ts.map