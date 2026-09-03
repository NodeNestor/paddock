/** Locale identifier (BCP 47 tag, e.g., 'en', 'sv-SE', 'de'). */
export type LocaleId = string;
/** A flat map of translation key → translated string. */
export type TranslationMap = Readonly<Record<string, string>>;
/** Interpolation parameters for parameterized strings (e.g., `{value}%`). */
export type InterpolationParams = Readonly<Record<string, string | number>>;
//# sourceMappingURL=types.d.ts.map