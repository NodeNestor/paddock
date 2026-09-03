/** Raw pointer into WASM linear memory. */
export type WasmPointer = number & {
    readonly __brand: 'WasmPointer';
};
/** Opaque handle to a pdfium document. */
export type FpdfDocument = number & {
    readonly __brand: 'FPDF_DOCUMENT';
};
/** Opaque handle to a pdfium page. */
export type FpdfPage = number & {
    readonly __brand: 'FPDF_PAGE';
};
/** Opaque handle to a pdfium bitmap. */
export type FpdfBitmap = number & {
    readonly __brand: 'FPDF_BITMAP';
};
/** Opaque handle to a pdfium text page (text extraction context). */
export type FpdfTextPage = number & {
    readonly __brand: 'FPDF_TEXTPAGE';
};
/** Opaque handle to a pdfium search context. */
export type FpdfSchHandle = number & {
    readonly __brand: 'FPDF_SCHHANDLE';
};
/** Opaque handle to a pdfium form fill environment. */
export type FpdfFormHandle = number & {
    readonly __brand: 'FPDF_FORMHANDLE';
};
/** Opaque handle to a pdfium annotation. */
export type FpdfAnnotation = number & {
    readonly __brand: 'FPDF_ANNOTATION';
};
/** Opaque handle to a pdfium action. */
export type FpdfAction = number & {
    readonly __brand: 'FPDF_ACTION';
};
/** Opaque handle to a pdfium destination. */
export type FpdfDest = number & {
    readonly __brand: 'FPDF_DEST';
};
/** Opaque handle to a pdfium bookmark. */
export type FpdfBookmark = number & {
    readonly __brand: 'FPDF_BOOKMARK';
};
/** Opaque handle to a pdfium link. */
export type FpdfLink = number & {
    readonly __brand: 'FPDF_LINK';
};
/** Opaque handle to a pdfium page link (web links from text analysis). */
export type FpdfPageLink = number & {
    readonly __brand: 'FPDF_PAGELINK';
};
/** Opaque handle to a pdfium page object (text, path, image, shading, form). */
export type FpdfPageObject = number & {
    readonly __brand: 'FPDF_PAGEOBJECT';
};
/** Opaque handle to a pdfium page object mark (tagged content). */
export type FpdfPageObjectMark = number & {
    readonly __brand: 'FPDF_PAGEOBJECTMARK';
};
/** Opaque handle to a pdfium font. */
export type FpdfFont = number & {
    readonly __brand: 'FPDF_FONT';
};
/** Opaque handle to a pdfium glyph path. */
export type FpdfGlyphPath = number & {
    readonly __brand: 'FPDF_GLYPHPATH';
};
/** Opaque handle to a pdfium path segment. */
export type FpdfPathSegment = number & {
    readonly __brand: 'FPDF_PATHSEGMENT';
};
/** Opaque handle to a pdfium clip path. */
export type FpdfClipPath = number & {
    readonly __brand: 'FPDF_CLIPPATH';
};
/** Opaque handle to a pdfium page range. */
export type FpdfPageRange = number & {
    readonly __brand: 'FPDF_PAGERANGE';
};
/** Opaque handle to a pdfium digital signature. */
export type FpdfSignature = number & {
    readonly __brand: 'FPDF_SIGNATURE';
};
/** Opaque handle to a pdfium file attachment. */
export type FpdfAttachment = number & {
    readonly __brand: 'FPDF_ATTACHMENT';
};
/** Opaque handle to a pdfium structure tree. */
export type FpdfStructTree = number & {
    readonly __brand: 'FPDF_STRUCTTREE';
};
/** Opaque handle to a pdfium structure element. */
export type FpdfStructElement = number & {
    readonly __brand: 'FPDF_STRUCTELEMENT';
};
/** Opaque handle to a pdfium structure element attribute. */
export type FpdfStructElementAttr = number & {
    readonly __brand: 'FPDF_STRUCTELEMENT_ATTR';
};
/** Opaque handle to a pdfium structure element attribute value. */
export type FpdfStructElementAttrValue = number & {
    readonly __brand: 'FPDF_STRUCTELEMENT_ATTR_VALUE';
};
/** Opaque handle to a pdfium data availability context. */
export type FpdfAvail = number & {
    readonly __brand: 'FPDF_AVAIL';
};
/** Opaque handle to a pdfium XObject. */
export type FpdfXObject = number & {
    readonly __brand: 'FPDF_XOBJECT';
};
/** Opaque handle to a pdfium JavaScript action. */
export type FpdfJavaScriptAction = number & {
    readonly __brand: 'FPDF_JAVASCRIPT_ACTION';
};
/** Opaque handle to a pdfium widget (form field). */
export type FpdfWidget = number & {
    readonly __brand: 'FPDF_WIDGET';
};
/** Opaque handle to a pdfium system font info. */
export type FpdfSystemFontInfo = number & {
    readonly __brand: 'FPDF_SYSFONTINFO';
};
//# sourceMappingURL=handles.d.ts.map