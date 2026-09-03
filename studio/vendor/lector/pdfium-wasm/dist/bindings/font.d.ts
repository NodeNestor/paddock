import type { FpdfDocument, FpdfFont, FpdfGlyphPath, FpdfPageObject, FpdfPathSegment, WasmPointer } from '../types/handles.js';
export declare const fontDescriptor: {
    readonly FPDFFont_Close: readonly [null, readonly ["number"]];
    readonly FPDFFont_GetBaseFontName: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFFont_GetFamilyName: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFFont_GetFontData: readonly ["number", readonly ["number", "number", "number", "number"]];
    readonly FPDFFont_GetIsEmbedded: readonly ["number", readonly ["number"]];
    readonly FPDFFont_GetFlags: readonly ["number", readonly ["number"]];
    readonly FPDFFont_GetWeight: readonly ["number", readonly ["number"]];
    readonly FPDFFont_GetItalicAngle: readonly ["number", readonly ["number", "number"]];
    readonly FPDFFont_GetAscent: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFFont_GetDescent: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFFont_GetGlyphWidth: readonly ["number", readonly ["number", "number", "number", "number"]];
    readonly FPDFFont_GetGlyphPath: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFGlyphPath_CountGlyphSegments: readonly ["number", readonly ["number"]];
    readonly FPDFGlyphPath_GetGlyphPathSegment: readonly ["number", readonly ["number", "number"]];
    readonly FPDFText_LoadFont: readonly ["number", readonly ["number", "number", "number", "number", "number"]];
    readonly FPDFText_LoadStandardFont: readonly ["number", readonly ["number", "number"]];
    readonly FPDFText_LoadCidType2Font: readonly ["number", readonly ["number", "number", "number", "number", "number", "number"]];
    readonly FPDFText_SetText: readonly ["number", readonly ["number", "number"]];
    readonly FPDFText_SetCharcodes: readonly ["number", readonly ["number", "number", "number"]];
};
export interface FontBindings {
    FPDFFont_Close(font: FpdfFont): void;
    FPDFFont_GetBaseFontName(font: FpdfFont, buffer: WasmPointer, length: number): number;
    FPDFFont_GetFamilyName(font: FpdfFont, buffer: WasmPointer, length: number): number;
    FPDFFont_GetFontData(font: FpdfFont, buffer: WasmPointer, buflen: number, outBuflen: WasmPointer): number;
    FPDFFont_GetIsEmbedded(font: FpdfFont): number;
    FPDFFont_GetFlags(font: FpdfFont): number;
    FPDFFont_GetWeight(font: FpdfFont): number;
    FPDFFont_GetItalicAngle(font: FpdfFont, angle: WasmPointer): number;
    FPDFFont_GetAscent(font: FpdfFont, fontSize: number, ascent: WasmPointer): number;
    FPDFFont_GetDescent(font: FpdfFont, fontSize: number, descent: WasmPointer): number;
    FPDFFont_GetGlyphWidth(font: FpdfFont, glyph: number, fontSize: number, width: WasmPointer): number;
    FPDFFont_GetGlyphPath(font: FpdfFont, glyph: number, fontSize: number): FpdfGlyphPath;
    FPDFGlyphPath_CountGlyphSegments(glyphpath: FpdfGlyphPath): number;
    FPDFGlyphPath_GetGlyphPathSegment(glyphpath: FpdfGlyphPath, index: number): FpdfPathSegment;
    FPDFText_LoadFont(document: FpdfDocument, data: WasmPointer, size: number, fontType: number, cid: number): FpdfFont;
    FPDFText_LoadStandardFont(document: FpdfDocument, font: WasmPointer): FpdfFont;
    FPDFText_LoadCidType2Font(document: FpdfDocument, fontData: WasmPointer, fontDataSize: number, toUnicodeCmap: WasmPointer, cidToGidMapData: WasmPointer, cidToGidMapDataSize: number): FpdfFont;
    FPDFText_SetText(textObject: FpdfPageObject, text: WasmPointer): number;
    FPDFText_SetCharcodes(textObject: FpdfPageObject, charcodes: WasmPointer, count: number): number;
}
//# sourceMappingURL=font.d.ts.map