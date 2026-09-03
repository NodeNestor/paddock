import type { FpdfDocument, FpdfFormHandle, PdfiumInstance } from '@truespar/lector-pdfium-wasm';
import type { WidgetData } from '../data/types.js';
/** Read all form fields (widget annotations) from a page. */
export declare function readPageFormFields(pdfium: PdfiumInstance, docHandle: FpdfDocument, formHandle: FpdfFormHandle, pageIndex: number): WidgetData[];
/** Set a form field value by field name. */
export declare function setFormFieldValue(pdfium: PdfiumInstance, docHandle: FpdfDocument, formHandle: FpdfFormHandle, pageIndex: number, fieldName: string, value: string): void;
/** Set a combobox/listbox selection by option index and annotation index. */
export declare function setComboBoxByIndex(pdfium: PdfiumInstance, docHandle: FpdfDocument, formHandle: FpdfFormHandle, pageIndex: number, annotIndex: number, optionIndex: number): void;
/**
 * Simulate a mouse click on a form widget using the C++ cached form
 * handle — the SAME one that lector_render_form_widgets uses for
 * FPDF_FFLDraw. This ensures the click updates the widget state that
 * the render path sees.
 */
export declare function clickFormWidget(pdfium: PdfiumInstance, docHandle: FpdfDocument, formHandle: FpdfFormHandle, pageIndex: number, pageX: number, pageY: number): void;
//# sourceMappingURL=form-ops.d.ts.map