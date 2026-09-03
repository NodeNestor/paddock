// src/memory/allocator.ts
var WasmAlloc = class {
  ptr;
  size;
  #module;
  #freed = false;
  constructor(module, size) {
    const ptr = module._malloc(size);
    if (ptr === 0) {
      throw new Error(`Failed to allocate ${size} bytes on WASM heap`);
    }
    this.ptr = ptr;
    this.size = size;
    this.#module = module;
  }
  [Symbol.dispose]() {
    if (!this.#freed) {
      this.#freed = true;
      this.#module._free(this.ptr);
    }
  }
};
function wasmAlloc(module, size) {
  return new WasmAlloc(module, size);
}
function wasmMalloc(module, size) {
  const ptr = module._malloc(size);
  if (ptr === 0) {
    throw new Error(`Failed to allocate ${size} bytes on WASM heap`);
  }
  return ptr;
}
function wasmFree(module, ptr) {
  if (ptr !== 0) {
    module._free(ptr);
  }
}

// src/memory/buffer.ts
function toHeap(module, data) {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const alloc = new WasmAlloc(module, bytes.byteLength);
  module.HEAPU8.set(bytes, alloc.ptr);
  return alloc;
}
function fromHeap(module, ptr, length) {
  return new Uint8Array(module.HEAPU8.buffer, ptr, length).slice();
}
function heapView(module, ptr, length) {
  return new Uint8Array(module.HEAPU8.buffer, ptr, length);
}

// src/memory/strings.ts
var encoder = new TextEncoder();
function toByteString(module, str) {
  const bytes = encoder.encode(str);
  const alloc = new WasmAlloc(module, bytes.byteLength + 1);
  module.stringToUTF8(str, alloc.ptr, bytes.byteLength + 1);
  return alloc;
}
function toWideString(module, str) {
  const byteLen = (str.length + 1) * 2;
  const alloc = new WasmAlloc(module, byteLen);
  module.stringToUTF16(str, alloc.ptr, byteLen);
  return alloc;
}
function fromByteString(module, ptr) {
  return module.UTF8ToString(ptr);
}
function fromWideString(module, ptr) {
  return module.UTF16ToString(ptr);
}

// src/memory/structs.ts
function readRectF(module, ptr) {
  const i = ptr >> 2;
  return {
    left: module.HEAPF32[i],
    top: module.HEAPF32[i + 1],
    right: module.HEAPF32[i + 2],
    bottom: module.HEAPF32[i + 3]
  };
}
function writeRectF(module, ptr, rect) {
  const i = ptr >> 2;
  module.HEAPF32[i] = rect.left;
  module.HEAPF32[i + 1] = rect.top;
  module.HEAPF32[i + 2] = rect.right;
  module.HEAPF32[i + 3] = rect.bottom;
}
function readMatrix(module, ptr) {
  const i = ptr >> 2;
  return {
    a: module.HEAPF32[i],
    b: module.HEAPF32[i + 1],
    c: module.HEAPF32[i + 2],
    d: module.HEAPF32[i + 3],
    e: module.HEAPF32[i + 4],
    f: module.HEAPF32[i + 5]
  };
}
function writeMatrix(module, ptr, matrix) {
  const i = ptr >> 2;
  module.HEAPF32[i] = matrix.a;
  module.HEAPF32[i + 1] = matrix.b;
  module.HEAPF32[i + 2] = matrix.c;
  module.HEAPF32[i + 3] = matrix.d;
  module.HEAPF32[i + 4] = matrix.e;
  module.HEAPF32[i + 5] = matrix.f;
}
function readSizeF(module, ptr) {
  const i = ptr >> 2;
  return {
    width: module.HEAPF32[i],
    height: module.HEAPF32[i + 1]
  };
}
function writeSizeF(module, ptr, size) {
  const i = ptr >> 2;
  module.HEAPF32[i] = size.width;
  module.HEAPF32[i + 1] = size.height;
}
function readPointF(module, ptr) {
  const i = ptr >> 2;
  return {
    x: module.HEAPF32[i],
    y: module.HEAPF32[i + 1]
  };
}
function writePointF(module, ptr, point) {
  const i = ptr >> 2;
  module.HEAPF32[i] = point.x;
  module.HEAPF32[i + 1] = point.y;
}
function readQuadPointsF(module, ptr) {
  const i = ptr >> 2;
  return {
    x1: module.HEAPF32[i],
    y1: module.HEAPF32[i + 1],
    x2: module.HEAPF32[i + 2],
    y2: module.HEAPF32[i + 3],
    x3: module.HEAPF32[i + 4],
    y3: module.HEAPF32[i + 5],
    x4: module.HEAPF32[i + 6],
    y4: module.HEAPF32[i + 7]
  };
}
function writeQuadPointsF(module, ptr, qp) {
  const i = ptr >> 2;
  module.HEAPF32[i] = qp.x1;
  module.HEAPF32[i + 1] = qp.y1;
  module.HEAPF32[i + 2] = qp.x2;
  module.HEAPF32[i + 3] = qp.y2;
  module.HEAPF32[i + 4] = qp.x3;
  module.HEAPF32[i + 5] = qp.y3;
  module.HEAPF32[i + 6] = qp.x4;
  module.HEAPF32[i + 7] = qp.y4;
}

// src/lifecycle.ts
async function createPdfiumInstance(createModule, config) {
  const module = await createModule(config);
  const fn = module;
  fn._FPDF_InitLibrary();
  const memory = {
    alloc: (size) => new WasmAlloc(module, size),
    toByteString: (str) => toByteString(module, str),
    toWideString: (str) => toWideString(module, str),
    fromByteString: (ptr) => fromByteString(module, ptr),
    fromWideString: (ptr) => fromWideString(module, ptr),
    toHeap: (data) => toHeap(module, data),
    fromHeap: (ptr, length) => fromHeap(module, ptr, length),
    heapView: (ptr, length) => heapView(module, ptr, length),
    readRectF: (ptr) => readRectF(module, ptr),
    writeRectF: (ptr, rect) => writeRectF(module, ptr, rect),
    readMatrix: (ptr) => readMatrix(module, ptr),
    writeMatrix: (ptr, matrix) => writeMatrix(module, ptr, matrix),
    readSizeF: (ptr) => readSizeF(module, ptr),
    writeSizeF: (ptr, size) => writeSizeF(module, ptr, size),
    readPointF: (ptr) => readPointF(module, ptr),
    writePointF: (ptr, point) => writePointF(module, ptr, point),
    readQuadPointsF: (ptr) => readQuadPointsF(module, ptr),
    writeQuadPointsF: (ptr, qp) => writeQuadPointsF(module, ptr, qp)
  };
  let destroyed = false;
  return {
    module,
    fn,
    memory,
    [Symbol.dispose]() {
      if (!destroyed) {
        destroyed = true;
        fn._FPDF_DestroyLibrary();
      }
    }
  };
}

// src/types/enums.ts
var FpdfError = {
  SUCCESS: 0,
  UNKNOWN: 1,
  FILE: 2,
  FORMAT: 3,
  PASSWORD: 4,
  SECURITY: 5,
  PAGE: 6,
  XFALOAD: 7,
  XFALAYOUT: 8
};
var FpdfRenderFlag = {
  ANNOT: 1,
  LCD_TEXT: 2,
  NO_NATIVETEXT: 4,
  GRAYSCALE: 8,
  REVERSE_BYTE_ORDER: 16,
  CONVERT_FILL_TO_STROKE: 32,
  NO_CATCH: 256,
  LIMITED_IMAGE_CACHE: 512,
  FORCE_HALFTONE: 1024,
  PRINTING: 2048,
  NO_SMOOTHTEXT: 4096,
  NO_SMOOTHIMAGE: 8192,
  NO_SMOOTHPATH: 16384
};
var FpdfBitmapFormat = {
  UNKNOWN: 0,
  GRAY: 1,
  BGR: 2,
  BGRX: 3,
  BGRA: 4,
  BGRA_PREMUL: 5
};
var FpdfAnnotSubtype = {
  UNKNOWN: 0,
  TEXT: 1,
  LINK: 2,
  FREETEXT: 3,
  LINE: 4,
  SQUARE: 5,
  CIRCLE: 6,
  POLYGON: 7,
  POLYLINE: 8,
  HIGHLIGHT: 9,
  UNDERLINE: 10,
  SQUIGGLY: 11,
  STRIKEOUT: 12,
  STAMP: 13,
  CARET: 14,
  INK: 15,
  POPUP: 16,
  FILEATTACHMENT: 17,
  SOUND: 18,
  MOVIE: 19,
  WIDGET: 20,
  SCREEN: 21,
  PRINTERMARK: 22,
  TRAPNET: 23,
  WATERMARK: 24,
  THREED: 25,
  RICHMEDIA: 26,
  XFAWIDGET: 27,
  REDACT: 28
};
var FpdfAnnotFlag = {
  NONE: 0,
  INVISIBLE: 1 << 0,
  HIDDEN: 1 << 1,
  PRINT: 1 << 2,
  NOZOOM: 1 << 3,
  NOROTATE: 1 << 4,
  NOVIEW: 1 << 5,
  READONLY: 1 << 6,
  LOCKED: 1 << 7,
  TOGGLENOVIEW: 1 << 8
};
var FpdfAnnotAppearanceMode = {
  NORMAL: 0,
  ROLLOVER: 1,
  DOWN: 2
};
var FpdfAnnotAAction = {
  KEY_STROKE: 12,
  FORMAT: 13,
  VALIDATE: 14,
  CALCULATE: 15
};
var FpdfAnnotColorType = {
  COLOR: 0,
  INTERIOR_COLOR: 1
};
var FpdfFormFieldType = {
  UNKNOWN: 0,
  PUSHBUTTON: 1,
  CHECKBOX: 2,
  RADIOBUTTON: 3,
  COMBOBOX: 4,
  LISTBOX: 5,
  TEXTFIELD: 6,
  SIGNATURE: 7
};
var FpdfFormType = {
  NONE: 0,
  ACRO_FORM: 1,
  XFA_FULL: 2,
  XFA_FOREGROUND: 3
};
var FpdfActionType = {
  UNSUPPORTED: 0,
  GOTO: 1,
  REMOTEGOTO: 2,
  URI: 3,
  LAUNCH: 4,
  EMBEDDEDGOTO: 5
};
var FpdfPageObjectType = {
  UNKNOWN: 0,
  TEXT: 1,
  PATH: 2,
  IMAGE: 3,
  SHADING: 4,
  FORM: 5
};
var FpdfSegmentType = {
  UNKNOWN: -1,
  LINETO: 0,
  BEZIERTO: 1,
  MOVETO: 2
};
var FpdfFillMode = {
  NONE: 0,
  ALTERNATE: 1,
  WINDING: 2
};
var FpdfLineCap = {
  BUTT: 0,
  ROUND: 1,
  PROJECTING_SQUARE: 2
};
var FpdfLineJoin = {
  MITER: 0,
  ROUND: 1,
  BEVEL: 2
};
var FpdfTextRenderMode = {
  UNKNOWN: -1,
  FILL: 0,
  STROKE: 1,
  FILL_STROKE: 2,
  INVISIBLE: 3,
  FILL_CLIP: 4,
  STROKE_CLIP: 5,
  FILL_STROKE_CLIP: 6,
  CLIP: 7
};
var FpdfDuplexType = {
  UNDEFINED: 0,
  SIMPLEX: 1,
  DUPLEX_FLIP_SHORT_EDGE: 2,
  DUPLEX_FLIP_LONG_EDGE: 3
};
var FpdfFlattenResult = {
  FAIL: 0,
  SUCCESS: 1,
  NOTHING_TO_DO: 2
};
var FpdfPageMode = {
  UNKNOWN: -1,
  USE_NONE: 0,
  USE_OUTLINES: 1,
  USE_THUMBS: 2,
  FULL_SCREEN: 3,
  USE_OC: 4,
  USE_ATTACHMENTS: 5
};
var FpdfObjectType = {
  UNKNOWN: 0,
  BOOLEAN: 1,
  NUMBER: 2,
  STRING: 3,
  NAME: 4,
  ARRAY: 5,
  DICTIONARY: 6,
  STREAM: 7,
  NULLOBJ: 8,
  REFERENCE: 9
};
var FpdfUnsupportedType = {
  DOC_XFAFORM: 1,
  DOC_PORTABLECOLLECTION: 2,
  DOC_ATTACHMENT: 3,
  DOC_SECURITY: 4,
  DOC_SHAREDREVIEW: 5,
  DOC_SHAREDFORM_ACROBAT: 6,
  DOC_SHAREDFORM_FILESYSTEM: 7,
  DOC_SHAREDFORM_EMAIL: 8,
  ANNOT_3DANNOT: 11,
  ANNOT_MOVIE: 12,
  ANNOT_SOUND: 13,
  ANNOT_SCREEN_MEDIA: 14,
  ANNOT_SCREEN_RICHMEDIA: 15,
  ANNOT_ATTACHMENT: 16,
  ANNOT_SIG: 17
};

// src/types/structs.ts
var FS_RECTF_SIZE = 16;
var FS_MATRIX_SIZE = 24;
var FS_SIZEF_SIZE = 8;
var FS_POINTF_SIZE = 8;
var FS_QUADPOINTSF_SIZE = 32;

// src/error.ts
var ERROR_MESSAGES = {
  [FpdfError.SUCCESS]: "Success",
  [FpdfError.UNKNOWN]: "Unknown error",
  [FpdfError.FILE]: "File not found or could not be opened",
  [FpdfError.FORMAT]: "File not in PDF format or corrupted",
  [FpdfError.PASSWORD]: "Password required or incorrect password",
  [FpdfError.SECURITY]: "Unsupported security scheme",
  [FpdfError.PAGE]: "Page not found or content error",
  [FpdfError.XFALOAD]: "XFA load error",
  [FpdfError.XFALAYOUT]: "XFA layout error"
};
var PdfiumError = class extends Error {
  code;
  constructor(code, context) {
    const base = ERROR_MESSAGES[code] ?? `Error code ${code}`;
    super(context ? `${context}: ${base}` : base);
    this.name = "PdfiumError";
    this.code = code;
  }
};
function checkBool(result, getLastErrorFn, context) {
  if (result === 0) {
    throw new PdfiumError(getLastErrorFn(), context);
  }
}
function checkHandle(handle, getLastErrorFn, context) {
  if (handle === 0) {
    throw new PdfiumError(getLastErrorFn(), context);
  }
}
export {
  FS_MATRIX_SIZE,
  FS_POINTF_SIZE,
  FS_QUADPOINTSF_SIZE,
  FS_RECTF_SIZE,
  FS_SIZEF_SIZE,
  FpdfActionType,
  FpdfAnnotAAction,
  FpdfAnnotAppearanceMode,
  FpdfAnnotColorType,
  FpdfAnnotFlag,
  FpdfAnnotSubtype,
  FpdfBitmapFormat,
  FpdfDuplexType,
  FpdfError,
  FpdfFillMode,
  FpdfFlattenResult,
  FpdfFormFieldType,
  FpdfFormType,
  FpdfLineCap,
  FpdfLineJoin,
  FpdfObjectType,
  FpdfPageMode,
  FpdfPageObjectType,
  FpdfRenderFlag,
  FpdfSegmentType,
  FpdfTextRenderMode,
  FpdfUnsupportedType,
  PdfiumError,
  WasmAlloc,
  checkBool,
  checkHandle,
  createPdfiumInstance,
  fromByteString,
  fromHeap,
  fromWideString,
  heapView,
  readMatrix,
  readPointF,
  readQuadPointsF,
  readRectF,
  readSizeF,
  toByteString,
  toHeap,
  toWideString,
  wasmAlloc,
  wasmFree,
  wasmMalloc,
  writeMatrix,
  writePointF,
  writeQuadPointsF,
  writeRectF,
  writeSizeF
};
