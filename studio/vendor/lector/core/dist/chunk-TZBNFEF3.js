// src/ui/lector-pane.ts
import { effect as effect2 } from "@truespar/lector-utils";

// src/types/render.ts
var RenderPriority = {
  /** Pages currently visible in the viewport. */
  VISIBLE: 0,
  /** Pages in the pre-render buffer zone adjacent to visible pages. */
  BUFFER: 1,
  /** Low-priority renders (thumbnails, prefetch). */
  LOW: 2
};
var DEFAULT_RENDER_OPTIONS = {
  flags: 2,
  // LCD_TEXT only — annotations are rendered as overlays
  rotation: 0,
  devicePixelRatio: 1,
  backgroundColor: 4294967295
  // opaque white
};

// src/ui/page-overlays.ts
import { effect } from "@truespar/lector-utils";

// src/plugins/measurement-plugin.ts
import { signal, computed } from "@truespar/lector-utils";

// src/plugin/define-plugin.ts
function definePlugin(definition) {
  return definition;
}

// src/data/types.ts
var LineCap = {
  NONE: 0,
  SQUARE: 1,
  CIRCLE: 2,
  DIAMOND: 3,
  OPEN_ARROW: 4,
  CLOSED_ARROW: 5,
  BUTT: 6,
  REVERSE_OPEN_ARROW: 7,
  REVERSE_CLOSED_ARROW: 8,
  SLASH: 9
};
var BlendMode = {
  NORMAL: "Normal",
  MULTIPLY: "Multiply",
  SCREEN: "Screen",
  OVERLAY: "Overlay",
  DARKEN: "Darken",
  LIGHTEN: "Lighten",
  COLOR_DODGE: "ColorDodge",
  COLOR_BURN: "ColorBurn",
  HARD_LIGHT: "HardLight",
  SOFT_LIGHT: "SoftLight",
  DIFFERENCE: "Difference",
  EXCLUSION: "Exclusion"
};
var NoteIcon = {
  COMMENT: "Comment",
  RIGHT_POINTER: "RightPointer",
  RIGHT_ARROW: "RightArrow",
  CHECK: "Check",
  CIRCLE: "Circle",
  CROSS: "Cross",
  INSERT: "Insert",
  NEW_PARAGRAPH: "NewParagraph",
  NOTE: "Note",
  PARAGRAPH: "Paragraph",
  HELP: "Help",
  STAR: "Star",
  KEY: "Key"
};
var MeasurementUnit = {
  PT: "pt",
  MM: "mm",
  IN: "in",
  CM: "cm",
  M: "m",
  FT: "ft",
  YD: "yd"
};

// src/plugins/measurement-plugin.ts
var POINTS_PER_UNIT = {
  pt: 1,
  in: 1 / 72,
  mm: 25.4 / 72,
  cm: 2.54 / 72,
  m: 0.0254 / 72,
  ft: 1 / (72 * 12),
  yd: 1 / (72 * 36)
};
function convertPointsToUnit(valuePt, unit) {
  return valuePt * POINTS_PER_UNIT[unit];
}
function convertLengthWithScale(valuePt, scale, fallbackUnit) {
  if (!scale || scale.source === 1 && scale.target === 1 && scale.sourceUnit === scale.targetUnit) {
    return { value: convertPointsToUnit(valuePt, fallbackUnit), unit: fallbackUnit };
  }
  const valueInSourceUnit = convertPointsToUnit(valuePt, scale.sourceUnit);
  const realValue = valueInSourceUnit / scale.source * scale.target;
  return { value: realValue, unit: scale.targetUnit };
}
function convertAreaWithScale(valuePt2, scale, fallbackUnit) {
  if (!scale || scale.source === 1 && scale.target === 1 && scale.sourceUnit === scale.targetUnit) {
    const linear = POINTS_PER_UNIT[fallbackUnit];
    return { value: valuePt2 * linear * linear, unit: fallbackUnit };
  }
  const linearInSource = POINTS_PER_UNIT[scale.sourceUnit];
  const areaInSource = valuePt2 * linearInSource * linearInSource;
  const ratio = scale.target / scale.source;
  return { value: areaInSource * ratio * ratio, unit: scale.targetUnit };
}
var measurementPlugin = definePlugin({
  id: "measurement",
  provides: ["measurement"],
  requires: ["annotation"],
  optional: ["i18n"],
  setup(ctx) {
    ctx.require("annotation");
    const scale$ = signal(null);
    const unit$ = signal(MeasurementUnit.CM);
    const precision$ = signal(2);
    function toUnit(ptValue) {
      const sc = scale$.peek();
      if (!sc) {
        const inches = ptValue / 72;
        switch (unit$.peek()) {
          case MeasurementUnit.PT:
            return ptValue;
          case MeasurementUnit.IN:
            return inches;
          case MeasurementUnit.MM:
            return inches * 25.4;
          case MeasurementUnit.CM:
            return inches * 2.54;
          case MeasurementUnit.M:
            return inches * 0.0254;
          case MeasurementUnit.FT:
            return inches / 12;
          case MeasurementUnit.YD:
            return inches / 36;
          default:
            return ptValue;
        }
      }
      const ratio = sc.target / sc.source;
      return ptValue * ratio;
    }
    return {
      setScale(scale) {
        scale$.value = scale;
        ctx.emit("measurement:scale-changed", scale);
      },
      getScale() {
        return scale$.peek();
      },
      activeUnit: computed(() => unit$.value),
      setActiveUnit(unit) {
        unit$.value = unit;
      },
      setPrecision(p) {
        precision$.value = Math.max(0, Math.min(6, p));
      },
      precision: computed(() => precision$.value),
      calculateDistance(p1, p2) {
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        return toUnit(Math.sqrt(dx * dx + dy * dy));
      },
      calculateArea(vertices) {
        let area = 0;
        const n = vertices.length;
        for (let i = 0; i < n; i++) {
          const j = (i + 1) % n;
          area += vertices[i].x * vertices[j].y;
          area -= vertices[j].x * vertices[i].y;
        }
        area = Math.abs(area) / 2;
        const linearScale = toUnit(1);
        return area * linearScale * linearScale;
      },
      calculatePerimeter(vertices, closed = false) {
        let total = 0;
        for (let i = 0; i < vertices.length - 1; i++) {
          const dx = vertices[i + 1].x - vertices[i].x;
          const dy = vertices[i + 1].y - vertices[i].y;
          total += Math.sqrt(dx * dx + dy * dy);
        }
        if (closed && vertices.length > 2) {
          const first = vertices[0];
          const last = vertices[vertices.length - 1];
          total += Math.sqrt((first.x - last.x) ** 2 + (first.y - last.y) ** 2);
        }
        return toUnit(total);
      },
      convert(valuePt) {
        return toUnit(valuePt);
      },
      format(valuePt) {
        const val = toUnit(valuePt);
        const p = precision$.peek();
        return `${val.toFixed(p)} ${unit$.peek()}`;
      }
    };
  }
});

// src/ui/page-overlays.ts
import { FpdfAnnotSubtype as FpdfAnnotSubtype2 } from "@truespar/lector-pdfium-wasm";

// src/utils/uuid.ts
function uuid() {
  const c = crypto;
  if (c.randomUUID) return c.randomUUID();
  const b = c.getRandomValues(new Uint8Array(16));
  b[6] = b[6] & 15 | 64;
  b[8] = b[8] & 63 | 128;
  let h = "";
  for (let i = 0; i < 16; i++) h += b[i].toString(16).padStart(2, "0");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

// src/plugins/annotation-tools.ts
import { FpdfAnnotSubtype } from "@truespar/lector-pdfium-wasm";

// src/ui/page-viewport.ts
var PageViewport = class _PageViewport {
  /** Page width in unrotated PDF points. */
  unrotatedWidthPts;
  /** Page height in unrotated PDF points. */
  unrotatedHeightPts;
  rotation;
  scale;
  /** Overlay width in CSS px (rotated, scaled). */
  width;
  /** Overlay height in CSS px (rotated, scaled). */
  height;
  /** PDF user space → CSS px. `px = a·x + c·y + e`, `py = b·x + d·y + f`. */
  matrix;
  /** CSS px → PDF user space (inverse of {@link matrix}). */
  #inv;
  /**
   * @param unrotatedWidthPts  Page width in PDF points, before rotation.
   * @param unrotatedHeightPts Page height in PDF points, before rotation.
   * @param rotation           Page rotation (0/1/2/3 = 0/90/180/270°).
   * @param scale              CSS px per PDF point.
   */
  constructor(unrotatedWidthPts, unrotatedHeightPts, rotation, scale) {
    this.unrotatedWidthPts = unrotatedWidthPts;
    this.unrotatedHeightPts = unrotatedHeightPts;
    this.rotation = rotation;
    this.scale = scale;
    const Wu = unrotatedWidthPts;
    const Hu = unrotatedHeightPts;
    const s = scale;
    let m;
    switch (rotation) {
      case 0:
        m = [s, 0, 0, -s, 0, Hu * s];
        this.width = Wu * s;
        this.height = Hu * s;
        break;
      case 1:
        m = [0, s, s, 0, 0, 0];
        this.width = Hu * s;
        this.height = Wu * s;
        break;
      case 2:
        m = [-s, 0, 0, s, Wu * s, 0];
        this.width = Wu * s;
        this.height = Hu * s;
        break;
      case 3:
        m = [0, -s, -s, 0, Hu * s, Wu * s];
        this.width = Hu * s;
        this.height = Wu * s;
        break;
    }
    this.matrix = m;
    this.#inv = invert(m);
  }
  /**
   * Build a viewport from the page's *rotated* dimensions — i.e. the size
   * pdfium reports for the page in its current rotation. The unrotated size is
   * recovered by un-swapping width/height at 90°/270°.
   */
  static fromRotatedSize(rotatedWidthPts, rotatedHeightPts, rotation, scale) {
    const swapped = rotation === 1 || rotation === 3;
    const Wu = swapped ? rotatedHeightPts : rotatedWidthPts;
    const Hu = swapped ? rotatedWidthPts : rotatedHeightPts;
    return new _PageViewport(Wu, Hu, rotation, scale);
  }
  /** Map a PDF user-space point to CSS px within the overlay. */
  pointToCss(x, y) {
    const [a, b, c, d, e, f] = this.matrix;
    return { x: a * x + c * y + e, y: b * x + d * y + f };
  }
  /**
   * Map a PDF user-space rect to its CSS axis-aligned bounding box. All four
   * corners are transformed and min/max'd, so the result is correct regardless
   * of rotation or corner ordering. For orthogonal rotations an axis-aligned
   * PDF rect maps to an axis-aligned CSS rect with no distortion.
   */
  rectToCss(rect) {
    const corners = [
      [rect.left, rect.bottom],
      [rect.left, rect.top],
      [rect.right, rect.bottom],
      [rect.right, rect.top]
    ];
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const [x, y] of corners) {
      const p = this.pointToCss(x, y);
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
  }
  /**
   * Map a CSS-pixel point (relative to the page overlay's top-left) back to
   * PDF user space. Used for hit-testing — e.g. converting a click to the PDF
   * coordinate where a new annotation should be created.
   */
  cssPointToPdf(px, py) {
    const [a, b, c, d, e, f] = this.#inv;
    return { x: a * px + c * py + e, y: b * px + d * py + f };
  }
  /**
   * Map a CSS-pixel delta (no translation) back to a PDF user-space delta.
   * Used for drag/resize: a pointer movement of (dx, dy) px becomes the
   * corresponding shift in PDF points, with rotation and Y-flip applied.
   */
  cssDeltaToPdf(dx, dy) {
    const [a, b, c, d] = this.#inv;
    return { x: a * dx + c * dy, y: b * dx + d * dy };
  }
};
function invert(m) {
  const [a, b, c, d, e, f] = m;
  const det = a * d - b * c;
  if (det === 0) {
    throw new Error("PageViewport: non-invertible matrix (scale must be non-zero)");
  }
  const ia = d / det;
  const ib = -b / det;
  const ic = -c / det;
  const id = a / det;
  const ie = (c * f - d * e) / det;
  const iff = (b * e - a * f) / det;
  return [ia, ib, ic, id, ie, iff];
}

// src/plugins/annotation-tools.ts
var ANNOTATION_TOOL_DEFAULTS = {
  "highlight": { color: { r: 255, g: 205, b: 69, a: 255 } },
  "underline": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "strikeout": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "squiggly": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "ink": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "ink-highlighter": { color: { r: 255, g: 205, b: 69, a: 255 }, opacity: 0.5 },
  "eraser": { color: { r: 255, g: 255, b: 255, a: 255 } },
  "freetext": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "sticky-note": { color: { r: 255, g: 205, b: 69, a: 255 } },
  "insert-text": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "callout": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "rectangle": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "circle": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "line": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "arrow": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "polygon": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "polyline": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "stamp": { color: { r: 228, g: 66, b: 52, a: 255 } },
  "image": { color: { r: 0, g: 0, b: 0, a: 0 } },
  "measure-distance": { color: { r: 59, g: 130, b: 246, a: 255 } },
  "measure-area": { color: { r: 59, g: 130, b: 246, a: 255 } },
  "measure-perimeter": { color: { r: 59, g: 130, b: 246, a: 255 } },
  "redaction": { color: { r: 0, g: 0, b: 0, a: 255 } }
};
var TOOL_TO_SUBTYPE = {
  "highlight": FpdfAnnotSubtype.HIGHLIGHT,
  "underline": FpdfAnnotSubtype.UNDERLINE,
  "strikeout": FpdfAnnotSubtype.STRIKEOUT,
  "squiggly": FpdfAnnotSubtype.SQUIGGLY,
  "ink": FpdfAnnotSubtype.INK,
  "ink-highlighter": FpdfAnnotSubtype.INK,
  "eraser": FpdfAnnotSubtype.INK,
  "freetext": FpdfAnnotSubtype.FREETEXT,
  "sticky-note": FpdfAnnotSubtype.TEXT,
  "insert-text": FpdfAnnotSubtype.CARET,
  "callout": FpdfAnnotSubtype.FREETEXT,
  "rectangle": FpdfAnnotSubtype.SQUARE,
  "circle": FpdfAnnotSubtype.CIRCLE,
  "line": FpdfAnnotSubtype.LINE,
  "arrow": FpdfAnnotSubtype.LINE,
  "polygon": FpdfAnnotSubtype.POLYGON,
  "polyline": FpdfAnnotSubtype.POLYLINE,
  "stamp": FpdfAnnotSubtype.STAMP,
  "image": FpdfAnnotSubtype.STAMP,
  "measure-distance": FpdfAnnotSubtype.LINE,
  "measure-area": FpdfAnnotSubtype.POLYGON,
  "measure-perimeter": FpdfAnnotSubtype.POLYLINE,
  "redaction": 28
  // REDACT subtype
};
var TOOL_BEHAVIORS = {
  "highlight": { selectAfterCreate: true, deactivateAfterCreate: true },
  "underline": { selectAfterCreate: true, deactivateAfterCreate: true },
  "strikeout": { selectAfterCreate: true, deactivateAfterCreate: true },
  "squiggly": { selectAfterCreate: true, deactivateAfterCreate: true },
  "ink": { selectAfterCreate: true, deactivateAfterCreate: true },
  "ink-highlighter": { selectAfterCreate: true, deactivateAfterCreate: true },
  "eraser": { selectAfterCreate: false, deactivateAfterCreate: false },
  "freetext": { selectAfterCreate: true, deactivateAfterCreate: true },
  "sticky-note": { selectAfterCreate: true, deactivateAfterCreate: true },
  "insert-text": { selectAfterCreate: true, deactivateAfterCreate: true },
  "callout": { selectAfterCreate: true, deactivateAfterCreate: true },
  "rectangle": { selectAfterCreate: true, deactivateAfterCreate: true },
  "circle": { selectAfterCreate: true, deactivateAfterCreate: true },
  "line": { selectAfterCreate: true, deactivateAfterCreate: true },
  "arrow": { selectAfterCreate: true, deactivateAfterCreate: true },
  "polygon": { selectAfterCreate: true, deactivateAfterCreate: true },
  "polyline": { selectAfterCreate: true, deactivateAfterCreate: true },
  "stamp": { selectAfterCreate: true, deactivateAfterCreate: true },
  "image": { selectAfterCreate: true, deactivateAfterCreate: true },
  "measure-distance": { selectAfterCreate: false, deactivateAfterCreate: true },
  "measure-area": { selectAfterCreate: false, deactivateAfterCreate: true },
  "measure-perimeter": { selectAfterCreate: false, deactivateAfterCreate: true },
  "redaction": { selectAfterCreate: false, deactivateAfterCreate: true }
};
function isMarkupTool(tool) {
  return tool === "highlight" || tool === "underline" || tool === "strikeout" || tool === "squiggly";
}
function isInkTool(tool) {
  return tool === "ink" || tool === "ink-highlighter";
}
function isEraserTool(tool) {
  return tool === "eraser";
}
function isShapeTool(tool) {
  return tool === "rectangle" || tool === "circle" || tool === "line" || tool === "arrow" || tool === "measure-distance" || tool === "redaction" || tool === "freetext";
}
function isPolygonTool(tool) {
  return tool === "polygon" || tool === "polyline" || tool === "measure-area" || tool === "measure-perimeter";
}
function isPlacementTool(tool) {
  return tool === "sticky-note" || tool === "insert-text" || tool === "stamp";
}
function isCalloutTool(tool) {
  return tool === "callout";
}
function isImageTool(tool) {
  return tool === "image";
}
function isMeasurementTool(tool) {
  return tool === "measure-distance" || tool === "measure-area" || tool === "measure-perimeter";
}
function isStampTool(tool) {
  return tool === "stamp";
}
function isRedactionTool(tool) {
  return tool === "redaction";
}
function isToolOutputTool(tool) {
  return isMeasurementTool(tool) || isRedactionTool(tool);
}
function isToolOutputAnnotation(tag, subtype) {
  if (tag === "measure-distance" || tag === "measure-area" || tag === "measure-perimeter") return true;
  if (tag === "redaction" || subtype === 28) return true;
  return false;
}
var USER_ANNOTATION_SUBTYPES = /* @__PURE__ */ new Set([
  1,
  // TEXT (sticky note)
  3,
  // FREETEXT
  4,
  // LINE
  5,
  // SQUARE
  6,
  // CIRCLE
  7,
  // POLYGON
  8,
  // POLYLINE
  9,
  // HIGHLIGHT
  10,
  // UNDERLINE
  11,
  // SQUIGGLY
  12,
  // STRIKEOUT
  13,
  // STAMP
  14,
  // CARET
  15,
  // INK
  28
  // REDACT
]);
function isUserAnnotation(subtype) {
  return USER_ANNOTATION_SUBTYPES.has(subtype);
}
function toPdf(pp, vp) {
  return vp.cssPointToPdf(pp.x, pp.y);
}
function createDrawModeHandler(ctx) {
  let drawState = null;
  function applyToolBehavior(tool, annotId) {
    const behavior = TOOL_BEHAVIORS[tool];
    if (behavior.selectAfterCreate) {
      ctx.annotation.selectAnnotation(annotId);
    }
    if (behavior.deactivateAfterCreate) {
      ctx.emit("annotation:tool-deactivated");
    }
  }
  function getPageHeight(docId, pageIndex) {
    const handle = ctx.document.getHandle(docId);
    return handle?.pageSizes[pageIndex]?.height ?? 792;
  }
  function pageVp(docId, pageIndex, scale = 1) {
    const ps = ctx.document.getHandle(docId)?.pageSizes[pageIndex];
    const rot = ctx.getPageRotation(docId, pageIndex);
    return PageViewport.fromRotatedSize(ps?.width ?? 612, ps?.height ?? 792, rot, scale);
  }
  let previewSvg = null;
  let previewPath = null;
  let previewPoints = [];
  function ensurePreviewSvg(pageIndex) {
    if (previewSvg) return previewSvg;
    if (!drawState) return null;
    const container = drawState.viewport.container ?? ctx.getContainer();
    if (!container) return null;
    const scrollArea = container.querySelector(".lector-canvas__scroll-area");
    if (!scrollArea) return null;
    const pos = drawState.viewport.pagePositions.peek().find((p) => p.pageIndex === pageIndex);
    if (!pos) return null;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", String(pos.width));
    svg.setAttribute("height", String(pos.height));
    svg.style.cssText = `position:absolute;left:${pos.x}px;top:${pos.y}px;pointer-events:none;z-index:9999;overflow:visible;`;
    svg.classList.add("lector-draw-preview");
    scrollArea.appendChild(svg);
    previewSvg = svg;
    return svg;
  }
  function clearPreview() {
    if (previewSvg) {
      previewSvg.remove();
      previewSvg = null;
    }
    previewPath = null;
    previewPoints = [];
  }
  function updateInkPreview(pdfX, pdfY) {
    if (!drawState) return;
    const scale = drawState.viewport.scale.peek();
    const vp = pageVp(drawState.docId, drawState.pageIndex, scale);
    const d0 = vp.pointToCss(pdfX, pdfY);
    const domX = d0.x;
    const domY = d0.y;
    const svg = ensurePreviewSvg(drawState.pageIndex);
    if (!svg) return;
    const style = ctx.style();
    const tool = ctx.activeTool();
    const color = `rgb(${style.color.r}, ${style.color.g}, ${style.color.b})`;
    const opacity = tool === "ink-highlighter" ? String(style.opacity) : "1";
    const baseWidth = style.borderWidth * scale;
    const stroke = drawState.currentStroke;
    const lastPt = stroke[stroke.length - 1];
    const isPenMode = lastPt?.pressure !== void 0;
    if (isPenMode) {
      if (!previewPath || previewPath.tagName.toLowerCase() !== "g") {
        if (previewPath) previewPath.remove();
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("opacity", opacity);
        svg.appendChild(g);
        previewPath = g;
        previewPoints = [];
      }
      previewPoints.push(`${domX},${domY}`);
      if (stroke.length >= 2) {
        const a = stroke[stroke.length - 2];
        const b = stroke[stroke.length - 1];
        const pa = a.pressure ?? 0.5;
        const pb = b.pressure ?? 0.5;
        const widthScale = 0.4 + (pa + pb) / 2 * 1.2;
        const da = vp.pointToCss(a.x, a.y);
        const db = vp.pointToCss(b.x, b.y);
        const seg = document.createElementNS("http://www.w3.org/2000/svg", "line");
        seg.setAttribute("x1", String(da.x));
        seg.setAttribute("y1", String(da.y));
        seg.setAttribute("x2", String(db.x));
        seg.setAttribute("y2", String(db.y));
        seg.setAttribute("stroke", color);
        seg.setAttribute("stroke-width", String(baseWidth * widthScale));
        seg.setAttribute("stroke-linecap", "round");
        previewPath.appendChild(seg);
      }
    } else {
      previewPoints.push(`${domX},${domY}`);
      if (!previewPath || previewPath.tagName.toLowerCase() !== "polyline") {
        if (previewPath) previewPath.remove();
        const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        polyline.setAttribute("fill", "none");
        polyline.setAttribute("stroke", color);
        polyline.setAttribute("stroke-opacity", opacity);
        polyline.setAttribute("stroke-width", String(baseWidth));
        polyline.setAttribute("stroke-linecap", "round");
        polyline.setAttribute("stroke-linejoin", "round");
        svg.appendChild(polyline);
        previewPath = polyline;
      }
      previewPath.setAttribute("points", previewPoints.join(" "));
    }
  }
  function updateShapePreview(startPdfX, startPdfY, endPdfX, endPdfY) {
    if (!drawState) return;
    const scale = drawState.viewport.scale.peek();
    const vp = pageVp(drawState.docId, drawState.pageIndex, scale);
    const p1 = vp.pointToCss(startPdfX, startPdfY);
    const p2 = vp.pointToCss(endPdfX, endPdfY);
    const svg = ensurePreviewSvg(drawState.pageIndex);
    if (!svg) return;
    const tool = ctx.activeTool();
    const style = ctx.style();
    const color = `rgb(${style.color.r}, ${style.color.g}, ${style.color.b})`;
    if (previewPath) previewPath.remove();
    if (tool === "line" || tool === "arrow" || tool === "measure-distance") {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", String(p1.x));
      line.setAttribute("y1", String(p1.y));
      line.setAttribute("x2", String(p2.x));
      line.setAttribute("y2", String(p2.y));
      line.setAttribute("stroke", color);
      line.setAttribute("stroke-width", String(style.borderWidth));
      line.setAttribute("stroke-dasharray", "6 3");
      if (tool === "measure-distance") {
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.appendChild(line);
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        const distPx = Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2);
        const distPt = distPx / scale;
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", String(midX));
        label.setAttribute("y", String(midY - 6));
        label.setAttribute("text-anchor", "middle");
        label.setAttribute("fill", color);
        label.setAttribute("font-size", "12");
        label.setAttribute("font-family", "system-ui, sans-serif");
        const m = ctx.getMeasurement();
        if (m) {
          label.textContent = m.format(distPt);
        } else if (ctx.formatting) {
          label.textContent = ctx.formatting.formatLengthFromPoints(distPt, 2);
        } else {
          label.textContent = `${(distPt / 72).toFixed(2)} in`;
        }
        g.appendChild(label);
        for (const pt of [p1, p2]) {
          const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          c.setAttribute("cx", String(pt.x));
          c.setAttribute("cy", String(pt.y));
          c.setAttribute("r", "4");
          c.setAttribute("fill", color);
          g.appendChild(c);
        }
        svg.appendChild(g);
        previewPath = g;
      } else if (tool === "arrow") {
        const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x);
        const headLen = 12;
        const a1x = p2.x - headLen * Math.cos(angle - Math.PI / 6);
        const a1y = p2.y - headLen * Math.sin(angle - Math.PI / 6);
        const a2x = p2.x - headLen * Math.cos(angle + Math.PI / 6);
        const a2y = p2.y - headLen * Math.sin(angle + Math.PI / 6);
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.appendChild(line);
        const arrow = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        arrow.setAttribute("points", `${a1x},${a1y} ${p2.x},${p2.y} ${a2x},${a2y}`);
        arrow.setAttribute("fill", "none");
        arrow.setAttribute("stroke", color);
        arrow.setAttribute("stroke-width", String(style.borderWidth));
        g.appendChild(arrow);
        svg.appendChild(g);
        previewPath = g;
      } else {
        svg.appendChild(line);
        previewPath = line;
      }
    } else if (tool === "circle") {
      const cx = (p1.x + p2.x) / 2;
      const cy = (p1.y + p2.y) / 2;
      const rx = Math.abs(p2.x - p1.x) / 2;
      const ry = Math.abs(p2.y - p1.y) / 2;
      const ellipse = document.createElementNS("http://www.w3.org/2000/svg", "ellipse");
      ellipse.setAttribute("cx", String(cx));
      ellipse.setAttribute("cy", String(cy));
      ellipse.setAttribute("rx", String(rx));
      ellipse.setAttribute("ry", String(ry));
      ellipse.setAttribute("fill", "none");
      ellipse.setAttribute("stroke", color);
      ellipse.setAttribute("stroke-width", String(style.borderWidth));
      ellipse.setAttribute("stroke-dasharray", "6 3");
      svg.appendChild(ellipse);
      previewPath = ellipse;
    } else {
      const x = Math.min(p1.x, p2.x);
      const y = Math.min(p1.y, p2.y);
      const w = Math.abs(p2.x - p1.x);
      const h = Math.abs(p2.y - p1.y);
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", String(x));
      rect.setAttribute("y", String(y));
      rect.setAttribute("width", String(w));
      rect.setAttribute("height", String(h));
      if (tool === "redaction") {
        rect.setAttribute("fill", "rgba(255, 0, 0, 0.3)");
        rect.setAttribute("stroke", "red");
      } else {
        rect.setAttribute("fill", "none");
        rect.setAttribute("stroke", color);
      }
      rect.setAttribute("stroke-width", String(style.borderWidth));
      rect.setAttribute("stroke-dasharray", "6 3");
      svg.appendChild(rect);
      previewPath = rect;
    }
  }
  function pressureFromEvent(domEvent) {
    if (!domEvent || !("pointerType" in domEvent)) return void 0;
    const pe = domEvent;
    if (pe.pointerType !== "pen") return void 0;
    const p = pe.pressure;
    if (typeof p !== "number" || !Number.isFinite(p) || p <= 0) return void 0;
    return p;
  }
  function onInkDown(pp, docId, vp, domEvent) {
    const h = getPageHeight(docId, pp.pageIndex);
    const pdf = toPdf(pp, pageVp(docId, pp.pageIndex));
    const pressure = pressureFromEvent(domEvent);
    drawState = {
      docId,
      pageIndex: pp.pageIndex,
      pageHeightPts: h,
      startPdfX: pdf.x,
      startPdfY: pdf.y,
      viewport: vp,
      currentStroke: [pressure !== void 0 ? { x: pdf.x, y: pdf.y, pressure } : { x: pdf.x, y: pdf.y }],
      markupStartCharIdx: -1,
      markupSelFirst: -1,
      markupSelLast: -1
    };
    updateInkPreview(pdf.x, pdf.y);
  }
  function onInkMove(pp, domEvent) {
    if (!drawState || pp.pageIndex !== drawState.pageIndex) return;
    const pdf = toPdf(pp, pageVp(drawState.docId, drawState.pageIndex));
    const pressure = pressureFromEvent(domEvent);
    drawState.currentStroke.push(pressure !== void 0 ? { x: pdf.x, y: pdf.y, pressure } : { x: pdf.x, y: pdf.y });
    updateInkPreview(pdf.x, pdf.y);
  }
  function onInkUp() {
    if (!drawState || drawState.currentStroke.length < 2) {
      clearPreview();
      drawState = null;
      return;
    }
    const tool = ctx.activeTool();
    const style = ctx.style();
    const stroke = drawState.currentStroke;
    const docId = drawState.docId;
    const pageIndex = drawState.pageIndex;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of stroke) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    const pad = style.borderWidth;
    const data = {
      subtype: FpdfAnnotSubtype.INK,
      pageIndex,
      rect: { left: minX - pad, bottom: minY - pad, right: maxX + pad, top: maxY + pad },
      color: tool === "ink-highlighter" ? { ...style.color, a: Math.round(style.opacity * 255) } : style.color,
      border: { horizontalRadius: 0, verticalRadius: 0, width: style.borderWidth },
      ink: { strokes: [stroke] }
    };
    drawState = null;
    void ctx.annotation.create(docId, pageIndex, data).then((tracked) => {
      clearPreview();
      if (ctx.history) {
        let annotId = tracked.id;
        ctx.history.push(docId, {
          id: uuid(),
          label: tool === "ink-highlighter" ? "Ink highlight" : "Ink drawing",
          topic: "annotation",
          timestamp: Date.now(),
          async execute() {
            const t = await ctx.annotation.create(docId, pageIndex, data);
            annotId = t.id;
          },
          undo() {
            void ctx.annotation.delete(docId, annotId);
          }
        });
      }
      if (tool) applyToolBehavior(tool, tracked.id);
    });
  }
  function hitTestAnnotation(annot, pdfX, pdfY, tolerance) {
    const r = annot.rect;
    const minX = Math.min(r.left, r.right) - tolerance;
    const maxX = Math.max(r.left, r.right) + tolerance;
    const minY = Math.min(r.top, r.bottom) - tolerance;
    const maxY = Math.max(r.top, r.bottom) + tolerance;
    return pdfX >= minX && pdfX <= maxX && pdfY >= minY && pdfY <= maxY;
  }
  function onEraserMove(pp, docId) {
    const pdf = toPdf(pp, pageVp(docId, pp.pageIndex));
    const annotations = ctx.annotation.getForPage(docId, pp.pageIndex);
    for (const tracked of annotations) {
      if (hitTestAnnotation(tracked.data, pdf.x, pdf.y, 5)) {
        let annotId = tracked.id;
        const erasedData = { ...tracked.data };
        const erasePageIndex = pp.pageIndex;
        void ctx.annotation.delete(docId, annotId);
        if (ctx.history) {
          ctx.history.push(docId, {
            id: uuid(),
            label: "Erase annotation",
            topic: "annotation",
            timestamp: Date.now(),
            execute() {
              void ctx.annotation.delete(docId, annotId);
            },
            async undo() {
              const t = await ctx.annotation.create(docId, erasePageIndex, erasedData);
              annotId = t.id;
            }
          });
        }
        break;
      }
    }
  }
  function onShapeDown(pp, docId, vp) {
    const h = getPageHeight(docId, pp.pageIndex);
    const pdf = toPdf(pp, pageVp(docId, pp.pageIndex));
    drawState = {
      docId,
      pageIndex: pp.pageIndex,
      pageHeightPts: h,
      startPdfX: pdf.x,
      startPdfY: pdf.y,
      viewport: vp,
      currentStroke: [],
      markupStartCharIdx: -1,
      markupSelFirst: -1,
      markupSelLast: -1
    };
  }
  let shiftHeld = false;
  function onShapeMove(pp, event) {
    if (!drawState || pp.pageIndex !== drawState.pageIndex) return;
    shiftHeld = "shiftKey" in event.domEvent && event.domEvent.shiftKey;
    const pdf = toPdf(pp, pageVp(drawState.docId, drawState.pageIndex));
    const { endX, endY } = constrainShape(drawState.startPdfX, drawState.startPdfY, pdf.x, pdf.y);
    updateShapePreview(drawState.startPdfX, drawState.startPdfY, endX, endY);
  }
  function constrainShape(startX, startY, endX, endY) {
    const tool = ctx.activeTool();
    if (shiftHeld && tool === "circle") {
      const dx = endX - startX;
      const dy = endY - startY;
      const size = Math.min(Math.abs(dx), Math.abs(dy));
      return {
        endX: startX + size * Math.sign(dx),
        endY: startY + size * Math.sign(dy)
      };
    }
    if (shiftHeld && tool === "rectangle") {
      const dx = endX - startX;
      const dy = endY - startY;
      const size = Math.min(Math.abs(dx), Math.abs(dy));
      return {
        endX: startX + size * Math.sign(dx),
        endY: startY + size * Math.sign(dy)
      };
    }
    if (shiftHeld && tool === "measure-distance") {
      const dx = endX - startX;
      const dy = endY - startY;
      if (Math.abs(dx) >= Math.abs(dy)) {
        return { endX, endY: startY };
      }
      return { endX: startX, endY };
    }
    if (shiftHeld && (tool === "line" || tool === "arrow")) {
      const dx = endX - startX;
      const dy = endY - startY;
      const angle = Math.atan2(dy, dx);
      const snapped = Math.round(angle / (Math.PI / 4)) * (Math.PI / 4);
      const dist = Math.sqrt(dx * dx + dy * dy);
      return {
        endX: startX + dist * Math.cos(snapped),
        endY: startY + dist * Math.sin(snapped)
      };
    }
    return { endX, endY };
  }
  function onShapeUp(pp, event) {
    if (!drawState) {
      clearPreview();
      return;
    }
    shiftHeld = "shiftKey" in event.domEvent && event.domEvent.shiftKey;
    const pdf = toPdf(pp, pageVp(drawState.docId, drawState.pageIndex));
    const { endX, endY } = constrainShape(drawState.startPdfX, drawState.startPdfY, pdf.x, pdf.y);
    const tool = ctx.activeTool();
    const style = ctx.style();
    const left = Math.min(drawState.startPdfX, endX);
    const right = Math.max(drawState.startPdfX, endX);
    const bottom = Math.min(drawState.startPdfY, endY);
    const top = Math.max(drawState.startPdfY, endY);
    if (right - left < 2 && top - bottom < 2) {
      clearPreview();
      drawState = null;
      return;
    }
    const pad = style.borderWidth;
    const isLine = tool === "line" || tool === "arrow" || tool === "measure-distance";
    const isRedaction = tool === "redaction";
    const subtype = isLine ? FpdfAnnotSubtype.INK : isRedaction ? FpdfAnnotSubtype.SQUARE : TOOL_TO_SUBTYPE[tool];
    const data = {
      subtype,
      pageIndex: drawState.pageIndex,
      rect: isLine ? { left: left - pad, bottom: bottom - pad, right: right + pad, top: top + pad } : { left, bottom, right, top },
      color: isRedaction ? { r: 255, g: 0, b: 0, a: 77 } : style.color,
      border: { horizontalRadius: 0, verticalRadius: 0, width: style.borderWidth },
      // Line/arrow/measure: store as ink stroke + line data for overlay rendering
      ...isLine ? {
        ink: { strokes: [[
          { x: drawState.startPdfX, y: drawState.startPdfY },
          { x: endX, y: endY }
        ]] },
        line: {
          start: { x: drawState.startPdfX, y: drawState.startPdfY },
          end: { x: endX, y: endY }
        },
        tag: tool === "arrow" ? "arrow" : tool === "measure-distance" ? "measure-distance" : "line"
      } : {},
      // Redaction: overlay text and fill
      ...isRedaction ? {
        tag: "redaction",
        interiorColor: { r: 0, g: 0, b: 0, a: 255 },
        redaction: { reason: "", overlayText: "", applied: false }
      } : {},
      // Interior color for filled shapes
      ...(tool === "rectangle" || tool === "circle") && style.interiorColor ? {
        interiorColor: style.interiorColor
      } : {},
      // FreeText annotation data
      ...tool === "freetext" ? {
        freeText: {
          text: "",
          fontSize: style.fontSize,
          fontColor: { r: style.color.r, g: style.color.g, b: style.color.b }
        }
      } : {},
      // Measurement annotation metadata. Embed a snapshot of the active
      // measurement scale + unit + precision so the annotation re-renders
      // correctly even if the user later changes the global scale.
      ...tool === "measure-distance" ? (() => {
        const m = ctx.getMeasurement();
        const valuePt = Math.sqrt(
          (endX - drawState.startPdfX) ** 2 + (endY - drawState.startPdfY) ** 2
        );
        const activeScale = m?.getScale() ?? null;
        const activeUnit = m?.activeUnit.peek() ?? "pt";
        const precision = m?.precision.peek() ?? 2;
        return {
          measurement: {
            type: "distance",
            value: valuePt,
            unit: activeUnit,
            scale: activeScale ?? {
              source: 1,
              sourceUnit: "in",
              target: 1,
              targetUnit: "in"
            },
            precision
          }
        };
      })() : {}
    };
    const docId = drawState.docId;
    const pageIndex = drawState.pageIndex;
    void ctx.annotation.create(docId, pageIndex, data).then((tracked) => {
      clearPreview();
      if (ctx.history) {
        let annotId = tracked.id;
        ctx.history.push(docId, {
          id: uuid(),
          label: `Add ${tool}`,
          topic: "annotation",
          timestamp: Date.now(),
          async execute() {
            const t = await ctx.annotation.create(docId, pageIndex, data);
            annotId = t.id;
          },
          undo() {
            void ctx.annotation.delete(docId, annotId);
          }
        });
      }
      applyToolBehavior(tool, tracked.id);
      if (tool === "freetext") {
        ctx.emit("annotation:edit-requested", tracked.id);
      }
    });
    drawState = null;
  }
  let polyVertices = [];
  let polyPageIndex = -1;
  let polyDocId = null;
  let polyViewport = null;
  const CLOSE_THRESHOLD = 15;
  let polyTooltip = null;
  function ensurePolyTooltip() {
    if (polyTooltip) return polyTooltip;
    const container = ctx.getContainer();
    if (!container) return null;
    const tip = document.createElement("div");
    tip.className = "lector-poly-tooltip";
    tip.textContent = "Click to place first point";
    container.appendChild(tip);
    polyTooltip = tip;
    return tip;
  }
  function updatePolyTooltipText() {
    if (!polyTooltip) return;
    const tool = ctx.activeTool();
    const n = polyVertices.length;
    if (n === 0) {
      polyTooltip.textContent = "Click to place first point";
    } else if (n === 1) {
      polyTooltip.textContent = "Click to add more points";
    } else if (n === 2 && tool === "polyline") {
      polyTooltip.textContent = "Click to add points \xB7 Double-click to finish";
    } else {
      const closeHint = tool === "polygon" ? "Click first point to close \xB7 " : "";
      polyTooltip.textContent = `${closeHint}Double-click to finish`;
    }
  }
  function movePolyTooltip(clientX, clientY) {
    const tip = ensurePolyTooltip();
    if (!tip) return;
    const container = ctx.getContainer();
    if (!container) return;
    const cRect = container.getBoundingClientRect();
    tip.style.left = `${clientX - cRect.left + 16}px`;
    tip.style.top = `${clientY - cRect.top + 16}px`;
  }
  function removePolyTooltip() {
    if (polyTooltip) {
      polyTooltip.remove();
      polyTooltip = null;
    }
  }
  function updatePolyPreview(cursorDomX, cursorDomY) {
    const svg = ensurePreviewSvg(polyPageIndex);
    if (!svg) return;
    for (const el of svg.querySelectorAll(".lector-poly-preview")) el.remove();
    const tool = ctx.activeTool();
    const style = ctx.style();
    const scale = (polyViewport ?? ctx.viewport.activeViewport.peek())?.scale.peek() ?? 1;
    const color = `rgb(${style.color.r}, ${style.color.g}, ${style.color.b})`;
    const allPts = polyVertices.map((v) => `${v.domX},${v.domY}`);
    if (cursorDomX !== void 0 && cursorDomY !== void 0 && polyVertices.length > 0) {
      allPts.push(`${cursorDomX},${cursorDomY}`);
    }
    const points = allPts.join(" ");
    const shape = document.createElementNS(
      "http://www.w3.org/2000/svg",
      tool === "polygon" ? "polygon" : "polyline"
    );
    shape.setAttribute("points", points);
    shape.setAttribute("fill", "none");
    shape.setAttribute("stroke", color);
    shape.setAttribute("stroke-width", String(style.borderWidth * scale));
    shape.setAttribute("stroke-dasharray", "6 3");
    shape.setAttribute("stroke-linejoin", "round");
    shape.classList.add("lector-poly-preview");
    svg.appendChild(shape);
    let nearFirst = false;
    if (tool === "polygon" && polyVertices.length >= 3 && cursorDomX !== void 0 && cursorDomY !== void 0) {
      const first = polyVertices[0];
      const dist = Math.sqrt((cursorDomX - first.domX) ** 2 + (cursorDomY - first.domY) ** 2);
      nearFirst = dist < CLOSE_THRESHOLD;
    }
    for (let i = 0; i < polyVertices.length; i++) {
      const v = polyVertices[i];
      const isOrigin = i === 0 && polyVertices.length > 2;
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", String(v.domX));
      dot.setAttribute("cy", String(v.domY));
      if (isOrigin) {
        dot.setAttribute("r", nearFirst ? "8" : "6");
        dot.setAttribute("fill", nearFirst ? color : "white");
        dot.setAttribute("stroke", nearFirst ? "white" : color);
        dot.setAttribute("stroke-width", "2");
        if (nearFirst) dot.setAttribute("opacity", "0.8");
      } else {
        dot.setAttribute("r", "4");
        dot.setAttribute("fill", color);
        dot.setAttribute("stroke", color);
        dot.setAttribute("stroke-width", "0");
      }
      dot.classList.add("lector-poly-preview");
      svg.appendChild(dot);
    }
  }
  function onPolyMove(pp, clientX, clientY) {
    if (polyVertices.length === 0) {
      movePolyTooltip(clientX, clientY);
      return;
    }
    if (pp.pageIndex !== polyPageIndex) return;
    if (!polyDocId) return;
    const pdf = toPdf(pp, pageVp(polyDocId, pp.pageIndex));
    const scale = (polyViewport ?? ctx.viewport.activeViewport.peek())?.scale.peek() ?? 1;
    const dom = pageVp(polyDocId, pp.pageIndex, scale).pointToCss(pdf.x, pdf.y);
    const domX = dom.x;
    const domY = dom.y;
    updatePolyPreview(domX, domY);
    movePolyTooltip(clientX, clientY);
    if (ctx.activeTool() === "polygon" && polyVertices.length >= 3) {
      const first = polyVertices[0];
      const dist = Math.sqrt((domX - first.domX) ** 2 + (domY - first.domY) ** 2);
      if (dist < CLOSE_THRESHOLD && polyTooltip) {
        polyTooltip.textContent = "Click to close shape";
        return;
      }
    }
    updatePolyTooltipText();
  }
  function onPolyClick(pp, docId, vp) {
    const pdf = toPdf(pp, pageVp(docId, pp.pageIndex));
    const scale = vp.scale.peek();
    const dom = pageVp(docId, pp.pageIndex, scale).pointToCss(pdf.x, pdf.y);
    const domX = dom.x;
    const domY = dom.y;
    if (polyVertices.length === 0) {
      polyPageIndex = pp.pageIndex;
      polyDocId = docId;
      polyViewport = vp;
    } else if (pp.pageIndex !== polyPageIndex) {
      return;
    }
    if (ctx.activeTool() === "polygon" && polyVertices.length >= 3) {
      const first = polyVertices[0];
      const dist = Math.sqrt((domX - first.domX) ** 2 + (domY - first.domY) ** 2);
      if (dist < CLOSE_THRESHOLD) {
        finishPolygon();
        return;
      }
    }
    polyVertices.push({ pdfX: pdf.x, pdfY: pdf.y, domX, domY });
    updatePolyPreview();
    updatePolyTooltipText();
  }
  function finishPolygon() {
    removePolyTooltip();
    if (polyVertices.length < 2 || !polyDocId) {
      clearPreview();
      polyVertices = [];
      polyDocId = null;
      polyViewport = null;
      return;
    }
    const tool = ctx.activeTool();
    const style = ctx.style();
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const v of polyVertices) {
      if (v.pdfX < minX) minX = v.pdfX;
      if (v.pdfY < minY) minY = v.pdfY;
      if (v.pdfX > maxX) maxX = v.pdfX;
      if (v.pdfY > maxY) maxY = v.pdfY;
    }
    const pad = style.borderWidth;
    const stroke = polyVertices.map((v) => ({ x: v.pdfX, y: v.pdfY }));
    if (tool === "polygon" && stroke.length > 0) {
      stroke.push({ x: stroke[0].x, y: stroke[0].y });
    }
    const isMeasure = tool === "measure-area" || tool === "measure-perimeter";
    const tag = isMeasure ? tool : tool === "polygon" ? "polygon" : "polyline";
    let measurementData = void 0;
    if (tool === "measure-area") {
      let area = 0;
      const pts = polyVertices;
      for (let i = 0; i < pts.length; i++) {
        const j = (i + 1) % pts.length;
        area += pts[i].pdfX * pts[j].pdfY;
        area -= pts[j].pdfX * pts[i].pdfY;
      }
      area = Math.abs(area) / 2;
      const m = ctx.getMeasurement();
      const activeScale = m?.getScale() ?? null;
      const activeUnit = m?.activeUnit.peek() ?? "pt";
      const precision = m?.precision.peek() ?? 2;
      measurementData = {
        type: "area",
        value: area,
        unit: activeUnit,
        scale: activeScale ?? { source: 1, sourceUnit: "in", target: 1, targetUnit: "in" },
        precision
      };
    } else if (tool === "measure-perimeter") {
      let perim = 0;
      const pts = polyVertices;
      for (let i = 0; i < pts.length; i++) {
        const j = (i + 1) % pts.length;
        perim += Math.sqrt(
          (pts[j].pdfX - pts[i].pdfX) ** 2 + (pts[j].pdfY - pts[i].pdfY) ** 2
        );
      }
      const m = ctx.getMeasurement();
      const activeScale = m?.getScale() ?? null;
      const activeUnit = m?.activeUnit.peek() ?? "pt";
      const precision = m?.precision.peek() ?? 2;
      measurementData = {
        type: "perimeter",
        value: perim,
        unit: activeUnit,
        scale: activeScale ?? { source: 1, sourceUnit: "in", target: 1, targetUnit: "in" },
        precision
      };
    }
    const data = {
      subtype: FpdfAnnotSubtype.INK,
      pageIndex: polyPageIndex,
      rect: { left: minX - pad, bottom: minY - pad, right: maxX + pad, top: maxY + pad },
      color: style.color,
      border: { horizontalRadius: 0, verticalRadius: 0, width: style.borderWidth },
      ink: { strokes: [stroke] },
      tag,
      ...measurementData ? { measurement: measurementData } : {}
    };
    const docId = polyDocId;
    const pageIndex = polyPageIndex;
    clearPreview();
    polyVertices = [];
    polyDocId = null;
    polyViewport = null;
    void ctx.annotation.create(docId, pageIndex, data).then((tracked) => {
      if (ctx.history) {
        let annotId = tracked.id;
        ctx.history.push(docId, {
          id: uuid(),
          label: `Add ${tool}`,
          topic: "annotation",
          timestamp: Date.now(),
          async execute() {
            const t = await ctx.annotation.create(docId, pageIndex, data);
            annotId = t.id;
          },
          undo() {
            void ctx.annotation.delete(docId, annotId);
          }
        });
      }
      applyToolBehavior(tool, tracked.id);
    });
  }
  let markupChars = null;
  let markupText = null;
  function onMarkupDown(pp, docId, vp) {
    if (!ctx.textLayer) return;
    const h = getPageHeight(docId, pp.pageIndex);
    const pdf = toPdf(pp, pageVp(docId, pp.pageIndex));
    drawState = {
      docId,
      pageIndex: pp.pageIndex,
      pageHeightPts: h,
      startPdfX: pdf.x,
      startPdfY: pdf.y,
      viewport: vp,
      currentStroke: [],
      markupStartCharIdx: -1,
      markupSelFirst: -1,
      markupSelLast: -1
    };
    markupChars = null;
    markupText = null;
    Promise.all([
      ctx.textLayer.getPageCharInfo(docId, pp.pageIndex),
      ctx.textLayer.getPageText(docId, pp.pageIndex)
    ]).then(([chars, text]) => {
      if (!drawState) return;
      markupChars = chars;
      markupText = text;
      const idx = localCharHitTest(chars, pdf.x, pdf.y, 20);
      if (idx >= 0) drawState.markupStartCharIdx = idx;
    }).catch(() => {
    });
  }
  function localCharHitTest(chars, x, y, tolerance) {
    let bestIdx = -1;
    let bestDist = tolerance;
    for (let i = 0; i < chars.length; i++) {
      const c = chars[i];
      if (x >= c.left && x <= c.right && y >= c.bottom && y <= c.top) return i;
      const cx = (c.left + c.right) / 2;
      const cy = (c.top + c.bottom) / 2;
      const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    return bestIdx;
  }
  function onMarkupMove(pp) {
    if (!drawState || !ctx.textLayer || drawState.markupStartCharIdx < 0) return;
    if (pp.pageIndex !== drawState.pageIndex) return;
    if (!markupChars || !markupText) return;
    const pdf = toPdf(pp, pageVp(drawState.docId, drawState.pageIndex));
    const endIdx = localCharHitTest(markupChars, pdf.x, pdf.y, 20);
    if (endIdx < 0) return;
    const first = Math.min(drawState.markupStartCharIdx, endIdx);
    const last = Math.max(drawState.markupStartCharIdx, endIdx);
    const count = last - first + 1;
    if (count <= 0) return;
    drawState.markupSelFirst = first;
    drawState.markupSelLast = last;
    const rects = [];
    let cl = markupChars[first];
    let left = cl.left, right = cl.right, top = cl.top, bottom = cl.bottom;
    for (let i = first + 1; i <= last && i < markupChars.length; i++) {
      const c = markupChars[i];
      const lineH = Math.min(top - bottom, c.top - c.bottom);
      const overlap = Math.min(top, c.top) - Math.max(bottom, c.bottom);
      if (overlap > lineH * 0.5 && c.left >= left - 1) {
        right = Math.max(right, c.right);
        top = Math.max(top, c.top);
        bottom = Math.min(bottom, c.bottom);
      } else {
        rects.push({ left, right, top, bottom });
        left = c.left;
        right = c.right;
        top = c.top;
        bottom = c.bottom;
      }
    }
    rects.push({ left, right, top, bottom });
    ctx.textLayer.setSelection({
      docId: drawState.docId,
      pageIndex: drawState.pageIndex,
      startCharIndex: first,
      endCharIndex: last,
      text: markupText.substring(first, first + count),
      rects
    });
  }
  function onMarkupUp(pp) {
    markupChars = null;
    markupText = null;
    if (!drawState || !ctx.textLayer) {
      drawState = null;
      return;
    }
    const endPdf = toPdf(pp, pageVp(drawState.docId, drawState.pageIndex));
    const tool = ctx.activeTool();
    const style = ctx.style();
    const docId = drawState.docId;
    const pageIndex = drawState.pageIndex;
    const startPdfX = drawState.startPdfX;
    const startPdfY = drawState.startPdfY;
    const storedStartCharIdx = drawState.markupStartCharIdx;
    const previewFirst = drawState.markupSelFirst;
    const previewLast = drawState.markupSelLast;
    drawState = null;
    if (ctx.textLayer) ctx.textLayer.setSelection(null);
    void (async () => {
      let first;
      let last;
      if (previewFirst >= 0 && previewLast >= 0) {
        first = previewFirst;
        last = previewLast;
      } else {
        let startIdx = storedStartCharIdx;
        if (startIdx < 0) {
          startIdx = await ctx.textLayer.getCharIndexAtPos(docId, pageIndex, startPdfX, startPdfY, 20);
        }
        let endIdx = await ctx.textLayer.getCharIndexAtPos(docId, pageIndex, endPdf.x, endPdf.y, 30);
        const resolvedEnd = endIdx >= 0 ? endIdx : startIdx;
        if (startIdx < 0) return;
        first = Math.min(startIdx, resolvedEnd);
        last = Math.max(startIdx, resolvedEnd);
      }
      const count = last - first + 1;
      if (count <= 0) return;
      let rects;
      try {
        rects = await ctx.textLayer.getTextRects(docId, pageIndex, first, count);
      } catch {
        return;
      }
      if (rects.length === 0) return;
      const quadPoints = rects.map((r) => ({
        x1: r.left,
        y1: Math.max(r.top, r.bottom),
        x2: r.right,
        y2: Math.max(r.top, r.bottom),
        x3: r.left,
        y3: Math.min(r.top, r.bottom),
        x4: r.right,
        y4: Math.min(r.top, r.bottom)
      }));
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const qp of quadPoints) {
        minX = Math.min(minX, qp.x1, qp.x3);
        maxX = Math.max(maxX, qp.x2, qp.x4);
        minY = Math.min(minY, qp.y3, qp.y4);
        maxY = Math.max(maxY, qp.y1, qp.y2);
      }
      const subtype = TOOL_TO_SUBTYPE[tool];
      const data = {
        subtype,
        pageIndex,
        rect: { left: minX, bottom: maxY, right: maxX, top: minY },
        color: style.color,
        // Full alpha; CSS mix-blend-mode handles visual blending
        markup: { quadPoints }
      };
      let tracked;
      try {
        tracked = await ctx.annotation.create(docId, pageIndex, data);
      } catch {
        return;
      }
      if (ctx.history) {
        let annotId = tracked.id;
        ctx.history.push(docId, {
          id: uuid(),
          label: `Add ${tool}`,
          topic: "annotation",
          timestamp: Date.now(),
          async execute() {
            const t = await ctx.annotation.create(docId, pageIndex, data);
            annotId = t.id;
          },
          undo() {
            void ctx.annotation.delete(docId, annotId);
          }
        });
      }
      applyToolBehavior(tool, tracked.id);
    })();
  }
  function onPlacementClick(pp, docId) {
    const tool = ctx.activeTool();
    const style = ctx.style();
    const pdf = toPdf(pp, pageVp(docId, pp.pageIndex));
    const subtype = TOOL_TO_SUBTYPE[tool];
    if (tool === "stamp") {
      const stampW = 200;
      const stampH = 60;
      const data2 = {
        subtype: FpdfAnnotSubtype.STAMP,
        pageIndex: pp.pageIndex,
        rect: {
          left: pdf.x - stampW / 2,
          bottom: pdf.y + stampH / 2,
          right: pdf.x + stampW / 2,
          top: pdf.y - stampH / 2
        },
        color: style.color,
        tag: ctx.getStampName?.() ?? "Approved",
        stamp: {
          name: ctx.getStampName?.() ?? "Approved"
        },
        contents: ctx.getStampName?.() ?? "Approved"
      };
      void ctx.annotation.create(docId, pp.pageIndex, data2).then((tracked) => {
        if (ctx.history) {
          let annotId = tracked.id;
          const pi = pp.pageIndex;
          ctx.history.push(docId, {
            id: uuid(),
            label: `Add stamp`,
            topic: "annotation",
            timestamp: Date.now(),
            async execute() {
              const t = await ctx.annotation.create(docId, pi, data2);
              annotId = t.id;
            },
            undo() {
              void ctx.annotation.delete(docId, annotId);
            }
          });
        }
        applyToolBehavior(tool, tracked.id);
      });
      return;
    }
    const size = tool === "sticky-note" ? 24 : tool === "insert-text" ? 12 : 24;
    const height = size;
    const data = {
      subtype,
      pageIndex: pp.pageIndex,
      rect: {
        left: pdf.x,
        bottom: pdf.y,
        right: pdf.x + size,
        top: pdf.y - height
      },
      color: style.color,
      contents: tool === "sticky-note" ? "" : void 0
    };
    void ctx.annotation.create(docId, pp.pageIndex, data).then((tracked) => {
      if (ctx.history) {
        let annotId = tracked.id;
        const pi = pp.pageIndex;
        ctx.history.push(docId, {
          id: uuid(),
          label: `Add ${tool}`,
          topic: "annotation",
          timestamp: Date.now(),
          async execute() {
            const t = await ctx.annotation.create(docId, pi, data);
            annotId = t.id;
          },
          undo() {
            void ctx.annotation.delete(docId, annotId);
          }
        });
      }
      applyToolBehavior(tool, tracked.id);
    });
  }
  let calloutPhase = "rect";
  let calloutRect = null;
  function resetCallout() {
    calloutPhase = "rect";
    calloutRect = null;
    clearPreview();
  }
  function calloutAnchorPoint(rect, endpoint) {
    const cx = (rect.left + rect.right) / 2;
    const cy = (rect.top + rect.bottom) / 2;
    const dx = endpoint.x - cx;
    const dy = endpoint.y - cy;
    if (Math.abs(dx) >= Math.abs(dy)) {
      return dx >= 0 ? { x: rect.right, y: cy } : { x: rect.left, y: cy };
    }
    return dy >= 0 ? { x: cx, y: rect.top } : { x: cx, y: rect.bottom };
  }
  function updateCalloutLeaderPreview(endPdfX, endPdfY) {
    if (!calloutRect) return;
    const scale = calloutRect.viewport.scale.peek();
    const vp = pageVp(calloutRect.docId, calloutRect.pageIndex, scale);
    const svg = ensurePreviewSvg(calloutRect.pageIndex);
    if (!svg) return;
    for (const el of svg.querySelectorAll(".lector-callout-leader-preview")) el.remove();
    const style = ctx.style();
    const color = `rgb(${style.color.r}, ${style.color.g}, ${style.color.b})`;
    const strokeWidth = String(style.borderWidth * scale);
    const anchor = calloutAnchorPoint(
      { left: calloutRect.left, right: calloutRect.right, top: calloutRect.top, bottom: calloutRect.bottom },
      { x: endPdfX, y: endPdfY }
    );
    const a = vp.pointToCss(anchor.x, anchor.y);
    const e = vp.pointToCss(endPdfX, endPdfY);
    const ax = a.x;
    const ay = a.y;
    const ex = e.x;
    const ey = e.y;
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.classList.add("lector-callout-leader-preview");
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(ax));
    line.setAttribute("y1", String(ay));
    line.setAttribute("x2", String(ex));
    line.setAttribute("y2", String(ey));
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", strokeWidth);
    line.setAttribute("stroke-dasharray", "6 3");
    g.appendChild(line);
    const angle = Math.atan2(ey - ay, ex - ax);
    const headLen = 12;
    const a1x = ex - headLen * Math.cos(angle - Math.PI / 6);
    const a1y = ey - headLen * Math.sin(angle - Math.PI / 6);
    const a2x = ex - headLen * Math.cos(angle + Math.PI / 6);
    const a2y = ey - headLen * Math.sin(angle + Math.PI / 6);
    const head = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    head.setAttribute("points", `${a1x},${a1y} ${ex},${ey} ${a2x},${a2y}`);
    head.setAttribute("fill", "none");
    head.setAttribute("stroke", color);
    head.setAttribute("stroke-width", strokeWidth);
    head.setAttribute("stroke-linecap", "round");
    head.setAttribute("stroke-linejoin", "round");
    g.appendChild(head);
    svg.appendChild(g);
  }
  function paintStaticCalloutRect() {
    if (!calloutRect) return;
    const scale = calloutRect.viewport.scale.peek();
    const svg = ensurePreviewSvg(calloutRect.pageIndex);
    if (!svg) return;
    for (const el of svg.querySelectorAll(".lector-callout-rect-static")) el.remove();
    const style = ctx.style();
    const color = `rgb(${style.color.r}, ${style.color.g}, ${style.color.b})`;
    const box = pageVp(calloutRect.docId, calloutRect.pageIndex, scale).rectToCss({
      left: calloutRect.left,
      top: calloutRect.top,
      right: calloutRect.right,
      bottom: calloutRect.bottom
    });
    const x = box.x;
    const y = box.y;
    const w = box.w;
    const h = box.h;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(x));
    rect.setAttribute("y", String(y));
    rect.setAttribute("width", String(w));
    rect.setAttribute("height", String(h));
    rect.setAttribute("fill", "none");
    rect.setAttribute("stroke", color);
    rect.setAttribute("stroke-width", String(style.borderWidth * scale));
    rect.classList.add("lector-callout-rect-static");
    svg.appendChild(rect);
  }
  function onCalloutDown(pp, docId, vp) {
    if (calloutPhase !== "rect") return;
    onShapeDown(pp, docId, vp);
  }
  function onCalloutMove(pp, event) {
    if (calloutPhase === "rect") {
      onShapeMove(pp, event);
      return;
    }
    if (!calloutRect) return;
    if (pp.pageIndex !== calloutRect.pageIndex) return;
    const pdf = toPdf(pp, pageVp(calloutRect.docId, calloutRect.pageIndex));
    updateCalloutLeaderPreview(pdf.x, pdf.y);
  }
  function onCalloutUp(pp) {
    if (calloutPhase !== "rect") return;
    if (!drawState) {
      resetCallout();
      return;
    }
    const pdf = toPdf(pp, pageVp(drawState.docId, drawState.pageIndex));
    const left = Math.min(drawState.startPdfX, pdf.x);
    const right = Math.max(drawState.startPdfX, pdf.x);
    const bottom = Math.min(drawState.startPdfY, pdf.y);
    const top = Math.max(drawState.startPdfY, pdf.y);
    if (right - left < 10 || top - bottom < 10) {
      drawState = null;
      clearPreview();
      return;
    }
    calloutRect = {
      left,
      right,
      top,
      bottom,
      pageIndex: drawState.pageIndex,
      docId: drawState.docId,
      pageHeightPts: drawState.pageHeightPts,
      viewport: drawState.viewport
    };
    drawState = null;
    if (previewPath) {
      previewPath.remove();
      previewPath = null;
    }
    paintStaticCalloutRect();
    calloutPhase = "leader";
  }
  function onCalloutClick(pp) {
    if (calloutPhase !== "leader" || !calloutRect) return;
    if (pp.pageIndex !== calloutRect.pageIndex) return;
    const pdf = toPdf(pp, pageVp(calloutRect.docId, calloutRect.pageIndex));
    commitCallout(pdf.x, pdf.y);
  }
  function commitCallout(endX, endY) {
    if (!calloutRect) return;
    const style = ctx.style();
    const docId = calloutRect.docId;
    const pageIndex = calloutRect.pageIndex;
    const rect = {
      left: calloutRect.left,
      right: calloutRect.right,
      top: calloutRect.top,
      bottom: calloutRect.bottom
    };
    const callout = {
      endpoint: { x: endX, y: endY },
      lineEnding: "OpenArrow"
    };
    const data = {
      subtype: FpdfAnnotSubtype.FREETEXT,
      pageIndex,
      rect,
      color: style.color,
      border: { horizontalRadius: 0, verticalRadius: 0, width: style.borderWidth },
      freeText: {
        text: "",
        fontSize: style.fontSize,
        fontColor: { r: style.color.r, g: style.color.g, b: style.color.b }
      },
      callout,
      tag: "callout"
    };
    resetCallout();
    void ctx.annotation.create(docId, pageIndex, data).then((tracked) => {
      if (ctx.history) {
        let annotId = tracked.id;
        ctx.history.push(docId, {
          id: uuid(),
          label: "Add callout",
          topic: "annotation",
          timestamp: Date.now(),
          async execute() {
            const t = await ctx.annotation.create(docId, pageIndex, data);
            annotId = t.id;
          },
          undo() {
            void ctx.annotation.delete(docId, annotId);
          }
        });
      }
      applyToolBehavior("callout", tracked.id);
    });
  }
  const IMAGE_DATA_URI_MAX_LENGTH = 6e6;
  const IMAGE_TARGET_MAX_PT = 360;
  function onImagePlace(pp, docId) {
    const staged = ctx.getStagedImage?.();
    if (!staged) {
      return;
    }
    if (staged.dataUri.length > IMAGE_DATA_URI_MAX_LENGTH) {
      ctx.emit("annotation:image-too-large");
      return;
    }
    const pdf = toPdf(pp, pageVp(docId, pp.pageIndex));
    const naturalW = Math.max(1, staged.naturalWidth);
    const naturalH = Math.max(1, staged.naturalHeight);
    const longest = Math.max(naturalW, naturalH);
    const k = longest > IMAGE_TARGET_MAX_PT ? IMAGE_TARGET_MAX_PT / longest : 1;
    const w = naturalW * k;
    const hRect = naturalH * k;
    const left = pdf.x - w / 2;
    const right = pdf.x + w / 2;
    const top = pdf.y + hRect / 2;
    const bottom = pdf.y - hRect / 2;
    const image = {
      imageRef: staged.dataUri,
      width: w,
      height: hRect,
      naturalWidth: naturalW,
      naturalHeight: naturalH
    };
    const data = {
      subtype: FpdfAnnotSubtype.STAMP,
      pageIndex: pp.pageIndex,
      rect: { left, right, top, bottom },
      color: { r: 0, g: 0, b: 0, a: 0 },
      border: { horizontalRadius: 0, verticalRadius: 0, width: 0 },
      tag: "image",
      image
    };
    const docIdCaptured = docId;
    const pageIndex = pp.pageIndex;
    void ctx.annotation.create(docIdCaptured, pageIndex, data).then((tracked) => {
      ctx.onImagePlaced?.();
      if (ctx.history) {
        let annotId = tracked.id;
        ctx.history.push(docIdCaptured, {
          id: uuid(),
          label: "Add image",
          topic: "annotation",
          timestamp: Date.now(),
          async execute() {
            const t = await ctx.annotation.create(docIdCaptured, pageIndex, data);
            annotId = t.id;
          },
          undo() {
            void ctx.annotation.delete(docIdCaptured, annotId);
          }
        });
      }
      applyToolBehavior("image", tracked.id);
    });
  }
  function getCursorForTool() {
    const tool = ctx.activeTool();
    if (!tool) return "default";
    if (isMarkupTool(tool)) return "text";
    if (isInkTool(tool)) return "crosshair";
    if (isEraserTool(tool)) return `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'%3E%3Ccircle cx='12' cy='12' r='10' fill='white' stroke='%23666' stroke-width='1.5'/%3E%3C/svg%3E") 12 12, crosshair`;
    if (isShapeTool(tool)) return "crosshair";
    if (isPolygonTool(tool)) return "crosshair";
    if (isCalloutTool(tool)) return "crosshair";
    if (isImageTool(tool)) return "cell";
    if (isPlacementTool(tool)) return "cell";
    return "crosshair";
  }
  ctx.interaction.registerHandler("draw", {
    get cursor() {
      return getCursorForTool();
    },
    onPointerDown(event) {
      const tool = ctx.activeTool();
      if (!tool) return;
      if (!event.pagePoint) {
        return;
      }
      const vp = event.viewport ?? ctx.viewport.activeViewport.peek();
      if (!vp) {
        return;
      }
      const docId = vp.docId.peek() ?? ctx.document.activeDocument.peek()?.id;
      if (!docId) return;
      if (isInkTool(tool)) {
        onInkDown(event.pagePoint, docId, vp, event.domEvent);
      } else if (isShapeTool(tool)) {
        onShapeDown(event.pagePoint, docId, vp);
      } else if (isMarkupTool(tool)) {
        onMarkupDown(event.pagePoint, docId, vp);
      } else if (isCalloutTool(tool)) {
        onCalloutDown(event.pagePoint, docId, vp);
      }
      if (isEraserTool(tool)) {
        onEraserMove(event.pagePoint, docId);
      }
    },
    onPointerMove(event) {
      const tool = ctx.activeTool();
      if (!tool || !event.pagePoint) return;
      if (isInkTool(tool)) {
        onInkMove(event.pagePoint, event.domEvent);
      } else if (isShapeTool(tool)) {
        onShapeMove(event.pagePoint, event);
      } else if (isMarkupTool(tool)) {
        onMarkupMove(event.pagePoint);
      } else if (isCalloutTool(tool)) {
        onCalloutMove(event.pagePoint, event);
      } else if (isEraserTool(tool)) {
        const vp = event.viewport ?? ctx.viewport.activeViewport.peek();
        const docId = vp?.docId.peek() ?? ctx.document.activeDocument.peek()?.id;
        if (docId) onEraserMove(event.pagePoint, docId);
      } else if (isPolygonTool(tool)) {
        onPolyMove(event.pagePoint, event.clientX, event.clientY);
      }
    },
    onPointerUp(event) {
      const tool = ctx.activeTool();
      if (!tool || !event.pagePoint) return;
      if (isInkTool(tool)) {
        onInkUp();
      } else if (isShapeTool(tool)) {
        onShapeUp(event.pagePoint, event);
      } else if (isMarkupTool(tool)) {
        onMarkupUp(event.pagePoint);
      } else if (isCalloutTool(tool)) {
        onCalloutUp(event.pagePoint);
      }
    },
    onClick(event) {
      const tool = ctx.activeTool();
      if (!tool || !event.pagePoint) return;
      const vp = event.viewport ?? ctx.viewport.activeViewport.peek();
      const docId = vp?.docId.peek() ?? ctx.document.activeDocument.peek()?.id;
      if (!docId) return;
      if (isPolygonTool(tool)) {
        if (vp) onPolyClick(event.pagePoint, docId, vp);
      } else if (isCalloutTool(tool)) {
        onCalloutClick(event.pagePoint);
      } else if (isImageTool(tool)) {
        onImagePlace(event.pagePoint, docId);
      } else if (isPlacementTool(tool)) {
        onPlacementClick(event.pagePoint, docId);
      }
    },
    onDoubleClick(_event) {
      const tool = ctx.activeTool();
      if (!tool) return;
      if (isPolygonTool(tool)) {
        finishPolygon();
      }
    },
    onKeyDown(event) {
      const tool = ctx.activeTool();
      if (event.key === "Escape") {
        if (isPolygonTool(tool) && polyVertices.length > 0) {
          clearPreview();
          removePolyTooltip();
          polyVertices = [];
          polyDocId = null;
          polyViewport = null;
          return;
        }
        if (isCalloutTool(tool) && (calloutPhase === "leader" || calloutRect)) {
          resetCallout();
          return;
        }
        removePolyTooltip();
        ctx.emit("annotation:tool-deactivated");
      }
      if (event.key === "Enter" && isPolygonTool(tool)) {
        finishPolygon();
      }
    },
    onDeactivate() {
      drawState = null;
      polyVertices = [];
      polyDocId = null;
      polyViewport = null;
      resetCallout();
      clearPreview();
      removePolyTooltip();
    }
  });
  return {
    activate() {
      ctx.interaction.setMode("draw");
    },
    deactivate() {
      drawState = null;
      ctx.interaction.setMode("pointer");
    }
  };
}

// src/ui/icons.ts
var S = 'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"';
var ICONS = {
  // ── Navigation ──
  "chevron-up": `<svg ${S}><path d="m18 15-6-6-6 6"/></svg>`,
  "chevron-down": `<svg ${S}><path d="m6 9 6 6 6-6"/></svg>`,
  "chevron-left": `<svg ${S}><path d="m15 18-6-6 6-6"/></svg>`,
  "chevron-right": `<svg ${S}><path d="m9 18 6-6-6-6"/></svg>`,
  // ── Zoom ──
  "zoom-in": `<svg ${S}><circle cx="11" cy="11" r="8"/><line x1="21" x2="16.65" y1="21" y2="16.65"/><line x1="11" x2="11" y1="8" y2="14"/><line x1="8" x2="14" y1="11" y2="11"/></svg>`,
  "zoom-out": `<svg ${S}><circle cx="11" cy="11" r="8"/><line x1="21" x2="16.65" y1="21" y2="16.65"/><line x1="8" x2="14" y1="11" y2="11"/></svg>`,
  "fit-width": `<svg ${S}><path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/><path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/></svg>`,
  "fit-page": `<svg ${S}><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/></svg>`,
  // ── Tools ──
  "cursor": `<svg ${S}><path d="M4.037 4.688a.495.495 0 0 1 .651-.651l16 6.5a.5.5 0 0 1-.063.947l-6.124 1.58a2 2 0 0 0-1.438 1.435l-1.579 6.126a.5.5 0 0 1-.947.063z"/></svg>`,
  "hand": `<svg ${S}><path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2"/><path d="M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></svg>`,
  "text-select": `<svg ${S}><path d="M12 4v16"/><path d="M4 7V5a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v2"/><path d="M9 20h6"/></svg>`,
  // ── Actions ──
  "search": `<svg ${S}><path d="m21 21-4.34-4.34"/><circle cx="11" cy="11" r="8"/></svg>`,
  "undo": `<svg ${S}><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11"/></svg>`,
  "redo": `<svg ${S}><path d="m15 14 5-5-5-5"/><path d="M20 9H9.5A5.5 5.5 0 0 0 4 14.5A5.5 5.5 0 0 0 9.5 20H13"/></svg>`,
  "copy": `<svg ${S}><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`,
  // ── Layout ──
  "sidebar": `<svg ${S}><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/></svg>`,
  "more-vertical": `<svg ${S}><circle cx="12" cy="12" r="1"/><circle cx="12" cy="5" r="1"/><circle cx="12" cy="19" r="1"/></svg>`,
  "layout-single": `<svg ${S}><rect width="18" height="18" x="3" y="3" rx="2"/></svg>`,
  "layout-continuous": `<svg ${S}><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M12 3v18"/><path d="M3 12h18"/></svg>`,
  "layout-double": `<svg ${S}><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M12 3v18"/></svg>`,
  // ── Sidebar panels ──
  "grid": `<svg ${S}><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 3v18"/><path d="M3 12h18"/></svg>`,
  "bookmark": `<svg ${S}><path d="M17 3a2 2 0 0 1 2 2v15a1 1 0 0 1-1.496.868l-4.512-2.578a2 2 0 0 0-1.984 0l-4.512 2.578A1 1 0 0 1 5 20V5a2 2 0 0 1 2-2z"/></svg>`,
  "annotation": `<svg ${S}><path d="M22 17a2 2 0 0 1-2 2H6.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 2 21.286V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z"/></svg>`,
  // ── File / document ──
  "file": `<svg ${S}><path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z"/><path d="M14 2v5a1 1 0 0 0 1 1h5"/></svg>`,
  "file-up": `<svg ${S}><path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z"/><path d="M14 2v5a1 1 0 0 0 1 1h5"/><path d="M12 12v6"/><path d="m15 15-3-3-3 3"/></svg>`,
  "menu": `<svg ${S}><path d="M4 5h16"/><path d="M4 12h16"/><path d="M4 19h16"/></svg>`,
  "printer": `<svg ${S}><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><path d="M6 9V3a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v6"/><rect x="6" y="14" width="12" height="8" rx="1"/></svg>`,
  "shield": `<svg ${S}><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>`,
  "camera": `<svg ${S}><path d="M13.997 4a2 2 0 0 1 1.76 1.05l.486.9A2 2 0 0 0 18.003 7H20a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1.997a2 2 0 0 0 1.759-1.048l.489-.904A2 2 0 0 1 10.004 4z"/><circle cx="12" cy="13" r="3"/></svg>`,
  "crop": `<svg ${S}><path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/></svg>`,
  "columns": `<svg ${S}><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M12 3v18"/></svg>`,
  "rows": `<svg ${S}><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 12h18"/></svg>`,
  "download": `<svg ${S}><path d="M12 15V3"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/></svg>`,
  "fullscreen": `<svg ${S}><path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/><path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/></svg>`,
  "eye": `<svg ${S}><path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/></svg>`,
  "plus": `<svg ${S}><path d="M5 12h14"/><path d="M12 5v14"/></svg>`,
  "paperclip": `<svg ${S}><path d="M13.234 20.252 21.02 12.47a3.767 3.767 0 0 0-5.327-5.328L7.907 14.93a2.511 2.511 0 0 0 3.551 3.551l5.058-5.06"/></svg>`,
  "pen-tool": `<svg ${S}><path d="M15.707 21.293a1 1 0 0 1-1.414 0l-1.586-1.586a1 1 0 0 1 0-1.414l5.586-5.586a1 1 0 0 1 1.414 0l1.586 1.586a1 1 0 0 1 0 1.414z"/><path d="m18 13-1.375-6.874a1 1 0 0 0-.746-.776L3.235 2.028a1 1 0 0 0-1.207 1.207L5.35 15.879a1 1 0 0 0 .776.746L13 18"/><path d="m2.3 2.3 7.286 7.286"/><circle cx="11" cy="11" r="2"/></svg>`,
  "chevron-right-sm": `<svg ${S}><path d="m9 18 6-6-6-6"/></svg>`,
  "layers": `<svg ${S}><path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.84z"/><path d="m2 12 8.58 3.91a2 2 0 0 0 1.66 0L21 12"/><path d="m2 17 8.58 3.91a2 2 0 0 0 1.66 0L21 17"/></svg>`,
  // ── Annotation tools (all Lucide ISC-licensed icons) ──
  "highlighter": `<svg ${S}><path d="m9 11-6 6v3h9l3-3"/><path d="m22 12-4.6 4.6a2 2 0 0 1-2.8 0l-5.2-5.2a2 2 0 0 1 0-2.8L14 4"/></svg>`,
  "strikethrough": `<svg ${S}><path d="M16 4H9a3 3 0 0 0-2.83 4"/><path d="M14 12a4 4 0 0 1 0 8H6"/><line x1="4" x2="20" y1="12" y2="12"/></svg>`,
  "underline-text": `<svg ${S}><path d="M6 4v6a6 6 0 0 0 12 0V4"/><line x1="4" x2="20" y1="20" y2="20"/></svg>`,
  "wave": `<svg ${S}><path d="M6 4v6a6 6 0 0 0 12 0V4"/><path d="M4 20c2 0 2-1 4-1s2 1 4 1 2-1 4-1 2 1 4 1"/></svg>`,
  "pencil": `<svg ${S}><path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/><path d="m15 5 4 4"/></svg>`,
  "pen-line": `<svg ${S}><path d="M12 20h9"/><path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.854z"/></svg>`,
  "type": `<svg ${S}><polyline points="4 7 4 4 20 4 20 7"/><line x1="9" x2="15" y1="20" y2="20"/><line x1="12" x2="12" y1="4" y2="20"/></svg>`,
  "align-left": `<svg ${S}><line x1="21" x2="3" y1="6" y2="6"/><line x1="15" x2="3" y1="12" y2="12"/><line x1="17" x2="3" y1="18" y2="18"/></svg>`,
  "align-center": `<svg ${S}><line x1="21" x2="3" y1="6" y2="6"/><line x1="17" x2="7" y1="12" y2="12"/><line x1="19" x2="5" y1="18" y2="18"/></svg>`,
  "align-right": `<svg ${S}><line x1="21" x2="3" y1="6" y2="6"/><line x1="21" x2="9" y1="12" y2="12"/><line x1="21" x2="7" y1="18" y2="18"/></svg>`,
  "message-square": `<svg ${S}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
  "text-cursor-input": `<svg ${S}><path d="M5 4h1a3 3 0 0 1 3 3 3 3 0 0 1 3-3h1"/><path d="M13 20h-1a3 3 0 0 1-3-3 3 3 0 0 1-3 3H5"/><path d="M5 16H4a2 2 0 0 1-2-2v-4a2 2 0 0 1 2-2h1"/><path d="M13 8h7a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2h-7"/><path d="M9 7v10"/></svg>`,
  "square-dashed": `<svg ${S}><path d="M5 3a2 2 0 0 0-2 2"/><path d="M19 3a2 2 0 0 1 2 2"/><path d="M21 19a2 2 0 0 1-2 2"/><path d="M5 21a2 2 0 0 1-2-2"/><path d="M9 3h1"/><path d="M9 21h1"/><path d="M14 3h1"/><path d="M14 21h1"/><path d="M3 9v1"/><path d="M21 9v1"/><path d="M3 14v1"/><path d="M21 14v1"/></svg>`,
  "circle-dashed": `<svg ${S}><path d="M10.1 2.182a10 10 0 0 1 3.8 0"/><path d="M13.9 21.818a10 10 0 0 1-3.8 0"/><path d="M17.609 3.721a10 10 0 0 1 2.69 2.7"/><path d="M2.182 13.9a10 10 0 0 1 0-3.8"/><path d="M20.279 17.609a10 10 0 0 1-2.7 2.69"/><path d="M21.818 10.1a10 10 0 0 1 0 3.8"/><path d="M3.721 6.391a10 10 0 0 1 2.7-2.69"/><path d="M6.391 20.279a10 10 0 0 1-2.69-2.7"/></svg>`,
  "line-tool": `<svg ${S}><line x1="5" y1="19" x2="19" y2="5"/><circle cx="5" cy="19" r="1.5" fill="currentColor"/><circle cx="19" cy="5" r="1.5" fill="currentColor"/></svg>`,
  "arrow-tool": `<svg ${S}><line x1="5" y1="19" x2="19" y2="5"/><polyline points="13 5 19 5 19 11"/></svg>`,
  "polygon-tool": `<svg ${S}><path d="M12 3 3 10l3.5 11h11L21 10z"/></svg>`,
  "polyline-tool": `<svg ${S}><polyline points="4 18 9 10 14 14 20 6"/><circle cx="4" cy="18" r="1.5" fill="currentColor"/><circle cx="20" cy="6" r="1.5" fill="currentColor"/></svg>`,
  "eraser": `<svg ${S}><path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/><path d="M22 21H7"/><path d="m5 11 9 9"/></svg>`,
  "palette": `<svg ${S}><circle cx="13.5" cy="6.5" r="0.5" fill="currentColor"/><circle cx="17.5" cy="10.5" r="0.5" fill="currentColor"/><circle cx="8.5" cy="7.5" r="0.5" fill="currentColor"/><circle cx="6.5" cy="12" r="0.5" fill="currentColor"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"/></svg>`,
  // ── Page operations ──
  "rotate-cw": `<svg ${S}><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/></svg>`,
  "rotate-ccw": `<svg ${S}><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>`,
  "trash": `<svg ${S}><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>`,
  "copy-plus": `<svg ${S}><line x1="15" x2="15" y1="12" y2="18"/><line x1="12" x2="18" y1="15" y2="15"/><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`,
  "file-plus": `<svg ${S}><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M9 15h6"/><path d="M12 18v-6"/></svg>`,
  "flatten": `<svg ${S}><path d="M3 3h18"/><path d="M3 21h18"/><path d="M12 8v8"/><path d="m8 12 4-4 4 4"/></svg>`,
  "grip": `<svg ${S}><circle cx="9" cy="5" r="1"/><circle cx="9" cy="12" r="1"/><circle cx="9" cy="19" r="1"/><circle cx="15" cy="5" r="1"/><circle cx="15" cy="12" r="1"/><circle cx="15" cy="19" r="1"/></svg>`,
  // ── Stamp tool ──
  "stamp": `<svg ${S}><path d="M5 22h14"/><path d="M19.27 13.73A2.5 2.5 0 0 0 17.5 13h-11A2.5 2.5 0 0 0 4 15.5V17a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-1.5c0-.66-.26-1.3-.73-1.77Z"/><path d="M14 13V8.5C14 7 15 7 15 5a3 3 0 0 0-6 0c0 2 1 2 1 3.5V13"/></svg>`,
  // ── Measurement tools ──
  "ruler": `<svg ${S}><path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z"/><path d="m14.5 12.5 2-2"/><path d="m11.5 9.5 2-2"/><path d="m8.5 6.5 2-2"/><path d="m17.5 15.5 2-2"/></svg>`,
  "ruler-area": `<svg ${S}><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h4"/><path d="M3 15h4"/><path d="M9 3v4"/><path d="M15 3v4"/></svg>`,
  "settings-2": `<svg ${S}><path d="M20 7h-9"/><path d="M14 17H5"/><circle cx="17" cy="17" r="3"/><circle cx="7" cy="7" r="3"/></svg>`,
  // ── Redaction ──
  "scissors": `<svg ${S}><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><line x1="20" x2="8.12" y1="4" y2="15.88"/><line x1="14.47" x2="20" y1="14.48" y2="20"/><line x1="8.12" x2="12" y1="8.12" y2="12"/></svg>`,
  "shield-off": `<svg ${S}><path d="m2 2 20 20"/><path d="M5 5a1 1 0 0 0-1 1v7c0 5 3.5 7.5 7.67 8.94a1 1 0 0 0 .67.01c2.35-.82 4.48-2.34 5.86-4.66"/><path d="M9.3 3.24A10.37 10.37 0 0 1 12 3c2 0 4.5 1.2 6.24 2.72a1.17 1.17 0 0 1 .42.46"/><path d="M20 6a1 1 0 0 1 1 1v7c0 .55-.04 1.09-.12 1.6"/></svg>`,
  "redact": `<svg ${S}><rect x="3" y="6" width="18" height="12" rx="1" fill="currentColor" stroke="none"/><line x1="6" y1="12" x2="18" y2="12" stroke="var(--lector-bg, #fff)" stroke-width="2"/></svg>`,
  // ── Save ──
  "save": `<svg ${S}><path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/></svg>`,
  // ── Misc ──
  "x": `<svg ${S}><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>`,
  "x-circle": `<svg ${S}><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>`,
  "check-circle": `<svg ${S}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>`,
  "alert-circle": `<svg ${S}><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>`,
  "loading": `<svg ${S}><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/></svg>`,
  // ── Sticky-note marker icons (replaces emoji glyphs) ──
  "sticky-note": `<svg ${S}><path d="M16 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11l5-5V5a2 2 0 0 0-2-2z"/><path d="M15 21v-5a2 2 0 0 1 2-2h5"/></svg>`,
  "help-circle": `<svg ${S}><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>`,
  "check": `<svg ${S}><path d="M20 6 9 17l-5-5"/></svg>`,
  "star": `<svg ${S}><path d="M11.525 2.295a.53.53 0 0 1 .95 0l2.31 4.679a2.123 2.123 0 0 0 1.595 1.16l5.166.756a.53.53 0 0 1 .294.904l-3.736 3.638a2.123 2.123 0 0 0-.611 1.878l.882 5.14a.53.53 0 0 1-.771.56l-4.618-2.428a2.122 2.122 0 0 0-1.973 0L6.396 21.01a.53.53 0 0 1-.77-.56l.881-5.139a2.122 2.122 0 0 0-.611-1.879L2.16 9.795a.53.53 0 0 1 .294-.906l5.165-.755a2.122 2.122 0 0 0 1.597-1.16z"/></svg>`,
  "circle": `<svg ${S}><circle cx="12" cy="12" r="10"/></svg>`,
  "key": `<svg ${S}><path d="m21 2-9.6 9.6"/><circle cx="7.5" cy="15.5" r="5.5"/><path d="m15.5 7.5 3 3L22 7l-3-3"/></svg>`,
  // ── Z-order + grouping ──
  "bring-to-front": `<svg ${S}><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M4 16V6a2 2 0 0 1 2-2h10"/></svg>`,
  "send-to-back": `<svg ${S}><rect x="14" y="14" width="8" height="8" rx="2"/><rect x="2" y="2" width="8" height="8" rx="2"/><path d="M7 14v1a2 2 0 0 0 2 2h1"/><path d="M14 7h1a2 2 0 0 1 2 2v1"/></svg>`,
  "group": `<svg ${S}><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><rect width="7" height="5" x="7" y="7" rx="1"/><rect width="7" height="5" x="10" y="12" rx="1"/></svg>`,
  "ungroup": `<svg ${S}><path d="M5 9V5a2 2 0 0 1 2-2h2"/><path d="M5 15v4a2 2 0 0 0 2 2h2"/><path d="M19 9V5a2 2 0 0 0-2-2h-2"/><path d="M19 15v4a2 2 0 0 1-2 2h-2"/><path d="M3 12h18"/></svg>`,
  // ── Callout & image annotations ──
  "callout": `<svg ${S}><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8z"/></svg>`,
  "image": `<svg ${S}><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>`,
  "image-plus": `<svg ${S}><path d="M16 5h6"/><path d="M19 2v6"/><path d="M21 11.5V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h8.5"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>`,
  // ── Comparison ──
  "git-compare": `<svg ${S}><circle cx="5" cy="6" r="3"/><circle cx="19" cy="18" r="3"/><path d="M12 6h5a2 2 0 0 1 2 2v7"/><path d="M12 18H7a2 2 0 0 1-2-2V9"/></svg>`,
  "git-compare-arrows": `<svg ${S}><circle cx="5" cy="6" r="3"/><circle cx="19" cy="18" r="3"/><path d="M12 6h5a2 2 0 0 1 2 2v3"/><path d="M12 18H7a2 2 0 0 1-2-2v-3"/><path d="m15 9-3-3 3-3"/><path d="m9 15 3 3-3 3"/></svg>`,
  "plus-square": `<svg ${S}><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M8 12h8"/><path d="M12 8v8"/></svg>`,
  "minus-square": `<svg ${S}><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M8 12h8"/></svg>`,
  "replace": `<svg ${S}><path d="M14 4c0-1.1.9-2 2-2"/><path d="M20 2c1.1 0 2 .9 2 2"/><path d="M22 8c0 1.1-.9 2-2 2"/><path d="M16 10c-1.1 0-2-.9-2-2"/><path d="m3 7 3 3 3-3"/><path d="M6 10V5a3 3 0 0 1 3-3h1"/><rect x="2" y="14" width="8" height="8" rx="2"/></svg>`
};
function getIcon(name) {
  return ICONS[name];
}
function isInlineSvg(ref) {
  return ref.startsWith("<svg");
}
function resolveIcon(ref) {
  if (ref === void 0) return void 0;
  if (isInlineSvg(ref)) return ref;
  return ICONS[ref];
}

// src/ui/page-overlays.ts
function pdfRectToDOM(rect, vp) {
  return vp.rectToCss(rect);
}
function positionElement(el, pos) {
  el.style.left = `${pos.x}px`;
  el.style.top = `${pos.y}px`;
  el.style.width = `${pos.w}px`;
  el.style.height = `${pos.h}px`;
}
function rgbaString(c) {
  return `rgba(${c.r}, ${c.g}, ${c.b}, ${c.a / 255})`;
}
var PageOverlayManager = class {
  #engine;
  #cleanups = [];
  #pages = /* @__PURE__ */ new Map();
  // Plugin capabilities (nullable — gracefully degrade if not loaded)
  /**
   * The viewport instance this overlay manager is bound to. In single-pane
   * usage this is the LectorViewer's primary viewport. In split-pane
   * usage each LectorPane has its own overlay manager bound to its own
   * viewport instance.
   */
  #viewport;
  #document;
  #textLayer;
  #search;
  #navigation;
  #annotation;
  #form;
  #formatting;
  /**
   * Active comparison overlay state. When non-null, every page that has
   * a corresponding `PageDiff` for this side renders coloured highlight
   * rectangles for each `ComparisonChange`. The active change index lets
   * the UI emphasise the change the user is currently navigating to.
   */
  #comparison = null;
  constructor(engine, viewport, formatting = null) {
    this.#engine = engine;
    this.#viewport = viewport;
    const p = engine.plugins;
    this.#document = p.get("document");
    this.#textLayer = p.tryGet("text-layer");
    this.#search = p.tryGet("search");
    this.#navigation = p.tryGet("navigation");
    this.#annotation = p.tryGet("annotation");
    this.#form = p.tryGet("form");
    this.#formatting = formatting ?? p.tryGet("formatting");
    this.#wireEffects();
  }
  /** Resolve a translation key via the i18n plugin, with fallback. */
  #t(key, params) {
    const i18n = this.#engine.plugins.tryGet("i18n");
    if (i18n) return i18n.t(key, params);
    const last = key.split(".").pop() ?? key;
    let result = last.replace(/([A-Z])/g, " $1").replace(/^./, (s) => s.toUpperCase()).trim();
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        result = result.replaceAll(`{${k}}`, String(v));
      }
    }
    return result;
  }
  // ── Doc resolution ──
  /**
   * Resolve the document handle this overlay manager should operate on.
   * Always prefers the bound viewport's pinned doc — falling back to
   * the engine's active document only when the viewport has no pin.
   *
   * Without this, in split-tab mode the left pane's overlay manager
   * would re-render with the right doc's annotations whenever the user
   * clicked the right pane (because activeDocument follows the active
   * viewport). The overlays must always reflect the doc shown in the
   * pane that owns them, regardless of which pane is currently active.
   */
  #resolveDoc() {
    const pinnedId = this.#viewport.docId.peek();
    if (pinnedId !== null) {
      return this.#document.getHandle(pinnedId) ?? null;
    }
    return this.#document.activeDocument.peek();
  }
  // ── Page lifecycle ──
  /**
   * Called by the viewer when a page element is created.
   * Adds an overlay div to the page element.
   */
  attachPage(pageIndex, pageEl, pageWidthPts, pageHeightPts) {
    const existing = this.#pages.get(pageIndex);
    if (existing) {
      if (existing.pageWidthPts !== pageWidthPts || existing.pageHeightPts !== pageHeightPts) {
        existing.pageWidthPts = pageWidthPts;
        existing.pageHeightPts = pageHeightPts;
      }
      const nowDoc = this.#resolveDoc()?.id ?? null;
      if (existing.docId !== nowDoc) {
        existing.docId = nowDoc;
        this.#engine.plugins.events.emit("ui:page-mounted", {
          pageIndex,
          overlayEl: existing.hostOverlay,
          docId: nowDoc
        });
      }
      return;
    }
    const overlay = document.createElement("div");
    overlay.className = "lector-page__overlay";
    pageEl.appendChild(overlay);
    const hostOverlay = document.createElement("div");
    hostOverlay.className = "lector-page__host-overlay";
    pageEl.appendChild(hostOverlay);
    const doc = this.#resolveDoc();
    const seedRotation = doc !== null ? this.#engine.pageRotation.get(doc.id, pageIndex) : 0;
    this.#pages.set(pageIndex, {
      overlay,
      hostOverlay,
      docId: doc?.id ?? null,
      pageWidthPts,
      pageHeightPts,
      rotation: seedRotation,
      linksLoaded: false,
      annotsLoaded: false,
      formsLoaded: false
    });
    this.#engine.plugins.events.emit("ui:page-mounted", {
      pageIndex,
      overlayEl: hostOverlay,
      docId: doc?.id ?? null
    });
  }
  /**
   * Build the {@link PageViewport} for a page at the current scale. Cheap to
   * construct (a matrix + its inverse), so callers build one per render pass
   * rather than caching across zoom levels.
   */
  #vp(state, scale) {
    return PageViewport.fromRotatedSize(
      state.pageWidthPts,
      state.pageHeightPts,
      state.rotation,
      scale
    );
  }
  /** Called by the viewer when a page element is removed. */
  detachPage(pageIndex) {
    const state = this.#pages.get(pageIndex);
    if (state) {
      state.overlay.remove();
      state.hostOverlay.remove();
      this.#pages.delete(pageIndex);
      this.#engine.plugins.events.emit("ui:page-unmounted", { pageIndex });
    }
  }
  // ── Reactive wiring ──
  #wireEffects() {
    if (this.#textLayer) {
      this.#cleanups.push(effect(() => {
        const sel = this.#textLayer.selection.value;
        this.#renderTextSelection(sel);
      }));
    }
    if (this.#search) {
      this.#cleanups.push(effect(() => {
        const result = this.#search.result.value;
        const activeIdx = this.#search.activeMatchIndex.value;
        this.#renderSearchHighlights(result, activeIdx);
      }));
    }
    if (this.#annotation) {
      this.#cleanups.push(effect(() => {
        const selectedId = this.#annotation.selectedAnnotation.value;
        this.#updateAnnotationSelection(selectedId);
      }));
    }
    this.#cleanups.push(effect(() => {
      const visible = this.#viewport.visiblePages.value;
      void this.#loadOverlaysForPages(visible);
    }));
    {
      const events = this.#engine.plugins.events;
      this.#cleanups.push(events.on("page-ops:page-rotated", (...args) => {
        const docId = args[0];
        const pageIndex = args[1];
        const doc = this.#resolveDoc();
        if (!doc || doc.id !== docId) return;
        this.#engine.pageRotation.invalidate(docId, pageIndex);
        void this.#ensureRotationAndRebuild(doc, pageIndex);
      }));
    }
    if (this.#annotation) {
      const events = this.#engine.plugins.events;
      const refreshAnnots = () => {
        const scale = this.#viewport.scale.peek();
        const doc = this.#resolveDoc();
        if (!doc) return;
        for (const [pageIndex, state] of this.#pages) {
          for (const el of state.overlay.querySelectorAll(".lector-annot-overlay, .lector-annot-note, .lector-annot-measure-label, .lector-line-handle, .lector-resize-handle, .lector-vertex-handle")) el.remove();
          const annotations = this.#annotation.getForPage(doc.id, pageIndex);
          if (annotations.length > 0) {
            this.#renderAnnotations(state, annotations, scale);
          }
        }
        const selId = this.#annotation.selectedAnnotation.peek();
        if (selId) this.#updateAnnotationSelection(selId);
      };
      this.#cleanups.push(events.on("annotation:page-loaded", refreshAnnots));
      this.#cleanups.push(events.on("annotation:created", refreshAnnots));
      this.#cleanups.push(events.on("annotation:deleted", refreshAnnots));
      this.#cleanups.push(events.on("annotation:updated", (...args) => {
        const event = args[0];
        if (event?.patch) {
          const keys = Object.keys(event.patch);
          if (keys.length === 1 && keys[0] === "readAt") return;
        }
        refreshAnnots();
      }));
    }
  }
  // ── Loading per-page overlay data ──
  async #loadOverlaysForPages(pages) {
    const doc = this.#resolveDoc();
    if (!doc) return;
    for (const pageIndex of pages) {
      const state = this.#pages.get(pageIndex);
      if (!state) continue;
      void this.#ensureRotationAndRebuild(doc, pageIndex);
      if (!state.linksLoaded && this.#navigation) {
        state.linksLoaded = true;
        void this.#loadLinks(doc, pageIndex, state);
      }
      if (!state.annotsLoaded && this.#annotation) {
        state.annotsLoaded = true;
        void this.#loadAnnotations(doc, pageIndex, state);
      }
      if (!state.formsLoaded && this.#form) {
        state.formsLoaded = true;
        void this.#loadFormFields(doc, pageIndex, state);
      }
    }
    if (this.#comparison) {
      this.#renderComparisonOverlays();
    }
  }
  // ── Page rotation resolution ──
  /**
   * Resolve a page's rotation via the shared engine cache, store it on the
   * state, and rebuild that page's overlays if the rotation differs from what
   * was already rendered. Safe to call fire-and-forget.
   */
  async #ensureRotationAndRebuild(doc, pageIndex) {
    const before = this.#pages.get(pageIndex)?.rotation ?? 0;
    const rot = await this.#engine.pageRotation.resolve(doc.id, pageIndex);
    const state = this.#pages.get(pageIndex);
    if (!state) return;
    if (this.#resolveDoc()?.id !== doc.id) return;
    state.rotation = rot;
    if (rot !== before) this.#rebuildPage(pageIndex);
  }
  /** Re-render every overlay layer for a single page (used after a rotation change). */
  #rebuildPage(pageIndex) {
    const state = this.#pages.get(pageIndex);
    if (!state) return;
    state.overlay.innerHTML = "";
    state.linksLoaded = false;
    state.annotsLoaded = false;
    state.formsLoaded = false;
    void this.#loadOverlaysForPages([pageIndex]);
    if (this.#textLayer) this.#renderTextSelection(this.#textLayer.selection.peek());
    if (this.#search) {
      this.#renderSearchHighlights(
        this.#search.result.peek(),
        this.#search.activeMatchIndex.peek()
      );
    }
    if (this.#comparison) this.#renderComparisonOverlays();
  }
  // ── Text selection ──
  #renderTextSelection(selection) {
    for (const [, state2] of this.#pages) {
      for (const el of state2.overlay.querySelectorAll(".lector-text-highlight")) {
        el.remove();
      }
    }
    if (!selection) return;
    const ownDoc = this.#resolveDoc();
    if (!ownDoc || selection.docId !== ownDoc.id) return;
    const state = this.#pages.get(selection.pageIndex);
    if (!state) return;
    const scale = this.#viewport.scale.peek();
    for (const rect of selection.rects) {
      const pos = pdfRectToDOM(rect, this.#vp(state, scale));
      const el = document.createElement("div");
      el.className = "lector-text-highlight";
      positionElement(el, pos);
      state.overlay.appendChild(el);
    }
  }
  // ── Search highlights ──
  #renderSearchHighlights(result, activeIndex) {
    for (const [, state] of this.#pages) {
      for (const el of state.overlay.querySelectorAll(".lector-search-highlight")) {
        el.remove();
      }
    }
    if (!result || result.matches.length === 0) return;
    const ownDoc = this.#resolveDoc();
    if (!ownDoc || result.docId !== ownDoc.id) return;
    const matches = result.matches;
    const scale = this.#viewport.scale.peek();
    for (let i = 0; i < matches.length; i++) {
      const match = matches[i];
      const state = this.#pages.get(match.pageIndex);
      if (!state) continue;
      for (const rect of match.rects) {
        const pos = pdfRectToDOM(rect, this.#vp(state, scale));
        const el = document.createElement("div");
        el.className = i === activeIndex ? "lector-search-highlight lector-search-highlight--active" : "lector-search-highlight";
        positionElement(el, pos);
        state.overlay.appendChild(el);
      }
    }
    if (activeIndex >= 0 && activeIndex < matches.length) {
      const match = matches[activeIndex];
      this.#viewport.scrollToPage(match.pageIndex);
    }
  }
  // ── Links ──
  async #loadLinks(doc, pageIndex, state) {
    try {
      const [links, webLinks] = await Promise.all([
        this.#navigation.getPageLinks(doc.id, pageIndex),
        this.#navigation.getPageWebLinks(doc.id, pageIndex)
      ]);
      if (this.#resolveDoc()?.id !== doc.id) return;
      const scale = this.#viewport.scale.peek();
      this.#renderLinks(state, links, webLinks, scale);
    } catch {
    }
  }
  #renderLinks(state, links, webLinks, scale) {
    for (const link of links) {
      const pos = pdfRectToDOM(link.rect, this.#vp(state, scale));
      const el = document.createElement("div");
      el.className = "lector-link-overlay";
      el.setAttribute("role", "link");
      el.tabIndex = 0;
      if (link.target.type === "uri") el.classList.add("lector-link-overlay--uri");
      positionElement(el, pos);
      const activate = (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.#handleLinkClick(link.target);
      };
      el.addEventListener("click", activate);
      el.addEventListener("keydown", (e) => {
        if (e.key === "Enter") activate(e);
      });
      if (link.target.type === "uri") {
        el.title = link.target.uri;
        el.setAttribute("aria-label", link.target.uri);
      } else if (link.target.type === "goto") {
        const label = this.#t("page.goToPageN", { page: link.target.destination.pageIndex + 1 });
        el.title = label;
        el.setAttribute("aria-label", label);
      }
      state.overlay.appendChild(el);
    }
    for (const webLink of webLinks) {
      for (const rect of webLink.rects) {
        const pos = pdfRectToDOM(rect, this.#vp(state, scale));
        const el = document.createElement("div");
        el.className = "lector-link-overlay lector-link-overlay--uri";
        el.setAttribute("role", "link");
        el.tabIndex = 0;
        el.title = webLink.url;
        el.setAttribute("aria-label", webLink.url);
        positionElement(el, pos);
        const activate = (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.#handleLinkClick({ type: "uri", uri: webLink.url });
        };
        el.addEventListener("click", activate);
        el.addEventListener("keydown", (e) => {
          if (e.key === "Enter") activate(e);
        });
        state.overlay.appendChild(el);
      }
    }
  }
  #handleLinkClick(target) {
    if (this.#navigation) {
      this.#navigation.navigateToTarget(target);
    }
    if (target.type === "uri") {
      this.#engine.plugins.events.emit("navigation:external-link-clicked", target.uri);
    }
  }
  // ── Annotations ──
  async #loadAnnotations(doc, pageIndex, state) {
    try {
      await this.#annotation.loadPage(doc.id, pageIndex);
      if (this.#resolveDoc()?.id !== doc.id) return;
      const annotations = this.#annotation.getForPage(doc.id, pageIndex);
      const scale = this.#viewport.scale.peek();
      this.#renderAnnotations(state, annotations, scale);
    } catch {
    }
  }
  #renderAnnotations(state, annotations, scale) {
    for (const tracked of annotations) {
      const annot = tracked.data;
      if (!isUserAnnotation(annot.subtype)) continue;
      if (annot.markup && annot.markup.quadPoints.length > 0) {
        this.#renderMarkupAnnotation(state, annot, scale);
        continue;
      }
      if (annot.subtype === 13 && annot.tag === "image" && annot.image) {
        this.#renderImageAnnotation(state, annot, scale);
        continue;
      }
      if (annot.subtype === 13) {
        this.#renderStampAnnotation(state, annot, scale);
        continue;
      }
      if (annot.subtype === 28 || annot.tag === "redaction") {
        this.#renderRedactionAnnotation(state, annot, scale);
        continue;
      }
      if (annot.tag === "measure-distance" || annot.tag === "measure-area" || annot.tag === "measure-perimeter") {
        this.#renderMeasurementAnnotation(state, annot, scale);
        continue;
      }
      const isLineTag = annot.tag === "line" || annot.tag === "arrow" || annot.tag === "arrow-start" || annot.tag === "arrow-both";
      if (isLineTag && annot.line) {
        this.#renderLineAnnotation(state, annot, scale);
        continue;
      } else if (annot.ink && annot.ink.strokes.length > 0 && isLineTag) {
        const stroke = annot.ink.strokes[0];
        if (stroke.length >= 2) {
          const lineAnnot = {
            ...annot,
            line: {
              start: stroke[0],
              end: stroke[stroke.length - 1]
            }
          };
          this.#renderLineAnnotation(state, lineAnnot, scale);
          continue;
        }
      }
      if (annot.ink && annot.ink.strokes.length > 0) {
        this.#renderInkAnnotation(state, annot, scale);
        continue;
      }
      if (annot.subtype === FpdfAnnotSubtype2.TEXT) {
        this.#renderStickyNote(state, annot, scale);
        continue;
      }
      this.#renderRectAnnotation(state, annot, scale);
    }
  }
  #renderMarkupAnnotation(state, annot, scale) {
    const color = annot.color ? rgbaString(annot.color) : "rgba(255, 255, 0, 0.4)";
    let modifier = "";
    if (annot.subtype === FpdfAnnotSubtype2.HIGHLIGHT) modifier = "--highlight";
    else if (annot.subtype === FpdfAnnotSubtype2.UNDERLINE) modifier = "--underline";
    else if (annot.subtype === FpdfAnnotSubtype2.SQUIGGLY) modifier = "--squiggly";
    else if (annot.subtype === FpdfAnnotSubtype2.STRIKEOUT) modifier = "--strikeout";
    for (const qp of annot.markup.quadPoints) {
      const left = Math.min(qp.x1, qp.x3);
      const right = Math.max(qp.x2, qp.x4);
      const top = Math.max(qp.y1, qp.y2);
      const bottom = Math.min(qp.y3, qp.y4);
      const rect = { left, top, right, bottom };
      const pos = pdfRectToDOM(rect, this.#vp(state, scale));
      const el = document.createElement("div");
      el.className = `lector-annot-overlay lector-annot-overlay${modifier}`;
      el.dataset["annotId"] = annot.id;
      if (annot.subtype === FpdfAnnotSubtype2.HIGHLIGHT) {
        el.style.background = color;
      } else {
        el.style.color = color;
        el.style.borderColor = color;
      }
      positionElement(el, pos);
      this.#attachDragToMove(el, annot, state);
      if (annot.contents) el.title = annot.contents;
      state.overlay.appendChild(el);
    }
  }
  #renderInkAnnotation(state, annot, scale) {
    const opaqueColor = annot.color ? `rgb(${annot.color.r}, ${annot.color.g}, ${annot.color.b})` : "rgb(0, 0, 0)";
    const opacity = annot.color ? annot.color.a / 255 : 1;
    const borderWidth = annot.border?.width ?? 2;
    const vp = this.#vp(state, scale);
    const vbW = vp.width;
    const vbH = vp.height;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("lector-annot-overlay", "lector-annot-overlay--ink");
    svg.setAttribute("viewBox", `0 0 ${vbW} ${vbH}`);
    svg.style.position = "absolute";
    svg.style.inset = "0";
    svg.style.width = "100%";
    svg.style.height = "100%";
    svg.style.overflow = "visible";
    svg.dataset["annotId"] = annot.id;
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    if (opacity < 1) g.setAttribute("opacity", String(opacity));
    for (const stroke of annot.ink.strokes) {
      if (stroke.length < 2) continue;
      const points = stroke.map((p) => {
        const d = vp.pointToCss(p.x, p.y);
        return `${d.x},${d.y}`;
      }).join(" ");
      const hitArea = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      hitArea.setAttribute("points", points);
      hitArea.setAttribute("fill", "none");
      hitArea.setAttribute("stroke", "transparent");
      hitArea.setAttribute("stroke-width", String(Math.max(12, borderWidth * scale * 3)));
      hitArea.setAttribute("stroke-linecap", "round");
      svg.appendChild(hitArea);
      const hasPressure = stroke.some((p) => typeof p.pressure === "number");
      if (hasPressure) {
        const baseWidth = borderWidth * scale;
        for (let i = 1; i < stroke.length; i++) {
          const a = stroke[i - 1];
          const b = stroke[i];
          const pa = a.pressure ?? 0.5;
          const pb = b.pressure ?? 0.5;
          const widthScale = 0.4 + (pa + pb) / 2 * 1.2;
          const da = vp.pointToCss(a.x, a.y);
          const db = vp.pointToCss(b.x, b.y);
          const seg = document.createElementNS("http://www.w3.org/2000/svg", "line");
          seg.setAttribute("x1", String(da.x));
          seg.setAttribute("y1", String(da.y));
          seg.setAttribute("x2", String(db.x));
          seg.setAttribute("y2", String(db.y));
          seg.setAttribute("stroke", opaqueColor);
          seg.setAttribute("stroke-width", String(baseWidth * widthScale));
          seg.setAttribute("stroke-linecap", "round");
          g.appendChild(seg);
        }
      } else {
        const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        polyline.setAttribute("points", points);
        polyline.setAttribute("fill", "none");
        polyline.setAttribute("stroke", opaqueColor);
        polyline.setAttribute("stroke-width", String(borderWidth * scale));
        polyline.setAttribute("stroke-linecap", "round");
        polyline.setAttribute("stroke-linejoin", "round");
        g.appendChild(polyline);
      }
    }
    svg.appendChild(g);
    {
      let sMinX = Infinity, sMinY = Infinity, sMaxX = -Infinity, sMaxY = -Infinity;
      for (const stroke of annot.ink.strokes) {
        for (const p of stroke) {
          const d = vp.pointToCss(p.x, p.y);
          if (d.x < sMinX) sMinX = d.x;
          if (d.y < sMinY) sMinY = d.y;
          if (d.x > sMaxX) sMaxX = d.x;
          if (d.y > sMaxY) sMaxY = d.y;
        }
      }
      const pad = 4;
      const selRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      selRect.setAttribute("x", String(sMinX - pad));
      selRect.setAttribute("y", String(sMinY - pad));
      selRect.setAttribute("width", String(sMaxX - sMinX + pad * 2));
      selRect.setAttribute("height", String(sMaxY - sMinY + pad * 2));
      selRect.setAttribute("fill", "transparent");
      selRect.setAttribute("stroke", "#3b82f6");
      selRect.setAttribute("stroke-width", "2");
      selRect.setAttribute("stroke-dasharray", "4 3");
      selRect.setAttribute("rx", "3");
      selRect.setAttribute("cursor", "grab");
      selRect.classList.add("lector-ink-select-rect");
      selRect.style.display = "none";
      svg.appendChild(selRect);
    }
    {
      let getCornerPos2 = function(idx, minX, minY, maxX, maxY) {
        const positions = [
          { x: minX - pad, y: minY - pad },
          { x: maxX + pad, y: minY - pad },
          { x: minX - pad, y: maxY + pad },
          { x: maxX + pad, y: maxY + pad }
        ];
        return positions[idx];
      }, repositionHandles2 = function(minX, minY, maxX, maxY) {
        for (let i = 0; i < 4; i++) {
          const pos = getCornerPos2(i, minX, minY, maxX, maxY);
          handleEls[i].style.left = `${pos.x}px`;
          handleEls[i].style.top = `${pos.y}px`;
        }
      };
      var getCornerPos = getCornerPos2, repositionHandles = repositionHandles2;
      const pad = 4;
      let sMinX = Infinity, sMinY = Infinity, sMaxX = -Infinity, sMaxY = -Infinity;
      for (const stroke of annot.ink.strokes) {
        for (const p of stroke) {
          const d = vp.pointToCss(p.x, p.y);
          if (d.x < sMinX) sMinX = d.x;
          if (d.y < sMinY) sMinY = d.y;
          if (d.x > sMaxX) sMaxX = d.x;
          if (d.y > sMaxY) sMaxY = d.y;
        }
      }
      const handleEls = [];
      const cornerDefs = [
        { cursor: "nwse-resize", anchorIdx: 3 },
        // NW → anchor SE
        { cursor: "nesw-resize", anchorIdx: 2 },
        // NE → anchor SW
        { cursor: "nesw-resize", anchorIdx: 1 },
        // SW → anchor NE
        { cursor: "nwse-resize", anchorIdx: 0 }
        // SE → anchor NW
      ];
      for (let ci = 0; ci < 4; ci++) {
        const def = cornerDefs[ci];
        const initPos = getCornerPos2(ci, sMinX, sMinY, sMaxX, sMaxY);
        const handle = document.createElement("div");
        handle.className = "lector-resize-handle";
        handle.dataset["forAnnot"] = annot.id;
        handle.style.cssText = `position:absolute;width:8px;height:8px;background:white;border:1.5px solid #3b82f6;border-radius:1px;cursor:${def.cursor};pointer-events:auto;transform:translate(-50%,-50%);display:none;z-index:10;`;
        handle.style.left = `${initPos.x}px`;
        handle.style.top = `${initPos.y}px`;
        handleEls.push(handle);
        let dragging = false;
        let origCornerX = 0;
        let origCornerY = 0;
        let anchorDomX = 0;
        let anchorDomY = 0;
        handle.addEventListener("pointerdown", (e) => {
          e.stopPropagation();
          e.preventDefault();
          dragging = true;
          origCornerX = parseFloat(handle.style.left);
          origCornerY = parseFloat(handle.style.top);
          const anchorEl = handleEls[def.anchorIdx];
          anchorDomX = parseFloat(anchorEl.style.left);
          anchorDomY = parseFloat(anchorEl.style.top);
          handle.setPointerCapture(e.pointerId);
          this.#engine.plugins.events.emit("annotation:drag-start");
        });
        handle.addEventListener("pointermove", (e) => {
          if (!dragging) return;
          e.stopPropagation();
          const container = state.overlay.closest(".lector-canvas");
          if (!container) return;
          const cRect = container.getBoundingClientRect();
          const pagePos = this.#viewport.pagePositions.peek().find((p) => p.pageIndex === annot.pageIndex);
          if (!pagePos) return;
          let domPx = e.clientX - cRect.left + container.scrollLeft - pagePos.x;
          let domPy = e.clientY - cRect.top + container.scrollTop - pagePos.y;
          const origW = origCornerX - anchorDomX;
          const origH = origCornerY - anchorDomY;
          let sx = origW !== 0 ? (domPx - anchorDomX) / origW : 1;
          let sy = origH !== 0 ? (domPy - anchorDomY) / origH : 1;
          if (e.shiftKey && origW !== 0 && origH !== 0) {
            const absSx = Math.abs(sx);
            const absSy = Math.abs(sy);
            const uniform = Math.max(absSx, absSy);
            sx = uniform * Math.sign(sx || 1);
            sy = uniform * Math.sign(sy || 1);
            domPx = anchorDomX + origW * sx;
            domPy = anchorDomY + origH * sy;
          }
          for (let si = 0; si < annot.ink.strokes.length; si++) {
            const stroke = annot.ink.strokes[si];
            if (stroke.length < 2) continue;
            const scaledPts = stroke.map((p) => {
              const d = vp.pointToCss(p.x, p.y);
              const nx = anchorDomX + (d.x - anchorDomX) * sx;
              const ny = anchorDomY + (d.y - anchorDomY) * sy;
              return `${nx},${ny}`;
            }).join(" ");
            const gPolylines = svg.querySelector("g")?.querySelectorAll("polyline");
            if (gPolylines && gPolylines[si]) {
              gPolylines[si].setAttribute("points", scaledPts);
            }
            const hitAreas = Array.from(svg.children).filter(
              (c) => c.tagName === "polyline" && c.getAttribute("stroke") === "transparent"
            );
            if (hitAreas[si]) {
              hitAreas[si].setAttribute("points", scaledPts);
            }
          }
          let nMinX = Infinity, nMinY = Infinity, nMaxX = -Infinity, nMaxY = -Infinity;
          for (const stroke of annot.ink.strokes) {
            for (const p of stroke) {
              const d = vp.pointToCss(p.x, p.y);
              const nx = anchorDomX + (d.x - anchorDomX) * sx;
              const ny = anchorDomY + (d.y - anchorDomY) * sy;
              if (nx < nMinX) nMinX = nx;
              if (ny < nMinY) nMinY = ny;
              if (nx > nMaxX) nMaxX = nx;
              if (ny > nMaxY) nMaxY = ny;
            }
          }
          const selRect = svg.querySelector(".lector-ink-select-rect");
          if (selRect) {
            const rPad = 4;
            selRect.setAttribute("x", String(nMinX - rPad));
            selRect.setAttribute("y", String(nMinY - rPad));
            selRect.setAttribute("width", String(nMaxX - nMinX + rPad * 2));
            selRect.setAttribute("height", String(nMaxY - nMinY + rPad * 2));
          }
          repositionHandles2(nMinX, nMinY, nMaxX, nMaxY);
        });
        const finishResize = () => {
          if (!dragging) return;
          dragging = false;
          const finalX = parseFloat(handle.style.left);
          const finalY = parseFloat(handle.style.top);
          const origW = origCornerX - anchorDomX;
          const origH = origCornerY - anchorDomY;
          const newW = finalX - anchorDomX;
          const newH = finalY - anchorDomY;
          const sx = origW !== 0 ? newW / origW : 1;
          const sy = origH !== 0 ? newH / origH : 1;
          if (this.#annotation) {
            const doc = this.#resolveDoc();
            if (doc && annot.ink) {
              const newStrokes = annot.ink.strokes.map(
                (s) => s.map((p) => {
                  const d = vp.pointToCss(p.x, p.y);
                  const np = vp.cssPointToPdf(
                    anchorDomX + (d.x - anchorDomX) * sx,
                    anchorDomY + (d.y - anchorDomY) * sy
                  );
                  return p.pressure !== void 0 ? { x: np.x, y: np.y, pressure: p.pressure } : { x: np.x, y: np.y };
                })
              );
              let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
              for (const s of newStrokes) {
                for (const p of s) {
                  if (p.x < minX) minX = p.x;
                  if (p.y < minY) minY = p.y;
                  if (p.x > maxX) maxX = p.x;
                  if (p.y > maxY) maxY = p.y;
                }
              }
              void this.#annotation.update(doc.id, annot.id, {
                ink: { strokes: newStrokes },
                rect: { left: minX, bottom: minY, right: maxX, top: maxY }
              }).then(() => {
                this.#engine.plugins.events.emit("annotation:drag-end", annot.id);
              });
            }
          }
        };
        handle.addEventListener("pointerup", finishResize);
        handle.addEventListener("pointercancel", finishResize);
        state.overlay.appendChild(handle);
      }
    }
    svg.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      if (!this.#annotation || this.#annotation.selectedAnnotation.peek() !== annot.id) return;
      this.#enterVertexEditMode(state, annot, scale);
    });
    this.#attachDragToMove(svg, annot, state);
    if (annot.contents) svg.setAttribute("title", annot.contents);
    state.overlay.appendChild(svg);
  }
  #renderRectAnnotation(state, annot, scale) {
    const pos = pdfRectToDOM(annot.rect, this.#vp(state, scale));
    const el = document.createElement("div");
    el.dataset["annotId"] = annot.id;
    if (annot.subtype === FpdfAnnotSubtype2.FREETEXT && annot.freeText) {
      el.className = "lector-annot-overlay lector-annot-overlay--freetext";
      el.textContent = annot.freeText.text || "";
      if (annot.freeText.fontSize > 0) {
        el.style.fontSize = `${annot.freeText.fontSize * scale}px`;
      }
      const borderW = (annot.border?.width ?? 1) * scale;
      if (annot.color && annot.color.a > 0) {
        el.style.border = `${borderW}px solid ${rgbaString(annot.color)}`;
      }
      if (annot.freeText.fontColor) {
        el.style.color = `rgb(${annot.freeText.fontColor.r}, ${annot.freeText.fontColor.g}, ${annot.freeText.fontColor.b})`;
      }
      if (annot.freeText.textAlign) {
        el.style.textAlign = annot.freeText.textAlign;
      }
      const enterEdit = () => {
        el.contentEditable = "true";
        el.style.outline = "2px solid var(--lector-accent)";
        el.style.background = "rgba(255,255,255,0.95)";
        el.style.cursor = "text";
        el.focus();
        const finish = () => {
          el.contentEditable = "false";
          el.style.outline = "";
          el.style.background = "";
          el.style.cursor = "";
          const text = el.textContent ?? "";
          if (this.#annotation) {
            const doc = this.#resolveDoc();
            if (doc) {
              void this.#annotation.update(doc.id, annot.id, {
                freeText: { ...annot.freeText, text },
                contents: text
              });
            }
          }
        };
        el.addEventListener("blur", finish, { once: true });
        el.addEventListener("keydown", (ke) => {
          if (ke.key === "Escape") {
            el.blur();
          }
        });
      };
      el.addEventListener("dblclick", (e) => {
        e.stopPropagation();
        enterEdit();
      });
      const unsub = this.#engine.plugins.events.on("annotation:edit-requested", (...args) => {
        if (args[0] === annot.id) {
          unsub();
          requestAnimationFrame(() => enterEdit());
        }
      });
      this.#cleanups.push(unsub);
    } else if (annot.subtype === FpdfAnnotSubtype2.LINE && annot.line) {
      this.#renderLineAnnotation(state, annot, scale);
      return;
    } else {
      el.className = "lector-annot-overlay lector-annot-overlay--rect";
      const borderW = (annot.border?.width ?? 2) * scale;
      if (annot.color && annot.color.a > 0) {
        el.style.borderColor = rgbaString(annot.color);
        el.style.borderWidth = `${borderW}px`;
      } else {
        el.style.border = "none";
      }
      if (annot.interiorColor && annot.interiorColor.a > 0) {
        el.style.background = rgbaString(annot.interiorColor);
      }
      if (annot.subtype === FpdfAnnotSubtype2.CIRCLE) {
        el.style.borderRadius = "50%";
      }
    }
    if (annot.opacity !== void 0 && annot.opacity < 1) {
      el.style.opacity = String(annot.opacity);
    }
    positionElement(el, pos);
    this.#attachDragToMove(el, annot, state);
    if (annot.contents) el.title = annot.contents;
    state.overlay.appendChild(el);
    this.#attachRectResizeHandles(state, annot, scale);
    if (annot.subtype === FpdfAnnotSubtype2.FREETEXT && annot.tag === "callout" && annot.callout) {
      this.#renderCalloutLeader(state, annot, scale);
    }
  }
  #renderStampAnnotation(state, annot, scale) {
    const pos = pdfRectToDOM(annot.rect, this.#vp(state, scale));
    const el = document.createElement("div");
    el.dataset["annotId"] = annot.id;
    el.className = "lector-annot-overlay lector-annot-stamp";
    const stampName = annot.stamp?.name ?? annot.contents ?? "STAMP";
    const display = stampName.replace(/([a-z])([A-Z])/g, "$1 $2").toUpperCase();
    el.textContent = display;
    const color = annot.color && annot.color.a > 0 ? `rgb(${annot.color.r}, ${annot.color.g}, ${annot.color.b})` : "#e44234";
    el.style.color = color;
    el.style.borderColor = color;
    if (annot.opacity !== void 0 && annot.opacity < 1) {
      el.style.opacity = String(annot.opacity);
    }
    positionElement(el, pos);
    this.#attachDragToMove(el, annot, state);
    if (annot.contents) el.title = annot.contents;
    state.overlay.appendChild(el);
    this.#attachRectResizeHandles(state, annot, scale);
  }
  /**
   * Render an image annotation: a STAMP-subtype annotation with
   * `tag='image'` and `image.imageRef` containing a data URI. The image
   * fills the annotation rect and is selectable / draggable like every
   * other overlay. Both the rect-positioned `<div>` wrapper and the
   * inner `<img>` are part of the same overlay element so the entire
   * thing reacts to drag, selection, and the popover toolbar.
   *
   * Image data is round-tripped through pdfium via private string keys
   * (see `worker/annotation-ops.ts`); this renderer only consumes the
   * already-decoded `image.imageRef`.
   */
  #renderImageAnnotation(state, annot, scale) {
    const pos = pdfRectToDOM(annot.rect, this.#vp(state, scale));
    const el = document.createElement("div");
    el.dataset["annotId"] = annot.id;
    el.className = "lector-annot-overlay lector-annot-image";
    const img = document.createElement("img");
    img.src = annot.image.imageRef;
    img.alt = annot.contents ?? "";
    img.draggable = false;
    img.style.cssText = "width:100%;height:100%;object-fit:contain;display:block;pointer-events:none;user-select:none;";
    el.appendChild(img);
    if (annot.opacity !== void 0 && annot.opacity < 1) {
      el.style.opacity = String(annot.opacity);
    }
    positionElement(el, pos);
    this.#attachDragToMove(el, annot, state);
    if (annot.contents) el.title = annot.contents;
    state.overlay.appendChild(el);
    this.#attachRectResizeHandles(state, annot, scale);
  }
  /**
   * Draw the leader line + arrowhead for a callout annotation. The
   * line starts at the midpoint of whichever rect edge is closest to
   * the endpoint, then either runs straight to the endpoint or — when
   * `callout.knee` is provided — bends through that knee point first.
   *
   * The SVG is positioned absolutely over the page area in the page
   * overlay state, sized to the page dimensions, so leader coordinates
   * are computed in the same `pdfX * scale` space as the text overlay.
   * One SVG element per callout — cheap, no z-index conflicts with
   * other overlays because the SVG is appended last per render pass.
   */
  #renderCalloutLeader(state, annot, scale) {
    const callout = annot.callout;
    const vp = this.#vp(state, scale);
    const pdfTop = Math.max(annot.rect.top, annot.rect.bottom);
    const pdfBottom = Math.min(annot.rect.top, annot.rect.bottom);
    const rectN = { left: annot.rect.left, right: annot.rect.right, top: pdfTop, bottom: pdfBottom };
    const cx = (rectN.left + rectN.right) / 2;
    const cy = (rectN.top + rectN.bottom) / 2;
    const dx = callout.endpoint.x - cx;
    const dy = callout.endpoint.y - cy;
    let anchor;
    if (Math.abs(dx) >= Math.abs(dy)) {
      anchor = dx >= 0 ? { x: rectN.right, y: cy } : { x: rectN.left, y: cy };
    } else {
      anchor = dy >= 0 ? { x: cx, y: rectN.top } : { x: cx, y: rectN.bottom };
    }
    const toDom = (p) => vp.pointToCss(p.x, p.y);
    const a = toDom(anchor);
    const e = toDom(callout.endpoint);
    const k = callout.knee ? toDom(callout.knee) : null;
    const color = annot.color && annot.color.a > 0 ? rgbaString(annot.color) : "#e44234";
    const strokeWidth = (annot.border?.width ?? 1) * scale;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "lector-annot-overlay lector-annot-callout-leader");
    svg.setAttribute("width", String(vp.width));
    svg.setAttribute("height", String(vp.height));
    svg.style.cssText = "position:absolute;left:0;top:0;pointer-events:none;overflow:visible;";
    svg.dataset["annotId"] = annot.id;
    const points = k ? `${a.x},${a.y} ${k.x},${k.y} ${e.x},${e.y}` : `${a.x},${a.y} ${e.x},${e.y}`;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    line.setAttribute("points", points);
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", String(strokeWidth));
    line.setAttribute("stroke-linecap", "round");
    line.setAttribute("stroke-linejoin", "round");
    svg.appendChild(line);
    const ending = callout.lineEnding ?? "OpenArrow";
    if (ending !== "None") {
      const tail = k ?? a;
      const angle = Math.atan2(e.y - tail.y, e.x - tail.x);
      const headLen = Math.max(8, strokeWidth * 4);
      const a1x = e.x - headLen * Math.cos(angle - Math.PI / 6);
      const a1y = e.y - headLen * Math.sin(angle - Math.PI / 6);
      const a2x = e.x - headLen * Math.cos(angle + Math.PI / 6);
      const a2y = e.y - headLen * Math.sin(angle + Math.PI / 6);
      if (ending === "ClosedArrow") {
        const tri = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        tri.setAttribute("points", `${a1x},${a1y} ${e.x},${e.y} ${a2x},${a2y}`);
        tri.setAttribute("fill", color);
        tri.setAttribute("stroke", color);
        tri.setAttribute("stroke-width", String(strokeWidth));
        tri.setAttribute("stroke-linejoin", "round");
        svg.appendChild(tri);
      } else {
        const head = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        head.setAttribute("points", `${a1x},${a1y} ${e.x},${e.y} ${a2x},${a2y}`);
        head.setAttribute("fill", "none");
        head.setAttribute("stroke", color);
        head.setAttribute("stroke-width", String(strokeWidth));
        head.setAttribute("stroke-linecap", "round");
        head.setAttribute("stroke-linejoin", "round");
        svg.appendChild(head);
      }
    }
    state.overlay.appendChild(svg);
    const endHandle = document.createElement("div");
    endHandle.className = "lector-line-handle";
    endHandle.dataset["forAnnot"] = annot.id;
    endHandle.style.cssText = `position:absolute;width:10px;height:10px;border-radius:50%;background:white;border:1.5px solid #3b82f6;cursor:move;pointer-events:auto;transform:translate(-50%,-50%);display:none;z-index:10;`;
    endHandle.style.left = `${e.x}px`;
    endHandle.style.top = `${e.y}px`;
    let dragging = false;
    endHandle.addEventListener("pointerdown", (ev) => {
      ev.stopPropagation();
      ev.preventDefault();
      dragging = true;
      endHandle.setPointerCapture(ev.pointerId);
      this.#engine.plugins.events.emit("annotation:drag-start");
    });
    endHandle.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      ev.stopPropagation();
      const container = state.overlay.closest(".lector-canvas");
      if (!container) return;
      const cRect = container.getBoundingClientRect();
      const pagePos = this.#viewport.pagePositions.peek().find((p) => p.pageIndex === annot.pageIndex);
      if (!pagePos) return;
      const domPx = ev.clientX - cRect.left + container.scrollLeft - pagePos.x;
      const domPy = ev.clientY - cRect.top + container.scrollTop - pagePos.y;
      endHandle.style.left = `${domPx}px`;
      endHandle.style.top = `${domPy}px`;
      const newPoints = k ? `${a.x},${a.y} ${k.x},${k.y} ${domPx},${domPy}` : `${a.x},${a.y} ${domPx},${domPy}`;
      line.setAttribute("points", newPoints);
    });
    const finishDrag = () => {
      if (!dragging) return;
      dragging = false;
      const finalPdf = vp.cssPointToPdf(
        parseFloat(endHandle.style.left),
        parseFloat(endHandle.style.top)
      );
      if (this.#annotation) {
        const doc = this.#resolveDoc();
        if (doc) {
          void this.#annotation.update(doc.id, annot.id, {
            callout: {
              ...callout,
              endpoint: { x: finalPdf.x, y: finalPdf.y }
            }
          }).then(() => {
            this.#engine.plugins.events.emit("annotation:drag-end", annot.id);
          });
        }
      }
    };
    endHandle.addEventListener("pointerup", finishDrag);
    endHandle.addEventListener("pointercancel", finishDrag);
    state.overlay.appendChild(endHandle);
  }
  #renderRedactionAnnotation(state, annot, scale) {
    const pos = pdfRectToDOM(annot.rect, this.#vp(state, scale));
    const el = document.createElement("div");
    el.dataset["annotId"] = annot.id;
    el.className = "lector-annot-overlay lector-annot-redaction";
    el.style.background = "rgba(255, 0, 0, 0.3)";
    el.style.border = "2px dashed red";
    if (annot.redaction?.overlayText) {
      el.textContent = annot.redaction.overlayText;
    }
    positionElement(el, pos);
    this.#attachDragToMove(el, annot, state);
    state.overlay.appendChild(el);
    this.#attachRectResizeHandles(state, annot, scale);
  }
  #renderMeasurementAnnotation(state, annot, scale) {
    if (annot.tag === "measure-distance") {
      if (annot.line) {
        const lineAnnot = { ...annot, tag: "line" };
        this.#renderLineAnnotation(state, lineAnnot, scale);
      } else if (annot.ink && annot.ink.strokes.length > 0) {
        const stroke = annot.ink.strokes[0];
        if (stroke.length >= 2) {
          const lineAnnot = { ...annot, line: { start: stroke[0], end: stroke[stroke.length - 1] }, tag: "line" };
          this.#renderLineAnnotation(state, lineAnnot, scale);
        }
      }
    } else if (annot.ink && annot.ink.strokes.length > 0) {
      this.#renderInkAnnotation(state, annot, scale);
    }
    if (annot.measurement) {
      const pos = pdfRectToDOM(annot.rect, this.#vp(state, scale));
      const label = document.createElement("div");
      label.className = "lector-annot-measure-label";
      const valuePt = annot.measurement.value;
      const storedUnit = annot.measurement.unit ?? "pt";
      const type = annot.measurement.type;
      const precision = annot.measurement.precision ?? 2;
      const scaleSnapshot = annot.measurement.scale;
      const fallbackUnit = storedUnit === "pt" ? "cm" : storedUnit;
      const converted = type === "area" ? convertAreaWithScale(valuePt, scaleSnapshot, fallbackUnit) : convertLengthWithScale(valuePt, scaleSnapshot, fallbackUnit);
      const localizedNumber = this.#formatting ? this.#formatting.formatNumber(converted.value, {
        minimumFractionDigits: precision,
        maximumFractionDigits: precision
      }) : converted.value.toFixed(precision);
      const unitSuffix = type === "area" ? `${converted.unit}\xB2` : converted.unit;
      label.textContent = `${localizedNumber} ${unitSuffix}`;
      label.style.left = `${pos.x + pos.w / 2}px`;
      label.style.top = `${pos.y - 20}px`;
      state.overlay.appendChild(label);
    }
  }
  #renderStickyNote(state, annot, scale) {
    const pos = pdfRectToDOM(annot.rect, this.#vp(state, scale));
    const color = annot.color ? `rgb(${annot.color.r}, ${annot.color.g}, ${annot.color.b})` : "rgb(255, 205, 69)";
    const el = document.createElement("div");
    el.className = "lector-annot-overlay lector-annot-note";
    el.dataset["annotId"] = annot.id;
    el.style.left = `${pos.x}px`;
    el.style.top = `${pos.y}px`;
    el.style.width = "24px";
    el.style.height = "24px";
    el.style.background = color;
    if (annot.contents) el.title = annot.contents;
    const iconMap = {
      "Comment": "message-square",
      "Note": "sticky-note",
      "Help": "help-circle",
      "Insert": "plus",
      "Check": "check",
      "Cross": "x",
      "Star": "star",
      "Circle": "circle",
      "Key": "key"
    };
    const iconName = iconMap[annot.noteIcon ?? "Comment"] ?? "message-square";
    const svg = resolveIcon(iconName);
    if (svg) {
      const cr = annot.color?.r ?? 255;
      const cg = annot.color?.g ?? 205;
      const cb = annot.color?.b ?? 69;
      const luminance = (0.299 * cr + 0.587 * cg + 0.114 * cb) / 255;
      const iconColor = luminance > 0.5 ? "rgba(0,0,0,0.6)" : "rgba(255,255,255,0.85)";
      const wrap = document.createElement("span");
      wrap.style.cssText = `display:flex;align-items:center;justify-content:center;width:16px;height:16px;color:${iconColor}`;
      wrap.innerHTML = svg;
      const svgEl = wrap.querySelector("svg");
      if (svgEl) {
        svgEl.style.width = "16px";
        svgEl.style.height = "16px";
      }
      el.appendChild(wrap);
    }
    this.#attachDragToMove(el, annot, state);
    state.overlay.appendChild(el);
  }
  #renderLineAnnotation(state, annot, scale) {
    if (!annot.line) return;
    const vp = this.#vp(state, scale);
    const opaqueColor = annot.color && annot.color.a > 0 ? `rgb(${annot.color.r}, ${annot.color.g}, ${annot.color.b})` : "rgb(0, 0, 0)";
    const opacity = annot.color ? annot.color.a / 255 : 1;
    const borderWidth = (annot.border?.width ?? 2) * scale;
    const start = vp.pointToCss(annot.line.start.x, annot.line.start.y);
    const end = vp.pointToCss(annot.line.end.x, annot.line.end.y);
    const x1 = start.x;
    const y1 = start.y;
    const x2 = end.x;
    const y2 = end.y;
    const vbW = vp.width;
    const vbH = vp.height;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("lector-annot-overlay", "lector-annot-overlay--ink");
    svg.setAttribute("viewBox", `0 0 ${vbW} ${vbH}`);
    svg.style.position = "absolute";
    svg.style.inset = "0";
    svg.style.width = "100%";
    svg.style.height = "100%";
    svg.style.overflow = "visible";
    svg.dataset["annotId"] = annot.id;
    const hitLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    hitLine.setAttribute("x1", String(x1));
    hitLine.setAttribute("y1", String(y1));
    hitLine.setAttribute("x2", String(x2));
    hitLine.setAttribute("y2", String(y2));
    hitLine.setAttribute("stroke", "transparent");
    hitLine.setAttribute("stroke-width", String(Math.max(12, borderWidth * 3)));
    svg.appendChild(hitLine);
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    if (opacity < 1) g.setAttribute("opacity", String(opacity));
    const dashStyle = annot.border?.horizontalRadius ?? 0;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(x1));
    line.setAttribute("y1", String(y1));
    line.setAttribute("x2", String(x2));
    line.setAttribute("y2", String(y2));
    line.setAttribute("stroke", opaqueColor);
    line.setAttribute("stroke-width", String(borderWidth));
    line.setAttribute("stroke-linecap", "round");
    if (dashStyle === 1) line.setAttribute("stroke-dasharray", `${borderWidth * 3} ${borderWidth * 2}`);
    if (dashStyle === 2) line.setAttribute("stroke-dasharray", `${borderWidth} ${borderWidth * 2}`);
    g.appendChild(line);
    const arrowStyle = annot.tag ?? "";
    const hasEndArrow = arrowStyle === "arrow" || arrowStyle === "arrow-both";
    const hasStartArrow = arrowStyle === "arrow-start" || arrowStyle === "arrow-both";
    const headLen = Math.max(8, borderWidth * 4);
    if (hasEndArrow) {
      const angle = Math.atan2(y2 - y1, x2 - x1);
      const a1x = x2 - headLen * Math.cos(angle - Math.PI / 6);
      const a1y = y2 - headLen * Math.sin(angle - Math.PI / 6);
      const a2x = x2 - headLen * Math.cos(angle + Math.PI / 6);
      const a2y = y2 - headLen * Math.sin(angle + Math.PI / 6);
      const head = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      head.setAttribute("points", `${a1x},${a1y} ${x2},${y2} ${a2x},${a2y}`);
      head.setAttribute("fill", "none");
      head.setAttribute("stroke", opaqueColor);
      head.setAttribute("stroke-width", String(borderWidth));
      head.setAttribute("stroke-linecap", "round");
      head.setAttribute("stroke-linejoin", "round");
      g.appendChild(head);
    }
    if (hasStartArrow) {
      const angle = Math.atan2(y1 - y2, x1 - x2);
      const a1x = x1 - headLen * Math.cos(angle - Math.PI / 6);
      const a1y = y1 - headLen * Math.sin(angle - Math.PI / 6);
      const a2x = x1 - headLen * Math.cos(angle + Math.PI / 6);
      const a2y = y1 - headLen * Math.sin(angle + Math.PI / 6);
      const head = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      head.setAttribute("points", `${a1x},${a1y} ${x1},${y1} ${a2x},${a2y}`);
      head.setAttribute("fill", "none");
      head.setAttribute("stroke", opaqueColor);
      head.setAttribute("stroke-width", String(borderWidth));
      head.setAttribute("stroke-linecap", "round");
      head.setAttribute("stroke-linejoin", "round");
      g.appendChild(head);
    }
    svg.appendChild(g);
    const pad = 6;
    const sMinX = Math.min(x1, x2) - pad;
    const sMinY = Math.min(y1, y2) - pad;
    const sW = Math.abs(x2 - x1) + pad * 2;
    const sH = Math.abs(y2 - y1) + pad * 2;
    const selRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    selRect.setAttribute("x", String(sMinX));
    selRect.setAttribute("y", String(sMinY));
    selRect.setAttribute("width", String(sW));
    selRect.setAttribute("height", String(sH));
    selRect.setAttribute("fill", "transparent");
    selRect.setAttribute("stroke", "#3b82f6");
    selRect.setAttribute("stroke-width", "2");
    selRect.setAttribute("stroke-dasharray", "4 3");
    selRect.setAttribute("rx", "3");
    selRect.setAttribute("cursor", "grab");
    selRect.classList.add("lector-ink-select-rect");
    selRect.style.display = "none";
    svg.appendChild(selRect);
    const arrowHeads = svg.querySelectorAll("polyline");
    const updateArrowHeads = (lx1, ly1, lx2, ly2) => {
      let idx = 0;
      if (hasEndArrow && idx < arrowHeads.length) {
        const a = Math.atan2(ly2 - ly1, lx2 - lx1);
        const ax1 = lx2 - headLen * Math.cos(a - Math.PI / 6);
        const ay1 = ly2 - headLen * Math.sin(a - Math.PI / 6);
        const ax2 = lx2 - headLen * Math.cos(a + Math.PI / 6);
        const ay2 = ly2 - headLen * Math.sin(a + Math.PI / 6);
        arrowHeads[idx].setAttribute("points", `${ax1},${ay1} ${lx2},${ly2} ${ax2},${ay2}`);
        idx++;
      }
      if (hasStartArrow && idx < arrowHeads.length) {
        const a = Math.atan2(ly1 - ly2, lx1 - lx2);
        const ax1 = lx1 - headLen * Math.cos(a - Math.PI / 6);
        const ay1 = ly1 - headLen * Math.sin(a - Math.PI / 6);
        const ax2 = lx1 - headLen * Math.cos(a + Math.PI / 6);
        const ay2 = ly1 - headLen * Math.sin(a + Math.PI / 6);
        arrowHeads[idx].setAttribute("points", `${ax1},${ay1} ${lx1},${ly1} ${ax2},${ay2}`);
        idx++;
      }
    };
    for (const [hx, hy, isStart] of [[x1, y1, true], [x2, y2, false]]) {
      const handle = document.createElement("div");
      handle.className = "lector-line-handle";
      handle.dataset["forAnnot"] = annot.id;
      handle.style.cssText = `position:absolute;width:10px;height:10px;border-radius:50%;background:white;border:1.5px solid #3b82f6;cursor:move;pointer-events:auto;transform:translate(-50%,-50%);display:none;z-index:10;`;
      handle.style.left = `${hx}px`;
      handle.style.top = `${hy}px`;
      let dragging = false;
      handle.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        e.preventDefault();
        dragging = true;
        handle.setPointerCapture(e.pointerId);
        this.#engine.plugins.events.emit("annotation:drag-start");
      });
      handle.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        e.stopPropagation();
        const container = state.overlay.closest(".lector-canvas");
        if (!container) return;
        const cRect = container.getBoundingClientRect();
        const pagePos = this.#viewport.pagePositions.peek().find((p) => p.pageIndex === annot.pageIndex);
        if (!pagePos) return;
        const domPx = e.clientX - cRect.left + container.scrollLeft - pagePos.x;
        const domPy = e.clientY - cRect.top + container.scrollTop - pagePos.y;
        handle.style.left = `${domPx}px`;
        handle.style.top = `${domPy}px`;
        if (isStart) {
          line.setAttribute("x1", String(domPx));
          line.setAttribute("y1", String(domPy));
          hitLine.setAttribute("x1", String(domPx));
          hitLine.setAttribute("y1", String(domPy));
        } else {
          line.setAttribute("x2", String(domPx));
          line.setAttribute("y2", String(domPy));
          hitLine.setAttribute("x2", String(domPx));
          hitLine.setAttribute("y2", String(domPy));
        }
        const curX1 = parseFloat(line.getAttribute("x1"));
        const curY1 = parseFloat(line.getAttribute("y1"));
        const curX2 = parseFloat(line.getAttribute("x2"));
        const curY2 = parseFloat(line.getAttribute("y2"));
        updateArrowHeads(curX1, curY1, curX2, curY2);
      });
      const finishDrag = (e) => {
        if (!dragging) return;
        dragging = false;
        e.stopPropagation();
        const domLeft = parseFloat(handle.style.left);
        const domTop = parseFloat(handle.style.top);
        const { x: pdfNewX, y: pdfNewY } = vp.cssPointToPdf(domLeft, domTop);
        if (this.#annotation) {
          const doc = this.#resolveDoc();
          if (doc && annot.ink && annot.ink.strokes.length > 0) {
            const stroke = [...annot.ink.strokes[0]];
            if (isStart) stroke[0] = { x: pdfNewX, y: pdfNewY };
            else stroke[stroke.length - 1] = { x: pdfNewX, y: pdfNewY };
            const startPt = stroke[0];
            const endPt = stroke[stroke.length - 1];
            const pad2 = annot.border?.width ?? 2;
            const minX = Math.min(startPt.x, endPt.x);
            const maxX = Math.max(startPt.x, endPt.x);
            const minY = Math.min(startPt.y, endPt.y);
            const maxY = Math.max(startPt.y, endPt.y);
            void this.#annotation.update(doc.id, annot.id, {
              ink: { strokes: [stroke] },
              line: { start: { x: startPt.x, y: startPt.y }, end: { x: endPt.x, y: endPt.y } },
              rect: { left: minX - pad2, bottom: minY - pad2, right: maxX + pad2, top: maxY + pad2 }
            }).then(() => {
              this.#engine.plugins.events.emit("annotation:drag-end", annot.id);
            });
          }
        }
      };
      handle.addEventListener("pointerup", finishDrag);
      handle.addEventListener("pointercancel", finishDrag);
      state.overlay.appendChild(handle);
    }
    this.#attachDragToMove(svg, annot, state);
    if (annot.contents) svg.setAttribute("title", annot.contents);
    state.overlay.appendChild(svg);
  }
  #updateAnnotationSelection(selectedId) {
    const multiIds = new Set(this.#annotation?.selectedAnnotations.peek() ?? []);
    if (selectedId) multiIds.add(selectedId);
    for (const [, state] of this.#pages) {
      for (const el of state.overlay.querySelectorAll(".lector-annot-overlay, .lector-annot-note")) {
        const annotEl = el;
        const id = annotEl.dataset["annotId"];
        const isPrimary = id === selectedId;
        const isInMulti = id !== void 0 && multiIds.has(id);
        annotEl.classList.toggle("lector-annot-overlay--selected", isPrimary);
        annotEl.classList.toggle("lector-annot-overlay--multi-selected", isInMulti && !isPrimary);
        const selRect = annotEl.querySelector(".lector-ink-select-rect");
        if (selRect) selRect.style.display = isInMulti ? "" : "none";
      }
      for (const handle of state.overlay.querySelectorAll(".lector-line-handle")) {
        const h = handle;
        h.style.display = h.dataset["forAnnot"] === selectedId ? "" : "none";
      }
      for (const handle of state.overlay.querySelectorAll(".lector-resize-handle")) {
        const h = handle;
        h.style.display = h.dataset["forAnnot"] === selectedId ? "" : "none";
      }
      for (const vh of state.overlay.querySelectorAll(".lector-vertex-handle")) {
        vh.remove();
      }
    }
  }
  /**
   * Enter vertex editing mode for an ink/polygon annotation.
   * Shows a draggable handle on each vertex point. Drag moves that vertex.
   * Press Escape or click outside the annotation to exit.
   */
  #enterVertexEditMode(state, annot, scale) {
    if (!annot.ink) return;
    const vp = this.#vp(state, scale);
    for (const vh of state.overlay.querySelectorAll(`.lector-vertex-handle[data-for-annot="${annot.id}"]`)) {
      vh.remove();
    }
    for (const rh of state.overlay.querySelectorAll(`.lector-resize-handle[data-for-annot="${annot.id}"]`)) {
      rh.style.display = "none";
    }
    const annotSvg = state.overlay.querySelector(`.lector-annot-overlay--ink[data-annot-id="${annot.id}"]`);
    if (annotSvg) {
      const selRect = annotSvg.querySelector(".lector-ink-select-rect");
      if (selRect) selRect.style.display = "none";
    }
    const handles = [];
    for (let si = 0; si < annot.ink.strokes.length; si++) {
      const stroke = annot.ink.strokes[si];
      for (let pi = 0; pi < stroke.length; pi++) {
        const p = stroke[pi];
        const { x: domX, y: domY } = vp.pointToCss(p.x, p.y);
        const handle = document.createElement("div");
        handle.className = "lector-vertex-handle";
        handle.dataset["forAnnot"] = annot.id;
        handle.dataset["strokeIdx"] = String(si);
        handle.dataset["pointIdx"] = String(pi);
        handle.style.cssText = `position:absolute;width:8px;height:8px;border-radius:50%;background:#3b82f6;border:1.5px solid white;cursor:move;pointer-events:auto;transform:translate(-50%,-50%);z-index:11;padding:8px;background-clip:content-box;box-sizing:content-box;`;
        handle.style.left = `${domX}px`;
        handle.style.top = `${domY}px`;
        let dragging = false;
        handle.addEventListener("pointerdown", (e) => {
          e.stopPropagation();
          e.preventDefault();
          dragging = true;
          handle.setPointerCapture(e.pointerId);
          this.#engine.plugins.events.emit("annotation:drag-start");
        });
        handle.addEventListener("pointermove", (e) => {
          if (!dragging) return;
          e.stopPropagation();
          const container2 = state.overlay.closest(".lector-canvas");
          if (!container2) return;
          const cRect = container2.getBoundingClientRect();
          const pagePos = this.#viewport.pagePositions.peek().find((pp) => pp.pageIndex === annot.pageIndex);
          if (!pagePos) return;
          const newDomX = e.clientX - cRect.left + container2.scrollLeft - pagePos.x;
          const newDomY = e.clientY - cRect.top + container2.scrollTop - pagePos.y;
          handle.style.left = `${newDomX}px`;
          handle.style.top = `${newDomY}px`;
          const annotSvgEl = state.overlay.querySelector(`.lector-annot-overlay--ink[data-annot-id="${annot.id}"]`);
          if (annotSvgEl) {
            const gPolylines = annotSvgEl.querySelector("g")?.querySelectorAll("polyline");
            const hitAreas = Array.from(annotSvgEl.children).filter(
              (c) => c.tagName === "polyline" && c.getAttribute("stroke") === "transparent"
            );
            const stk = annot.ink.strokes[si];
            const pts = stk.map((sp, idx) => {
              if (idx === pi) return `${newDomX},${newDomY}`;
              const d = vp.pointToCss(sp.x, sp.y);
              return `${d.x},${d.y}`;
            }).join(" ");
            if (gPolylines && gPolylines[si]) gPolylines[si].setAttribute("points", pts);
            if (hitAreas[si]) hitAreas[si].setAttribute("points", pts);
          }
        });
        const finishVertexDrag = () => {
          if (!dragging) return;
          dragging = false;
          const finalDomX = parseFloat(handle.style.left);
          const finalDomY = parseFloat(handle.style.top);
          const { x: newPdfX, y: newPdfY } = vp.cssPointToPdf(finalDomX, finalDomY);
          if (this.#annotation) {
            const doc = this.#resolveDoc();
            if (doc && annot.ink) {
              const newStrokes = annot.ink.strokes.map(
                (s, sIdx) => s.map((pt, pIdx) => {
                  if (sIdx !== si || pIdx !== pi) return pt;
                  return pt.pressure !== void 0 ? { x: newPdfX, y: newPdfY, pressure: pt.pressure } : { x: newPdfX, y: newPdfY };
                })
              );
              let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
              for (const s of newStrokes) {
                for (const pt of s) {
                  if (pt.x < minX) minX = pt.x;
                  if (pt.y < minY) minY = pt.y;
                  if (pt.x > maxX) maxX = pt.x;
                  if (pt.y > maxY) maxY = pt.y;
                }
              }
              void this.#annotation.update(doc.id, annot.id, {
                ink: { strokes: newStrokes },
                rect: { left: minX, bottom: minY, right: maxX, top: maxY }
              }).then(() => {
                this.#engine.plugins.events.emit("annotation:drag-end", annot.id);
              });
            }
          }
        };
        handle.addEventListener("pointerup", finishVertexDrag);
        handle.addEventListener("pointercancel", finishVertexDrag);
        state.overlay.appendChild(handle);
        handles.push(handle);
      }
    }
    const onKeyDown = (e) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        exitVertexMode();
      }
    };
    const container = state.overlay.closest(".lector-canvas");
    container?.addEventListener("keydown", onKeyDown);
    const exitVertexMode = () => {
      for (const h of handles) h.remove();
      handles.length = 0;
      container?.removeEventListener("keydown", onKeyDown);
      if (this.#annotation?.selectedAnnotation.peek() === annot.id) {
        for (const rh of state.overlay.querySelectorAll(`.lector-resize-handle[data-for-annot="${annot.id}"]`)) {
          rh.style.display = "";
        }
        if (annotSvg) {
          const selRect = annotSvg.querySelector(".lector-ink-select-rect");
          if (selRect) selRect.style.display = "";
        }
      }
    };
  }
  // ── Form fields ──
  async #loadFormFields(doc, pageIndex, state) {
    try {
      await this.#form.loadPage(doc.id, pageIndex);
      const rawFields = await this.#engine.workerProxy.getFormFields(doc.id, pageIndex);
      if (this.#resolveDoc()?.id !== doc.id) return;
      const scale = this.#viewport.scale.peek();
      this.#renderFormFieldsRaw(doc.id, pageIndex, state, rawFields, scale);
    } catch {
    }
  }
  /**
   * Handle a click on a checkbox or radio button widget.
   * Simulates a pdfium mouse click, re-renders the page bitmap,
   * and rebuilds the form overlays to reflect the new state.
   */
  async #handleCheckableClick(docId, pageIndex, state, pageX, pageY) {
    try {
      await this.#form.clickWidget(docId, pageIndex, pageX, pageY);
      const pageEl = state.overlay.parentElement;
      const canvas = pageEl?.querySelector("canvas");
      if (canvas) {
        const bmp = await this.#engine.workerProxy.renderPage(
          docId,
          pageIndex,
          canvas.width,
          canvas.height
        );
        const ctx2d = canvas.getContext("2d");
        if (ctx2d && bmp) ctx2d.drawImage(bmp, 0, 0, canvas.width, canvas.height);
      }
    } catch {
    }
  }
  #renderFormFieldsRaw(docId, pageIndex, state, fields, scale) {
    const annotations = this.#annotation ? this.#annotation.getForPage(docId, pageIndex) : [];
    const annotByIndex = /* @__PURE__ */ new Map();
    for (const a of annotations) {
      if (a.data.subtype === FpdfAnnotSubtype2.WIDGET && a.data.widget) {
        annotByIndex.set(a.data.widget.annotIndex, a);
      }
    }
    for (const field of fields) {
      const widgetAnnot = annotByIndex.get(field.annotIndex);
      if (!widgetAnnot) continue;
      const pos = pdfRectToDOM(widgetAnnot.data.rect, this.#vp(state, scale));
      const wrapper = document.createElement("div");
      wrapper.className = "lector-form-field";
      positionElement(wrapper, pos);
      switch (field.fieldType) {
        case 2:
        // CheckBox
        case 3: {
          const hitArea = document.createElement("div");
          hitArea.className = "lector-form-field__check-hit";
          if (field.fieldName) hitArea.setAttribute("role", field.fieldType === 2 ? "checkbox" : "radio");
          if (field.fieldName) hitArea.setAttribute("aria-label", field.exportValue || field.fieldName);
          hitArea.setAttribute("aria-checked", String(field.isChecked ?? false));
          hitArea.tabIndex = 0;
          if (!this.#form.readOnly.peek()) {
            const rect = widgetAnnot.data.rect;
            const centerX = (rect.left + rect.right) / 2;
            const centerY = (rect.top + rect.bottom) / 2;
            hitArea.addEventListener("pointerup", (e) => {
              e.stopPropagation();
              void this.#handleCheckableClick(docId, pageIndex, state, centerX, centerY);
            });
            hitArea.addEventListener("keydown", (e) => {
              if (e.key === " " || e.key === "Enter") {
                e.preventDefault();
                void this.#handleCheckableClick(docId, pageIndex, state, centerX, centerY);
              }
            });
          }
          wrapper.appendChild(hitArea);
          break;
        }
        case 4:
        // ComboBox
        case 5: {
          const select = document.createElement("select");
          select.className = "lector-form-field__select";
          if (field.fieldName) select.setAttribute("aria-label", field.fieldName);
          const isFieldReadOnly = (field.fieldFlags & 1) !== 0;
          select.disabled = this.#form.readOnly.peek() || isFieldReadOnly;
          if (field.options) {
            for (const opt of field.options) {
              const optEl = document.createElement("option");
              optEl.value = opt.label;
              optEl.textContent = opt.label;
              optEl.selected = opt.selected;
              optEl.dataset.optionIndex = String(opt.index);
              select.appendChild(optEl);
            }
          }
          select.value = field.fieldValue;
          const capturedAnnotIdx = field.annotIndex;
          select.addEventListener("change", () => {
            const selectedOption = select.selectedOptions[0];
            const optIdx = selectedOption?.dataset.optionIndex;
            if (optIdx !== void 0) {
              void this.#engine.workerProxy.setComboBoxByIndex(
                docId,
                pageIndex,
                capturedAnnotIdx,
                parseInt(optIdx, 10)
              );
            }
          });
          wrapper.appendChild(select);
          break;
        }
        case 6: {
          const isFieldReadOnly = (field.fieldFlags & 1) !== 0;
          const isMultiline = (field.fieldFlags & 1 << 12) !== 0;
          const isPassword = (field.fieldFlags & 1 << 13) !== 0;
          const el = isMultiline ? document.createElement("textarea") : document.createElement("input");
          if (!isMultiline) {
            el.type = isPassword ? "password" : "text";
          }
          el.autocomplete = "off";
          el.className = "lector-form-field__input";
          if (field.fieldName) el.setAttribute("aria-label", field.fieldName);
          el.value = field.fieldValue;
          if (isMultiline) el.textContent = field.fieldValue;
          el.readOnly = this.#form.readOnly.peek() || isFieldReadOnly;
          el.style.fontSize = `${Math.max(8, 12 * scale)}px`;
          el.addEventListener("focus", () => {
            this.#form.focusField(field.fieldName);
          });
          el.addEventListener("blur", () => {
            this.#form.focusField(null);
            void this.#form.setFieldValue(docId, pageIndex, field.fieldName, el.value);
          });
          if (!isMultiline) {
            el.addEventListener("keydown", ((e) => {
              if (e.key === "Enter") el.blur();
            }));
          }
          wrapper.appendChild(el);
          break;
        }
        case 1: {
          const btn = document.createElement("button");
          btn.className = "lector-form-field__button";
          if (field.fieldName) btn.setAttribute("aria-label", field.fieldValue || field.fieldName);
          btn.disabled = this.#form.readOnly.peek();
          btn.addEventListener("pointerup", (e) => {
            e.stopPropagation();
            const rect = widgetAnnot.data.rect;
            const cx = (rect.left + rect.right) / 2;
            const cy = (rect.top + rect.bottom) / 2;
            void (async () => {
              await this.#form.clickWidget(docId, pageIndex, cx, cy);
              this.#engine.plugins.events.emit("form:button-click", {
                docId,
                pageIndex,
                fieldName: field.fieldName,
                fieldValue: field.fieldValue
              });
              const name = field.fieldName.toLowerCase();
              if (name.includes("print")) {
                this.#engine.plugins.events.emit("ui:document-print");
              } else if (name.includes("reset") || name.includes("clear")) {
                for (const el of state.overlay.querySelectorAll(".lector-form-field")) {
                  el.remove();
                }
                const rawFields = await this.#engine.workerProxy.getFormFields(docId, pageIndex);
                const s = this.#viewport.scale.peek();
                this.#renderFormFieldsRaw(docId, pageIndex, state, rawFields, s);
              }
              const pageEl = state.overlay.parentElement;
              const canvas = pageEl?.querySelector("canvas");
              if (canvas) {
                const bmp = await this.#engine.workerProxy.renderPage(
                  docId,
                  pageIndex,
                  canvas.width,
                  canvas.height
                );
                const ctx2d = canvas.getContext("2d");
                if (ctx2d && bmp) ctx2d.drawImage(bmp, 0, 0, canvas.width, canvas.height);
              }
            })();
          });
          wrapper.appendChild(btn);
          break;
        }
        case 7: {
          const sigEl = document.createElement("div");
          sigEl.className = "lector-form-field__signature";
          sigEl.style.fontSize = `${Math.max(8, 10 * scale)}px`;
          if (field.fieldValue) {
            sigEl.textContent = field.fieldValue;
            sigEl.classList.add("lector-form-field__signature--signed");
          } else {
            sigEl.textContent = this.#t("form.clickToSign");
            sigEl.style.cursor = "pointer";
            sigEl.addEventListener("click", () => {
              this.#engine.plugins.events.emit("signature:field-click", {
                docId,
                pageIndex,
                fieldName: field.fieldName,
                rect: widgetAnnot ? widgetAnnot.data.rect : null
              });
            });
          }
          wrapper.appendChild(sigEl);
          break;
        }
        default:
          break;
      }
      state.overlay.appendChild(wrapper);
    }
  }
  // ── Scale change: rebuild all overlays ──
  /**
   * Called when the scale changes. Clears and reloads all overlays
   * for currently visible pages.
   */
  // ── Drag-to-move for selected annotations ──
  /**
   * Attach drag-to-move behavior to an annotation element.
   * Only activates when the annotation is already selected.
   * Works for all annotation types (rect, ink, line, sticky note).
   */
  #attachDragToMove(el, annot, state) {
    const htmlEl = el;
    htmlEl.tabIndex = 0;
    htmlEl.setAttribute("role", "button");
    const label = annot.contents || annot.author || annot.tag || this.#t("annotation.defaultLabel");
    htmlEl.setAttribute("aria-label", label);
    htmlEl.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === " ") && this.#annotation) {
        e.preventDefault();
        this.#annotation.selectAnnotation(annot.id);
      }
    });
    let dragging = false;
    let startX = 0;
    let startY = 0;
    let offsetX = 0;
    let offsetY = 0;
    const DRAG_THRESHOLD = 3;
    htmlEl.addEventListener("pointerdown", (e) => {
      if (!this.#annotation) return;
      const targetCl = e.target.classList;
      if (targetCl?.contains("lector-line-handle") || targetCl?.contains("lector-resize-handle") || targetCl?.contains("lector-vertex-handle")) return;
      const multiSet = this.#annotation.selectedAnnotations.peek();
      const wasInMulti = multiSet.includes(annot.id);
      const wasSelected = this.#annotation.selectedAnnotation.peek() === annot.id;
      e.stopPropagation();
      e.preventDefault();
      startX = e.clientX;
      startY = e.clientY;
      offsetX = 0;
      offsetY = 0;
      dragging = false;
      if (e.shiftKey) {
        this.#annotation.toggleAnnotationSelection(annot.id);
        htmlEl.closest(".lector-canvas")?.focus({ preventScroll: true });
        return;
      }
      if (wasInMulti && multiSet.length > 1) {
        this.#annotation.selectAnnotation(annot.id);
        for (const id of multiSet) {
          if (id !== annot.id) {
            this.#annotation.toggleAnnotationSelection(id);
          }
        }
      } else if (!wasSelected) {
        this.#annotation.selectAnnotation(annot.id);
        htmlEl.closest(".lector-canvas")?.focus({ preventScroll: true });
      }
      const isSvg = htmlEl.tagName === "svg" || htmlEl.tagName === "SVG";
      const captureEl = isSvg ? e.target : htmlEl;
      captureEl.setPointerCapture(e.pointerId);
      const onMove = (me) => {
        const dx = me.clientX - startX;
        const dy = me.clientY - startY;
        if (!dragging && Math.abs(dx) + Math.abs(dy) > DRAG_THRESHOLD) {
          dragging = true;
          htmlEl.style.cursor = "grabbing";
          htmlEl.classList.add("lector-annot-note--dragging");
          this.#engine.plugins.events.emit("annotation:drag-start");
        }
        if (!dragging) return;
        offsetX = dx;
        offsetY = dy;
        if (isSvg) {
          for (const child of htmlEl.children) {
            child.style.transform = `translate(${dx}px, ${dy}px)`;
          }
        } else {
          htmlEl.style.transform = `translate(${dx}px, ${dy}px)`;
        }
        const parent = htmlEl.parentElement;
        if (parent) {
          for (const h of parent.querySelectorAll(`.lector-line-handle[data-for-annot="${annot.id}"], .lector-resize-handle[data-for-annot="${annot.id}"]`)) {
            h.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
          }
        }
      };
      const onUp = () => {
        captureEl.removeEventListener("pointermove", onMove);
        captureEl.removeEventListener("pointerup", onUp);
        captureEl.removeEventListener("pointercancel", onUp);
        htmlEl.style.cursor = "";
        htmlEl.style.transform = "";
        htmlEl.classList.remove("lector-annot-note--dragging");
        if (isSvg) {
          for (const child of htmlEl.children) {
            child.style.transform = "";
          }
        }
        const parent = htmlEl.parentElement;
        if (parent) {
          for (const h of parent.querySelectorAll(`.lector-line-handle[data-for-annot="${annot.id}"], .lector-resize-handle[data-for-annot="${annot.id}"]`)) {
            h.style.transform = "translate(-50%, -50%)";
          }
        }
        if (!dragging) return;
        const scale = this.#viewport.scale.peek();
        const { x: pdfDx, y: pdfDy } = this.#vp(state, scale).cssDeltaToPdf(offsetX, offsetY);
        const doc = this.#resolveDoc();
        if (!doc || !this.#annotation) return;
        const r = annot.rect;
        const patch = {
          rect: {
            left: r.left + pdfDx,
            right: r.right + pdfDx,
            top: r.top + pdfDy,
            bottom: r.bottom + pdfDy
          },
          // Shift ink strokes — preserve per-point `pressure` so
          // dragging a pen-drawn stroke doesn't flatten its width.
          ...annot.ink ? {
            ink: {
              strokes: annot.ink.strokes.map(
                (s) => s.map((p) => p.pressure !== void 0 ? { x: p.x + pdfDx, y: p.y + pdfDy, pressure: p.pressure } : { x: p.x + pdfDx, y: p.y + pdfDy })
              )
            }
          } : {},
          // Shift line endpoints
          ...annot.line ? {
            line: {
              start: { x: annot.line.start.x + pdfDx, y: annot.line.start.y + pdfDy },
              end: { x: annot.line.end.x + pdfDx, y: annot.line.end.y + pdfDy }
            }
          } : {},
          // Shift markup quad points
          ...annot.markup ? {
            markup: {
              quadPoints: annot.markup.quadPoints.map((qp) => ({
                x1: qp.x1 + pdfDx,
                y1: qp.y1 + pdfDy,
                x2: qp.x2 + pdfDx,
                y2: qp.y2 + pdfDy,
                x3: qp.x3 + pdfDx,
                y3: qp.y3 + pdfDy,
                x4: qp.x4 + pdfDx,
                y4: qp.y4 + pdfDy
              }))
            }
          } : {},
          // Shift callout endpoint + knee in lockstep with the rect so
          // the leader line moves as part of the same drag. Without
          // this the leader would slingshot to keep pointing at the
          // original anchor while the text rect moves away.
          ...annot.callout ? {
            callout: {
              endpoint: {
                x: annot.callout.endpoint.x + pdfDx,
                y: annot.callout.endpoint.y + pdfDy
              },
              ...annot.callout.knee ? {
                knee: {
                  x: annot.callout.knee.x + pdfDx,
                  y: annot.callout.knee.y + pdfDy
                }
              } : {},
              ...annot.callout.lineEnding ? { lineEnding: annot.callout.lineEnding } : {}
            }
          } : {}
        };
        void this.#annotation.update(doc.id, annot.id, patch).then(() => {
          this.#engine.plugins.events.emit("annotation:drag-end", annot.id);
        });
      };
      captureEl.addEventListener("pointermove", onMove);
      captureEl.addEventListener("pointerup", onUp);
      captureEl.addEventListener("pointercancel", onUp);
    });
  }
  // ── Rect resize handles (8-point: NW, N, NE, E, SE, S, SW, W) ──
  /**
   * Attach 8-point resize handles to a rect-based annotation overlay.
   * Handles are appended to the page overlay (not the annotation element)
   * and identified via `data-for-annot`. Hidden by default — shown when
   * the annotation is the primary selection.
   *
   * Corner handles (NW/NE/SW/SE) resize freely on both axes.
   * Edge handles (N/S/E/W) resize only one axis.
   * Shift constrains corners to uniform scaling.
   */
  #attachRectResizeHandles(state, annot, scale) {
    const vp = this.#vp(state, scale);
    const pos = pdfRectToDOM(annot.rect, vp);
    const elLeft = pos.x;
    const elTop = pos.y;
    const elRight = pos.x + pos.w;
    const elBottom = pos.y + pos.h;
    const handleDefs = [
      { x: elLeft, y: elTop, cursor: "nwse-resize", edges: "lt" },
      { x: (elLeft + elRight) / 2, y: elTop, cursor: "ns-resize", edges: "t" },
      { x: elRight, y: elTop, cursor: "nesw-resize", edges: "rt" },
      { x: elRight, y: (elTop + elBottom) / 2, cursor: "ew-resize", edges: "r" },
      { x: elRight, y: elBottom, cursor: "nwse-resize", edges: "rb" },
      { x: (elLeft + elRight) / 2, y: elBottom, cursor: "ns-resize", edges: "b" },
      { x: elLeft, y: elBottom, cursor: "nesw-resize", edges: "lb" },
      { x: elLeft, y: (elTop + elBottom) / 2, cursor: "ew-resize", edges: "l" }
    ];
    const handleEls = [];
    const repositionAll = (l, t, r, b) => {
      const mx = (l + r) / 2;
      const my = (t + b) / 2;
      const positions = [
        { x: l, y: t },
        { x: mx, y: t },
        { x: r, y: t },
        { x: r, y: my },
        { x: r, y: b },
        { x: mx, y: b },
        { x: l, y: b },
        { x: l, y: my }
      ];
      for (let i = 0; i < 8; i++) {
        handleEls[i].style.left = `${positions[i].x}px`;
        handleEls[i].style.top = `${positions[i].y}px`;
      }
    };
    for (let hi = 0; hi < 8; hi++) {
      const def = handleDefs[hi];
      const handle = document.createElement("div");
      handle.className = "lector-resize-handle";
      handle.dataset["forAnnot"] = annot.id;
      handle.style.cssText = `position:absolute;width:8px;height:8px;background:white;border:1.5px solid #3b82f6;border-radius:1px;cursor:${def.cursor};pointer-events:auto;transform:translate(-50%,-50%);display:none;z-index:10;`;
      handle.style.left = `${def.x}px`;
      handle.style.top = `${def.y}px`;
      handleEls.push(handle);
      let dragging = false;
      let startL = 0, startT = 0, startR = 0, startB = 0;
      handle.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        e.preventDefault();
        dragging = true;
        handle.setPointerCapture(e.pointerId);
        this.#engine.plugins.events.emit("annotation:drag-start");
        const annotEl = state.overlay.querySelector(`[data-annot-id="${annot.id}"]`);
        if (annotEl) {
          startL = parseFloat(annotEl.style.left);
          startT = parseFloat(annotEl.style.top);
          startR = startL + parseFloat(annotEl.style.width);
          startB = startT + parseFloat(annotEl.style.height);
        } else {
          startL = elLeft;
          startT = elTop;
          startR = elRight;
          startB = elBottom;
        }
      });
      handle.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        e.stopPropagation();
        const container = state.overlay.closest(".lector-canvas");
        if (!container) return;
        const cRect = container.getBoundingClientRect();
        const pagePos = this.#viewport.pagePositions.peek().find((p) => p.pageIndex === annot.pageIndex);
        if (!pagePos) return;
        const domPx = e.clientX - cRect.left + container.scrollLeft - pagePos.x;
        const domPy = e.clientY - cRect.top + container.scrollTop - pagePos.y;
        let newL = startL, newT = startT, newR = startR, newB = startB;
        if (def.edges.includes("l")) newL = Math.min(domPx, newR - 4);
        if (def.edges.includes("r")) newR = Math.max(domPx, newL + 4);
        if (def.edges.includes("t")) newT = Math.min(domPy, newB - 4);
        if (def.edges.includes("b")) newB = Math.max(domPy, newT + 4);
        if (e.shiftKey && def.edges.length === 2) {
          const ow = startR - startL;
          const oh = startB - startT;
          if (ow > 0 && oh > 0) {
            const nw = newR - newL;
            const nh = newB - newT;
            const aspect = ow / oh;
            if (nw / nh > aspect) {
              const adjH = nw / aspect;
              if (def.edges.includes("t")) newT = newB - adjH;
              else newB = newT + adjH;
            } else {
              const adjW = nh * aspect;
              if (def.edges.includes("l")) newL = newR - adjW;
              else newR = newL + adjW;
            }
          }
        }
        const annotEl = state.overlay.querySelector(`[data-annot-id="${annot.id}"]`);
        if (annotEl) {
          annotEl.style.left = `${newL}px`;
          annotEl.style.top = `${newT}px`;
          annotEl.style.width = `${newR - newL}px`;
          annotEl.style.height = `${newB - newT}px`;
        }
        repositionAll(newL, newT, newR, newB);
      });
      const finishResize = () => {
        if (!dragging) return;
        dragging = false;
        const annotEl = state.overlay.querySelector(`[data-annot-id="${annot.id}"]`);
        if (!annotEl || !this.#annotation) return;
        const finalL = parseFloat(annotEl.style.left);
        const finalT = parseFloat(annotEl.style.top);
        const finalW = parseFloat(annotEl.style.width);
        const finalH = parseFloat(annotEl.style.height);
        const c1 = vp.cssPointToPdf(finalL, finalT);
        const c2 = vp.cssPointToPdf(finalL + finalW, finalT + finalH);
        const pdfLeft = Math.min(c1.x, c2.x);
        const pdfRight = Math.max(c1.x, c2.x);
        const pdfTop = Math.max(c1.y, c2.y);
        const pdfBottom = Math.min(c1.y, c2.y);
        const doc = this.#resolveDoc();
        if (doc) {
          void this.#annotation.update(doc.id, annot.id, {
            rect: { left: pdfLeft, bottom: pdfBottom, right: pdfRight, top: pdfTop }
          }).then(() => {
            this.#engine.plugins.events.emit("annotation:drag-end", annot.id);
          });
        }
      };
      handle.addEventListener("pointerup", finishResize);
      handle.addEventListener("pointercancel", finishResize);
      state.overlay.appendChild(handle);
    }
  }
  // ── Comparison overlays ──
  /**
   * Apply a comparison result to this pane. Pass `null` to clear.
   *
   * The same `ComparisonResult` is shared across both panes — each pane's
   * manager only renders the rects that belong to its own side
   * (`'A'` = left/deletions, `'B'` = right/insertions). The third
   * argument is the change index to highlight as "active" (for prev/next
   * navigation).
   */
  setComparison(side, pageDiffs, activeFlatIndex = -1) {
    if (!pageDiffs) {
      this.#comparison = null;
      this.#clearComparisonOverlays();
      return;
    }
    const flat = [];
    for (const diff of pageDiffs) {
      for (const change of diff.changes) {
        flat.push({ diff, change });
      }
    }
    this.#comparison = { side, pageDiffs, flat, activeFlatIndex };
    this.#renderComparisonOverlays();
  }
  /** Update only the active change index without rebuilding the geometry. */
  setComparisonActiveIndex(activeFlatIndex) {
    if (!this.#comparison) return;
    this.#comparison = { ...this.#comparison, activeFlatIndex };
    this.#renderComparisonOverlays();
  }
  /**
   * Look up the page index that contains a specific change for this
   * side. Returns -1 if the change has no rect on this side (e.g. an
   * insertion-only change has no `pageA` for the A pane).
   */
  comparisonPageForChange(flatIndex) {
    if (!this.#comparison) return -1;
    const item = this.#comparison.flat[flatIndex];
    if (!item) return -1;
    if (this.#comparison.side === "A") {
      return item.diff.pageA ?? -1;
    }
    return item.diff.pageB ?? -1;
  }
  /** The flat change list snapshot — used by the sidebar panel. */
  get comparisonFlatChanges() {
    return this.#comparison?.flat ?? [];
  }
  #clearComparisonOverlays() {
    for (const [, state] of this.#pages) {
      for (const el of state.overlay.querySelectorAll(".lector-compare-overlay")) {
        el.remove();
      }
    }
  }
  #renderComparisonOverlays() {
    this.#clearComparisonOverlays();
    if (!this.#comparison) return;
    const { side, pageDiffs, activeFlatIndex } = this.#comparison;
    const scale = this.#viewport.scale.peek();
    const byPage = /* @__PURE__ */ new Map();
    let flatIdx = 0;
    for (const diff of pageDiffs) {
      const pageOnSide = side === "A" ? diff.pageA : diff.pageB;
      for (const change of diff.changes) {
        const idx = flatIdx++;
        if (pageOnSide == null) continue;
        const list = byPage.get(pageOnSide) ?? [];
        list.push({ diff, change, flatIdx: idx });
        byPage.set(pageOnSide, list);
      }
    }
    for (const [pageIndex, items] of byPage) {
      const state = this.#pages.get(pageIndex);
      if (!state) continue;
      for (const { change, flatIdx: idx } of items) {
        const rect = side === "A" ? change.rectA : change.rectB;
        if (!rect) continue;
        const pos = pdfRectToDOM(rect, this.#vp(state, scale));
        const PAD = 2;
        const el = document.createElement("div");
        el.className = `lector-compare-overlay lector-compare-overlay--${change.type}`;
        if (idx === activeFlatIndex) {
          el.classList.add("lector-compare-overlay--active");
        }
        el.style.left = `${pos.x - PAD}px`;
        el.style.top = `${pos.y - PAD}px`;
        el.style.width = `${pos.w + PAD * 2}px`;
        el.style.height = `${pos.h + PAD * 2}px`;
        el.dataset["compareIdx"] = String(idx);
        state.overlay.appendChild(el);
      }
    }
  }
  rebuildOverlays() {
    for (const [, state] of this.#pages) {
      state.overlay.innerHTML = "";
      state.linksLoaded = false;
      state.annotsLoaded = false;
      state.formsLoaded = false;
    }
    const visible = this.#viewport.visiblePages.peek();
    void this.#loadOverlaysForPages(visible);
    if (this.#textLayer) {
      this.#renderTextSelection(this.#textLayer.selection.peek());
    }
    if (this.#search) {
      const result = this.#search.result.peek();
      const activeIdx = this.#search.activeMatchIndex.peek();
      this.#renderSearchHighlights(result, activeIdx);
    }
    if (this.#comparison) {
      this.#renderComparisonOverlays();
    }
  }
  // ── Cleanup ──
  destroy() {
    for (const u of this.#cleanups) u();
    this.#cleanups.length = 0;
    for (const [, state] of this.#pages) {
      state.overlay.remove();
    }
    this.#pages.clear();
  }
  [Symbol.dispose]() {
    this.destroy();
  }
};

// src/ui/lector-pane.ts
var LectorPane = class {
  #engine;
  #container;
  #cleanups = [];
  // ── Plugin handles ──
  #viewportCap;
  #document;
  #formatting;
  #viewport;
  #overlays;
  // ── DOM ──
  #canvas;
  #scrollArea;
  // ── Per-pane page state ──
  #pageElements = /* @__PURE__ */ new Map();
  #pageCanvases = /* @__PURE__ */ new Map();
  #renderedPages = /* @__PURE__ */ new Set();
  #destroyed = false;
  constructor(options) {
    this.#engine = options.engine;
    this.#container = options.container;
    const p = options.engine.plugins;
    this.#viewportCap = p.get("viewport");
    this.#document = p.get("document");
    this.#formatting = p.tryGet("formatting");
    this.#buildDOM();
    this.#viewport = this.#viewportCap.createViewport({
      docId: options.docId
    });
    this.#viewport.attach(this.#canvas);
    this.#overlays = new PageOverlayManager(
      this.#engine,
      this.#viewport,
      this.#formatting
    );
    options.engine.plugins.events.emit("viewport:container-attached", this.#canvas);
    this.#wireRenderLoop();
  }
  /** The pane's viewport instance. */
  get viewport() {
    return this.#viewport;
  }
  /** The pane's canvas element (the scrollable host). */
  get canvas() {
    return this.#canvas;
  }
  /** The pane's outer container element. */
  get container() {
    return this.#container;
  }
  /**
   * The pane's PageOverlayManager. Exposed so chrome (LectorViewer) can
   * push comparison overlays onto the pane in compare mode without
   * having to mirror the overlay state inside the pane itself.
   */
  get overlays() {
    return this.#overlays;
  }
  /**
   * Pin a document to this pane (or null to follow the active document).
   * Use this in split-view to show different docs in different panes.
   */
  setDocument(docId) {
    this.#viewport.setDocument(docId);
    this.#renderedPages.clear();
    for (const [, el] of this.#pageElements) {
      this.#overlays.detachPage(this.#pageIndexOf(el) ?? 0);
      el.remove();
    }
    this.#pageElements.clear();
    this.#pageCanvases.clear();
  }
  /** Force a re-render of currently visible pages. */
  rerenderVisible() {
    this.#renderedPages.clear();
    const vis = this.#viewport.visiblePages.peek();
    if (vis.length > 0) void this.#renderVisiblePages(vis);
  }
  /** Tear down the pane: viewport, overlays, DOM, listeners. */
  destroy() {
    if (this.#destroyed) return;
    this.#destroyed = true;
    for (const cleanup of this.#cleanups) cleanup();
    this.#cleanups.length = 0;
    this.#overlays[Symbol.dispose]?.();
    this.#viewport.destroy();
    for (const el of this.#pageElements.values()) el.remove();
    this.#pageElements.clear();
    this.#pageCanvases.clear();
    this.#renderedPages.clear();
    this.#canvas.remove();
  }
  [Symbol.dispose]() {
    this.destroy();
  }
  // ─── DOM construction ──────────────────────────────────
  #buildDOM() {
    this.#canvas = window.document.createElement("div");
    this.#canvas.className = "lector-canvas lector-pane__canvas";
    this.#canvas.tabIndex = 0;
    this.#scrollArea = window.document.createElement("div");
    this.#scrollArea.className = "lector-canvas__scroll-area";
    this.#canvas.appendChild(this.#scrollArea);
    this.#container.appendChild(this.#canvas);
  }
  // ─── Render loop ───────────────────────────────────────
  #wireRenderLoop() {
    this.#cleanups.push(effect2(() => {
      const positions = this.#viewport.pagePositions.value;
      const totalHeight = this.#viewport.totalHeight.value;
      this.#updatePages(positions, totalHeight);
    }));
    this.#cleanups.push(effect2(() => {
      const visible = this.#viewport.visiblePages.value;
      if (visible.length > 0) {
        void this.#renderVisiblePages(visible);
      }
    }));
  }
  #updatePages(positions, totalHeight) {
    this.#scrollArea.style.height = `${totalHeight}px`;
    const active = new Set(positions.map((p) => p.pageIndex));
    for (const [idx, el] of this.#pageElements) {
      if (!active.has(idx)) {
        this.#overlays.detachPage(idx);
        el.remove();
        this.#pageElements.delete(idx);
        this.#pageCanvases.delete(idx);
        this.#renderedPages.delete(idx);
      }
    }
    for (const pos of positions) {
      let pe = this.#pageElements.get(pos.pageIndex);
      let cv = this.#pageCanvases.get(pos.pageIndex);
      if (!pe) {
        pe = window.document.createElement("div");
        pe.className = "lector-page lector-page--loading";
        pe.dataset.pageIndex = String(pos.pageIndex);
        cv = window.document.createElement("canvas");
        cv.className = "lector-page__canvas";
        pe.appendChild(cv);
        this.#scrollArea.appendChild(pe);
        this.#pageElements.set(pos.pageIndex, pe);
        this.#pageCanvases.set(pos.pageIndex, cv);
        const docHandle = this.#resolveHandle();
        if (docHandle) {
          const ps = docHandle.pageSizes[pos.pageIndex];
          this.#overlays.attachPage(pos.pageIndex, pe, ps?.width ?? 0, ps?.height ?? 0);
        }
      }
      pe.style.left = `${pos.x}px`;
      pe.style.top = `${pos.y}px`;
      pe.style.width = `${pos.width}px`;
      pe.style.height = `${pos.height}px`;
      if (cv) {
        const dpr = window.devicePixelRatio || 1;
        let cw = Math.round(pos.width * dpr);
        let ch = Math.round(pos.height * dpr);
        cv.style.width = `${pos.width}px`;
        cv.style.height = `${pos.height}px`;
        const MAX_CANVAS_DIM = 16384;
        if (cw > MAX_CANVAS_DIM || ch > MAX_CANVAS_DIM) {
          const scale = Math.min(MAX_CANVAS_DIM / cw, MAX_CANVAS_DIM / ch);
          cw = Math.round(cw * scale);
          ch = Math.round(ch * scale);
        }
        if (cv.width !== cw || cv.height !== ch) {
          cv.width = cw;
          cv.height = ch;
          this.#renderedPages.delete(pos.pageIndex);
          pe.classList.add("lector-page--loading");
        }
      }
    }
  }
  async #renderVisiblePages(visible) {
    const docHandle = this.#resolveHandle();
    if (!docHandle) return;
    for (const idx of visible) {
      if (this.#renderedPages.has(idx)) continue;
      const cv = this.#pageCanvases.get(idx);
      if (!cv) continue;
      if (cv.width === 0 || cv.height === 0) continue;
      this.#renderedPages.add(idx);
      try {
        const rw = cv.width;
        const rh = cv.height;
        const bmp = await this.#engine.renderPage(
          docHandle.id,
          idx,
          rw,
          rh,
          { priority: RenderPriority.VISIBLE }
        );
        const stale = cv.width !== rw || cv.height !== rh;
        const ctx = cv.getContext("2d");
        if (ctx) ctx.drawImage(bmp, 0, 0, cv.width, cv.height);
        bmp.close();
        const pe = this.#pageElements.get(idx);
        if (pe) pe.classList.remove("lector-page--loading");
        if (stale) this.#renderedPages.delete(idx);
      } catch {
        this.#renderedPages.delete(idx);
      }
    }
  }
  /**
   * Resolve the document this pane currently shows. Either the pinned
   * doc id (if set) or the engine's active document.
   */
  #resolveHandle() {
    const pinnedId = this.#viewport.docId.peek();
    if (pinnedId !== null) {
      return this.#document.getHandle(pinnedId) ?? null;
    }
    return this.#document.activeDocument.peek();
  }
  #pageIndexOf(el) {
    for (const [idx, e] of this.#pageElements) {
      if (e === el) return idx;
    }
    return null;
  }
};

export {
  definePlugin,
  RenderPriority,
  DEFAULT_RENDER_OPTIONS,
  PageViewport,
  uuid,
  ANNOTATION_TOOL_DEFAULTS,
  TOOL_TO_SUBTYPE,
  isMarkupTool,
  isInkTool,
  isEraserTool,
  isShapeTool,
  isPolygonTool,
  isPlacementTool,
  isMeasurementTool,
  isStampTool,
  isRedactionTool,
  isToolOutputTool,
  isToolOutputAnnotation,
  isUserAnnotation,
  createDrawModeHandler,
  LineCap,
  BlendMode,
  NoteIcon,
  MeasurementUnit,
  measurementPlugin,
  getIcon,
  isInlineSvg,
  resolveIcon,
  PageOverlayManager,
  LectorPane
};
