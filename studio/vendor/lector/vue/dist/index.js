// src/context.ts
var LECTOR_KEY = /* @__PURE__ */ Symbol("lector-engine");

// src/signal-bridge.ts
import { shallowRef, onScopeDispose, readonly } from "vue";
function useSignalRef(sig) {
  const r = shallowRef(sig.peek());
  const unsub = sig.subscribe((v) => {
    r.value = v;
  });
  onScopeDispose(unsub);
  return readonly(r);
}

// src/plugins.ts
import {
  documentPlugin,
  renderPlugin,
  viewportPlugin,
  zoomPlugin,
  interactionPlugin,
  textLayerPlugin,
  searchPlugin,
  navigationPlugin,
  annotationPlugin,
  annotationPresetsPlugin,
  comparisonPlugin,
  formPlugin,
  historyPlugin,
  signaturePlugin,
  signatureValidationPlugin,
  signatureSigningPlugin,
  attachmentPlugin,
  pageOpsPlugin,
  measurementPlugin,
  redactionPlugin,
  layerPlugin,
  i18nPlugin,
  formattingPlugin,
  capturePlugin,
  documentManagerPlugin,
  performancePlugin,
  uiPlugin
} from "@truespar/lector-core";
var ALL_PLUGINS = [
  documentPlugin,
  renderPlugin,
  viewportPlugin,
  zoomPlugin,
  interactionPlugin,
  textLayerPlugin,
  searchPlugin,
  navigationPlugin,
  annotationPlugin,
  annotationPresetsPlugin,
  comparisonPlugin,
  formPlugin,
  historyPlugin,
  signaturePlugin,
  signatureValidationPlugin,
  signatureSigningPlugin,
  attachmentPlugin,
  pageOpsPlugin,
  measurementPlugin,
  redactionPlugin,
  layerPlugin,
  i18nPlugin,
  formattingPlugin,
  capturePlugin,
  documentManagerPlugin,
  performancePlugin,
  uiPlugin
];
var CORE_PLUGINS = [
  documentPlugin,
  renderPlugin,
  viewportPlugin,
  zoomPlugin,
  interactionPlugin
];
var READER_PLUGINS = [
  documentPlugin,
  renderPlugin,
  viewportPlugin,
  zoomPlugin,
  interactionPlugin,
  textLayerPlugin,
  searchPlugin,
  navigationPlugin,
  documentManagerPlugin,
  i18nPlugin,
  uiPlugin
];

// src/composables/use-lector.ts
import { inject } from "vue";
function useLector() {
  const engine = inject(LECTOR_KEY);
  if (!engine) {
    throw new Error(
      "useLector() must be used within a <LectorPdfViewer> or a component that provides LECTOR_KEY."
    );
  }
  return engine;
}

// src/composables/use-plugin.ts
function usePlugin(capability) {
  const engine = useLector();
  return engine.plugins.get(capability);
}
function useOptionalPlugin(capability) {
  const engine = useLector();
  return engine.plugins.tryGet(capability);
}

// src/composables/use-document.ts
function useDocument() {
  const doc = usePlugin("document");
  const activeDocument = useSignalRef(doc.activeDocument);
  return {
    activeDocument,
    open: doc.load,
    close: doc.close,
    getHandle: doc.getHandle,
    setActive: doc.setActive
  };
}

// src/composables/use-viewport.ts
function useViewport() {
  const vp = usePlugin("viewport");
  return {
    visiblePages: useSignalRef(vp.visiblePages),
    scale: useSignalRef(vp.scale),
    layoutMode: useSignalRef(vp.layoutMode),
    pagePositions: useSignalRef(vp.pagePositions),
    totalHeight: useSignalRef(vp.totalHeight),
    containerSize: useSignalRef(vp.containerSize),
    attach: vp.attach,
    detach: vp.detach,
    scrollToPage: vp.scrollToPage,
    setLayoutMode: vp.setLayoutMode,
    setScale: vp.setScale,
    setBufferSize: vp.setBufferSize
  };
}

// src/composables/use-zoom.ts
function useZoom() {
  const zoom = usePlugin("zoom");
  return {
    level: useSignalRef(zoom.level),
    fitMode: useSignalRef(zoom.fitMode),
    setLevel: zoom.setLevel,
    zoomIn: zoom.zoomIn,
    zoomOut: zoom.zoomOut,
    fitPage: zoom.fitPage,
    fitWidth: zoom.fitWidth,
    resetZoom: zoom.resetZoom
  };
}

// src/composables/use-search.ts
function useSearch() {
  const s = usePlugin("search");
  return {
    result: useSignalRef(s.result),
    activeMatchIndex: useSignalRef(s.activeMatchIndex),
    searching: useSignalRef(s.searching),
    progress: useSignalRef(s.progress),
    search: s.search,
    nextMatch: s.nextMatch,
    previousMatch: s.previousMatch,
    goToMatch: s.goToMatch,
    clear: s.clear
  };
}

// src/composables/use-navigation.ts
function useNavigation() {
  const nav = usePlugin("navigation");
  return {
    currentPage: useSignalRef(nav.currentPage),
    canGoBack: useSignalRef(nav.canGoBack),
    canGoForward: useSignalRef(nav.canGoForward),
    goToPage: nav.goToPage,
    goBack: nav.goBack,
    goForward: nav.goForward,
    getBookmarks: nav.getBookmarks,
    navigateToTarget: nav.navigateToTarget
  };
}

// src/composables/use-annotations.ts
function useAnnotations() {
  const a = usePlugin("annotation");
  return {
    selectedAnnotation: useSignalRef(a.selectedAnnotation),
    selectedAnnotations: useSignalRef(a.selectedAnnotations),
    activeTool: useSignalRef(a.activeTool),
    toolStyle: useSignalRef(a.toolStyle),
    lockMode: useSignalRef(a.lockMode),
    store: a.store,
    create: a.create,
    update: a.update,
    delete: a.delete,
    loadPage: a.loadPage,
    getForPage: a.getForPage,
    getForDocument: a.getForDocument,
    selectAnnotation: a.selectAnnotation,
    toggleAnnotationSelection: a.toggleAnnotationSelection,
    clearAnnotationSelection: a.clearAnnotationSelection,
    setActiveTool: a.setActiveTool,
    setToolStyle: a.setToolStyle,
    setLockMode: a.setLockMode,
    getDirty: a.getDirty,
    markSynced: a.markSynced,
    markAllSynced: a.markAllSynced,
    subscribe: a.subscribe,
    setCommentStatus: a.setCommentStatus,
    toggleResolved: a.toggleResolved,
    editComment: a.editComment,
    deleteComment: a.deleteComment,
    bringToFront: a.bringToFront,
    sendToBack: a.sendToBack,
    canEdit: a.canEdit,
    canDelete: a.canDelete
  };
}

// src/composables/use-form.ts
function useForm() {
  const f = usePlugin("form");
  return {
    readOnly: useSignalRef(f.readOnly),
    focusedField: useSignalRef(f.focusedField),
    store: f.store,
    loadPage: f.loadPage,
    getPageFields: f.getPageFields,
    getDocumentFields: f.getDocumentFields,
    getFieldValue: f.getFieldValue,
    isPageLoaded: f.isPageLoaded,
    setFieldValue: f.setFieldValue,
    populateFields: f.populateFields,
    extractFormData: f.extractFormData,
    clickWidget: f.clickWidget,
    setReadOnly: f.setReadOnly,
    focusField: f.focusField,
    hasDirty: f.hasDirty,
    markAllSynced: f.markAllSynced,
    subscribe: f.subscribe
  };
}

// src/composables/use-history.ts
function useHistory() {
  const h2 = usePlugin("history");
  return {
    canUndo: useSignalRef(h2.canUndo),
    canRedo: useSignalRef(h2.canRedo),
    undoLabel: useSignalRef(h2.undoLabel),
    redoLabel: useSignalRef(h2.redoLabel),
    undo: h2.undo,
    redo: h2.redo,
    clear: h2.clear
  };
}

// src/composables/use-text-selection.ts
function useTextSelection() {
  const tl = usePlugin("text-layer");
  return {
    selection: useSignalRef(tl.selection),
    setSelection: tl.setSelection,
    copySelection: tl.copySelection,
    getPageText: tl.getPageText
  };
}

// src/composables/use-signatures.ts
function useSignatures() {
  const sig = usePlugin("signature");
  return {
    getCount: sig.getCount,
    getInfo: sig.getInfo,
    getAllInfo: sig.getAllInfo
  };
}

// src/composables/use-i18n.ts
function useI18n() {
  const i18n = usePlugin("i18n");
  return {
    locale: useSignalRef(i18n.locale),
    t: i18n.t,
    setLocale: i18n.setLocale,
    addTranslations: i18n.addTranslations,
    hasLocale: i18n.hasLocale,
    getLocales: i18n.getLocales
  };
}

// src/components/LectorPdfViewer.ts
import {
  defineComponent,
  ref,
  onMounted,
  onBeforeUnmount,
  watch,
  h
} from "vue";
import {
  LectorEngine,
  LectorViewer
} from "@truespar/lector-core";
var LectorPdfViewer = defineComponent({
  name: "LectorPdfViewer",
  props: {
    /** Pre-created and initialized `LectorEngine`. */
    engine: { type: Object, default: void 0 },
    /** Options to create a new engine internally. */
    engineOptions: { type: Object, default: void 0 },
    /** Plugins to register. Defaults to all built-in plugins. */
    plugins: { type: Array, default: void 0 },
    /** PDF source: URL string, ArrayBuffer, or File. */
    src: { type: [String, Object], default: void 0 },
    /** Password for encrypted PDFs. */
    password: { type: String, default: void 0 },
    /** Theme mode. */
    theme: { type: String, default: void 0 },
    /** Whether the sidebar starts open. */
    sidebarOpen: { type: Boolean, default: void 0 },
    /** Initial sidebar panel. */
    initialPanel: { type: String, default: void 0 },
    /** Page layout mode. */
    layoutMode: { type: String, default: void 0 },
    /** Initial zoom level or fit mode. */
    initialZoom: { type: [Number, String], default: void 0 },
    /** Visible sidebar panels. */
    panels: { type: Array, default: void 0 },
    /** Allow opening local files from UI. */
    allowLocalOpen: { type: Boolean, default: false },
    /**
     * Show the document tab bar. Defaults to true; pass false when the host
     * app has its own document switcher (see `LectorViewerOptions`).
     * `default: undefined` rather than `true` on purpose - Vue casts an ABSENT
     * Boolean prop to false, which would silently turn the bar off for every
     * embedder that never mentions it.
     */
    documentTabs: { type: Boolean, default: void 0 },
    /**
     * Host-contributed toolbar buttons, rendered by lector so they match its
     * own metrics (see `LectorViewerOptions.toolbarExtras`).
     */
    toolbarExtras: {
      type: Array,
      default: void 0
    },
    /**
     * Partial UI schema override, merged over `DEFAULT_UI_SCHEMA` — the same
     * option `LectorViewer` takes. Lets an embedding app trim or reorder the
     * toolbar without giving up the drop-in component.
     */
    uiSchema: { type: Object, default: void 0 }
  },
  emits: {
    /** Fired once the engine is ready and viewer mounted. */
    ready: (_engine) => true,
    /** Fired when a document is loaded. */
    "document-loaded": (_handle) => true,
    /** Fired on initialization or loading errors. */
    error: (_err) => true
  },
  setup(props, { emit, expose }) {
    const containerEl = ref(null);
    let engine = null;
    let viewer = null;
    let ownsEngine = false;
    let disposed = false;
    let loadToken = 0;
    async function initEngine() {
      if (props.engine) {
        ownsEngine = false;
        engine = props.engine;
        return;
      }
      if (!props.engineOptions) return;
      const eng = new LectorEngine(props.engineOptions);
      const pluginList = props.plugins ?? ALL_PLUGINS;
      for (const plugin of pluginList) {
        eng.plugins.register(plugin);
      }
      try {
        await eng.init();
        ownsEngine = true;
        engine = eng;
      } catch (err) {
        void eng.destroy();
        emit("error", err instanceof Error ? err : new Error(String(err)));
      }
    }
    function createViewer() {
      const eng = engine;
      const el = containerEl.value;
      if (!eng || !el) return;
      viewer = new LectorViewer({
        container: el,
        engine: eng,
        theme: props.theme,
        sidebarOpen: props.sidebarOpen,
        initialPanel: props.initialPanel,
        layoutMode: props.layoutMode,
        initialZoom: props.initialZoom,
        panels: props.panels,
        allowLocalOpen: props.allowLocalOpen,
        documentTabs: props.documentTabs,
        toolbarExtras: props.toolbarExtras,
        uiSchema: props.uiSchema
      });
      emit("ready", eng);
    }
    async function loadDocument() {
      if (!viewer || props.src == null) return;
      const token = ++loadToken;
      try {
        let data;
        let filename;
        if (typeof props.src === "string") {
          filename = props.src.split("/").pop() ?? "document.pdf";
          const res = await fetch(props.src);
          if (!res.ok) throw new Error(`Failed to fetch PDF: ${res.status}`);
          data = await res.arrayBuffer();
        } else if (props.src instanceof File) {
          filename = props.src.name;
          data = await props.src.arrayBuffer();
        } else {
          filename = "document.pdf";
          data = props.src;
        }
        if (disposed || token !== loadToken || !viewer) return;
        await viewer.loadDocument(data, props.password, filename);
        if (token !== loadToken) return;
        const eng = engine;
        if (eng) {
          const doc = eng.plugins.get("document");
          const handle = doc.activeDocument.peek();
          if (handle) emit("document-loaded", handle);
        }
      } catch (err) {
        emit("error", err instanceof Error ? err : new Error(String(err)));
      }
    }
    function cleanup() {
      if (viewer) {
        viewer.destroy();
        viewer = null;
      }
      if (ownsEngine && engine) {
        void engine.destroy();
        engine = null;
      }
    }
    onMounted(async () => {
      await initEngine();
      if (disposed) {
        cleanup();
        return;
      }
      createViewer();
      if (disposed) {
        cleanup();
        return;
      }
      await loadDocument();
      if (disposed) cleanup();
    });
    onBeforeUnmount(() => {
      disposed = true;
      cleanup();
    });
    watch([() => props.src, () => props.password], () => {
      void loadDocument();
    });
    watch(() => props.theme, (newTheme) => {
      if (!engine || !newTheme) return;
      const ui = engine.plugins.tryGet("ui");
      ui?.setTheme(newTheme);
    });
    expose({
      /** The underlying LectorEngine. */
      get engine() {
        return engine;
      },
      /** The underlying LectorViewer. */
      get viewer() {
        return viewer;
      }
    });
    return () => h("div", {
      ref: containerEl,
      style: { height: "100%" }
    });
  }
});
export {
  ALL_PLUGINS,
  CORE_PLUGINS,
  LECTOR_KEY,
  LectorPdfViewer,
  READER_PLUGINS,
  useAnnotations,
  useDocument,
  useForm,
  useHistory,
  useI18n,
  useLector,
  useNavigation,
  useOptionalPlugin,
  usePlugin,
  useSearch,
  useSignalRef,
  useSignatures,
  useTextSelection,
  useViewport,
  useZoom
};
