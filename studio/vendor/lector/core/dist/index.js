import {
  ANNOTATION_TOOL_DEFAULTS,
  BlendMode,
  DEFAULT_RENDER_OPTIONS,
  LectorPane,
  LineCap,
  MeasurementUnit,
  NoteIcon,
  PageOverlayManager,
  PageViewport,
  RenderPriority,
  TOOL_TO_SUBTYPE,
  createDrawModeHandler,
  definePlugin,
  getIcon,
  isEraserTool,
  isInkTool,
  isInlineSvg,
  isMarkupTool,
  isMeasurementTool,
  isPlacementTool,
  isPolygonTool,
  isRedactionTool,
  isShapeTool,
  isStampTool,
  isToolOutputAnnotation,
  isToolOutputTool,
  isUserAnnotation,
  measurementPlugin,
  resolveIcon,
  uuid
} from "./chunk-TZBNFEF3.js";

// src/engine/lector-engine.ts
import * as Comlink2 from "comlink";

// src/types/errors.ts
function isSerializedPdfiumError(value) {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const obj = value;
  return obj["name"] === "PdfiumError" && typeof obj["message"] === "string" && typeof obj["code"] === "number";
}

// src/errors/engine-error.ts
var EngineErrorCode = {
  UNKNOWN: "UNKNOWN",
  FILE: "FILE",
  FORMAT: "FORMAT",
  PASSWORD_REQUIRED: "PASSWORD_REQUIRED",
  SECURITY: "SECURITY",
  PAGE: "PAGE",
  NOT_INITIALIZED: "NOT_INITIALIZED",
  ALREADY_DESTROYED: "ALREADY_DESTROYED",
  DOCUMENT_NOT_FOUND: "DOCUMENT_NOT_FOUND",
  WORKER_TERMINATED: "WORKER_TERMINATED",
  RENDER_ABORTED: "RENDER_ABORTED"
};
var PDFIUM_CODE_MAP = {
  1: EngineErrorCode.UNKNOWN,
  // FPDF_ERR_UNKNOWN
  2: EngineErrorCode.FILE,
  // FPDF_ERR_FILE
  3: EngineErrorCode.FORMAT,
  // FPDF_ERR_FORMAT
  4: EngineErrorCode.PASSWORD_REQUIRED,
  // FPDF_ERR_PASSWORD
  5: EngineErrorCode.SECURITY,
  // FPDF_ERR_SECURITY
  6: EngineErrorCode.PAGE
  // FPDF_ERR_PAGE
};
var EngineError = class extends Error {
  name = "EngineError";
  code;
  pdfiumCode;
  constructor(code, message, pdfiumCode) {
    super(message);
    this.code = code;
    this.pdfiumCode = pdfiumCode;
  }
};
function fromSerializedError(err) {
  if (err instanceof EngineError) {
    return err;
  }
  if (isSerializedPdfiumError(err)) {
    return fromPdfiumEnvelope(err);
  }
  if (err instanceof Error) {
    return new EngineError(
      EngineErrorCode.UNKNOWN,
      err.message
    );
  }
  return new EngineError(
    EngineErrorCode.UNKNOWN,
    String(err)
  );
}
function fromPdfiumEnvelope(envelope) {
  const engineCode = PDFIUM_CODE_MAP[envelope.code] ?? EngineErrorCode.UNKNOWN;
  const message = envelope.context ? `${envelope.context}: ${envelope.message}` : envelope.message;
  return new EngineError(engineCode, message, envelope.code);
}

// src/plugin/event-bus.ts
var EventBus = class {
  #listeners = /* @__PURE__ */ new Map();
  /**
   * Subscribe to an event. Returns an unsubscribe function.
   *
   * @param event - The event name to listen for.
   * @param handler - Callback invoked when the event is emitted.
   * @returns A function that removes this listener.
   */
  on(event, handler) {
    let handlers = this.#listeners.get(event);
    if (handlers === void 0) {
      handlers = /* @__PURE__ */ new Set();
      this.#listeners.set(event, handlers);
    }
    handlers.add(handler);
    return () => {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this.#listeners.delete(event);
      }
    };
  }
  /**
   * Emit an event, invoking all registered listeners synchronously.
   *
   * @param event - The event name to emit.
   * @param args - Arguments forwarded to each listener.
   */
  emit(event, ...args) {
    const handlers = this.#listeners.get(event);
    if (handlers === void 0) return;
    for (const handler of [...handlers]) {
      handler(...args);
    }
  }
  /** Dispose: clear all listeners. */
  [Symbol.dispose]() {
    this.#listeners.clear();
  }
};

// src/plugin/commands.ts
var CommandRegistry = class {
  #commands = /* @__PURE__ */ new Map();
  /**
   * Register a command. Throws if a command with the same ID already exists.
   *
   * @param command - The command to register.
   */
  register(command) {
    if (this.#commands.has(command.id)) {
      throw new Error(`Command "${command.id}" is already registered`);
    }
    this.#commands.set(command.id, command);
  }
  /**
   * Unregister a command by ID.
   *
   * @param id - The command ID to remove.
   */
  unregister(id) {
    this.#commands.delete(id);
  }
  /**
   * Execute a command by ID.
   *
   * @param id - The command ID to execute.
   * @throws If the command is not found or is disabled.
   */
  async execute(id) {
    const command = this.#commands.get(id);
    if (command === void 0) {
      throw new Error(`Command "${id}" not found`);
    }
    if (command.enabled !== void 0 && !command.enabled.value) {
      throw new Error(`Command "${id}" is disabled`);
    }
    await command.execute();
  }
  /**
   * Get a command by ID.
   *
   * @param id - The command ID to look up.
   * @returns The command, or undefined if not found.
   */
  get(id) {
    return this.#commands.get(id);
  }
  /**
   * Get all registered commands as a read-only map.
   */
  getAll() {
    return this.#commands;
  }
  /**
   * Get all commands in a given category.
   *
   * @param category - The category to filter by.
   * @returns Commands whose category matches.
   */
  getByCategory(category) {
    const result = [];
    for (const command of this.#commands.values()) {
      if (command.category === category) {
        result.push(command);
      }
    }
    return result;
  }
  /** Dispose: clear all commands. */
  [Symbol.dispose]() {
    this.#commands.clear();
  }
};

// src/plugin/dependency-resolver.ts
function resolveDependencies(plugins) {
  const capabilityToPlugin = /* @__PURE__ */ new Map();
  for (const plugin of plugins) {
    for (const cap of plugin.provides) {
      if (capabilityToPlugin.has(cap)) {
        throw new Error(
          `Duplicate capability "${cap}": provided by both "${capabilityToPlugin.get(cap)}" and "${plugin.id}"`
        );
      }
      capabilityToPlugin.set(cap, plugin.id);
    }
  }
  for (const plugin of plugins) {
    for (const req of plugin.requires) {
      if (!capabilityToPlugin.has(req)) {
        throw new Error(
          `Plugin "${plugin.id}" requires capability "${req}" but no plugin provides it`
        );
      }
    }
  }
  const pluginIds = new Set(plugins.map((p) => p.id));
  const adjacency = /* @__PURE__ */ new Map();
  const inDegree = /* @__PURE__ */ new Map();
  for (const id of pluginIds) {
    adjacency.set(id, /* @__PURE__ */ new Set());
    inDegree.set(id, 0);
  }
  for (const plugin of plugins) {
    const deps = [...plugin.requires, ...plugin.optional];
    for (const dep of deps) {
      const providerId = capabilityToPlugin.get(dep);
      if (providerId === void 0) {
        continue;
      }
      if (providerId === plugin.id) {
        continue;
      }
      const edges = adjacency.get(providerId);
      if (!edges.has(plugin.id)) {
        edges.add(plugin.id);
        inDegree.set(plugin.id, inDegree.get(plugin.id) + 1);
      }
    }
  }
  const queue = [];
  for (const [id, degree] of inDegree) {
    if (degree === 0) {
      queue.push(id);
    }
  }
  const sorted = [];
  while (queue.length > 0) {
    const current = queue.shift();
    sorted.push(current);
    for (const neighbor of adjacency.get(current)) {
      const newDegree = inDegree.get(neighbor) - 1;
      inDegree.set(neighbor, newDegree);
      if (newDegree === 0) {
        queue.push(neighbor);
      }
    }
  }
  if (sorted.length !== pluginIds.size) {
    const cyclePath = detectCycle(pluginIds, adjacency, inDegree);
    throw new Error(`Circular dependency detected: ${cyclePath.join(" \u2192 ")}`);
  }
  return sorted;
}
function detectCycle(pluginIds, adjacency, inDegree) {
  const remaining = /* @__PURE__ */ new Set();
  for (const id of pluginIds) {
    if (inDegree.get(id) > 0) {
      remaining.add(id);
    }
  }
  const visited = /* @__PURE__ */ new Set();
  const stack = /* @__PURE__ */ new Set();
  const parent = /* @__PURE__ */ new Map();
  for (const start of remaining) {
    if (visited.has(start)) continue;
    const found = dfs(start, remaining, adjacency, visited, stack, parent);
    if (found !== null) {
      const cycle = [found];
      let current = parent.get(found);
      while (current !== void 0 && current !== found) {
        cycle.push(current);
        current = parent.get(current);
      }
      cycle.push(found);
      cycle.reverse();
      return cycle;
    }
  }
  return [...remaining];
}
function dfs(node, remaining, adjacency, visited, stack, parent) {
  visited.add(node);
  stack.add(node);
  for (const neighbor of adjacency.get(node) ?? []) {
    if (!remaining.has(neighbor)) continue;
    if (!visited.has(neighbor)) {
      parent.set(neighbor, node);
      const result = dfs(neighbor, remaining, adjacency, visited, stack, parent);
      if (result !== null) return result;
    } else if (stack.has(neighbor)) {
      parent.set(neighbor, node);
      return neighbor;
    }
  }
  stack.delete(node);
  return null;
}

// src/plugin/context.ts
import { effect } from "@truespar/lector-utils";
function createPluginContext(options) {
  const { capabilities, events, engine, commands, state, pluginId } = options;
  return {
    require(capability) {
      if (!capabilities.has(capability)) {
        throw new Error(
          `Plugin "${pluginId}" requires capability "${capability}" but it is not available`
        );
      }
      return capabilities.get(capability);
    },
    optional(capability) {
      const value = capabilities.get(capability);
      return value === void 0 ? null : value;
    },
    state,
    on(event, handler) {
      return events.on(event, handler);
    },
    emit(event, ...args) {
      events.emit(event, ...args);
    },
    effect(fn) {
      return effect(fn);
    },
    engine,
    registerCommand(command) {
      commands.register(command);
    }
  };
}

// src/plugin/registry.ts
var PluginRegistry = class {
  #engine;
  #definitions = [];
  #capabilities = /* @__PURE__ */ new Map();
  #disposeCallbacks = [];
  #initialized = false;
  /** Shared event bus for inter-plugin communication. */
  events;
  /** Shared command registry for keyboard shortcuts and actions. */
  commands;
  constructor(engine) {
    this.#engine = engine;
    this.events = new EventBus();
    this.commands = new CommandRegistry();
  }
  /**
   * Register a plugin definition. Must be called before `init()`.
   *
   * @param definition - The plugin definition to register.
   * @throws If the registry has already been initialized.
   */
  register(definition) {
    if (this.#initialized) {
      throw new Error(
        "Cannot register plugins after initialization \u2014 call register() before init()"
      );
    }
    this.#definitions.push(definition);
  }
  /**
   * Initialize all registered plugins in dependency order.
   *
   * 1. Validates no duplicate capabilities across plugins.
   * 2. Resolves topological initialization order via dependency graph.
   * 3. For each plugin (in order): creates state, builds context, calls setup().
   * 4. Stores returned capabilities and dispose callbacks.
   *
   * @throws If already initialized, if dependencies cannot be resolved,
   *         or if a plugin's setup() throws.
   */
  async init() {
    if (this.#initialized) {
      throw new Error("Plugin registry is already initialized");
    }
    const definitionById = /* @__PURE__ */ new Map();
    for (const definition of this.#definitions) {
      if (definitionById.has(definition.id)) {
        throw new Error(`Duplicate plugin ID: "${definition.id}"`);
      }
      definitionById.set(definition.id, definition);
    }
    const order = resolveDependencies(
      this.#definitions.map((d) => ({
        id: d.id,
        provides: d.provides,
        requires: d.requires,
        optional: d.optional
      }))
    );
    for (const pluginId of order) {
      const definition = definitionById.get(pluginId);
      const state = definition.state !== void 0 ? definition.state() : {};
      const ctx = createPluginContext({
        capabilities: this.#capabilities,
        events: this.events,
        engine: this.#engine,
        commands: this.commands,
        state,
        pluginId
      });
      const capability = await definition.setup(ctx);
      for (const cap of definition.provides) {
        this.#capabilities.set(cap, capability);
      }
      if (definition.dispose !== void 0) {
        this.#disposeCallbacks.push(definition.dispose);
      }
    }
    this.#initialized = true;
  }
  /**
   * Get a capability by name. Throws if not found.
   *
   * @typeParam T - The expected capability type.
   * @param capability - The capability name to look up.
   * @returns The capability value.
   * @throws If the capability is not registered.
   */
  get(capability) {
    if (!this.#capabilities.has(capability)) {
      throw new Error(`Capability "${capability}" not found`);
    }
    return this.#capabilities.get(capability);
  }
  /**
   * Get a capability by name, or null if not found.
   *
   * @typeParam T - The expected capability type.
   * @param capability - The capability name to look up.
   * @returns The capability value, or null.
   */
  tryGet(capability) {
    const value = this.#capabilities.get(capability);
    return value === void 0 ? null : value;
  }
  /**
   * Dispose: run plugin dispose callbacks in reverse order,
   * then dispose the event bus and command registry.
   */
  [Symbol.dispose]() {
    for (let i = this.#disposeCallbacks.length - 1; i >= 0; i--) {
      const callback = this.#disposeCallbacks[i];
      try {
        const result = callback();
        if (result instanceof Promise) {
          void result.catch(() => {
          });
        }
      } catch {
      }
    }
    this.#disposeCallbacks.length = 0;
    this.#capabilities.clear();
    this.#definitions.length = 0;
    this.events[Symbol.dispose]();
    this.commands[Symbol.dispose]();
  }
};

// src/engine/abort-tracker.ts
var AbortTracker = class {
  #listeners = /* @__PURE__ */ new Map();
  /**
   * Register an abort listener for a task.
   *
   * If the signal is already aborted, `onAbort` is called synchronously
   * and the task is not stored.
   */
  track(taskId, signal25, onAbort) {
    if (signal25.aborted) {
      onAbort();
      return;
    }
    const handler = () => {
      this.#listeners.delete(taskId);
      onAbort();
    };
    signal25.addEventListener("abort", handler, { once: true });
    this.#listeners.set(taskId, { signal: signal25, handler });
  }
  /** Remove the abort listener for a task. No-op if the task is not tracked. */
  untrack(taskId) {
    const entry = this.#listeners.get(taskId);
    if (entry !== void 0) {
      entry.signal.removeEventListener("abort", entry.handler);
      this.#listeners.delete(taskId);
    }
  }
  /** Remove all abort listeners. */
  [Symbol.dispose]() {
    for (const [, entry] of this.#listeners) {
      entry.signal.removeEventListener("abort", entry.handler);
    }
    this.#listeners.clear();
  }
};

// src/engine/render-scheduler.ts
function deduplicationKey(docId, pageIndex, width, height) {
  return `${docId}:${pageIndex}:${width}:${height}`;
}
function coalesceKey(docId, pageIndex, priority) {
  return `${docId}:${pageIndex}:${priority}`;
}
var RenderScheduler = class {
  #queue = [];
  #active = null;
  #insertionCounter = 0;
  #taskCounter = 0;
  #disposed = false;
  #abortTracker = new AbortTracker();
  #proxy;
  #pool = null;
  /** Number of concurrent renders allowed (1 = serial, >1 = parallel via pool). */
  #concurrency = 1;
  /** Currently active renders (dispatched but not yet resolved). */
  #activeCount = 0;
  /** Maps deduplication keys to pending promises for identical requests. */
  #pending = /* @__PURE__ */ new Map();
  /**
   * Maps `${docId}:${pageIndex}` to the currently-pending task for that
   * page. When a new render arrives for the same page at different
   * dimensions, the older one is cancelled — only the latest dimensions
   * matter to the user. Without this, sidebar/zoom drags flood the
   * queue with hundreds of stale requests and the worker never catches
   * up.
   */
  #pendingByPage = /* @__PURE__ */ new Map();
  constructor(proxy, pool) {
    this.#proxy = proxy;
    if (pool && pool.size > 0) {
      this.#pool = pool;
      this.#concurrency = pool.size + 1;
    }
  }
  /**
   * Enqueue a page render request.
   *
   * Returns a promise that resolves with the rendered ImageBitmap.
   * If an identical request is already pending, returns the existing promise.
   */
  enqueue(request) {
    if (this.#disposed) {
      return Promise.reject(new DOMException("RenderScheduler is disposed", "AbortError"));
    }
    const priority = request.priority ?? 0;
    const dedupKey = deduplicationKey(request.docId, request.pageIndex, request.width, request.height);
    const coalKey = coalesceKey(request.docId, request.pageIndex, priority);
    const existing = this.#pending.get(dedupKey);
    if (existing !== void 0) {
      return existing.promise;
    }
    const stalePageTaskId = this.#pendingByPage.get(coalKey);
    if (stalePageTaskId !== void 0) {
      this.#cancelTask(stalePageTaskId);
    }
    const taskId = `task_${this.#taskCounter++}`;
    const insertionOrder = this.#insertionCounter++;
    const promise = new Promise((resolve, reject) => {
      const task = {
        taskId,
        request,
        priority,
        insertionOrder,
        resolve,
        reject
      };
      let insertIdx = this.#queue.length;
      for (let i = 0; i < this.#queue.length; i++) {
        const queued = this.#queue[i];
        if (priority < queued.priority || priority === queued.priority && insertionOrder < queued.insertionOrder) {
          insertIdx = i;
          break;
        }
      }
      this.#queue.splice(insertIdx, 0, task);
      if (request.signal !== void 0) {
        this.#abortTracker.track(taskId, request.signal, () => {
          this.#cancelTask(taskId, dedupKey);
        });
      }
    });
    this.#pending.set(dedupKey, { promise, taskId });
    this.#pendingByPage.set(coalKey, taskId);
    this.#dispatch();
    return promise;
  }
  /** Cancel a task by ID. Removes it from the queue or discards the active result. */
  cancel(taskId) {
    this.#cancelTask(taskId);
  }
  /**
   * Cancel every queued and in-flight render for a document. Call this when a
   * document is closing so outstanding renders don't resolve against a closed
   * document: their promises reject with AbortError, and any late-arriving
   * pool bitmap is dropped and closed by the success guard (no leak).
   */
  cancelDocument(docId) {
    const ids = [];
    for (const task of this.#queue) {
      if (task.request.docId === docId) ids.push(task.taskId);
    }
    for (const [taskId, task] of this.#activeTasks) {
      if (task.request.docId === docId) ids.push(taskId);
    }
    for (const id of ids) this.#cancelTask(id);
  }
  /**
   * Change the priority of all pending tasks matching a specific document and page.
   * Tasks that are already actively rendering are not affected.
   */
  reprioritize(docId, pageIndex, newPriority) {
    const updated = [];
    const unchanged = [];
    for (const task of this.#queue) {
      if (task.request.docId === docId && task.request.pageIndex === pageIndex) {
        updated.push({
          ...task,
          priority: newPriority
        });
      } else {
        unchanged.push(task);
      }
    }
    if (updated.length === 0) {
      return;
    }
    const merged = [...unchanged, ...updated];
    merged.sort((a, b) => {
      if (a.priority !== b.priority) {
        return a.priority - b.priority;
      }
      return a.insertionOrder - b.insertionOrder;
    });
    this.#queue = merged;
  }
  /** Dispose the scheduler, rejecting all pending tasks. */
  [Symbol.dispose]() {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    const abortError = new DOMException("RenderScheduler disposed", "AbortError");
    for (const task of this.#queue) {
      task.reject(abortError);
    }
    this.#queue = [];
    this.#pending.clear();
    this.#pendingByPage.clear();
    this.#abortTracker[Symbol.dispose]();
  }
  #cancelTask(taskId, dedupKey) {
    const queueIdx = this.#queue.findIndex((t) => t.taskId === taskId);
    if (queueIdx !== -1) {
      const task = this.#queue[queueIdx];
      this.#queue.splice(queueIdx, 1);
      this.#abortTracker.untrack(taskId);
      if (dedupKey !== void 0) {
        this.#pending.delete(dedupKey);
      } else {
        const key = deduplicationKey(
          task.request.docId,
          task.request.pageIndex,
          task.request.width,
          task.request.height
        );
        this.#pending.delete(key);
      }
      const coalKey = coalesceKey(task.request.docId, task.request.pageIndex, task.priority);
      if (this.#pendingByPage.get(coalKey) === taskId) {
        this.#pendingByPage.delete(coalKey);
      }
      task.reject(new DOMException("Render cancelled", "AbortError"));
      return;
    }
    const activeTask = this.#activeTasks.get(taskId);
    if (activeTask) {
      this.#activeTasks.delete(taskId);
      this.#activeCount--;
      this.#abortTracker.untrack(taskId);
      if (this.#active?.taskId === taskId) this.#active = null;
      activeTask.reject(new DOMException("Render cancelled", "AbortError"));
    }
  }
  /** Tasks currently being rendered, by ID (for cancellation + settle). */
  #activeTasks = /* @__PURE__ */ new Map();
  #dispatch() {
    while (!this.#disposed && this.#activeCount < this.#concurrency && this.#queue.length > 0) {
      const task = this.#queue.shift();
      this.#activeCount++;
      this.#activeTasks.set(task.taskId, task);
      if (this.#concurrency === 1) this.#active = task;
      this.#runTask(task);
    }
  }
  /**
   * Dispatch one task to a render target. The pool is a best-effort
   * optimization: if a pool render REJECTS (worker lacks the doc, isn't ready
   * yet, a transient error), we fall back to the authoritative primary worker
   * before giving up, so a desynced or failing pool worker can never break a
   * render. When the pool returns null (no worker holds the doc) we go
   * straight to the primary.
   */
  #runTask(task) {
    const { request } = task;
    const poolPromise = this.#pool?.renderPage(
      request.docId,
      request.pageIndex,
      request.width,
      request.height,
      request.options
    );
    const renderOnPrimary = () => this.#proxy.renderPage(
      request.docId,
      request.pageIndex,
      request.width,
      request.height,
      request.options
    );
    if (poolPromise) {
      poolPromise.then(
        (bitmap) => this.#onRenderSuccess(task, bitmap),
        () => {
          if (!this.#activeTasks.has(task.taskId)) return;
          renderOnPrimary().then(
            (bitmap) => this.#onRenderSuccess(task, bitmap),
            (err) => this.#onRenderError(task, err)
          );
        }
      );
    } else {
      renderOnPrimary().then(
        (bitmap) => this.#onRenderSuccess(task, bitmap),
        (err) => this.#onRenderError(task, err)
      );
    }
  }
  #settleBookkeeping(task) {
    this.#activeCount--;
    this.#activeTasks.delete(task.taskId);
    if (this.#active?.taskId === task.taskId) this.#active = null;
    this.#abortTracker.untrack(task.taskId);
    const { request } = task;
    const key = deduplicationKey(request.docId, request.pageIndex, request.width, request.height);
    this.#pending.delete(key);
    const coalKey = coalesceKey(request.docId, request.pageIndex, task.priority);
    if (this.#pendingByPage.get(coalKey) === task.taskId) {
      this.#pendingByPage.delete(coalKey);
    }
  }
  #onRenderSuccess(task, bitmap) {
    if (!this.#activeTasks.has(task.taskId)) {
      bitmap.close();
      return;
    }
    this.#settleBookkeeping(task);
    task.resolve(bitmap);
    this.#dispatch();
  }
  #onRenderError(task, err) {
    if (!this.#activeTasks.has(task.taskId)) return;
    this.#settleBookkeeping(task);
    task.reject(err);
    this.#dispatch();
  }
};

// src/engine/render-pool.ts
import * as Comlink from "comlink";
var RenderPool = class {
  #workers = [];
  #proxies = [];
  #roundRobin = 0;
  #initialized = false;
  /** Track which documents are open on pool workers, so we can close them. */
  #openDocs = /* @__PURE__ */ new Map();
  /**
   * Maps the primary worker's DocumentId to each pool worker's OWN
   * DocumentId for the same document (null when that worker doesn't hold it —
   * its open failed, or its copy was invalidated/marked stale). Pool workers
   * mint ids from an independent counter, so the primary's id must never be
   * assumed to match a pool worker's; every call translates per worker.
   */
  #docMap = /* @__PURE__ */ new Map();
  get size() {
    return this.#workers.length;
  }
  /**
   * Create and initialize the pool.
   *
   * @param poolSize - Number of render workers (0 = disabled)
   * @param workerUrl - URL of the worker script (same as primary worker)
   * @param wasmUrl - URL of pdfium.wasm
   * @param wasmJsUrl - URL of pdfium.js
   */
  async init(poolSize, workerUrl, wasmUrl, wasmJsUrl) {
    if (poolSize <= 0 || this.#initialized) return;
    const initPromises = [];
    for (let i = 0; i < poolSize; i++) {
      const worker = new Worker(workerUrl, {
        type: "module",
        name: `lector-render-${i}`
      });
      const proxy = Comlink.wrap(worker);
      this.#workers.push(worker);
      this.#proxies.push(proxy);
      initPromises.push(proxy.init(wasmUrl, wasmJsUrl));
    }
    await Promise.all(initPromises);
    this.#initialized = true;
  }
  /**
   * Open a document on all pool workers. Must be called after the primary
   * worker opens the document (so we have the data + docId).
   *
   * @param docId - The document ID assigned by the primary worker
   * @param data - Raw PDF bytes
   * @param password - Optional password
   */
  async openDocument(docId, data, password) {
    if (this.#proxies.length === 0) return;
    this.#openDocs.set(docId, { data, password });
    const ids = await Promise.all(
      this.#proxies.map(
        (proxy) => proxy.openDocument(data.slice(0), password).then(
          (workerDocId) => workerDocId,
          () => null
        )
      )
    );
    this.#docMap.set(docId, ids);
  }
  /**
   * Close a document on all pool workers.
   */
  async closeDocument(docId) {
    this.#openDocs.delete(docId);
    const ids = this.#docMap.get(docId);
    this.#docMap.delete(docId);
    if (this.#proxies.length === 0 || !ids) return;
    await Promise.all(
      this.#proxies.map((proxy, i) => {
        const workerDocId = ids[i];
        if (workerDocId == null) return Promise.resolve();
        return proxy.closeDocument(workerDocId).catch(() => {
        });
      })
    );
  }
  /**
   * Re-run a destructive page mutation on every pool worker so their
   * pdfium copies stay in sync with the primary worker. Caller is
   * responsible for having already executed the same op on the primary.
   * Resolves only after all pool workers complete.
   */
  async applyRedactions(docId, pageIndex, specs) {
    if (this.#proxies.length === 0 || specs.length === 0) return;
    const ids = this.#docMap.get(docId);
    if (!ids) return;
    await Promise.all(
      this.#proxies.map((proxy, i) => {
        const workerDocId = ids[i];
        if (workerDocId == null) return Promise.resolve();
        return proxy.applyRedactions(workerDocId, pageIndex, specs, false).catch(() => {
          ids[i] = null;
        });
      })
    );
  }
  /**
   * Render a page on an available pool worker (round-robin over the workers
   * that actually hold this document). Returns null when no pool worker can
   * serve it — the document isn't mapped, every copy is stale, or the pool is
   * empty — and the scheduler then renders on the primary worker. The
   * primary's `docId` is translated to each worker's own id; we never assume
   * the ids coincide.
   */
  renderPage(docId, pageIndex, widthPx, heightPx, options) {
    const ids = this.#docMap.get(docId);
    if (!ids || this.#proxies.length === 0) return null;
    const n = this.#proxies.length;
    for (let k = 0; k < n; k++) {
      const i = (this.#roundRobin + k) % n;
      const workerDocId = ids[i];
      if (workerDocId != null) {
        this.#roundRobin = (i + 1) % n;
        return this.#proxies[i].renderPage(workerDocId, pageIndex, widthPx, heightPx, options);
      }
    }
    return null;
  }
  /**
   * Mark a document's pool copies stale — used after a structural page
   * mutation (delete/insert/move/rotate/duplicate/flatten) that the pool
   * cannot replay. Renders fall back to the primary worker until the document
   * is closed, guaranteeing correct content at the cost of pool parallelism
   * for that document.
   */
  invalidate(docId) {
    const ids = this.#docMap.get(docId);
    if (!ids) return;
    ids.fill(null);
  }
  /** Dispose all pool workers. */
  [Symbol.dispose]() {
    for (const worker of this.#workers) {
      worker.terminate();
    }
    this.#workers.length = 0;
    this.#proxies.length = 0;
    this.#openDocs.clear();
    this.#docMap.clear();
  }
};

// src/engine/page-rotation-cache.ts
var PageRotationCache = class {
  #cache = /* @__PURE__ */ new Map();
  #pending = /* @__PURE__ */ new Map();
  #fetch;
  /**
   * @param fetch Loads a page's raw rotation from the worker
   *   (`FPDFPage_GetRotation`). Only invoked after the engine is initialized.
   */
  constructor(fetch2) {
    this.#fetch = fetch2;
  }
  /** Synchronous read. Returns the cached rotation, or 0 if not yet resolved. */
  get(docId, pageIndex) {
    return this.#cache.get(docId)?.get(pageIndex) ?? 0;
  }
  /** Whether this page's rotation has been resolved (vs. defaulting to 0). */
  has(docId, pageIndex) {
    return this.#cache.get(docId)?.has(pageIndex) ?? false;
  }
  /**
   * Resolve and cache a page's rotation. Concurrent calls for the same page
   * share one worker request. Returns 0 if the worker call fails.
   */
  async resolve(docId, pageIndex) {
    const cached = this.#cache.get(docId)?.get(pageIndex);
    if (cached !== void 0) return cached;
    const key = `${docId}\0${pageIndex}`;
    let pending = this.#pending.get(key);
    if (pending === void 0) {
      pending = this.#fetch(docId, pageIndex).then((raw) => {
        const rot = (Math.trunc(raw) % 4 + 4) % 4;
        this.#store(docId, pageIndex, rot);
        this.#pending.delete(key);
        return rot;
      }).catch(() => {
        this.#pending.delete(key);
        return 0;
      });
      this.#pending.set(key, pending);
    }
    return pending;
  }
  /** Invalidate one page's cached rotation (e.g. after a rotate operation). */
  invalidate(docId, pageIndex) {
    this.#cache.get(docId)?.delete(pageIndex);
    this.#pending.delete(`${docId}\0${pageIndex}`);
  }
  /** Drop all cached rotations for a document (on close). */
  clearDocument(docId) {
    this.#cache.delete(docId);
    const prefix = `${docId}\0`;
    for (const key of [...this.#pending.keys()]) {
      if (key.startsWith(prefix)) this.#pending.delete(key);
    }
  }
  #store(docId, pageIndex, rot) {
    let pages = this.#cache.get(docId);
    if (pages === void 0) {
      pages = /* @__PURE__ */ new Map();
      this.#cache.set(docId, pages);
    }
    pages.set(pageIndex, rot);
  }
  [Symbol.dispose]() {
    this.#cache.clear();
    this.#pending.clear();
  }
};

// src/engine/lector-engine.ts
var LectorEngine = class {
  #worker = null;
  #proxy = null;
  #scheduler = null;
  #renderPool = null;
  /** Lazily-populated, session-wide per-page rotation cache. */
  #pageRotation;
  /** Teardown for the rotation-cache event subscriptions. */
  #rotationUnsubscribe = null;
  #options;
  #initialized = false;
  #destroyed = false;
  /** Plugin registry — register plugins before calling init(). */
  plugins;
  /** Current user identity, if provided by the consumer. */
  user;
  /** Mention user resolver for @mention autocomplete. */
  mentionUsers;
  /** Initial UI translation locale for i18n. */
  locale;
  /** Custom translation overrides. */
  translations;
  /** BCP 47 format locale (or undefined to auto-detect from `navigator.language`). */
  formatLocale;
  /** Measurement system override (or undefined to derive from `formatLocale`). */
  measurementSystem;
  /** Hour-cycle override (or undefined to use the locale default). */
  hourCycle;
  /** Custom recent-files store, if provided. */
  recentFilesStore;
  /** Max number of recent-files entries. */
  recentFilesMax;
  /** localStorage key for recent files. */
  recentFilesStorageKey;
  /** Auto-register the viewer container as a drop zone. */
  enableViewerDropZone;
  /** Document-level permission overrides. */
  permissions;
  /** Whether annotation tools stay active after creation. */
  keepSelectedTool;
  /** Custom stamp templates. */
  customStamps;
  /** Named annotation style presets. */
  annotationPresets;
  /** Zoom limits and step — consumed by the zoom plugin. */
  zoomMin;
  zoomMax;
  zoomStep;
  /** Viewport spacing — consumed by the viewport plugin. */
  pageGap;
  viewportPadding;
  /** Annotation tool defaults — consumed by the annotation plugin. */
  annotationDefaults;
  /** Render quality — consumed by render and capture plugins. */
  renderDpi;
  captureDpi;
  constructor(options) {
    this.#options = options;
    this.user = options.user;
    this.mentionUsers = options.mentionUsers;
    this.locale = options.locale ?? "en";
    this.translations = options.translations;
    this.formatLocale = options.formatLocale;
    this.measurementSystem = options.measurementSystem;
    this.hourCycle = options.hourCycle;
    this.recentFilesStore = options.recentFilesStore;
    this.recentFilesMax = options.recentFilesMax;
    this.recentFilesStorageKey = options.recentFilesStorageKey;
    this.enableViewerDropZone = options.enableViewerDropZone ?? true;
    this.permissions = options.permissions;
    this.keepSelectedTool = options.keepSelectedTool ?? false;
    this.customStamps = options.customStamps;
    this.annotationPresets = options.annotationPresets;
    this.zoomMin = options.zoomMin;
    this.zoomMax = options.zoomMax;
    this.zoomStep = options.zoomStep;
    this.pageGap = options.pageGap;
    this.viewportPadding = options.viewportPadding;
    this.annotationDefaults = options.annotationDefaults;
    this.renderDpi = options.renderDpi;
    this.captureDpi = options.captureDpi;
    this.plugins = new PluginRegistry(this);
    this.#pageRotation = new PageRotationCache(
      (docId, pageIndex) => this.workerProxy.getPageRotation(docId, pageIndex)
    );
  }
  /**
   * Initialize the engine: create the worker, load the WASM module.
   *
   * Must be called exactly once before any other method.
   * @throws {EngineError} with code NOT_INITIALIZED if init fails.
   */
  async init() {
    if (this.#destroyed) {
      throw new EngineError(
        EngineErrorCode.ALREADY_DESTROYED,
        "Cannot initialize a destroyed engine"
      );
    }
    if (this.#initialized) {
      return;
    }
    const workerUrl = this.#options.workerUrl;
    if (workerUrl === void 0) {
      throw new EngineError(
        EngineErrorCode.NOT_INITIALIZED,
        "workerUrl is required \u2014 provide the URL to the bundled pdfium-worker.js"
      );
    }
    try {
      this.#worker = new Worker(workerUrl, {
        type: "module",
        name: "lector-pdfium"
      });
      this.#proxy = Comlink2.wrap(this.#worker);
      const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
      const useThreaded = isolated || !this.#options.wasmUrlFallback;
      const wasmUrl = useThreaded ? this.#options.wasmUrl : this.#options.wasmUrlFallback;
      const wasmJsUrl = useThreaded ? this.#options.wasmJsUrl : this.#options.wasmJsUrlFallback ?? this.#options.wasmJsUrl;
      await this.#proxy.init(wasmUrl, wasmJsUrl);
      const poolSize = useThreaded ? this.#options.renderPoolSize ?? 0 : 0;
      if (poolSize > 0) {
        this.#renderPool = new RenderPool();
        await this.#renderPool.init(
          poolSize,
          workerUrl,
          wasmUrl,
          wasmJsUrl
        );
      }
      this.#scheduler = new RenderScheduler(this.#proxy, this.#renderPool ?? void 0);
      const events = this.plugins.events;
      const offRotated = events.on("page-ops:page-rotated", (...args) => {
        this.#pageRotation.invalidate(args[0], args[1]);
      });
      const offClosed = events.on("document:closed", (...args) => {
        this.#pageRotation.clearDocument(args[0]);
      });
      this.#rotationUnsubscribe = () => {
        offRotated();
        offClosed();
      };
      await this.plugins.init();
      this.#initialized = true;
      this.plugins.events.emit("engine:ready");
    } catch (err) {
      this.#scheduler = null;
      this.#renderPool?.[Symbol.dispose]();
      this.#renderPool = null;
      this.#proxy = null;
      this.#worker?.terminate();
      this.#worker = null;
      throw toEngineError(err);
    }
  }
  /**
   * Open a PDF document from a URL, fetching it with optional custom headers.
   *
   * @param url URL to fetch the PDF from.
   * @param options Fetch options (headers, credentials, etc.) and optional password.
   * @returns A DocumentHandle with metadata and a close() method.
   *
   * @example
   * ```ts
   * const doc = await engine.openDocumentFromUrl('/api/documents/123.pdf', {
   *   headers: { Authorization: `Bearer ${token}`, 'X-Team-Id': teamId },
   *   credentials: 'include',
   *   password: 'secret',
   * });
   * ```
   */
  async openDocumentFromUrl(url, options) {
    const { password, ...fetchOptions } = options ?? {};
    const response = await fetch(url, fetchOptions);
    if (!response.ok) {
      throw new Error(`Failed to fetch PDF: ${response.status} ${response.statusText}`);
    }
    const data = await response.arrayBuffer();
    return this.openDocument(data, password);
  }
  /**
   * Open a PDF document from raw bytes.
   *
   * The ArrayBuffer is transferred to the worker (zero-copy). After this call,
   * the provided ArrayBuffer is detached and cannot be used.
   *
   * @param data PDF file contents as an ArrayBuffer.
   * @param password Optional password for encrypted PDFs.
   * @returns A DocumentHandle with metadata and a close() method.
   */
  async openDocument(data, password) {
    this.#assertReady();
    try {
      const poolData = this.#renderPool ? data.slice(0) : null;
      const docId = await this.#proxy.openDocument(
        Comlink2.transfer(data, [data]),
        password
      );
      if (this.#renderPool && poolData) {
        void this.#renderPool.openDocument(docId, poolData, password);
      }
      const [pageSizes, sha256] = await Promise.all([
        this.#proxy.getAllPageSizes(docId),
        this.#proxy.getDocumentHash(docId)
      ]);
      const pageCount = pageSizes.length;
      let closed = false;
      const proxy = this.#proxy;
      const pool = this.#renderPool;
      const scheduler = this.#scheduler;
      const events = this.plugins.events;
      const handle = {
        id: docId,
        pageCount,
        pageSizes,
        sha256,
        async close() {
          if (!closed) {
            closed = true;
            scheduler?.cancelDocument(docId);
            events.emit("document:closed", docId);
            await proxy.closeDocument(docId);
            void pool?.closeDocument(docId);
          }
        },
        [Symbol.dispose]() {
          if (!closed) {
            closed = true;
            scheduler?.cancelDocument(docId);
            events.emit("document:closed", docId);
            void proxy.closeDocument(docId);
            void pool?.closeDocument(docId);
          }
        }
      };
      events.emit("document:opened", docId);
      return handle;
    } catch (err) {
      throw toEngineError(err);
    }
  }
  /**
   * Render a page to an ImageBitmap at the specified pixel dimensions.
   *
   * The returned ImageBitmap is transferred from the worker (zero-copy).
   * The caller owns the bitmap and must call `.close()` when done.
   *
   * @param docId Document identifier from a DocumentHandle.
   * @param pageIndex Zero-based page index.
   * @param width Target width in pixels.
   * @param height Target height in pixels.
   * @param options Render options including flags, rotation, DPI, and scheduling hints.
   * @returns The rendered page as an ImageBitmap.
   */
  async renderPage(docId, pageIndex, width, height, options) {
    this.#assertReady();
    try {
      return await this.#scheduler.enqueue({
        docId,
        pageIndex,
        width,
        height,
        options,
        priority: options?.priority,
        signal: options?.signal
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new EngineError(EngineErrorCode.RENDER_ABORTED, err.message);
      }
      throw toEngineError(err);
    }
  }
  /**
   * Render a rectangular tile of a PDF page. Used by the tile-based
   * rendering system for large pages at high zoom where allocating a
   * full-page bitmap would exceed memory limits.
   *
   * Bypasses the render scheduler (tiles have their own dedup/cancel
   * logic in TileManager) and calls the worker directly.
   */
  async renderPageTile(docId, pageIndex, tileX, tileY, tileW, tileH, fullW, fullH, options) {
    this.#assertReady();
    return this.#proxy.renderPageTile(
      docId,
      pageIndex,
      tileX,
      tileY,
      tileW,
      tileH,
      fullW,
      fullH,
      options
    );
  }
  /**
   * Access the Comlink proxy to the pdfium worker.
   *
   * This is intended for internal use by plugins that need direct access
   * to worker APIs beyond rendering (text extraction, navigation, annotations, etc.).
   *
   * @throws {EngineError} if the engine is not initialized.
   */
  get workerProxy() {
    this.#assertReady();
    return this.#proxy;
  }
  /**
   * Render pool, if one is configured (via `renderPoolSize`). Plugins
   * that perform destructive page mutations on the primary worker must
   * also propagate the same op to the pool so the pool workers' pdfium
   * copies stay in sync; otherwise renders served from the pool will
   * show pre-mutation content.
   */
  get renderPool() {
    return this.#renderPool;
  }
  /**
   * Session-wide cache of per-page rotation. Used by the overlay, text, and
   * annotation layers to map coordinates correctly on rotated pages. The
   * overlay layer warms it for visible pages via `resolve()`; synchronous
   * consumers (annotation drawing) read it via `get()`.
   */
  get pageRotation() {
    return this.#pageRotation;
  }
  /**
   * Change the priority of pending render tasks for a specific page.
   *
   * Tasks already actively rendering are not affected. This is typically
   * called by the viewport/render plugins when visible pages change.
   *
   * @param docId Document identifier.
   * @param pageIndex Zero-based page index.
   * @param priority New priority level.
   */
  reprioritize(docId, pageIndex, priority) {
    this.#assertReady();
    this.#scheduler.reprioritize(docId, pageIndex, priority);
  }
  /**
   * Synchronous dispose — schedules async cleanup.
   *
   * Use this with the `using` keyword for automatic cleanup.
   * For awaitable cleanup, use `destroy()` instead.
   */
  [Symbol.dispose]() {
    if (!this.#destroyed) {
      this.#destroyed = true;
      this.#rotationUnsubscribe?.();
      this.#rotationUnsubscribe = null;
      this.#pageRotation[Symbol.dispose]();
      this.#scheduler?.[Symbol.dispose]();
      this.#scheduler = null;
      void this.#proxy?.destroy();
      this.#proxy = null;
      this.#worker?.terminate();
      this.#worker = null;
    }
  }
  /**
   * Async dispose — waits for all cleanup to complete.
   *
   * Closes all documents, tears down pdfium, and terminates the worker.
   */
  async destroy() {
    if (!this.#destroyed) {
      this.#destroyed = true;
      this.#rotationUnsubscribe?.();
      this.#rotationUnsubscribe = null;
      this.#pageRotation[Symbol.dispose]();
      this.#scheduler?.[Symbol.dispose]();
      this.#scheduler = null;
      this.#renderPool?.[Symbol.dispose]();
      this.#renderPool = null;
      try {
        await this.#proxy?.destroy();
      } catch {
      }
      this.#proxy = null;
      this.#worker?.terminate();
      this.#worker = null;
    }
  }
  /**
   * Assert the engine is initialized and not destroyed.
   * @throws {EngineError} if not ready.
   */
  #assertReady() {
    if (this.#destroyed) {
      throw new EngineError(
        EngineErrorCode.ALREADY_DESTROYED,
        "Engine has been destroyed"
      );
    }
    if (!this.#initialized || this.#proxy === null || this.#scheduler === null) {
      throw new EngineError(
        EngineErrorCode.NOT_INITIALIZED,
        "Engine not initialized \u2014 call init() first"
      );
    }
  }
};
function toEngineError(err) {
  if (err instanceof EngineError) {
    return err;
  }
  return fromSerializedError(err);
}

// src/plugins/document-plugin.ts
import { signal, computed } from "@truespar/lector-utils";
async function resolveSource(source) {
  if (source instanceof ArrayBuffer) {
    return source;
  }
  if (typeof source === "string" || source instanceof URL) {
    const response = await fetch(source instanceof URL ? source.href : source);
    if (!response.ok) {
      throw new Error(`Failed to fetch PDF: ${response.status} ${response.statusText}`);
    }
    return response.arrayBuffer();
  }
  if (source instanceof File) {
    return source.arrayBuffer();
  }
  throw new Error("Unsupported source type \u2014 expected ArrayBuffer, File, URL, or string");
}
var documentPlugin = definePlugin({
  id: "document",
  provides: ["document"],
  requires: [],
  optional: [],
  setup(ctx) {
    const handles = signal(/* @__PURE__ */ new Map());
    const activeDoc = signal(null);
    const capability = {
      async load(source, password) {
        const arrayBuffer = await resolveSource(source);
        const handle = await ctx.engine.openDocument(arrayBuffer, password);
        handles.update((map) => {
          const next = new Map(map);
          next.set(handle.id, handle);
          return next;
        });
        if (activeDoc.peek() === null) {
          activeDoc.value = handle;
        }
        ctx.emit("document:loaded", handle);
        return handle;
      },
      async close(docId) {
        const handle = handles.peek().get(docId);
        if (handle === void 0) {
          throw new Error(`No document with id "${docId}" is open`);
        }
        if (activeDoc.peek()?.id === docId) {
          activeDoc.value = null;
        }
        handles.update((map) => {
          const next = new Map(map);
          next.delete(docId);
          return next;
        });
        await handle.close();
        ctx.emit("document:closed", docId);
      },
      getHandle(docId) {
        return handles.peek().get(docId);
      },
      activeDocument: computed(() => activeDoc.value),
      setActive(docId) {
        const handle = handles.peek().get(docId);
        if (handle === void 0) {
          throw new Error(`No document with id "${docId}" is open`);
        }
        activeDoc.value = handle;
      }
    };
    ctx.on("page-ops:pages-changed", (...args) => {
      const docId = args[0];
      const newSizes = args[1];
      if (!newSizes) return;
      const oldHandle = handles.peek().get(docId);
      if (!oldHandle) return;
      const updatedHandle = {
        id: oldHandle.id,
        pageCount: newSizes.length,
        pageSizes: newSizes,
        sha256: oldHandle.sha256,
        close: () => oldHandle.close(),
        [Symbol.dispose]: () => oldHandle[Symbol.dispose]()
      };
      handles.update((map) => {
        const next = new Map(map);
        next.set(docId, updatedHandle);
        return next;
      });
      if (activeDoc.peek()?.id === docId) {
        activeDoc.value = updatedHandle;
      }
    });
    return capability;
  },
  async dispose() {
  }
});

// src/plugins/render-plugin.ts
var renderPlugin = definePlugin({
  id: "render",
  provides: ["render"],
  requires: ["document"],
  optional: [],
  setup(ctx) {
    const document2 = ctx.require("document");
    const capability = {
      async renderPage(docId, pageIndex, widthPx, heightPx, options) {
        const handle = document2.getHandle(docId);
        if (handle === void 0) {
          throw new Error(`No document with id "${docId}" is open`);
        }
        if (pageIndex < 0 || pageIndex >= handle.pageCount) {
          throw new RangeError(
            `Page index ${pageIndex} out of range [0, ${handle.pageCount - 1}]`
          );
        }
        return ctx.engine.renderPage(docId, pageIndex, widthPx, heightPx, {
          flags: options?.flags ?? DEFAULT_RENDER_OPTIONS.flags,
          rotation: options?.rotation ?? DEFAULT_RENDER_OPTIONS.rotation,
          priority: options?.priority,
          signal: options?.signal
        });
      },
      reprioritize(docId, pageIndex, priority) {
        ctx.engine.reprioritize(docId, pageIndex, priority);
      }
    };
    return capability;
  }
});

// src/plugins/viewport-plugin.ts
import { signal as signal2, computed as computed2 } from "@truespar/lector-utils";
var DEFAULT_PAGE_GAP = 8;
var DEFAULT_VIEWPORT_PAD = 12;
function computePagePositions(handle, scale, container, mode, pageGap = DEFAULT_PAGE_GAP, viewportPad = DEFAULT_VIEWPORT_PAD) {
  const PAGE_GAP = pageGap;
  const VIEWPORT_PAD = viewportPad;
  const pageSizes = handle.pageSizes;
  const positions = [];
  if (mode === "continuous") {
    let y = VIEWPORT_PAD;
    for (let i = 0; i < pageSizes.length; i++) {
      const ps = pageSizes[i];
      const w = ps.width * scale;
      const h = ps.height * scale;
      const x = Math.max(0, (container.width - w) / 2);
      positions.push({ pageIndex: i, x, y, width: w, height: h });
      y += h + PAGE_GAP;
    }
  } else if (mode === "single") {
    for (let i = 0; i < pageSizes.length; i++) {
      const ps = pageSizes[i];
      const w = ps.width * scale;
      const h = ps.height * scale;
      const x = Math.max(0, (container.width - w) / 2);
      const y = i * (h + PAGE_GAP);
      positions.push({ pageIndex: i, x, y, width: w, height: h });
    }
  } else if (mode === "double" || mode === "book") {
    let y = 0;
    let i = 0;
    if (mode === "book" && pageSizes.length > 0) {
      const ps = pageSizes[0];
      const w = ps.width * scale;
      const h = ps.height * scale;
      const x = Math.max(0, (container.width - w) / 2);
      positions.push({ pageIndex: 0, x, y, width: w, height: h });
      y += h + PAGE_GAP;
      i = 1;
    }
    while (i < pageSizes.length) {
      const leftPs = pageSizes[i];
      const leftW = leftPs.width * scale;
      const leftH = leftPs.height * scale;
      if (i + 1 < pageSizes.length) {
        const rightPs = pageSizes[i + 1];
        const rightW = rightPs.width * scale;
        const rightH = rightPs.height * scale;
        const pairWidth = leftW + PAGE_GAP + rightW;
        const rowHeight = Math.max(leftH, rightH);
        const startX = Math.max(0, (container.width - pairWidth) / 2);
        positions.push({
          pageIndex: i,
          x: startX,
          y,
          width: leftW,
          height: leftH
        });
        positions.push({
          pageIndex: i + 1,
          x: startX + leftW + PAGE_GAP,
          y,
          width: rightW,
          height: rightH
        });
        y += rowHeight + PAGE_GAP;
        i += 2;
      } else {
        const x = Math.max(0, (container.width - leftW) / 2);
        positions.push({ pageIndex: i, x, y, width: leftW, height: leftH });
        y += leftH + PAGE_GAP;
        i += 1;
      }
    }
  }
  return positions;
}
var nextViewportId = 0;
function makeViewportId() {
  return `vp-${++nextViewportId}`;
}
var ViewportInstanceImpl = class {
  id;
  // ── State ──
  #containerSize$ = signal2({ width: 0, height: 0 });
  #scrollOffset$ = signal2({ x: 0, y: 0 });
  #layoutMode$;
  #scale$;
  #pinnedDocId$ = signal2(null);
  #bufferSize$ = signal2(2);
  // ── Observer / listener handles ──
  #resizeObserver = null;
  #scrollListener = null;
  #attachedContainer = null;
  #resizeObserverPaused = false;
  #deferredContainerSize = null;
  #destroyed = false;
  /** True while a programmatic scrollToPage is in progress. */
  #programmaticScroll = false;
  // ── Derived ──
  #docId$;
  #handle$;
  #pagePositions$;
  #totalHeight$;
  #visiblePages$;
  #ctx;
  #pageGap;
  #viewportPad;
  constructor(id, options, ctx) {
    this.id = id;
    this.#ctx = ctx;
    this.#pageGap = options.pageGap ?? DEFAULT_PAGE_GAP;
    this.#viewportPad = options.viewportPadding ?? DEFAULT_VIEWPORT_PAD;
    this.#layoutMode$ = signal2(options.layoutMode ?? "continuous");
    this.#scale$ = signal2(options.scale ?? 1);
    if (options.docId !== void 0) {
      this.#pinnedDocId$.value = options.docId;
    }
    this.#docId$ = computed2(() => {
      const pinned = this.#pinnedDocId$.value;
      if (pinned !== null) return pinned;
      const active = ctx.document.activeDocument.value;
      return active?.id ?? null;
    });
    this.#handle$ = computed2(() => {
      const active = ctx.document.activeDocument.value;
      const pinned = this.#pinnedDocId$.value;
      if (pinned !== null) {
        const handle = ctx.document.getHandle(pinned);
        return handle ?? null;
      }
      return active;
    });
    this.#pagePositions$ = computed2(() => {
      const handle = this.#handle$.value;
      if (handle === null) return [];
      return computePagePositions(
        handle,
        this.#scale$.value,
        this.#containerSize$.value,
        this.#layoutMode$.value,
        this.#pageGap,
        this.#viewportPad
      );
    });
    this.#totalHeight$ = computed2(() => {
      const positions = this.#pagePositions$.value;
      if (positions.length === 0) return 0;
      const last = positions[positions.length - 1];
      return last.y + last.height + this.#viewportPad;
    });
    this.#visiblePages$ = computed2(() => {
      const positions = this.#pagePositions$.value;
      const offset = this.#scrollOffset$.value;
      const container = this.#containerSize$.value;
      const viewTop = offset.y;
      const viewBottom = viewTop + container.height;
      const visible = [];
      for (const pos of positions) {
        const pageTop = pos.y;
        const pageBottom = pos.y + pos.height;
        if (pageBottom > viewTop && pageTop < viewBottom) {
          visible.push(pos.pageIndex);
        }
      }
      return visible;
    });
  }
  // ── Public reactive accessors ──
  get container() {
    return this.#attachedContainer;
  }
  get docId() {
    return this.#docId$;
  }
  get containerSize() {
    return this.#containerSize$;
  }
  get scrollOffset() {
    return this.#scrollOffset$;
  }
  get scale() {
    return this.#scale$;
  }
  get layoutMode() {
    return this.#layoutMode$;
  }
  get pagePositions() {
    return this.#pagePositions$;
  }
  get totalHeight() {
    return this.#totalHeight$;
  }
  get visiblePages() {
    return this.#visiblePages$;
  }
  // ── Mutating methods ──
  attach(container) {
    if (this.#destroyed) throw new Error(`Viewport ${this.id} has been destroyed`);
    if (this.#attachedContainer !== null) {
      this.detach();
    }
    this.#attachedContainer = container;
    this.#resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (this.#resizeObserverPaused) {
          this.#deferredContainerSize = { width, height };
        } else {
          this.#containerSize$.value = { width, height };
        }
      }
    });
    this.#resizeObserver.observe(container);
    this.#containerSize$.value = {
      width: container.clientWidth,
      height: container.clientHeight
    };
    this.#scrollListener = (e) => {
      if (this.#programmaticScroll) return;
      const target = e.target;
      this.#scrollOffset$.value = { x: target.scrollLeft, y: target.scrollTop };
    };
    container.addEventListener("scroll", this.#scrollListener, { passive: true });
    this.#scrollOffset$.value = {
      x: container.scrollLeft,
      y: container.scrollTop
    };
  }
  detach() {
    if (this.#resizeObserver !== null) {
      this.#resizeObserver.disconnect();
      this.#resizeObserver = null;
    }
    if (this.#attachedContainer !== null && this.#scrollListener !== null) {
      this.#attachedContainer.removeEventListener("scroll", this.#scrollListener);
      this.#scrollListener = null;
    }
    this.#attachedContainer = null;
    this.#containerSize$.value = { width: 0, height: 0 };
    this.#scrollOffset$.value = { x: 0, y: 0 };
  }
  setDocument(docId) {
    this.#pinnedDocId$.value = docId;
    this.#scrollOffset$.value = { x: 0, y: 0 };
    if (this.#attachedContainer !== null) {
      this.#attachedContainer.scrollTop = 0;
      this.#attachedContainer.scrollLeft = 0;
    }
  }
  setScale(scale) {
    this.#scale$.value = scale;
  }
  setLayoutMode(mode) {
    this.#layoutMode$.value = mode;
  }
  scrollToPage(pageIndex, smooth = true) {
    const positions = this.#pagePositions$.peek();
    const target = positions.find((p) => p.pageIndex === pageIndex);
    if (target === void 0) return;
    const scrollY = Math.max(0, target.y - this.#pageGap);
    this.#programmaticScroll = true;
    if (this.#attachedContainer !== null) {
      this.#attachedContainer.scrollTo({
        top: scrollY,
        left: 0,
        behavior: smooth ? "smooth" : "instant"
      });
    }
    this.#scrollOffset$.value = { x: 0, y: scrollY };
    const delay = smooth ? 600 : 0;
    if (delay === 0) {
      queueMicrotask(() => {
        this.#programmaticScroll = false;
      });
    } else {
      setTimeout(() => {
        this.#programmaticScroll = false;
      }, delay);
    }
  }
  setResizeObserverPaused(paused) {
    if (this.#resizeObserverPaused === paused) return;
    this.#resizeObserverPaused = paused;
    if (!paused && this.#deferredContainerSize !== null) {
      this.#containerSize$.value = this.#deferredContainerSize;
      this.#deferredContainerSize = null;
    }
  }
  destroy() {
    if (this.#destroyed) return;
    this.detach();
    this.#destroyed = true;
    this.#ctx.onDestroy(this.id);
  }
  /** Internal: read the current buffer size. */
  getBufferSize() {
    return this.#bufferSize$.peek();
  }
  /** Internal: set the buffer size. */
  setBufferSize(pages) {
    this.#bufferSize$.value = Math.max(0, Math.floor(pages));
  }
  /** Internal: notify document change for scroll reset (legacy behavior). */
  resetScroll() {
    this.#scrollOffset$.value = { x: 0, y: 0 };
    if (this.#attachedContainer !== null) {
      this.#attachedContainer.scrollTop = 0;
      this.#attachedContainer.scrollLeft = 0;
    }
  }
};
var viewportPlugin = definePlugin({
  id: "viewport",
  provides: ["viewport"],
  requires: ["document", "render"],
  optional: [],
  setup(ctx) {
    const PAGE_GAP = ctx.engine.pageGap ?? DEFAULT_PAGE_GAP;
    const VIEWPORT_PAD = ctx.engine.viewportPadding ?? DEFAULT_VIEWPORT_PAD;
    const documentCap = ctx.require("document");
    ctx.require("render");
    const viewports$ = signal2([]);
    const activeViewportId$ = signal2(null);
    let primaryViewport = null;
    function getOrCreatePrimary() {
      if (primaryViewport === null) {
        primaryViewport = createInstance({});
      }
      return primaryViewport;
    }
    function createInstance(options) {
      const id = options.id ?? makeViewportId();
      const mergedOptions = {
        pageGap: PAGE_GAP,
        viewportPadding: VIEWPORT_PAD,
        ...options
      };
      const instance = new ViewportInstanceImpl(id, mergedOptions, {
        document: documentCap,
        onDestroy: (destroyedId) => {
          viewports$.value = viewports$.peek().filter((vp) => vp.id !== destroyedId);
          if (activeViewportId$.peek() === destroyedId) {
            const remaining = viewports$.peek();
            activeViewportId$.value = remaining[0]?.id ?? null;
          }
          if (primaryViewport?.id === destroyedId) {
            const remaining = viewports$.peek();
            primaryViewport = remaining[0] ?? null;
          }
        }
      });
      viewports$.value = [...viewports$.peek(), instance];
      if (primaryViewport === null) {
        primaryViewport = instance;
      }
      if (activeViewportId$.peek() === null) {
        activeViewportId$.value = id;
      }
      ctx.emit("viewport:instance-created", instance);
      return instance;
    }
    const activeViewport$ = computed2(() => {
      const id = activeViewportId$.value;
      if (id === null) return null;
      return viewports$.value.find((vp) => vp.id === id) ?? null;
    });
    ctx.on("document:loaded", () => {
      for (const vp of viewports$.peek()) {
        vp.resetScroll();
      }
    });
    function readPrimary(read, fallback) {
      const vp = primaryViewport;
      if (vp === null) return fallback;
      return read(vp);
    }
    const visiblePages$ = computed2(() => {
      void viewports$.value;
      const vp = primaryViewport;
      if (vp === null) return [];
      return vp.visiblePages.value;
    });
    const containerSize$ = computed2(() => {
      void viewports$.value;
      const vp = primaryViewport;
      if (vp === null) return { width: 0, height: 0 };
      return vp.containerSize.value;
    });
    const scrollOffset$ = computed2(() => {
      void viewports$.value;
      const vp = primaryViewport;
      if (vp === null) return { x: 0, y: 0 };
      return vp.scrollOffset.value;
    });
    const layoutMode$ = computed2(() => {
      void viewports$.value;
      const vp = primaryViewport;
      if (vp === null) return "continuous";
      return vp.layoutMode.value;
    });
    const pagePositions$ = computed2(() => {
      void viewports$.value;
      const vp = primaryViewport;
      if (vp === null) return [];
      return vp.pagePositions.value;
    });
    const totalHeight$ = computed2(() => {
      void viewports$.value;
      const vp = primaryViewport;
      if (vp === null) return 0;
      return vp.totalHeight.value;
    });
    const scale$ = computed2(() => {
      void viewports$.value;
      const vp = primaryViewport;
      if (vp === null) return 1;
      return vp.scale.value;
    });
    const capability = {
      // Multi-instance API
      createViewport(options) {
        return createInstance(options ?? {});
      },
      destroyViewport(id) {
        const vp = viewports$.peek().find((v) => v.id === id);
        if (vp) vp.destroy();
      },
      getViewport(id) {
        return viewports$.peek().find((v) => v.id === id) ?? null;
      },
      viewports: computed2(() => viewports$.value),
      activeViewport: activeViewport$,
      setActiveViewport(id) {
        if (!viewports$.peek().some((v) => v.id === id)) return;
        if (activeViewportId$.peek() === id) return;
        activeViewportId$.value = id;
        ctx.emit("viewport:active-changed", id);
      },
      // Singleton facade
      attach(container) {
        const primary = getOrCreatePrimary();
        primary.attach(container);
      },
      detach() {
        primaryViewport?.detach();
      },
      scrollToPage(pageIndex, smooth = true) {
        primaryViewport?.scrollToPage(pageIndex, smooth);
        ctx.emit("viewport:scroll-to-page", pageIndex);
      },
      setLayoutMode(mode) {
        primaryViewport?.setLayoutMode(mode);
      },
      setBufferSize(pages) {
        primaryViewport?.setBufferSize(pages);
      },
      setScale(scale) {
        primaryViewport?.setScale(scale);
      },
      setResizeObserverPaused(paused) {
        primaryViewport?.setResizeObserverPaused(paused);
      },
      visiblePages: visiblePages$,
      containerSize: containerSize$,
      scrollOffset: scrollOffset$,
      layoutMode: layoutMode$,
      pagePositions: pagePositions$,
      totalHeight: totalHeight$,
      scale: scale$
    };
    void readPrimary;
    return capability;
  },
  dispose() {
  }
});

// src/plugins/zoom-plugin.ts
import { signal as signal3, computed as computed3 } from "@truespar/lector-utils";
var DEFAULT_MIN_LEVEL = 0.1;
var DEFAULT_MAX_LEVEL = 10;
var DEFAULT_ZOOM_STEP = 1.1;
var zoomPlugin = definePlugin({
  id: "zoom",
  provides: ["zoom"],
  requires: ["viewport", "render"],
  optional: [],
  setup(ctx) {
    const viewport = ctx.require("viewport");
    ctx.require("render");
    const document2 = ctx.require("document");
    const MIN_LEVEL = ctx.engine.zoomMin ?? DEFAULT_MIN_LEVEL;
    const MAX_LEVEL = ctx.engine.zoomMax ?? DEFAULT_MAX_LEVEL;
    const ZOOM_STEP = ctx.engine.zoomStep ?? DEFAULT_ZOOM_STEP;
    const level$ = signal3(1);
    const fitMode$ = signal3("width");
    function clamp(value) {
      return Math.min(MAX_LEVEL, Math.max(MIN_LEVEL, value));
    }
    function computeFitLevel(mode) {
      if (mode === "none") {
        return null;
      }
      const handle = document2.activeDocument.value;
      if (handle === null || handle.pageCount === 0) {
        return null;
      }
      const active = viewport.activeViewport.value;
      const container = active?.containerSize.value ?? viewport.containerSize.value;
      if (container.width === 0 || container.height === 0) {
        return null;
      }
      const visiblePages = active?.visiblePages.value ?? viewport.visiblePages.value;
      const targetPageIndex = visiblePages.length > 0 ? visiblePages[0] : 0;
      const pageSize = handle.pageSizes[targetPageIndex];
      if (pageSize === void 0) {
        return null;
      }
      const VIEWPORT_PADDING = 24;
      if (mode === "width") {
        return clamp((container.width - VIEWPORT_PADDING) / pageSize.width);
      }
      const scaleW = (container.width - VIEWPORT_PADDING) / pageSize.width;
      const scaleH = (container.height - VIEWPORT_PADDING) / pageSize.height;
      return clamp(Math.min(scaleW, scaleH));
    }
    function syncViewportScale() {
      const target = level$.peek();
      const active = viewport.activeViewport.peek();
      const current = active ? active.scale.peek() : viewport.scale.peek();
      if (Math.abs(target - current) < 1e-6) return;
      if (active) active.setScale(target);
      else viewport.setScale(target);
    }
    let fitRecalculating = false;
    ctx.effect(() => {
      const mode = fitMode$.value;
      if (mode === "none" || fitRecalculating) return;
      const active = viewport.activeViewport.value;
      void (active?.containerSize.value ?? viewport.containerSize.value);
      void document2.activeDocument.value;
      const fitted = computeFitLevel(mode);
      if (fitted !== null && Math.abs(fitted - level$.peek()) > 1e-6) {
        fitRecalculating = true;
        level$.value = fitted;
        syncViewportScale();
        fitRecalculating = false;
      }
    });
    const capability = {
      level: computed3(() => level$.value),
      fitMode: computed3(() => fitMode$.value),
      setLevel(factor) {
        fitMode$.value = "none";
        level$.value = clamp(factor);
        syncViewportScale();
      },
      zoomIn() {
        fitMode$.value = "none";
        level$.value = clamp(level$.peek() * ZOOM_STEP);
        syncViewportScale();
      },
      zoomOut() {
        fitMode$.value = "none";
        level$.value = clamp(level$.peek() / ZOOM_STEP);
        syncViewportScale();
      },
      fitPage() {
        fitMode$.value = "page";
        const fitted = computeFitLevel("page");
        if (fitted !== null) {
          level$.value = fitted;
          syncViewportScale();
        }
      },
      fitWidth() {
        fitMode$.value = "width";
        const fitted = computeFitLevel("width");
        if (fitted !== null) {
          level$.value = fitted;
          syncViewportScale();
        }
      },
      resetZoom() {
        fitMode$.value = "none";
        level$.value = 1;
      }
    };
    ctx.registerCommand({
      id: "zoom.in",
      label: "Zoom In",
      shortcut: "Ctrl+=",
      category: "Zoom",
      execute: () => {
        capability.zoomIn();
      }
    });
    ctx.registerCommand({
      id: "zoom.out",
      label: "Zoom Out",
      shortcut: "Ctrl+-",
      category: "Zoom",
      execute: () => {
        capability.zoomOut();
      }
    });
    ctx.registerCommand({
      id: "zoom.fit-page",
      label: "Fit Page",
      shortcut: "Ctrl+0",
      category: "Zoom",
      execute: () => {
        capability.fitPage();
      }
    });
    ctx.registerCommand({
      id: "zoom.fit-width",
      label: "Fit Width",
      shortcut: "Ctrl+1",
      category: "Zoom",
      execute: () => {
        capability.fitWidth();
      }
    });
    ctx.registerCommand({
      id: "zoom.reset",
      label: "Actual Size (100%)",
      shortcut: "Ctrl+Shift+0",
      category: "Zoom",
      execute: () => {
        capability.setLevel(1);
      }
    });
    return capability;
  }
});

// src/plugins/interaction-plugin.ts
import { signal as signal4, computed as computed4 } from "@truespar/lector-utils";
var interactionPlugin = definePlugin({
  id: "interaction",
  provides: ["interaction"],
  requires: ["viewport"],
  optional: [],
  setup(ctx) {
    const viewport = ctx.require("viewport");
    const mode$ = signal4("pointer");
    const cursorOverride$ = signal4(null);
    const handlers = /* @__PURE__ */ new Map();
    const cursor$ = computed4(() => {
      const override = cursorOverride$.value;
      if (override !== null) return override;
      const handler = handlers.get(mode$.value);
      return handler?.cursor ?? "default";
    });
    function findViewportForContainer(container) {
      if (!container) return null;
      const viewports = viewport.viewports.peek();
      for (const vp of viewports) {
        if (vp.container === container) return vp;
      }
      return null;
    }
    function findContainerAtPoint(clientX, clientY) {
      for (const container of attachedContainers.keys()) {
        const r = container.getBoundingClientRect();
        if (clientX >= r.left && clientX <= r.right && clientY >= r.top && clientY <= r.bottom) {
          return container;
        }
      }
      return null;
    }
    function viewportToPage(clientX, clientY, container) {
      const targetContainer = container ?? findContainerAtPoint(clientX, clientY);
      if (!targetContainer) return null;
      const vp = findViewportForContainer(targetContainer);
      if (!vp) return null;
      const positions = vp.pagePositions.peek();
      const offset = vp.scrollOffset.peek();
      const scale = vp.scale.peek();
      const rect = targetContainer.getBoundingClientRect();
      const canvasX = clientX - rect.left + offset.x;
      const canvasY = clientY - rect.top + offset.y;
      let targetPos;
      for (const pos of positions) {
        if (canvasX >= pos.x && canvasX <= pos.x + pos.width && canvasY >= pos.y && canvasY <= pos.y + pos.height) {
          targetPos = pos;
          break;
        }
      }
      if (targetPos === void 0) return null;
      const pageX = (canvasX - targetPos.x) / scale;
      const pageY = (canvasY - targetPos.y) / scale;
      return {
        pageIndex: targetPos.pageIndex,
        x: pageX,
        y: pageY
      };
    }
    function makePageEvent(domEvent) {
      let clientX = 0;
      let clientY = 0;
      let container = null;
      if ("clientX" in domEvent) {
        clientX = domEvent.clientX;
        clientY = domEvent.clientY;
        container = domEvent.currentTarget instanceof HTMLElement ? domEvent.currentTarget : null;
      }
      const vp = findViewportForContainer(container);
      return {
        pagePoint: viewportToPage(clientX, clientY, container),
        clientX,
        clientY,
        domEvent,
        container,
        viewport: vp
      };
    }
    function onPointerDown(e) {
      const container = e.currentTarget instanceof HTMLElement ? e.currentTarget : null;
      if (container !== null && document.activeElement !== container) {
        container.focus({ preventScroll: true });
      }
      const vp = findViewportForContainer(container);
      if (vp) viewport.setActiveViewport(vp.id);
      const handler = handlers.get(mode$.peek());
      handler?.onPointerDown?.(makePageEvent(e));
    }
    function onPointerMove(e) {
      const handler = handlers.get(mode$.peek());
      handler?.onPointerMove?.(makePageEvent(e));
    }
    function onPointerUp(e) {
      const handler = handlers.get(mode$.peek());
      handler?.onPointerUp?.(makePageEvent(e));
    }
    function onClick(e) {
      const handler = handlers.get(mode$.peek());
      handler?.onClick?.(makePageEvent(e));
    }
    function onDblClick(e) {
      const handler = handlers.get(mode$.peek());
      handler?.onDoubleClick?.(makePageEvent(e));
    }
    function onKeyDown(e) {
      const handler = handlers.get(mode$.peek());
      handler?.onKeyDown?.(e);
    }
    function onKeyUp(e) {
      const handler = handlers.get(mode$.peek());
      handler?.onKeyUp?.(e);
    }
    const attachedContainers = /* @__PURE__ */ new Map();
    ctx.effect(() => {
      const live = /* @__PURE__ */ new Set();
      for (const vp of viewport.viewports.value) {
        if (vp.container !== null) live.add(vp.container);
      }
      for (const [container, listeners] of attachedContainers) {
        if (!live.has(container)) {
          listeners.removeAll();
          attachedContainers.delete(container);
        }
      }
    });
    const capability = {
      mode: computed4(() => mode$.value),
      cursor: cursor$,
      setMode(newMode) {
        const oldMode = mode$.peek();
        if (oldMode === newMode) return;
        const oldHandler = handlers.get(oldMode);
        oldHandler?.onDeactivate?.();
        mode$.value = newMode;
        const newHandler = handlers.get(newMode);
        newHandler?.onActivate?.();
        ctx.emit("interaction:mode-changed", newMode, oldMode);
      },
      registerHandler(mode, handler) {
        handlers.set(mode, handler);
      },
      unregisterHandler(mode) {
        handlers.delete(mode);
      },
      viewportToPage,
      setCursorOverride(cursor) {
        cursorOverride$.value = cursor;
      }
    };
    let panState = null;
    capability.registerHandler("pointer", { cursor: "default" });
    capability.registerHandler("pan", {
      cursor: "grab",
      onPointerDown(event) {
        if (event.container === null) return;
        const pe = event.domEvent;
        event.container.setPointerCapture(pe.pointerId);
        panState = {
          pointerId: pe.pointerId,
          lastX: pe.clientX,
          lastY: pe.clientY,
          container: event.container
        };
        cursorOverride$.value = "grabbing";
      },
      onPointerMove(event) {
        if (panState === null) return;
        const pe = event.domEvent;
        const dx = pe.clientX - panState.lastX;
        const dy = pe.clientY - panState.lastY;
        panState.container.scrollLeft -= dx;
        panState.container.scrollTop -= dy;
        panState.lastX = pe.clientX;
        panState.lastY = pe.clientY;
      },
      onPointerUp(event) {
        if (panState === null) return;
        const pe = event.domEvent;
        try {
          panState.container.releasePointerCapture(pe.pointerId);
        } catch {
        }
        panState = null;
        cursorOverride$.value = null;
      },
      onDeactivate() {
        panState = null;
        cursorOverride$.value = null;
      }
    });
    capability.registerHandler("text-select", { cursor: "text" });
    ctx.registerCommand({
      id: "interaction.pointer-mode",
      label: "Pointer Mode",
      category: "Tools",
      execute: () => {
        capability.setMode("pointer");
      }
    });
    ctx.registerCommand({
      id: "interaction.pan-mode",
      label: "Pan Mode",
      shortcut: "H",
      category: "Tools",
      execute: () => {
        capability.setMode("pan");
      }
    });
    ctx.registerCommand({
      id: "interaction.text-select-mode",
      label: "Text Select Mode",
      shortcut: "T",
      category: "Tools",
      execute: () => {
        capability.setMode("text-select");
      }
    });
    ctx.on("viewport:container-attached", (...args) => {
      const container = args[0];
      if (attachedContainers.has(container)) return;
      container.addEventListener("pointerdown", onPointerDown);
      container.addEventListener("pointermove", onPointerMove);
      container.addEventListener("pointerup", onPointerUp);
      container.addEventListener("click", onClick);
      container.addEventListener("dblclick", onDblClick);
      container.addEventListener("keydown", onKeyDown);
      container.addEventListener("keyup", onKeyUp);
      attachedContainers.set(container, {
        removeAll: () => {
          container.removeEventListener("pointerdown", onPointerDown);
          container.removeEventListener("pointermove", onPointerMove);
          container.removeEventListener("pointerup", onPointerUp);
          container.removeEventListener("click", onClick);
          container.removeEventListener("dblclick", onDblClick);
          container.removeEventListener("keydown", onKeyDown);
          container.removeEventListener("keyup", onKeyUp);
        }
      });
    });
    return capability;
  },
  dispose() {
  }
});

// src/utils/clipboard.ts
async function copyText(text) {
  if (navigator.clipboard) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {
    document.execCommand("copy");
  } finally {
    ta.remove();
  }
}

// src/plugins/text-layer-plugin.ts
import { signal as signal5, computed as computed5 } from "@truespar/lector-utils";
var textLayerPlugin = definePlugin({
  id: "text-layer",
  provides: ["text-layer"],
  requires: ["document", "render"],
  optional: ["interaction"],
  setup(ctx) {
    ctx.require("document");
    ctx.require("render");
    const textCache = /* @__PURE__ */ new Map();
    const charInfoCache = /* @__PURE__ */ new Map();
    function cacheKey(docId, pageIndex) {
      return `${docId}:${pageIndex}`;
    }
    function localCharIndexAtPos(chars, x, y, tolerance) {
      let bestIdx = -1;
      let bestDist = tolerance;
      for (let i = 0; i < chars.length; i++) {
        const c = chars[i];
        if (x >= c.left && x <= c.right && y >= c.bottom && y <= c.top) {
          return i;
        }
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
    function localTextRects(chars, startIdx, count) {
      if (count <= 0 || startIdx < 0 || startIdx >= chars.length) return [];
      const end = Math.min(startIdx + count, chars.length);
      const rects = [];
      let cur = chars[startIdx];
      let left = cur.left;
      let right = cur.right;
      let top = cur.top;
      let bottom = cur.bottom;
      for (let i = startIdx + 1; i < end; i++) {
        const c = chars[i];
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
      return rects;
    }
    const selection$ = signal5(null);
    ctx.on("document:closed", (...args) => {
      const docId = args[0];
      for (const key of textCache.keys()) {
        if (key.startsWith(`${docId}:`)) textCache.delete(key);
      }
      for (const key of charInfoCache.keys()) {
        if (key.startsWith(`${docId}:`)) charInfoCache.delete(key);
      }
      const sel = selection$.peek();
      if (sel !== null && sel.docId === docId) {
        selection$.value = null;
      }
    });
    const capability = {
      async getPageText(docId, pageIndex) {
        const key = cacheKey(docId, pageIndex);
        let promise = textCache.get(key);
        if (promise === void 0) {
          promise = ctx.engine.workerProxy.getPageText(docId, pageIndex).catch((err) => {
            textCache.delete(key);
            throw err;
          });
          textCache.set(key, promise);
        }
        return promise;
      },
      async getPageCharInfo(docId, pageIndex) {
        const key = cacheKey(docId, pageIndex);
        let promise = charInfoCache.get(key);
        if (promise === void 0) {
          promise = ctx.engine.workerProxy.getPageCharInfo(docId, pageIndex).catch((err) => {
            charInfoCache.delete(key);
            throw err;
          });
          charInfoCache.set(key, promise);
        }
        return promise;
      },
      async getTextRects(docId, pageIndex, charIndex, count) {
        return ctx.engine.workerProxy.getTextRects(docId, pageIndex, charIndex, count);
      },
      async getCharIndexAtPos(docId, pageIndex, x, y, tolerance = 5) {
        return ctx.engine.workerProxy.getCharIndexAtPos(docId, pageIndex, x, y, tolerance);
      },
      selection: computed5(() => selection$.value),
      setSelection(selection) {
        selection$.value = selection;
        if (selection !== null) {
          ctx.emit("text:selection-changed", selection);
        } else {
          ctx.emit("text:selection-cleared");
        }
      },
      async copySelection() {
        const sel = selection$.peek();
        if (sel === null || sel.text.length === 0) return;
        await copyText(sel.text);
        ctx.emit("text:copied", sel.text);
      },
      clearCache(docId) {
        for (const key of textCache.keys()) {
          if (key.startsWith(`${docId}:`)) textCache.delete(key);
        }
        for (const key of charInfoCache.keys()) {
          if (key.startsWith(`${docId}:`)) charInfoCache.delete(key);
        }
      }
    };
    const interaction = ctx.optional("interaction");
    if (interaction) {
      let resolveEventDocId2 = function(event) {
        const vpDocId = event.viewport?.docId.peek() ?? null;
        if (vpDocId !== null) return vpDocId;
        return document2.activeDocument.peek()?.id ?? null;
      };
      var resolveEventDocId = resolveEventDocId2;
      const document2 = ctx.require("document");
      let selectState = null;
      let selectChars = null;
      let selectText = null;
      interaction.registerHandler("text-select", {
        cursor: "text",
        onPointerDown(event) {
          const pp = event.pagePoint;
          if (!pp) {
            selectState = null;
            return;
          }
          const docId = resolveEventDocId2(event);
          if (!docId) return;
          const handle = document2.getHandle(docId);
          if (!handle) return;
          const ps = handle.pageSizes[pp.pageIndex];
          if (!ps) return;
          selectState = null;
          selectChars = null;
          selectText = null;
          selection$.value = null;
          Promise.all([
            capability.getPageCharInfo(docId, pp.pageIndex),
            capability.getPageText(docId, pp.pageIndex),
            ctx.engine.pageRotation.resolve(docId, pp.pageIndex)
          ]).then(([chars, text, rot]) => {
            selectChars = chars;
            selectText = text;
            const vp = PageViewport.fromRotatedSize(ps.width, ps.height, rot, 1);
            const { x: pdfX, y: pdfY } = vp.cssPointToPdf(pp.x, pp.y);
            const charIdx = localCharIndexAtPos(chars, pdfX, pdfY, 10);
            if (charIdx < 0) return;
            selectState = {
              docId,
              pageIndex: pp.pageIndex,
              viewport: vp,
              startCharIndex: charIdx
            };
          }).catch(() => {
          });
        },
        onPointerMove(event) {
          if (!selectState || !selectChars || !selectText) return;
          const pp = event.pagePoint;
          if (!pp || pp.pageIndex !== selectState.pageIndex) return;
          const eventDocId = resolveEventDocId2(event);
          if (eventDocId !== null && eventDocId !== selectState.docId) return;
          const { x: pdfX, y: pdfY } = selectState.viewport.cssPointToPdf(pp.x, pp.y);
          const endIdx = localCharIndexAtPos(selectChars, pdfX, pdfY, 10);
          if (endIdx < 0) return;
          const startIdx = Math.min(selectState.startCharIndex, endIdx);
          const endCharIdx = Math.max(selectState.startCharIndex, endIdx);
          const count = endCharIdx - startIdx + 1;
          if (count <= 0) return;
          const rects = localTextRects(selectChars, startIdx, count);
          capability.setSelection({
            docId: selectState.docId,
            pageIndex: selectState.pageIndex,
            startCharIndex: startIdx,
            endCharIndex: endCharIdx,
            text: selectText.substring(startIdx, startIdx + count),
            rects
          });
        },
        onPointerUp() {
          selectState = null;
          selectChars = null;
          selectText = null;
        },
        onKeyDown(event) {
          if ((event.ctrlKey || event.metaKey) && event.key === "c") {
            if (selection$.peek() !== null) {
              event.preventDefault();
              void capability.copySelection();
            }
          }
          if ((event.ctrlKey || event.metaKey) && event.key === "a") {
            event.preventDefault();
            ctx.emit("text:select-all-requested");
          }
        },
        onDeactivate() {
          selectState = null;
        }
      });
    }
    ctx.registerCommand({
      id: "text.copy",
      label: "Copy",
      shortcut: "Ctrl+C",
      category: "Edit",
      enabled: computed5(() => selection$.value !== null),
      execute: () => {
        void capability.copySelection();
      }
    });
    ctx.registerCommand({
      id: "text.select-all",
      label: "Select All",
      shortcut: "Ctrl+A",
      category: "Edit",
      execute: () => {
        ctx.emit("text:select-all-requested");
      }
    });
    return capability;
  }
});

// src/plugins/search-plugin.ts
import { signal as signal6, computed as computed6 } from "@truespar/lector-utils";
function toSearchFlags(options) {
  let flags = 0;
  if (options?.matchCase) flags |= 1;
  if (options?.matchWholeWord) flags |= 2;
  return flags;
}
var searchPlugin = definePlugin({
  id: "search",
  provides: ["search"],
  requires: ["document"],
  optional: [],
  setup(ctx) {
    ctx.require("document");
    const result$ = signal6(null);
    const activeMatchIndex$ = signal6(-1);
    const searching$ = signal6(false);
    const progress$ = signal6(null);
    let abortController = null;
    let searchDocId = null;
    ctx.on("document:closed", (...args) => {
      if (args[0] === searchDocId) clear();
    });
    function clear() {
      abortController?.abort();
      abortController = null;
      searchDocId = null;
      result$.value = null;
      activeMatchIndex$.value = -1;
      searching$.value = false;
      progress$.value = null;
      ctx.emit("search:cleared");
    }
    const capability = {
      result: computed6(() => result$.value),
      activeMatchIndex: computed6(() => activeMatchIndex$.value),
      searching: computed6(() => searching$.value),
      progress: computed6(() => progress$.value),
      async search(docId, query, options) {
        abortController?.abort();
        const controller = new AbortController();
        abortController = controller;
        searchDocId = docId;
        if (query.length === 0) {
          clear();
          return { docId, query, matches: [], totalCount: 0 };
        }
        searching$.value = true;
        result$.value = null;
        activeMatchIndex$.value = -1;
        const flags = toSearchFlags(options);
        const proxy = ctx.engine.workerProxy;
        const pageCount = await proxy.getPageCount(docId);
        const allMatches = [];
        progress$.value = { pagesSearched: 0, totalPages: pageCount, matchesSoFar: 0 };
        try {
          for (let i = 0; i < pageCount; i++) {
            if (controller.signal.aborted) break;
            const pageMatches = await proxy.searchPage(docId, i, query, flags);
            allMatches.push(...pageMatches);
            progress$.value = {
              pagesSearched: i + 1,
              totalPages: pageCount,
              matchesSoFar: allMatches.length
            };
            ctx.emit("search:progress", progress$.peek());
          }
        } catch (err) {
          if (controller.signal.aborted) {
            return { docId, query, matches: [], totalCount: 0 };
          }
          throw err;
        }
        if (controller.signal.aborted) {
          return { docId, query, matches: [], totalCount: 0 };
        }
        const searchResult = {
          docId,
          query,
          matches: allMatches,
          totalCount: allMatches.length
        };
        result$.value = searchResult;
        searching$.value = false;
        progress$.value = null;
        if (allMatches.length > 0) {
          activeMatchIndex$.value = 0;
        }
        ctx.emit("search:completed", searchResult);
        return searchResult;
      },
      nextMatch() {
        const res = result$.peek();
        if (res === null || res.totalCount === 0) return;
        const next = (activeMatchIndex$.peek() + 1) % res.totalCount;
        activeMatchIndex$.value = next;
        ctx.emit("search:match-changed", next, res.matches[next]);
      },
      previousMatch() {
        const res = result$.peek();
        if (res === null || res.totalCount === 0) return;
        const current = activeMatchIndex$.peek();
        const prev = current <= 0 ? res.totalCount - 1 : current - 1;
        activeMatchIndex$.value = prev;
        ctx.emit("search:match-changed", prev, res.matches[prev]);
      },
      goToMatch(index) {
        const res = result$.peek();
        if (res === null || index < 0 || index >= res.totalCount) return;
        activeMatchIndex$.value = index;
        ctx.emit("search:match-changed", index, res.matches[index]);
      },
      clear
    };
    ctx.registerCommand({
      id: "search.open",
      label: "Find",
      shortcut: "Ctrl+F",
      category: "Search",
      execute: () => {
        ctx.emit("search:open");
      }
    });
    ctx.registerCommand({
      id: "search.next",
      label: "Next Match",
      shortcut: "F3",
      category: "Search",
      enabled: computed6(() => {
        const res = result$.value;
        return res !== null && res.totalCount > 0;
      }),
      execute: () => {
        capability.nextMatch();
      }
    });
    ctx.registerCommand({
      id: "search.previous",
      label: "Previous Match",
      shortcut: "Shift+F3",
      category: "Search",
      enabled: computed6(() => {
        const res = result$.value;
        return res !== null && res.totalCount > 0;
      }),
      execute: () => {
        capability.previousMatch();
      }
    });
    ctx.registerCommand({
      id: "search.clear",
      label: "Clear Search",
      shortcut: "Escape",
      category: "Search",
      execute: () => {
        capability.clear();
      }
    });
    return capability;
  }
});

// src/plugins/navigation-plugin.ts
import { signal as signal7, computed as computed7 } from "@truespar/lector-utils";
var navigationPlugin = definePlugin({
  id: "navigation",
  provides: ["navigation"],
  requires: ["document", "viewport"],
  optional: [],
  setup(ctx) {
    const document2 = ctx.require("document");
    const viewport = ctx.require("viewport");
    const bookmarkCache = /* @__PURE__ */ new Map();
    const linkCache = /* @__PURE__ */ new Map();
    const webLinkCache = /* @__PURE__ */ new Map();
    function pageCacheKey(docId, pageIndex) {
      return `${docId}:${pageIndex}`;
    }
    const historyStack = [];
    let historyIndex = -1;
    const canGoBack$ = signal7(false);
    const canGoForward$ = signal7(false);
    function pushHistory(pageIndex) {
      if (historyIndex >= 0 && historyStack[historyIndex] === pageIndex) return;
      historyStack.length = historyIndex + 1;
      historyStack.push(pageIndex);
      historyIndex = historyStack.length - 1;
      canGoBack$.value = historyIndex > 0;
      canGoForward$.value = false;
    }
    const currentPage$ = computed7(() => {
      const active = viewport.activeViewport.value;
      const visible = active?.visiblePages.value ?? viewport.visiblePages.value;
      return visible.length > 0 ? visible[0] : 0;
    });
    let lastTrackedPage = -1;
    currentPage$.subscribe((page) => {
      if (page !== lastTrackedPage) {
        lastTrackedPage = page;
        queueMicrotask(() => pushHistory(page));
      }
    });
    ctx.on("document:closed", (...args) => {
      const docId = args[0];
      bookmarkCache.delete(docId);
      for (const key of linkCache.keys()) {
        if (key.startsWith(`${docId}:`)) linkCache.delete(key);
      }
      for (const key of webLinkCache.keys()) {
        if (key.startsWith(`${docId}:`)) webLinkCache.delete(key);
      }
    });
    const capability = {
      async getBookmarks(docId) {
        let promise = bookmarkCache.get(docId);
        if (promise === void 0) {
          promise = ctx.engine.workerProxy.getBookmarks(docId).catch((err) => {
            bookmarkCache.delete(docId);
            throw err;
          });
          bookmarkCache.set(docId, promise);
        }
        return promise;
      },
      async getPageLinks(docId, pageIndex) {
        const key = pageCacheKey(docId, pageIndex);
        let promise = linkCache.get(key);
        if (promise === void 0) {
          promise = ctx.engine.workerProxy.getPageLinks(docId, pageIndex).catch((err) => {
            linkCache.delete(key);
            throw err;
          });
          linkCache.set(key, promise);
        }
        return promise;
      },
      async getPageWebLinks(docId, pageIndex) {
        const key = pageCacheKey(docId, pageIndex);
        let promise = webLinkCache.get(key);
        if (promise === void 0) {
          promise = ctx.engine.workerProxy.getPageWebLinks(docId, pageIndex).catch((err) => {
            webLinkCache.delete(key);
            throw err;
          });
          webLinkCache.set(key, promise);
        }
        return promise;
      },
      navigateToTarget(target) {
        switch (target.type) {
          case "goto": {
            const dest = target.destination;
            capability.goToPage(dest.pageIndex);
            ctx.emit("navigation:goto", dest);
            break;
          }
          case "uri": {
            ctx.emit("navigation:external-link", target.uri);
            break;
          }
          case "remote-goto": {
            ctx.emit("navigation:remote-goto", target.filePath, target.destination);
            break;
          }
          case "launch": {
            ctx.emit("navigation:launch", target.filePath);
            break;
          }
          case "unknown":
            break;
        }
      },
      goToPage(pageIndex) {
        const handle = document2.activeDocument.peek();
        if (handle === null) return;
        const clamped = Math.max(0, Math.min(pageIndex, handle.pageCount - 1));
        const target = viewport.activeViewport.peek();
        if (target) target.scrollToPage(clamped);
        else viewport.scrollToPage(clamped);
        ctx.emit("navigation:page-changed", clamped);
      },
      goForward() {
        if (historyIndex < historyStack.length - 1) {
          historyIndex++;
          const page = historyStack[historyIndex];
          const target = viewport.activeViewport.peek();
          if (target) target.scrollToPage(page);
          else viewport.scrollToPage(page);
          canGoBack$.value = historyIndex > 0;
          canGoForward$.value = historyIndex < historyStack.length - 1;
        }
      },
      goBack() {
        if (historyIndex > 0) {
          historyIndex--;
          const page = historyStack[historyIndex];
          const target = viewport.activeViewport.peek();
          if (target) target.scrollToPage(page);
          else viewport.scrollToPage(page);
          canGoBack$.value = historyIndex > 0;
          canGoForward$.value = historyIndex < historyStack.length - 1;
        }
      },
      canGoBack: computed7(() => canGoBack$.value),
      canGoForward: computed7(() => canGoForward$.value),
      currentPage: currentPage$,
      clearCache(docId) {
        bookmarkCache.delete(docId);
        for (const key of linkCache.keys()) {
          if (key.startsWith(`${docId}:`)) linkCache.delete(key);
        }
        for (const key of webLinkCache.keys()) {
          if (key.startsWith(`${docId}:`)) webLinkCache.delete(key);
        }
      }
    };
    ctx.registerCommand({
      id: "navigation.next-page",
      label: "Next Page",
      shortcut: "PageDown",
      category: "Navigation",
      execute: () => {
        const page = currentPage$.peek();
        capability.goToPage(page + 1);
      }
    });
    ctx.registerCommand({
      id: "navigation.previous-page",
      label: "Previous Page",
      shortcut: "PageUp",
      category: "Navigation",
      execute: () => {
        const page = currentPage$.peek();
        capability.goToPage(page - 1);
      }
    });
    ctx.registerCommand({
      id: "navigation.first-page",
      label: "First Page",
      shortcut: "Home",
      category: "Navigation",
      execute: () => {
        capability.goToPage(0);
      }
    });
    ctx.registerCommand({
      id: "navigation.last-page",
      label: "Last Page",
      shortcut: "End",
      category: "Navigation",
      execute: () => {
        const handle = document2.activeDocument.peek();
        if (handle !== null) {
          capability.goToPage(handle.pageCount - 1);
        }
      }
    });
    ctx.registerCommand({
      id: "navigation.go-back",
      label: "Go Back",
      shortcut: "Alt+ArrowLeft",
      category: "Navigation",
      enabled: capability.canGoBack,
      execute: () => {
        capability.goBack();
      }
    });
    ctx.registerCommand({
      id: "navigation.go-forward",
      label: "Go Forward",
      shortcut: "Alt+ArrowRight",
      category: "Navigation",
      enabled: capability.canGoForward,
      execute: () => {
        capability.goForward();
      }
    });
    return capability;
  }
});

// src/plugins/annotation-plugin.ts
import { signal as signal9, computed as computed9 } from "@truespar/lector-utils";

// src/data/operation-log.ts
var OperationLog = class {
  #events = [];
  #subscribers = /* @__PURE__ */ new Set();
  #disposed = false;
  /** Append an event to the log. Notifies all subscribers synchronously. */
  append(event) {
    if (this.#disposed) return;
    this.#events.push(event);
    for (const fn of [...this.#subscribers]) {
      fn(event);
    }
  }
  /** Subscribe to new events. Returns an unsubscribe function. */
  subscribe(fn) {
    this.#subscribers.add(fn);
    return () => {
      this.#subscribers.delete(fn);
    };
  }
  /** Get all events (shallow copy -- event objects themselves are readonly). */
  getAll() {
    return [...this.#events];
  }
  /** Get events for a specific document. */
  getForDocument(documentId) {
    return this.#events.filter((e) => e.documentId === documentId);
  }
  /**
   * Get events appended after the event with the given operation ID.
   *
   * If the operation ID is not found the full log is returned, which is the
   * safe default for an initial sync.
   */
  getSince(operationId) {
    const index = this.#events.findIndex((e) => e.operationId === operationId);
    if (index === -1) {
      return [...this.#events];
    }
    return this.#events.slice(index + 1);
  }
  /** Get the total number of events in the log. */
  get size() {
    return this.#events.length;
  }
  /** Remove all events for a document (typically when the document is closed). */
  clearDocument(documentId) {
    this.#events = this.#events.filter((e) => e.documentId !== documentId);
  }
  /** Dispose: clear all events and subscribers. */
  [Symbol.dispose]() {
    this.#disposed = true;
    this.#events = [];
    this.#subscribers.clear();
  }
};

// src/data/dirty-tracker.ts
import { signal as signal8, computed as computed8 } from "@truespar/lector-utils";
var DirtyTracker = class {
  #objects = /* @__PURE__ */ new Map();
  /**
   * Internal counter signal: incremented when an object becomes dirty
   * (new / dirty / deleted), decremented when synced or removed.
   * The `hasDirty` computed signal derives from this.
   */
  #dirtyCount = signal8(0);
  /** Reactive signal that is `true` when at least one object is dirty. */
  hasDirty = computed8(() => this.#dirtyCount.value > 0);
  /** Track a new object. */
  add(id, data, state = "new") {
    const prev = this.#objects.get(id);
    if (prev !== void 0 && prev.commitState.peek() !== "synced") {
      this.#dirtyCount.update((n) => Math.max(0, n - 1));
    }
    const commitState = signal8(state);
    this.#objects.set(id, { data, commitState });
    if (state !== "synced") {
      this.#dirtyCount.update((n) => n + 1);
    }
    return this.#toTracked(id, data, commitState);
  }
  /** Update a tracked object's data. Sets commitState to 'dirty' if currently 'synced'. */
  update(id, data) {
    const entry = this.#objects.get(id);
    if (entry === void 0) return;
    const wasSynced = entry.commitState.peek() === "synced";
    entry.data = data;
    if (wasSynced) {
      entry.commitState.value = "dirty";
      this.#dirtyCount.update((n) => n + 1);
    }
  }
  /** Mark an object as deleted. */
  delete(id) {
    const entry = this.#objects.get(id);
    if (entry === void 0) return;
    const wasSynced = entry.commitState.peek() === "synced";
    entry.commitState.value = "deleted";
    if (wasSynced) {
      this.#dirtyCount.update((n) => n + 1);
    }
  }
  /** Mark an object as synced (consumer has persisted it). */
  markSynced(id) {
    const entry = this.#objects.get(id);
    if (entry === void 0) return;
    const wasDirty = entry.commitState.peek() !== "synced";
    entry.commitState.value = "synced";
    if (wasDirty) {
      this.#dirtyCount.update((n) => Math.max(0, n - 1));
    }
  }
  /** Mark all tracked objects as synced. */
  markAllSynced() {
    let remainingDirty = 0;
    for (const [, entry] of this.#objects) {
      if (entry.commitState.peek() === "deleted") {
        remainingDirty++;
        continue;
      }
      entry.commitState.value = "synced";
    }
    this.#dirtyCount.value = remainingDirty;
  }
  /** Get a tracked object by ID. */
  get(id) {
    const entry = this.#objects.get(id);
    if (entry === void 0) return void 0;
    return this.#toTracked(id, entry.data, entry.commitState);
  }
  /** Get all objects with a specific commit state. */
  getByState(state) {
    const result = [];
    for (const [id, entry] of this.#objects) {
      if (entry.commitState.peek() === state) {
        result.push(this.#toTracked(id, entry.data, entry.commitState));
      }
    }
    return result;
  }
  /** Get all dirty objects (new + dirty + deleted). */
  getDirty() {
    const result = [];
    for (const [id, entry] of this.#objects) {
      const s = entry.commitState.peek();
      if (s === "new" || s === "dirty" || s === "deleted") {
        result.push(this.#toTracked(id, entry.data, entry.commitState));
      }
    }
    return result;
  }
  /** Remove an object completely (after deletion is confirmed by the server). */
  remove(id) {
    const entry = this.#objects.get(id);
    if (entry === void 0) return;
    const wasDirty = entry.commitState.peek() !== "synced";
    this.#objects.delete(id);
    if (wasDirty) {
      this.#dirtyCount.update((n) => Math.max(0, n - 1));
    }
  }
  /** Clear all tracked objects. */
  clear() {
    this.#objects.clear();
    this.#dirtyCount.value = 0;
  }
  /** Dispose: clear all tracked objects. */
  [Symbol.dispose]() {
    this.clear();
  }
  /** Build a {@link TrackedObject} view over an internal entry. */
  #toTracked(id, data, commitState) {
    return {
      id,
      data,
      commitState,
      markSynced: () => {
        this.markSynced(id);
      }
    };
  }
};

// src/data/annotation-store.ts
var AnnotationStore = class {
  #log;
  #trackers = /* @__PURE__ */ new Map();
  #eventBus;
  #userId;
  /** Page-index lookup: documentId -> pageIndex -> Set<annotationId> */
  #pageIndex = /* @__PURE__ */ new Map();
  constructor(eventBus, userId) {
    this.#eventBus = eventBus;
    this.#userId = userId;
    this.#log = new OperationLog();
  }
  /**
   * Load annotations for a page from the worker result.
   *
   * Existing annotations for the same page are replaced. Each loaded
   * annotation starts in the 'synced' commit state because it already
   * exists in the PDF.
   */
  loadPage(documentId, pageIndex, annotations) {
    const tracker = this.#ensureTracker(documentId);
    const pageMap = this.#ensurePageMap(documentId);
    const existing = pageMap.get(pageIndex);
    if (existing !== void 0) {
      for (const id of existing) {
        tracker.remove(id);
      }
    }
    const ids = /* @__PURE__ */ new Set();
    for (const annotation of annotations) {
      tracker.add(annotation.id, annotation, "synced");
      ids.add(annotation.id);
    }
    pageMap.set(pageIndex, ids);
  }
  /** Record a new annotation (from user action or API). */
  create(documentId, annotation) {
    const tracker = this.#ensureTracker(documentId);
    const tracked = tracker.add(annotation.id, annotation, "new");
    const pageMap = this.#ensurePageMap(documentId);
    let ids = pageMap.get(annotation.pageIndex);
    if (ids === void 0) {
      ids = /* @__PURE__ */ new Set();
      pageMap.set(annotation.pageIndex, ids);
    }
    ids.add(annotation.id);
    const event = this.#buildEvent("created", documentId, annotation.pageIndex, annotation.id, annotation);
    this.#log.append(event);
    this.#eventBus.emit("annotation:created", event);
    return tracked;
  }
  /** Record an annotation update. */
  update(documentId, annotationId, patch) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return;
    const existing = tracker.get(annotationId);
    if (existing === void 0) return;
    const oldPage = existing.data.pageIndex;
    const merged = { ...existing.data, ...patch };
    tracker.update(annotationId, merged);
    if (merged.pageIndex !== oldPage) {
      const pageMap = this.#pageIndex.get(documentId);
      if (pageMap) {
        pageMap.get(oldPage)?.delete(annotationId);
        let ids = pageMap.get(merged.pageIndex);
        if (ids === void 0) {
          ids = /* @__PURE__ */ new Set();
          pageMap.set(merged.pageIndex, ids);
        }
        ids.add(annotationId);
      }
    }
    const event = this.#buildEvent("updated", documentId, merged.pageIndex, annotationId, merged, patch);
    this.#log.append(event);
    this.#eventBus.emit("annotation:updated", event);
  }
  /** Record an annotation deletion. */
  delete(documentId, annotationId) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return;
    const existing = tracker.get(annotationId);
    if (existing === void 0) return;
    tracker.delete(annotationId);
    const pageMap = this.#pageIndex.get(documentId);
    if (pageMap) {
      const ids = pageMap.get(existing.data.pageIndex);
      if (ids) ids.delete(annotationId);
    }
    const event = this.#buildEvent("deleted", documentId, existing.data.pageIndex, annotationId, existing.data);
    this.#log.append(event);
    this.#eventBus.emit("annotation:deleted", event);
  }
  /** Mark a single annotation as synced. */
  markSynced(documentId, annotationId) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return;
    tracker.markSynced(annotationId);
  }
  /** Mark all annotations for a document as synced. */
  markAllSynced(documentId) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return;
    tracker.markAllSynced();
  }
  /** Get all annotations for a document. */
  getForDocument(documentId) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return [];
    const result = [];
    const pageMap = this.#pageIndex.get(documentId);
    if (pageMap === void 0) return [];
    for (const [, ids] of pageMap) {
      for (const id of ids) {
        const tracked = tracker.get(id);
        if (tracked !== void 0) {
          result.push(tracked);
        }
      }
    }
    return result;
  }
  /** Get annotations for a specific page. */
  getForPage(documentId, pageIndex) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return [];
    const pageMap = this.#pageIndex.get(documentId);
    if (pageMap === void 0) return [];
    const ids = pageMap.get(pageIndex);
    if (ids === void 0) return [];
    const result = [];
    for (const id of ids) {
      const tracked = tracker.get(id);
      if (tracked !== void 0) {
        result.push(tracked);
      }
    }
    return result;
  }
  /** Get dirty annotations for a document. */
  getDirty(documentId) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return [];
    return tracker.getDirty();
  }
  /** Check if a document has unsaved annotation changes. */
  hasDirty(documentId) {
    const tracker = this.#ensureTracker(documentId);
    return tracker.hasDirty;
  }
  /** Subscribe to annotation events from the operation log. */
  subscribe(fn) {
    return this.#log.subscribe(fn);
  }
  /** Get the operation log (for sync/replay). */
  get log() {
    return this.#log;
  }
  /** Clean up all state for a document. */
  clearDocument(documentId) {
    const tracker = this.#trackers.get(documentId);
    if (tracker !== void 0) {
      tracker[Symbol.dispose]();
      this.#trackers.delete(documentId);
    }
    this.#pageIndex.delete(documentId);
    this.#log.clearDocument(documentId);
  }
  /** Dispose: clean up all documents. */
  [Symbol.dispose]() {
    for (const [, tracker] of this.#trackers) {
      tracker[Symbol.dispose]();
    }
    this.#trackers.clear();
    this.#pageIndex.clear();
    this.#log[Symbol.dispose]();
  }
  // ── Private helpers ──
  #ensureTracker(documentId) {
    let tracker = this.#trackers.get(documentId);
    if (tracker === void 0) {
      tracker = new DirtyTracker();
      this.#trackers.set(documentId, tracker);
    }
    return tracker;
  }
  #ensurePageMap(documentId) {
    let pageMap = this.#pageIndex.get(documentId);
    if (pageMap === void 0) {
      pageMap = /* @__PURE__ */ new Map();
      this.#pageIndex.set(documentId, pageMap);
    }
    return pageMap;
  }
  #buildEvent(type, documentId, pageIndex, objectId, data, patch) {
    return {
      type,
      documentId,
      pageIndex,
      objectId,
      data,
      patch,
      timestamp: Date.now(),
      operationId: uuid(),
      userId: this.#userId
    };
  }
};

// src/plugins/annotation-plugin.ts
var annotationPlugin = definePlugin({
  id: "annotation",
  provides: ["annotation"],
  requires: ["document", "render", "viewport"],
  optional: ["interaction", "text-layer", "history", "formatting"],
  setup(ctx) {
    const document2 = ctx.require("document");
    ctx.require("render");
    const viewport = ctx.require("viewport");
    const interaction = ctx.optional("interaction");
    const textLayer = ctx.optional("text-layer");
    const history = ctx.optional("history");
    const formatting = ctx.optional("formatting");
    const store = new AnnotationStore(ctx.engine.plugins.events, ctx.engine.user?.id);
    const lockMode$ = signal9("none");
    const selectedAnnotation$ = signal9(null);
    const selectedAnnotations$ = signal9([]);
    function userCanAccess(annot, allowList) {
      const userId = ctx.engine.user?.id;
      if (!userId) return true;
      if (annot.authorId === userId) return true;
      if (allowList && allowList.length > 0) return allowList.includes(userId);
      if (allowList && allowList.length === 0) return false;
      return true;
    }
    const activeTool$ = signal9(null);
    const defaults = ctx.engine.annotationDefaults;
    const toolStyle$ = signal9({
      color: defaults?.color ?? { r: 228, g: 66, b: 52, a: 255 },
      interiorColor: null,
      borderWidth: defaults?.borderWidth ?? 2,
      fontSize: defaults?.fontSize ?? 14,
      opacity: defaults?.opacity ?? 1
    });
    const loadedPages = /* @__PURE__ */ new Map();
    const loadingPages = /* @__PURE__ */ new Map();
    function markPageLoaded(docId, pageIndex) {
      let pages = loadedPages.get(docId);
      if (pages === void 0) {
        pages = /* @__PURE__ */ new Set();
        loadedPages.set(docId, pages);
      }
      pages.add(pageIndex);
    }
    const annotIndexMap = /* @__PURE__ */ new Map();
    let stagedImage = null;
    ctx.on("document:closed", (...args) => {
      const docId = args[0];
      store.clearDocument(docId);
      loadedPages.delete(docId);
      for (const key of annotIndexMap.keys()) {
        if (key.startsWith(`${docId}:`)) {
          annotIndexMap.delete(key);
        }
      }
      if (selectedAnnotation$.peek() !== null) {
        selectedAnnotation$.value = null;
      }
      if (selectedAnnotations$.peek().length > 0) {
        selectedAnnotations$.value = [];
      }
    });
    const capability = {
      store,
      async loadPage(docId, pageIndex) {
        if (loadedPages.get(docId)?.has(pageIndex)) return;
        const key = `${docId}:${pageIndex}`;
        const existing = loadingPages.get(key);
        if (existing) return existing;
        const load = (async () => {
          const annotations = await ctx.engine.workerProxy.getAnnotations(docId, pageIndex);
          annotations.forEach((annot, index) => {
            annotIndexMap.set(`${docId}:${annot.id}`, { pageIndex, annotIndex: index });
          });
          store.loadPage(docId, pageIndex, annotations);
          markPageLoaded(docId, pageIndex);
          ctx.emit("annotation:page-loaded", docId, pageIndex, annotations.length);
        })();
        loadingPages.set(key, load);
        try {
          await load;
        } finally {
          loadingPages.delete(key);
        }
      },
      async reloadPage(docId, pageIndex) {
        for (const [key, val] of annotIndexMap.entries()) {
          if (key.startsWith(`${docId}:`) && val.pageIndex === pageIndex) {
            annotIndexMap.delete(key);
          }
        }
        loadedPages.get(docId)?.delete(pageIndex);
        loadingPages.delete(`${docId}:${pageIndex}`);
        await capability.loadPage(docId, pageIndex);
      },
      async create(docId, pageIndex, data) {
        const mode = lockMode$.peek();
        if (mode === "read-only" || mode === "no-create") {
          throw new Error(`Cannot create annotations in '${mode}' lock mode`);
        }
        const user = ctx.engine.user;
        const now = (/* @__PURE__ */ new Date()).toISOString();
        const enriched = {
          ...data,
          author: data.author ?? user?.name,
          authorId: data.authorId ?? user?.id,
          createdDate: data.createdDate ?? now,
          modifiedDate: now
        };
        let created = await ctx.engine.workerProxy.createAnnotation(docId, pageIndex, enriched);
        if (enriched.line) created = { ...created, line: enriched.line };
        if (enriched.measurement) created = { ...created, measurement: enriched.measurement };
        if (enriched.ink && enriched.ink.strokes.some(
          (s) => s.some((p) => p.pressure !== void 0)
        )) {
          created = { ...created, ink: enriched.ink };
        }
        const allAnnotations = await ctx.engine.workerProxy.getAnnotations(docId, pageIndex);
        const newIndex = allAnnotations.length - 1;
        annotIndexMap.set(`${docId}:${created.id}`, { pageIndex, annotIndex: newIndex });
        const tracked = store.create(docId, created);
        return tracked;
      },
      async update(docId, annotationId, patch) {
        const mode = lockMode$.peek();
        if (mode === "read-only") {
          throw new Error("Cannot update annotations in read-only lock mode");
        }
        const existing = store.getForDocument(docId).find((t) => t.id === annotationId);
        if (existing && !userCanAccess(existing.data, existing.data.editableBy)) {
          throw new Error("Permission denied: cannot edit this annotation");
        }
        const indexInfo = annotIndexMap.get(`${docId}:${annotationId}`);
        if (indexInfo === void 0) {
          throw new Error(`Annotation ${annotationId} not found in index map`);
        }
        const enrichedPatch = {
          ...patch,
          modifiedDate: (/* @__PURE__ */ new Date()).toISOString(),
          modifiedBy: ctx.engine.user?.id
        };
        await ctx.engine.workerProxy.updateAnnotation(
          docId,
          indexInfo.pageIndex,
          indexInfo.annotIndex,
          enrichedPatch
        );
        store.update(docId, annotationId, enrichedPatch);
      },
      async delete(docId, annotationId) {
        const mode = lockMode$.peek();
        if (mode === "read-only" || mode === "no-delete") {
          throw new Error(`Cannot delete annotations in '${mode}' lock mode`);
        }
        const tracked = store.getForDocument(docId).find((t) => t.id === annotationId);
        if (tracked && !userCanAccess(tracked.data, tracked.data.deletableBy)) {
          throw new Error("Permission denied: cannot delete this annotation");
        }
        if (!tracked) {
          if (selectedAnnotation$.peek() === annotationId) {
            selectedAnnotation$.value = null;
          }
          if (selectedAnnotations$.peek().includes(annotationId)) {
            selectedAnnotations$.value = selectedAnnotations$.peek().filter((id) => id !== annotationId);
          }
          return;
        }
        const pageIndex = tracked.data.pageIndex;
        let indexInfo = annotIndexMap.get(`${docId}:${annotationId}`);
        if (!indexInfo) {
          const workerAnnots = await ctx.engine.workerProxy.getAnnotations(docId, pageIndex);
          const rect = tracked.data.rect;
          const matchIdx = workerAnnots.findIndex(
            (a) => Math.abs(a.rect.left - rect.left) < 0.5 && Math.abs(a.rect.right - rect.right) < 0.5 && a.subtype === tracked.data.subtype
          );
          if (matchIdx >= 0) {
            indexInfo = { pageIndex, annotIndex: matchIdx };
          }
        }
        if (indexInfo) {
          await ctx.engine.workerProxy.deleteAnnotation(docId, indexInfo.pageIndex, indexInfo.annotIndex);
          const remaining = await ctx.engine.workerProxy.getAnnotations(docId, pageIndex);
          for (const [key, val] of annotIndexMap.entries()) {
            if (val.pageIndex === pageIndex && key.startsWith(`${docId}:`)) {
              annotIndexMap.delete(key);
            }
          }
          remaining.forEach((annot, idx) => {
            annotIndexMap.set(`${docId}:${annot.id}`, { pageIndex, annotIndex: idx });
          });
        }
        annotIndexMap.delete(`${docId}:${annotationId}`);
        store.delete(docId, annotationId);
        if (selectedAnnotation$.peek() === annotationId) {
          selectedAnnotation$.value = null;
        }
        if (selectedAnnotations$.peek().includes(annotationId)) {
          const next = selectedAnnotations$.peek().filter((id) => id !== annotationId);
          selectedAnnotations$.value = next;
          if (selectedAnnotation$.peek() === null && next.length > 0) {
            selectedAnnotation$.value = next[next.length - 1];
          }
        }
      },
      getForPage(docId, pageIndex) {
        return store.getForPage(docId, pageIndex);
      },
      getForDocument(docId) {
        return store.getForDocument(docId);
      },
      getDirty(docId) {
        return store.getDirty(docId);
      },
      hasDirty(docId) {
        return store.hasDirty(docId);
      },
      markSynced(docId, annotationId) {
        store.markSynced(docId, annotationId);
      },
      markAllSynced(docId) {
        store.markAllSynced(docId);
      },
      subscribe(fn) {
        return store.subscribe(fn);
      },
      lockMode: computed9(() => lockMode$.value),
      setLockMode(mode) {
        lockMode$.value = mode;
        ctx.emit("annotation:lock-mode-changed", mode);
      },
      selectedAnnotation: computed9(() => selectedAnnotation$.value),
      selectedAnnotations: computed9(() => selectedAnnotations$.value),
      selectAnnotation(annotationId) {
        selectedAnnotation$.value = annotationId;
        selectedAnnotations$.value = annotationId !== null ? [annotationId] : [];
        ctx.emit("annotation:selection-changed", annotationId);
      },
      toggleAnnotationSelection(annotationId) {
        const cur = selectedAnnotations$.peek();
        if (cur.includes(annotationId)) {
          const next = cur.filter((id) => id !== annotationId);
          selectedAnnotations$.value = next;
          selectedAnnotation$.value = next.length > 0 ? next[next.length - 1] : null;
        } else {
          const next = [...cur, annotationId];
          selectedAnnotations$.value = next;
          selectedAnnotation$.value = annotationId;
        }
        ctx.emit("annotation:selection-changed", selectedAnnotation$.peek());
      },
      clearAnnotationSelection() {
        if (selectedAnnotations$.peek().length === 0 && selectedAnnotation$.peek() === null) {
          return;
        }
        selectedAnnotations$.value = [];
        selectedAnnotation$.value = null;
        ctx.emit("annotation:selection-changed", null);
      },
      isPageLoaded(docId, pageIndex) {
        return loadedPages.get(docId)?.has(pageIndex) ?? false;
      },
      // ── Annotation tools ──
      activeTool: computed9(() => activeTool$.value),
      setActiveTool(tool) {
        const prev = activeTool$.peek();
        activeTool$.value = tool;
        if (tool !== null) {
          const defaults2 = ANNOTATION_TOOL_DEFAULTS[tool];
          toolStyle$.value = {
            ...toolStyle$.peek(),
            color: defaults2.color,
            opacity: defaults2.opacity ?? 1
          };
          drawHandler?.activate();
        } else if (prev !== null) {
          drawHandler?.deactivate();
        }
        ctx.emit("annotation:tool-changed", tool);
      },
      toolStyle: computed9(() => toolStyle$.value),
      setToolStyle(patch) {
        toolStyle$.value = { ...toolStyle$.peek(), ...patch };
      },
      user: ctx.engine.user,
      // ── Comment management ──
      async setCommentStatus(docId, annotationId, status) {
        await capability.update(docId, annotationId, { commentStatus: status });
        ctx.emit("annotation:comment-status-changed", annotationId, status);
      },
      async toggleResolved(docId, annotationId) {
        const tracked = store.getForDocument(docId).find((t) => t.id === annotationId);
        const current = tracked?.data.resolved ?? false;
        await capability.update(docId, annotationId, { resolved: !current });
        ctx.emit("annotation:resolved-changed", annotationId, !current);
      },
      async editComment(docId, annotationId, commentId, newText) {
        const tracked = store.getForDocument(docId).find((t) => t.id === annotationId);
        if (!tracked) return;
        const userId = ctx.engine.user?.id;
        if (commentId === "initial") {
          if (userId && tracked.data.authorId && tracked.data.authorId !== userId) {
            throw new Error("Permission denied: only the author can edit this comment");
          }
          await capability.update(docId, annotationId, { contents: newText });
        } else {
          const comments = tracked.data.comments ?? [];
          const idx = comments.findIndex((c) => c.id === commentId);
          if (idx < 0) return;
          if (userId && comments[idx].authorId && comments[idx].authorId !== userId) {
            throw new Error("Permission denied: only the author can edit this comment");
          }
          const updated = [...comments];
          updated[idx] = {
            ...comments[idx],
            text: newText,
            edited: true,
            editedAt: (/* @__PURE__ */ new Date()).toISOString()
          };
          await capability.update(docId, annotationId, { comments: updated });
        }
      },
      async deleteComment(docId, annotationId, commentId) {
        const tracked = store.getForDocument(docId).find((t) => t.id === annotationId);
        if (!tracked) return;
        const userId = ctx.engine.user?.id;
        if (commentId === "initial") {
          const comments = tracked.data.comments ?? [];
          if (comments.length > 0) {
            const [first, ...rest] = comments;
            await capability.update(docId, annotationId, {
              contents: first.text,
              author: first.authorName,
              authorId: first.authorId,
              comments: rest.length > 0 ? rest : void 0
            });
          } else {
            await capability.update(docId, annotationId, { contents: void 0 });
          }
        } else {
          const comments = tracked.data.comments ?? [];
          const idx = comments.findIndex((c) => c.id === commentId);
          if (idx < 0) return;
          if (userId && comments[idx].authorId && comments[idx].authorId !== userId) {
            throw new Error("Permission denied: only the author can delete this comment");
          }
          const updated = comments.filter((c) => c.id !== commentId);
          await capability.update(docId, annotationId, {
            comments: updated.length > 0 ? updated : void 0
          });
        }
      },
      markAsRead(docId, annotationId) {
        const now = (/* @__PURE__ */ new Date()).toISOString();
        const tracked = store.getForDocument(docId).find((t) => t.id === annotationId);
        if (!tracked) return;
        store.update(docId, annotationId, { readAt: now });
      },
      // ── Z-order ──
      async bringToFront(docId, annotationId) {
        const pageAnnots = store.getForDocument(docId);
        const maxZ = pageAnnots.reduce((max, t) => Math.max(max, t.data.zIndex ?? 0), 0);
        await capability.update(docId, annotationId, { zIndex: maxZ + 1 });
      },
      async sendToBack(docId, annotationId) {
        const pageAnnots = store.getForDocument(docId);
        const minZ = pageAnnots.reduce((min, t) => Math.min(min, t.data.zIndex ?? 0), 0);
        await capability.update(docId, annotationId, { zIndex: minZ - 1 });
      },
      // ── Grouping ──
      async groupAnnotations(docId, annotationIds) {
        if (annotationIds.length < 2) return;
        const groupId = uuid();
        for (const id of annotationIds) {
          await capability.update(docId, id, { groupId });
        }
        ctx.emit("annotation:grouped", groupId, annotationIds);
      },
      async ungroupAnnotations(docId, groupId) {
        const grouped = store.getForDocument(docId).filter((t) => t.data.groupId === groupId);
        for (const t of grouped) {
          await capability.update(docId, t.id, { groupId: void 0 });
        }
        ctx.emit("annotation:ungrouped", groupId);
      },
      canEdit(docId, annotationId) {
        const tracked = store.getForDocument(docId).find((t) => t.id === annotationId);
        if (!tracked) return false;
        const mode = lockMode$.peek();
        if (mode === "read-only") return false;
        return userCanAccess(tracked.data, tracked.data.editableBy);
      },
      canDelete(docId, annotationId) {
        const tracked = store.getForDocument(docId).find((t) => t.id === annotationId);
        if (!tracked) return false;
        const mode = lockMode$.peek();
        if (mode === "read-only" || mode === "no-delete") return false;
        return userCanAccess(tracked.data, tracked.data.deletableBy);
      },
      activeStampName: "Approved",
      setStagedImage(image) {
        stagedImage = image;
      },
      getStagedImage() {
        return stagedImage;
      }
    };
    let drawHandler = null;
    let containerEl = null;
    ctx.on("viewport:container-attached", (...args) => {
      containerEl = args[0];
    });
    if (interaction) {
      drawHandler = createDrawModeHandler({
        document: document2,
        viewport,
        annotation: capability,
        textLayer,
        history,
        interaction,
        formatting,
        // Lazy lookup: the measurement plugin requires the annotation
        // plugin, so it hasn't been set up yet at this point in time.
        // Resolving inside the closure means draw-mode handlers get the
        // live capability if/when the consumer registered it.
        getMeasurement: () => ctx.optional("measurement"),
        getPageRotation: (docId, pageIndex) => {
          if (!ctx.engine.pageRotation.has(docId, pageIndex)) {
            void ctx.engine.pageRotation.resolve(docId, pageIndex);
          }
          return ctx.engine.pageRotation.get(docId, pageIndex);
        },
        activeTool: () => activeTool$.peek(),
        style: () => toolStyle$.peek(),
        emit: (event, ...args) => ctx.emit(event, ...args),
        getContainer: () => containerEl,
        getStampName: () => capability.activeStampName,
        getStagedImage: () => capability.getStagedImage(),
        onImagePlaced: () => capability.setStagedImage(null)
      });
      ctx.on("annotation:tool-deactivated", () => {
        capability.setActiveTool(null);
      });
    }
    ctx.on("viewport:container-attached", (...args) => {
      const container = args[0];
      const NUDGE_PT = 2;
      const NUDGE_PT_LARGE = 20;
      const buildMovePatch = (annot, dx, dy) => ({
        rect: {
          left: annot.rect.left + dx,
          right: annot.rect.right + dx,
          top: annot.rect.top + dy,
          bottom: annot.rect.bottom + dy
        },
        ...annot.ink ? {
          ink: {
            strokes: annot.ink.strokes.map(
              (s) => s.map((p) => p.pressure !== void 0 ? { x: p.x + dx, y: p.y + dy, pressure: p.pressure } : { x: p.x + dx, y: p.y + dy })
            )
          }
        } : {},
        ...annot.line ? {
          line: {
            start: { x: annot.line.start.x + dx, y: annot.line.start.y + dy },
            end: { x: annot.line.end.x + dx, y: annot.line.end.y + dy }
          }
        } : {},
        ...annot.markup ? {
          markup: {
            quadPoints: annot.markup.quadPoints.map((qp) => ({
              x1: qp.x1 + dx,
              y1: qp.y1 + dy,
              x2: qp.x2 + dx,
              y2: qp.y2 + dy,
              x3: qp.x3 + dx,
              y3: qp.y3 + dy,
              x4: qp.x4 + dx,
              y4: qp.y4 + dy
            }))
          }
        } : {},
        ...annot.callout ? {
          callout: {
            endpoint: {
              x: annot.callout.endpoint.x + dx,
              y: annot.callout.endpoint.y + dy
            },
            ...annot.callout.knee ? {
              knee: {
                x: annot.callout.knee.x + dx,
                y: annot.callout.knee.y + dy
              }
            } : {},
            ...annot.callout.lineEnding ? { lineEnding: annot.callout.lineEnding } : {}
          }
        } : {}
      });
      const buildResizePatch = (annot, dir, step) => {
        const r = annot.rect;
        switch (dir) {
          case "ArrowRight":
            return { rect: { ...r, right: r.right + step } };
          case "ArrowLeft":
            return { rect: { ...r, right: Math.max(r.left + step, r.right - step) } };
          case "ArrowUp":
            return { rect: { ...r, top: r.top + step } };
          case "ArrowDown":
            return { rect: { ...r, top: Math.max(r.bottom + step, r.top - step) } };
        }
      };
      const onKeyDown = (e) => {
        const selId = selectedAnnotation$.peek();
        if (!selId) return;
        if (e.key === "Delete" || e.key === "Backspace") {
          e.preventDefault();
          const doc = document2.activeDocument.peek();
          if (doc) {
            void capability.delete(doc.id, selId);
          }
        }
        if (e.key === "Escape") {
          capability.selectAnnotation(null);
        }
        const isArrow = e.key === "ArrowLeft" || e.key === "ArrowRight" || e.key === "ArrowUp" || e.key === "ArrowDown";
        if (isArrow) {
          e.preventDefault();
          const doc = document2.activeDocument.peek();
          if (!doc) return;
          const tracked = store.getForDocument(doc.id).find((t) => t.id === selId);
          if (!tracked) return;
          if (e.shiftKey) {
            const patch = buildResizePatch(tracked.data, e.key, NUDGE_PT);
            void capability.update(doc.id, selId, patch);
          } else {
            const step = e.ctrlKey || e.metaKey ? NUDGE_PT_LARGE : NUDGE_PT;
            const dx = e.key === "ArrowRight" ? step : e.key === "ArrowLeft" ? -step : 0;
            const dy = e.key === "ArrowUp" ? step : e.key === "ArrowDown" ? -step : 0;
            const patch = buildMovePatch(tracked.data, dx, dy);
            void capability.update(doc.id, selId, patch);
          }
        }
      };
      container.addEventListener("keydown", onKeyDown);
      container.addEventListener("pointerdown", (e) => {
        if (selectedAnnotation$.peek() !== null && activeTool$.peek() === null) {
          const target = e.target;
          if (target.closest(".lector-annot-overlay, .lector-annot-note, .lector-annot-popover, .lector-line-handle, .lector-resize-handle, .lector-vertex-handle")) return;
          capability.selectAnnotation(null);
        }
      });
    });
    return capability;
  },
  dispose() {
  }
});

// src/plugins/annotation-presets-plugin.ts
import { signal as signal10, computed as computed10 } from "@truespar/lector-utils";
var STORAGE_KEY = "lector.annotationPresets.user";
function loadFromStorage() {
  if (typeof localStorage === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (p) => p !== null && typeof p === "object" && typeof p.name === "string"
    );
  } catch {
    return [];
  }
}
function saveToStorage(presets) {
  if (typeof localStorage === "undefined") return;
  try {
    const userPresets = presets.filter((p) => !p.builtin);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(userPresets));
  } catch {
  }
}
function coerceColor(value) {
  if (!value || typeof value !== "object") return void 0;
  const v = value;
  if (typeof v.r === "number" && typeof v.g === "number" && typeof v.b === "number") {
    return {
      r: v.r,
      g: v.g,
      b: v.b,
      a: typeof v.a === "number" ? v.a : 255
    };
  }
  return void 0;
}
function coerceConfigPresets(raw) {
  if (!raw) return [];
  const out = [];
  for (const [name, val] of Object.entries(raw)) {
    if (!val || typeof val !== "object") continue;
    const labelRaw = val.label;
    const iconRaw = val.icon;
    const borderRaw = val.borderWidth;
    const fontRaw = val.fontSize;
    const opacityRaw = val.opacity;
    const interiorRaw = val.interiorColor;
    const preset = {
      name,
      label: typeof labelRaw === "string" ? labelRaw : name,
      icon: typeof iconRaw === "string" ? iconRaw : void 0,
      color: coerceColor(val.color),
      interiorColor: interiorRaw === null ? null : coerceColor(interiorRaw),
      borderWidth: typeof borderRaw === "number" ? borderRaw : void 0,
      fontSize: typeof fontRaw === "number" ? fontRaw : void 0,
      opacity: typeof opacityRaw === "number" ? opacityRaw : void 0,
      builtin: true
    };
    out.push(preset);
  }
  return out;
}
var annotationPresetsPlugin = definePlugin({
  id: "annotation-presets",
  provides: ["annotation-presets"],
  requires: ["annotation"],
  optional: [],
  setup(ctx) {
    const annotation = ctx.require("annotation");
    const builtin = coerceConfigPresets(ctx.engine.annotationPresets);
    const user = loadFromStorage();
    const seeded = [...builtin];
    for (const p of user) {
      const idx = seeded.findIndex((b) => b.name === p.name);
      const normalized = { ...p, builtin: false };
      if (idx >= 0) seeded[idx] = normalized;
      else seeded.push(normalized);
    }
    const presets$ = signal10(seeded);
    const activePreset$ = signal10(null);
    function applyPresetToToolStyle(p) {
      const patch = {};
      if (p.color !== void 0) patch.color = p.color;
      if (p.interiorColor !== void 0) patch.interiorColor = p.interiorColor;
      if (typeof p.borderWidth === "number") patch.borderWidth = p.borderWidth;
      if (typeof p.fontSize === "number") patch.fontSize = p.fontSize;
      if (typeof p.opacity === "number") patch.opacity = p.opacity;
      if (Object.keys(patch).length > 0) annotation.setToolStyle(patch);
    }
    ctx.on("annotation:tool-changed", () => {
      const name = activePreset$.peek();
      if (name === null) return;
      const p = presets$.peek().find((x) => x.name === name);
      if (p) applyPresetToToolStyle(p);
    });
    const capability = {
      presets: computed10(() => presets$.value),
      activePreset: computed10(() => activePreset$.value),
      getPreset(name) {
        return presets$.peek().find((p) => p.name === name);
      },
      setActivePreset(name) {
        if (name === null) {
          activePreset$.value = null;
          ctx.emit("annotation-presets:active-changed", null);
          return;
        }
        const p = presets$.peek().find((x) => x.name === name);
        if (!p) return;
        activePreset$.value = name;
        applyPresetToToolStyle(p);
        ctx.emit("annotation-presets:active-changed", name);
      },
      savePreset(preset) {
        if (preset.name.trim().length === 0) return;
        const list = [...presets$.peek()];
        const idx = list.findIndex((p) => p.name === preset.name);
        const normalized = { ...preset, builtin: false };
        if (idx >= 0) list[idx] = normalized;
        else list.push(normalized);
        presets$.value = list;
        saveToStorage(list);
        ctx.emit("annotation-presets:changed");
      },
      deletePreset(name) {
        const list = presets$.peek();
        const target = list.find((p) => p.name === name);
        if (!target || target.builtin) return false;
        const next = list.filter((p) => p.name !== name);
        presets$.value = next;
        saveToStorage(next);
        if (activePreset$.peek() === name) {
          activePreset$.value = null;
          ctx.emit("annotation-presets:active-changed", null);
        }
        ctx.emit("annotation-presets:changed");
        return true;
      },
      saveCurrentAsPreset(name) {
        const trimmed = name.trim();
        if (trimmed.length === 0) return null;
        const style = annotation.toolStyle.peek();
        const preset = {
          name: trimmed,
          label: trimmed,
          color: style.color,
          interiorColor: style.interiorColor,
          borderWidth: style.borderWidth,
          fontSize: style.fontSize,
          opacity: style.opacity
        };
        capability.savePreset(preset);
        return preset;
      }
    };
    return capability;
  }
});

// src/plugins/comparison-plugin.ts
import { signal as signal11, computed as computed11 } from "@truespar/lector-utils";
var comparisonPlugin = definePlugin({
  id: "comparison",
  provides: ["comparison"],
  requires: ["document"],
  optional: [],
  setup(ctx) {
    const document2 = ctx.require("document");
    const state$ = signal11("inactive");
    const result$ = signal11(null);
    const error$ = signal11(null);
    const pair$ = signal11(null);
    function makeIdenticalResult(docA, docB) {
      const handleA = document2.getHandle(docA);
      const handleB = document2.getHandle(docB);
      const pageCount = handleA?.pageCount ?? handleB?.pageCount ?? 0;
      const pageDiffs = [];
      for (let i = 0; i < pageCount; i++) {
        pageDiffs.push({
          pageA: i,
          pageB: i,
          mode: "identical",
          changes: []
        });
      }
      return {
        pageCountA: handleA?.pageCount ?? 0,
        pageCountB: handleB?.pageCount ?? 0,
        pageDiffs,
        totalChanges: 0
      };
    }
    function hashesMatch(docA, docB) {
      const a = document2.getHandle(docA)?.sha256 ?? "";
      const b = document2.getHandle(docB)?.sha256 ?? "";
      return a.length > 0 && a === b;
    }
    ctx.on("document:closed", (...args) => {
      const closedId = args[0];
      const cur = pair$.peek();
      if (cur && (cur.docA === closedId || cur.docB === closedId)) {
        capability.exit();
      }
    });
    const capability = {
      state: computed11(() => state$.value),
      result: computed11(() => result$.value),
      error: computed11(() => error$.value),
      activePair: computed11(() => pair$.value),
      async compare(docA, docB) {
        if (docA === docB) {
          throw new Error("Cannot compare a document against itself");
        }
        if (hashesMatch(docA, docB)) {
          return makeIdenticalResult(docA, docB);
        }
        return ctx.engine.workerProxy.compareDocuments(docA, docB);
      },
      async enter(docA, docB) {
        if (docA === docB) {
          const msg = "Cannot compare a document against itself";
          state$.value = "error";
          error$.value = msg;
          ctx.emit("comparison:error", msg);
          return;
        }
        if (hashesMatch(docA, docB)) {
          const res = makeIdenticalResult(docA, docB);
          pair$.value = { docA, docB };
          result$.value = res;
          state$.value = "active";
          error$.value = null;
          ctx.emit("comparison:entered", res);
          return;
        }
        state$.value = "computing";
        error$.value = null;
        pair$.value = { docA, docB };
        ctx.emit("comparison:computing", docA, docB);
        try {
          const res = await ctx.engine.workerProxy.compareDocuments(docA, docB);
          const stillCurrent = pair$.peek();
          if (!stillCurrent || stillCurrent.docA !== docA || stillCurrent.docB !== docB) {
            return;
          }
          result$.value = res;
          state$.value = "active";
          ctx.emit("comparison:entered", res);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          state$.value = "error";
          error$.value = msg;
          result$.value = null;
          pair$.value = null;
          ctx.emit("comparison:error", msg);
        }
      },
      exit() {
        if (state$.peek() === "inactive") return;
        state$.value = "inactive";
        result$.value = null;
        error$.value = null;
        pair$.value = null;
        ctx.emit("comparison:exited");
      }
    };
    ctx.registerCommand({
      id: "comparison.exit",
      label: "Exit compare mode",
      icon: "x",
      category: "Comparison",
      execute: () => {
        capability.exit();
      }
    });
    return capability;
  }
});

// src/plugins/form-plugin.ts
import { signal as signal12, computed as computed12 } from "@truespar/lector-utils";

// src/data/form-store.ts
var FormStore = class {
  #log;
  #trackers = /* @__PURE__ */ new Map();
  #eventBus;
  #userId;
  /** Page-index lookup: documentId -> pageIndex -> Set<fieldName> */
  #pageIndex = /* @__PURE__ */ new Map();
  constructor(eventBus, userId) {
    this.#eventBus = eventBus;
    this.#userId = userId;
    this.#log = new OperationLog();
  }
  /**
   * Load form fields for a page from the worker result.
   *
   * Existing fields for the same page are replaced. Each loaded field
   * starts in the 'synced' commit state because it reflects the current
   * PDF state.
   */
  loadPage(documentId, pageIndex, fields) {
    const tracker = this.#ensureTracker(documentId);
    const pageMap = this.#ensurePageMap(documentId);
    const existing = pageMap.get(pageIndex);
    if (existing !== void 0) {
      for (const name of existing) {
        tracker.remove(name);
      }
    }
    const names = /* @__PURE__ */ new Set();
    for (const field of fields) {
      tracker.add(field.fieldName, field, "synced");
      names.add(field.fieldName);
    }
    pageMap.set(pageIndex, names);
  }
  /**
   * Record a form field value change.
   *
   * Builds a new {@link WidgetData} with the updated `fieldValue`, records
   * it in the tracker and operation log, and emits a `'form:field-changed'`
   * event.
   */
  updateField(documentId, fieldName, value, pageIndex) {
    const tracker = this.#ensureTracker(documentId);
    const existing = tracker.get(fieldName);
    let updated;
    if (existing !== void 0) {
      updated = { ...existing.data, fieldValue: value };
    } else {
      updated = {
        fieldType: 0,
        fieldName,
        fieldValue: value,
        annotIndex: -1,
        fieldFlags: 0
      };
      const pageMap = this.#ensurePageMap(documentId);
      let names = pageMap.get(pageIndex);
      if (names === void 0) {
        names = /* @__PURE__ */ new Set();
        pageMap.set(pageIndex, names);
      }
      names.add(fieldName);
    }
    if (existing !== void 0) {
      tracker.update(fieldName, updated);
    } else {
      tracker.add(fieldName, updated, "new");
    }
    const event = this.#buildEvent(
      existing !== void 0 ? "updated" : "created",
      documentId,
      pageIndex,
      fieldName,
      updated,
      { fieldValue: value }
    );
    this.#log.append(event);
    this.#eventBus.emit("form:field-changed", event);
  }
  /** Get all form fields for a document. */
  getForDocument(documentId) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return [];
    const result = [];
    const pageMap = this.#pageIndex.get(documentId);
    if (pageMap === void 0) return [];
    for (const [, names] of pageMap) {
      for (const name of names) {
        const tracked = tracker.get(name);
        if (tracked !== void 0) {
          result.push(tracked);
        }
      }
    }
    return result;
  }
  /** Get form fields for a specific page. */
  getForPage(documentId, pageIndex) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return [];
    const pageMap = this.#pageIndex.get(documentId);
    if (pageMap === void 0) return [];
    const names = pageMap.get(pageIndex);
    if (names === void 0) return [];
    const result = [];
    for (const name of names) {
      const tracked = tracker.get(name);
      if (tracked !== void 0) {
        result.push(tracked);
      }
    }
    return result;
  }
  /** Get all changed fields for a document. */
  getDirty(documentId) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return [];
    return tracker.getDirty();
  }
  /** Check if a document's form has unsaved changes. */
  hasDirty(documentId) {
    const tracker = this.#ensureTracker(documentId);
    return tracker.hasDirty;
  }
  /** Mark a single field as synced. */
  markSynced(documentId, fieldName) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return;
    tracker.markSynced(fieldName);
  }
  /** Convenience alias for `updateField` — matches the plugin API naming. */
  setFieldValue(documentId, pageIndex, fieldName, value) {
    this.updateField(documentId, fieldName, value, pageIndex);
  }
  /** Get the current value of a field by name. Returns undefined if not loaded. */
  getFieldValue(documentId, fieldName) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return void 0;
    const tracked = tracker.get(fieldName);
    return tracked?.data.fieldValue;
  }
  /** Extract all form data as a flat record (fieldName → fieldValue). */
  extractFormData(documentId) {
    const result = {};
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return result;
    const pageMap = this.#pageIndex.get(documentId);
    if (pageMap === void 0) return result;
    for (const [, names] of pageMap) {
      for (const name of names) {
        const tracked = tracker.get(name);
        if (tracked !== void 0) {
          result[name] = tracked.data.fieldValue;
        }
      }
    }
    return result;
  }
  /** Mark all fields as synced for a document. */
  markAllSynced(documentId) {
    const tracker = this.#trackers.get(documentId);
    if (tracker === void 0) return;
    tracker.markAllSynced();
  }
  /** Subscribe to form events from the operation log. */
  subscribe(fn) {
    return this.#log.subscribe(fn);
  }
  /** Get the operation log (for sync/replay). */
  get log() {
    return this.#log;
  }
  /** Clean up all state for a document. */
  clearDocument(documentId) {
    const tracker = this.#trackers.get(documentId);
    if (tracker !== void 0) {
      tracker[Symbol.dispose]();
      this.#trackers.delete(documentId);
    }
    this.#pageIndex.delete(documentId);
    this.#log.clearDocument(documentId);
  }
  /** Dispose: clean up all documents. */
  [Symbol.dispose]() {
    for (const [, tracker] of this.#trackers) {
      tracker[Symbol.dispose]();
    }
    this.#trackers.clear();
    this.#pageIndex.clear();
    this.#log[Symbol.dispose]();
  }
  // ── Private helpers ──
  #ensureTracker(documentId) {
    let tracker = this.#trackers.get(documentId);
    if (tracker === void 0) {
      tracker = new DirtyTracker();
      this.#trackers.set(documentId, tracker);
    }
    return tracker;
  }
  #ensurePageMap(documentId) {
    let pageMap = this.#pageIndex.get(documentId);
    if (pageMap === void 0) {
      pageMap = /* @__PURE__ */ new Map();
      this.#pageIndex.set(documentId, pageMap);
    }
    return pageMap;
  }
  #buildEvent(type, documentId, pageIndex, objectId, data, patch) {
    return {
      type,
      documentId,
      pageIndex,
      objectId,
      data,
      patch,
      timestamp: Date.now(),
      operationId: uuid(),
      userId: this.#userId
    };
  }
};

// src/plugins/form-plugin.ts
var formPlugin = definePlugin({
  id: "form",
  provides: ["form"],
  requires: ["document", "render"],
  optional: ["interaction"],
  setup(ctx) {
    ctx.require("document");
    ctx.require("render");
    const store = new FormStore(ctx.engine.plugins.events);
    const readOnly$ = signal12(false);
    const focusedField$ = signal12(null);
    const loadedPages = /* @__PURE__ */ new Map();
    const loadingPages = /* @__PURE__ */ new Map();
    function markPageLoaded(docId, pageIndex) {
      let pages = loadedPages.get(docId);
      if (pages === void 0) {
        pages = /* @__PURE__ */ new Set();
        loadedPages.set(docId, pages);
      }
      pages.add(pageIndex);
    }
    ctx.on("document:closed", (...args) => {
      const docId = args[0];
      store.clearDocument(docId);
      loadedPages.delete(docId);
      focusedField$.value = null;
    });
    const capability = {
      store,
      async loadPage(docId, pageIndex) {
        if (loadedPages.get(docId)?.has(pageIndex)) return;
        const key = `${docId}:${pageIndex}`;
        const existing = loadingPages.get(key);
        if (existing) return existing;
        const load = (async () => {
          const fields = await ctx.engine.workerProxy.getFormFields(docId, pageIndex);
          store.loadPage(docId, pageIndex, fields);
          markPageLoaded(docId, pageIndex);
          ctx.emit("form:page-loaded", docId, pageIndex, fields.length);
        })();
        loadingPages.set(key, load);
        try {
          await load;
        } finally {
          loadingPages.delete(key);
        }
      },
      getPageFields(docId, pageIndex) {
        return store.getForPage(docId, pageIndex);
      },
      getDocumentFields(docId) {
        return store.getForDocument(docId);
      },
      async setFieldValue(docId, pageIndex, fieldName, value) {
        if (readOnly$.peek()) {
          throw new Error("Cannot set form field value in read-only mode");
        }
        await ctx.engine.workerProxy.setFormFieldValue(docId, pageIndex, fieldName, value);
        store.setFieldValue(docId, pageIndex, fieldName, value);
      },
      getFieldValue(docId, fieldName) {
        return store.getFieldValue(docId, fieldName);
      },
      async populateFields(docId, fields) {
        if (readOnly$.peek()) {
          throw new Error("Cannot populate form fields in read-only mode");
        }
        for (const field of fields) {
          await ctx.engine.workerProxy.setFormFieldValue(
            docId,
            field.pageIndex,
            field.fieldName,
            field.value
          );
          store.setFieldValue(docId, field.pageIndex, field.fieldName, field.value);
        }
        ctx.emit("form:fields-populated", docId, fields.length);
      },
      extractFormData(docId) {
        return store.extractFormData(docId);
      },
      hasDirty(docId) {
        return store.hasDirty(docId);
      },
      markAllSynced(docId) {
        store.markAllSynced(docId);
      },
      subscribe(fn) {
        return store.subscribe(fn);
      },
      readOnly: computed12(() => readOnly$.value),
      setReadOnly(ro) {
        readOnly$.value = ro;
        ctx.emit("form:read-only-changed", ro);
      },
      focusedField: computed12(() => focusedField$.value),
      focusField(fieldName) {
        focusedField$.value = fieldName;
        if (fieldName !== null) {
          ctx.emit("form:field-focused", fieldName);
        } else {
          ctx.emit("form:field-blurred");
        }
      },
      isPageLoaded(docId, pageIndex) {
        return loadedPages.get(docId)?.has(pageIndex) ?? false;
      },
      async clickWidget(docId, pageIndex, pageX, pageY) {
        const updatedFields = await ctx.engine.workerProxy.clickFormWidget(
          docId,
          pageIndex,
          pageX,
          pageY
        );
        for (const field of updatedFields) {
          if (store.getFieldValue(docId, field.fieldName) !== field.fieldValue) {
            store.setFieldValue(docId, pageIndex, field.fieldName, field.fieldValue);
          }
        }
        ctx.emit("form:widget-clicked", docId, pageIndex);
        return updatedFields;
      }
    };
    return capability;
  },
  dispose() {
  }
});

// src/plugins/history-plugin.ts
import { signal as signal13, computed as computed13 } from "@truespar/lector-utils";
var MAX_UNDO_DEPTH = 200;
var historyPlugin = definePlugin({
  id: "history",
  provides: ["history"],
  requires: ["document"],
  optional: [],
  setup(ctx) {
    const document2 = ctx.require("document");
    const histories = /* @__PURE__ */ new Map();
    const revision$ = signal13(0);
    function ensureHistory(docId) {
      let h = histories.get(docId);
      if (h === void 0) {
        h = { undoStack: [], redoStack: [], batch: null, busy: false };
        histories.set(docId, h);
      }
      return h;
    }
    function activeDocHistory() {
      const handle = document2.activeDocument.peek();
      if (handle === null) return null;
      return histories.get(handle.id) ?? null;
    }
    function bumpRevision() {
      revision$.update((n) => n + 1);
    }
    ctx.on("document:closed", (...args) => {
      const docId = args[0];
      histories.delete(docId);
      bumpRevision();
    });
    const canUndo$ = computed13(() => {
      void revision$.value;
      void document2.activeDocument.value;
      const h = activeDocHistory();
      return h !== null && h.undoStack.length > 0;
    });
    const canRedo$ = computed13(() => {
      void revision$.value;
      void document2.activeDocument.value;
      const h = activeDocHistory();
      return h !== null && h.redoStack.length > 0;
    });
    const undoLabel$ = computed13(() => {
      void revision$.value;
      void document2.activeDocument.value;
      const h = activeDocHistory();
      if (h === null || h.undoStack.length === 0) return null;
      return h.undoStack[h.undoStack.length - 1].label;
    });
    const redoLabel$ = computed13(() => {
      void revision$.value;
      void document2.activeDocument.value;
      const h = activeDocHistory();
      if (h === null || h.redoStack.length === 0) return null;
      return h.redoStack[h.redoStack.length - 1].label;
    });
    const capability = {
      push(docId, entry) {
        const h = ensureHistory(docId);
        if (h.batch !== null) {
          h.batch.entries.push(entry);
          return;
        }
        h.undoStack.push(entry);
        if (h.undoStack.length > MAX_UNDO_DEPTH) h.undoStack.shift();
        h.redoStack.length = 0;
        bumpRevision();
        ctx.emit("history:pushed", docId, entry);
      },
      async undo(docId) {
        const h = histories.get(docId);
        if (h === void 0 || h.busy || h.undoStack.length === 0) return;
        h.busy = true;
        try {
          const entry = h.undoStack.pop();
          await entry.undo();
          h.redoStack.push(entry);
          bumpRevision();
          ctx.emit("history:undo", docId, entry);
        } finally {
          h.busy = false;
        }
      },
      async redo(docId) {
        const h = histories.get(docId);
        if (h === void 0 || h.busy || h.redoStack.length === 0) return;
        h.busy = true;
        try {
          const entry = h.redoStack.pop();
          await entry.execute();
          h.undoStack.push(entry);
          bumpRevision();
          ctx.emit("history:redo", docId, entry);
        } finally {
          h.busy = false;
        }
      },
      canUndo: canUndo$,
      canRedo: canRedo$,
      undoLabel: undoLabel$,
      redoLabel: redoLabel$,
      beginBatch(docId, label) {
        const h = ensureHistory(docId);
        if (h.batch !== null) {
          throw new Error("A batch is already in progress");
        }
        h.batch = { label, entries: [] };
      },
      endBatch(docId) {
        const h = histories.get(docId);
        if (h === void 0 || h.batch === null) return;
        const batch = h.batch;
        h.batch = null;
        if (batch.entries.length === 0) return;
        const batchEntry = {
          id: uuid(),
          label: batch.label,
          topic: batch.entries[0].topic,
          timestamp: Date.now(),
          async execute() {
            for (const e of batch.entries) {
              await e.execute();
            }
          },
          async undo() {
            for (let i = batch.entries.length - 1; i >= 0; i--) {
              await batch.entries[i].undo();
            }
          }
        };
        h.undoStack.push(batchEntry);
        h.redoStack.length = 0;
        bumpRevision();
        ctx.emit("history:pushed", docId, batchEntry);
      },
      clear(docId) {
        const h = histories.get(docId);
        if (h === void 0) return;
        h.undoStack.length = 0;
        h.redoStack.length = 0;
        h.batch = null;
        bumpRevision();
        ctx.emit("history:cleared", docId);
      },
      undoSize(docId) {
        return histories.get(docId)?.undoStack.length ?? 0;
      },
      redoSize(docId) {
        return histories.get(docId)?.redoStack.length ?? 0;
      }
    };
    ctx.registerCommand({
      id: "history.undo",
      label: "Undo",
      shortcut: "Ctrl+Z",
      category: "Edit",
      enabled: canUndo$,
      execute: () => {
        const handle = document2.activeDocument.peek();
        if (handle !== null) {
          void capability.undo(handle.id);
        }
      }
    });
    ctx.registerCommand({
      id: "history.redo",
      label: "Redo",
      shortcut: "Ctrl+Y",
      category: "Edit",
      enabled: canRedo$,
      execute: () => {
        const handle = document2.activeDocument.peek();
        if (handle !== null) {
          void capability.redo(handle.id);
        }
      }
    });
    return capability;
  }
});

// src/plugins/signature-plugin.ts
var signaturePlugin = definePlugin({
  id: "signature",
  provides: ["signature"],
  requires: ["document"],
  optional: [],
  setup(ctx) {
    ctx.require("document");
    return {
      async getCount(docId) {
        return ctx.engine.workerProxy.getSignatureCount(docId);
      },
      async getInfo(docId, sigIndex) {
        return ctx.engine.workerProxy.getSignatureInfo(docId, sigIndex);
      },
      async getAllInfo(docId) {
        const count = await ctx.engine.workerProxy.getSignatureCount(docId);
        const results = [];
        for (let i = 0; i < count; i++) {
          results.push(await ctx.engine.workerProxy.getSignatureInfo(docId, i));
        }
        return results;
      }
    };
  }
});

// src/plugins/attachment-plugin.ts
var attachmentPlugin = definePlugin({
  id: "attachment",
  provides: ["attachment"],
  requires: ["document"],
  optional: [],
  setup(ctx) {
    ctx.require("document");
    return {
      async getCount(docId) {
        return ctx.engine.workerProxy.getAttachmentCount(docId);
      },
      async list(docId) {
        const count = await ctx.engine.workerProxy.getAttachmentCount(docId);
        const results = [];
        for (let i = 0; i < count; i++) {
          results.push(await ctx.engine.workerProxy.getAttachmentInfo(docId, i));
        }
        return results;
      },
      async download(docId, index) {
        const info = await ctx.engine.workerProxy.getAttachmentInfo(docId, index);
        const data = await ctx.engine.workerProxy.getAttachmentData(docId, index);
        return { name: info.name, data };
      },
      async add(docId, name, data) {
        await ctx.engine.workerProxy.addAttachment(docId, name, data);
        ctx.emit("attachment:added", docId, name);
      },
      async delete(docId, index) {
        const info = await ctx.engine.workerProxy.getAttachmentInfo(docId, index);
        await ctx.engine.workerProxy.deleteAttachment(docId, index);
        ctx.emit("attachment:deleted", docId, info.name);
      }
    };
  }
});

// src/plugins/page-ops-plugin.ts
var pageOpsPlugin = definePlugin({
  id: "page-ops",
  provides: ["page-ops"],
  requires: ["document", "render", "viewport"],
  optional: ["history"],
  setup(ctx) {
    ctx.require("document");
    ctx.require("render");
    ctx.require("viewport");
    const history = ctx.optional("history");
    async function onPagesChanged(docId) {
      ctx.engine.renderPool?.invalidate(docId);
      const newSizes = await ctx.engine.workerProxy.getAllPageSizes(docId);
      ctx.emit("page-ops:pages-changed", docId, newSizes);
    }
    function pushHistory(docId, label, undoFn) {
      if (!history) return;
      history.push(docId, {
        id: uuid(),
        label,
        topic: "page-ops",
        timestamp: Date.now(),
        execute() {
        },
        undo: undoFn
      });
    }
    return {
      async deletePage(docId, pageIndex) {
        await ctx.engine.workerProxy.deletePage(docId, pageIndex);
        await onPagesChanged(docId);
        ctx.emit("page-ops:page-deleted", docId, pageIndex);
      },
      async insertBlankPage(docId, pageIndex, width, height) {
        await ctx.engine.workerProxy.insertBlankPage(docId, pageIndex, width, height);
        await onPagesChanged(docId);
        pushHistory(docId, "Insert page", () => {
          void ctx.engine.workerProxy.deletePage(docId, pageIndex);
        });
        ctx.emit("page-ops:page-inserted", docId, pageIndex);
      },
      async rotatePage(docId, pageIndex, degrees) {
        const current = await ctx.engine.workerProxy.getPageRotation(docId, pageIndex);
        const newRotation = (current + degrees / 90) % 4;
        await ctx.engine.workerProxy.rotatePage(docId, pageIndex, newRotation);
        await onPagesChanged(docId);
        ctx.emit("page-ops:page-rotated", docId, pageIndex, degrees);
      },
      async movePage(docId, fromIndex, toIndex) {
        await ctx.engine.workerProxy.movePage(docId, fromIndex, toIndex);
        await onPagesChanged(docId);
        pushHistory(docId, "Move page", () => {
          void ctx.engine.workerProxy.movePage(docId, toIndex, fromIndex);
        });
        ctx.emit("page-ops:page-moved", docId, fromIndex, toIndex);
      },
      async duplicatePage(docId, pageIndex) {
        await ctx.engine.workerProxy.duplicatePage(docId, pageIndex);
        await onPagesChanged(docId);
        pushHistory(docId, "Duplicate page", () => {
          void ctx.engine.workerProxy.deletePage(docId, pageIndex + 1);
        });
        ctx.emit("page-ops:page-duplicated", docId, pageIndex);
      },
      async flattenPage(docId, pageIndex) {
        const result = await ctx.engine.workerProxy.flattenPage(docId, pageIndex);
        if (result === 2) throw new Error("Failed to flatten page");
        ctx.engine.renderPool?.invalidate(docId);
        ctx.emit("page-ops:page-flattened", docId, pageIndex);
      }
    };
  }
});

// src/plugins/redaction-plugin.ts
import { FpdfAnnotSubtype } from "@truespar/lector-pdfium-wasm";
function toRedactionSpec(t) {
  const { rect, redaction } = t.data;
  const overlayColor = redaction?.overlayColor;
  return {
    rect: { left: rect.left, bottom: rect.bottom, right: rect.right, top: rect.top },
    overlayText: redaction?.overlayText,
    overlayColor: overlayColor ? { r: overlayColor.r, g: overlayColor.g, b: overlayColor.b } : void 0,
    overlayFontSize: redaction?.overlayFontSize
  };
}
var redactionPlugin = definePlugin({
  id: "redaction",
  provides: ["redaction"],
  requires: ["annotation", "document"],
  optional: ["history"],
  setup(ctx) {
    const annotation = ctx.require("annotation");
    ctx.require("document");
    const history = ctx.optional("history");
    return {
      async markForRedaction(docId, pageIndex, rect, options) {
        const tracked = await annotation.create(docId, pageIndex, {
          subtype: FpdfAnnotSubtype.SQUARE,
          tag: "redaction",
          rect,
          color: { r: 255, g: 0, b: 0, a: 77 },
          // Semi-transparent red
          redaction: {
            reason: options?.reason,
            overlayText: options?.overlayText,
            overlayColor: options?.overlayColor,
            applied: false
          }
        });
        if (history) {
          const id = tracked.id;
          history.push(docId, {
            id: uuid(),
            label: "Mark for redaction",
            topic: "redaction",
            timestamp: Date.now(),
            execute() {
            },
            undo() {
              void annotation.delete(docId, id);
            }
          });
        }
        ctx.emit("redaction:marked", docId, pageIndex, tracked.id);
        return tracked;
      },
      async applyRedactions(docId, pageIndex) {
        const specs = annotation.getForPage(docId, pageIndex).filter(
          (t) => (t.data.tag === "redaction" || t.data.subtype === FpdfAnnotSubtype.REDACT) && !t.data.redaction?.applied
        ).map(toRedactionSpec);
        if (specs.length === 0) return;
        await ctx.engine.workerProxy.applyRedactions(docId, pageIndex, specs, true);
        const pool = ctx.engine.renderPool;
        if (pool) {
          await pool.applyRedactions(docId, pageIndex, specs);
        }
        await annotation.reloadPage(docId, pageIndex);
        ctx.emit("redaction:applied", docId, pageIndex);
      },
      getMarkedRedactions(docId) {
        return annotation.getForDocument(docId).filter(
          (t) => (t.data.tag === "redaction" || t.data.subtype === FpdfAnnotSubtype.REDACT) && !t.data.redaction?.applied
        );
      },
      getPageRedactionCount(docId, pageIndex) {
        return annotation.getForPage(docId, pageIndex).filter(
          (t) => (t.data.tag === "redaction" || t.data.subtype === FpdfAnnotSubtype.REDACT) && !t.data.redaction?.applied
        ).length;
      }
    };
  }
});

// src/i18n/i18n-manager.ts
import { signal as signal14 } from "@truespar/lector-utils";

// src/i18n/translations/en.ts
var en = {
  // ── Toolbar ──
  "toolbar.menu": "Menu",
  "toolbar.open": "Open",
  "toolbar.close": "Close",
  "toolbar.save": "Save",
  "toolbar.print": "Print",
  "toolbar.export": "Export",
  "toolbar.sidebar": "Toggle sidebar",
  "toolbar.prevPage": "Previous page",
  "toolbar.nextPage": "Next page",
  "toolbar.zoomIn": "Zoom in",
  "toolbar.zoomOut": "Zoom out",
  "toolbar.fitWidth": "Fit width",
  "toolbar.fitPage": "Fit page",
  "toolbar.actualSize": "Actual size",
  "toolbar.search": "Search",
  "toolbar.fullscreen": "Fullscreen",
  "toolbar.annotate": "Annotate",
  "toolbar.undo": "Undo",
  "toolbar.redo": "Redo",
  "toolbar.more": "More",
  "toolbar.moreTools": "More tools",
  "toolbar.zoomPresets": "Zoom presets",
  "toolbar.skipToContent": "Skip to document",
  "toolbar.screenshot": "Export page as image",
  // ── Tools ──
  "tool.pointer": "Pointer",
  "tool.hand": "Pan",
  "tool.textSelect": "Select text",
  "tool.capture": "Capture region",
  // ── Marquee capture ──
  "capture.copy": "Copy",
  "capture.save": "Save",
  "capture.actions": "Capture actions",
  "capture.copiedToClipboard": "Copied to clipboard",
  "capture.copyFailed": "Copy to clipboard failed",
  "capture.failed": "Capture failed",
  // ── Drag and drop ──
  "dropzone.releaseToOpen": "Release to load PDF",
  "dropzone.releaseToOpenMany": "Release to load PDFs",
  "dropzone.notPdf": "Only PDF files can be opened",
  "dropzone.partialNotPdf": "Some files were ignored \u2014 only PDF files can be opened",
  // ── Split-pane view ──
  "split.horizontal": "Split horizontally",
  "split.vertical": "Split vertically",
  "split.closeAll": "Close split panes",
  "split.openToCompare": "Open PDF to compare",
  "split.dropHint": "or drop a PDF here",
  "split.emptySideLabel": "Empty",
  // ── Annotations ──
  "annotation.highlight": "Highlight",
  "annotation.underline": "Underline",
  "annotation.strikeout": "Strikethrough",
  "annotation.squiggly": "Squiggly",
  "annotation.ink": "Pen",
  "annotation.inkHighlighter": "Highlighter pen",
  "annotation.eraser": "Eraser",
  "annotation.freetext": "Text box",
  "annotation.stickyNote": "Sticky note",
  "annotation.insertText": "Insert text",
  "annotation.rectangle": "Rectangle",
  "annotation.circle": "Circle",
  "annotation.line": "Line",
  "annotation.arrow": "Arrow",
  "annotation.polygon": "Polygon",
  "annotation.polyline": "Polyline",
  "annotation.stamp": "Stamp",
  "annotation.image": "Image",
  "annotation.imageHelp": "Drop or pick an image to place on the page",
  "annotation.imagePickFile": "Choose image\u2026",
  "annotation.imageInvalid": "Could not load that image",
  "annotation.imageTooLarge": "Image too large (max 4 MB)",
  "annotation.callout": "Callout",
  "annotation.calloutHint": "Click to place text box, then click again for the leader point",
  "annotation.calloutEdit": "Double-click to edit text",
  "annotation.redaction": "Redaction",
  "annotation.applyRedaction": "Apply Redaction",
  "annotation.measurement": "Measurement",
  "annotation.signature": "Signature",
  "annotation.color": "Color",
  "annotation.fill": "Fill",
  "annotation.borderWidth": "Border width",
  "annotation.lineStyle": "Line style",
  "annotation.opacity": "Opacity",
  "annotation.fontSize": "Font size",
  "annotation.fontColor": "Font color",
  "annotation.textAlign": "Alignment",
  "annotation.alignLeft": "Left",
  "annotation.alignCenter": "Center",
  "annotation.alignRight": "Right",
  "annotation.delete": "Delete",
  "annotation.icon": "Icon",
  "annotation.thickness": "Thickness",
  "annotation.bringToFront": "Bring to front",
  "annotation.sendToBack": "Send to back",
  "annotation.group": "Group",
  "annotation.ungroup": "Ungroup",
  "annotation.multiSelectCount": "{count} annotations selected",
  // ── Line styles ──
  "lineStyle.solid": "Solid",
  "lineStyle.dashed": "Dashed",
  "lineStyle.dotted": "Dotted",
  // ── Context menu ──
  "contextMenu.insertPageBefore": "Insert page before",
  "contextMenu.insertPageAfter": "Insert page after",
  "contextMenu.duplicatePage": "Duplicate page",
  "contextMenu.rotateCW": "Rotate clockwise",
  "contextMenu.rotateCCW": "Rotate counter-clockwise",
  "contextMenu.deletePage": "Delete page",
  "contextMenu.copyText": "Copy text",
  "contextMenu.addNote": "Add note",
  "contextMenu.highlight": "Highlight",
  // ── Attachments ──
  "attachment.addFile": "Add file",
  // ── Page / navigation ──
  "page.viewer": "Document viewer",
  "page.goToPage": "Go to page",
  "page.goToPageN": "Go to page {page}",
  "page.documents": "Documents",
  // ── Freetext / overlays ──
  "annotation.doubleClickToEdit": "Double-click to edit",
  "annotation.defaultLabel": "Annotation",
  "form.button": "Button",
  "form.clickToSign": "Click to sign",
  // ── Sidebar ──
  "sidebar.thumbnails": "Pages",
  "sidebar.outline": "Bookmarks",
  "sidebar.annotations": "Annotations",
  "sidebar.attachments": "Attachments",
  "sidebar.signatures": "Signatures",
  "sidebar.layers": "Layers",
  "sidebar.comparison": "Changes",
  // ── Comparison ──
  "comparison.title": "Document comparison",
  "comparison.compare": "Compare",
  "comparison.exit": "Exit compare",
  "comparison.computing": "Comparing documents\u2026",
  "comparison.error": "Comparison failed: {error}",
  "comparison.requireSplit": "Open two PDFs in split view to compare them",
  "comparison.identical": "Both documents are identical \u2014 no changes detected",
  "comparison.totalChanges": "{count} change",
  "comparison.totalChangesPlural": "{count} changes",
  "comparison.noChangesOnPage": "No changes on this page",
  "comparison.pageHeading": "Page {a} \u2194 Page {b}",
  "comparison.pageHeadingInsert": "Page {b} (added)",
  "comparison.pageHeadingDelete": "Page {a} (removed)",
  "comparison.pageHeadingMismatch": "Page {a} \u2194 Page {b} (mixed text/scan)",
  "comparison.pageHeadingRegion": "Page {a} \u2194 Page {b} (visual)",
  "comparison.changeInsert": "Added",
  "comparison.changeDelete": "Removed",
  "comparison.changeReplace": "Changed",
  "comparison.changeRegion": "Visual change",
  "comparison.filterAll": "All",
  "comparison.filterInsert": "Added",
  "comparison.filterDelete": "Removed",
  "comparison.filterReplace": "Changed",
  "comparison.prevChange": "Previous change",
  "comparison.nextChange": "Next change",
  "comparison.syncScroll": "Sync scroll",
  "comparison.changeOf": "{n} of {total}",
  "comparison.openSidebar": "Open changes panel",
  // ── Comments ──
  "comment.addComment": "Add a comment...",
  "comment.reply": "Reply...",
  "comment.replyOrMention": "Reply or use @ to mention...",
  "comment.commentOrMention": "Comment or use @ to mention...",
  "comment.edit": "Edit",
  "comment.delete": "Delete",
  "comment.cancel": "Cancel",
  "comment.post": "Post",
  "comment.save": "Save",
  "comment.resolve": "Resolve",
  "comment.reopen": "Reopen",
  "comment.edited": "(edited)",
  // ── Comments sidebar ──
  "comments.title": "Comments",
  "comments.empty": "No comments yet.\nClick any annotation in the page to start a thread.",
  "comments.noDocument": "No document open",
  "comments.noBody": "No text",
  "comments.reply": "reply",
  "comments.replies": "replies",
  "comments.filter.all": "All",
  "comments.filter.open": "Open",
  "comments.filter.resolved": "Resolved",
  "comments.filter.mine": "Mine",
  "comments.filter.mentions": "@ me",
  "comments.sort.page": "By page",
  "comments.sort.date": "By date",
  "comments.sort.author": "By author",
  "common.page": "Page",
  "common.close": "Close",
  "toolbar.comments": "Toggle comments",
  // ── Comment status ──
  "commentStatus.open": "Open",
  "commentStatus.accepted": "Accepted",
  "commentStatus.rejected": "Rejected",
  "commentStatus.completed": "Completed",
  "commentStatus.cancelled": "Cancelled",
  "commentStatus.resolved": "Resolved",
  // ── Page operations ──
  "pageOps.delete": "Delete page",
  "pageOps.rotate": "Rotate page",
  "pageOps.rotateCW": "Rotate clockwise",
  "pageOps.rotateCCW": "Rotate counter-clockwise",
  "pageOps.insert": "Insert blank page",
  "pageOps.duplicate": "Duplicate page",
  "pageOps.move": "Move page",
  // ── Search ──
  "search.placeholder": "Search in document...",
  "search.noResults": "No results",
  "search.matchCase": "Match case",
  "search.matchWholeWord": "Whole words",
  "search.resultsCount": "{current} of {total}",
  "search.previous": "Previous match (Shift+Enter)",
  "search.next": "Next match (Enter)",
  "search.close": "Close search (Esc)",
  // ── Pagination ──
  "pagination.of": "of",
  "pagination.page": "Page",
  "pagination.pageOf": "Page {current} of {total}",
  // ── Zoom ──
  "zoom.level": "{value}%",
  // ── Layout ──
  "layout.single": "Single page",
  "layout.continuous": "Continuous",
  "layout.double": "Two-page spread",
  // ── Theme ──
  "theme.light": "Light",
  "theme.dark": "Dark",
  "theme.system": "System",
  // ── Stamps ──
  "stamp.approved": "Approved",
  "stamp.notApproved": "Not Approved",
  "stamp.draft": "Draft",
  "stamp.final": "Final",
  "stamp.completed": "Completed",
  "stamp.confidential": "Confidential",
  "stamp.forPublicRelease": "For Public Release",
  "stamp.notForPublicRelease": "Not For Public Release",
  "stamp.forComment": "For Comment",
  "stamp.void": "Void",
  "stamp.asIs": "As Is",
  "stamp.departmental": "Departmental",
  "stamp.experimental": "Experimental",
  "stamp.expired": "Expired",
  "stamp.informationOnly": "Information Only",
  "stamp.preliminaryResults": "Preliminary Results",
  "stamp.sold": "Sold",
  "stamp.topSecret": "Top Secret",
  // ── Signatures ──
  "signature.signed": "Signed",
  "signature.unsigned": "Unsigned",
  "signature.valid": "Valid signature",
  "signature.invalid": "Invalid signature",
  "signature.unknown": "Signature validity unknown",
  // ── Measurement ──
  "measurement.distance": "Distance",
  "measurement.area": "Area",
  "measurement.perimeter": "Perimeter",
  "measurement.calibrate": "Set scale\u2026",
  "measurement.calibrateTitle": "Calibrate measurement scale",
  "measurement.calibrateDesc": "Define how PDF distances map to real-world units.",
  "measurement.currentScale": "Current scale: {source} {sourceUnit} = {target} {targetUnit}",
  "measurement.pdfDistance": "PDF distance",
  "measurement.realDistance": "Real distance",
  "measurement.unit": "Unit",
  "measurement.precision": "Decimals",
  "measurement.clearScale": "Clear scale",
  "toast.calibrationSaved": "Measurement scale saved",
  "toast.calibrationCleared": "Measurement scale cleared",
  "toast.calibrationInvalid": "Both distances must be greater than zero",
  // ── Errors ──
  "error.loadFailed": "Failed to load document",
  "error.passwordRequired": "This document is password-protected",
  "error.renderFailed": "Failed to render page",
  // ── Misc ──
  "misc.loading": "Loading...",
  "misc.unknown": "Unknown",
  "misc.none": "None",
  // ── Common UI ──
  "common.cancel": "Cancel",
  "common.save": "Save",
  "common.open": "Open",
  "common.delete": "Delete",
  "common.apply": "Apply",
  "common.ok": "OK",
  "common.yes": "Yes",
  "common.no": "No",
  "common.selectAll": "Select all",
  "common.download": "Download",
  // ── Toolbar extras (shortcut hints) ──
  "toolbar.undoShortcut": "Undo (Ctrl+Z)",
  "toolbar.redoShortcut": "Redo (Ctrl+Y)",
  "toolbar.searchShortcut": "Search (Ctrl+F)",
  "toolbar.passwordProtect": "Password Protect",
  "toolbar.signDocument": "Sign Document",
  // ── Zoom control tooltips ──
  "zoom.zoomIn": "Zoom In",
  "zoom.zoomOut": "Zoom Out",
  // ── Toasts / ephemeral feedback ──
  "toast.bookmarkAdded": "Bookmark added",
  "toast.bookmarkDeleted": "Bookmark deleted",
  "toast.attached": "Attached: {name}",
  "toast.removed": "Removed: {name}",
  "toast.noRedactions": "No redactions to apply",
  "toast.redactionsApplied": "{count} redaction applied",
  "toast.redactionsAppliedPlural": "{count} redactions applied",
  "toast.redactionsFailed": "Failed to apply redactions: {error}",
  "toast.noDocument": "No document open",
  "toast.protectedSaved": "Protected document saved",
  "toast.invalidPageRange": "Invalid page range",
  "toast.printFailed": "Print failed: {error}",
  "toast.documentSaved": "Document saved",
  "toast.saveFailed": "Save failed: {error}",
  "toast.pageExported": "Page {page} exported",
  "toast.exportFailed": "Export failed: {error}",
  "toast.signatureMissing": "Please create a signature or upload a certificate",
  "toast.signatureApplied": "Signature applied",
  "toast.signedSaved": "Signed document saved",
  "toast.copied": "Copied",
  "toast.openFailed": "Failed to open: {error}",
  "toast.openNamedFailed": "Failed to open {name}: {error}",
  // ── Bookmarks panel ──
  "bookmarks.add": "Add Bookmark",
  "bookmarks.empty": "No bookmarks in this document",
  "bookmarks.delete": "Delete",
  "bookmarks.rename": "Rename",
  "bookmarks.dragHint": "Drag to reorder",
  "bookmarks.navigationPluginMissing": "Navigation plugin not loaded",
  "toast.bookmarkRenamed": "Bookmark renamed",
  "toast.bookmarkMoved": "Bookmark moved",
  // ── Annotation presets ──
  "presets.tooltip": "Annotation presets",
  "presets.none": "No preset",
  "presets.empty": "No presets yet",
  "presets.saveCurrent": "Save current style\u2026",
  "presets.namePrompt": "Name for this preset:",
  "presets.dialogTitle": "Save preset",
  "presets.namePlaceholder": "My preset",
  "presets.nameRequired": "Please enter a name for the preset",
  "toast.presetSaved": "Preset saved",
  "toast.presetDeleted": "Preset deleted",
  // ── Annotations list panel ──
  "annotations.empty": "No annotations in this document",
  "annotations.pluginMissing": "Annotation plugin not loaded",
  "annotations.resolved": "Resolved",
  "annotations.pageHeader": "Page {page}",
  // ── Attachments panel ──
  "attachments.empty": "No attachments in this document",
  "attachments.pluginMissing": "Attachment plugin not loaded",
  // ── Signatures panel ──
  "signatures.empty": "No digital signatures in this document",
  "signatures.pluginMissing": "Signature plugin not loaded",
  "signatures.validating": "Validating signatures\u2026",
  "signatures.validatingShort": "Validating\u2026",
  "signatures.validationError": "Validation error: {error}",
  "signatures.itemTitle": "Signature {index}",
  "signatures.badgeValid": "Valid",
  "signatures.badgeInvalid": "Invalid",
  "signatures.badgeError": "Error",
  "signatures.badgeUnknown": "Unknown",
  "signatures.integrityVerified": "Document integrity verified",
  "signatures.integrityModified": "Document modified after signing",
  "signatures.integrityUnknown": "Integrity unknown",
  "signatures.cryptoValid": "Signature cryptographically valid",
  "signatures.signatureValid": "Signature valid",
  "signatures.signatureInvalid": "Signature invalid",
  "signatures.certNotTrusted": "Certificate not trusted",
  "signatures.verificationNeeded": "Signature verification needed",
  "signatures.certExpired": "Certificate has expired",
  "signatures.certSelfSigned": "Self-signed certificate",
  "signatures.certIssuer": "Issuer: {issuer}",
  "signatures.algorithm": "Algorithm: {algorithm}",
  "signatures.format": "Format: {format}",
  "signatures.reason": "Reason: {reason}",
  "signatures.signedAt": "Signed: {time}",
  "signatures.covers": "Covers: {size}",
  "signatures.validationHint": "Validation: {error}",
  "signatures.badgeStatusSingle": "Digital signature",
  "signatures.badgeStatusMany": "{count} digital signatures",
  "signatures.fullDetailsHint": "Click for full details",
  "signatures.perm.approval": "Approval signature (further changes allowed)",
  "signatures.perm.noChanges": "Certified \u2014 no changes allowed",
  "signatures.perm.formSigning": "Certified \u2014 form fill & signing only",
  "signatures.perm.formCommenting": "Certified \u2014 form fill, signing & comments",
  "signatures.perm.unknown": "Permission level: {level}",
  "signatures.format.pkcs7Detached": "PKCS#7 detached (Adobe)",
  "signatures.format.pkcs7Sha1": "PKCS#7 SHA-1 (Adobe, legacy)",
  "signatures.format.x509RsaSha1": "X.509 RSA SHA-1 (Adobe, legacy)",
  "signatures.format.cadesDetached": "CAdES detached (PAdES)",
  "signatures.format.rfc3161": "RFC 3161 timestamp",
  // ── Layers panel ──
  "layers.empty": "No layers in this document",
  "layers.pluginMissing": "Layer plugin not loaded",
  // ── Apply Redactions dialog ──
  "redact.dialogTitle": "Apply Redactions",
  "redact.applyButton": "Apply Redactions",
  "redact.applying": "Applying...",
  "redact.summarySingle": "<strong>1</strong> redaction will be applied.",
  "redact.summaryPlural": "<strong>{count}</strong> redactions will be applied.",
  "redact.warning": "This permanently removes all content under the redacted areas. This action <strong>cannot be undone</strong>.",
  // ── Password (open) dialog ──
  "password.openTitle": "Password Required",
  "password.openPlaceholder": "Enter password",
  "password.opening": "Opening...",
  "password.incorrect": "Incorrect password. Please try again.",
  "password.openFailed": "Failed to open: {error}",
  "password.descNamed": 'The document "{name}" is password protected.',
  "password.desc": "This document is password protected.",
  // ── Password protect dialog ──
  "protect.dialogTitle": "Password Protect Document",
  "protect.userLabel": "Open password (required to view)",
  "protect.userPlaceholder": "Enter password",
  "protect.confirmLabel": "Confirm password",
  "protect.confirmPlaceholder": "Confirm password",
  "protect.ownerLabel": "Owner password (optional \u2014 for full access)",
  "protect.ownerPlaceholder": "Owner password (leave empty = same as open password)",
  "protect.permissionsLabel": "Permissions",
  "protect.applyButton": "Protect & Save",
  "protect.applying": "Protecting...",
  "protect.errorRequired": "Password is required",
  "protect.errorMismatch": "Passwords do not match",
  "protect.errorFailed": "Failed: {error}",
  "protect.perm.print": "Allow printing",
  "protect.perm.extract": "Allow copy/extract",
  "protect.perm.annotate": "Allow annotations & forms",
  "protect.perm.modify": "Allow content modification",
  // ── Print dialog ──
  "print.dialogTitle": "Print Document",
  "print.pagesLabel": "Pages",
  "print.allPages": "All pages (1-{total})",
  "print.customRange": "Custom range:",
  "print.rangePlaceholder": "e.g. 1-3, 5, 8-10",
  "print.printButton": "Print",
  "print.rendering": "Rendering...",
  // ── Signature dialog ──
  "sign.dialogTitle": "Sign Document",
  "sign.clearButton": "Clear",
  "sign.uploadPrompt": "Click to upload or drag image",
  "sign.typedPlaceholder": "Type your name",
  "sign.certLabel": "Digital Certificate (optional)",
  "sign.certUpload": "Upload PFX/P12...",
  "sign.certNone": "No certificate selected",
  "sign.certPasswordPlaceholder": "Certificate password",
  "sign.reasonPlaceholder": "Reason for signing (optional)",
  "sign.tsaPlaceholder": "TSA URL (optional, e.g. http://timestamp.digicert.com)",
  "sign.applyButton": "Apply Signature",
  "sign.applying": "Signing...",
  "sign.tab.draw": "Draw",
  "sign.tab.type": "Type",
  "sign.tab.image": "Image",
  "sign.mdp.approval": "Allow further changes (approval signature)",
  "sign.mdp.formCommenting": "Allow form fill-in, signing, and commenting",
  "sign.mdp.formSigning": "Allow form fill-in and signing only",
  "sign.mdp.noChanges": "No changes allowed (lock document)",
  // ── Split pane close buttons (tooltips) ──
  "split.closeThisPane": "Close this pane",
  "split.closeSplit": "Close split",
  "split.openPdf": "Open PDF",
  // ── Stamp picker ──
  "stamp.pickerTooltip": "Stamp"
};

// src/i18n/i18n-manager.ts
var I18nManager = class {
  #translations = /* @__PURE__ */ new Map();
  #locale$;
  #fallback = "en";
  constructor(initialLocale = "en") {
    this.#translations.set("en", en);
    this.#locale$ = signal14(initialLocale);
  }
  /** Current locale as a reactive signal. */
  get locale() {
    return this.#locale$;
  }
  /** Switch the active locale. */
  setLocale(locale) {
    this.#locale$.value = locale;
  }
  /**
   * Add or merge translations for a locale.
   * Existing keys for the same locale are overwritten.
   */
  addTranslations(locale, translations) {
    const existing = this.#translations.get(locale);
    this.#translations.set(locale, { ...existing, ...translations });
  }
  /** Check if a locale has been registered. */
  hasLocale(locale) {
    return this.#translations.has(locale);
  }
  /** Get all registered locale IDs. */
  getLocales() {
    return [...this.#translations.keys()];
  }
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
  t(key, params) {
    const locale = this.#locale$.peek();
    const map = this.#translations.get(locale);
    let value = map?.[key] ?? this.#translations.get(this.#fallback)?.[key] ?? key;
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        value = value.replaceAll(`{${k}}`, String(v));
      }
    }
    return value;
  }
  [Symbol.dispose]() {
    this.#translations.clear();
  }
};

// src/i18n/translations/sv.ts
var sv = {
  // ── Toolbar ──
  "toolbar.menu": "Meny",
  "toolbar.open": "\xD6ppna",
  "toolbar.close": "St\xE4ng",
  "toolbar.save": "Spara",
  "toolbar.print": "Skriv ut",
  "toolbar.export": "Exportera",
  "toolbar.sidebar": "Visa/d\xF6lj sidopanel",
  "toolbar.prevPage": "F\xF6reg\xE5ende sida",
  "toolbar.nextPage": "N\xE4sta sida",
  "toolbar.zoomIn": "Zooma in",
  "toolbar.zoomOut": "Zooma ut",
  "toolbar.fitWidth": "Anpassa till bredd",
  "toolbar.fitPage": "Anpassa till sida",
  "toolbar.actualSize": "Verklig storlek",
  "toolbar.search": "S\xF6k",
  "toolbar.fullscreen": "Helsk\xE4rm",
  "toolbar.annotate": "Anteckna",
  "toolbar.undo": "\xC5ngra",
  "toolbar.redo": "G\xF6r om",
  "toolbar.more": "Mer",
  "toolbar.moreTools": "Fler verktyg",
  "toolbar.zoomPresets": "Zoomf\xF6rval",
  "toolbar.skipToContent": "Hoppa till dokumentet",
  "toolbar.screenshot": "Exportera sida som bild",
  // ── Tools ──
  "tool.pointer": "Pekare",
  "tool.hand": "Panorera",
  "tool.textSelect": "Markera text",
  "tool.capture": "F\xE5nga region",
  // ── Marquee capture ──
  "capture.copy": "Kopiera",
  "capture.save": "Spara",
  "capture.actions": "F\xE5ngst\xE5tg\xE4rder",
  "capture.copiedToClipboard": "Kopierat till urklipp",
  "capture.copyFailed": "Det gick inte att kopiera till urklipp",
  "capture.failed": "F\xE5ngst misslyckades",
  // ── Drag and drop ──
  "dropzone.releaseToOpen": "Sl\xE4pp f\xF6r att l\xE4sa in PDF",
  "dropzone.releaseToOpenMany": "Sl\xE4pp f\xF6r att l\xE4sa in PDF-filer",
  "dropzone.notPdf": "Endast PDF-filer kan \xF6ppnas",
  "dropzone.partialNotPdf": "Vissa filer ignorerades \u2014 endast PDF-filer kan \xF6ppnas",
  // ── Split-pane view ──
  "split.horizontal": "Dela horisontellt",
  "split.vertical": "Dela vertikalt",
  "split.closeAll": "St\xE4ng delade paneler",
  "split.openToCompare": "\xD6ppna PDF f\xF6r att j\xE4mf\xF6ra",
  "split.dropHint": "eller sl\xE4pp en PDF h\xE4r",
  "split.emptySideLabel": "Tom",
  // ── Annotations ──
  "annotation.highlight": "Markering",
  "annotation.underline": "Understrykning",
  "annotation.strikeout": "Genomstrykning",
  "annotation.squiggly": "V\xE5gig understrykning",
  "annotation.ink": "Penna",
  "annotation.inkHighlighter": "\xD6verstrykningspenna",
  "annotation.eraser": "Suddgummi",
  "annotation.freetext": "Textruta",
  "annotation.stickyNote": "Anteckningslapp",
  "annotation.insertText": "Infoga text",
  "annotation.rectangle": "Rektangel",
  "annotation.circle": "Cirkel",
  "annotation.line": "Linje",
  "annotation.arrow": "Pil",
  "annotation.polygon": "Polygon",
  "annotation.polyline": "Polylinje",
  "annotation.stamp": "St\xE4mpel",
  "annotation.image": "Bild",
  "annotation.imageHelp": "Sl\xE4pp eller v\xE4lj en bild att placera p\xE5 sidan",
  "annotation.imagePickFile": "V\xE4lj bild\u2026",
  "annotation.imageInvalid": "Det gick inte att l\xE4sa in bilden",
  "annotation.imageTooLarge": "Bilden \xE4r f\xF6r stor (max 4 MB)",
  "annotation.callout": "Pratbubbla",
  "annotation.calloutHint": "Klicka f\xF6r att placera textrutan, klicka sedan igen f\xF6r ledarpunkten",
  "annotation.calloutEdit": "Dubbelklicka f\xF6r att redigera text",
  "annotation.redaction": "Maskering",
  "annotation.applyRedaction": "Till\xE4mpa maskering",
  "annotation.measurement": "M\xE4tning",
  "annotation.signature": "Signatur",
  "annotation.color": "F\xE4rg",
  "annotation.fill": "Fyllning",
  "annotation.borderWidth": "Kantbredd",
  "annotation.lineStyle": "Linjestil",
  "annotation.opacity": "Opacitet",
  "annotation.delete": "Ta bort",
  "annotation.icon": "Ikon",
  "annotation.thickness": "Tjocklek",
  "annotation.bringToFront": "Flytta l\xE4ngst fram",
  "annotation.sendToBack": "Flytta l\xE4ngst bak",
  "annotation.group": "Gruppera",
  "annotation.ungroup": "Dela upp grupp",
  "annotation.multiSelectCount": "{count} anteckningar markerade",
  // ── Line styles ──
  "lineStyle.solid": "Heldragen",
  "lineStyle.dashed": "Streckad",
  "lineStyle.dotted": "Prickad",
  // ── Context menu ──
  "contextMenu.insertPageBefore": "Infoga sida f\xF6re",
  "contextMenu.insertPageAfter": "Infoga sida efter",
  "contextMenu.duplicatePage": "Duplicera sida",
  "contextMenu.rotateCW": "Rotera medurs",
  "contextMenu.rotateCCW": "Rotera moturs",
  "contextMenu.deletePage": "Ta bort sida",
  "contextMenu.copyText": "Kopiera text",
  "contextMenu.addNote": "L\xE4gg till anteckning",
  "contextMenu.highlight": "Markera",
  // ── Attachments ──
  "attachment.addFile": "L\xE4gg till fil",
  // ── Page / navigation ──
  "page.viewer": "Dokumentvisare",
  "page.goToPage": "G\xE5 till sida",
  "page.goToPageN": "G\xE5 till sida {page}",
  "page.documents": "Dokument",
  // ── Freetext / overlays ──
  "annotation.doubleClickToEdit": "Dubbelklicka f\xF6r att redigera",
  "annotation.defaultLabel": "Anteckning",
  "form.button": "Knapp",
  "form.clickToSign": "Klicka f\xF6r att signera",
  // ── Sidebar ──
  "sidebar.thumbnails": "Sidor",
  "sidebar.outline": "Bokm\xE4rken",
  "sidebar.annotations": "Anteckningar",
  "sidebar.attachments": "Bilagor",
  "sidebar.signatures": "Signaturer",
  "sidebar.layers": "Lager",
  "sidebar.comparison": "\xC4ndringar",
  // ── Comparison ──
  "comparison.title": "Dokumentj\xE4mf\xF6relse",
  "comparison.compare": "J\xE4mf\xF6r",
  "comparison.exit": "Avsluta j\xE4mf\xF6relse",
  "comparison.computing": "J\xE4mf\xF6r dokument\u2026",
  "comparison.error": "J\xE4mf\xF6relse misslyckades: {error}",
  "comparison.requireSplit": "\xD6ppna tv\xE5 PDF-filer i delad vy f\xF6r att j\xE4mf\xF6ra dem",
  "comparison.identical": "B\xE5da dokumenten \xE4r identiska \u2014 inga \xE4ndringar hittades",
  "comparison.totalChanges": "{count} \xE4ndring",
  "comparison.totalChangesPlural": "{count} \xE4ndringar",
  "comparison.noChangesOnPage": "Inga \xE4ndringar p\xE5 denna sida",
  "comparison.pageHeading": "Sida {a} \u2194 Sida {b}",
  "comparison.pageHeadingInsert": "Sida {b} (tillagd)",
  "comparison.pageHeadingDelete": "Sida {a} (borttagen)",
  "comparison.pageHeadingMismatch": "Sida {a} \u2194 Sida {b} (blandat text/bild)",
  "comparison.pageHeadingRegion": "Sida {a} \u2194 Sida {b} (visuell)",
  "comparison.changeInsert": "Tillagd",
  "comparison.changeDelete": "Borttagen",
  "comparison.changeReplace": "\xC4ndrad",
  "comparison.changeRegion": "Visuell \xE4ndring",
  "comparison.filterAll": "Alla",
  "comparison.filterInsert": "Tillagda",
  "comparison.filterDelete": "Borttagna",
  "comparison.filterReplace": "\xC4ndrade",
  "comparison.prevChange": "F\xF6reg\xE5ende \xE4ndring",
  "comparison.nextChange": "N\xE4sta \xE4ndring",
  "comparison.syncScroll": "Synkronisera rullning",
  "comparison.changeOf": "{n} av {total}",
  "comparison.openSidebar": "\xD6ppna \xE4ndringspanelen",
  // ── Comments ──
  "comment.addComment": "L\xE4gg till en kommentar...",
  "comment.reply": "Svara...",
  "comment.replyOrMention": "Svara eller anv\xE4nd @ f\xF6r att n\xE4mna...",
  "comment.commentOrMention": "Kommentera eller anv\xE4nd @ f\xF6r att n\xE4mna...",
  "comment.edit": "Redigera",
  "comment.delete": "Ta bort",
  "comment.cancel": "Avbryt",
  "comment.post": "Publicera",
  "comment.save": "Spara",
  "comment.resolve": "L\xF6s",
  "comment.reopen": "\xD6ppna igen",
  "comment.edited": "(redigerad)",
  // ── Comments sidebar ──
  "comments.title": "Kommentarer",
  "comments.empty": "Inga kommentarer \xE4nnu.\nKlicka p\xE5 en anteckning p\xE5 sidan f\xF6r att starta en tr\xE5d.",
  "comments.noDocument": "Inget dokument \xF6ppet",
  "comments.noBody": "Ingen text",
  "comments.reply": "svar",
  "comments.replies": "svar",
  "comments.filter.all": "Alla",
  "comments.filter.open": "\xD6ppna",
  "comments.filter.resolved": "L\xF6sta",
  "comments.filter.mine": "Mina",
  "comments.filter.mentions": "@ mig",
  "comments.sort.page": "Per sida",
  "comments.sort.date": "Per datum",
  "comments.sort.author": "Per f\xF6rfattare",
  "common.page": "Sida",
  "common.close": "St\xE4ng",
  "toolbar.comments": "Visa/d\xF6lj kommentarer",
  // ── Comment status ──
  "commentStatus.open": "\xD6ppen",
  "commentStatus.accepted": "Godk\xE4nd",
  "commentStatus.rejected": "Avvisad",
  "commentStatus.completed": "Slutf\xF6rd",
  "commentStatus.cancelled": "Avbruten",
  "commentStatus.resolved": "L\xF6st",
  // ── Page operations ──
  "pageOps.delete": "Ta bort sida",
  "pageOps.rotate": "Rotera sida",
  "pageOps.rotateCW": "Rotera medurs",
  "pageOps.rotateCCW": "Rotera moturs",
  "pageOps.insert": "Infoga tom sida",
  "pageOps.duplicate": "Duplicera sida",
  "pageOps.move": "Flytta sida",
  // ── Search ──
  "search.placeholder": "S\xF6k i dokumentet...",
  "search.noResults": "Inga resultat",
  "search.matchCase": "Matcha versaler/gemener",
  "search.matchWholeWord": "Hela ord",
  "search.resultsCount": "{current} av {total}",
  "search.previous": "F\xF6reg\xE5ende tr\xE4ff (Shift+Enter)",
  "search.next": "N\xE4sta tr\xE4ff (Enter)",
  "search.close": "St\xE4ng s\xF6kning (Esc)",
  // ── Pagination ──
  "pagination.of": "av",
  "pagination.page": "Sida",
  "pagination.pageOf": "Sida {current} av {total}",
  // ── Zoom ──
  "zoom.level": "{value}%",
  // ── Layout ──
  "layout.single": "Enkel sida",
  "layout.continuous": "Kontinuerlig",
  "layout.double": "Tv\xE5sidig vy",
  // ── Theme ──
  "theme.light": "Ljust",
  "theme.dark": "M\xF6rkt",
  "theme.system": "System",
  // ── Stamps ──
  "stamp.approved": "Godk\xE4nd",
  "stamp.notApproved": "Ej godk\xE4nd",
  "stamp.draft": "Utkast",
  "stamp.final": "Slutgiltig",
  "stamp.completed": "Slutf\xF6rd",
  "stamp.confidential": "Konfidentiellt",
  "stamp.forPublicRelease": "F\xF6r offentligg\xF6rande",
  "stamp.notForPublicRelease": "Ej f\xF6r offentligg\xF6rande",
  "stamp.forComment": "F\xF6r kommentar",
  "stamp.void": "Ogiltig",
  "stamp.asIs": "I befintligt skick",
  "stamp.departmental": "Avdelning",
  "stamp.experimental": "Experimentell",
  "stamp.expired": "Utg\xE5ngen",
  "stamp.informationOnly": "Endast information",
  "stamp.preliminaryResults": "Prelimin\xE4ra resultat",
  "stamp.sold": "S\xE5ld",
  "stamp.topSecret": "Topphemligt",
  // ── Signatures ──
  "signature.signed": "Signerad",
  "signature.unsigned": "Osignerad",
  "signature.valid": "Giltig signatur",
  "signature.invalid": "Ogiltig signatur",
  "signature.unknown": "Signaturens giltighet ok\xE4nd",
  // ── Measurement ──
  "measurement.distance": "Avst\xE5nd",
  "measurement.area": "Area",
  "measurement.perimeter": "Omkrets",
  "measurement.calibrate": "St\xE4ll in skala\u2026",
  "measurement.calibrateTitle": "Kalibrera m\xE4tskala",
  "measurement.calibrateDesc": "Definiera hur PDF-avst\xE5nd motsvarar verkliga enheter.",
  "measurement.currentScale": "Aktuell skala: {source} {sourceUnit} = {target} {targetUnit}",
  "measurement.pdfDistance": "PDF-avst\xE5nd",
  "measurement.realDistance": "Verkligt avst\xE5nd",
  "measurement.unit": "Enhet",
  "measurement.precision": "Decimaler",
  "measurement.clearScale": "Rensa skala",
  "toast.calibrationSaved": "M\xE4tskala sparad",
  "toast.calibrationCleared": "M\xE4tskala rensad",
  "toast.calibrationInvalid": "B\xE5da avst\xE5nden m\xE5ste vara st\xF6rre \xE4n noll",
  // ── Errors ──
  "error.loadFailed": "Det gick inte att l\xE4sa in dokumentet",
  "error.passwordRequired": "Detta dokument \xE4r l\xF6senordsskyddat",
  "error.renderFailed": "Det gick inte att rendera sidan",
  // ── Misc ──
  "misc.loading": "L\xE4ser in...",
  "misc.unknown": "Ok\xE4nd",
  "misc.none": "Ingen",
  // ── Common UI ──
  "common.cancel": "Avbryt",
  "common.save": "Spara",
  "common.open": "\xD6ppna",
  "common.delete": "Ta bort",
  "common.apply": "Till\xE4mpa",
  "common.ok": "OK",
  "common.yes": "Ja",
  "common.no": "Nej",
  "common.selectAll": "Markera alla",
  "common.download": "Ladda ned",
  // ── Toolbar extras (shortcut hints) ──
  "toolbar.undoShortcut": "\xC5ngra (Ctrl+Z)",
  "toolbar.redoShortcut": "G\xF6r om (Ctrl+Y)",
  "toolbar.searchShortcut": "S\xF6k (Ctrl+F)",
  "toolbar.passwordProtect": "L\xF6senordsskydda",
  "toolbar.signDocument": "Signera dokument",
  // ── Zoom control tooltips ──
  "zoom.zoomIn": "Zooma in",
  "zoom.zoomOut": "Zooma ut",
  // ── Toasts / ephemeral feedback ──
  "toast.bookmarkAdded": "Bokm\xE4rke tillagt",
  "toast.bookmarkDeleted": "Bokm\xE4rke borttaget",
  "toast.attached": "Bifogad: {name}",
  "toast.removed": "Borttagen: {name}",
  "toast.noRedactions": "Inga maskeringar att till\xE4mpa",
  "toast.redactionsApplied": "{count} maskering till\xE4mpad",
  "toast.redactionsAppliedPlural": "{count} maskeringar till\xE4mpade",
  "toast.redactionsFailed": "Det gick inte att till\xE4mpa maskeringar: {error}",
  "toast.noDocument": "Inget dokument \xF6ppet",
  "toast.protectedSaved": "Skyddat dokument sparat",
  "toast.invalidPageRange": "Ogiltigt sidintervall",
  "toast.printFailed": "Utskrift misslyckades: {error}",
  "toast.documentSaved": "Dokument sparat",
  "toast.saveFailed": "Det gick inte att spara: {error}",
  "toast.pageExported": "Sida {page} exporterad",
  "toast.exportFailed": "Export misslyckades: {error}",
  "toast.signatureMissing": "Skapa en signatur eller ladda upp ett certifikat",
  "toast.signatureApplied": "Signatur tillagd",
  "toast.signedSaved": "Signerat dokument sparat",
  "toast.copied": "Kopierat",
  "toast.openFailed": "Det gick inte att \xF6ppna: {error}",
  "toast.openNamedFailed": "Det gick inte att \xF6ppna {name}: {error}",
  // ── Bookmarks panel ──
  "bookmarks.add": "L\xE4gg till bokm\xE4rke",
  "bookmarks.empty": "Inga bokm\xE4rken i detta dokument",
  "bookmarks.delete": "Ta bort",
  "bookmarks.rename": "Byt namn",
  "bookmarks.dragHint": "Dra f\xF6r att \xE4ndra ordning",
  "bookmarks.navigationPluginMissing": "Navigeringstill\xE4gget \xE4r inte laddat",
  "toast.bookmarkRenamed": "Bokm\xE4rke omd\xF6pt",
  "toast.bookmarkMoved": "Bokm\xE4rke flyttat",
  // ── Annotation presets ──
  "presets.tooltip": "Anteckningsf\xF6rval",
  "presets.none": "Inget f\xF6rval",
  "presets.empty": "Inga f\xF6rval \xE4nnu",
  "presets.saveCurrent": "Spara aktuell stil\u2026",
  "presets.namePrompt": "Namn p\xE5 detta f\xF6rval:",
  "presets.dialogTitle": "Spara f\xF6rval",
  "presets.namePlaceholder": "Mitt f\xF6rval",
  "presets.nameRequired": "Ange ett namn f\xF6r f\xF6rvalet",
  "toast.presetSaved": "F\xF6rval sparat",
  "toast.presetDeleted": "F\xF6rval borttaget",
  // ── Annotations list panel ──
  "annotations.empty": "Inga anteckningar i detta dokument",
  "annotations.pluginMissing": "Anteckningstill\xE4gget \xE4r inte laddat",
  "annotations.resolved": "L\xF6st",
  "annotations.pageHeader": "Sida {page}",
  // ── Attachments panel ──
  "attachments.empty": "Inga bilagor i detta dokument",
  "attachments.pluginMissing": "Bilagetill\xE4gget \xE4r inte laddat",
  // ── Signatures panel ──
  "signatures.empty": "Inga digitala signaturer i detta dokument",
  "signatures.pluginMissing": "Signaturtill\xE4gget \xE4r inte laddat",
  "signatures.validating": "Validerar signaturer\u2026",
  "signatures.validatingShort": "Validerar\u2026",
  "signatures.validationError": "Valideringsfel: {error}",
  "signatures.itemTitle": "Signatur {index}",
  "signatures.badgeValid": "Giltig",
  "signatures.badgeInvalid": "Ogiltig",
  "signatures.badgeError": "Fel",
  "signatures.badgeUnknown": "Ok\xE4nd",
  "signatures.integrityVerified": "Dokumentets integritet verifierad",
  "signatures.integrityModified": "Dokumentet \xE4ndrat efter signering",
  "signatures.integrityUnknown": "Integritet ok\xE4nd",
  "signatures.cryptoValid": "Signaturen \xE4r kryptografiskt giltig",
  "signatures.signatureValid": "Signaturen \xE4r giltig",
  "signatures.signatureInvalid": "Signaturen \xE4r ogiltig",
  "signatures.certNotTrusted": "Certifikatet \xE4r inte betrott",
  "signatures.verificationNeeded": "Signaturverifiering kr\xE4vs",
  "signatures.certExpired": "Certifikatet har utg\xE5tt",
  "signatures.certSelfSigned": "Sj\xE4lvsignerat certifikat",
  "signatures.certIssuer": "Utf\xE4rdare: {issuer}",
  "signatures.algorithm": "Algoritm: {algorithm}",
  "signatures.format": "Format: {format}",
  "signatures.reason": "Orsak: {reason}",
  "signatures.signedAt": "Signerat: {time}",
  "signatures.covers": "Omfattar: {size}",
  "signatures.validationHint": "Validering: {error}",
  "signatures.badgeStatusSingle": "Digital signatur",
  "signatures.badgeStatusMany": "{count} digitala signaturer",
  "signatures.fullDetailsHint": "Klicka f\xF6r fullst\xE4ndig information",
  "signatures.perm.approval": "Godk\xE4nnandesignatur (ytterligare \xE4ndringar till\xE5tna)",
  "signatures.perm.noChanges": "Certifierad \u2014 inga \xE4ndringar till\xE5tna",
  "signatures.perm.formSigning": "Certifierad \u2014 endast formul\xE4rifyllning och signering",
  "signatures.perm.formCommenting": "Certifierad \u2014 formul\xE4rifyllning, signering och kommentarer",
  "signatures.perm.unknown": "Beh\xF6righetsniv\xE5: {level}",
  "signatures.format.pkcs7Detached": "PKCS#7 frist\xE5ende (Adobe)",
  "signatures.format.pkcs7Sha1": "PKCS#7 SHA-1 (Adobe, \xE4ldre)",
  "signatures.format.x509RsaSha1": "X.509 RSA SHA-1 (Adobe, \xE4ldre)",
  "signatures.format.cadesDetached": "CAdES frist\xE5ende (PAdES)",
  "signatures.format.rfc3161": "RFC 3161-tidsst\xE4mpel",
  // ── Layers panel ──
  "layers.empty": "Inga lager i detta dokument",
  "layers.pluginMissing": "Lagertill\xE4gget \xE4r inte laddat",
  // ── Apply Redactions dialog ──
  "redact.dialogTitle": "Till\xE4mpa maskeringar",
  "redact.applyButton": "Till\xE4mpa maskeringar",
  "redact.applying": "Till\xE4mpar...",
  "redact.summarySingle": "<strong>1</strong> maskering kommer att till\xE4mpas.",
  "redact.summaryPlural": "<strong>{count}</strong> maskeringar kommer att till\xE4mpas.",
  "redact.warning": "Detta tar permanent bort allt inneh\xE5ll under de maskerade omr\xE5dena. Denna \xE5tg\xE4rd <strong>kan inte \xE5ngras</strong>.",
  // ── Password (open) dialog ──
  "password.openTitle": "L\xF6senord kr\xE4vs",
  "password.openPlaceholder": "Ange l\xF6senord",
  "password.opening": "\xD6ppnar...",
  "password.incorrect": "Felaktigt l\xF6senord. F\xF6rs\xF6k igen.",
  "password.openFailed": "Det gick inte att \xF6ppna: {error}",
  "password.descNamed": 'Dokumentet "{name}" \xE4r l\xF6senordsskyddat.',
  "password.desc": "Detta dokument \xE4r l\xF6senordsskyddat.",
  // ── Password protect dialog ──
  "protect.dialogTitle": "L\xF6senordsskydda dokument",
  "protect.userLabel": "\xD6ppningsl\xF6senord (kr\xE4vs f\xF6r att visa)",
  "protect.userPlaceholder": "Ange l\xF6senord",
  "protect.confirmLabel": "Bekr\xE4fta l\xF6senord",
  "protect.confirmPlaceholder": "Bekr\xE4fta l\xF6senord",
  "protect.ownerLabel": "\xC4garl\xF6senord (valfritt \u2014 f\xF6r fullst\xE4ndig \xE5tkomst)",
  "protect.ownerPlaceholder": "\xC4garl\xF6senord (l\xE4mna tomt = samma som \xF6ppningsl\xF6senord)",
  "protect.permissionsLabel": "Beh\xF6righeter",
  "protect.applyButton": "Skydda och spara",
  "protect.applying": "Skyddar...",
  "protect.errorRequired": "L\xF6senord kr\xE4vs",
  "protect.errorMismatch": "L\xF6senorden matchar inte",
  "protect.errorFailed": "Misslyckades: {error}",
  "protect.perm.print": "Till\xE5t utskrift",
  "protect.perm.extract": "Till\xE5t kopiering/extrahering",
  "protect.perm.annotate": "Till\xE5t anteckningar och formul\xE4r",
  "protect.perm.modify": "Till\xE5t inneh\xE5lls\xE4ndring",
  // ── Print dialog ──
  "print.dialogTitle": "Skriv ut dokument",
  "print.pagesLabel": "Sidor",
  "print.allPages": "Alla sidor (1-{total})",
  "print.customRange": "Anpassat intervall:",
  "print.rangePlaceholder": "t.ex. 1-3, 5, 8-10",
  "print.printButton": "Skriv ut",
  "print.rendering": "Renderar...",
  // ── Signature dialog ──
  "sign.dialogTitle": "Signera dokument",
  "sign.clearButton": "Rensa",
  "sign.uploadPrompt": "Klicka f\xF6r att ladda upp eller dra bild",
  "sign.typedPlaceholder": "Skriv ditt namn",
  "sign.certLabel": "Digitalt certifikat (valfritt)",
  "sign.certUpload": "Ladda upp PFX/P12...",
  "sign.certNone": "Inget certifikat valt",
  "sign.certPasswordPlaceholder": "Certifikatl\xF6senord",
  "sign.reasonPlaceholder": "Orsak till signering (valfritt)",
  "sign.tsaPlaceholder": "TSA-URL (valfritt, t.ex. http://timestamp.digicert.com)",
  "sign.applyButton": "Till\xE4mpa signatur",
  "sign.applying": "Signerar...",
  "sign.tab.draw": "Rita",
  "sign.tab.type": "Skriv",
  "sign.tab.image": "Bild",
  "sign.mdp.approval": "Till\xE5t ytterligare \xE4ndringar (godk\xE4nnandesignatur)",
  "sign.mdp.formCommenting": "Till\xE5t formul\xE4rifyllning, signering och kommentarer",
  "sign.mdp.formSigning": "Till\xE5t endast formul\xE4rifyllning och signering",
  "sign.mdp.noChanges": "Inga \xE4ndringar till\xE5tna (l\xE5s dokumentet)",
  // ── Split pane close buttons (tooltips) ──
  "split.closeThisPane": "St\xE4ng denna panel",
  "split.closeSplit": "St\xE4ng delning",
  "split.openPdf": "\xD6ppna PDF",
  // ── Stamp picker ──
  "stamp.pickerTooltip": "St\xE4mpel"
};

// src/i18n/translations/nb.ts
var nb = {
  // ── Toolbar ──
  "toolbar.menu": "Meny",
  "toolbar.open": "\xC5pne",
  "toolbar.close": "Lukk",
  "toolbar.save": "Lagre",
  "toolbar.print": "Skriv ut",
  "toolbar.export": "Eksporter",
  "toolbar.sidebar": "Vis/skjul sidepanel",
  "toolbar.prevPage": "Forrige side",
  "toolbar.nextPage": "Neste side",
  "toolbar.zoomIn": "Zoom inn",
  "toolbar.zoomOut": "Zoom ut",
  "toolbar.fitWidth": "Tilpass bredde",
  "toolbar.fitPage": "Tilpass side",
  "toolbar.actualSize": "Faktisk st\xF8rrelse",
  "toolbar.search": "S\xF8k",
  "toolbar.fullscreen": "Fullskjerm",
  "toolbar.annotate": "Annotere",
  "toolbar.undo": "Angre",
  "toolbar.redo": "Gj\xF8r om",
  "toolbar.more": "Mer",
  "toolbar.moreTools": "Flere verkt\xF8y",
  "toolbar.zoomPresets": "Zoom-forh\xE5ndsvalg",
  "toolbar.skipToContent": "G\xE5 til dokumentet",
  "toolbar.screenshot": "Eksporter side som bilde",
  // ── Tools ──
  "tool.pointer": "Peker",
  "tool.hand": "Panorer",
  "tool.textSelect": "Merk tekst",
  "tool.capture": "Fang omr\xE5de",
  // ── Marquee capture ──
  "capture.copy": "Kopier",
  "capture.save": "Lagre",
  "capture.actions": "Fangsthandlinger",
  "capture.copiedToClipboard": "Kopiert til utklippstavlen",
  "capture.copyFailed": "Kopiering til utklippstavlen mislyktes",
  "capture.failed": "Fangst mislyktes",
  // ── Drag and drop ──
  "dropzone.releaseToOpen": "Slipp for \xE5 laste inn PDF",
  "dropzone.releaseToOpenMany": "Slipp for \xE5 laste inn PDF-er",
  "dropzone.notPdf": "Bare PDF-filer kan \xE5pnes",
  "dropzone.partialNotPdf": "Noen filer ble ignorert \u2014 bare PDF-filer kan \xE5pnes",
  // ── Split-pane view ──
  "split.horizontal": "Del horisontalt",
  "split.vertical": "Del vertikalt",
  "split.closeAll": "Lukk delte paneler",
  "split.openToCompare": "\xC5pne PDF for \xE5 sammenligne",
  "split.dropHint": "eller slipp en PDF her",
  "split.emptySideLabel": "Tom",
  // ── Annotations ──
  "annotation.highlight": "Utheving",
  "annotation.underline": "Understreking",
  "annotation.strikeout": "Gjennomstreking",
  "annotation.squiggly": "B\xF8lget understreking",
  "annotation.ink": "Penn",
  "annotation.inkHighlighter": "Markeringspenn",
  "annotation.eraser": "Viskel\xE6r",
  "annotation.freetext": "Tekstboks",
  "annotation.stickyNote": "Notatmerke",
  "annotation.insertText": "Sett inn tekst",
  "annotation.rectangle": "Rektangel",
  "annotation.circle": "Sirkel",
  "annotation.line": "Linje",
  "annotation.arrow": "Pil",
  "annotation.polygon": "Polygon",
  "annotation.polyline": "Polylinje",
  "annotation.stamp": "Stempel",
  "annotation.image": "Bilde",
  "annotation.imageHelp": "Slipp eller velg et bilde \xE5 plassere p\xE5 siden",
  "annotation.imagePickFile": "Velg bilde\u2026",
  "annotation.imageInvalid": "Kunne ikke laste inn bildet",
  "annotation.imageTooLarge": "Bildet er for stort (maks 4 MB)",
  "annotation.callout": "Forklaring",
  "annotation.calloutHint": "Klikk for \xE5 plassere tekstboks, klikk deretter for lederpunktet",
  "annotation.calloutEdit": "Dobbeltklikk for \xE5 redigere tekst",
  "annotation.redaction": "Sladding",
  "annotation.applyRedaction": "Bruk sladding",
  "annotation.measurement": "M\xE5ling",
  "annotation.signature": "Signatur",
  "annotation.color": "Farge",
  "annotation.fill": "Fyll",
  "annotation.borderWidth": "Kantbredde",
  "annotation.lineStyle": "Linjestil",
  "annotation.opacity": "Gjennomsiktighet",
  "annotation.delete": "Slett",
  "annotation.icon": "Ikon",
  "annotation.thickness": "Tykkelse",
  "annotation.bringToFront": "Flytt fremst",
  "annotation.sendToBack": "Flytt bakerst",
  "annotation.group": "Grupper",
  "annotation.ungroup": "Opphev gruppering",
  "annotation.multiSelectCount": "{count} merknader valgt",
  // ── Line styles ──
  "lineStyle.solid": "Heltrukken",
  "lineStyle.dashed": "Stiplet",
  "lineStyle.dotted": "Prikket",
  // ── Context menu ──
  "contextMenu.insertPageBefore": "Sett inn side foran",
  "contextMenu.insertPageAfter": "Sett inn side etter",
  "contextMenu.duplicatePage": "Dupliser side",
  "contextMenu.rotateCW": "Roter med klokken",
  "contextMenu.rotateCCW": "Roter mot klokken",
  "contextMenu.deletePage": "Slett side",
  "contextMenu.copyText": "Kopier tekst",
  "contextMenu.addNote": "Legg til notat",
  "contextMenu.highlight": "Uthev",
  // ── Attachments ──
  "attachment.addFile": "Legg til fil",
  // ── Page / navigation ──
  "page.viewer": "Dokumentvisning",
  "page.goToPage": "G\xE5 til side",
  "page.goToPageN": "G\xE5 til side {page}",
  "page.documents": "Dokumenter",
  // ── Freetext / overlays ──
  "annotation.doubleClickToEdit": "Dobbeltklikk for \xE5 redigere",
  "annotation.defaultLabel": "Merknad",
  "form.button": "Knapp",
  "form.clickToSign": "Klikk for \xE5 signere",
  // ── Sidebar ──
  "sidebar.thumbnails": "Sider",
  "sidebar.outline": "Bokmerker",
  "sidebar.annotations": "Merknader",
  "sidebar.attachments": "Vedlegg",
  "sidebar.signatures": "Signaturer",
  "sidebar.layers": "Lag",
  "sidebar.comparison": "Endringer",
  // ── Comparison ──
  "comparison.title": "Dokumentsammenligning",
  "comparison.compare": "Sammenlign",
  "comparison.exit": "Avslutt sammenligning",
  "comparison.computing": "Sammenligner dokumenter\u2026",
  "comparison.error": "Sammenligning mislyktes: {error}",
  "comparison.requireSplit": "\xC5pne to PDF-er i delt visning for \xE5 sammenligne dem",
  "comparison.identical": "Begge dokumentene er identiske \u2014 ingen endringer funnet",
  "comparison.totalChanges": "{count} endring",
  "comparison.totalChangesPlural": "{count} endringer",
  "comparison.noChangesOnPage": "Ingen endringer p\xE5 denne siden",
  "comparison.pageHeading": "Side {a} \u2194 Side {b}",
  "comparison.pageHeadingInsert": "Side {b} (lagt til)",
  "comparison.pageHeadingDelete": "Side {a} (fjernet)",
  "comparison.pageHeadingMismatch": "Side {a} \u2194 Side {b} (blandet tekst/skann)",
  "comparison.pageHeadingRegion": "Side {a} \u2194 Side {b} (visuell)",
  "comparison.changeInsert": "Lagt til",
  "comparison.changeDelete": "Fjernet",
  "comparison.changeReplace": "Endret",
  "comparison.changeRegion": "Visuell endring",
  "comparison.filterAll": "Alle",
  "comparison.filterInsert": "Lagt til",
  "comparison.filterDelete": "Fjernet",
  "comparison.filterReplace": "Endret",
  "comparison.prevChange": "Forrige endring",
  "comparison.nextChange": "Neste endring",
  "comparison.syncScroll": "Synkroniser rulling",
  "comparison.changeOf": "{n} av {total}",
  "comparison.openSidebar": "\xC5pne endringspanelet",
  // ── Comments ──
  "comment.addComment": "Legg til en kommentar...",
  "comment.reply": "Svar...",
  "comment.replyOrMention": "Svar eller bruk @ for \xE5 nevne...",
  "comment.commentOrMention": "Kommenter eller bruk @ for \xE5 nevne...",
  "comment.edit": "Rediger",
  "comment.delete": "Slett",
  "comment.cancel": "Avbryt",
  "comment.post": "Publiser",
  "comment.save": "Lagre",
  "comment.resolve": "L\xF8s",
  "comment.reopen": "Gjen\xE5pne",
  "comment.edited": "(redigert)",
  // ── Comments sidebar ──
  "comments.title": "Kommentarer",
  "comments.empty": "Ingen kommentarer enn\xE5.\nKlikk p\xE5 en merknad p\xE5 siden for \xE5 starte en tr\xE5d.",
  "comments.noDocument": "Ingen dokumenter \xE5pne",
  "comments.noBody": "Ingen tekst",
  "comments.reply": "svar",
  "comments.replies": "svar",
  "comments.filter.all": "Alle",
  "comments.filter.open": "\xC5pne",
  "comments.filter.resolved": "L\xF8ste",
  "comments.filter.mine": "Mine",
  "comments.filter.mentions": "@ meg",
  "comments.sort.page": "Etter side",
  "comments.sort.date": "Etter dato",
  "comments.sort.author": "Etter forfatter",
  "common.page": "Side",
  "common.close": "Lukk",
  "toolbar.comments": "Vis/skjul kommentarer",
  // ── Comment status ──
  "commentStatus.open": "\xC5pen",
  "commentStatus.accepted": "Godtatt",
  "commentStatus.rejected": "Avvist",
  "commentStatus.completed": "Fullf\xF8rt",
  "commentStatus.cancelled": "Avbrutt",
  "commentStatus.resolved": "L\xF8st",
  // ── Page operations ──
  "pageOps.delete": "Slett side",
  "pageOps.rotate": "Roter side",
  "pageOps.rotateCW": "Roter med klokken",
  "pageOps.rotateCCW": "Roter mot klokken",
  "pageOps.insert": "Sett inn blank side",
  "pageOps.duplicate": "Dupliser side",
  "pageOps.move": "Flytt side",
  // ── Search ──
  "search.placeholder": "S\xF8k i dokumentet...",
  "search.noResults": "Ingen resultater",
  "search.matchCase": "Skill store/sm\xE5 bokstaver",
  "search.matchWholeWord": "Hele ord",
  "search.resultsCount": "{current} av {total}",
  "search.previous": "Forrige treff (Shift+Enter)",
  "search.next": "Neste treff (Enter)",
  "search.close": "Lukk s\xF8k (Esc)",
  // ── Pagination ──
  "pagination.of": "av",
  "pagination.page": "Side",
  "pagination.pageOf": "Side {current} av {total}",
  // ── Zoom ──
  "zoom.level": "{value} %",
  // ── Layout ──
  "layout.single": "Enkeltside",
  "layout.continuous": "Fortl\xF8pende",
  "layout.double": "Tosidig visning",
  // ── Theme ──
  "theme.light": "Lyst",
  "theme.dark": "M\xF8rkt",
  "theme.system": "System",
  // ── Stamps ──
  "stamp.approved": "Godkjent",
  "stamp.notApproved": "Ikke godkjent",
  "stamp.draft": "Utkast",
  "stamp.final": "Endelig",
  "stamp.completed": "Fullf\xF8rt",
  "stamp.confidential": "Konfidensielt",
  "stamp.forPublicRelease": "For offentliggj\xF8ring",
  "stamp.notForPublicRelease": "Ikke for offentliggj\xF8ring",
  "stamp.forComment": "Til uttalelse",
  "stamp.void": "Ugyldig",
  "stamp.asIs": "Som den er",
  "stamp.departmental": "Avdelingsintern",
  "stamp.experimental": "Eksperimentell",
  "stamp.expired": "Utl\xF8pt",
  "stamp.informationOnly": "Kun til informasjon",
  "stamp.preliminaryResults": "Forel\xF8pige resultater",
  "stamp.sold": "Solgt",
  "stamp.topSecret": "Strengt hemmelig",
  // ── Signatures ──
  "signature.signed": "Signert",
  "signature.unsigned": "Usignert",
  "signature.valid": "Gyldig signatur",
  "signature.invalid": "Ugyldig signatur",
  "signature.unknown": "Signaturgyldighetsstatus ukjent",
  // ── Measurement ──
  "measurement.distance": "Avstand",
  "measurement.area": "Areal",
  "measurement.perimeter": "Omkrets",
  "measurement.calibrate": "Angi skala\u2026",
  "measurement.calibrateTitle": "Kalibrer m\xE5leskala",
  "measurement.calibrateDesc": "Definer hvordan PDF-avstander tilsvarer virkelige m\xE5l.",
  "measurement.currentScale": "Gjeldende skala: {source} {sourceUnit} = {target} {targetUnit}",
  "measurement.pdfDistance": "PDF-avstand",
  "measurement.realDistance": "Virkelig avstand",
  "measurement.unit": "Enhet",
  "measurement.precision": "Desimaler",
  "measurement.clearScale": "Fjern skala",
  "toast.calibrationSaved": "M\xE5leskala lagret",
  "toast.calibrationCleared": "M\xE5leskala fjernet",
  "toast.calibrationInvalid": "Begge avstander m\xE5 v\xE6re st\xF8rre enn null",
  // ── Errors ──
  "error.loadFailed": "Kunne ikke laste dokumentet",
  "error.passwordRequired": "Dette dokumentet er passordbeskyttet",
  "error.renderFailed": "Kunne ikke gjengi siden",
  // ── Misc ──
  "misc.loading": "Laster...",
  "misc.unknown": "Ukjent",
  "misc.none": "Ingen",
  // ── Common UI ──
  "common.cancel": "Avbryt",
  "common.save": "Lagre",
  "common.open": "\xC5pne",
  "common.delete": "Slett",
  "common.apply": "Bruk",
  "common.ok": "OK",
  "common.yes": "Ja",
  "common.no": "Nei",
  "common.selectAll": "Merk alt",
  "common.download": "Last ned",
  // ── Toolbar extras (shortcut hints) ──
  "toolbar.undoShortcut": "Angre (Ctrl+Z)",
  "toolbar.redoShortcut": "Gj\xF8r om (Ctrl+Y)",
  "toolbar.searchShortcut": "S\xF8k (Ctrl+F)",
  "toolbar.passwordProtect": "Passordbeskyttelse",
  "toolbar.signDocument": "Signer dokument",
  // ── Zoom control tooltips ──
  "zoom.zoomIn": "Zoom inn",
  "zoom.zoomOut": "Zoom ut",
  // ── Toasts / ephemeral feedback ──
  "toast.bookmarkAdded": "Bokmerke lagt til",
  "toast.bookmarkDeleted": "Bokmerke slettet",
  "toast.attached": "Vedlagt: {name}",
  "toast.removed": "Fjernet: {name}",
  "toast.noRedactions": "Ingen sladdinger \xE5 bruke",
  "toast.redactionsApplied": "{count} sladding brukt",
  "toast.redactionsAppliedPlural": "{count} sladdinger brukt",
  "toast.redactionsFailed": "Kunne ikke bruke sladdinger: {error}",
  "toast.noDocument": "Ingen dokumenter \xE5pne",
  "toast.protectedSaved": "Beskyttet dokument lagret",
  "toast.invalidPageRange": "Ugyldig sideomr\xE5de",
  "toast.printFailed": "Utskrift mislyktes: {error}",
  "toast.documentSaved": "Dokument lagret",
  "toast.saveFailed": "Lagring mislyktes: {error}",
  "toast.pageExported": "Side {page} eksportert",
  "toast.exportFailed": "Eksport mislyktes: {error}",
  "toast.signatureMissing": "Opprett en signatur eller last opp et sertifikat",
  "toast.signatureApplied": "Signatur brukt",
  "toast.signedSaved": "Signert dokument lagret",
  "toast.copied": "Kopiert",
  "toast.openFailed": "Kunne ikke \xE5pne: {error}",
  "toast.openNamedFailed": "Kunne ikke \xE5pne {name}: {error}",
  // ── Bookmarks panel ──
  "bookmarks.add": "Legg til bokmerke",
  "bookmarks.empty": "Ingen bokmerker i dette dokumentet",
  "bookmarks.delete": "Slett",
  "bookmarks.rename": "Gi nytt navn",
  "bookmarks.dragHint": "Dra for \xE5 endre rekkef\xF8lge",
  "bookmarks.navigationPluginMissing": "Navigasjonsplugin er ikke lastet",
  "toast.bookmarkRenamed": "Bokmerke omd\xF8pt",
  "toast.bookmarkMoved": "Bokmerke flyttet",
  // ── Annotation presets ──
  "presets.tooltip": "Forh\xE5ndsvalg for merknader",
  "presets.none": "Ingen forh\xE5ndsvalg",
  "presets.empty": "Ingen forh\xE5ndsvalg enn\xE5",
  "presets.saveCurrent": "Lagre gjeldende stil\u2026",
  "presets.namePrompt": "Navn for dette forh\xE5ndsvalget:",
  "presets.dialogTitle": "Lagre forh\xE5ndsvalg",
  "presets.namePlaceholder": "Mitt forh\xE5ndsvalg",
  "presets.nameRequired": "Vennligst oppgi et navn for forh\xE5ndsvalget",
  "toast.presetSaved": "Forh\xE5ndsvalg lagret",
  "toast.presetDeleted": "Forh\xE5ndsvalg slettet",
  // ── Annotations list panel ──
  "annotations.empty": "Ingen merknader i dette dokumentet",
  "annotations.pluginMissing": "Merknadsplugin er ikke lastet",
  "annotations.resolved": "L\xF8st",
  "annotations.pageHeader": "Side {page}",
  // ── Attachments panel ──
  "attachments.empty": "Ingen vedlegg i dette dokumentet",
  "attachments.pluginMissing": "Vedleggsplugin er ikke lastet",
  // ── Signatures panel ──
  "signatures.empty": "Ingen digitale signaturer i dette dokumentet",
  "signatures.pluginMissing": "Signaturplugin er ikke lastet",
  "signatures.validating": "Validerer signaturer\u2026",
  "signatures.validatingShort": "Validerer\u2026",
  "signatures.validationError": "Valideringsfeil: {error}",
  "signatures.itemTitle": "Signatur {index}",
  "signatures.badgeValid": "Gyldig",
  "signatures.badgeInvalid": "Ugyldig",
  "signatures.badgeError": "Feil",
  "signatures.badgeUnknown": "Ukjent",
  "signatures.integrityVerified": "Dokumentintegritet bekreftet",
  "signatures.integrityModified": "Dokumentet er endret etter signering",
  "signatures.integrityUnknown": "Integritet ukjent",
  "signatures.cryptoValid": "Signatur er kryptografisk gyldig",
  "signatures.signatureValid": "Signatur gyldig",
  "signatures.signatureInvalid": "Signatur ugyldig",
  "signatures.certNotTrusted": "Sertifikatet er ikke klarert",
  "signatures.verificationNeeded": "Signaturverifisering kreves",
  "signatures.certExpired": "Sertifikatet har utl\xF8pt",
  "signatures.certSelfSigned": "Selvsignert sertifikat",
  "signatures.certIssuer": "Utsteder: {issuer}",
  "signatures.algorithm": "Algoritme: {algorithm}",
  "signatures.format": "Format: {format}",
  "signatures.reason": "Grunn: {reason}",
  "signatures.signedAt": "Signert: {time}",
  "signatures.covers": "Dekker: {size}",
  "signatures.validationHint": "Validering: {error}",
  "signatures.badgeStatusSingle": "Digital signatur",
  "signatures.badgeStatusMany": "{count} digitale signaturer",
  "signatures.fullDetailsHint": "Klikk for alle detaljer",
  "signatures.perm.approval": "Godkjenningssignatur (ytterligere endringer tillatt)",
  "signatures.perm.noChanges": "Sertifisert \u2014 ingen endringer tillatt",
  "signatures.perm.formSigning": "Sertifisert \u2014 kun skjemautfylling og signering",
  "signatures.perm.formCommenting": "Sertifisert \u2014 skjemautfylling, signering og kommentarer",
  "signatures.perm.unknown": "Tilgangsniv\xE5: {level}",
  "signatures.format.pkcs7Detached": "PKCS#7 frakoblet (Adobe)",
  "signatures.format.pkcs7Sha1": "PKCS#7 SHA-1 (Adobe, eldre)",
  "signatures.format.x509RsaSha1": "X.509 RSA SHA-1 (Adobe, eldre)",
  "signatures.format.cadesDetached": "CAdES frakoblet (PAdES)",
  "signatures.format.rfc3161": "RFC 3161 tidsstempel",
  // ── Layers panel ──
  "layers.empty": "Ingen lag i dette dokumentet",
  "layers.pluginMissing": "Lagplugin er ikke lastet",
  // ── Apply Redactions dialog ──
  "redact.dialogTitle": "Bruk sladdinger",
  "redact.applyButton": "Bruk sladdinger",
  "redact.applying": "Bruker\u2026",
  "redact.summarySingle": "<strong>1</strong> sladding vil bli brukt.",
  "redact.summaryPlural": "<strong>{count}</strong> sladdinger vil bli brukt.",
  "redact.warning": "Dette fjerner permanent alt innhold under de sladdede omr\xE5dene. Denne handlingen <strong>kan ikke angres</strong>.",
  // ── Password (open) dialog ──
  "password.openTitle": "Passord kreves",
  "password.openPlaceholder": "Skriv inn passord",
  "password.opening": "\xC5pner\u2026",
  "password.incorrect": "Feil passord. Vennligst pr\xF8v igjen.",
  "password.openFailed": "Kunne ikke \xE5pne: {error}",
  "password.descNamed": "Dokumentet \xAB{name}\xBB er passordbeskyttet.",
  "password.desc": "Dette dokumentet er passordbeskyttet.",
  // ── Password protect dialog ──
  "protect.dialogTitle": "Passordbeskyttelse av dokument",
  "protect.userLabel": "\xC5pnepassord (kreves for \xE5 vise)",
  "protect.userPlaceholder": "Skriv inn passord",
  "protect.confirmLabel": "Bekreft passord",
  "protect.confirmPlaceholder": "Bekreft passord",
  "protect.ownerLabel": "Eierpassord (valgfritt \u2014 for full tilgang)",
  "protect.ownerPlaceholder": "Eierpassord (la st\xE5 tomt = samme som \xE5pnepassord)",
  "protect.permissionsLabel": "Tillatelser",
  "protect.applyButton": "Beskytt og lagre",
  "protect.applying": "Beskytter\u2026",
  "protect.errorRequired": "Passord er p\xE5krevd",
  "protect.errorMismatch": "Passordene stemmer ikke overens",
  "protect.errorFailed": "Mislyktes: {error}",
  "protect.perm.print": "Tillat utskrift",
  "protect.perm.extract": "Tillat kopiering/uttrekk",
  "protect.perm.annotate": "Tillat merknader og skjemaer",
  "protect.perm.modify": "Tillat innholdsendring",
  // ── Print dialog ──
  "print.dialogTitle": "Skriv ut dokument",
  "print.pagesLabel": "Sider",
  "print.allPages": "Alle sider (1-{total})",
  "print.customRange": "Egendefinert omr\xE5de:",
  "print.rangePlaceholder": "f.eks. 1-3, 5, 8-10",
  "print.printButton": "Skriv ut",
  "print.rendering": "Gjengir\u2026",
  // ── Signature dialog ──
  "sign.dialogTitle": "Signer dokument",
  "sign.clearButton": "T\xF8m",
  "sign.uploadPrompt": "Klikk for \xE5 laste opp eller dra bilde",
  "sign.typedPlaceholder": "Skriv navnet ditt",
  "sign.certLabel": "Digitalt sertifikat (valgfritt)",
  "sign.certUpload": "Last opp PFX/P12\u2026",
  "sign.certNone": "Ingen sertifikat valgt",
  "sign.certPasswordPlaceholder": "Sertifikatpassord",
  "sign.reasonPlaceholder": "Grunn for signering (valgfritt)",
  "sign.tsaPlaceholder": "TSA-URL (valgfritt, f.eks. http://timestamp.digicert.com)",
  "sign.applyButton": "Bruk signatur",
  "sign.applying": "Signerer\u2026",
  "sign.tab.draw": "Tegn",
  "sign.tab.type": "Skriv",
  "sign.tab.image": "Bilde",
  "sign.mdp.approval": "Tillat ytterligere endringer (godkjenningssignatur)",
  "sign.mdp.formCommenting": "Tillat skjemautfylling, signering og kommentering",
  "sign.mdp.formSigning": "Tillat kun skjemautfylling og signering",
  "sign.mdp.noChanges": "Ingen endringer tillatt (l\xE5s dokumentet)",
  // ── Split pane close buttons (tooltips) ──
  "split.closeThisPane": "Lukk dette panelet",
  "split.closeSplit": "Lukk deling",
  "split.openPdf": "\xC5pne PDF",
  // ── Stamp picker ──
  "stamp.pickerTooltip": "Stempel"
};

// src/i18n/translations/da.ts
var da = {
  // ── Toolbar ──
  "toolbar.menu": "Menu",
  "toolbar.open": "\xC5bn",
  "toolbar.close": "Luk",
  "toolbar.save": "Gem",
  "toolbar.print": "Udskriv",
  "toolbar.export": "Eksport\xE9r",
  "toolbar.sidebar": "Vis/skjul sidepanel",
  "toolbar.prevPage": "Forrige side",
  "toolbar.nextPage": "N\xE6ste side",
  "toolbar.zoomIn": "Zoom ind",
  "toolbar.zoomOut": "Zoom ud",
  "toolbar.fitWidth": "Tilpas bredde",
  "toolbar.fitPage": "Tilpas side",
  "toolbar.actualSize": "Faktisk st\xF8rrelse",
  "toolbar.search": "S\xF8g",
  "toolbar.fullscreen": "Fuld sk\xE6rm",
  "toolbar.annotate": "Annot\xE9r",
  "toolbar.undo": "Fortryd",
  "toolbar.redo": "Annuller Fortryd",
  "toolbar.more": "Mere",
  "toolbar.moreTools": "Flere v\xE6rkt\xF8jer",
  "toolbar.zoomPresets": "Zoom-forudindstillinger",
  "toolbar.skipToContent": "G\xE5 til dokument",
  "toolbar.screenshot": "Eksport\xE9r side som billede",
  // ── Tools ──
  "tool.pointer": "Mark\xF8r",
  "tool.hand": "Panorer",
  "tool.textSelect": "Mark\xE9r tekst",
  "tool.capture": "Optag omr\xE5de",
  // ── Marquee capture ──
  "capture.copy": "Kopi\xE9r",
  "capture.save": "Gem",
  "capture.actions": "Optagelseshandlinger",
  "capture.copiedToClipboard": "Kopieret til udklipsholder",
  "capture.copyFailed": "Kopiering til udklipsholder mislykkedes",
  "capture.failed": "Optagelse mislykkedes",
  // ── Drag and drop ──
  "dropzone.releaseToOpen": "Slip for at indl\xE6se PDF",
  "dropzone.releaseToOpenMany": "Slip for at indl\xE6se PDF-filer",
  "dropzone.notPdf": "Kun PDF-filer kan \xE5bnes",
  "dropzone.partialNotPdf": "Nogle filer blev ignoreret \u2014 kun PDF-filer kan \xE5bnes",
  // ── Split-pane view ──
  "split.horizontal": "Opdel vandret",
  "split.vertical": "Opdel lodret",
  "split.closeAll": "Luk opdelte ruder",
  "split.openToCompare": "\xC5bn PDF til sammenligning",
  "split.dropHint": "eller slip en PDF her",
  "split.emptySideLabel": "Tom",
  // ── Annotations ──
  "annotation.highlight": "Fremh\xE6v",
  "annotation.underline": "Understregning",
  "annotation.strikeout": "Gennemstregning",
  "annotation.squiggly": "B\xF8lget understregning",
  "annotation.ink": "Pen",
  "annotation.inkHighlighter": "Overstregningspen",
  "annotation.eraser": "Viskel\xE6der",
  "annotation.freetext": "Tekstfelt",
  "annotation.stickyNote": "Note",
  "annotation.insertText": "Inds\xE6t tekst",
  "annotation.rectangle": "Rektangel",
  "annotation.circle": "Cirkel",
  "annotation.line": "Linje",
  "annotation.arrow": "Pil",
  "annotation.polygon": "Polygon",
  "annotation.polyline": "Polylinje",
  "annotation.stamp": "Stempel",
  "annotation.image": "Billede",
  "annotation.imageHelp": "Slip eller v\xE6lg et billede for at placere det p\xE5 siden",
  "annotation.imagePickFile": "V\xE6lg billede\u2026",
  "annotation.imageInvalid": "Billedet kunne ikke indl\xE6ses",
  "annotation.imageTooLarge": "Billedet er for stort (maks. 4 MB)",
  "annotation.callout": "Billedforklaring",
  "annotation.calloutHint": "Klik for at placere tekstfeltet, klik derefter igen for ledepunktet",
  "annotation.calloutEdit": "Dobbeltklik for at redigere tekst",
  "annotation.redaction": "Redigering",
  "annotation.applyRedaction": "Anvend redigering",
  "annotation.measurement": "M\xE5ling",
  "annotation.signature": "Underskrift",
  "annotation.color": "Farve",
  "annotation.fill": "Fyld",
  "annotation.borderWidth": "Kantbredde",
  "annotation.lineStyle": "Linjestil",
  "annotation.opacity": "Gennemsigtighed",
  "annotation.delete": "Slet",
  "annotation.icon": "Ikon",
  "annotation.thickness": "Tykkelse",
  "annotation.bringToFront": "Placer forrest",
  "annotation.sendToBack": "Placer bagerst",
  "annotation.group": "Grupp\xE9r",
  "annotation.ungroup": "Oph\xE6v gruppering",
  "annotation.multiSelectCount": "{count} annotationer valgt",
  // ── Line styles ──
  "lineStyle.solid": "Ubrudt",
  "lineStyle.dashed": "Stiplet",
  "lineStyle.dotted": "Prikket",
  // ── Context menu ──
  "contextMenu.insertPageBefore": "Inds\xE6t side f\xF8r",
  "contextMenu.insertPageAfter": "Inds\xE6t side efter",
  "contextMenu.duplicatePage": "Duplik\xE9r side",
  "contextMenu.rotateCW": "Rot\xE9r med uret",
  "contextMenu.rotateCCW": "Rot\xE9r mod uret",
  "contextMenu.deletePage": "Slet side",
  "contextMenu.copyText": "Kopi\xE9r tekst",
  "contextMenu.addNote": "Tilf\xF8j note",
  "contextMenu.highlight": "Fremh\xE6v",
  // ── Attachments ──
  "attachment.addFile": "Tilf\xF8j fil",
  // ── Page / navigation ──
  "page.viewer": "Dokumentvisning",
  "page.goToPage": "G\xE5 til side",
  "page.goToPageN": "G\xE5 til side {page}",
  "page.documents": "Dokumenter",
  // ── Freetext / overlays ──
  "annotation.doubleClickToEdit": "Dobbeltklik for at redigere",
  "annotation.defaultLabel": "Annotation",
  "form.button": "Knap",
  "form.clickToSign": "Klik for at underskrive",
  // ── Sidebar ──
  "sidebar.thumbnails": "Sider",
  "sidebar.outline": "Bogm\xE6rker",
  "sidebar.annotations": "Annotationer",
  "sidebar.attachments": "Vedh\xE6ftede filer",
  "sidebar.signatures": "Underskrifter",
  "sidebar.layers": "Lag",
  "sidebar.comparison": "\xC6ndringer",
  // ── Comparison ──
  "comparison.title": "Dokumentsammenligning",
  "comparison.compare": "Sammenlign",
  "comparison.exit": "Afslut sammenligning",
  "comparison.computing": "Sammenligner dokumenter\u2026",
  "comparison.error": "Sammenligning mislykkedes: {error}",
  "comparison.requireSplit": "\xC5bn to PDF-filer i opdelt visning for at sammenligne dem",
  "comparison.identical": "Begge dokumenter er identiske \u2014 ingen \xE6ndringer fundet",
  "comparison.totalChanges": "{count} \xE6ndring",
  "comparison.totalChangesPlural": "{count} \xE6ndringer",
  "comparison.noChangesOnPage": "Ingen \xE6ndringer p\xE5 denne side",
  "comparison.pageHeading": "Side {a} \u2194 Side {b}",
  "comparison.pageHeadingInsert": "Side {b} (tilf\xF8jet)",
  "comparison.pageHeadingDelete": "Side {a} (fjernet)",
  "comparison.pageHeadingMismatch": "Side {a} \u2194 Side {b} (blandet tekst/scanning)",
  "comparison.pageHeadingRegion": "Side {a} \u2194 Side {b} (visuel)",
  "comparison.changeInsert": "Tilf\xF8jet",
  "comparison.changeDelete": "Fjernet",
  "comparison.changeReplace": "\xC6ndret",
  "comparison.changeRegion": "Visuel \xE6ndring",
  "comparison.filterAll": "Alle",
  "comparison.filterInsert": "Tilf\xF8jet",
  "comparison.filterDelete": "Fjernet",
  "comparison.filterReplace": "\xC6ndret",
  "comparison.prevChange": "Forrige \xE6ndring",
  "comparison.nextChange": "N\xE6ste \xE6ndring",
  "comparison.syncScroll": "Synkronis\xE9r rulning",
  "comparison.changeOf": "{n} af {total}",
  "comparison.openSidebar": "\xC5bn \xE6ndringspanel",
  // ── Comments ──
  "comment.addComment": "Tilf\xF8j en kommentar...",
  "comment.reply": "Svar...",
  "comment.replyOrMention": "Svar eller brug @ til at n\xE6vne...",
  "comment.commentOrMention": "Komment\xE9r eller brug @ til at n\xE6vne...",
  "comment.edit": "Rediger",
  "comment.delete": "Slet",
  "comment.cancel": "Annuller",
  "comment.post": "Opsl\xE5",
  "comment.save": "Gem",
  "comment.resolve": "Mark\xE9r som l\xF8st",
  "comment.reopen": "Gen\xE5bn",
  "comment.edited": "(redigeret)",
  // ── Comments sidebar ──
  "comments.title": "Kommentarer",
  "comments.empty": "Ingen kommentarer endnu.\nKlik p\xE5 en annotation p\xE5 siden for at starte en tr\xE5d.",
  "comments.noDocument": "Intet dokument \xE5bent",
  "comments.noBody": "Ingen tekst",
  "comments.reply": "svar",
  "comments.replies": "svar",
  "comments.filter.all": "Alle",
  "comments.filter.open": "\xC5bne",
  "comments.filter.resolved": "L\xF8ste",
  "comments.filter.mine": "Mine",
  "comments.filter.mentions": "@ mig",
  "comments.sort.page": "Efter side",
  "comments.sort.date": "Efter dato",
  "comments.sort.author": "Efter forfatter",
  "common.page": "Side",
  "common.close": "Luk",
  "toolbar.comments": "Vis/skjul kommentarer",
  // ── Comment status ──
  "commentStatus.open": "\xC5ben",
  "commentStatus.accepted": "Accepteret",
  "commentStatus.rejected": "Afvist",
  "commentStatus.completed": "Fuldf\xF8rt",
  "commentStatus.cancelled": "Annulleret",
  "commentStatus.resolved": "L\xF8st",
  // ── Page operations ──
  "pageOps.delete": "Slet side",
  "pageOps.rotate": "Rot\xE9r side",
  "pageOps.rotateCW": "Rot\xE9r med uret",
  "pageOps.rotateCCW": "Rot\xE9r mod uret",
  "pageOps.insert": "Inds\xE6t blank side",
  "pageOps.duplicate": "Duplik\xE9r side",
  "pageOps.move": "Flyt side",
  // ── Search ──
  "search.placeholder": "S\xF8g i dokument...",
  "search.noResults": "Ingen resultater",
  "search.matchCase": "Forskel p\xE5 store/sm\xE5 bogstaver",
  "search.matchWholeWord": "Hele ord",
  "search.resultsCount": "{current} af {total}",
  "search.previous": "Forrige resultat (Shift+Enter)",
  "search.next": "N\xE6ste resultat (Enter)",
  "search.close": "Luk s\xF8gning (Esc)",
  // ── Pagination ──
  "pagination.of": "af",
  "pagination.page": "Side",
  "pagination.pageOf": "Side {current} af {total}",
  // ── Zoom ──
  "zoom.level": "{value}%",
  // ── Layout ──
  "layout.single": "Enkelt side",
  "layout.continuous": "Fortl\xF8bende",
  "layout.double": "Tosidet visning",
  // ── Theme ──
  "theme.light": "Lys",
  "theme.dark": "M\xF8rk",
  "theme.system": "System",
  // ── Stamps ──
  "stamp.approved": "Godkendt",
  "stamp.notApproved": "Ikke godkendt",
  "stamp.draft": "Udkast",
  "stamp.final": "Endelig",
  "stamp.completed": "Afsluttet",
  "stamp.confidential": "Fortroligt",
  "stamp.forPublicRelease": "Til offentligg\xF8relse",
  "stamp.notForPublicRelease": "Ikke til offentligg\xF8relse",
  "stamp.forComment": "Til kommentering",
  "stamp.void": "Ugyldig",
  "stamp.asIs": "Som den er",
  "stamp.departmental": "Afdelingsinternt",
  "stamp.experimental": "Eksperimentel",
  "stamp.expired": "Udl\xF8bet",
  "stamp.informationOnly": "Kun til orientering",
  "stamp.preliminaryResults": "Forel\xF8bige resultater",
  "stamp.sold": "Solgt",
  "stamp.topSecret": "Yderst fortroligt",
  // ── Signatures ──
  "signature.signed": "Underskrevet",
  "signature.unsigned": "Ikke underskrevet",
  "signature.valid": "Gyldig underskrift",
  "signature.invalid": "Ugyldig underskrift",
  "signature.unknown": "Underskriftens gyldighed ukendt",
  // ── Measurement ──
  "measurement.distance": "Afstand",
  "measurement.area": "Areal",
  "measurement.perimeter": "Omkreds",
  "measurement.calibrate": "Indstil skala\u2026",
  "measurement.calibrateTitle": "Kalibr\xE9r m\xE5leskala",
  "measurement.calibrateDesc": "Defin\xE9r hvordan PDF-afstande svarer til virkelige enheder.",
  "measurement.currentScale": "Aktuel skala: {source} {sourceUnit} = {target} {targetUnit}",
  "measurement.pdfDistance": "PDF-afstand",
  "measurement.realDistance": "Virkelig afstand",
  "measurement.unit": "Enhed",
  "measurement.precision": "Decimaler",
  "measurement.clearScale": "Ryd skala",
  "toast.calibrationSaved": "M\xE5leskala gemt",
  "toast.calibrationCleared": "M\xE5leskala ryddet",
  "toast.calibrationInvalid": "Begge afstande skal v\xE6re st\xF8rre end nul",
  // ── Errors ──
  "error.loadFailed": "Kunne ikke indl\xE6se dokument",
  "error.passwordRequired": "Dette dokument er beskyttet med adgangskode",
  "error.renderFailed": "Kunne ikke gengive side",
  // ── Misc ──
  "misc.loading": "Indl\xE6ser...",
  "misc.unknown": "Ukendt",
  "misc.none": "Ingen",
  // ── Common UI ──
  "common.cancel": "Annuller",
  "common.save": "Gem",
  "common.open": "\xC5bn",
  "common.delete": "Slet",
  "common.apply": "Anvend",
  "common.ok": "OK",
  "common.yes": "Ja",
  "common.no": "Nej",
  "common.selectAll": "Mark\xE9r alt",
  "common.download": "Download",
  // ── Toolbar extras (shortcut hints) ──
  "toolbar.undoShortcut": "Fortryd (Ctrl+Z)",
  "toolbar.redoShortcut": "Annuller Fortryd (Ctrl+Y)",
  "toolbar.searchShortcut": "S\xF8g (Ctrl+F)",
  "toolbar.passwordProtect": "Adgangskodebeskyttelse",
  "toolbar.signDocument": "Underskriv dokument",
  // ── Zoom control tooltips ──
  "zoom.zoomIn": "Zoom ind",
  "zoom.zoomOut": "Zoom ud",
  // ── Toasts / ephemeral feedback ──
  "toast.bookmarkAdded": "Bogm\xE6rke tilf\xF8jet",
  "toast.bookmarkDeleted": "Bogm\xE6rke slettet",
  "toast.attached": "Vedh\xE6ftet: {name}",
  "toast.removed": "Fjernet: {name}",
  "toast.noRedactions": "Ingen redigeringer at anvende",
  "toast.redactionsApplied": "{count} redigering anvendt",
  "toast.redactionsAppliedPlural": "{count} redigeringer anvendt",
  "toast.redactionsFailed": "Kunne ikke anvende redigeringer: {error}",
  "toast.noDocument": "Intet dokument \xE5bent",
  "toast.protectedSaved": "Beskyttet dokument gemt",
  "toast.invalidPageRange": "Ugyldigt sideinterval",
  "toast.printFailed": "Udskrivning mislykkedes: {error}",
  "toast.documentSaved": "Dokument gemt",
  "toast.saveFailed": "Lagring mislykkedes: {error}",
  "toast.pageExported": "Side {page} eksporteret",
  "toast.exportFailed": "Eksportering mislykkedes: {error}",
  "toast.signatureMissing": "Opret venligst en underskrift eller upload et certifikat",
  "toast.signatureApplied": "Underskrift anvendt",
  "toast.signedSaved": "Underskrevet dokument gemt",
  "toast.copied": "Kopieret",
  "toast.openFailed": "Kunne ikke \xE5bne: {error}",
  "toast.openNamedFailed": "Kunne ikke \xE5bne {name}: {error}",
  // ── Bookmarks panel ──
  "bookmarks.add": "Tilf\xF8j bogm\xE6rke",
  "bookmarks.empty": "Ingen bogm\xE6rker i dette dokument",
  "bookmarks.delete": "Slet",
  "bookmarks.rename": "Omd\xF8b",
  "bookmarks.dragHint": "Tr\xE6k for at omarrangere",
  "bookmarks.navigationPluginMissing": "Navigationsplugin ikke indl\xE6st",
  "toast.bookmarkRenamed": "Bogm\xE6rke omd\xF8bt",
  "toast.bookmarkMoved": "Bogm\xE6rke flyttet",
  // ── Annotation presets ──
  "presets.tooltip": "Annotationsforudindstillinger",
  "presets.none": "Ingen forudindstilling",
  "presets.empty": "Ingen forudindstillinger endnu",
  "presets.saveCurrent": "Gem aktuel stil\u2026",
  "presets.namePrompt": "Navn til denne forudindstilling:",
  "presets.dialogTitle": "Gem forudindstilling",
  "presets.namePlaceholder": "Min forudindstilling",
  "presets.nameRequired": "Angiv venligst et navn til forudindstillingen",
  "toast.presetSaved": "Forudindstilling gemt",
  "toast.presetDeleted": "Forudindstilling slettet",
  // ── Annotations list panel ──
  "annotations.empty": "Ingen annotationer i dette dokument",
  "annotations.pluginMissing": "Annotationsplugin ikke indl\xE6st",
  "annotations.resolved": "L\xF8st",
  "annotations.pageHeader": "Side {page}",
  // ── Attachments panel ──
  "attachments.empty": "Ingen vedh\xE6ftede filer i dette dokument",
  "attachments.pluginMissing": "Plugin for vedh\xE6ftede filer ikke indl\xE6st",
  // ── Signatures panel ──
  "signatures.empty": "Ingen digitale underskrifter i dette dokument",
  "signatures.pluginMissing": "Underskriftsplugin ikke indl\xE6st",
  "signatures.validating": "Validerer underskrifter\u2026",
  "signatures.validatingShort": "Validerer\u2026",
  "signatures.validationError": "Valideringsfejl: {error}",
  "signatures.itemTitle": "Underskrift {index}",
  "signatures.badgeValid": "Gyldig",
  "signatures.badgeInvalid": "Ugyldig",
  "signatures.badgeError": "Fejl",
  "signatures.badgeUnknown": "Ukendt",
  "signatures.integrityVerified": "Dokumentets integritet bekr\xE6ftet",
  "signatures.integrityModified": "Dokumentet er \xE6ndret efter underskrift",
  "signatures.integrityUnknown": "Integritet ukendt",
  "signatures.cryptoValid": "Underskrift er kryptografisk gyldig",
  "signatures.signatureValid": "Underskrift gyldig",
  "signatures.signatureInvalid": "Underskrift ugyldig",
  "signatures.certNotTrusted": "Certifikat er ikke betroet",
  "signatures.verificationNeeded": "Underskriftsbekr\xE6ftelse p\xE5kr\xE6vet",
  "signatures.certExpired": "Certifikat er udl\xF8bet",
  "signatures.certSelfSigned": "Selvunderskrevet certifikat",
  "signatures.certIssuer": "Udsteder: {issuer}",
  "signatures.algorithm": "Algoritme: {algorithm}",
  "signatures.format": "Format: {format}",
  "signatures.reason": "\xC5rsag: {reason}",
  "signatures.signedAt": "Underskrevet: {time}",
  "signatures.covers": "D\xE6kker: {size}",
  "signatures.validationHint": "Validering: {error}",
  "signatures.badgeStatusSingle": "Digital underskrift",
  "signatures.badgeStatusMany": "{count} digitale underskrifter",
  "signatures.fullDetailsHint": "Klik for alle detaljer",
  "signatures.perm.approval": "Godkendelsesunderskrift (yderligere \xE6ndringer tilladt)",
  "signatures.perm.noChanges": "Certificeret \u2014 ingen \xE6ndringer tilladt",
  "signatures.perm.formSigning": "Certificeret \u2014 kun formularudfyldning og underskrift",
  "signatures.perm.formCommenting": "Certificeret \u2014 formularudfyldning, underskrift og kommentarer",
  "signatures.perm.unknown": "Tilladelsesniveau: {level}",
  "signatures.format.pkcs7Detached": "PKCS#7 adskilt (Adobe)",
  "signatures.format.pkcs7Sha1": "PKCS#7 SHA-1 (Adobe, for\xE6ldet)",
  "signatures.format.x509RsaSha1": "X.509 RSA SHA-1 (Adobe, for\xE6ldet)",
  "signatures.format.cadesDetached": "CAdES adskilt (PAdES)",
  "signatures.format.rfc3161": "RFC 3161 tidsstempel",
  // ── Layers panel ──
  "layers.empty": "Ingen lag i dette dokument",
  "layers.pluginMissing": "Lagplugin ikke indl\xE6st",
  // ── Apply Redactions dialog ──
  "redact.dialogTitle": "Anvend redigeringer",
  "redact.applyButton": "Anvend redigeringer",
  "redact.applying": "Anvender...",
  "redact.summarySingle": "<strong>1</strong> redigering vil blive anvendt.",
  "redact.summaryPlural": "<strong>{count}</strong> redigeringer vil blive anvendt.",
  "redact.warning": "Dette fjerner permanent alt indhold under de redigerede omr\xE5der. Denne handling <strong>kan ikke fortrydes</strong>.",
  // ── Password (open) dialog ──
  "password.openTitle": "Adgangskode p\xE5kr\xE6vet",
  "password.openPlaceholder": "Indtast adgangskode",
  "password.opening": "\xC5bner...",
  "password.incorrect": "Forkert adgangskode. Pr\xF8v venligst igen.",
  "password.openFailed": "Kunne ikke \xE5bne: {error}",
  "password.descNamed": 'Dokumentet "{name}" er beskyttet med adgangskode.',
  "password.desc": "Dette dokument er beskyttet med adgangskode.",
  // ── Password protect dialog ──
  "protect.dialogTitle": "Adgangskodebeskyt dokument",
  "protect.userLabel": "\xC5bningsadgangskode (p\xE5kr\xE6vet for at se)",
  "protect.userPlaceholder": "Indtast adgangskode",
  "protect.confirmLabel": "Bekr\xE6ft adgangskode",
  "protect.confirmPlaceholder": "Bekr\xE6ft adgangskode",
  "protect.ownerLabel": "Ejeradgangskode (valgfri \u2014 til fuld adgang)",
  "protect.ownerPlaceholder": "Ejeradgangskode (tom = samme som \xE5bningsadgangskode)",
  "protect.permissionsLabel": "Tilladelser",
  "protect.applyButton": "Beskyt og gem",
  "protect.applying": "Beskytter...",
  "protect.errorRequired": "Adgangskode er p\xE5kr\xE6vet",
  "protect.errorMismatch": "Adgangskoderne stemmer ikke overens",
  "protect.errorFailed": "Mislykkedes: {error}",
  "protect.perm.print": "Tillad udskrivning",
  "protect.perm.extract": "Tillad kopiering/udtr\xE6k",
  "protect.perm.annotate": "Tillad annotationer og formularer",
  "protect.perm.modify": "Tillad \xE6ndring af indhold",
  // ── Print dialog ──
  "print.dialogTitle": "Udskriv dokument",
  "print.pagesLabel": "Sider",
  "print.allPages": "Alle sider (1-{total})",
  "print.customRange": "Brugerdefineret interval:",
  "print.rangePlaceholder": "f.eks. 1-3, 5, 8-10",
  "print.printButton": "Udskriv",
  "print.rendering": "Gengiver...",
  // ── Signature dialog ──
  "sign.dialogTitle": "Underskriv dokument",
  "sign.clearButton": "Ryd",
  "sign.uploadPrompt": "Klik for at uploade eller tr\xE6k et billede",
  "sign.typedPlaceholder": "Skriv dit navn",
  "sign.certLabel": "Digitalt certifikat (valgfrit)",
  "sign.certUpload": "Upload PFX/P12...",
  "sign.certNone": "Intet certifikat valgt",
  "sign.certPasswordPlaceholder": "Certifikatadgangskode",
  "sign.reasonPlaceholder": "\xC5rsag til underskrift (valgfrit)",
  "sign.tsaPlaceholder": "TSA-URL (valgfrit, f.eks. http://timestamp.digicert.com)",
  "sign.applyButton": "Anvend underskrift",
  "sign.applying": "Underskriver...",
  "sign.tab.draw": "Tegn",
  "sign.tab.type": "Skriv",
  "sign.tab.image": "Billede",
  "sign.mdp.approval": "Tillad yderligere \xE6ndringer (godkendelsesunderskrift)",
  "sign.mdp.formCommenting": "Tillad formularudfyldning, underskrift og kommentering",
  "sign.mdp.formSigning": "Tillad kun formularudfyldning og underskrift",
  "sign.mdp.noChanges": "Ingen \xE6ndringer tilladt (l\xE5s dokument)",
  // ── Split pane close buttons (tooltips) ──
  "split.closeThisPane": "Luk denne rude",
  "split.closeSplit": "Luk opdeling",
  "split.openPdf": "\xC5bn PDF",
  // ── Stamp picker ──
  "stamp.pickerTooltip": "Stempel"
};

// src/i18n/translations/fi.ts
var fi = {
  // ── Toolbar ──
  "toolbar.menu": "Valikko",
  "toolbar.open": "Avaa",
  "toolbar.close": "Sulje",
  "toolbar.save": "Tallenna",
  "toolbar.print": "Tulosta",
  "toolbar.export": "Vie",
  "toolbar.sidebar": "N\xE4yt\xE4/piilota sivupalkki",
  "toolbar.prevPage": "Edellinen sivu",
  "toolbar.nextPage": "Seuraava sivu",
  "toolbar.zoomIn": "L\xE4henn\xE4",
  "toolbar.zoomOut": "Loitonna",
  "toolbar.fitWidth": "Sovita leveyteen",
  "toolbar.fitPage": "Sovita sivuun",
  "toolbar.actualSize": "Todellinen koko",
  "toolbar.search": "Hae",
  "toolbar.fullscreen": "Koko n\xE4ytt\xF6",
  "toolbar.annotate": "Merkinn\xE4t",
  "toolbar.undo": "Kumoa",
  "toolbar.redo": "Tee uudelleen",
  "toolbar.more": "Lis\xE4\xE4",
  "toolbar.moreTools": "Lis\xE4\xE4 ty\xF6kaluja",
  "toolbar.zoomPresets": "Zoomausasetukset",
  "toolbar.skipToContent": "Siirry asiakirjaan",
  "toolbar.screenshot": "Vie sivu kuvana",
  // ── Tools ──
  "tool.pointer": "Osoitin",
  "tool.hand": "Panoroi",
  "tool.textSelect": "Valitse teksti",
  "tool.capture": "Kaappaa alue",
  // ── Marquee capture ──
  "capture.copy": "Kopioi",
  "capture.save": "Tallenna",
  "capture.actions": "Kaappaustoiminnot",
  "capture.copiedToClipboard": "Kopioitu leikep\xF6yd\xE4lle",
  "capture.copyFailed": "Kopiointi leikep\xF6yd\xE4lle ep\xE4onnistui",
  "capture.failed": "Kaappaus ep\xE4onnistui",
  // ── Drag and drop ──
  "dropzone.releaseToOpen": "Vapauta ladataksesi PDF:n",
  "dropzone.releaseToOpenMany": "Vapauta ladataksesi PDF-tiedostot",
  "dropzone.notPdf": "Vain PDF-tiedostoja voidaan avata",
  "dropzone.partialNotPdf": "Osa tiedostoista ohitettiin \u2014 vain PDF-tiedostoja voidaan avata",
  // ── Split-pane view ──
  "split.horizontal": "Jaa vaakasuunnassa",
  "split.vertical": "Jaa pystysuunnassa",
  "split.closeAll": "Sulje jaetut ruudut",
  "split.openToCompare": "Avaa PDF vertailua varten",
  "split.dropHint": "tai pudota PDF t\xE4h\xE4n",
  "split.emptySideLabel": "Tyhj\xE4",
  // ── Annotations ──
  "annotation.highlight": "Korostus",
  "annotation.underline": "Alleviivaus",
  "annotation.strikeout": "Yliviivaus",
  "annotation.squiggly": "Aaltoviivaus",
  "annotation.ink": "Kyn\xE4",
  "annotation.inkHighlighter": "Korostuskyn\xE4",
  "annotation.eraser": "Pyyhekumi",
  "annotation.freetext": "Tekstikentt\xE4",
  "annotation.stickyNote": "Muistilappu",
  "annotation.insertText": "Lis\xE4\xE4 teksti",
  "annotation.rectangle": "Suorakulmio",
  "annotation.circle": "Ympyr\xE4",
  "annotation.line": "Viiva",
  "annotation.arrow": "Nuoli",
  "annotation.polygon": "Monikulmio",
  "annotation.polyline": "Murtoviiva",
  "annotation.stamp": "Leima",
  "annotation.image": "Kuva",
  "annotation.imageHelp": "Pudota tai valitse kuva sijoitettavaksi sivulle",
  "annotation.imagePickFile": "Valitse kuva\u2026",
  "annotation.imageInvalid": "Kuvaa ei voitu ladata",
  "annotation.imageTooLarge": "Kuva on liian suuri (enint\xE4\xE4n 4 Mt)",
  "annotation.callout": "Selite",
  "annotation.calloutHint": "Napsauta sijoittaaksesi tekstikentt\xE4 ja napsauta uudelleen johtolinjaa varten",
  "annotation.calloutEdit": "Kaksoisnapsauta muokataksesi teksti\xE4",
  "annotation.redaction": "Peittomerkint\xE4",
  "annotation.applyRedaction": "Toteuta peitto",
  "annotation.measurement": "Mittaus",
  "annotation.signature": "Allekirjoitus",
  "annotation.color": "V\xE4ri",
  "annotation.fill": "T\xE4ytt\xF6",
  "annotation.borderWidth": "Reunan leveys",
  "annotation.lineStyle": "Viivatyyli",
  "annotation.opacity": "L\xE4pin\xE4kyvyys",
  "annotation.delete": "Poista",
  "annotation.icon": "Kuvake",
  "annotation.thickness": "Paksuus",
  "annotation.bringToFront": "Tuo eteen",
  "annotation.sendToBack": "Vie taakse",
  "annotation.group": "Ryhmit\xE4",
  "annotation.ungroup": "Pura ryhm\xE4",
  "annotation.multiSelectCount": "{count} merkint\xE4\xE4 valittu",
  // ── Line styles ──
  "lineStyle.solid": "Yhten\xE4inen",
  "lineStyle.dashed": "Katkoviiva",
  "lineStyle.dotted": "Pisteviiva",
  // ── Context menu ──
  "contextMenu.insertPageBefore": "Lis\xE4\xE4 sivu ennen",
  "contextMenu.insertPageAfter": "Lis\xE4\xE4 sivu j\xE4lkeen",
  "contextMenu.duplicatePage": "Kopioi sivu",
  "contextMenu.rotateCW": "Kierr\xE4 my\xF6t\xE4p\xE4iv\xE4\xE4n",
  "contextMenu.rotateCCW": "Kierr\xE4 vastap\xE4iv\xE4\xE4n",
  "contextMenu.deletePage": "Poista sivu",
  "contextMenu.copyText": "Kopioi teksti",
  "contextMenu.addNote": "Lis\xE4\xE4 muistiinpano",
  "contextMenu.highlight": "Korosta",
  // ── Attachments ──
  "attachment.addFile": "Lis\xE4\xE4 tiedosto",
  // ── Page / navigation ──
  "page.viewer": "Asiakirjan katselin",
  "page.goToPage": "Siirry sivulle",
  "page.goToPageN": "Siirry sivulle {page}",
  "page.documents": "Asiakirjat",
  // ── Freetext / overlays ──
  "annotation.doubleClickToEdit": "Kaksoisnapsauta muokataksesi",
  "annotation.defaultLabel": "Merkint\xE4",
  "form.button": "Painike",
  "form.clickToSign": "Napsauta allekirjoittaaksesi",
  // ── Sidebar ──
  "sidebar.thumbnails": "Sivut",
  "sidebar.outline": "Kirjanmerkit",
  "sidebar.annotations": "Merkinn\xE4t",
  "sidebar.attachments": "Liitteet",
  "sidebar.signatures": "Allekirjoitukset",
  "sidebar.layers": "Tasot",
  "sidebar.comparison": "Muutokset",
  // ── Comparison ──
  "comparison.title": "Asiakirjojen vertailu",
  "comparison.compare": "Vertaa",
  "comparison.exit": "Poistu vertailusta",
  "comparison.computing": "Vertaillaan asiakirjoja\u2026",
  "comparison.error": "Vertailu ep\xE4onnistui: {error}",
  "comparison.requireSplit": "Avaa kaksi PDF-tiedostoa jaettuun n\xE4kym\xE4\xE4n vertaillaksesi niit\xE4",
  "comparison.identical": "Molemmat asiakirjat ovat identtisi\xE4 \u2014 muutoksia ei havaittu",
  "comparison.totalChanges": "{count} muutos",
  "comparison.totalChangesPlural": "{count} muutosta",
  "comparison.noChangesOnPage": "Ei muutoksia t\xE4ll\xE4 sivulla",
  "comparison.pageHeading": "Sivu {a} \u2194 Sivu {b}",
  "comparison.pageHeadingInsert": "Sivu {b} (lis\xE4tty)",
  "comparison.pageHeadingDelete": "Sivu {a} (poistettu)",
  "comparison.pageHeadingMismatch": "Sivu {a} \u2194 Sivu {b} (teksti/skannaus)",
  "comparison.pageHeadingRegion": "Sivu {a} \u2194 Sivu {b} (visuaalinen)",
  "comparison.changeInsert": "Lis\xE4tty",
  "comparison.changeDelete": "Poistettu",
  "comparison.changeReplace": "Muutettu",
  "comparison.changeRegion": "Visuaalinen muutos",
  "comparison.filterAll": "Kaikki",
  "comparison.filterInsert": "Lis\xE4tty",
  "comparison.filterDelete": "Poistettu",
  "comparison.filterReplace": "Muutettu",
  "comparison.prevChange": "Edellinen muutos",
  "comparison.nextChange": "Seuraava muutos",
  "comparison.syncScroll": "Synkronoi vieritys",
  "comparison.changeOf": "{n}/{total}",
  "comparison.openSidebar": "Avaa muutospaneeli",
  // ── Comments ──
  "comment.addComment": "Lis\xE4\xE4 kommentti...",
  "comment.reply": "Vastaa...",
  "comment.replyOrMention": "Vastaa tai mainitse @...",
  "comment.commentOrMention": "Kommentoi tai mainitse @...",
  "comment.edit": "Muokkaa",
  "comment.delete": "Poista",
  "comment.cancel": "Peruuta",
  "comment.post": "L\xE4het\xE4",
  "comment.save": "Tallenna",
  "comment.resolve": "Ratkaise",
  "comment.reopen": "Avaa uudelleen",
  "comment.edited": "(muokattu)",
  // ── Comments sidebar ──
  "comments.title": "Kommentit",
  "comments.empty": "Ei viel\xE4 kommentteja.\nNapsauta merkint\xE4\xE4 sivulla aloittaaksesi keskustelun.",
  "comments.noDocument": "Asiakirjaa ei ole avattu",
  "comments.noBody": "Ei teksti\xE4",
  "comments.reply": "vastaus",
  "comments.replies": "vastausta",
  "comments.filter.all": "Kaikki",
  "comments.filter.open": "Avoimet",
  "comments.filter.resolved": "Ratkaistut",
  "comments.filter.mine": "Omat",
  "comments.filter.mentions": "@ min\xE4",
  "comments.sort.page": "Sivun mukaan",
  "comments.sort.date": "P\xE4iv\xE4m\xE4\xE4r\xE4n mukaan",
  "comments.sort.author": "Tekij\xE4n mukaan",
  "common.page": "Sivu",
  "common.close": "Sulje",
  "toolbar.comments": "N\xE4yt\xE4/piilota kommentit",
  // ── Comment status ──
  "commentStatus.open": "Avoin",
  "commentStatus.accepted": "Hyv\xE4ksytty",
  "commentStatus.rejected": "Hyl\xE4tty",
  "commentStatus.completed": "Valmis",
  "commentStatus.cancelled": "Peruutettu",
  "commentStatus.resolved": "Ratkaistu",
  // ── Page operations ──
  "pageOps.delete": "Poista sivu",
  "pageOps.rotate": "Kierr\xE4 sivua",
  "pageOps.rotateCW": "Kierr\xE4 my\xF6t\xE4p\xE4iv\xE4\xE4n",
  "pageOps.rotateCCW": "Kierr\xE4 vastap\xE4iv\xE4\xE4n",
  "pageOps.insert": "Lis\xE4\xE4 tyhj\xE4 sivu",
  "pageOps.duplicate": "Kopioi sivu",
  "pageOps.move": "Siirr\xE4 sivu",
  // ── Search ──
  "search.placeholder": "Hae asiakirjasta...",
  "search.noResults": "Ei tuloksia",
  "search.matchCase": "Erota isot ja pienet kirjaimet",
  "search.matchWholeWord": "Kokonaiset sanat",
  "search.resultsCount": "{current}/{total}",
  "search.previous": "Edellinen osuma (Shift+Enter)",
  "search.next": "Seuraava osuma (Enter)",
  "search.close": "Sulje haku (Esc)",
  // ── Pagination ──
  "pagination.of": "/",
  "pagination.page": "Sivu",
  "pagination.pageOf": "Sivu {current}/{total}",
  // ── Zoom ──
  "zoom.level": "{value} %",
  // ── Layout ──
  "layout.single": "Yksi sivu",
  "layout.continuous": "Jatkuva",
  "layout.double": "Kaksisivun\xE4kym\xE4",
  // ── Theme ──
  "theme.light": "Vaalea",
  "theme.dark": "Tumma",
  "theme.system": "J\xE4rjestelm\xE4",
  // ── Stamps ──
  "stamp.approved": "Hyv\xE4ksytty",
  "stamp.notApproved": "Ei hyv\xE4ksytty",
  "stamp.draft": "Luonnos",
  "stamp.final": "Lopullinen",
  "stamp.completed": "Valmis",
  "stamp.confidential": "Luottamuksellinen",
  "stamp.forPublicRelease": "Julkinen",
  "stamp.notForPublicRelease": "Ei julkinen",
  "stamp.forComment": "Kommentoitavaksi",
  "stamp.void": "Mit\xE4t\xF6n",
  "stamp.asIs": "Sellaisenaan",
  "stamp.departmental": "Osastokohtainen",
  "stamp.experimental": "Kokeellinen",
  "stamp.expired": "Vanhentunut",
  "stamp.informationOnly": "Vain tiedoksi",
  "stamp.preliminaryResults": "Alustavat tulokset",
  "stamp.sold": "Myyty",
  "stamp.topSecret": "Eritt\xE4in salainen",
  // ── Signatures ──
  "signature.signed": "Allekirjoitettu",
  "signature.unsigned": "Allekirjoittamaton",
  "signature.valid": "Kelvollinen allekirjoitus",
  "signature.invalid": "Virheellinen allekirjoitus",
  "signature.unknown": "Allekirjoituksen kelpoisuus tuntematon",
  // ── Measurement ──
  "measurement.distance": "Et\xE4isyys",
  "measurement.area": "Pinta-ala",
  "measurement.perimeter": "Keh\xE4",
  "measurement.calibrate": "Aseta mittakaava\u2026",
  "measurement.calibrateTitle": "Kalibroi mittakaava",
  "measurement.calibrateDesc": "M\xE4\xE4rit\xE4 miten PDF-et\xE4isyydet vastaavat todellisia mittoja.",
  "measurement.currentScale": "Nykyinen mittakaava: {source} {sourceUnit} = {target} {targetUnit}",
  "measurement.pdfDistance": "PDF-et\xE4isyys",
  "measurement.realDistance": "Todellinen et\xE4isyys",
  "measurement.unit": "Yksikk\xF6",
  "measurement.precision": "Desimaalit",
  "measurement.clearScale": "Tyhjenn\xE4 mittakaava",
  "toast.calibrationSaved": "Mittakaava tallennettu",
  "toast.calibrationCleared": "Mittakaava tyhjennetty",
  "toast.calibrationInvalid": "Molempien et\xE4isyyksien on oltava suurempia kuin nolla",
  // ── Errors ──
  "error.loadFailed": "Asiakirjan lataaminen ep\xE4onnistui",
  "error.passwordRequired": "T\xE4m\xE4 asiakirja on salasanasuojattu",
  "error.renderFailed": "Sivun piirt\xE4minen ep\xE4onnistui",
  // ── Misc ──
  "misc.loading": "Ladataan...",
  "misc.unknown": "Tuntematon",
  "misc.none": "Ei mit\xE4\xE4n",
  // ── Common UI ──
  "common.cancel": "Peruuta",
  "common.save": "Tallenna",
  "common.open": "Avaa",
  "common.delete": "Poista",
  "common.apply": "K\xE4yt\xE4",
  "common.ok": "OK",
  "common.yes": "Kyll\xE4",
  "common.no": "Ei",
  "common.selectAll": "Valitse kaikki",
  "common.download": "Lataa",
  // ── Toolbar extras (shortcut hints) ──
  "toolbar.undoShortcut": "Kumoa (Ctrl+Z)",
  "toolbar.redoShortcut": "Tee uudelleen (Ctrl+Y)",
  "toolbar.searchShortcut": "Hae (Ctrl+F)",
  "toolbar.passwordProtect": "Salasanasuojaus",
  "toolbar.signDocument": "Allekirjoita asiakirja",
  // ── Zoom control tooltips ──
  "zoom.zoomIn": "L\xE4henn\xE4",
  "zoom.zoomOut": "Loitonna",
  // ── Toasts / ephemeral feedback ──
  "toast.bookmarkAdded": "Kirjanmerkki lis\xE4tty",
  "toast.bookmarkDeleted": "Kirjanmerkki poistettu",
  "toast.attached": "Liitetty: {name}",
  "toast.removed": "Poistettu: {name}",
  "toast.noRedactions": "Ei peittomerkint\xF6j\xE4 toteutettavaksi",
  "toast.redactionsApplied": "{count} peittomerkint\xE4 toteutettu",
  "toast.redactionsAppliedPlural": "{count} peittomerkint\xE4\xE4 toteutettu",
  "toast.redactionsFailed": "Peittomerkint\xF6jen toteuttaminen ep\xE4onnistui: {error}",
  "toast.noDocument": "Asiakirjaa ei ole avattu",
  "toast.protectedSaved": "Suojattu asiakirja tallennettu",
  "toast.invalidPageRange": "Virheellinen sivualue",
  "toast.printFailed": "Tulostus ep\xE4onnistui: {error}",
  "toast.documentSaved": "Asiakirja tallennettu",
  "toast.saveFailed": "Tallennus ep\xE4onnistui: {error}",
  "toast.pageExported": "Sivu {page} viety",
  "toast.exportFailed": "Vienti ep\xE4onnistui: {error}",
  "toast.signatureMissing": "Luo allekirjoitus tai lataa varmenne",
  "toast.signatureApplied": "Allekirjoitus lis\xE4tty",
  "toast.signedSaved": "Allekirjoitettu asiakirja tallennettu",
  "toast.copied": "Kopioitu",
  "toast.openFailed": "Avaaminen ep\xE4onnistui: {error}",
  "toast.openNamedFailed": "Tiedoston {name} avaaminen ep\xE4onnistui: {error}",
  // ── Bookmarks panel ──
  "bookmarks.add": "Lis\xE4\xE4 kirjanmerkki",
  "bookmarks.empty": "Ei kirjanmerkkej\xE4 t\xE4ss\xE4 asiakirjassa",
  "bookmarks.delete": "Poista",
  "bookmarks.rename": "Nime\xE4 uudelleen",
  "bookmarks.dragHint": "J\xE4rjest\xE4 vet\xE4m\xE4ll\xE4",
  "bookmarks.navigationPluginMissing": "Navigointilis\xE4osaa ei ole ladattu",
  "toast.bookmarkRenamed": "Kirjanmerkki nimetty uudelleen",
  "toast.bookmarkMoved": "Kirjanmerkki siirretty",
  // ── Annotation presets ──
  "presets.tooltip": "Merkint\xE4asetukset",
  "presets.none": "Ei esiasetusta",
  "presets.empty": "Ei viel\xE4 esiasetuksia",
  "presets.saveCurrent": "Tallenna nykyinen tyyli\u2026",
  "presets.namePrompt": "Esiasetuksen nimi:",
  "presets.dialogTitle": "Tallenna esiasetus",
  "presets.namePlaceholder": "Oma esiasetus",
  "presets.nameRequired": "Anna esiasetukselle nimi",
  "toast.presetSaved": "Esiasetus tallennettu",
  "toast.presetDeleted": "Esiasetus poistettu",
  // ── Annotations list panel ──
  "annotations.empty": "Ei merkint\xF6j\xE4 t\xE4ss\xE4 asiakirjassa",
  "annotations.pluginMissing": "Merkint\xE4lis\xE4osaa ei ole ladattu",
  "annotations.resolved": "Ratkaistu",
  "annotations.pageHeader": "Sivu {page}",
  // ── Attachments panel ──
  "attachments.empty": "Ei liitteit\xE4 t\xE4ss\xE4 asiakirjassa",
  "attachments.pluginMissing": "Liitelis\xE4osaa ei ole ladattu",
  // ── Signatures panel ──
  "signatures.empty": "Ei digitaalisia allekirjoituksia t\xE4ss\xE4 asiakirjassa",
  "signatures.pluginMissing": "Allekirjoituslis\xE4osaa ei ole ladattu",
  "signatures.validating": "Vahvistetaan allekirjoituksia\u2026",
  "signatures.validatingShort": "Vahvistetaan\u2026",
  "signatures.validationError": "Vahvistusvirhe: {error}",
  "signatures.itemTitle": "Allekirjoitus {index}",
  "signatures.badgeValid": "Kelvollinen",
  "signatures.badgeInvalid": "Virheellinen",
  "signatures.badgeError": "Virhe",
  "signatures.badgeUnknown": "Tuntematon",
  "signatures.integrityVerified": "Asiakirjan eheys vahvistettu",
  "signatures.integrityModified": "Asiakirjaa muokattu allekirjoituksen j\xE4lkeen",
  "signatures.integrityUnknown": "Eheys tuntematon",
  "signatures.cryptoValid": "Allekirjoitus kryptografisesti kelvollinen",
  "signatures.signatureValid": "Allekirjoitus kelvollinen",
  "signatures.signatureInvalid": "Allekirjoitus virheellinen",
  "signatures.certNotTrusted": "Varmennetta ei luoteta",
  "signatures.verificationNeeded": "Allekirjoituksen vahvistus tarvitaan",
  "signatures.certExpired": "Varmenne on vanhentunut",
  "signatures.certSelfSigned": "Itse allekirjoitettu varmenne",
  "signatures.certIssuer": "My\xF6nt\xE4j\xE4: {issuer}",
  "signatures.algorithm": "Algoritmi: {algorithm}",
  "signatures.format": "Muoto: {format}",
  "signatures.reason": "Syy: {reason}",
  "signatures.signedAt": "Allekirjoitettu: {time}",
  "signatures.covers": "Kattaa: {size}",
  "signatures.validationHint": "Vahvistus: {error}",
  "signatures.badgeStatusSingle": "Digitaalinen allekirjoitus",
  "signatures.badgeStatusMany": "{count} digitaalista allekirjoitusta",
  "signatures.fullDetailsHint": "Napsauta n\xE4hd\xE4ksesi tiedot",
  "signatures.perm.approval": "Hyv\xE4ksynt\xE4allekirjoitus (lis\xE4muutokset sallittu)",
  "signatures.perm.noChanges": "Varmennettu \u2014 muutokset eiv\xE4t ole sallittuja",
  "signatures.perm.formSigning": "Varmennettu \u2014 vain lomakkeiden t\xE4ytt\xF6 ja allekirjoitus",
  "signatures.perm.formCommenting": "Varmennettu \u2014 lomakkeiden t\xE4ytt\xF6, allekirjoitus ja kommentointi",
  "signatures.perm.unknown": "K\xE4ytt\xF6oikeustaso: {level}",
  "signatures.format.pkcs7Detached": "PKCS#7 detached (Adobe)",
  "signatures.format.pkcs7Sha1": "PKCS#7 SHA-1 (Adobe, vanha)",
  "signatures.format.x509RsaSha1": "X.509 RSA SHA-1 (Adobe, vanha)",
  "signatures.format.cadesDetached": "CAdES detached (PAdES)",
  "signatures.format.rfc3161": "RFC 3161 -aikaleima",
  // ── Layers panel ──
  "layers.empty": "Ei tasoja t\xE4ss\xE4 asiakirjassa",
  "layers.pluginMissing": "Tasot-lis\xE4osaa ei ole ladattu",
  // ── Apply Redactions dialog ──
  "redact.dialogTitle": "Toteuta peittomerkinn\xE4t",
  "redact.applyButton": "Toteuta peittomerkinn\xE4t",
  "redact.applying": "Toteutetaan...",
  "redact.summarySingle": "<strong>1</strong> peittomerkint\xE4 toteutetaan.",
  "redact.summaryPlural": "<strong>{count}</strong> peittomerkint\xE4\xE4 toteutetaan.",
  "redact.warning": "T\xE4m\xE4 poistaa pysyv\xE4sti kaiken sis\xE4ll\xF6n peitetyilt\xE4 alueilta. T\xE4t\xE4 toimintoa <strong>ei voi kumota</strong>.",
  // ── Password (open) dialog ──
  "password.openTitle": "Salasana vaaditaan",
  "password.openPlaceholder": "Sy\xF6t\xE4 salasana",
  "password.opening": "Avataan...",
  "password.incorrect": "V\xE4\xE4r\xE4 salasana. Yrit\xE4 uudelleen.",
  "password.openFailed": "Avaaminen ep\xE4onnistui: {error}",
  "password.descNamed": 'Asiakirja "{name}" on salasanasuojattu.',
  "password.desc": "T\xE4m\xE4 asiakirja on salasanasuojattu.",
  // ── Password protect dialog ──
  "protect.dialogTitle": "Salasanasuojaa asiakirja",
  "protect.userLabel": "Avaussalasana (vaaditaan katselua varten)",
  "protect.userPlaceholder": "Sy\xF6t\xE4 salasana",
  "protect.confirmLabel": "Vahvista salasana",
  "protect.confirmPlaceholder": "Vahvista salasana",
  "protect.ownerLabel": "Omistajan salasana (valinnainen \u2014 t\xE4ydet oikeudet)",
  "protect.ownerPlaceholder": "Omistajan salasana (tyhj\xE4 = sama kuin avaussalasana)",
  "protect.permissionsLabel": "K\xE4ytt\xF6oikeudet",
  "protect.applyButton": "Suojaa ja tallenna",
  "protect.applying": "Suojataan...",
  "protect.errorRequired": "Salasana on pakollinen",
  "protect.errorMismatch": "Salasanat eiv\xE4t t\xE4sm\xE4\xE4",
  "protect.errorFailed": "Ep\xE4onnistui: {error}",
  "protect.perm.print": "Salli tulostus",
  "protect.perm.extract": "Salli kopiointi/poiminta",
  "protect.perm.annotate": "Salli merkinn\xE4t ja lomakkeet",
  "protect.perm.modify": "Salli sis\xE4ll\xF6n muokkaus",
  // ── Print dialog ──
  "print.dialogTitle": "Tulosta asiakirja",
  "print.pagesLabel": "Sivut",
  "print.allPages": "Kaikki sivut (1-{total})",
  "print.customRange": "Mukautettu alue:",
  "print.rangePlaceholder": "esim. 1-3, 5, 8-10",
  "print.printButton": "Tulosta",
  "print.rendering": "Piirret\xE4\xE4n...",
  // ── Signature dialog ──
  "sign.dialogTitle": "Allekirjoita asiakirja",
  "sign.clearButton": "Tyhjenn\xE4",
  "sign.uploadPrompt": "Napsauta ladataksesi tai ved\xE4 kuva",
  "sign.typedPlaceholder": "Kirjoita nimesi",
  "sign.certLabel": "Digitaalinen varmenne (valinnainen)",
  "sign.certUpload": "Lataa PFX/P12...",
  "sign.certNone": "Varmennetta ei valittu",
  "sign.certPasswordPlaceholder": "Varmenteen salasana",
  "sign.reasonPlaceholder": "Allekirjoituksen syy (valinnainen)",
  "sign.tsaPlaceholder": "TSA URL (valinnainen, esim. http://timestamp.digicert.com)",
  "sign.applyButton": "Lis\xE4\xE4 allekirjoitus",
  "sign.applying": "Allekirjoitetaan...",
  "sign.tab.draw": "Piirr\xE4",
  "sign.tab.type": "Kirjoita",
  "sign.tab.image": "Kuva",
  "sign.mdp.approval": "Salli lis\xE4muutokset (hyv\xE4ksynt\xE4allekirjoitus)",
  "sign.mdp.formCommenting": "Salli lomakkeiden t\xE4ytt\xF6, allekirjoitus ja kommentointi",
  "sign.mdp.formSigning": "Salli vain lomakkeiden t\xE4ytt\xF6 ja allekirjoitus",
  "sign.mdp.noChanges": "Muutokset eiv\xE4t ole sallittuja (lukitse asiakirja)",
  // ── Split pane close buttons (tooltips) ──
  "split.closeThisPane": "Sulje t\xE4m\xE4 ruutu",
  "split.closeSplit": "Sulje jako",
  "split.openPdf": "Avaa PDF",
  // ── Stamp picker ──
  "stamp.pickerTooltip": "Leima"
};

// src/i18n/translations/es.ts
var es = {
  // ── Toolbar ──
  "toolbar.menu": "Men\xFA",
  "toolbar.open": "Abrir",
  "toolbar.close": "Cerrar",
  "toolbar.save": "Guardar",
  "toolbar.print": "Imprimir",
  "toolbar.export": "Exportar",
  "toolbar.sidebar": "Alternar barra lateral",
  "toolbar.prevPage": "P\xE1gina anterior",
  "toolbar.nextPage": "P\xE1gina siguiente",
  "toolbar.zoomIn": "Acercar",
  "toolbar.zoomOut": "Alejar",
  "toolbar.fitWidth": "Ajustar al ancho",
  "toolbar.fitPage": "Ajustar a la p\xE1gina",
  "toolbar.actualSize": "Tama\xF1o real",
  "toolbar.search": "Buscar",
  "toolbar.fullscreen": "Pantalla completa",
  "toolbar.annotate": "Anotar",
  "toolbar.undo": "Deshacer",
  "toolbar.redo": "Rehacer",
  "toolbar.more": "M\xE1s",
  "toolbar.moreTools": "M\xE1s herramientas",
  "toolbar.zoomPresets": "Niveles de zoom",
  "toolbar.skipToContent": "Ir al documento",
  "toolbar.screenshot": "Exportar p\xE1gina como imagen",
  // ── Tools ──
  "tool.pointer": "Puntero",
  "tool.hand": "Desplazar",
  "tool.textSelect": "Seleccionar texto",
  "tool.capture": "Capturar regi\xF3n",
  // ── Marquee capture ──
  "capture.copy": "Copiar",
  "capture.save": "Guardar",
  "capture.actions": "Acciones de captura",
  "capture.copiedToClipboard": "Copiado al portapapeles",
  "capture.copyFailed": "Error al copiar al portapapeles",
  "capture.failed": "Error en la captura",
  // ── Drag and drop ──
  "dropzone.releaseToOpen": "Suelte para cargar el PDF",
  "dropzone.releaseToOpenMany": "Suelte para cargar los PDF",
  "dropzone.notPdf": "Solo se pueden abrir archivos PDF",
  "dropzone.partialNotPdf": "Algunos archivos fueron ignorados \u2014 solo se pueden abrir archivos PDF",
  // ── Split-pane view ──
  "split.horizontal": "Dividir horizontalmente",
  "split.vertical": "Dividir verticalmente",
  "split.closeAll": "Cerrar paneles divididos",
  "split.openToCompare": "Abrir PDF para comparar",
  "split.dropHint": "o arrastre un PDF aqu\xED",
  "split.emptySideLabel": "Vac\xEDo",
  // ── Annotations ──
  "annotation.highlight": "Resaltado",
  "annotation.underline": "Subrayado",
  "annotation.strikeout": "Tachado",
  "annotation.squiggly": "Ondulado",
  "annotation.ink": "L\xE1piz",
  "annotation.inkHighlighter": "Resaltador",
  "annotation.eraser": "Borrador",
  "annotation.freetext": "Cuadro de texto",
  "annotation.stickyNote": "Nota adhesiva",
  "annotation.insertText": "Insertar texto",
  "annotation.rectangle": "Rect\xE1ngulo",
  "annotation.circle": "C\xEDrculo",
  "annotation.line": "L\xEDnea",
  "annotation.arrow": "Flecha",
  "annotation.polygon": "Pol\xEDgono",
  "annotation.polyline": "Polil\xEDnea",
  "annotation.stamp": "Sello",
  "annotation.image": "Imagen",
  "annotation.imageHelp": "Arrastre o seleccione una imagen para colocarla en la p\xE1gina",
  "annotation.imagePickFile": "Elegir imagen\u2026",
  "annotation.imageInvalid": "No se pudo cargar esa imagen",
  "annotation.imageTooLarge": "Imagen demasiado grande (m\xE1x. 4 MB)",
  "annotation.callout": "Llamada",
  "annotation.calloutHint": "Haga clic para colocar el cuadro de texto, luego haga clic de nuevo para el punto de referencia",
  "annotation.calloutEdit": "Doble clic para editar el texto",
  "annotation.redaction": "Redacci\xF3n",
  "annotation.applyRedaction": "Aplicar redacci\xF3n",
  "annotation.measurement": "Medici\xF3n",
  "annotation.signature": "Firma",
  "annotation.color": "Color",
  "annotation.fill": "Relleno",
  "annotation.borderWidth": "Ancho de borde",
  "annotation.lineStyle": "Estilo de l\xEDnea",
  "annotation.opacity": "Opacidad",
  "annotation.delete": "Eliminar",
  "annotation.icon": "Icono",
  "annotation.thickness": "Grosor",
  "annotation.bringToFront": "Traer al frente",
  "annotation.sendToBack": "Enviar al fondo",
  "annotation.group": "Agrupar",
  "annotation.ungroup": "Desagrupar",
  "annotation.multiSelectCount": "{count} anotaciones seleccionadas",
  // ── Line styles ──
  "lineStyle.solid": "S\xF3lido",
  "lineStyle.dashed": "Discontinuo",
  "lineStyle.dotted": "Punteado",
  // ── Context menu ──
  "contextMenu.insertPageBefore": "Insertar p\xE1gina antes",
  "contextMenu.insertPageAfter": "Insertar p\xE1gina despu\xE9s",
  "contextMenu.duplicatePage": "Duplicar p\xE1gina",
  "contextMenu.rotateCW": "Girar en sentido horario",
  "contextMenu.rotateCCW": "Girar en sentido antihorario",
  "contextMenu.deletePage": "Eliminar p\xE1gina",
  "contextMenu.copyText": "Copiar texto",
  "contextMenu.addNote": "Agregar nota",
  "contextMenu.highlight": "Resaltar",
  // ── Attachments ──
  "attachment.addFile": "Agregar archivo",
  // ── Page / navigation ──
  "page.viewer": "Visor de documentos",
  "page.goToPage": "Ir a la p\xE1gina",
  "page.goToPageN": "Ir a la p\xE1gina {page}",
  "page.documents": "Documentos",
  // ── Freetext / overlays ──
  "annotation.doubleClickToEdit": "Doble clic para editar",
  "annotation.defaultLabel": "Anotaci\xF3n",
  "form.button": "Bot\xF3n",
  "form.clickToSign": "Haga clic para firmar",
  // ── Sidebar ──
  "sidebar.thumbnails": "P\xE1ginas",
  "sidebar.outline": "Marcadores",
  "sidebar.annotations": "Anotaciones",
  "sidebar.attachments": "Adjuntos",
  "sidebar.signatures": "Firmas",
  "sidebar.layers": "Capas",
  "sidebar.comparison": "Cambios",
  // ── Comparison ──
  "comparison.title": "Comparaci\xF3n de documentos",
  "comparison.compare": "Comparar",
  "comparison.exit": "Salir de la comparaci\xF3n",
  "comparison.computing": "Comparando documentos\u2026",
  "comparison.error": "Error en la comparaci\xF3n: {error}",
  "comparison.requireSplit": "Abra dos PDF en vista dividida para compararlos",
  "comparison.identical": "Ambos documentos son id\xE9nticos \u2014 no se detectaron cambios",
  "comparison.totalChanges": "{count} cambio",
  "comparison.totalChangesPlural": "{count} cambios",
  "comparison.noChangesOnPage": "Sin cambios en esta p\xE1gina",
  "comparison.pageHeading": "P\xE1gina {a} \u2194 P\xE1gina {b}",
  "comparison.pageHeadingInsert": "P\xE1gina {b} (a\xF1adida)",
  "comparison.pageHeadingDelete": "P\xE1gina {a} (eliminada)",
  "comparison.pageHeadingMismatch": "P\xE1gina {a} \u2194 P\xE1gina {b} (texto/escaneo mixto)",
  "comparison.pageHeadingRegion": "P\xE1gina {a} \u2194 P\xE1gina {b} (visual)",
  "comparison.changeInsert": "A\xF1adido",
  "comparison.changeDelete": "Eliminado",
  "comparison.changeReplace": "Modificado",
  "comparison.changeRegion": "Cambio visual",
  "comparison.filterAll": "Todos",
  "comparison.filterInsert": "A\xF1adidos",
  "comparison.filterDelete": "Eliminados",
  "comparison.filterReplace": "Modificados",
  "comparison.prevChange": "Cambio anterior",
  "comparison.nextChange": "Cambio siguiente",
  "comparison.syncScroll": "Sincronizar desplazamiento",
  "comparison.changeOf": "{n} de {total}",
  "comparison.openSidebar": "Abrir panel de cambios",
  // ── Comments ──
  "comment.addComment": "Agregar un comentario...",
  "comment.reply": "Responder...",
  "comment.replyOrMention": "Responder o usar @ para mencionar...",
  "comment.commentOrMention": "Comentar o usar @ para mencionar...",
  "comment.edit": "Editar",
  "comment.delete": "Eliminar",
  "comment.cancel": "Cancelar",
  "comment.post": "Publicar",
  "comment.save": "Guardar",
  "comment.resolve": "Resolver",
  "comment.reopen": "Reabrir",
  "comment.edited": "(editado)",
  // ── Comments sidebar ──
  "comments.title": "Comentarios",
  "comments.empty": "A\xFAn no hay comentarios.\nHaga clic en cualquier anotaci\xF3n de la p\xE1gina para iniciar un hilo.",
  "comments.noDocument": "No hay documento abierto",
  "comments.noBody": "Sin texto",
  "comments.reply": "respuesta",
  "comments.replies": "respuestas",
  "comments.filter.all": "Todos",
  "comments.filter.open": "Abiertos",
  "comments.filter.resolved": "Resueltos",
  "comments.filter.mine": "M\xEDos",
  "comments.filter.mentions": "@ m\xED",
  "comments.sort.page": "Por p\xE1gina",
  "comments.sort.date": "Por fecha",
  "comments.sort.author": "Por autor",
  "common.page": "P\xE1gina",
  "common.close": "Cerrar",
  "toolbar.comments": "Alternar comentarios",
  // ── Comment status ──
  "commentStatus.open": "Abierto",
  "commentStatus.accepted": "Aceptado",
  "commentStatus.rejected": "Rechazado",
  "commentStatus.completed": "Completado",
  "commentStatus.cancelled": "Cancelado",
  "commentStatus.resolved": "Resuelto",
  // ── Page operations ──
  "pageOps.delete": "Eliminar p\xE1gina",
  "pageOps.rotate": "Girar p\xE1gina",
  "pageOps.rotateCW": "Girar en sentido horario",
  "pageOps.rotateCCW": "Girar en sentido antihorario",
  "pageOps.insert": "Insertar p\xE1gina en blanco",
  "pageOps.duplicate": "Duplicar p\xE1gina",
  "pageOps.move": "Mover p\xE1gina",
  // ── Search ──
  "search.placeholder": "Buscar en el documento...",
  "search.noResults": "Sin resultados",
  "search.matchCase": "Distinguir may\xFAsculas",
  "search.matchWholeWord": "Palabras completas",
  "search.resultsCount": "{current} de {total}",
  "search.previous": "Coincidencia anterior (Shift+Enter)",
  "search.next": "Coincidencia siguiente (Enter)",
  "search.close": "Cerrar b\xFAsqueda (Esc)",
  // ── Pagination ──
  "pagination.of": "de",
  "pagination.page": "P\xE1gina",
  "pagination.pageOf": "P\xE1gina {current} de {total}",
  // ── Zoom ──
  "zoom.level": "{value}%",
  // ── Layout ──
  "layout.single": "P\xE1gina \xFAnica",
  "layout.continuous": "Continuo",
  "layout.double": "Doble p\xE1gina",
  // ── Theme ──
  "theme.light": "Claro",
  "theme.dark": "Oscuro",
  "theme.system": "Sistema",
  // ── Stamps ──
  "stamp.approved": "Aprobado",
  "stamp.notApproved": "No aprobado",
  "stamp.draft": "Borrador",
  "stamp.final": "Final",
  "stamp.completed": "Completado",
  "stamp.confidential": "Confidencial",
  "stamp.forPublicRelease": "Para difusi\xF3n p\xFAblica",
  "stamp.notForPublicRelease": "No para difusi\xF3n p\xFAblica",
  "stamp.forComment": "Para comentarios",
  "stamp.void": "Nulo",
  "stamp.asIs": "Tal cual",
  "stamp.departmental": "Departamental",
  "stamp.experimental": "Experimental",
  "stamp.expired": "Vencido",
  "stamp.informationOnly": "Solo informativo",
  "stamp.preliminaryResults": "Resultados preliminares",
  "stamp.sold": "Vendido",
  "stamp.topSecret": "Alto secreto",
  // ── Signatures ──
  "signature.signed": "Firmado",
  "signature.unsigned": "Sin firmar",
  "signature.valid": "Firma v\xE1lida",
  "signature.invalid": "Firma no v\xE1lida",
  "signature.unknown": "Validez de firma desconocida",
  // ── Measurement ──
  "measurement.distance": "Distancia",
  "measurement.area": "\xC1rea",
  "measurement.perimeter": "Per\xEDmetro",
  "measurement.calibrate": "Definir escala\u2026",
  "measurement.calibrateTitle": "Calibrar escala de medici\xF3n",
  "measurement.calibrateDesc": "Defina c\xF3mo las distancias del PDF se corresponden con unidades reales.",
  "measurement.currentScale": "Escala actual: {source} {sourceUnit} = {target} {targetUnit}",
  "measurement.pdfDistance": "Distancia en el PDF",
  "measurement.realDistance": "Distancia real",
  "measurement.unit": "Unidad",
  "measurement.precision": "Decimales",
  "measurement.clearScale": "Borrar escala",
  "toast.calibrationSaved": "Escala de medici\xF3n guardada",
  "toast.calibrationCleared": "Escala de medici\xF3n borrada",
  "toast.calibrationInvalid": "Ambas distancias deben ser mayores que cero",
  // ── Errors ──
  "error.loadFailed": "Error al cargar el documento",
  "error.passwordRequired": "Este documento est\xE1 protegido con contrase\xF1a",
  "error.renderFailed": "Error al renderizar la p\xE1gina",
  // ── Misc ──
  "misc.loading": "Cargando...",
  "misc.unknown": "Desconocido",
  "misc.none": "Ninguno",
  // ── Common UI ──
  "common.cancel": "Cancelar",
  "common.save": "Guardar",
  "common.open": "Abrir",
  "common.delete": "Eliminar",
  "common.apply": "Aplicar",
  "common.ok": "Aceptar",
  "common.yes": "S\xED",
  "common.no": "No",
  "common.selectAll": "Seleccionar todo",
  "common.download": "Descargar",
  // ── Toolbar extras (shortcut hints) ──
  "toolbar.undoShortcut": "Deshacer (Ctrl+Z)",
  "toolbar.redoShortcut": "Rehacer (Ctrl+Y)",
  "toolbar.searchShortcut": "Buscar (Ctrl+F)",
  "toolbar.passwordProtect": "Proteger con contrase\xF1a",
  "toolbar.signDocument": "Firmar documento",
  // ── Zoom control tooltips ──
  "zoom.zoomIn": "Acercar",
  "zoom.zoomOut": "Alejar",
  // ── Toasts / ephemeral feedback ──
  "toast.bookmarkAdded": "Marcador agregado",
  "toast.bookmarkDeleted": "Marcador eliminado",
  "toast.attached": "Adjuntado: {name}",
  "toast.removed": "Eliminado: {name}",
  "toast.noRedactions": "No hay redacciones que aplicar",
  "toast.redactionsApplied": "{count} redacci\xF3n aplicada",
  "toast.redactionsAppliedPlural": "{count} redacciones aplicadas",
  "toast.redactionsFailed": "Error al aplicar redacciones: {error}",
  "toast.noDocument": "No hay documento abierto",
  "toast.protectedSaved": "Documento protegido guardado",
  "toast.invalidPageRange": "Rango de p\xE1ginas no v\xE1lido",
  "toast.printFailed": "Error de impresi\xF3n: {error}",
  "toast.documentSaved": "Documento guardado",
  "toast.saveFailed": "Error al guardar: {error}",
  "toast.pageExported": "P\xE1gina {page} exportada",
  "toast.exportFailed": "Error de exportaci\xF3n: {error}",
  "toast.signatureMissing": "Cree una firma o cargue un certificado",
  "toast.signatureApplied": "Firma aplicada",
  "toast.signedSaved": "Documento firmado guardado",
  "toast.copied": "Copiado",
  "toast.openFailed": "Error al abrir: {error}",
  "toast.openNamedFailed": "Error al abrir {name}: {error}",
  // ── Bookmarks panel ──
  "bookmarks.add": "Agregar marcador",
  "bookmarks.empty": "No hay marcadores en este documento",
  "bookmarks.delete": "Eliminar",
  "bookmarks.rename": "Cambiar nombre",
  "bookmarks.dragHint": "Arrastre para reordenar",
  "bookmarks.navigationPluginMissing": "Complemento de navegaci\xF3n no cargado",
  "toast.bookmarkRenamed": "Marcador renombrado",
  "toast.bookmarkMoved": "Marcador movido",
  // ── Annotation presets ──
  "presets.tooltip": "Estilos predefinidos",
  "presets.none": "Sin estilo predefinido",
  "presets.empty": "A\xFAn no hay estilos predefinidos",
  "presets.saveCurrent": "Guardar estilo actual\u2026",
  "presets.namePrompt": "Nombre para este estilo predefinido:",
  "presets.dialogTitle": "Guardar estilo predefinido",
  "presets.namePlaceholder": "Mi estilo",
  "presets.nameRequired": "Ingrese un nombre para el estilo predefinido",
  "toast.presetSaved": "Estilo predefinido guardado",
  "toast.presetDeleted": "Estilo predefinido eliminado",
  // ── Annotations list panel ──
  "annotations.empty": "No hay anotaciones en este documento",
  "annotations.pluginMissing": "Complemento de anotaciones no cargado",
  "annotations.resolved": "Resuelto",
  "annotations.pageHeader": "P\xE1gina {page}",
  // ── Attachments panel ──
  "attachments.empty": "No hay adjuntos en este documento",
  "attachments.pluginMissing": "Complemento de adjuntos no cargado",
  // ── Signatures panel ──
  "signatures.empty": "No hay firmas digitales en este documento",
  "signatures.pluginMissing": "Complemento de firmas no cargado",
  "signatures.validating": "Validando firmas\u2026",
  "signatures.validatingShort": "Validando\u2026",
  "signatures.validationError": "Error de validaci\xF3n: {error}",
  "signatures.itemTitle": "Firma {index}",
  "signatures.badgeValid": "V\xE1lida",
  "signatures.badgeInvalid": "No v\xE1lida",
  "signatures.badgeError": "Error",
  "signatures.badgeUnknown": "Desconocida",
  "signatures.integrityVerified": "Integridad del documento verificada",
  "signatures.integrityModified": "Documento modificado despu\xE9s de la firma",
  "signatures.integrityUnknown": "Integridad desconocida",
  "signatures.cryptoValid": "Firma criptogr\xE1ficamente v\xE1lida",
  "signatures.signatureValid": "Firma v\xE1lida",
  "signatures.signatureInvalid": "Firma no v\xE1lida",
  "signatures.certNotTrusted": "Certificado no confiable",
  "signatures.verificationNeeded": "Verificaci\xF3n de firma necesaria",
  "signatures.certExpired": "El certificado ha expirado",
  "signatures.certSelfSigned": "Certificado autofirmado",
  "signatures.certIssuer": "Emisor: {issuer}",
  "signatures.algorithm": "Algoritmo: {algorithm}",
  "signatures.format": "Formato: {format}",
  "signatures.reason": "Motivo: {reason}",
  "signatures.signedAt": "Firmado: {time}",
  "signatures.covers": "Abarca: {size}",
  "signatures.validationHint": "Validaci\xF3n: {error}",
  "signatures.badgeStatusSingle": "Firma digital",
  "signatures.badgeStatusMany": "{count} firmas digitales",
  "signatures.fullDetailsHint": "Haga clic para ver los detalles completos",
  "signatures.perm.approval": "Firma de aprobaci\xF3n (se permiten cambios adicionales)",
  "signatures.perm.noChanges": "Certificado \u2014 no se permiten cambios",
  "signatures.perm.formSigning": "Certificado \u2014 solo rellenar formularios y firmar",
  "signatures.perm.formCommenting": "Certificado \u2014 rellenar formularios, firmar y comentar",
  "signatures.perm.unknown": "Nivel de permiso: {level}",
  "signatures.format.pkcs7Detached": "PKCS#7 separado (Adobe)",
  "signatures.format.pkcs7Sha1": "PKCS#7 SHA-1 (Adobe, heredado)",
  "signatures.format.x509RsaSha1": "X.509 RSA SHA-1 (Adobe, heredado)",
  "signatures.format.cadesDetached": "CAdES separado (PAdES)",
  "signatures.format.rfc3161": "Marca de tiempo RFC 3161",
  // ── Layers panel ──
  "layers.empty": "No hay capas en este documento",
  "layers.pluginMissing": "Complemento de capas no cargado",
  // ── Apply Redactions dialog ──
  "redact.dialogTitle": "Aplicar redacciones",
  "redact.applyButton": "Aplicar redacciones",
  "redact.applying": "Aplicando...",
  "redact.summarySingle": "Se aplicar\xE1 <strong>1</strong> redacci\xF3n.",
  "redact.summaryPlural": "Se aplicar\xE1n <strong>{count}</strong> redacciones.",
  "redact.warning": "Esto elimina permanentemente todo el contenido bajo las \xE1reas redactadas. Esta acci\xF3n <strong>no se puede deshacer</strong>.",
  // ── Password (open) dialog ──
  "password.openTitle": "Contrase\xF1a requerida",
  "password.openPlaceholder": "Ingrese la contrase\xF1a",
  "password.opening": "Abriendo...",
  "password.incorrect": "Contrase\xF1a incorrecta. Intente de nuevo.",
  "password.openFailed": "Error al abrir: {error}",
  "password.descNamed": 'El documento "{name}" est\xE1 protegido con contrase\xF1a.',
  "password.desc": "Este documento est\xE1 protegido con contrase\xF1a.",
  // ── Password protect dialog ──
  "protect.dialogTitle": "Proteger documento con contrase\xF1a",
  "protect.userLabel": "Contrase\xF1a de apertura (requerida para ver)",
  "protect.userPlaceholder": "Ingrese la contrase\xF1a",
  "protect.confirmLabel": "Confirmar contrase\xF1a",
  "protect.confirmPlaceholder": "Confirmar contrase\xF1a",
  "protect.ownerLabel": "Contrase\xF1a de propietario (opcional \u2014 para acceso completo)",
  "protect.ownerPlaceholder": "Contrase\xF1a de propietario (dejar vac\xEDo = igual que la de apertura)",
  "protect.permissionsLabel": "Permisos",
  "protect.applyButton": "Proteger y guardar",
  "protect.applying": "Protegiendo...",
  "protect.errorRequired": "La contrase\xF1a es obligatoria",
  "protect.errorMismatch": "Las contrase\xF1as no coinciden",
  "protect.errorFailed": "Error: {error}",
  "protect.perm.print": "Permitir impresi\xF3n",
  "protect.perm.extract": "Permitir copiar/extraer",
  "protect.perm.annotate": "Permitir anotaciones y formularios",
  "protect.perm.modify": "Permitir modificaci\xF3n del contenido",
  // ── Print dialog ──
  "print.dialogTitle": "Imprimir documento",
  "print.pagesLabel": "P\xE1ginas",
  "print.allPages": "Todas las p\xE1ginas (1-{total})",
  "print.customRange": "Rango personalizado:",
  "print.rangePlaceholder": "ej. 1-3, 5, 8-10",
  "print.printButton": "Imprimir",
  "print.rendering": "Renderizando...",
  // ── Signature dialog ──
  "sign.dialogTitle": "Firmar documento",
  "sign.clearButton": "Borrar",
  "sign.uploadPrompt": "Haga clic para cargar o arrastre una imagen",
  "sign.typedPlaceholder": "Escriba su nombre",
  "sign.certLabel": "Certificado digital (opcional)",
  "sign.certUpload": "Cargar PFX/P12...",
  "sign.certNone": "Ning\xFAn certificado seleccionado",
  "sign.certPasswordPlaceholder": "Contrase\xF1a del certificado",
  "sign.reasonPlaceholder": "Motivo de la firma (opcional)",
  "sign.tsaPlaceholder": "URL de TSA (opcional, ej. http://timestamp.digicert.com)",
  "sign.applyButton": "Aplicar firma",
  "sign.applying": "Firmando...",
  "sign.tab.draw": "Dibujar",
  "sign.tab.type": "Escribir",
  "sign.tab.image": "Imagen",
  "sign.mdp.approval": "Permitir cambios adicionales (firma de aprobaci\xF3n)",
  "sign.mdp.formCommenting": "Permitir rellenar formularios, firmar y comentar",
  "sign.mdp.formSigning": "Permitir solo rellenar formularios y firmar",
  "sign.mdp.noChanges": "No se permiten cambios (bloquear documento)",
  // ── Split pane close buttons (tooltips) ──
  "split.closeThisPane": "Cerrar este panel",
  "split.closeSplit": "Cerrar divisi\xF3n",
  "split.openPdf": "Abrir PDF",
  // ── Stamp picker ──
  "stamp.pickerTooltip": "Sello"
};

// src/i18n/translations/de.ts
var de = {
  // ── Toolbar ──
  "toolbar.menu": "Men\xFC",
  "toolbar.open": "\xD6ffnen",
  "toolbar.close": "Schlie\xDFen",
  "toolbar.save": "Speichern",
  "toolbar.print": "Drucken",
  "toolbar.export": "Exportieren",
  "toolbar.sidebar": "Seitenleiste ein-/ausblenden",
  "toolbar.prevPage": "Vorherige Seite",
  "toolbar.nextPage": "N\xE4chste Seite",
  "toolbar.zoomIn": "Vergr\xF6\xDFern",
  "toolbar.zoomOut": "Verkleinern",
  "toolbar.fitWidth": "Seitenbreite",
  "toolbar.fitPage": "Ganze Seite",
  "toolbar.actualSize": "Tats\xE4chliche Gr\xF6\xDFe",
  "toolbar.search": "Suchen",
  "toolbar.fullscreen": "Vollbild",
  "toolbar.annotate": "Kommentieren",
  "toolbar.undo": "R\xFCckg\xE4ngig",
  "toolbar.redo": "Wiederholen",
  "toolbar.more": "Mehr",
  "toolbar.moreTools": "Weitere Werkzeuge",
  "toolbar.zoomPresets": "Zoomvoreinstellungen",
  "toolbar.skipToContent": "Zum Dokument springen",
  "toolbar.screenshot": "Seite als Bild exportieren",
  // ── Tools ──
  "tool.pointer": "Zeiger",
  "tool.hand": "Schwenken",
  "tool.textSelect": "Text ausw\xE4hlen",
  "tool.capture": "Bereich erfassen",
  // ── Marquee capture ──
  "capture.copy": "Kopieren",
  "capture.save": "Speichern",
  "capture.actions": "Erfassungsaktionen",
  "capture.copiedToClipboard": "In die Zwischenablage kopiert",
  "capture.copyFailed": "Kopieren in die Zwischenablage fehlgeschlagen",
  "capture.failed": "Erfassung fehlgeschlagen",
  // ── Drag and drop ──
  "dropzone.releaseToOpen": "Loslassen, um PDF zu laden",
  "dropzone.releaseToOpenMany": "Loslassen, um PDFs zu laden",
  "dropzone.notPdf": "Es k\xF6nnen nur PDF-Dateien ge\xF6ffnet werden",
  "dropzone.partialNotPdf": "Einige Dateien wurden ignoriert \u2014 es k\xF6nnen nur PDF-Dateien ge\xF6ffnet werden",
  // ── Split-pane view ──
  "split.horizontal": "Horizontal teilen",
  "split.vertical": "Vertikal teilen",
  "split.closeAll": "Geteilte Ansicht schlie\xDFen",
  "split.openToCompare": "PDF zum Vergleichen \xF6ffnen",
  "split.dropHint": "oder PDF hierher ziehen",
  "split.emptySideLabel": "Leer",
  // ── Annotations ──
  "annotation.highlight": "Hervorheben",
  "annotation.underline": "Unterstreichen",
  "annotation.strikeout": "Durchstreichen",
  "annotation.squiggly": "Wellenunterstreichung",
  "annotation.ink": "Stift",
  "annotation.inkHighlighter": "Textmarker",
  "annotation.eraser": "Radierer",
  "annotation.freetext": "Textfeld",
  "annotation.stickyNote": "Haftnotiz",
  "annotation.insertText": "Text einf\xFCgen",
  "annotation.rectangle": "Rechteck",
  "annotation.circle": "Kreis",
  "annotation.line": "Linie",
  "annotation.arrow": "Pfeil",
  "annotation.polygon": "Polygon",
  "annotation.polyline": "Polylinie",
  "annotation.stamp": "Stempel",
  "annotation.image": "Bild",
  "annotation.imageHelp": "Bild hierher ziehen oder ausw\xE4hlen, um es auf der Seite zu platzieren",
  "annotation.imagePickFile": "Bild ausw\xE4hlen\u2026",
  "annotation.imageInvalid": "Dieses Bild konnte nicht geladen werden",
  "annotation.imageTooLarge": "Bild zu gro\xDF (max. 4 MB)",
  "annotation.callout": "Legende",
  "annotation.calloutHint": "Klicken, um das Textfeld zu platzieren, dann erneut klicken f\xFCr den F\xFChrungspunkt",
  "annotation.calloutEdit": "Doppelklicken zum Bearbeiten",
  "annotation.redaction": "Schw\xE4rzung",
  "annotation.applyRedaction": "Schw\xE4rzung anwenden",
  "annotation.measurement": "Messung",
  "annotation.signature": "Unterschrift",
  "annotation.color": "Farbe",
  "annotation.fill": "F\xFCllung",
  "annotation.borderWidth": "Rahmenbreite",
  "annotation.lineStyle": "Linienstil",
  "annotation.opacity": "Deckkraft",
  "annotation.delete": "L\xF6schen",
  "annotation.icon": "Symbol",
  "annotation.thickness": "St\xE4rke",
  "annotation.bringToFront": "In den Vordergrund",
  "annotation.sendToBack": "In den Hintergrund",
  "annotation.group": "Gruppieren",
  "annotation.ungroup": "Gruppierung aufheben",
  "annotation.multiSelectCount": "{count} Anmerkungen ausgew\xE4hlt",
  // ── Line styles ──
  "lineStyle.solid": "Durchgezogen",
  "lineStyle.dashed": "Gestrichelt",
  "lineStyle.dotted": "Gepunktet",
  // ── Context menu ──
  "contextMenu.insertPageBefore": "Seite davor einf\xFCgen",
  "contextMenu.insertPageAfter": "Seite danach einf\xFCgen",
  "contextMenu.duplicatePage": "Seite duplizieren",
  "contextMenu.rotateCW": "Im Uhrzeigersinn drehen",
  "contextMenu.rotateCCW": "Gegen den Uhrzeigersinn drehen",
  "contextMenu.deletePage": "Seite l\xF6schen",
  "contextMenu.copyText": "Text kopieren",
  "contextMenu.addNote": "Notiz hinzuf\xFCgen",
  "contextMenu.highlight": "Hervorheben",
  // ── Attachments ──
  "attachment.addFile": "Datei hinzuf\xFCgen",
  // ── Page / navigation ──
  "page.viewer": "Dokumentanzeige",
  "page.goToPage": "Gehe zu Seite",
  "page.goToPageN": "Gehe zu Seite {page}",
  "page.documents": "Dokumente",
  // ── Freetext / overlays ──
  "annotation.doubleClickToEdit": "Doppelklicken zum Bearbeiten",
  "annotation.defaultLabel": "Anmerkung",
  "form.button": "Schaltfl\xE4che",
  "form.clickToSign": "Zum Unterschreiben klicken",
  // ── Sidebar ──
  "sidebar.thumbnails": "Seiten",
  "sidebar.outline": "Lesezeichen",
  "sidebar.annotations": "Anmerkungen",
  "sidebar.attachments": "Anh\xE4nge",
  "sidebar.signatures": "Signaturen",
  "sidebar.layers": "Ebenen",
  "sidebar.comparison": "\xC4nderungen",
  // ── Comparison ──
  "comparison.title": "Dokumentvergleich",
  "comparison.compare": "Vergleichen",
  "comparison.exit": "Vergleich beenden",
  "comparison.computing": "Dokumente werden verglichen\u2026",
  "comparison.error": "Vergleich fehlgeschlagen: {error}",
  "comparison.requireSplit": "\xD6ffnen Sie zwei PDFs in der geteilten Ansicht, um sie zu vergleichen",
  "comparison.identical": "Beide Dokumente sind identisch \u2014 keine \xC4nderungen erkannt",
  "comparison.totalChanges": "{count} \xC4nderung",
  "comparison.totalChangesPlural": "{count} \xC4nderungen",
  "comparison.noChangesOnPage": "Keine \xC4nderungen auf dieser Seite",
  "comparison.pageHeading": "Seite {a} \u2194 Seite {b}",
  "comparison.pageHeadingInsert": "Seite {b} (hinzugef\xFCgt)",
  "comparison.pageHeadingDelete": "Seite {a} (entfernt)",
  "comparison.pageHeadingMismatch": "Seite {a} \u2194 Seite {b} (gemischt Text/Scan)",
  "comparison.pageHeadingRegion": "Seite {a} \u2194 Seite {b} (visuell)",
  "comparison.changeInsert": "Hinzugef\xFCgt",
  "comparison.changeDelete": "Entfernt",
  "comparison.changeReplace": "Ge\xE4ndert",
  "comparison.changeRegion": "Visuelle \xC4nderung",
  "comparison.filterAll": "Alle",
  "comparison.filterInsert": "Hinzugef\xFCgt",
  "comparison.filterDelete": "Entfernt",
  "comparison.filterReplace": "Ge\xE4ndert",
  "comparison.prevChange": "Vorherige \xC4nderung",
  "comparison.nextChange": "N\xE4chste \xC4nderung",
  "comparison.syncScroll": "Scrollen synchronisieren",
  "comparison.changeOf": "{n} von {total}",
  "comparison.openSidebar": "\xC4nderungsansicht \xF6ffnen",
  // ── Comments ──
  "comment.addComment": "Kommentar hinzuf\xFCgen...",
  "comment.reply": "Antworten...",
  "comment.replyOrMention": "Antworten oder @ zum Erw\xE4hnen...",
  "comment.commentOrMention": "Kommentieren oder @ zum Erw\xE4hnen...",
  "comment.edit": "Bearbeiten",
  "comment.delete": "L\xF6schen",
  "comment.cancel": "Abbrechen",
  "comment.post": "Senden",
  "comment.save": "Speichern",
  "comment.resolve": "Erledigen",
  "comment.reopen": "Erneut \xF6ffnen",
  "comment.edited": "(bearbeitet)",
  // ── Comments sidebar ──
  "comments.title": "Kommentare",
  "comments.empty": "Noch keine Kommentare.\nKlicken Sie auf eine Anmerkung auf der Seite, um einen Thread zu starten.",
  "comments.noDocument": "Kein Dokument ge\xF6ffnet",
  "comments.noBody": "Kein Text",
  "comments.reply": "Antwort",
  "comments.replies": "Antworten",
  "comments.filter.all": "Alle",
  "comments.filter.open": "Offen",
  "comments.filter.resolved": "Erledigt",
  "comments.filter.mine": "Meine",
  "comments.filter.mentions": "@ ich",
  "comments.sort.page": "Nach Seite",
  "comments.sort.date": "Nach Datum",
  "comments.sort.author": "Nach Autor",
  "common.page": "Seite",
  "common.close": "Schlie\xDFen",
  "toolbar.comments": "Kommentare ein-/ausblenden",
  // ── Comment status ──
  "commentStatus.open": "Offen",
  "commentStatus.accepted": "Akzeptiert",
  "commentStatus.rejected": "Abgelehnt",
  "commentStatus.completed": "Abgeschlossen",
  "commentStatus.cancelled": "Abgebrochen",
  "commentStatus.resolved": "Erledigt",
  // ── Page operations ──
  "pageOps.delete": "Seite l\xF6schen",
  "pageOps.rotate": "Seite drehen",
  "pageOps.rotateCW": "Im Uhrzeigersinn drehen",
  "pageOps.rotateCCW": "Gegen den Uhrzeigersinn drehen",
  "pageOps.insert": "Leere Seite einf\xFCgen",
  "pageOps.duplicate": "Seite duplizieren",
  "pageOps.move": "Seite verschieben",
  // ── Search ──
  "search.placeholder": "Im Dokument suchen...",
  "search.noResults": "Keine Ergebnisse",
  "search.matchCase": "Gro\xDF-/Kleinschreibung beachten",
  "search.matchWholeWord": "Ganze W\xF6rter",
  "search.resultsCount": "{current} von {total}",
  "search.previous": "Vorheriges Ergebnis (Shift+Enter)",
  "search.next": "N\xE4chstes Ergebnis (Enter)",
  "search.close": "Suche schlie\xDFen (Esc)",
  // ── Pagination ──
  "pagination.of": "von",
  "pagination.page": "Seite",
  "pagination.pageOf": "Seite {current} von {total}",
  // ── Zoom ──
  "zoom.level": "{value} %",
  // ── Layout ──
  "layout.single": "Einzelseite",
  "layout.continuous": "Fortlaufend",
  "layout.double": "Doppelseite",
  // ── Theme ──
  "theme.light": "Hell",
  "theme.dark": "Dunkel",
  "theme.system": "System",
  // ── Stamps ──
  "stamp.approved": "Genehmigt",
  "stamp.notApproved": "Nicht genehmigt",
  "stamp.draft": "Entwurf",
  "stamp.final": "Endg\xFCltig",
  "stamp.completed": "Abgeschlossen",
  "stamp.confidential": "Vertraulich",
  "stamp.forPublicRelease": "Zur Ver\xF6ffentlichung",
  "stamp.notForPublicRelease": "Nicht zur Ver\xF6ffentlichung",
  "stamp.forComment": "Zur Stellungnahme",
  "stamp.void": "Ung\xFCltig",
  "stamp.asIs": "Wie vorliegend",
  "stamp.departmental": "Abteilungsintern",
  "stamp.experimental": "Experimentell",
  "stamp.expired": "Abgelaufen",
  "stamp.informationOnly": "Nur zur Information",
  "stamp.preliminaryResults": "Vorl\xE4ufige Ergebnisse",
  "stamp.sold": "Verkauft",
  "stamp.topSecret": "Streng geheim",
  // ── Signatures ──
  "signature.signed": "Signiert",
  "signature.unsigned": "Nicht signiert",
  "signature.valid": "G\xFCltige Signatur",
  "signature.invalid": "Ung\xFCltige Signatur",
  "signature.unknown": "Signaturg\xFCltigkeit unbekannt",
  // ── Measurement ──
  "measurement.distance": "Abstand",
  "measurement.area": "Fl\xE4che",
  "measurement.perimeter": "Umfang",
  "measurement.calibrate": "Ma\xDFstab festlegen\u2026",
  "measurement.calibrateTitle": "Messma\xDFstab kalibrieren",
  "measurement.calibrateDesc": "Legen Sie fest, wie PDF-Abst\xE4nde realen Einheiten entsprechen.",
  "measurement.currentScale": "Aktueller Ma\xDFstab: {source} {sourceUnit} = {target} {targetUnit}",
  "measurement.pdfDistance": "PDF-Abstand",
  "measurement.realDistance": "Realer Abstand",
  "measurement.unit": "Einheit",
  "measurement.precision": "Dezimalstellen",
  "measurement.clearScale": "Ma\xDFstab zur\xFCcksetzen",
  "toast.calibrationSaved": "Messma\xDFstab gespeichert",
  "toast.calibrationCleared": "Messma\xDFstab zur\xFCckgesetzt",
  "toast.calibrationInvalid": "Beide Abst\xE4nde m\xFCssen gr\xF6\xDFer als null sein",
  // ── Errors ──
  "error.loadFailed": "Dokument konnte nicht geladen werden",
  "error.passwordRequired": "Dieses Dokument ist passwortgesch\xFCtzt",
  "error.renderFailed": "Seite konnte nicht gerendert werden",
  // ── Misc ──
  "misc.loading": "Laden...",
  "misc.unknown": "Unbekannt",
  "misc.none": "Keine",
  // ── Common UI ──
  "common.cancel": "Abbrechen",
  "common.save": "Speichern",
  "common.open": "\xD6ffnen",
  "common.delete": "L\xF6schen",
  "common.apply": "\xDCbernehmen",
  "common.ok": "OK",
  "common.yes": "Ja",
  "common.no": "Nein",
  "common.selectAll": "Alle ausw\xE4hlen",
  "common.download": "Herunterladen",
  // ── Toolbar extras (shortcut hints) ──
  "toolbar.undoShortcut": "R\xFCckg\xE4ngig (Strg+Z)",
  "toolbar.redoShortcut": "Wiederholen (Strg+Y)",
  "toolbar.searchShortcut": "Suchen (Strg+F)",
  "toolbar.passwordProtect": "Passwortschutz",
  "toolbar.signDocument": "Dokument signieren",
  // ── Zoom control tooltips ──
  "zoom.zoomIn": "Vergr\xF6\xDFern",
  "zoom.zoomOut": "Verkleinern",
  // ── Toasts / ephemeral feedback ──
  "toast.bookmarkAdded": "Lesezeichen hinzugef\xFCgt",
  "toast.bookmarkDeleted": "Lesezeichen gel\xF6scht",
  "toast.attached": "Angeh\xE4ngt: {name}",
  "toast.removed": "Entfernt: {name}",
  "toast.noRedactions": "Keine Schw\xE4rzungen vorhanden",
  "toast.redactionsApplied": "{count} Schw\xE4rzung angewendet",
  "toast.redactionsAppliedPlural": "{count} Schw\xE4rzungen angewendet",
  "toast.redactionsFailed": "Schw\xE4rzungen konnten nicht angewendet werden: {error}",
  "toast.noDocument": "Kein Dokument ge\xF6ffnet",
  "toast.protectedSaved": "Gesch\xFCtztes Dokument gespeichert",
  "toast.invalidPageRange": "Ung\xFCltiger Seitenbereich",
  "toast.printFailed": "Drucken fehlgeschlagen: {error}",
  "toast.documentSaved": "Dokument gespeichert",
  "toast.saveFailed": "Speichern fehlgeschlagen: {error}",
  "toast.pageExported": "Seite {page} exportiert",
  "toast.exportFailed": "Export fehlgeschlagen: {error}",
  "toast.signatureMissing": "Bitte erstellen Sie eine Unterschrift oder laden Sie ein Zertifikat hoch",
  "toast.signatureApplied": "Unterschrift angewendet",
  "toast.signedSaved": "Signiertes Dokument gespeichert",
  "toast.copied": "Kopiert",
  "toast.openFailed": "\xD6ffnen fehlgeschlagen: {error}",
  "toast.openNamedFailed": "{name} konnte nicht ge\xF6ffnet werden: {error}",
  // ── Bookmarks panel ──
  "bookmarks.add": "Lesezeichen hinzuf\xFCgen",
  "bookmarks.empty": "Keine Lesezeichen in diesem Dokument",
  "bookmarks.delete": "L\xF6schen",
  "bookmarks.rename": "Umbenennen",
  "bookmarks.dragHint": "Ziehen zum Sortieren",
  "bookmarks.navigationPluginMissing": "Navigations-Plugin nicht geladen",
  "toast.bookmarkRenamed": "Lesezeichen umbenannt",
  "toast.bookmarkMoved": "Lesezeichen verschoben",
  // ── Annotation presets ──
  "presets.tooltip": "Anmerkungsvorlagen",
  "presets.none": "Keine Vorlage",
  "presets.empty": "Noch keine Vorlagen",
  "presets.saveCurrent": "Aktuellen Stil speichern\u2026",
  "presets.namePrompt": "Name f\xFCr diese Vorlage:",
  "presets.dialogTitle": "Vorlage speichern",
  "presets.namePlaceholder": "Meine Vorlage",
  "presets.nameRequired": "Bitte geben Sie einen Namen f\xFCr die Vorlage ein",
  "toast.presetSaved": "Vorlage gespeichert",
  "toast.presetDeleted": "Vorlage gel\xF6scht",
  // ── Annotations list panel ──
  "annotations.empty": "Keine Anmerkungen in diesem Dokument",
  "annotations.pluginMissing": "Anmerkungs-Plugin nicht geladen",
  "annotations.resolved": "Erledigt",
  "annotations.pageHeader": "Seite {page}",
  // ── Attachments panel ──
  "attachments.empty": "Keine Anh\xE4nge in diesem Dokument",
  "attachments.pluginMissing": "Anhang-Plugin nicht geladen",
  // ── Signatures panel ──
  "signatures.empty": "Keine digitalen Signaturen in diesem Dokument",
  "signatures.pluginMissing": "Signatur-Plugin nicht geladen",
  "signatures.validating": "Signaturen werden \xFCberpr\xFCft\u2026",
  "signatures.validatingShort": "\xDCberpr\xFCfung\u2026",
  "signatures.validationError": "Validierungsfehler: {error}",
  "signatures.itemTitle": "Signatur {index}",
  "signatures.badgeValid": "G\xFCltig",
  "signatures.badgeInvalid": "Ung\xFCltig",
  "signatures.badgeError": "Fehler",
  "signatures.badgeUnknown": "Unbekannt",
  "signatures.integrityVerified": "Dokumentintegrit\xE4t \xFCberpr\xFCft",
  "signatures.integrityModified": "Dokument nach der Signierung ge\xE4ndert",
  "signatures.integrityUnknown": "Integrit\xE4t unbekannt",
  "signatures.cryptoValid": "Signatur kryptografisch g\xFCltig",
  "signatures.signatureValid": "Signatur g\xFCltig",
  "signatures.signatureInvalid": "Signatur ung\xFCltig",
  "signatures.certNotTrusted": "Zertifikat nicht vertrauensw\xFCrdig",
  "signatures.verificationNeeded": "Signatur\xFCberpr\xFCfung erforderlich",
  "signatures.certExpired": "Zertifikat abgelaufen",
  "signatures.certSelfSigned": "Selbstsigniertes Zertifikat",
  "signatures.certIssuer": "Aussteller: {issuer}",
  "signatures.algorithm": "Algorithmus: {algorithm}",
  "signatures.format": "Format: {format}",
  "signatures.reason": "Grund: {reason}",
  "signatures.signedAt": "Signiert: {time}",
  "signatures.covers": "Abdeckung: {size}",
  "signatures.validationHint": "Validierung: {error}",
  "signatures.badgeStatusSingle": "Digitale Signatur",
  "signatures.badgeStatusMany": "{count} digitale Signaturen",
  "signatures.fullDetailsHint": "Klicken f\xFCr alle Details",
  "signatures.perm.approval": "Genehmigungssignatur (weitere \xC4nderungen erlaubt)",
  "signatures.perm.noChanges": "Zertifiziert \u2014 keine \xC4nderungen erlaubt",
  "signatures.perm.formSigning": "Zertifiziert \u2014 nur Formularfelder und Signierung",
  "signatures.perm.formCommenting": "Zertifiziert \u2014 Formularfelder, Signierung und Kommentare",
  "signatures.perm.unknown": "Berechtigungsstufe: {level}",
  "signatures.format.pkcs7Detached": "PKCS#7 detached (Adobe)",
  "signatures.format.pkcs7Sha1": "PKCS#7 SHA-1 (Adobe, veraltet)",
  "signatures.format.x509RsaSha1": "X.509 RSA SHA-1 (Adobe, veraltet)",
  "signatures.format.cadesDetached": "CAdES detached (PAdES)",
  "signatures.format.rfc3161": "RFC 3161 Zeitstempel",
  // ── Layers panel ──
  "layers.empty": "Keine Ebenen in diesem Dokument",
  "layers.pluginMissing": "Ebenen-Plugin nicht geladen",
  // ── Apply Redactions dialog ──
  "redact.dialogTitle": "Schw\xE4rzungen anwenden",
  "redact.applyButton": "Schw\xE4rzungen anwenden",
  "redact.applying": "Wird angewendet...",
  "redact.summarySingle": "<strong>1</strong> Schw\xE4rzung wird angewendet.",
  "redact.summaryPlural": "<strong>{count}</strong> Schw\xE4rzungen werden angewendet.",
  "redact.warning": "Dadurch werden alle Inhalte unter den geschw\xE4rzten Bereichen dauerhaft entfernt. Diese Aktion kann <strong>nicht r\xFCckg\xE4ngig gemacht</strong> werden.",
  // ── Password (open) dialog ──
  "password.openTitle": "Passwort erforderlich",
  "password.openPlaceholder": "Passwort eingeben",
  "password.opening": "Wird ge\xF6ffnet...",
  "password.incorrect": "Falsches Passwort. Bitte versuchen Sie es erneut.",
  "password.openFailed": "\xD6ffnen fehlgeschlagen: {error}",
  "password.descNamed": 'Das Dokument \u201E{name}" ist passwortgesch\xFCtzt.',
  "password.desc": "Dieses Dokument ist passwortgesch\xFCtzt.",
  // ── Password protect dialog ──
  "protect.dialogTitle": "Dokument mit Passwort sch\xFCtzen",
  "protect.userLabel": "Passwort zum \xD6ffnen (zum Anzeigen erforderlich)",
  "protect.userPlaceholder": "Passwort eingeben",
  "protect.confirmLabel": "Passwort best\xE4tigen",
  "protect.confirmPlaceholder": "Passwort best\xE4tigen",
  "protect.ownerLabel": "Besitzerpasswort (optional \u2014 f\xFCr Vollzugriff)",
  "protect.ownerPlaceholder": "Besitzerpasswort (leer lassen = gleich wie \xD6ffnen-Passwort)",
  "protect.permissionsLabel": "Berechtigungen",
  "protect.applyButton": "Sch\xFCtzen und speichern",
  "protect.applying": "Wird gesch\xFCtzt...",
  "protect.errorRequired": "Passwort ist erforderlich",
  "protect.errorMismatch": "Passw\xF6rter stimmen nicht \xFCberein",
  "protect.errorFailed": "Fehlgeschlagen: {error}",
  "protect.perm.print": "Drucken erlauben",
  "protect.perm.extract": "Kopieren/Extrahieren erlauben",
  "protect.perm.annotate": "Anmerkungen und Formulare erlauben",
  "protect.perm.modify": "Inhalts\xE4nderungen erlauben",
  // ── Print dialog ──
  "print.dialogTitle": "Dokument drucken",
  "print.pagesLabel": "Seiten",
  "print.allPages": "Alle Seiten (1-{total})",
  "print.customRange": "Benutzerdefinierter Bereich:",
  "print.rangePlaceholder": "z. B. 1-3, 5, 8-10",
  "print.printButton": "Drucken",
  "print.rendering": "Wird gerendert...",
  // ── Signature dialog ──
  "sign.dialogTitle": "Dokument signieren",
  "sign.clearButton": "L\xF6schen",
  "sign.uploadPrompt": "Klicken zum Hochladen oder Bild hierher ziehen",
  "sign.typedPlaceholder": "Namen eingeben",
  "sign.certLabel": "Digitales Zertifikat (optional)",
  "sign.certUpload": "PFX/P12 hochladen...",
  "sign.certNone": "Kein Zertifikat ausgew\xE4hlt",
  "sign.certPasswordPlaceholder": "Zertifikatspasswort",
  "sign.reasonPlaceholder": "Grund der Signierung (optional)",
  "sign.tsaPlaceholder": "TSA-URL (optional, z. B. http://timestamp.digicert.com)",
  "sign.applyButton": "Unterschrift anwenden",
  "sign.applying": "Wird signiert...",
  "sign.tab.draw": "Zeichnen",
  "sign.tab.type": "Eintippen",
  "sign.tab.image": "Bild",
  "sign.mdp.approval": "Weitere \xC4nderungen erlauben (Genehmigungssignatur)",
  "sign.mdp.formCommenting": "Formularfelder, Signierung und Kommentare erlauben",
  "sign.mdp.formSigning": "Nur Formularfelder und Signierung erlauben",
  "sign.mdp.noChanges": "Keine \xC4nderungen erlaubt (Dokument sperren)",
  // ── Split pane close buttons (tooltips) ──
  "split.closeThisPane": "Diesen Bereich schlie\xDFen",
  "split.closeSplit": "Geteilte Ansicht schlie\xDFen",
  "split.openPdf": "PDF \xF6ffnen",
  // ── Stamp picker ──
  "stamp.pickerTooltip": "Stempel"
};

// src/plugins/i18n-plugin.ts
var i18nPlugin = definePlugin({
  id: "i18n",
  provides: ["i18n"],
  requires: [],
  optional: [],
  setup(ctx) {
    const engineAny = ctx.engine;
    const initialLocale = engineAny.locale;
    const manager = new I18nManager(initialLocale ?? "en");
    manager.addTranslations("sv", sv);
    manager.addTranslations("nb", nb);
    manager.addTranslations("da", da);
    manager.addTranslations("fi", fi);
    manager.addTranslations("es", es);
    manager.addTranslations("de", de);
    const translations = engineAny.translations;
    if (translations) {
      for (const [locale, map] of Object.entries(translations)) {
        manager.addTranslations(locale, map);
      }
    }
    const bundledLocales = [
      { id: "en", label: "English" },
      { id: "sv", label: "Svenska" },
      { id: "nb", label: "Norsk bokm\xE5l" },
      { id: "da", label: "Dansk" },
      { id: "fi", label: "Suomi" },
      { id: "es", label: "Espa\xF1ol" },
      { id: "de", label: "Deutsch" }
    ];
    for (const loc of bundledLocales) {
      ctx.registerCommand({
        id: `i18n.locale-${loc.id}`,
        label: loc.label,
        category: "Language",
        execute: () => {
          manager.setLocale(loc.id);
          ctx.emit("i18n:locale-changed", loc.id);
        }
      });
    }
    return {
      t: (key, params) => manager.t(key, params),
      locale: manager.locale,
      setLocale(locale) {
        manager.setLocale(locale);
        ctx.emit("i18n:locale-changed", locale);
      },
      addTranslations: (locale, map) => manager.addTranslations(locale, map),
      hasLocale: (locale) => manager.hasLocale(locale),
      getLocales: () => manager.getLocales()
    };
  },
  dispose() {
  }
});

// src/i18n/formatting-manager.ts
import { signal as signal15 } from "@truespar/lector-utils";
var DEFAULT_LOCALE = "en-US";
var IMPERIAL_REGIONS = /* @__PURE__ */ new Set(["US", "LR", "MM"]);
function detectBrowserLocale() {
  if (typeof navigator !== "undefined" && typeof navigator.language === "string" && navigator.language.length > 0) {
    return navigator.language;
  }
  return DEFAULT_LOCALE;
}
function deriveMeasurementSystem(locale) {
  try {
    const region = new Intl.Locale(locale).maximize().region;
    if (region && IMPERIAL_REGIONS.has(region)) return "imperial";
  } catch {
  }
  return "metric";
}
var FormattingManager = class {
  #locale$;
  #system$;
  #hourCycle$;
  #explicitSystem;
  // Cached formatters keyed by JSON of options. Recreated when locale changes.
  #cacheKey = "";
  #fmtCache = /* @__PURE__ */ new Map();
  constructor(options = {}) {
    const initialLocale = options.formatLocale ?? detectBrowserLocale();
    this.#locale$ = signal15(initialLocale);
    this.#explicitSystem = options.measurementSystem !== void 0;
    this.#system$ = signal15(
      options.measurementSystem ?? deriveMeasurementSystem(initialLocale)
    );
    this.#hourCycle$ = signal15(options.hourCycle);
    this.#cacheKey = this.#computeCacheKey();
  }
  /** Active BCP 47 format locale as a reactive signal. */
  get locale() {
    return this.#locale$;
  }
  /** Active measurement system as a reactive signal. */
  get measurementSystem() {
    return this.#system$;
  }
  /** Active hour-cycle override (or `undefined` to use locale default). */
  get hourCycle() {
    return this.#hourCycle$;
  }
  /**
   * Change the active format locale at runtime.
   *
   * If the measurement system was auto-derived (not explicitly provided
   * at construction), it is re-derived from the new locale's region.
   */
  setFormatLocale(locale) {
    if (this.#locale$.peek() === locale) return;
    this.#locale$.value = locale;
    if (!this.#explicitSystem) {
      this.#system$.value = deriveMeasurementSystem(locale);
    }
    this.#invalidateCache();
  }
  /** Change the measurement system. */
  setMeasurementSystem(system) {
    if (this.#system$.peek() === system) return;
    this.#system$.value = system;
  }
  /** Change the hour-cycle override (or pass `undefined` to clear it). */
  setHourCycle(cycle) {
    if (this.#hourCycle$.peek() === cycle) return;
    this.#hourCycle$.value = cycle;
    this.#invalidateCache();
  }
  // ─── Date / time ───────────────────────────────────────────────
  /**
   * Parse a PDF date string (`D:YYYYMMDDHHmmSSOHH'mm'`) into a `Date`,
   * or fall back to `Date` constructor for ISO/RFC strings.
   *
   * Returns `null` if parsing fails.
   */
  parsePdfDate(input) {
    if (!input) return null;
    const pdfDate = input.replace(/^D:/, "");
    const match = /^(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?([Z+-])?(\d{2})?'?(\d{2})?'?$/.exec(pdfDate);
    let d;
    if (match) {
      const [, year, month, day, hh = "00", mm = "00", ss = "00", tzSign, tzH = "00", tzM = "00"] = match;
      const tz = tzSign === "Z" || !tzSign ? "Z" : `${tzSign}${tzH}:${tzM}`;
      d = /* @__PURE__ */ new Date(`${year}-${month}-${day}T${hh}:${mm}:${ss}${tz}`);
    } else {
      d = new Date(input);
    }
    return Number.isNaN(d.getTime()) ? null : d;
  }
  /** Format a date as a localized medium-style date (no time). */
  formatDate(input) {
    const d = this.#coerce(input);
    if (!d) return typeof input === "string" ? input : "";
    return this.#dateTimeFmt({ dateStyle: "medium" }).format(d);
  }
  /** Format a date as a localized short-style time (no date). */
  formatTime(input) {
    const d = this.#coerce(input);
    if (!d) return typeof input === "string" ? input : "";
    return this.#dateTimeFmt({ timeStyle: "short" }).format(d);
  }
  /** Format a date as combined medium date + short time. */
  formatDateTime(input) {
    const d = this.#coerce(input);
    if (!d) return typeof input === "string" ? input : "";
    return this.#dateTimeFmt({ dateStyle: "medium", timeStyle: "short" }).format(d);
  }
  // ─── Numbers ───────────────────────────────────────────────────
  /** Locale-aware number formatting. */
  formatNumber(value, options) {
    return this.#numberFmt(options).format(value);
  }
  /**
   * Format a byte count as a human-readable string with locale-correct
   * decimal separator. Uses binary prefixes (1 KB = 1024 B) consistent
   * with how the rest of the codebase reports sizes.
   */
  formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) return "";
    if (bytes < 1024) {
      return `${this.formatNumber(bytes, { maximumFractionDigits: 0 })} B`;
    }
    if (bytes < 1024 * 1024) {
      return `${this.formatNumber(bytes / 1024, { maximumFractionDigits: 1 })} KB`;
    }
    if (bytes < 1024 * 1024 * 1024) {
      return `${this.formatNumber(bytes / (1024 * 1024), { maximumFractionDigits: 1 })} MB`;
    }
    return `${this.formatNumber(bytes / (1024 * 1024 * 1024), { maximumFractionDigits: 2 })} GB`;
  }
  // ─── Lengths and areas ─────────────────────────────────────────
  /**
   * Format a length given in PDF points (1 pt = 1/72 inch) using the
   * active measurement system. Picks an appropriate sub-unit based on
   * magnitude.
   *
   * **Metric** uses CAD/engineering conventions: millimetres for sub-metre
   * lengths, metres for ≥1 m. Centimetres are intentionally skipped — they
   * are an everyday unit and are not used in technical drawings, which is
   * the dominant use case for a PDF measurement tool.
   *
   * **Imperial** uses inches up to 12, feet up to 3, yards beyond.
   */
  formatLengthFromPoints(points, precision = 2) {
    const inches = points / 72;
    const system = this.#system$.peek();
    if (system === "imperial") {
      if (Math.abs(inches) < 12) return `${this.formatNumber(inches, { maximumFractionDigits: precision })} in`;
      const ft = inches / 12;
      if (Math.abs(ft) < 3) return `${this.formatNumber(ft, { maximumFractionDigits: precision })} ft`;
      return `${this.formatNumber(ft / 3, { maximumFractionDigits: precision })} yd`;
    }
    const mm = inches * 25.4;
    if (Math.abs(mm) < 1e3) return `${this.formatNumber(mm, { maximumFractionDigits: precision })} mm`;
    return `${this.formatNumber(mm / 1e3, { maximumFractionDigits: precision })} m`;
  }
  /**
   * Format an area given in square PDF points using the active
   * measurement system.
   *
   * **Metric** mirrors the length convention: square millimetres for
   * sub-square-metre areas, square metres for ≥1 m². No cm² for the same
   * reason length skips cm — it isn't a technical drawing convention.
   *
   * **Imperial** uses square inches up to 1 ft² (144 in²), then square feet.
   */
  formatAreaFromSquarePoints(squarePoints, precision = 2) {
    const sqInches = squarePoints / (72 * 72);
    const system = this.#system$.peek();
    if (system === "imperial") {
      if (Math.abs(sqInches) < 144) return `${this.formatNumber(sqInches, { maximumFractionDigits: precision })} in\xB2`;
      const sqFt = sqInches / 144;
      return `${this.formatNumber(sqFt, { maximumFractionDigits: precision })} ft\xB2`;
    }
    const sqMm = sqInches * (25.4 * 25.4);
    if (Math.abs(sqMm) < 1e6) return `${this.formatNumber(sqMm, { maximumFractionDigits: precision })} mm\xB2`;
    return `${this.formatNumber(sqMm / 1e6, { maximumFractionDigits: precision })} m\xB2`;
  }
  [Symbol.dispose]() {
    this.#fmtCache.clear();
  }
  // ─── Internals ─────────────────────────────────────────────────
  #coerce(input) {
    if (input instanceof Date) return Number.isNaN(input.getTime()) ? null : input;
    if (typeof input === "number") {
      const d = new Date(input);
      return Number.isNaN(d.getTime()) ? null : d;
    }
    return this.parsePdfDate(input);
  }
  #computeCacheKey() {
    return `${this.#locale$.peek()}|${this.#hourCycle$.peek() ?? ""}`;
  }
  #invalidateCache() {
    const next = this.#computeCacheKey();
    if (next !== this.#cacheKey) {
      this.#cacheKey = next;
      this.#fmtCache.clear();
    }
  }
  #dateTimeFmt(options) {
    const hour = this.#hourCycle$.peek();
    const opts = hour ? { ...options, hourCycle: hour } : options;
    const key = `dt:${JSON.stringify(opts)}`;
    let fmt = this.#fmtCache.get(key);
    if (!fmt) {
      fmt = new Intl.DateTimeFormat(this.#locale$.peek(), opts);
      this.#fmtCache.set(key, fmt);
    }
    return fmt;
  }
  #numberFmt(options) {
    const key = `n:${options ? JSON.stringify(options) : ""}`;
    let fmt = this.#fmtCache.get(key);
    if (!fmt) {
      fmt = new Intl.NumberFormat(this.#locale$.peek(), options);
      this.#fmtCache.set(key, fmt);
    }
    return fmt;
  }
};

// src/plugins/formatting-plugin.ts
var formattingPlugin = definePlugin({
  id: "formatting",
  provides: ["formatting"],
  requires: [],
  optional: [],
  setup(ctx) {
    const engineAny = ctx.engine;
    const opts = {
      formatLocale: engineAny.formatLocale,
      measurementSystem: engineAny.measurementSystem,
      hourCycle: engineAny.hourCycle
    };
    const manager = new FormattingManager(opts);
    return {
      locale: manager.locale,
      measurementSystem: manager.measurementSystem,
      hourCycle: manager.hourCycle,
      setFormatLocale(locale) {
        manager.setFormatLocale(locale);
        ctx.emit("formatting:locale-changed", locale);
      },
      setMeasurementSystem(system) {
        manager.setMeasurementSystem(system);
        ctx.emit("formatting:system-changed", system);
      },
      setHourCycle(cycle) {
        manager.setHourCycle(cycle);
        ctx.emit("formatting:hour-cycle-changed", cycle);
      },
      parsePdfDate: (input) => manager.parsePdfDate(input),
      formatDate: (input) => manager.formatDate(input),
      formatTime: (input) => manager.formatTime(input),
      formatDateTime: (input) => manager.formatDateTime(input),
      formatNumber: (value, options) => manager.formatNumber(value, options),
      formatFileSize: (bytes) => manager.formatFileSize(bytes),
      formatLengthFromPoints: (points, precision) => manager.formatLengthFromPoints(points, precision),
      formatAreaFromSquarePoints: (sq, precision) => manager.formatAreaFromSquarePoints(sq, precision)
    };
  },
  dispose() {
  }
});

// src/plugins/capture-plugin.ts
import { signal as signal16 } from "@truespar/lector-utils";
var BUILTIN_DEFAULT_DPI = 300;
var capturePlugin = definePlugin({
  id: "capture",
  provides: ["capture"],
  requires: ["document", "viewport", "interaction"],
  optional: [],
  setup(ctx) {
    const document2 = ctx.require("document");
    const viewport = ctx.require("viewport");
    const interaction = ctx.require("interaction");
    const isMarqueeActive$ = signal16(false);
    let dragState = null;
    function getScrollAreaFor(container) {
      if (!container) return null;
      return container.querySelector(".lector-canvas__scroll-area");
    }
    function rectFromDragState(state) {
      if (state.startPagePoint.pageIndex !== state.currentPagePoint.pageIndex) {
        return null;
      }
      const x = Math.min(state.startPagePoint.x, state.currentPagePoint.x);
      const y = Math.min(state.startPagePoint.y, state.currentPagePoint.y);
      const width = Math.abs(state.currentPagePoint.x - state.startPagePoint.x);
      const height = Math.abs(state.currentPagePoint.y - state.startPagePoint.y);
      if (width < 4 || height < 4) return null;
      return { pageIndex: state.startPagePoint.pageIndex, x, y, width, height };
    }
    function clearDragVisual() {
      if (dragState) {
        dragState.svg.remove();
        dragState = null;
      }
    }
    interaction.registerHandler("marquee", {
      cursor: "crosshair",
      onPointerDown(event) {
        const pp = event.pagePoint;
        if (!pp) return;
        const vp = event.viewport ?? viewport.activeViewport.peek();
        if (!vp) return;
        const scrollArea = getScrollAreaFor(event.container ?? vp.container);
        if (!scrollArea) return;
        const positions = vp.pagePositions.peek();
        const pos = positions.find((p) => p.pageIndex === pp.pageIndex);
        if (!pos) return;
        const svg = window.document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("width", String(pos.width));
        svg.setAttribute("height", String(pos.height));
        svg.style.cssText = `position:absolute;left:${pos.x}px;top:${pos.y}px;pointer-events:none;z-index:200;overflow:visible;`;
        svg.classList.add("lector-marquee-preview");
        const rectEl = window.document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rectEl.setAttribute("fill", "rgba(59, 130, 246, 0.18)");
        rectEl.setAttribute("stroke", "rgb(37, 99, 235)");
        rectEl.setAttribute("stroke-width", "1.5");
        rectEl.setAttribute("stroke-dasharray", "6 3");
        svg.appendChild(rectEl);
        scrollArea.appendChild(svg);
        dragState = { startPagePoint: pp, currentPagePoint: pp, svg, rectEl, viewport: vp };
        ctx.emit("capture:drag-started", pp);
      },
      onPointerMove(event) {
        if (!dragState || !event.pagePoint) return;
        if (event.pagePoint.pageIndex !== dragState.startPagePoint.pageIndex) return;
        dragState.currentPagePoint = event.pagePoint;
        const scale = dragState.viewport.scale.peek();
        const x1 = dragState.startPagePoint.x * scale;
        const y1 = dragState.startPagePoint.y * scale;
        const x2 = dragState.currentPagePoint.x * scale;
        const y2 = dragState.currentPagePoint.y * scale;
        dragState.rectEl.setAttribute("x", String(Math.min(x1, x2)));
        dragState.rectEl.setAttribute("y", String(Math.min(y1, y2)));
        dragState.rectEl.setAttribute("width", String(Math.abs(x2 - x1)));
        dragState.rectEl.setAttribute("height", String(Math.abs(y2 - y1)));
      },
      onPointerUp(_event) {
        if (!dragState) return;
        const rect = rectFromDragState(dragState);
        const sourceVp = dragState.viewport;
        clearDragVisual();
        if (!rect) {
          ctx.emit("capture:cancelled");
          return;
        }
        ctx.emit("capture:region-selected", {
          rect,
          docId: sourceVp.docId.peek(),
          viewportId: sourceVp.id
        });
      },
      onKeyDown(event) {
        if (event.key === "Escape") {
          clearDragVisual();
          capability.disableMarquee();
        }
      },
      onDeactivate() {
        clearDragVisual();
      }
    });
    ctx.effect(() => {
      isMarqueeActive$.value = interaction.mode.value === "marquee";
    });
    const capability = {
      isMarqueeActive: isMarqueeActive$,
      enableMarquee() {
        interaction.setMode("marquee");
        ctx.emit("capture:mode-enabled");
      },
      disableMarquee() {
        if (interaction.mode.peek() === "marquee") {
          interaction.setMode("pointer");
          ctx.emit("capture:mode-disabled");
        }
      },
      toggleMarquee() {
        if (interaction.mode.peek() === "marquee") {
          this.disableMarquee();
        } else {
          this.enableMarquee();
        }
      },
      async captureRegion(docId, rect, options) {
        const dpi = options?.dpi ?? ctx.engine.captureDpi ?? BUILTIN_DEFAULT_DPI;
        const bitmap = await ctx.engine.workerProxy.captureRegion(docId, {
          pageIndex: rect.pageIndex,
          rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
          dpi,
          rotation: options?.rotation ?? 0
        });
        const blob = await imageBitmapToPngBlob(bitmap);
        return { bitmap, blob, rect, dpi };
      }
    };
    ctx.registerCommand({
      id: "capture.toggle-marquee",
      label: "Capture region",
      icon: "crop",
      shortcut: "C",
      category: "Tools",
      execute: () => capability.toggleMarquee()
    });
    ctx.registerCommand({
      id: "capture.disable",
      label: "Cancel capture",
      category: "Tools",
      execute: () => capability.disableMarquee()
    });
    void document2;
    return capability;
  },
  dispose() {
  }
});
async function imageBitmapToPngBlob(bitmap) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("OffscreenCanvas 2D context unavailable");
  ctx.drawImage(bitmap, 0, 0);
  return await canvas.convertToBlob({ type: "image/png" });
}

// src/plugins/document-manager-plugin.ts
import { signal as signal17, computed as computed14 } from "@truespar/lector-utils";
var DEFAULT_STORAGE_KEY = "lector.recent-files";
var DEFAULT_MAX_RECENT = 20;
function sourceKey(source) {
  switch (source.type) {
    case "url":
      return `url:${source.url}`;
    case "file":
      return `file:${source.fileName}`;
    case "buffer":
      return `buffer:${source.name ?? "<unnamed>"}`;
  }
}
function deserialize(raw) {
  return {
    name: raw.name,
    source: raw.source,
    size: raw.size,
    lastOpenedAt: new Date(raw.lastOpenedAt)
  };
}
function serialize(entry) {
  return {
    name: entry.name,
    source: entry.source,
    size: entry.size,
    lastOpenedAt: entry.lastOpenedAt.toISOString()
  };
}
function createLocalStorageStore(storageKey, maxEntries) {
  function read() {
    try {
      if (typeof localStorage === "undefined") return [];
      const raw = localStorage.getItem(storageKey);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.map(deserialize);
    } catch {
      return [];
    }
  }
  function write(entries) {
    try {
      if (typeof localStorage === "undefined") return;
      localStorage.setItem(storageKey, JSON.stringify(entries.map(serialize)));
    } catch {
    }
  }
  return {
    list() {
      return read();
    },
    add(entry) {
      const existing = read();
      const key = sourceKey(entry.source);
      const filtered = existing.filter((e) => sourceKey(e.source) !== key);
      filtered.unshift(entry);
      write(filtered.slice(0, maxEntries));
    },
    remove(source) {
      const key = sourceKey(source);
      const remaining = read().filter((e) => sourceKey(e.source) !== key);
      write(remaining);
    },
    clear() {
      try {
        if (typeof localStorage === "undefined") return;
        localStorage.removeItem(storageKey);
      } catch {
      }
    }
  };
}
function urlBaseName(url) {
  try {
    const parsed = new URL(url);
    const last = parsed.pathname.split("/").filter(Boolean).pop();
    return last ?? parsed.hostname;
  } catch {
    const last = url.split(/[\\/]/).filter(Boolean).pop();
    return last ?? url;
  }
}
function isPdfFile(file) {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}
var documentManagerPlugin = definePlugin({
  id: "document-manager",
  provides: ["document-manager"],
  requires: ["document"],
  optional: [],
  setup(ctx) {
    const document2 = ctx.require("document");
    const engineAny = ctx.engine;
    const customStore = engineAny["recentFilesStore"];
    const maxRecent = engineAny["recentFilesMax"] ?? DEFAULT_MAX_RECENT;
    const storageKey = engineAny["recentFilesStorageKey"] ?? DEFAULT_STORAGE_KEY;
    const store = customStore ?? createLocalStorageStore(storageKey, maxRecent);
    const openDocs$ = signal17(/* @__PURE__ */ new Map());
    const recentFiles$ = signal17([]);
    void Promise.resolve(store.list()).then((entries) => {
      const sorted = [...entries].sort(
        (a, b) => b.lastOpenedAt.getTime() - a.lastOpenedAt.getTime()
      );
      recentFiles$.value = sorted;
    }).catch(() => {
    });
    function setInfo(info) {
      openDocs$.update((map) => {
        const next = new Map(map);
        next.set(info.id, info);
        return next;
      });
    }
    function trackUnknownOpen(handle) {
      const existing = openDocs$.peek();
      if (existing.has(handle.id)) return;
      const placeholder = {
        id: handle.id,
        handle,
        name: `Document ${existing.size + 1}`,
        size: 0,
        source: { type: "buffer" },
        openedAt: /* @__PURE__ */ new Date()
      };
      openDocs$.update((map) => {
        const next = new Map(map);
        next.set(handle.id, placeholder);
        return next;
      });
    }
    function untrack(docId) {
      openDocs$.update((map) => {
        if (!map.has(docId)) return map;
        const next = new Map(map);
        next.delete(docId);
        return next;
      });
    }
    async function recordRecent(info) {
      const entry = {
        name: info.name,
        source: info.source,
        size: info.size > 0 ? info.size : void 0,
        lastOpenedAt: info.openedAt
      };
      try {
        await store.add(entry);
        const next = await Promise.resolve(store.list());
        recentFiles$.value = [...next].sort(
          (a, b) => b.lastOpenedAt.getTime() - a.lastOpenedAt.getTime()
        );
        ctx.emit("document-manager:recent-changed", recentFiles$.peek());
      } catch {
      }
    }
    ctx.on("document:loaded", (...args) => {
      const handle = args[0];
      trackUnknownOpen(handle);
    });
    ctx.on("document:closed", (...args) => {
      const docId = args[0];
      untrack(docId);
    });
    const capability = {
      async openFromBuffer(buffer, options) {
        ctx.emit("document-manager:opening", { source: { type: "buffer", name: options?.name } });
        const handle = await document2.load(buffer, options?.password);
        const info = {
          id: handle.id,
          handle,
          name: options?.name ?? `Document ${openDocs$.peek().size}`,
          size: buffer.byteLength,
          source: { type: "buffer", name: options?.name },
          openedAt: /* @__PURE__ */ new Date()
        };
        setInfo(info);
        if (!options?.skipRecent) await recordRecent(info);
        ctx.emit("document-manager:opened", info);
        return info;
      },
      async openFromFile(file, options) {
        const source = { type: "file", fileName: file.name };
        ctx.emit("document-manager:opening", { source, name: file.name });
        const buffer = await file.arrayBuffer();
        const handle = await document2.load(buffer, options?.password);
        const info = {
          id: handle.id,
          handle,
          name: options?.name ?? file.name,
          size: file.size,
          source,
          openedAt: /* @__PURE__ */ new Date()
        };
        setInfo(info);
        if (!options?.skipRecent) await recordRecent(info);
        ctx.emit("document-manager:opened", info);
        return info;
      },
      async openFromUrl(url, options) {
        const urlStr = url instanceof URL ? url.href : url;
        const source = { type: "url", url: urlStr };
        const baseName = urlBaseName(urlStr);
        ctx.emit("document-manager:opening", { source, name: baseName });
        const response = await fetch(urlStr);
        if (!response.ok) {
          const err = new Error(`Failed to fetch PDF: ${response.status} ${response.statusText}`);
          ctx.emit("document-manager:open-failed", { source, error: err });
          throw err;
        }
        const buffer = await response.arrayBuffer();
        const handle = await document2.load(buffer, options?.password);
        const info = {
          id: handle.id,
          handle,
          name: options?.name ?? baseName,
          size: buffer.byteLength,
          source,
          openedAt: /* @__PURE__ */ new Date()
        };
        setInfo(info);
        if (!options?.skipRecent) await recordRecent(info);
        ctx.emit("document-manager:opened", info);
        return info;
      },
      openFileDialog(options) {
        return new Promise((resolve, reject) => {
          if (typeof window === "undefined") {
            reject(new Error("openFileDialog requires a DOM environment"));
            return;
          }
          const input = window.document.createElement("input");
          input.type = "file";
          input.accept = options?.accept ?? ".pdf,application/pdf";
          input.multiple = options?.multiple ?? false;
          input.style.display = "none";
          let settled = false;
          const finishResolve = (result) => {
            if (settled) return;
            settled = true;
            window.removeEventListener("focus", onWindowFocus);
            input.remove();
            resolve(result);
          };
          const finishReject = (err) => {
            if (settled) return;
            settled = true;
            window.removeEventListener("focus", onWindowFocus);
            input.remove();
            reject(err);
          };
          input.addEventListener("change", () => {
            const files = Array.from(input.files ?? []);
            if (files.length === 0) {
              finishResolve([]);
              return;
            }
            (async () => {
              const opened = [];
              for (const file of files) {
                if (!isPdfFile(file)) continue;
                try {
                  const info = await capability.openFromFile(file, options?.openOptions);
                  opened.push(info);
                } catch (err) {
                  ctx.emit("document-manager:open-failed", {
                    source: { type: "file", fileName: file.name },
                    error: err
                  });
                }
              }
              finishResolve(opened);
            })().catch(finishReject);
          });
          input.addEventListener("cancel", () => finishResolve([]));
          const onWindowFocus = () => {
            setTimeout(() => {
              if (!settled && (input.files?.length ?? 0) === 0) finishResolve([]);
            }, 350);
          };
          window.addEventListener("focus", onWindowFocus);
          window.document.body.appendChild(input);
          input.click();
        });
      },
      async close(docId) {
        await document2.close(docId);
      },
      async closeAll() {
        const ids = [...openDocs$.peek().keys()];
        for (const id of ids) {
          await document2.close(id).catch(() => {
          });
        }
      },
      getInfo(docId) {
        return openDocs$.peek().get(docId);
      },
      openDocuments: computed14(
        () => (
          // Iteration order of Map preserves insertion order, so this gives
          // open-order. Most consumers iterate; copying to an array is cheap.
          [...openDocs$.value.values()]
        )
      ),
      recentFiles: computed14(() => recentFiles$.value),
      async openRecentFile(entry, options) {
        switch (entry.source.type) {
          case "url":
            return capability.openFromUrl(entry.source.url, {
              ...options,
              name: options?.name ?? entry.name
            });
          case "file":
            throw new Error(
              "Cannot reopen a recent File entry \u2014 File objects cannot be persisted. Use openFileDialog() to let the user pick the file again."
            );
          case "buffer":
            throw new Error(
              "Cannot reopen a recent buffer entry \u2014 buffers are not persisted."
            );
        }
      },
      async removeRecentFile(source) {
        await store.remove(source);
        const next = await Promise.resolve(store.list());
        recentFiles$.value = [...next].sort(
          (a, b) => b.lastOpenedAt.getTime() - a.lastOpenedAt.getTime()
        );
        ctx.emit("document-manager:recent-changed", recentFiles$.peek());
      },
      async clearRecentFiles() {
        await store.clear();
        recentFiles$.value = [];
        ctx.emit("document-manager:recent-changed", []);
      },
      registerDropZone(el, options) {
        const multiple = options?.multiple ?? true;
        const hoverClass = options?.hoverClass ?? "lector-drop-zone--over";
        const promptText = options?.promptText;
        const showOverlay = options?.showOverlay ?? promptText !== void 0;
        let dragDepth = 0;
        let overlayEl = null;
        function showPromptOverlay() {
          if (!showOverlay || !promptText || overlayEl) return;
          if (typeof window === "undefined") return;
          const overlay = window.document.createElement("div");
          overlay.className = "lector-drop-zone__prompt";
          const text = window.document.createElement("div");
          text.className = "lector-drop-zone__prompt-text";
          text.textContent = promptText;
          overlay.appendChild(text);
          el.appendChild(overlay);
          overlayEl = overlay;
        }
        function hidePromptOverlay() {
          if (overlayEl) {
            overlayEl.remove();
            overlayEl = null;
          }
        }
        const onDragEnter = (e) => {
          if (!e.dataTransfer?.types.includes("Files")) return;
          e.preventDefault();
          dragDepth++;
          if (dragDepth === 1) {
            el.classList.add(hoverClass);
            showPromptOverlay();
          }
        };
        const onDragOver = (e) => {
          if (!e.dataTransfer?.types.includes("Files")) return;
          e.preventDefault();
          if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
        };
        const onDragLeave = (_e) => {
          dragDepth = Math.max(0, dragDepth - 1);
          if (dragDepth === 0) {
            el.classList.remove(hoverClass);
            hidePromptOverlay();
          }
        };
        const onDrop = (e) => {
          e.preventDefault();
          dragDepth = 0;
          el.classList.remove(hoverClass);
          hidePromptOverlay();
          const allFiles = Array.from(e.dataTransfer?.files ?? []);
          if (allFiles.length === 0) return;
          const accepted = allFiles.filter(isPdfFile);
          const rejected = allFiles.filter((f) => !isPdfFile(f));
          if (accepted.length === 0) {
            ctx.emit("document-manager:drop-rejected", {
              files: rejected,
              reason: "not-pdf",
              partial: false
            });
            return;
          }
          if (rejected.length > 0) {
            ctx.emit("document-manager:drop-rejected", {
              files: rejected,
              reason: "not-pdf",
              partial: true
            });
          }
          const toOpen = multiple ? accepted : accepted.slice(0, 1);
          (async () => {
            for (const f of toOpen) {
              try {
                await capability.openFromFile(f, options?.openOptions);
              } catch (err) {
                ctx.emit("document-manager:open-failed", {
                  source: { type: "file", fileName: f.name },
                  error: err
                });
              }
            }
          })().catch(() => {
          });
        };
        el.addEventListener("dragenter", onDragEnter);
        el.addEventListener("dragover", onDragOver);
        el.addEventListener("dragleave", onDragLeave);
        el.addEventListener("drop", onDrop);
        return () => {
          el.removeEventListener("dragenter", onDragEnter);
          el.removeEventListener("dragover", onDragOver);
          el.removeEventListener("dragleave", onDragLeave);
          el.removeEventListener("drop", onDrop);
          el.classList.remove(hoverClass);
          hidePromptOverlay();
        };
      }
    };
    return capability;
  },
  async dispose() {
  }
});

// src/plugins/merge-split-plugin.ts
var mergeSplitPlugin = definePlugin({
  id: "merge-split",
  provides: ["merge-split"],
  requires: ["document"],
  optional: [],
  setup(ctx) {
    ctx.require("document");
    const capability = {
      async mergeDocuments(docIds) {
        return ctx.engine.workerProxy.mergeDocuments(docIds);
      },
      async splitDocument(docId, ranges) {
        return ctx.engine.workerProxy.splitDocument(docId, ranges);
      },
      async extractPages(docId, pageIndices) {
        return ctx.engine.workerProxy.extractPages(docId, pageIndices);
      }
    };
    ctx.registerCommand({
      id: "document.merge",
      label: "Merge Documents",
      category: "Document",
      execute: () => {
        ctx.emit("merge-split:merge-requested");
      }
    });
    ctx.registerCommand({
      id: "document.split",
      label: "Split Document",
      category: "Document",
      execute: () => {
        ctx.emit("merge-split:split-requested");
      }
    });
    ctx.registerCommand({
      id: "document.extract-pages",
      label: "Extract Pages",
      category: "Document",
      execute: () => {
        ctx.emit("merge-split:extract-requested");
      }
    });
    return capability;
  }
});

// src/plugins/signature-validation-plugin.ts
import * as Comlink3 from "comlink";
var signatureValidationPlugin = definePlugin({
  id: "signature-validation",
  provides: ["signature-validation"],
  requires: ["document", "signature"],
  optional: [],
  setup(ctx) {
    ctx.require("document");
    const capability = {
      async validate(docId, sigIndex) {
        const placeholder = new ArrayBuffer(0);
        return ctx.engine.workerProxy.validateSignature(
          docId,
          sigIndex,
          Comlink3.transfer(placeholder, [placeholder])
        );
      },
      async validateAll(docId) {
        const placeholder = new ArrayBuffer(0);
        return ctx.engine.workerProxy.validateAllSignatures(
          docId,
          Comlink3.transfer(placeholder, [placeholder])
        );
      }
    };
    ctx.registerCommand({
      id: "signature.validate-all",
      label: "Validate All Signatures",
      category: "Signature",
      execute: () => {
        ctx.emit("signature:validation-requested");
      }
    });
    return capability;
  }
});

// src/plugins/signature-signing-plugin.ts
var signatureSigningPlugin = definePlugin({
  id: "signature-signing",
  provides: ["signature-signing"],
  requires: ["document"],
  optional: ["signature", "signature-validation"],
  setup(ctx) {
    ctx.require("document");
    const capability = {
      async sign(docId, options) {
        return ctx.engine.workerProxy.signDocument(docId, options);
      }
    };
    ctx.registerCommand({
      id: "signature.sign",
      label: "Sign Document",
      icon: "pen-tool",
      category: "Signature",
      execute: () => {
        ctx.emit("signature:sign-requested");
      }
    });
    return capability;
  }
});

// src/plugins/acroform-js-plugin.ts
import { signal as signal18 } from "@truespar/lector-utils";
var ACROFORM_BUILTINS = `
// \u2500\u2500 Event object \u2500\u2500
var event = { value: '', change: '', rc: true, willCommit: false, targetName: '', selStart: 0, selEnd: 0 };

// \u2500\u2500 Utility \u2500\u2500
var util = {
  printf: function(fmt) {
    var args = Array.prototype.slice.call(arguments, 1);
    var i = 0;
    return fmt.replace(/%[\\d.]*[dfs]/g, function(m) {
      var v = args[i++];
      if (m.indexOf('d') >= 0) return Math.floor(Number(v));
      if (m.indexOf('f') >= 0) {
        var prec = 6;
        var match = m.match(/\\.(\\d+)/);
        if (match) prec = parseInt(match[1]);
        return Number(v).toFixed(prec);
      }
      return String(v);
    });
  },
  printd: function(fmt, d) {
    if (typeof d === 'string') d = new Date(d);
    if (!(d instanceof Date)) return '';
    var map = {
      'yyyy': d.getFullYear(),
      'yy': String(d.getFullYear()).slice(-2),
      'mmmm': ['January','February','March','April','May','June','July','August','September','October','November','December'][d.getMonth()],
      'mmm': ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][d.getMonth()],
      'mm': ('0' + (d.getMonth()+1)).slice(-2),
      'm': d.getMonth()+1,
      'dd': ('0' + d.getDate()).slice(-2),
      'd': d.getDate(),
      'HH': ('0' + d.getHours()).slice(-2),
      'H': d.getHours(),
      'MM': ('0' + d.getMinutes()).slice(-2),
      'ss': ('0' + d.getSeconds()).slice(-2)
    };
    var result = fmt;
    for (var k in map) result = result.replace(k, map[k]);
    return result;
  }
};

// \u2500\u2500 App stub \u2500\u2500
var app = { alert: function(msg) {}, beep: function() {} };

// \u2500\u2500 Field bridge (calls native __lector_getFieldValue) \u2500\u2500
var doc = {
  getField: function(name) {
    return {
      value: __lector_getFieldValue(name),
      name: name,
      getArray: function() { return [this]; }
    };
  }
};

// \u2500\u2500 AFNumber_Format \u2500\u2500
function AFNumber_Format(nDec, sepStyle, negStyle, currStyle, strCurrency, bCurrencyPrepend) {
  var val = parseFloat(event.value);
  if (isNaN(val)) { event.value = ''; return; }
  var neg = val < 0;
  val = Math.abs(val);
  var str = val.toFixed(nDec);
  // Thousands separator
  if (sepStyle === 0 || sepStyle === 2) {
    var parts = str.split('.');
    parts[0] = parts[0].replace(/\\B(?=(\\d{3})+(?!\\d))/g, sepStyle === 0 ? ',' : '.');
    str = parts.join(sepStyle === 0 ? '.' : ',');
  } else if (sepStyle === 3) {
    str = str.replace('.', ',');
  }
  // Currency
  if (strCurrency) {
    str = bCurrencyPrepend ? strCurrency + str : str + strCurrency;
  }
  // Negative
  if (neg) {
    if (negStyle === 0 || negStyle === 1) str = '-' + str;
    if (negStyle === 2 || negStyle === 3) str = '(' + str + ')';
  }
  event.value = str;
}

function AFNumber_Keystroke(nDec, sepStyle, negStyle, currStyle, strCurrency, bCurrencyPrepend) {
  // Allow valid numeric input
  event.rc = true;
}

// \u2500\u2500 AFDate_Format \u2500\u2500
function AFDate_Format(cFormat) {
  if (!event.value) return;
  var d = new Date(event.value);
  if (isNaN(d.getTime())) return;
  var fmts = ['m/d', 'm/d/yy', 'mm/dd/yy', 'mm/yy', 'd-mmm', 'd-mmm-yy', 'dd-mmm-yy', 'yy-mm-dd', 'mmm-yy', 'mmmm-yy', 'mmm d, yyyy', 'mmmm d, yyyy', 'm/d/yy h:MM tt', 'm/d/yy HH:MM'];
  var fmt = (typeof cFormat === 'number') ? (fmts[cFormat] || 'mm/dd/yy') : cFormat;
  event.value = util.printd(fmt, d);
}

function AFDate_Keystroke(cFormat) { event.rc = true; }

// \u2500\u2500 AFSimple_Calculate \u2500\u2500
function AFSimple_Calculate(cFunction, cFields) {
  var fields = (typeof cFields === 'string') ? cFields.split(/[, ]+/) : cFields;
  var result = (cFunction === 'PRD') ? 1 : 0;
  var count = 0;
  for (var i = 0; i < fields.length; i++) {
    var f = doc.getField(fields[i]);
    if (!f) continue;
    var v = parseFloat(f.value);
    if (isNaN(v)) v = 0;
    if (cFunction === 'SUM' || cFunction === 'AVG') result += v;
    else if (cFunction === 'PRD') result *= v;
    else if (cFunction === 'MIN') result = (count === 0) ? v : Math.min(result, v);
    else if (cFunction === 'MAX') result = (count === 0) ? v : Math.max(result, v);
    count++;
  }
  if (cFunction === 'AVG' && count > 0) result /= count;
  event.value = String(result);
}

// \u2500\u2500 AFPercent_Format \u2500\u2500
function AFPercent_Format(nDec, sepStyle) {
  var val = parseFloat(event.value) * 100;
  if (isNaN(val)) { event.value = ''; return; }
  event.value = val.toFixed(nDec) + '%';
}
function AFPercent_Keystroke(nDec, sepStyle) { event.rc = true; }

// \u2500\u2500 AFSpecial_Format \u2500\u2500
function AFSpecial_Format(psf) {
  var v = event.value.replace(/\\D/g, '');
  if (psf === 0) event.value = v.slice(0,5); // ZIP
  else if (psf === 1) event.value = v.slice(0,5) + '-' + v.slice(5,9); // ZIP+4
  else if (psf === 2) { // Phone
    if (v.length >= 10) event.value = '(' + v.slice(0,3) + ') ' + v.slice(3,6) + '-' + v.slice(6,10);
    else if (v.length >= 7) event.value = v.slice(0,3) + '-' + v.slice(3,7);
  }
  else if (psf === 3) event.value = v.slice(0,3) + '-' + v.slice(3,5) + '-' + v.slice(5,9); // SSN
}
function AFSpecial_Keystroke(psf) { event.rc = true; }

// \u2500\u2500 AFRange_Validate \u2500\u2500
function AFRange_Validate(bGreaterThan, nGreaterThan, bLessThan, nLessThan) {
  var v = parseFloat(event.value);
  if (bGreaterThan && v < nGreaterThan) event.rc = false;
  if (bLessThan && v > nLessThan) event.rc = false;
}

// \u2500\u2500 AFTime_Format \u2500\u2500
function AFTime_Format(ptf) {
  if (!event.value) return;
  var d = new Date(event.value);
  if (isNaN(d.getTime())) return;
  var fmts = ['HH:MM', 'h:MM tt', 'HH:MM:ss', 'h:MM:ss tt'];
  var fmt = (typeof ptf === 'number') ? (fmts[ptf] || 'HH:MM') : ptf;
  event.value = util.printd(fmt, d);
}
function AFTime_Keystroke(ptf) { event.rc = true; }
`;
var acroformJSPlugin = definePlugin({
  id: "acroform-js",
  provides: ["acroform-js"],
  requires: ["document", "form"],
  optional: [],
  setup(ctx) {
    ctx.require("document");
    ctx.require("form");
    const enabled$ = signal18(true);
    let runtimeId = null;
    async function ensureRuntime() {
      if (runtimeId === null) {
        runtimeId = await ctx.engine.workerProxy.createJSRuntime();
        await ctx.engine.workerProxy.evalScript(runtimeId, ACROFORM_BUILTINS);
      }
      return runtimeId;
    }
    const capability = {
      enabled: enabled$,
      setEnabled(enabled) {
        enabled$.value = enabled;
      },
      async loadDocumentScripts(docId) {
        if (!enabled$.peek()) return;
        const id = await ensureRuntime();
        const actions = await ctx.engine.workerProxy.getJavaScriptActions(docId);
        for (const action of actions) {
          if (action.script) {
            await ctx.engine.workerProxy.evalScript(id, action.script);
          }
        }
      },
      async executeFieldScript(script, fieldName, currentValue) {
        if (!enabled$.peek()) return void 0;
        const id = await ensureRuntime();
        const setup = `event.value = ${JSON.stringify(currentValue)}; event.targetName = ${JSON.stringify(fieldName)}; event.rc = true;`;
        await ctx.engine.workerProxy.evalScript(id, setup);
        const ok = await ctx.engine.workerProxy.evalScript(id, script);
        if (!ok) return void 0;
        const result = await ctx.engine.workerProxy.getJSGlobal(id, "__lector_getEventValue");
        return result ?? void 0;
      },
      async getActions(docId) {
        return ctx.engine.workerProxy.getJavaScriptActions(docId);
      },
      async dispose() {
        if (runtimeId !== null) {
          await ctx.engine.workerProxy.destroyJSRuntime(runtimeId);
          runtimeId = null;
        }
      }
    };
    return capability;
  }
});

// src/plugins/linearization-plugin.ts
import { signal as signal19 } from "@truespar/lector-utils";
var INITIAL_CHUNK_SIZE = 65536;
var linearizationPlugin = definePlugin({
  id: "linearization",
  provides: ["linearization"],
  requires: ["document"],
  optional: [],
  setup(ctx) {
    const docCap = ctx.require("document");
    const progress$ = signal19(null);
    const capability = {
      progress: progress$,
      async loadProgressive(url, options) {
        const fetchOpts = {};
        if (options?.signal) fetchOpts.signal = options.signal;
        progress$.value = { phase: "probing", bytesReceived: 0, totalBytes: 0, pagesAvailable: 0, totalPages: 0 };
        const headRes = await fetch(url, { ...fetchOpts, method: "HEAD" });
        const contentLength = parseInt(headRes.headers.get("Content-Length") ?? "0", 10);
        const acceptsRanges = headRes.headers.get("Accept-Ranges") === "bytes";
        if (!acceptsRanges || contentLength <= 0) {
          progress$.value = { phase: "fallback", bytesReceived: 0, totalBytes: contentLength, pagesAvailable: 0, totalPages: 0 };
          const res = await fetch(url, fetchOpts);
          const buf = await res.arrayBuffer();
          progress$.value = { phase: "complete", bytesReceived: buf.byteLength, totalBytes: buf.byteLength, pagesAvailable: 0, totalPages: 0 };
          const handle = await docCap.load(buf, options?.password);
          return handle.id;
        }
        progress$.value = { phase: "header", bytesReceived: 0, totalBytes: contentLength, pagesAvailable: 0, totalPages: 0 };
        const initialRes = await fetch(url, {
          ...fetchOpts,
          headers: { Range: `bytes=0-${INITIAL_CHUNK_SIZE - 1}` }
        });
        const initialChunk = new Uint8Array(await initialRes.arrayBuffer());
        let bytesReceived = initialChunk.length;
        progress$.value = { phase: "header", bytesReceived, totalBytes: contentLength, pagesAvailable: 0, totalPages: 0 };
        const ctxId = await ctx.engine.workerProxy.createLinearContext(contentLength, initialChunk.buffer);
        const linResult = await ctx.engine.workerProxy.isLinearized(ctxId);
        if (linResult !== 1) {
          await ctx.engine.workerProxy.destroyLinearContext(ctxId);
          progress$.value = { phase: "fallback", bytesReceived, totalBytes: contentLength, pagesAvailable: 0, totalPages: 0 };
          const res = await fetch(url, fetchOpts);
          const buf = await res.arrayBuffer();
          progress$.value = { phase: "complete", bytesReceived: buf.byteLength, totalBytes: buf.byteLength, pagesAvailable: 0, totalPages: 0 };
          const handle2 = await docCap.load(buf, options?.password);
          return handle2.id;
        }
        progress$.value = { phase: "first-page", bytesReceived, totalBytes: contentLength, pagesAvailable: 0, totalPages: 0 };
        const fetchRange = async (offset, length) => {
          const end = Math.min(offset + length - 1, contentLength - 1);
          const res = await fetch(url, {
            ...fetchOpts,
            headers: { Range: `bytes=${offset}-${end}` }
          });
          return new Uint8Array(await res.arrayBuffer());
        };
        for (let attempt = 0; attempt < 100; attempt++) {
          const docStatus = await ctx.engine.workerProxy.isDocAvail(ctxId);
          if (docStatus.available) break;
          for (const hint of docStatus.hints) {
            const chunk = await fetchRange(hint.offset, hint.length);
            await ctx.engine.workerProxy.feedLinearData(ctxId, hint.offset, chunk.buffer);
            bytesReceived += chunk.length;
          }
          progress$.value = { phase: "first-page", bytesReceived, totalBytes: contentLength, pagesAvailable: 0, totalPages: 0 };
        }
        const docId = await ctx.engine.workerProxy.getLinearDocument(ctxId, options?.password);
        const firstPage = await ctx.engine.workerProxy.getLinearFirstPage(ctxId);
        const totalPages = await ctx.engine.workerProxy.getPageCount(docId);
        ctx.emit("document:loaded", docId);
        progress$.value = { phase: "loading", bytesReceived, totalBytes: contentLength, pagesAvailable: 1, totalPages };
        (async () => {
          let pagesAvailable = 1;
          for (let pageIdx = 0; pageIdx < totalPages; pageIdx++) {
            if (pageIdx === firstPage) continue;
            for (let attempt = 0; attempt < 50; attempt++) {
              const pageStatus = await ctx.engine.workerProxy.isPageAvail(ctxId, pageIdx);
              if (pageStatus.available) break;
              for (const hint of pageStatus.hints) {
                const chunk = await fetchRange(hint.offset, hint.length);
                await ctx.engine.workerProxy.feedLinearData(ctxId, hint.offset, chunk.buffer);
                bytesReceived += chunk.length;
              }
            }
            pagesAvailable++;
            ctx.emit("linearization:page-available", docId, pageIdx);
            progress$.value = { phase: "loading", bytesReceived, totalBytes: contentLength, pagesAvailable, totalPages };
          }
          progress$.value = { phase: "complete", bytesReceived, totalBytes: contentLength, pagesAvailable: totalPages, totalPages };
          await ctx.engine.workerProxy.destroyLinearContext(ctxId);
        })();
        return docId;
      }
    };
    return capability;
  }
});

// src/plugins/layer-plugin.ts
import { signal as signal20, computed as computed15 } from "@truespar/lector-utils";
var layerPlugin = definePlugin({
  id: "layer",
  provides: ["layer"],
  requires: ["document"],
  optional: [],
  setup(ctx) {
    ctx.require("document");
    const layers$ = signal20([]);
    ctx.on("document:opened", async (...args) => {
      const docId = args[0];
      try {
        const layerList = await ctx.engine.workerProxy.getLayers(docId);
        layers$.value = layerList;
      } catch {
        layers$.value = [];
      }
    });
    ctx.on("document:closed", () => {
      layers$.value = [];
    });
    return {
      layers: computed15(() => layers$.value),
      hasLayers: computed15(() => layers$.value.length > 0),
      async loadLayers(docId) {
        try {
          const layerList = await ctx.engine.workerProxy.getLayers(docId);
          layers$.value = layerList;
        } catch {
          layers$.value = [];
        }
      },
      async setVisible(docId, layerIndex, visible) {
        await ctx.engine.workerProxy.setLayerVisible(docId, layerIndex, visible);
        const updated = layers$.peek().map(
          (l) => l.index === layerIndex ? { ...l, visible } : l
        );
        layers$.value = updated;
        ctx.emit("layer:visibility-changed", docId, layerIndex, visible);
      },
      async setAllVisible(docId, visible) {
        const current = layers$.peek();
        for (const layer of current) {
          await ctx.engine.workerProxy.setLayerVisible(docId, layer.index, visible);
        }
        layers$.value = current.map((l) => ({ ...l, visible }));
        ctx.emit("layer:visibility-changed", docId, -1, visible);
      }
    };
  }
});

// src/plugins/performance-plugin.ts
import { signal as signal21, computed as computed16 } from "@truespar/lector-utils";
var MAX_SAMPLES = 2e3;
function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const idx = Math.ceil(p / 100 * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}
function computeStats(durations) {
  if (durations.length === 0) {
    return { count: 0, meanMs: 0, medianMs: 0, p95Ms: 0, p99Ms: 0, minMs: 0, maxMs: 0 };
  }
  const sorted = [...durations].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    count: sorted.length,
    meanMs: Math.round(sum / sorted.length * 100) / 100,
    medianMs: percentile(sorted, 50),
    p95Ms: percentile(sorted, 95),
    p99Ms: percentile(sorted, 99),
    minMs: sorted[0],
    maxMs: sorted[sorted.length - 1]
  };
}
var performancePlugin = definePlugin({
  id: "performance",
  provides: ["performance"],
  requires: ["document", "render"],
  optional: [],
  setup(ctx) {
    const doc = ctx.require("document");
    ctx.require("render");
    const samples = [];
    let sampleCursor = 0;
    function addSample(label, durationMs) {
      const sample = {
        label,
        durationMs: Math.round(durationMs * 100) / 100,
        timestamp: Date.now()
      };
      if (samples.length < MAX_SAMPLES) {
        samples.push(sample);
      } else {
        samples[sampleCursor % MAX_SAMPLES] = sample;
      }
      sampleCursor++;
    }
    const initTimeMs$ = signal21(null);
    const totalRenders$ = signal21(0);
    const totalDocLoads$ = signal21(0);
    const renderDurations = [];
    const docLoadDurations = [];
    const meanRenderMs$ = signal21(0);
    const meanDocLoadMs$ = signal21(0);
    const memory$ = signal21({
      totalBytes: 0,
      tileCacheBytes: 0,
      tileCacheCount: 0,
      renderedPageCount: 0,
      wasmHeapBytes: null
    });
    const ROLLING_WINDOW = 200;
    function pushRenderDuration(ms) {
      renderDurations.push(ms);
      if (renderDurations.length > ROLLING_WINDOW) renderDurations.shift();
      totalRenders$.value = totalRenders$.peek() + 1;
      meanRenderMs$.value = Math.round(renderDurations.reduce((a, b) => a + b, 0) / renderDurations.length * 100) / 100;
    }
    function pushDocLoadDuration(ms) {
      docLoadDurations.push(ms);
      if (docLoadDurations.length > ROLLING_WINDOW) docLoadDurations.shift();
      totalDocLoads$.value = totalDocLoads$.peek() + 1;
      meanDocLoadMs$.value = Math.round(docLoadDurations.reduce((a, b) => a + b, 0) / docLoadDurations.length * 100) / 100;
    }
    const pluginSetupTime = performance.now();
    ctx.on("engine:ready", () => {
      const elapsed = performance.now() - pluginSetupTime;
      initTimeMs$.value = Math.round(elapsed * 100) / 100;
      addSample("engine:init", elapsed);
    });
    const originalLoad = doc.load.bind(doc);
    const instrumentedLoad = async (source, password) => {
      const t0 = performance.now();
      try {
        const handle = await originalLoad(source, password);
        const elapsed = performance.now() - t0;
        const label = `document:load:${handle.id}`;
        addSample(label, elapsed);
        pushDocLoadDuration(elapsed);
        return handle;
      } catch (err) {
        const elapsed = performance.now() - t0;
        addSample("document:load:error", elapsed);
        throw err;
      }
    };
    doc.load = instrumentedLoad;
    const originalRenderPage = ctx.engine.renderPage.bind(ctx.engine);
    const instrumentedRenderPage = async (docId, pageIndex, widthPx, heightPx, options) => {
      const t0 = performance.now();
      try {
        const bitmap = await originalRenderPage(docId, pageIndex, widthPx, heightPx, options);
        const elapsed = performance.now() - t0;
        addSample(`render:${docId}:page-${pageIndex}`, elapsed);
        pushRenderDuration(elapsed);
        return bitmap;
      } catch (err) {
        const elapsed = performance.now() - t0;
        addSample(`render:error:${docId}:page-${pageIndex}`, elapsed);
        throw err;
      }
    };
    ctx.engine.renderPage = instrumentedRenderPage;
    if (typeof ctx.engine.renderPageTile === "function") {
      const originalRenderTile = ctx.engine.renderPageTile.bind(ctx.engine);
      const instrumentedTile = async (...args) => {
        const t0 = performance.now();
        try {
          const bitmap = await originalRenderTile(...args);
          const elapsed = performance.now() - t0;
          addSample(`tile:${args[0]}:page-${args[1]}`, elapsed);
          pushRenderDuration(elapsed);
          return bitmap;
        } catch (err) {
          const elapsed = performance.now() - t0;
          addSample(`tile:error:${args[0]}:page-${args[1]}`, elapsed);
          throw err;
        }
      };
      ctx.engine.renderPageTile = instrumentedTile;
    }
    let memoryInterval = null;
    function estimateMemory() {
      const canvases = typeof document !== "undefined" ? document.querySelectorAll(".lector-page__canvas") : [];
      let canvasBytes = 0;
      canvases.forEach((c) => {
        const canvas = c;
        canvasBytes += canvas.width * canvas.height * 4;
      });
      const tileCanvases = typeof document !== "undefined" ? document.querySelectorAll(".lector-tile-canvas") : [];
      let tileBytes = 0;
      tileCanvases.forEach((c) => {
        const canvas = c;
        tileBytes += canvas.width * canvas.height * 4;
      });
      const perfMemory = performance.memory;
      return {
        totalBytes: canvasBytes + tileBytes,
        tileCacheBytes: tileBytes,
        tileCacheCount: tileCanvases.length,
        renderedPageCount: canvases.length,
        wasmHeapBytes: perfMemory?.usedJSHeapSize ?? null
      };
    }
    if (typeof document !== "undefined") {
      memoryInterval = setInterval(() => {
        memory$.value = estimateMemory();
      }, 2e3);
    }
    ctx.on("engine:destroy", () => {
      if (memoryInterval) clearInterval(memoryInterval);
    });
    const capability = {
      initTimeMs: computed16(() => initTimeMs$.value),
      totalRenders: computed16(() => totalRenders$.value),
      meanRenderMs: computed16(() => meanRenderMs$.value),
      totalDocumentsLoaded: computed16(() => totalDocLoads$.value),
      meanDocLoadMs: computed16(() => meanDocLoadMs$.value),
      memory: computed16(() => memory$.value),
      mark(label, durationMs) {
        addSample(label, durationMs);
      },
      startTimer(label) {
        const t0 = performance.now();
        return () => {
          const elapsed = performance.now() - t0;
          addSample(label, elapsed);
        };
      },
      getReport() {
        const allSamples = samples.slice(0, Math.min(samples.length, MAX_SAMPLES));
        const renderSamples = allSamples.filter((s) => s.label.startsWith("render:") || s.label.startsWith("tile:")).map((s) => s.durationMs);
        const docSamples = allSamples.filter((s) => s.label.startsWith("document:load:")).map((s) => s.durationMs);
        return {
          initTimeMs: initTimeMs$.peek(),
          documentLoad: computeStats(docSamples),
          pageRender: computeStats(renderSamples),
          memory: estimateMemory(),
          samples: Object.freeze([...allSamples]),
          generatedAt: Date.now()
        };
      },
      reset() {
        samples.length = 0;
        sampleCursor = 0;
        renderDurations.length = 0;
        docLoadDurations.length = 0;
        totalRenders$.value = 0;
        totalDocLoads$.value = 0;
        meanRenderMs$.value = 0;
        meanDocLoadMs$.value = 0;
      }
    };
    ctx.registerCommand({
      id: "performance.log-report",
      label: "Log Performance Report",
      shortcut: "Ctrl+Shift+P",
      category: "Developer",
      execute: () => {
        const report = capability.getReport();
        console.group("Lector Performance Report");
        console.log("Init time:", report.initTimeMs ? `${report.initTimeMs}ms` : "n/a");
        console.log("Document loads:", report.documentLoad);
        console.log("Page renders:", report.pageRender);
        console.log("Memory:", report.memory);
        console.log("Total samples:", report.samples.length);
        console.groupEnd();
      }
    });
    return capability;
  }
});

// src/ui/tile-manager.ts
var DEFAULT_TILE_SIZE = 512;
var DEFAULT_OVERLAP_PX = 1;
var DEFAULT_MAX_CACHED_TILES = 256;
var DEFAULT_TILE_THRESHOLD = 4096 * 4096;
function tileKey(docId, pageIndex, scale, col, row) {
  return `${docId}:${pageIndex}:${scale}:${col}:${row}`;
}
function rectsIntersect(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}
var TileManager = class {
  // Resolved configuration
  #tileSize;
  #overlapPx;
  #maxCachedTiles;
  #tileThreshold;
  /**
   * Optional callback invoked whenever an async tile render completes
   * successfully. The consumer (viewer) uses this to trigger a re-paint
   * so the newly-ready tile is drawn onto the canvas.
   */
  onTileReady = null;
  /** All cache entries keyed by their canonical cache key. */
  #cache = /* @__PURE__ */ new Map();
  /** Monotonically increasing counter used for LRU access ordering. */
  #accessCounter = 0;
  /**
   * Tracks the last scale value seen for each page so we can detect zoom
   * changes and invalidate stale tiles.
   *
   * Key: `${docId}:${pageIndex}`
   */
  #pageScale = /* @__PURE__ */ new Map();
  /**
   * Set of cache keys that currently have an in-flight render promise.
   * Prevents duplicate render requests for the same tile.
   */
  #inflight = /* @__PURE__ */ new Set();
  // -----------------------------------------------------------------------
  // Constructor
  // -----------------------------------------------------------------------
  constructor(config) {
    this.#tileSize = config?.tileSize ?? DEFAULT_TILE_SIZE;
    this.#overlapPx = config?.overlapPx ?? DEFAULT_OVERLAP_PX;
    this.#maxCachedTiles = config?.maxCachedTiles ?? DEFAULT_MAX_CACHED_TILES;
    this.#tileThreshold = config?.tileThreshold ?? DEFAULT_TILE_THRESHOLD;
  }
  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------
  /**
   * Determine whether a page at the given pixel dimensions should use tile
   * mode. Returns `false` when the total pixel area is below
   * {@link TileConfig.tileThreshold} (full-page rendering is cheaper).
   *
   * @param fullW - Full page width in pixels at the current scale and DPR.
   * @param fullH - Full page height in pixels.
   */
  shouldTile(fullW, fullH) {
    return fullW * fullH > this.#tileThreshold;
  }
  /**
   * Compute visible tiles for a page and kick off rendering for any that are
   * not yet cached.
   *
   * Call this on every scroll or zoom event. The method is synchronous and
   * returns immediately with the current state of each visible tile. Render
   * callbacks are fired asynchronously; call this method again on the next
   * frame to pick up newly-ready tiles.
   *
   * @param docId      - Document identifier.
   * @param pageIndex  - Zero-based page index.
   * @param fullW      - Full page width in pixels (at current scale x DPR).
   * @param fullH      - Full page height in pixels.
   * @param viewportRect - Visible area within the page, in page-pixel coords.
   * @param scale      - Current zoom scale. Tiles are invalidated when this
   *                     changes because different scales produce different
   *                     bitmaps.
   * @param renderFn   - Async callback the consumer provides to render a tile
   *                     region via the worker pipeline.
   * @returns An array of {@link TileDescriptor}s for every tile that
   *          intersects the viewport.
   */
  updateVisibleTiles(docId, pageIndex, fullW, fullH, viewportRect, scale, renderFn) {
    const pageKey = `${docId}:${pageIndex}`;
    const prevScale = this.#pageScale.get(pageKey);
    if (prevScale !== void 0 && prevScale !== scale) {
      this.#invalidatePageAtScale(docId, pageIndex, prevScale);
    }
    this.#pageScale.set(pageKey, scale);
    const step = this.#tileSize - this.#overlapPx;
    const cols = Math.ceil(fullW / step);
    const rows = Math.ceil(fullH / step);
    const result = [];
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const x = col * step;
        const y = row * step;
        const w = Math.min(this.#tileSize, fullW - x);
        const h = Math.min(this.#tileSize, fullH - y);
        if (w <= 0 || h <= 0) {
          continue;
        }
        const tileRect = { x, y, w, h };
        if (!rectsIntersect(tileRect, viewportRect)) {
          continue;
        }
        const key = tileKey(docId, pageIndex, scale, col, row);
        let entry = this.#cache.get(key);
        if (entry === void 0) {
          entry = {
            key,
            col,
            row,
            x,
            y,
            w,
            h,
            status: "queued",
            bitmap: void 0,
            lastAccess: this.#accessCounter++
          };
          this.#cache.set(key, entry);
          this.#requestRender(entry, docId, pageIndex, fullW, fullH, scale, renderFn);
        } else {
          entry.lastAccess = this.#accessCounter++;
        }
        result.push(this.#toDescriptor(entry));
      }
    }
    this.#evict();
    return result;
  }
  /**
   * Return all tiles for a page at a specific scale whose bitmaps are ready
   * for drawing. Unlike {@link updateVisibleTiles} this does not issue any
   * new render requests — it only returns what is already cached.
   *
   * @param pageIndex - Zero-based page index.
   * @param scale     - Zoom scale to match.
   */
  getReadyTiles(pageIndex, scale) {
    const result = [];
    for (const entry of this.#cache.values()) {
      if (entry.status === "ready" && this.#matchesPageScale(entry.key, pageIndex, scale)) {
        entry.lastAccess = this.#accessCounter++;
        result.push(this.#toDescriptor(entry));
      }
    }
    return result;
  }
  /**
   * Remove all cached tiles for a specific page (all scales). Use this when
   * a page is scrolled far out of view and removed from the DOM.
   *
   * @param pageIndex - Zero-based page index.
   */
  clearPage(pageIndex) {
    const toDelete = [];
    for (const [key, entry] of this.#cache) {
      if (this.#matchesPage(key, pageIndex)) {
        entry.bitmap?.close();
        toDelete.push(key);
      }
    }
    for (const key of toDelete) {
      this.#cache.delete(key);
      this.#inflight.delete(key);
    }
    for (const [k] of this.#pageScale) {
      if (k.endsWith(`:${pageIndex}`)) {
        this.#pageScale.delete(k);
      }
    }
  }
  /**
   * Clear the entire tile cache. Call this when the document changes or the
   * engine is disposed.
   */
  clearAll() {
    for (const entry of this.#cache.values()) {
      entry.bitmap?.close();
    }
    this.#cache.clear();
    this.#inflight.clear();
    this.#pageScale.clear();
    this.#accessCounter = 0;
  }
  /**
   * Dispose the tile manager, closing all cached {@link ImageBitmap}s and
   * releasing memory. The instance should not be used after calling this.
   */
  destroy() {
    this.clearAll();
  }
  // -----------------------------------------------------------------------
  // Private helpers
  // -----------------------------------------------------------------------
  /** Fire the consumer's render callback and update the cache entry on completion. */
  #requestRender(entry, docId, pageIndex, fullW, fullH, scale, renderFn) {
    const { key } = entry;
    if (this.#inflight.has(key)) {
      return;
    }
    this.#inflight.add(key);
    entry.status = "rendering";
    const req = {
      docId,
      pageIndex,
      tileX: entry.x,
      tileY: entry.y,
      tileW: entry.w,
      tileH: entry.h,
      fullW,
      fullH,
      scale
    };
    renderFn(req).then(
      (bitmap) => {
        this.#inflight.delete(key);
        const current = this.#cache.get(key);
        if (current === void 0) {
          bitmap.close();
          return;
        }
        current.status = "ready";
        current.bitmap = bitmap;
        current.lastAccess = this.#accessCounter++;
        this.onTileReady?.();
      },
      (_err) => {
        this.#inflight.delete(key);
        this.#cache.delete(key);
      }
    );
  }
  /**
   * Invalidate (remove and close) all cached tiles for a given page at a
   * specific scale. Called when the zoom level changes.
   */
  #invalidatePageAtScale(docId, pageIndex, scale) {
    const prefix = `${docId}:${pageIndex}:${scale}:`;
    const toDelete = [];
    for (const [key, entry] of this.#cache) {
      if (key.startsWith(prefix)) {
        entry.bitmap?.close();
        toDelete.push(key);
      }
    }
    for (const key of toDelete) {
      this.#cache.delete(key);
      this.#inflight.delete(key);
    }
  }
  /** Evict least-recently-used entries when the cache exceeds its budget. */
  #evict() {
    if (this.#cache.size <= this.#maxCachedTiles) {
      return;
    }
    const entries = [...this.#cache.values()].sort((a, b) => a.lastAccess - b.lastAccess);
    const excess = this.#cache.size - this.#maxCachedTiles;
    for (let i = 0; i < excess; i++) {
      const entry = entries[i];
      entry.bitmap?.close();
      this.#cache.delete(entry.key);
      this.#inflight.delete(entry.key);
    }
  }
  /** Create a read-only {@link TileDescriptor} from an internal cache entry. */
  #toDescriptor(entry) {
    return {
      key: entry.key,
      col: entry.col,
      row: entry.row,
      x: entry.x,
      y: entry.y,
      w: entry.w,
      h: entry.h,
      status: entry.status,
      bitmap: entry.bitmap
    };
  }
  /**
   * Check whether a cache key belongs to a given page index (any doc, any
   * scale). The key format is `docId:pageIndex:scale:col:row`. We split on
   * `:` and compare the second segment.
   */
  #matchesPage(key, pageIndex) {
    const secondColon = key.indexOf(":", key.indexOf(":") + 1);
    const firstColon = key.indexOf(":");
    const segment = key.slice(firstColon + 1, secondColon);
    return segment === String(pageIndex);
  }
  /**
   * Check whether a cache key matches a given page index AND scale.
   * Key format: `docId:pageIndex:scale:col:row`.
   */
  #matchesPageScale(key, pageIndex, scale) {
    const firstColon = key.indexOf(":");
    const secondColon = key.indexOf(":", firstColon + 1);
    const thirdColon = key.indexOf(":", secondColon + 1);
    const pageSegment = key.slice(firstColon + 1, secondColon);
    const scaleSegment = key.slice(secondColon + 1, thirdColon);
    return pageSegment === String(pageIndex) && scaleSegment === String(scale);
  }
};

// src/plugins/ui-plugin.ts
import { computed as computed19 } from "@truespar/lector-utils";

// src/ui/ui-manager.ts
import { signal as signal23, computed as computed18 } from "@truespar/lector-utils";

// src/ui/responsive.ts
import { signal as signal22, computed as computed17 } from "@truespar/lector-utils";
var DEFAULT_BREAKPOINTS = {
  compact: 640,
  wide: 1024
};
var BreakpointObserver = class {
  breakpoint;
  #config;
  #width = signal22(0);
  #observer = null;
  #container = null;
  constructor(config = DEFAULT_BREAKPOINTS) {
    this.#config = config;
    this.breakpoint = computed17(() => {
      const w = this.#width.value;
      if (w < this.#config.compact) return "compact";
      if (w >= this.#config.wide) return "wide";
      return "medium";
    });
  }
  /** Start observing a container element's width. */
  observe(container) {
    this.disconnect();
    this.#container = container;
    this.#width.value = container.clientWidth;
    this.#observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        this.#width.value = entry.contentRect.width;
      }
    });
    this.#observer.observe(container);
  }
  /** Stop observing. */
  disconnect() {
    if (this.#observer !== null) {
      this.#observer.disconnect();
      this.#observer = null;
    }
    this.#container = null;
  }
  /** Update breakpoint thresholds at runtime. */
  setConfig(config) {
    this.#config = config;
    if (this.#container !== null) {
      this.#width.value = this.#container.clientWidth;
    }
  }
  [Symbol.dispose]() {
    this.disconnect();
  }
};
function isCategoryVisible(category, tier) {
  if (category === void 0 || category === "essential") return true;
  if (category === "standard") return tier !== "compact";
  return tier === "wide";
}

// src/ui/ui-manager.ts
var UIManager = class {
  state;
  #schema;
  #commands;
  #breakpointObserver;
  #theme = signal23("light");
  #sidebarCollapsed;
  #activePanel;
  #cleanups = [];
  #systemDarkQuery = null;
  #translator = null;
  constructor(schema, commands, translator) {
    this.#schema = schema;
    this.#commands = commands;
    this.#translator = translator ?? null;
    this.#breakpointObserver = new BreakpointObserver(
      schema.breakpoints ?? DEFAULT_BREAKPOINTS
    );
    this.#sidebarCollapsed = signal23(schema.sidebar?.collapsed ?? false);
    this.#activePanel = signal23(
      schema.sidebar?.panels[0]?.id ?? null
    );
    const resolvedItems = this.#resolveToolbarItems(schema.toolbar?.items ?? []);
    const visibleToolbarItems = computed18(() => {
      const tier = this.#breakpointObserver.breakpoint.value;
      return resolvedItems.filter((item) => {
        if (item.schema.visible === false) return false;
        return isCategoryVisible(item.schema.category, tier);
      });
    });
    const toolbarVisible = computed18(() => this.#schema.toolbar?.visible !== false);
    const sidebarVisible = computed18(() => {
      const panels = this.#schema.sidebar?.panels ?? [];
      return panels.length > 0;
    });
    const statusBarVisible = computed18(() => this.#schema.statusBar?.visible !== false);
    const self = this;
    this.state = {
      toolbar: {
        visible: toolbarVisible,
        items: visibleToolbarItems
      },
      sidebar: {
        visible: sidebarVisible,
        collapsed: computed18(() => this.#sidebarCollapsed.value),
        activePanel: computed18(() => this.#activePanel.value),
        // Getter, not snapshot: consumers rebuild after updateSchema (the
        // viewer calls #buildSidebar), so a plain live read suffices.
        get panels() {
          return self.#schema.sidebar?.panels ?? [];
        }
      },
      statusBar: {
        visible: statusBarVisible,
        get items() {
          return self.#schema.statusBar?.items ?? [];
        }
      },
      breakpoint: this.#breakpointObserver.breakpoint,
      theme: computed18(() => this.#theme.value)
    };
  }
  /** Start observing a container for responsive breakpoints. */
  observe(container) {
    this.#breakpointObserver.observe(container);
  }
  /** Stop observing. */
  disconnect() {
    this.#breakpointObserver.disconnect();
  }
  // ── Schema ──
  /** Replace the UI schema at runtime. */
  updateSchema(schema) {
    this.#schema = schema;
    if (schema.breakpoints) {
      this.#breakpointObserver.setConfig(schema.breakpoints);
    }
    if (schema.sidebar?.panels) {
      const panels = schema.sidebar.panels;
      const current = this.#activePanel.peek();
      if (current !== null && !panels.some((p) => p.id === current)) {
        this.#activePanel.value = panels[0]?.id ?? null;
      }
    }
  }
  /** Get the current schema. */
  get schema() {
    return this.#schema;
  }
  // ── Theme ──
  /** Set the theme mode. */
  setTheme(mode) {
    this.#theme.value = mode;
  }
  /** Get resolved (effective) theme considering 'system'. */
  get effectiveTheme() {
    return computed18(() => {
      const mode = this.#theme.value;
      if (mode !== "system") return mode;
      return this.#isSystemDark() ? "dark" : "light";
    });
  }
  #isSystemDark() {
    if (typeof window === "undefined") return false;
    if (this.#systemDarkQuery === null) {
      this.#systemDarkQuery = window.matchMedia("(prefers-color-scheme: dark)");
    }
    return this.#systemDarkQuery.matches;
  }
  // ── Sidebar ──
  /** Toggle sidebar collapsed state. */
  toggleSidebar() {
    this.#sidebarCollapsed.update((v) => !v);
  }
  /** Set sidebar collapsed state explicitly. */
  setSidebarCollapsed(collapsed) {
    this.#sidebarCollapsed.value = collapsed;
  }
  /** Set the active sidebar panel by ID. */
  setActivePanel(panelId) {
    this.#activePanel.value = panelId;
    if (panelId !== null) {
      this.#sidebarCollapsed.value = false;
    }
  }
  // ── Toolbar item resolution ──
  #resolveToolbarItems(items) {
    const resolved = [];
    for (const item of items) {
      if (item.type === "group") {
        const groupItems = this.#resolveToolbarItems(item.items);
        resolved.push({
          schema: item,
          enabled: computed18(() => true),
          visible: computed18(() => {
            const tier = this.#breakpointObserver.breakpoint.value;
            if (item.visible === false) return false;
            return isCategoryVisible(item.category, tier);
          }),
          label: ""
        });
        resolved.push(...groupItems);
        continue;
      }
      if (item.type === "separator") {
        resolved.push({
          schema: item,
          enabled: computed18(() => true),
          visible: computed18(() => item.visible !== false),
          label: ""
        });
        continue;
      }
      if (item.type === "custom") {
        resolved.push({
          schema: item,
          enabled: computed18(() => true),
          visible: computed18(() => {
            const tier = this.#breakpointObserver.breakpoint.value;
            if (item.visible === false) return false;
            return isCategoryVisible(item.category, tier);
          }),
          label: ""
        });
        continue;
      }
      const commandId = "commandId" in item ? item.commandId : void 0;
      const command = commandId !== void 0 ? this.#commands.get(commandId) : void 0;
      const labelKey = "labelKey" in item ? item.labelKey : void 0;
      const tooltipKey = "tooltipKey" in item ? item.tooltipKey : void 0;
      const translatedLabel = labelKey && this.#translator ? this.#translator(labelKey) : void 0;
      const translatedTooltip = tooltipKey && this.#translator ? this.#translator(tooltipKey) : void 0;
      const label = translatedLabel ?? item.label ?? command?.label ?? "";
      const icon = item.icon ?? command?.icon;
      const literalTooltip = "tooltip" in item ? item.tooltip : void 0;
      const tooltip = translatedTooltip ?? literalTooltip ?? label;
      const enabled = computed18(() => {
        if (command?.enabled !== void 0) return command.enabled.value;
        return true;
      });
      const visible = computed18(() => {
        const tier = this.#breakpointObserver.breakpoint.value;
        if (item.visible === false) return false;
        return isCategoryVisible(item.category, tier);
      });
      resolved.push({ schema: item, enabled, visible, label, icon, tooltip });
    }
    return resolved;
  }
  [Symbol.dispose]() {
    this.#breakpointObserver[Symbol.dispose]();
    for (const unsub of this.#cleanups) unsub();
    this.#cleanups.length = 0;
  }
};

// src/ui/default-schema.ts
var DEFAULT_UI_SCHEMA = {
  toolbar: {
    visible: true,
    position: "top",
    items: [
      // ── Left: hamburger + sidebar + page navigation ──
      {
        id: "tb-hamburger",
        type: "dropdown",
        icon: "menu",
        tooltipKey: "toolbar.menu",
        section: "left",
        category: "essential",
        priority: 110,
        items: [
          {
            id: "menu-open",
            type: "item",
            commandId: "document.open",
            labelKey: "toolbar.open",
            icon: "file-up"
          },
          {
            id: "menu-close",
            type: "item",
            commandId: "document.close",
            labelKey: "toolbar.close",
            icon: "x"
          },
          { id: "menu-sep-file", type: "separator" },
          {
            id: "menu-print",
            type: "item",
            commandId: "document.print",
            labelKey: "toolbar.print",
            icon: "printer"
          },
          {
            id: "menu-export",
            type: "item",
            commandId: "document.export",
            labelKey: "toolbar.export",
            icon: "download"
          },
          {
            id: "menu-screenshot",
            type: "item",
            commandId: "document.screenshot",
            labelKey: "toolbar.screenshot",
            icon: "camera"
          },
          { id: "menu-sep-tools", type: "separator" },
          {
            id: "menu-security",
            type: "item",
            commandId: "document.security",
            labelKey: "toolbar.passwordProtect",
            icon: "shield"
          },
          {
            id: "menu-fullscreen",
            type: "item",
            commandId: "ui.fullscreen",
            labelKey: "toolbar.fullscreen",
            icon: "fullscreen"
          }
        ]
      },
      {
        id: "tb-sidebar-toggle",
        type: "button",
        commandId: "ui.toggle-sidebar",
        icon: "sidebar",
        tooltipKey: "toolbar.sidebar",
        section: "left",
        category: "essential",
        priority: 100
      },
      {
        id: "tb-sep-1",
        type: "separator",
        section: "left",
        category: "essential",
        priority: 95
      },
      {
        id: "tb-prev-page",
        type: "button",
        commandId: "navigation.previous-page",
        icon: "chevron-up",
        tooltipKey: "toolbar.prevPage",
        section: "left",
        category: "essential",
        priority: 90
      },
      {
        id: "tb-page-input",
        type: "custom",
        component: "page-input",
        section: "left",
        category: "essential",
        priority: 85
      },
      {
        id: "tb-next-page",
        type: "button",
        commandId: "navigation.next-page",
        icon: "chevron-down",
        tooltipKey: "toolbar.nextPage",
        section: "left",
        category: "essential",
        priority: 80
      },
      // ── Center: integrated zoom control + fit ──
      {
        id: "tb-zoom",
        type: "custom",
        component: "zoom-display",
        section: "center",
        category: "essential",
        priority: 70
      },
      {
        id: "tb-sep-2",
        type: "separator",
        section: "center",
        category: "standard",
        priority: 55
      },
      {
        id: "tb-fit-page",
        type: "button",
        commandId: "zoom.fit-page",
        icon: "fit-page",
        tooltipKey: "toolbar.fitPage",
        section: "center",
        category: "standard",
        priority: 50
      },
      {
        id: "tb-fit-width",
        type: "button",
        commandId: "zoom.fit-width",
        icon: "fit-width",
        tooltipKey: "toolbar.fitWidth",
        section: "center",
        category: "standard",
        priority: 45
      },
      // ── Right: search, tools, more ──
      {
        id: "tb-search",
        type: "button",
        commandId: "search.open",
        icon: "search",
        tooltipKey: "toolbar.searchShortcut",
        section: "right",
        category: "essential",
        priority: 50
      },
      {
        id: "tb-sep-3",
        type: "separator",
        section: "right",
        category: "standard",
        priority: 45
      },
      {
        id: "tb-pointer-mode",
        type: "toggle",
        commandId: "interaction.pointer-mode",
        icon: "cursor",
        tooltipKey: "tool.pointer",
        section: "right",
        category: "standard",
        priority: 40
      },
      {
        id: "tb-text-select-mode",
        type: "toggle",
        commandId: "interaction.text-select-mode",
        icon: "text-select",
        tooltipKey: "tool.textSelect",
        section: "right",
        category: "standard",
        priority: 35
      },
      {
        id: "tb-pan-mode",
        type: "toggle",
        commandId: "interaction.pan-mode",
        icon: "hand",
        tooltipKey: "tool.hand",
        section: "right",
        category: "standard",
        priority: 30
      },
      {
        id: "tb-sep-4",
        type: "separator",
        section: "right",
        category: "standard",
        priority: 25
      },
      {
        // The annotate-toolbar toggle. In the schema so embedders can drop
        // it (the viewer's builder honors toolbar.items as the visibility
        // list); permission gating still applies on top.
        id: "tb-annotate",
        type: "button",
        commandId: "ui.toggle-annotation-toolbar",
        icon: "annotation",
        tooltipKey: "toolbar.annotate",
        section: "right",
        category: "standard"
      },
      {
        id: "tb-comments",
        type: "button",
        commandId: "ui.toggle-comments-sidebar",
        icon: "message-square",
        tooltipKey: "toolbar.comments",
        section: "right",
        category: "standard",
        priority: 22
      },
      {
        id: "tb-undo",
        type: "button",
        commandId: "history.undo",
        icon: "undo",
        tooltipKey: "toolbar.undoShortcut",
        section: "right",
        category: "standard",
        priority: 20
      },
      {
        id: "tb-redo",
        type: "button",
        commandId: "history.redo",
        icon: "redo",
        tooltipKey: "toolbar.redoShortcut",
        section: "right",
        category: "standard",
        priority: 15
      },
      {
        id: "tb-more",
        type: "dropdown",
        icon: "more-vertical",
        tooltipKey: "toolbar.more",
        section: "right",
        category: "essential",
        priority: 1,
        items: [
          {
            id: "menu-layout-single",
            type: "item",
            commandId: "viewport.layout-single",
            labelKey: "layout.single"
          },
          {
            id: "menu-layout-continuous",
            type: "item",
            commandId: "viewport.layout-continuous",
            labelKey: "layout.continuous"
          },
          {
            id: "menu-layout-double",
            type: "item",
            commandId: "viewport.layout-double",
            labelKey: "layout.double"
          },
          { id: "menu-sep-1", type: "separator" },
          {
            id: "menu-theme-light",
            type: "item",
            commandId: "ui.theme-light",
            labelKey: "theme.light"
          },
          {
            id: "menu-theme-dark",
            type: "item",
            commandId: "ui.theme-dark",
            labelKey: "theme.dark"
          },
          {
            id: "menu-theme-system",
            type: "item",
            commandId: "ui.theme-system",
            labelKey: "theme.system"
          }
        ]
      }
    ]
  },
  sidebar: {
    position: "left",
    collapsed: false,
    panels: [
      {
        id: "thumbnails",
        label: "Pages",
        labelKey: "sidebar.thumbnails",
        icon: "grid",
        component: "thumbnails",
        defaultWidth: 200,
        resizable: true,
        minWidth: 160,
        maxWidth: 400
      },
      {
        id: "bookmarks",
        label: "Bookmarks",
        labelKey: "sidebar.outline",
        icon: "bookmark",
        component: "bookmarks",
        defaultWidth: 260,
        resizable: true,
        minWidth: 180,
        maxWidth: 480
      },
      {
        id: "annotations",
        label: "Annotations",
        labelKey: "sidebar.annotations",
        icon: "annotation",
        component: "annotations",
        defaultWidth: 280,
        resizable: true,
        minWidth: 200,
        maxWidth: 480
      },
      {
        id: "attachments",
        label: "Attachments",
        labelKey: "sidebar.attachments",
        icon: "paperclip",
        component: "attachments",
        defaultWidth: 260,
        resizable: true,
        minWidth: 180,
        maxWidth: 480
      },
      {
        id: "signatures",
        label: "Signatures",
        labelKey: "sidebar.signatures",
        icon: "pen-tool",
        component: "signatures",
        defaultWidth: 260,
        resizable: true,
        minWidth: 180,
        maxWidth: 480
      },
      {
        id: "layers",
        label: "Layers",
        labelKey: "sidebar.layers",
        icon: "layers",
        component: "layers",
        defaultWidth: 240,
        resizable: true,
        minWidth: 160,
        maxWidth: 400
      },
      {
        id: "comparison",
        label: "Changes",
        labelKey: "sidebar.comparison",
        icon: "git-compare",
        component: "comparison",
        defaultWidth: 320,
        resizable: true,
        minWidth: 220,
        maxWidth: 520
      }
    ]
  },
  statusBar: {
    visible: true,
    items: [
      {
        id: "sb-page-count",
        component: "page-count",
        section: "left",
        priority: 100
      },
      {
        id: "sb-zoom-level",
        component: "zoom-level",
        section: "right",
        priority: 100
      }
    ]
  },
  contextMenu: {
    items: [
      {
        id: "ctx-copy",
        type: "item",
        commandId: "text.copy",
        labelKey: "capture.copy",
        shortcutHint: "Ctrl+C",
        when: "text-selected"
      },
      {
        id: "ctx-select-all",
        type: "item",
        commandId: "text.select-all",
        labelKey: "common.selectAll",
        shortcutHint: "Ctrl+A"
      },
      { id: "ctx-sep-1", type: "separator" },
      {
        id: "ctx-zoom-in",
        type: "item",
        commandId: "zoom.in",
        labelKey: "toolbar.zoomIn",
        shortcutHint: "Ctrl+="
      },
      {
        id: "ctx-zoom-out",
        type: "item",
        commandId: "zoom.out",
        labelKey: "toolbar.zoomOut",
        shortcutHint: "Ctrl+-"
      }
    ]
  },
  overlays: [
    {
      id: "search-bar",
      component: "search-bar",
      position: "top-right",
      dismissOnClickOutside: true
    }
  ],
  breakpoints: {
    compact: 640,
    wide: 1024
  }
};
function mergeSchema(base, override) {
  return {
    toolbar: override.toolbar !== void 0 ? { ...base.toolbar, ...override.toolbar } : base.toolbar,
    sidebar: override.sidebar !== void 0 ? { ...base.sidebar, ...override.sidebar } : base.sidebar,
    statusBar: override.statusBar !== void 0 ? { ...base.statusBar, ...override.statusBar } : base.statusBar,
    contextMenu: override.contextMenu !== void 0 ? { ...base.contextMenu, ...override.contextMenu } : base.contextMenu,
    overlays: override.overlays !== void 0 ? override.overlays : base.overlays,
    breakpoints: override.breakpoints !== void 0 ? { ...base.breakpoints, ...override.breakpoints } : base.breakpoints
  };
}

// src/ui/css/index.ts
var TOKENS_CSS = new URL(/* @vite-ignore */ "./css/tokens.css", import.meta.url).href;
var BASE_CSS = new URL(/* @vite-ignore */ "./css/base.css", import.meta.url).href;
var injected = false;
function injectLectorStyles() {
  if (injected || typeof document === "undefined") return;
  injected = true;
  const head = document.head;
  const tokensLink = document.createElement("link");
  tokensLink.rel = "stylesheet";
  tokensLink.href = TOKENS_CSS;
  tokensLink.dataset["lector"] = "tokens";
  head.appendChild(tokensLink);
  const baseLink = document.createElement("link");
  baseLink.rel = "stylesheet";
  baseLink.href = BASE_CSS;
  baseLink.dataset["lector"] = "base";
  head.appendChild(baseLink);
}
function buildViewerClass(theme, customClass) {
  const classes = ["lector-viewer"];
  if (theme === "dark") {
    classes.push("lector-viewer--dark");
  } else if (theme === "system") {
    classes.push("lector-viewer--system");
  }
  if (customClass) {
    classes.push(customClass);
  }
  return classes.join(" ");
}

// src/plugins/ui-plugin.ts
var uiPlugin = definePlugin({
  id: "ui",
  provides: ["ui"],
  requires: ["viewport"],
  optional: ["i18n"],
  setup(ctx) {
    const viewport = ctx.require("viewport");
    const i18n = ctx.optional("i18n");
    const schema = DEFAULT_UI_SCHEMA;
    const manager = new UIManager(
      schema,
      ctx.engine.plugins.commands,
      i18n ? (key) => i18n.t(key) : void 0
    );
    const viewerClass = computed19(() => {
      const theme = manager.state.theme.value;
      return buildViewerClass(theme);
    });
    ctx.registerCommand({
      id: "ui.toggle-sidebar",
      label: "Toggle Sidebar",
      icon: "sidebar",
      category: "UI",
      execute: () => {
        manager.toggleSidebar();
      }
    });
    ctx.registerCommand({
      id: "ui.toggle-comments-sidebar",
      label: "Toggle Comments",
      icon: "message-square",
      category: "UI",
      // The viewer owns the comments sidebar element and state, so this
      // command emits an event the viewer listens for instead of poking
      // at UIManager directly.
      execute: () => {
        ctx.emit("ui:toggle-comments-sidebar");
      }
    });
    ctx.registerCommand({
      id: "ui.theme-light",
      label: "Light Theme",
      category: "UI",
      execute: () => {
        manager.setTheme("light");
      }
    });
    ctx.registerCommand({
      id: "ui.theme-dark",
      label: "Dark Theme",
      category: "UI",
      execute: () => {
        manager.setTheme("dark");
      }
    });
    ctx.registerCommand({
      id: "ui.theme-system",
      label: "System Theme",
      category: "UI",
      execute: () => {
        manager.setTheme("system");
      }
    });
    function setActiveLayout(mode) {
      const active = viewport.activeViewport.peek();
      if (active) active.setLayoutMode(mode);
      else viewport.setLayoutMode(mode);
    }
    ctx.registerCommand({
      id: "viewport.layout-single",
      label: "Single Page",
      icon: "layout-single",
      category: "Layout",
      execute: () => {
        setActiveLayout("single");
      }
    });
    ctx.registerCommand({
      id: "viewport.layout-continuous",
      label: "Continuous Scroll",
      icon: "layout-continuous",
      category: "Layout",
      execute: () => {
        setActiveLayout("continuous");
      }
    });
    ctx.registerCommand({
      id: "viewport.layout-double",
      label: "Two-Page Spread",
      icon: "layout-double",
      category: "Layout",
      execute: () => {
        setActiveLayout("double");
      }
    });
    ctx.registerCommand({ id: "document.open", label: "Open", icon: "file-up", category: "Document", execute: () => {
      ctx.emit("ui:document-open");
    } });
    ctx.registerCommand({ id: "document.close", label: "Close", icon: "x", category: "Document", execute: () => {
      ctx.emit("ui:document-close");
    } });
    ctx.registerCommand({ id: "document.save", label: "Save", icon: "save", category: "Document", execute: () => {
      ctx.emit("ui:document-save");
    } });
    ctx.registerCommand({ id: "document.print", label: "Print", icon: "printer", category: "Document", execute: () => {
      ctx.emit("ui:document-print");
    } });
    ctx.registerCommand({ id: "document.export", label: "Download PDF", icon: "download", category: "Document", execute: () => {
      ctx.emit("ui:document-export");
    } });
    ctx.registerCommand({ id: "document.screenshot", label: "Export Page as Image", icon: "camera", category: "Document", execute: () => {
      ctx.emit("ui:document-screenshot");
    } });
    ctx.registerCommand({ id: "document.protect", label: "Password Protect", icon: "shield", category: "Document", execute: () => {
      ctx.emit("ui:document-protect");
    } });
    ctx.registerCommand({ id: "document.sign", label: "Sign Document", icon: "pen-tool", category: "Document", execute: () => {
      ctx.emit("ui:document-sign");
    } });
    ctx.registerCommand({ id: "viewer.split-horizontal", label: "Split horizontally", icon: "columns", category: "View", execute: () => {
      ctx.emit("ui:split-horizontal");
    } });
    ctx.registerCommand({ id: "viewer.split-vertical", label: "Split vertically", icon: "rows", category: "View", execute: () => {
      ctx.emit("ui:split-vertical");
    } });
    ctx.registerCommand({ id: "viewer.close-extra-panes", label: "Close split panes", icon: "x", category: "View", execute: () => {
      ctx.emit("ui:close-extra-panes");
    } });
    ctx.registerCommand({
      id: "ui.fullscreen",
      label: "Fullscreen",
      icon: "fullscreen",
      category: "UI",
      execute: () => {
        const viewer = document.querySelector(".lector-viewer");
        if (viewer) {
          if (document.fullscreenElement) {
            void document.exitFullscreen();
          } else {
            void viewer.requestFullscreen();
          }
        }
      }
    });
    const capability = {
      state: manager.state,
      viewerClass,
      setTheme(mode) {
        manager.setTheme(mode);
      },
      get effectiveTheme() {
        return manager.effectiveTheme;
      },
      toggleSidebar() {
        manager.toggleSidebar();
      },
      setSidebarCollapsed(collapsed) {
        manager.setSidebarCollapsed(collapsed);
      },
      setActivePanel(panelId) {
        manager.setActivePanel(panelId);
      },
      updateSchema(override) {
        const merged = mergeSchema(manager.schema, override);
        manager.updateSchema(merged);
      },
      get schema() {
        return manager.schema;
      },
      manager
    };
    return capability;
  },
  dispose() {
  }
});

// src/ui/lector-viewer.ts
import { effect as effect2, signal as signal24 } from "@truespar/lector-utils";
var LectorViewer = class _LectorViewer {
  #engine;
  #root;
  #cleanups = [];
  /**
   * Per-section teardowns for rebuildable UI chrome (toolbar, annotation
   * toolbar, sidebar, page controls). Each rebuild of a section drains its own
   * list before re-registering, so repeated rebuilds (locale switches, dynamic
   * toolbar config) don't accumulate listeners / effects / subscriptions on
   * detached DOM. Drained in destroy() too.
   */
  #sections = /* @__PURE__ */ new Map();
  /** When non-null, #pushCleanup routes teardowns into this section's list. */
  #sectionSink = null;
  #ui;
  #viewport;
  /**
   * The viewport instance bound to this viewer's canvas. The capability
   * `#viewport` is the multi-instance manager; this is the actual
   * viewport whose state (scroll, scale, layout) drives this viewer's
   * rendering. Created in the constructor, destroyed on `destroy()`.
   */
  #viewportInstance;
  #zoom;
  #document;
  #annotation = null;
  #presets = null;
  #measurement = null;
  #comparison = null;
  /**
   * Index of the currently-focused change inside the active comparison
   * result, or -1 when nothing is selected. Mirrors the click / prev /
   * next navigation in the Changes panel and drives the highlighted
   * overlay on both panes.
   */
  #activeChangeIndex = -1;
  /**
   * Reentrancy guard for synchronised scroll between the two panes in
   * compare mode. Set to true when one pane's scroll handler is about to
   * programmatically scroll the other so the second pane's scroll
   * handler doesn't echo back and produce a feedback loop.
   */
  #syncScrollLock = false;
  /**
   * Active sync-scroll listener cleanups, registered when comparison
   * enters and torn down on exit. Kept separate from `#cleanups` so the
   * viewer doesn't accumulate stale subscriptions across compare cycles.
   */
  #compareSyncCleanups = [];
  /** Filter chip state for the Changes sidebar panel. */
  #compareFilter = "all";
  #navigation = null;
  #attachment = null;
  #signature = null;
  #sigValidation = null;
  // Cache validation results per document to avoid re-running on every
  // panel render (validation is expensive — full BER parse + crypto verify).
  #validationCache = /* @__PURE__ */ new Map();
  /**
   * Per-document cache of signature info, populated eagerly on document
   * load. The first click on the signatures tab is then instant — no
   * round-trip to the worker is needed before the loading banner shows.
   */
  #sigInfoCache = /* @__PURE__ */ new Map();
  #layer = null;
  #pageOps = null;
  #redaction = null;
  #search = null;
  #i18n = null;
  #formatting = null;
  #capture = null;
  /** The text layer owns SELECTION — its own model, built by hit-testing
   *  pdfium char boxes, not a DOM Selection. The viewer had no reference to
   *  it at all, which is why two of its own features reached for
   *  `window.getSelection()` and always found nothing (2026-08-14). */
  #textLayer = null;
  #captureActionBar = null;
  #docManager = null;
  // DOM
  #viewerEl;
  #toolbar;
  #annotToolbar;
  #docTabs;
  #sidebarEl;
  /**
   * Right-hand comments sidebar element. Independent of the left
   * sidebar — both can be open at once. Holds the active doc's
   * comment threads with reply / resolve / filter controls.
   */
  #commentsSidebarEl;
  #commentsSidebarBody;
  #canvas;
  #scrollArea;
  #pageControlsEl;
  #tooltipEl;
  #annotPopover;
  /** ID of the annotation the popover is currently shown for (for scroll tracking). */
  #annotPopoverAnnotId = null;
  #pageElements = /* @__PURE__ */ new Map();
  #pageCanvases = /* @__PURE__ */ new Map();
  #renderedPages = /* @__PURE__ */ new Set();
  /** Debounce timer for pixel-buffer resizes during zoom. */
  #pixelResizeTimer = null;
  /** Pending pixel-buffer resizes, keyed by page index. */
  #pendingResizes = /* @__PURE__ */ new Map();
  /**
   * Additional `LectorPane`s mounted alongside the main canvas in
   * split-pane mode. Each pane has its own viewport instance, render
   * loop, and overlays — all sharing the engine, document, and chrome.
   * Empty in single-pane mode.
   */
  #extraPanes = /* @__PURE__ */ new Map();
  /** Wrapper element holding the main canvas plus any extra panes. */
  #canvasWrap;
  /**
   * Non-scrolling host element that wraps the primary `#canvas`. In
   * single-pane mode it fills `#canvasWrap` absolutely; in split mode
   * it becomes the flex child (sibling to the divider + right pane).
   *
   * Why a separate host: the active-pane frame is an `::after`
   * pseudo-element sized to the viewport. If it lived on `#canvas`
   * directly, it would scroll with the page content (because `#canvas`
   * is the scroll container) and only cover the first viewport height.
   * Putting the frame on this non-scrolling wrapper pins it to the
   * pane edges regardless of scroll offset. The extra-pane case
   * already had this structure (`.lector-pane` wrapping `.lector-pane__canvas`);
   * the host brings the primary canvas in line with it.
   */
  #canvasHost;
  /** Counter used to mint unique extra-pane ids. */
  #nextExtraPaneId = 0;
  #tooltipTimer = null;
  #searchBarEl = null;
  #overlays;
  /**
   * Signature status badge in the right side of the main toolbar.
   * Hidden when the active document has no signatures. Otherwise shows
   * a shield icon coloured by the worst-case validation status across
   * all signatures (green=valid, amber=unknown, red=invalid). Click
   * jumps to the signatures sidebar panel; hover shows a structured
   * popover with per-signature integrity + signature details.
   */
  #sigStatusBtn = null;
  /**
   * Floating popover anchored to #sigStatusBtn on hover. Content is
   * rebuilt on each show from the latest #sigInfoCache + #validationCache.
   */
  #sigStatusPopover = null;
  /**
   * Compare button in the right toolbar group. Hidden when the
   * comparison plugin is not registered. Visible-but-disabled when the
   * active tab is not a split with two distinct docs. Active state
   * mirrors the comparison plugin's state signal so the button toggles
   * between "Compare" and "Exit compare".
   */
  #compareBtn = null;
  /**
   * Floating action bar shown at the bottom-centre of the viewport
   * whenever 2+ annotations are selected (multi-select via shift-click).
   * Hidden when 0 or 1 annotations are selected. Provides Group /
   * Ungroup / Delete actions across the whole selection set.
   */
  #multiSelectBar = null;
  // State
  /**
   * Whether the right-hand comments sidebar is collapsed. Lives outside
   * the UIManager because it's a viewer-level concern, not part of the
   * left-sidebar schema. Auto-opened when the user selects an annotation
   * (per the agreed UX), closed via its own close button or the toolbar
   * toggle. Per-session only — not persisted.
   */
  #commentsSidebarCollapsed = signal24(true);
  /**
   * The annotation id whose thread is currently expanded inside the
   * comments sidebar. Mirrors the annotation plugin's selectedAnnotation
   * but is scoped to the sidebar's UI state — kept here so the panel
   * can re-render its expanded thread on selection changes.
   */
  #activeThreadAnnotId = null;
  /** Current filter for the comments sidebar list. */
  #commentsFilter = "all";
  /** Current sort order for the comments sidebar list. */
  #commentsSort = "page";
  /**
   * Timer for the debounced comments sidebar refresh. Held so we can
   * cancel a pending refresh when an annotation drag starts (the
   * sidebar rebuild would block move events and make the drag jank).
   */
  #commentsRefreshTimer = null;
  /**
   * True while the user is actively dragging an annotation on the
   * canvas. Used to suppress sidebar refreshes that would otherwise
   * rebuild DOM during the drag and block subsequent pointermove
   * events from being painted smoothly. Cleared on drag-end.
   */
  #annotDragging = false;
  /**
   * Tab list. Each entry is either a single doc or a split-pane pair of
   * docs. Splits are created via "Split horizontally"/"Split vertically"
   * which prompts the user for a second file. Closing a side of a split
   * demotes the tab back to a single. Closing the whole tab closes both
   * docs in the engine.
   */
  #tabs = [];
  /** Index into `#tabs` of the currently displayed tab, or -1 if empty. */
  #activeTabIndex = -1;
  /**
   * The currently active LectorPane in a split tab — `'left'` or
   * `'right'`. Used to decide which side of a split tab to focus when
   * switching back to it. Always `'left'` for single tabs.
   */
  #activeTabSide = "left";
  #allowLocalOpen;
  #documentTabs;
  #toolbarExtras;
  #pendingInitialZoom;
  #tileManager = new TileManager();
  #tileRepaintScheduled = false;
  constructor(options) {
    this.#engine = options.engine;
    this.#root = options.container;
    this.#allowLocalOpen = options.allowLocalOpen ?? false;
    this.#documentTabs = options.documentTabs ?? true;
    this.#toolbarExtras = options.toolbarExtras ?? [];
    const p = this.#engine.plugins;
    this.#ui = p.get("ui");
    this.#viewport = p.get("viewport");
    this.#zoom = p.get("zoom");
    this.#document = p.get("document");
    this.#annotation = p.tryGet("annotation");
    this.#presets = p.tryGet("annotation-presets");
    this.#measurement = p.tryGet("measurement");
    this.#comparison = p.tryGet("comparison");
    this.#navigation = p.tryGet("navigation");
    this.#attachment = p.tryGet("attachment");
    this.#signature = p.tryGet("signature");
    this.#sigValidation = p.tryGet("signature-validation");
    this.#layer = p.tryGet("layer");
    this.#pageOps = p.tryGet("page-ops");
    this.#redaction = p.tryGet("redaction");
    this.#search = p.tryGet("search");
    this.#i18n = p.tryGet("i18n");
    this.#formatting = p.tryGet("formatting");
    this.#capture = p.tryGet("capture");
    this.#textLayer = p.tryGet("text-layer");
    this.#docManager = p.tryGet("document-manager");
    this.#buildDOM();
    this.#wireEffects();
    this.#tileManager.onTileReady = () => {
      if (this.#tileRepaintScheduled) return;
      this.#tileRepaintScheduled = true;
      requestAnimationFrame(() => {
        this.#tileRepaintScheduled = false;
        const vis = this.#viewportInstance.visiblePages.peek();
        if (vis.length > 0) void this.#renderVisiblePages(vis);
      });
    };
    this.#applyInitialOptions(options);
  }
  #applyInitialOptions(opts) {
    const theme = opts.theme ?? "system";
    if (theme !== "system") {
      try {
        this.#engine.plugins.commands.execute(`ui.theme-${theme}`);
      } catch {
      }
    }
    if (opts.sidebarOpen !== void 0) {
      this.#ui.setSidebarCollapsed(!opts.sidebarOpen);
    }
    if (opts.initialPanel) {
      const wasCollapsed = this.#ui.state.sidebar.collapsed.peek();
      this.#ui.setActivePanel(opts.initialPanel);
      if (opts.sidebarOpen === void 0) {
        this.#ui.setSidebarCollapsed(wasCollapsed);
      }
    }
    if (opts.panels) {
      const allowed = new Set(opts.panels);
      const currentPanels = this.#ui.state.sidebar.panels;
      const filtered = currentPanels.filter((p) => allowed.has(p.id));
      if (filtered.length !== currentPanels.length) {
        try {
          this.#ui.updateSchema({ sidebar: { panels: filtered } });
          this.#buildSidebar();
        } catch {
        }
      }
    }
    if (opts.uiSchema) {
      try {
        this.#ui.updateSchema(opts.uiSchema);
        this.#buildToolbar();
        this.#buildSidebar();
      } catch {
      }
    }
    if (opts.layoutMode) {
      const layoutMap = {
        "single": "viewport.layout-single",
        "continuous": "viewport.layout-continuous",
        "double": "viewport.layout-double",
        "book": "viewport.layout-book"
      };
      const cmd = layoutMap[opts.layoutMode];
      if (cmd) try {
        this.#engine.plugins.commands.execute(cmd);
      } catch {
      }
    }
    if (opts.initialZoom !== void 0) {
      this.#pendingInitialZoom = opts.initialZoom;
    }
  }
  // ─── DOM Construction ──────────────────────────────────
  #buildDOM() {
    this.#root.innerHTML = "";
    const ws = this.#el("div", "lector-workspace");
    if (this.#i18n) {
      ws.setAttribute("lang", this.#i18n.locale.peek());
    }
    this.#root.appendChild(ws);
    ws.addEventListener("contextmenu", (e) => {
      const tag = e.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      e.preventDefault();
    });
    const skip = document.createElement("a");
    skip.href = "#";
    skip.className = "lector-skip-link";
    skip.textContent = this.#t("toolbar.skipToContent");
    skip.addEventListener("click", (e) => {
      e.preventDefault();
      this.#canvas?.focus({ preventScroll: false });
    });
    ws.appendChild(skip);
    this.#docTabs = this.#el("div", "lector-doctabs");
    if (this.#documentTabs) ws.appendChild(this.#docTabs);
    this.#viewerEl = this.#el("div", "lector-viewer");
    ws.appendChild(this.#viewerEl);
    this.#toolbar = this.#el("div", "lector-toolbar");
    this.#toolbar.setAttribute("role", "toolbar");
    this.#toolbar.setAttribute("aria-label", this.#t("toolbar.menu"));
    this.#viewerEl.appendChild(this.#toolbar);
    this.#annotToolbar = this.#el("div", "lector-annot-toolbar lector-annot-toolbar--hidden");
    this.#annotToolbar.setAttribute("role", "toolbar");
    this.#annotToolbar.setAttribute("aria-label", this.#t("toolbar.annotate"));
    this.#viewerEl.appendChild(this.#annotToolbar);
    this.#sidebarEl = this.#el("div", "lector-sidebar");
    this.#sidebarEl.setAttribute("role", "complementary");
    this.#sidebarEl.setAttribute("aria-label", this.#t("toolbar.sidebar"));
    this.#viewerEl.appendChild(this.#sidebarEl);
    const sidebarBackdrop = this.#el("div", "lector-sidebar-backdrop");
    sidebarBackdrop.addEventListener("click", () => {
      this.#ui.setSidebarCollapsed(true);
    });
    ws.appendChild(sidebarBackdrop);
    this.#canvasWrap = this.#el("div", "lector-canvas-wrap");
    this.#viewerEl.appendChild(this.#canvasWrap);
    this.#commentsSidebarEl = this.#el(
      "div",
      "lector-comments-sidebar lector-comments-sidebar--collapsed"
    );
    this.#viewerEl.appendChild(this.#commentsSidebarEl);
    this.#buildCommentsSidebarShell();
    this.#canvasHost = this.#el("div", "lector-canvas-host");
    this.#canvasHost.setAttribute("role", "main");
    this.#canvasHost.setAttribute("aria-label", this.#t("page.viewer"));
    this.#canvasWrap.appendChild(this.#canvasHost);
    this.#canvas = this.#el("div", "lector-canvas");
    this.#canvas.tabIndex = 0;
    this.#canvasHost.appendChild(this.#canvas);
    this.#scrollArea = this.#el("div", "lector-canvas__scroll-area");
    this.#canvas.appendChild(this.#scrollArea);
    this.#pageControlsEl = this.#el("div", "lector-page-controls");
    this.#canvasWrap.appendChild(this.#pageControlsEl);
    this.#annotPopover = this.#el("div", "lector-annot-popover");
    this.#annotPopover.style.display = "none";
    this.#canvasWrap.appendChild(this.#annotPopover);
    this.#tooltipEl = this.#el("div", "lector-tooltip");
    this.#tooltipEl.setAttribute("role", "tooltip");
    this.#tooltipEl.id = "lector-tooltip";
    this.#tooltipEl.style.display = "none";
    document.body.appendChild(this.#tooltipEl);
    this.#sigStatusPopover = this.#el("div", "lector-sig-status-popover");
    this.#sigStatusPopover.style.display = "none";
    ws.appendChild(this.#sigStatusPopover);
    this.#ui.manager.observe(ws);
    this.#viewportInstance = this.#viewport.createViewport();
    this.#viewportInstance.attach(this.#canvas);
    this.#overlays = new PageOverlayManager(this.#engine, this.#viewportInstance, this.#formatting);
    this.#engine.plugins.events.emit("viewport:container-attached", this.#canvas);
  }
  #el(tag, cls) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
  }
  #btn(cls) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    return b;
  }
  #icon(name) {
    const s = this.#el("span", "lector-btn__icon");
    s.innerHTML = resolveIcon(name) ?? "";
    return s;
  }
  // ─── Tooltip ───────────────────────────────────────────
  #tooltipTarget = null;
  #showTooltip(target, text) {
    if (this.#tooltipTimer) clearTimeout(this.#tooltipTimer);
    this.#tooltipTimer = setTimeout(() => {
      const r = target.getBoundingClientRect();
      this.#tooltipEl.textContent = text;
      this.#tooltipEl.style.display = "";
      this.#tooltipEl.style.left = `${r.left + r.width / 2}px`;
      this.#tooltipEl.style.top = `${r.bottom + 8}px`;
      this.#tooltipEl.style.transform = "translateX(-50%)";
      target.setAttribute("aria-describedby", "lector-tooltip");
      this.#tooltipTarget = target;
    }, 500);
  }
  #hideTooltip() {
    if (this.#tooltipTimer) {
      clearTimeout(this.#tooltipTimer);
      this.#tooltipTimer = null;
    }
    this.#tooltipEl.style.display = "none";
    if (this.#tooltipTarget) {
      this.#tooltipTarget.removeAttribute("aria-describedby");
      this.#tooltipTarget = null;
    }
  }
  #tip(el, text) {
    el.setAttribute("aria-label", text);
    el.addEventListener("mouseenter", () => this.#showTooltip(el, text));
    el.addEventListener("mouseleave", () => this.#hideTooltip());
    el.addEventListener("pointerdown", () => this.#hideTooltip());
  }
  /**
   * Wire WAI-ARIA menu keyboard behaviour into a trigger + menu pair.
   *
   * Sets `role="menu"` on the container, `role="menuitem"` on every
   * `<button>` child that isn't a separator, `role="separator"` on
   * dividers. Adds `aria-haspopup` and `aria-expanded` to the trigger.
   * Keyboard: Enter/Space open the menu, Escape closes it, Arrow
   * Up/Down cycle through items with roving tabindex, Home/End jump to
   * first/last. Focus returns to the trigger on close.
   *
   * @param openClass CSS class toggled on `menu` to show/hide it.
   */
  #wireMenu(trigger, menu, openClass) {
    trigger.setAttribute("aria-haspopup", "menu");
    trigger.setAttribute("aria-expanded", "false");
    menu.setAttribute("role", "menu");
    for (const child of menu.children) {
      const el = child;
      if (el.tagName === "BUTTON") {
        el.setAttribute("role", "menuitem");
        el.tabIndex = -1;
      } else if (el.classList.contains("lector-dropdown__separator") || el.classList.contains("lector-toolbar__divider")) {
        el.setAttribute("role", "separator");
      }
    }
    const isOpen = () => menu.classList.contains(openClass);
    const items = () => Array.from(menu.querySelectorAll('[role="menuitem"]:not([disabled])'));
    const open = () => {
      menu.classList.add(openClass);
      trigger.setAttribute("aria-expanded", "true");
      const first = items()[0];
      if (first) first.focus();
    };
    const close = () => {
      menu.classList.remove(openClass);
      trigger.setAttribute("aria-expanded", "false");
      trigger.focus();
    };
    const focusItem = (offset) => {
      const list = items();
      if (list.length === 0) return;
      const current = list.indexOf(document.activeElement);
      let next = current + offset;
      if (next < 0) next = list.length - 1;
      if (next >= list.length) next = 0;
      list[next].focus();
    };
    trigger.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " " || e.key === "ArrowDown") {
        e.preventDefault();
        if (isOpen()) close();
        else open();
      }
      if (e.key === "Escape" && isOpen()) {
        e.preventDefault();
        close();
      }
    });
    menu.addEventListener("keydown", (e) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          focusItem(1);
          break;
        case "ArrowUp":
          e.preventDefault();
          focusItem(-1);
          break;
        case "Home":
          e.preventDefault();
          {
            const list = items();
            if (list[0]) list[0].focus();
          }
          break;
        case "End":
          e.preventDefault();
          {
            const list = items();
            if (list.length) list[list.length - 1].focus();
          }
          break;
        case "Escape":
          e.preventDefault();
          close();
          break;
        case "Tab":
          close();
          break;
      }
    });
    menu.addEventListener("click", (e) => {
      const target = e.target.closest('[role="menuitem"]');
      if (target) {
        trigger.setAttribute("aria-expanded", "false");
      }
    });
    const observer = new MutationObserver(() => {
      trigger.setAttribute("aria-expanded", String(isOpen()));
    });
    observer.observe(menu, { attributes: true, attributeFilter: ["class"] });
    this.#pushCleanup(() => observer.disconnect());
  }
  /**
   * Enhance a modal overlay+dialog pair with ARIA attributes and a
   * keyboard focus trap. Call this AFTER the dialog DOM is fully built
   * (title, body, actions) but BEFORE appending the overlay to the
   * workspace.
   *
   * What it does:
   *  1. `role="dialog"`, `aria-modal="true"`, `aria-labelledby` on the
   *     dialog element (pointing at the first `h3` inside it).
   *  2. Focus trap: Tab / Shift+Tab cycle through focusable descendants
   *     only. Focus never leaves the dialog while it's open.
   *  3. Escape key closes (removes) the overlay.
   *  4. On close, focus returns to `triggerEl` (the element that opened
   *     the modal). If null, no return-focus.
   *  5. Appends overlay to the workspace and moves focus into the dialog
   *     (to `initialFocus` if given, otherwise the first focusable
   *     element, otherwise the dialog itself).
   *
   * @returns A `close()` function the caller can invoke programmatically.
   */
  #openModal(overlay, dialog, triggerEl = null, initialFocus) {
    const titleEl = dialog.querySelector('h3, [class*="__title"]');
    if (titleEl) {
      const titleId = titleEl.id || `lector-modal-title-${Date.now()}`;
      titleEl.id = titleId;
      dialog.setAttribute("aria-labelledby", titleId);
    }
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    const close = () => {
      overlay.remove();
      if (triggerEl && triggerEl.isConnected) triggerEl.focus();
    };
    const focusableSelector = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
    const trapFocus = (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        close();
        return;
      }
      if (e.key !== "Tab") return;
      const focusable = Array.from(dialog.querySelectorAll(focusableSelector));
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    dialog.addEventListener("keydown", trapFocus);
    const ws = this.#root.querySelector(".lector-workspace") ?? this.#root;
    ws.appendChild(overlay);
    requestAnimationFrame(() => {
      if (initialFocus) {
        initialFocus.focus();
        return;
      }
      const first = dialog.querySelector(focusableSelector);
      if (first) first.focus();
      else dialog.focus();
    });
    return close;
  }
  /**
   * Wire keyboard navigation into a context menu (already appended to
   * DOM). Arrow Up/Down cycle items, Escape removes the menu, Enter
   * activates the focused item.
   */
  #wireContextMenuKeyboard(menu) {
    const items = () => Array.from(menu.querySelectorAll('[role="menuitem"]:not([disabled])'));
    const focusItem = (offset) => {
      const list = items();
      if (list.length === 0) return;
      const current = list.indexOf(document.activeElement);
      let next = current + offset;
      if (next < 0) next = list.length - 1;
      if (next >= list.length) next = 0;
      list[next].focus();
    };
    menu.addEventListener("keydown", (e) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          focusItem(1);
          break;
        case "ArrowUp":
          e.preventDefault();
          focusItem(-1);
          break;
        case "Home":
          e.preventDefault();
          {
            const l = items();
            if (l[0]) l[0].focus();
          }
          break;
        case "End":
          e.preventDefault();
          {
            const l = items();
            if (l.length) l[l.length - 1].focus();
          }
          break;
        case "Escape":
          e.preventDefault();
          menu.remove();
          break;
        case "Tab":
          menu.remove();
          break;
      }
    });
  }
  /** Resolve a translation key, falling back to the last segment of the key. */
  #t(key, params) {
    if (this.#i18n) return this.#i18n.t(key, params);
    const last = key.split(".").pop() ?? key;
    let result = last.replace(/([A-Z])/g, " $1").replace(/^./, (s) => s.toUpperCase()).trim();
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        result = result.replaceAll(`{${k}}`, String(v));
      }
    }
    return result;
  }
  // ─── Toolbar ───────────────────────────────────────────
  /**
   * Register a teardown. While a rebuildable section is being (re)built (see
   * {@link #runSection}), teardowns route into that section's list so the next
   * rebuild can drain them; otherwise they go to the viewer-lifetime list.
   */
  #pushCleanup(...fns) {
    (this.#sectionSink ?? this.#cleanups).push(...fns);
  }
  /**
   * (Re)build a named UI section: drain the section's prior teardowns, then
   * run `build` with teardowns routed into a fresh list for that section.
   */
  #runSection(key, build) {
    const prev = this.#sections.get(key);
    if (prev) {
      for (const fn of prev) {
        try {
          fn();
        } catch {
        }
      }
    }
    const sink = [];
    this.#sections.set(key, sink);
    const outer = this.#sectionSink;
    this.#sectionSink = sink;
    try {
      build();
    } finally {
      this.#sectionSink = outer;
    }
  }
  #buildToolbar() {
    this.#runSection("toolbar", () => this.#buildToolbarImpl());
  }
  /**
   * Whether the merged UI schema still lists a toolbar item. The builder is
   * hand-rolled for layout and behavior, but WHICH items exist is the
   * schema's call — an embedder dropping an id from `toolbar.items` (via
   * the `uiSchema` option) hides that piece. Ids the default schema never
   * carried (plugin-gated extras like capture and compare) stay visible
   * whenever their plugin is loaded.
   */
  #tbHas(id) {
    try {
      const items = this.#ui.schema.toolbar?.items;
      if (!items) return true;
      return items.some((i) => i.id === id);
    } catch {
      return true;
    }
  }
  /**
   * The icon the SCHEMA asks for, falling back to the builder's own default.
   *
   * `ToolbarItem.icon` has always been part of the schema and the builder has
   * always ignored it — every glyph was hardcoded at its call site. So an
   * embedder passing `uiSchema` could drop an item but not re-skin one, and
   * the type advertised a knob that did nothing. Found 2026-08-14 by paddock,
   * whose document pane contributes a collapse button that collided with the
   * sidebar toggle's own glyph and had nowhere to say so.
   */
  #tbIcon(id, fallback) {
    try {
      const item = this.#ui.schema.toolbar?.items?.find((i) => i.id === id);
      const icon = item && "icon" in item ? item.icon : void 0;
      return typeof icon === "string" && icon.length > 0 ? icon : fallback;
    } catch {
      return fallback;
    }
  }
  /** Drop leading/trailing/doubled dividers left behind by hidden items. */
  #pruneDividers(group) {
    const isDiv = (el) => !!el && el.classList.contains("lector-toolbar__divider");
    for (const d of [...group.querySelectorAll(".lector-toolbar__divider")]) {
      if (isDiv(d.previousElementSibling) || !d.previousElementSibling || !d.nextElementSibling) {
        d.remove();
      }
    }
  }
  #buildToolbarImpl() {
    this.#toolbar.innerHTML = "";
    const left = this.#el("div", "lector-toolbar__group");
    this.#addExtras(left, "left", "start");
    if (this.#tbHas("tb-hamburger")) {
      this.#addToolbarBtn(left, this.#tbIcon("tb-hamburger", "menu"), this.#t("toolbar.menu"), "dropdown-hamburger");
    }
    this.#addDivider(left);
    if (this.#tbHas("tb-sidebar-toggle")) {
      this.#addToolbarBtn(
        left,
        this.#tbIcon("tb-sidebar-toggle", "sidebar"),
        this.#t("toolbar.sidebar"),
        "ui.toggle-sidebar",
        { active: () => !this.#ui.state.sidebar.collapsed.peek() }
      );
    }
    this.#addExtras(left, "left", "end");
    this.#pruneDividers(left);
    this.#toolbar.appendChild(left);
    this.#toolbar.appendChild(this.#el("div", "lector-toolbar__spacer"));
    const center = this.#el("div", "lector-toolbar__group");
    this.#addExtras(center, "center", "start");
    if (this.#tbHas("tb-zoom")) {
      center.appendChild(this.#buildZoomControl());
    }
    this.#addDivider(center);
    if (this.#tbHas("tb-fit-page")) {
      this.#addToolbarBtn(center, this.#tbIcon("tb-fit-page", "fit-page"), this.#t("toolbar.fitPage"), "zoom.fit-page");
    }
    if (this.#tbHas("tb-fit-width")) {
      this.#addToolbarBtn(center, this.#tbIcon("tb-fit-width", "fit-width"), this.#t("toolbar.fitWidth"), "zoom.fit-width");
    }
    this.#addDivider(center);
    if (this.#tbHas("tb-pan-mode")) {
      this.#addToolbarBtn(center, this.#tbIcon("tb-pan-mode", "hand"), this.#t("tool.hand"), "interaction.pan-mode", { tool: "pan" });
    }
    if (this.#tbHas("tb-pointer-mode")) {
      this.#addToolbarBtn(center, this.#tbIcon("tb-pointer-mode", "cursor"), this.#t("tool.pointer"), "interaction.pointer-mode", { tool: "pointer" });
    }
    if (this.#tbHas("tb-text-select-mode")) {
      this.#addToolbarBtn(center, this.#tbIcon("tb-text-select-mode", "text-select"), this.#t("tool.textSelect"), "interaction.text-select-mode", { tool: "text-select" });
    }
    if (this.#capture) {
      this.#addToolbarBtn(center, "crop", this.#t("tool.capture"), "capture.toggle-marquee", { tool: "marquee" });
    }
    this.#addExtras(center, "center", "end");
    this.#pruneDividers(center);
    this.#toolbar.appendChild(center);
    this.#toolbar.appendChild(this.#el("div", "lector-toolbar__spacer"));
    const right = this.#el("div", "lector-toolbar__group");
    this.#addExtras(right, "right", "start");
    if (this.#tbHas("tb-annotate") && this.#isCommandAllowed("annotation.mode-highlight")) {
      const annotBtn = this.#btn("lector-btn");
      annotBtn.appendChild(this.#icon("annotation"));
      const annotLabel = this.#el("span", "lector-btn__label");
      annotLabel.textContent = this.#t("toolbar.annotate");
      annotBtn.appendChild(annotLabel);
      this.#tip(annotBtn, this.#t("toolbar.annotate"));
      annotBtn.dataset["action"] = "toggle-annot-toolbar";
      annotBtn.addEventListener("click", () => {
        const isHidden = this.#annotToolbar.classList.contains("lector-annot-toolbar--hidden");
        this.#annotToolbar.classList.toggle("lector-annot-toolbar--hidden", !isHidden);
        annotBtn.classList.toggle("lector-btn--active", isHidden);
        if (!isHidden && this.#annotation) {
          this.#annotation.setActiveTool(null);
        }
      });
      right.appendChild(annotBtn);
    }
    this.#addDivider(right);
    if (this.#tbHas("tb-undo")) {
      this.#addToolbarBtn(right, this.#tbIcon("tb-undo", "undo"), this.#t("toolbar.undoShortcut"), "history.undo");
    }
    if (this.#tbHas("tb-redo")) {
      this.#addToolbarBtn(right, this.#tbIcon("tb-redo", "redo"), this.#t("toolbar.redoShortcut"), "history.redo");
    }
    this.#addDivider(right);
    if (this.#tbHas("tb-search")) {
      this.#addToolbarBtn(right, this.#tbIcon("tb-search", "search"), this.#t("toolbar.search"), "search.open");
    }
    if (this.#comparison) {
      const cmpBtn = this.#btn("lector-btn");
      cmpBtn.appendChild(this.#icon("git-compare"));
      const cmpLabel = this.#el("span", "lector-btn__label");
      cmpLabel.textContent = this.#t("comparison.compare");
      cmpBtn.appendChild(cmpLabel);
      this.#tip(cmpBtn, this.#t("comparison.compare"));
      cmpBtn.addEventListener("click", () => {
        if (!this.#comparison) return;
        const state = this.#comparison.state.peek();
        if (state === "active" || state === "computing") {
          this.#exitComparison();
        } else {
          this.#startComparison();
        }
      });
      this.#compareBtn = cmpBtn;
      right.appendChild(cmpBtn);
      this.#updateCompareButton();
    }
    this.#sigStatusBtn = this.#btn("lector-btn lector-sig-status");
    this.#sigStatusBtn.style.display = "none";
    this.#sigStatusBtn.setAttribute("aria-label", this.#t("annotation.signature"));
    this.#sigStatusBtn.appendChild(this.#icon("shield"));
    this.#sigStatusBtn.addEventListener("click", () => {
      if (this.#ui.state.sidebar.collapsed.peek()) {
        this.#ui.setSidebarCollapsed(false);
      }
      this.#ui.setActivePanel("signatures");
      this.#hideSigStatusPopover();
    });
    this.#sigStatusBtn.addEventListener("mouseenter", () => {
      this.#showSigStatusPopover();
    });
    this.#sigStatusBtn.addEventListener("mouseleave", () => {
      this.#hideSigStatusPopover();
    });
    right.appendChild(this.#sigStatusBtn);
    if (this.#tbHas("tb-more")) {
      this.#addToolbarBtn(right, this.#tbIcon("tb-more", "more-vertical"), this.#t("toolbar.more"), "dropdown-more");
    }
    this.#addExtras(right, "right", "end");
    this.#pruneDividers(right);
    this.#toolbar.appendChild(right);
    this.#updateSignatureStatusBadge();
  }
  /**
   * Refresh the signature status badge in the toolbar. Called on:
   *   - active document change
   *   - signature info pre-fetch resolve (loadDocument)
   *   - validation start / done / error
   *   - document close
   *
   * Worst-case status across all signatures wins:
   *   invalid > unknown/error > valid
   * Hidden entirely when the active document has no signatures.
   */
  #updateSignatureStatusBadge() {
    const btn = this.#sigStatusBtn;
    if (!btn) return;
    const handle = this.#document.activeDocument.peek();
    if (!handle) {
      btn.style.display = "none";
      return;
    }
    const sigs = this.#sigInfoCache.get(handle.id);
    if (sigs === void 0) {
      btn.style.display = "none";
      return;
    }
    if (sigs.length === 0) {
      btn.style.display = "none";
      return;
    }
    btn.style.display = "";
    btn.classList.remove(
      "lector-sig-status--valid",
      "lector-sig-status--unknown",
      "lector-sig-status--invalid",
      "lector-sig-status--validating"
    );
    const cached = this.#validationCache.get(handle.id);
    let state = "validating";
    if (Array.isArray(cached)) {
      let hasInvalid = false;
      let hasUnknown = false;
      let hasError = false;
      for (const r of cached) {
        if (r.status === "invalid") hasInvalid = true;
        else if (r.status === "error") hasError = true;
        else if (r.status === "unknown") hasUnknown = true;
      }
      if (hasInvalid) state = "invalid";
      else if (hasError || hasUnknown) state = "unknown";
      else state = "valid";
    } else if (cached && typeof cached === "object" && "error" in cached) {
      state = "unknown";
    }
    btn.classList.add(`lector-sig-status--${state}`);
    const iconName = state === "invalid" ? "shield-off" : "shield";
    btn.innerHTML = "";
    btn.appendChild(this.#icon(iconName));
    if (this.#sigStatusPopover && this.#sigStatusPopover.style.display !== "none") {
      this.#buildSigStatusPopoverContent();
      this.#positionSigStatusPopover();
    }
  }
  // ── Signature status popover (richer hover detail) ──
  #showSigStatusPopover() {
    if (!this.#sigStatusPopover || !this.#sigStatusBtn) return;
    this.#buildSigStatusPopoverContent();
    this.#sigStatusPopover.style.display = "";
    this.#positionSigStatusPopover();
  }
  #hideSigStatusPopover() {
    if (!this.#sigStatusPopover) return;
    this.#sigStatusPopover.style.display = "none";
  }
  #positionSigStatusPopover() {
    if (!this.#sigStatusPopover || !this.#sigStatusBtn) return;
    const r = this.#sigStatusBtn.getBoundingClientRect();
    const pop = this.#sigStatusPopover;
    pop.style.top = `${r.bottom + 8}px`;
    const popRect = pop.getBoundingClientRect();
    let left = r.right - popRect.width;
    if (left < 8) left = 8;
    pop.style.left = `${left}px`;
  }
  #buildSigStatusPopoverContent() {
    const pop = this.#sigStatusPopover;
    if (!pop) return;
    pop.innerHTML = "";
    const handle = this.#document.activeDocument.peek();
    if (!handle) return;
    const sigs = this.#sigInfoCache.get(handle.id) ?? [];
    const cached = this.#validationCache.get(handle.id);
    const isPending = cached === void 0 || cached === "pending";
    const isError = cached !== void 0 && cached !== "pending" && !Array.isArray(cached);
    const results = Array.isArray(cached) ? cached : null;
    const header = this.#el("div", "lector-sig-status-popover__header");
    header.textContent = sigs.length === 1 ? this.#t("signatures.badgeStatusSingle") : this.#t("signatures.badgeStatusMany", { count: sigs.length });
    pop.appendChild(header);
    if (isPending) {
      const row = this.#el("div", "lector-sig-status-popover__row lector-sig-status-popover__row--muted");
      row.textContent = this.#t("signatures.validatingShort");
      pop.appendChild(row);
      return;
    }
    if (isError) {
      const row = this.#el("div", "lector-sig-status-popover__row lector-sig-status-popover__row--warn");
      const errMsg = cached.error;
      row.textContent = this.#t("signatures.validationError", { error: errMsg });
      pop.appendChild(row);
      return;
    }
    for (let i = 0; i < sigs.length; i++) {
      const sig = sigs[i];
      const v = results?.[i] ?? null;
      const block = this.#el("div", "lector-sig-status-popover__sig");
      const title = this.#el("div", "lector-sig-status-popover__sig-title");
      title.textContent = v?.signerCertificate?.subject || this.#t("signatures.itemTitle", { index: i + 1 });
      block.appendChild(title);
      const intRow = this.#el("div", "lector-sig-status-popover__row");
      if (v?.integrityValid) {
        intRow.classList.add("lector-sig-status-popover__row--ok");
        intRow.appendChild(this.#popoverIcon("check-circle"));
        intRow.appendChild(this.#popoverText(this.#t("signatures.integrityVerified")));
      } else if (v && v.status !== "error") {
        intRow.classList.add("lector-sig-status-popover__row--bad");
        intRow.appendChild(this.#popoverIcon("x-circle"));
        intRow.appendChild(this.#popoverText(this.#t("signatures.integrityModified")));
      } else {
        intRow.classList.add("lector-sig-status-popover__row--muted");
        intRow.appendChild(this.#popoverIcon("alert-circle"));
        intRow.appendChild(this.#popoverText(this.#t("signatures.integrityUnknown")));
      }
      block.appendChild(intRow);
      const sigRow = this.#el("div", "lector-sig-status-popover__row");
      if (v?.status === "valid") {
        sigRow.classList.add("lector-sig-status-popover__row--ok");
        sigRow.appendChild(this.#popoverIcon("check-circle"));
        sigRow.appendChild(this.#popoverText(this.#t("signatures.signatureValid")));
      } else if (v?.status === "invalid") {
        sigRow.classList.add("lector-sig-status-popover__row--bad");
        sigRow.appendChild(this.#popoverIcon("x-circle"));
        sigRow.appendChild(this.#popoverText(this.#t("signatures.signatureInvalid")));
      } else if (v?.signatureValid && !v?.certificateValid) {
        sigRow.classList.add("lector-sig-status-popover__row--warn");
        sigRow.appendChild(this.#popoverIcon("alert-circle"));
        sigRow.appendChild(this.#popoverText(
          v.signerCertificate?.isSelfSigned ? this.#t("signatures.certSelfSigned") : this.#t("signatures.certNotTrusted")
        ));
      } else {
        sigRow.classList.add("lector-sig-status-popover__row--warn");
        sigRow.appendChild(this.#popoverIcon("alert-circle"));
        sigRow.appendChild(this.#popoverText(this.#t("signatures.verificationNeeded")));
      }
      block.appendChild(sigRow);
      if (sig.reason) {
        const meta = this.#el("div", "lector-sig-status-popover__meta");
        meta.textContent = this.#t("signatures.reason", { reason: sig.reason });
        block.appendChild(meta);
      }
      if (sig.time) {
        const meta = this.#el("div", "lector-sig-status-popover__meta");
        meta.textContent = this.#t("signatures.signedAt", { time: this.#formatDate(sig.time) });
        block.appendChild(meta);
      }
      pop.appendChild(block);
    }
    const footer = this.#el("div", "lector-sig-status-popover__footer");
    footer.textContent = this.#t("signatures.fullDetailsHint");
    pop.appendChild(footer);
  }
  #popoverIcon(name) {
    const s = this.#el("span", "lector-sig-status-popover__icon");
    s.innerHTML = resolveIcon(name) ?? "";
    return s;
  }
  #popoverText(text) {
    const s = this.#el("span", "lector-sig-status-popover__text");
    s.textContent = text;
    return s;
  }
  #addToolbarBtn(parent, icon, tooltip, action, opts) {
    if (action.startsWith("dropdown-")) {
      this.#addDropdown(parent, icon, tooltip, action);
      return;
    }
    const btn = this.#btn("lector-btn");
    btn.appendChild(this.#icon(icon));
    this.#tip(btn, tooltip);
    btn.dataset["action"] = action;
    if (opts?.tool) btn.dataset["tool"] = opts.tool;
    btn.addEventListener("click", () => {
      try {
        void this.#engine.plugins.commands.execute(action);
      } catch {
      }
      if (opts?.active) {
        const isActive = opts.active();
        btn.classList.toggle("lector-btn--active", isActive);
        btn.setAttribute("aria-pressed", String(isActive));
      }
    });
    if (opts?.active) {
      const isActive = opts.active();
      if (isActive) btn.classList.add("lector-btn--active");
      btn.setAttribute("aria-pressed", String(isActive));
    }
    parent.appendChild(btn);
  }
  #addDivider(parent) {
    parent.appendChild(this.#el("div", "lector-toolbar__divider"));
  }
  /** Host-contributed buttons for one group, built with lector's own button
   *  so they match its metrics exactly. A leading divider separates them from
   *  the viewer's own controls; `#pruneDividers` removes it when the section
   *  contributed nothing. */
  #addExtras(parent, section, where) {
    const mine = this.#toolbarExtras.filter(
      (e) => (e.section ?? "right") === section && (e.placement ?? "end") === where
    );
    if (mine.length === 0) return;
    if (where === "end") this.#addDivider(parent);
    for (const e of mine) {
      const btn = this.#btn("lector-btn");
      btn.appendChild(this.#icon(e.icon));
      this.#tip(btn, e.tooltip);
      btn.dataset["action"] = e.id;
      const off = e.disabled?.() ?? false;
      if (off) {
        btn.setAttribute("disabled", "");
        btn.setAttribute("aria-disabled", "true");
      }
      btn.addEventListener("click", () => {
        if (e.disabled?.()) return;
        try {
          e.onSelect();
        } catch {
        }
      });
      parent.appendChild(btn);
    }
    if (where === "start") this.#addDivider(parent);
  }
  #addDropdown(parent, icon, tooltip, dropdownId) {
    const wrapper = this.#el("div", "lector-dropdown");
    const trigger = this.#btn("lector-btn");
    trigger.appendChild(this.#icon(icon));
    this.#tip(trigger, tooltip);
    const items = dropdownId === "dropdown-hamburger" ? this.#getHamburgerItems() : this.#getMoreItems();
    const menu = this.#el("div", "lector-dropdown__menu" + (dropdownId === "dropdown-more" ? " lector-dropdown__menu--right" : ""));
    for (const item of items) {
      if (item.type === "separator") {
        menu.appendChild(this.#el("div", "lector-dropdown__separator"));
        continue;
      }
      const btn = this.#btn("lector-dropdown__item");
      if (item.icon) {
        const ic = this.#el("span", "lector-dropdown__item-icon");
        ic.innerHTML = resolveIcon(item.icon) ?? "";
        btn.appendChild(ic);
      }
      const lbl = this.#el("span", "lector-dropdown__item-label");
      lbl.textContent = item.label ?? "";
      btn.appendChild(lbl);
      btn.addEventListener("click", () => {
        menu.classList.remove("lector-dropdown__menu--open");
        if (item.command) {
          try {
            void this.#engine.plugins.commands.execute(item.command);
          } catch {
          }
        }
      });
      menu.appendChild(btn);
    }
    trigger.addEventListener("click", (e) => {
      e.stopPropagation();
      this.#hideTooltip();
      menu.classList.toggle("lector-dropdown__menu--open");
    });
    const close = (e) => {
      if (!wrapper.contains(e.target)) menu.classList.remove("lector-dropdown__menu--open");
    };
    document.addEventListener("click", close);
    this.#pushCleanup(() => document.removeEventListener("click", close));
    this.#wireMenu(trigger, menu, "lector-dropdown__menu--open");
    wrapper.appendChild(trigger);
    wrapper.appendChild(menu);
    parent.appendChild(wrapper);
  }
  /**
   * Maps command IDs to permission keys. Commands not listed here
   * are always allowed.
   */
  static #COMMAND_PERMISSIONS = {
    "document.print": "canPrint",
    "document.export": "canExport",
    "document.save": "canExport",
    "document.screenshot": "canExport",
    "document.protect": "canEdit",
    "document.sign": "canSign",
    "annotation.mode-highlight": "canCreate",
    "annotation.mode-underline": "canCreate",
    "annotation.mode-strikeout": "canCreate",
    "annotation.mode-squiggly": "canCreate",
    "annotation.mode-freetext": "canCreate",
    "annotation.mode-sticky-note": "canCreate",
    "annotation.mode-ink": "canCreate",
    "annotation.mode-ink-highlighter": "canCreate",
    "annotation.mode-rectangle": "canCreate",
    "annotation.mode-circle": "canCreate",
    "annotation.mode-line": "canCreate",
    "annotation.mode-arrow": "canCreate",
    "annotation.mode-polygon": "canCreate",
    "annotation.mode-polyline": "canCreate",
    "annotation.mode-stamp": "canCreate",
    "annotation.mode-image": "canCreate",
    "annotation.mode-callout": "canCreate",
    "annotation.mode-insert-text": "canCreate",
    "annotation.mode-eraser": "canDelete",
    "annotation.mode-measurement": "canCreate",
    "redaction.mark": "canRedact",
    "redaction.apply": "canRedact"
  };
  /** Check if a command is allowed by the current document permissions. */
  #isCommandAllowed(commandId) {
    if (!commandId) return true;
    const permKey = _LectorViewer.#COMMAND_PERMISSIONS[commandId];
    if (!permKey) return true;
    const perms = this.#engine.permissions;
    if (!perms) return true;
    return perms[permKey] !== false;
  }
  /**
   * Filter a menu item list based on document permissions.
   * Removes disallowed items and cleans up orphaned/leading/trailing separators.
   */
  #filterByPermissions(items) {
    const filtered = items.filter(
      (item) => item.type === "separator" || this.#isCommandAllowed(item.command)
    );
    const cleaned = [];
    for (const item of filtered) {
      if (item.type === "separator") {
        if (cleaned.length === 0) continue;
        if (cleaned[cleaned.length - 1].type === "separator") continue;
      }
      cleaned.push(item);
    }
    if (cleaned.length > 0 && cleaned[cleaned.length - 1].type === "separator") {
      cleaned.pop();
    }
    return cleaned;
  }
  #getHamburgerItems() {
    return this.#filterByPermissions([
      { type: "item", label: this.#t("toolbar.open"), icon: "file-up", command: "document.open" },
      { type: "item", label: this.#t("toolbar.save"), icon: "save", command: "document.save" },
      { type: "item", label: this.#t("toolbar.close"), icon: "x", command: "document.close" },
      { type: "separator" },
      { type: "item", label: this.#t("toolbar.print"), icon: "printer", command: "document.print" },
      { type: "item", label: this.#t("toolbar.export"), icon: "download", command: "document.export" },
      { type: "item", label: this.#t("toolbar.screenshot"), icon: "camera", command: "document.screenshot" },
      { type: "separator" },
      { type: "item", label: this.#t("toolbar.passwordProtect"), icon: "shield", command: "document.protect" },
      { type: "item", label: this.#t("toolbar.signDocument"), icon: "pen-tool", command: "document.sign" },
      { type: "separator" },
      { type: "item", label: this.#t("toolbar.fullscreen"), icon: "fullscreen", command: "ui.fullscreen" }
    ]);
  }
  #getMoreItems() {
    const splitItems = this.isSplit ? [
      { type: "item", label: this.#t("split.closeAll") || "Close split panes", icon: "x", command: "viewer.close-extra-panes" }
    ] : [
      { type: "item", label: this.#t("split.horizontal") || "Split horizontally", icon: "columns", command: "viewer.split-horizontal" },
      { type: "item", label: this.#t("split.vertical") || "Split vertically", icon: "rows", command: "viewer.split-vertical" }
    ];
    return [
      { type: "item", label: this.#t("layout.single"), icon: "layout-single", command: "viewport.layout-single" },
      { type: "item", label: this.#t("layout.continuous"), icon: "layout-continuous", command: "viewport.layout-continuous" },
      { type: "item", label: this.#t("layout.double"), icon: "layout-double", command: "viewport.layout-double" },
      { type: "separator" },
      ...splitItems,
      { type: "separator" },
      { type: "item", label: this.#t("theme.light"), command: "ui.theme-light" },
      { type: "item", label: this.#t("theme.dark"), command: "ui.theme-dark" },
      { type: "item", label: this.#t("theme.system"), command: "ui.theme-system" },
      { type: "separator" },
      { type: "item", label: "English", command: "i18n.locale-en" },
      { type: "item", label: "Svenska", command: "i18n.locale-sv" },
      { type: "item", label: "Norsk bokm\xE5l", command: "i18n.locale-nb" },
      { type: "item", label: "Dansk", command: "i18n.locale-da" },
      { type: "item", label: "Suomi", command: "i18n.locale-fi" },
      { type: "item", label: "Espa\xF1ol", command: "i18n.locale-es" },
      { type: "item", label: "Deutsch", command: "i18n.locale-de" }
    ];
  }
  // ─── Annotation toolbar ─────────────────────────────────
  #buildAnnotToolbar() {
    this.#runSection("annot-toolbar", () => this.#buildAnnotToolbarImpl());
  }
  #buildAnnotToolbarImpl() {
    if (!this.#annotation) return;
    const perms = this.#engine.permissions;
    if (perms && perms.canCreate === false) return;
    this.#annotToolbar.innerHTML = "";
    this.#annotToolbar.appendChild(this.#el("div", "lector-toolbar__spacer"));
    this.#annotToolbar.appendChild(this.#buildAnnotToolGroup("highlighter", [
      { icon: "highlighter", tooltip: this.#t("annotation.highlight"), tool: "highlight" },
      { icon: "strikethrough", tooltip: this.#t("annotation.strikeout"), tool: "strikeout" },
      { icon: "underline-text", tooltip: this.#t("annotation.underline"), tool: "underline" },
      { icon: "wave", tooltip: this.#t("annotation.squiggly"), tool: "squiggly" }
    ]));
    this.#annotToolbar.appendChild(this.#el("div", "lector-toolbar__divider"));
    this.#annotToolbar.appendChild(this.#buildAnnotToolGroup("pencil", [
      { icon: "pencil", tooltip: this.#t("annotation.ink"), tool: "ink" },
      { icon: "pen-line", tooltip: this.#t("annotation.inkHighlighter"), tool: "ink-highlighter" },
      { icon: "eraser", tooltip: this.#t("annotation.eraser"), tool: "eraser" }
    ]));
    this.#annotToolbar.appendChild(this.#el("div", "lector-toolbar__divider"));
    this.#annotToolbar.appendChild(this.#buildAnnotToolGroup("type", [
      { icon: "type", tooltip: this.#t("annotation.freetext"), tool: "freetext" },
      { icon: "message-square", tooltip: this.#t("annotation.stickyNote"), tool: "sticky-note" },
      { icon: "callout", tooltip: this.#t("annotation.callout"), tool: "callout" }
    ]));
    this.#annotToolbar.appendChild(this.#el("div", "lector-toolbar__divider"));
    this.#annotToolbar.appendChild(this.#buildAnnotToolGroup("square-dashed", [
      { icon: "square-dashed", tooltip: this.#t("annotation.rectangle"), tool: "rectangle" },
      { icon: "circle-dashed", tooltip: this.#t("annotation.circle"), tool: "circle" },
      { icon: "line-tool", tooltip: this.#t("annotation.line"), tool: "line" },
      { icon: "arrow-tool", tooltip: this.#t("annotation.arrow"), tool: "arrow" },
      { icon: "polygon-tool", tooltip: this.#t("annotation.polygon"), tool: "polygon" },
      { icon: "polyline-tool", tooltip: this.#t("annotation.polyline"), tool: "polyline" }
    ]));
    this.#annotToolbar.appendChild(this.#el("div", "lector-toolbar__divider"));
    this.#annotToolbar.appendChild(this.#buildStampPicker());
    this.#annotToolbar.appendChild(this.#buildImageButton());
    this.#annotToolbar.appendChild(this.#el("div", "lector-toolbar__divider"));
    this.#annotToolbar.appendChild(this.#buildAnnotToolGroup("ruler", [
      { icon: "ruler", tooltip: this.#t("measurement.distance"), tool: "measure-distance" },
      { icon: "ruler-area", tooltip: this.#t("measurement.area"), tool: "measure-area" },
      { icon: "polyline-tool", tooltip: this.#t("measurement.perimeter"), tool: "measure-perimeter" }
    ]));
    if (this.#measurement) {
      const calBtn = this.#btn("lector-btn lector-annot-toolbar__calibrate");
      calBtn.appendChild(this.#icon("settings-2"));
      this.#tip(calBtn, this.#t("measurement.calibrate"));
      calBtn.addEventListener("click", () => {
        this.#showCalibrationDialog();
      });
      this.#annotToolbar.appendChild(calBtn);
    }
    this.#annotToolbar.appendChild(this.#el("div", "lector-toolbar__divider"));
    if (!perms || perms.canRedact !== false) {
      const redact = this.#el("div", "lector-toolbar__group");
      this.#addAnnotToolBtn(redact, "redact", this.#t("annotation.redaction"), "redaction");
      if (this.#redaction) {
        redact.appendChild(this.#buildApplyRedactionsBtn());
      }
      this.#annotToolbar.appendChild(redact);
    }
    if (this.#presets) {
      this.#annotToolbar.appendChild(this.#el("div", "lector-toolbar__divider"));
      this.#annotToolbar.appendChild(this.#buildPresetPicker());
    }
    this.#annotToolbar.appendChild(this.#el("div", "lector-toolbar__spacer"));
  }
  /**
   * Build the annotation-presets dropdown button. Same visual pattern as
   * `#buildStampPicker`: a primary button + chevron opening a panel of
   * preset chips. Each chip shows a colour swatch and the preset's label;
   * non-builtin presets get a delete affordance on hover. The panel
   * footer has a "save current style as preset…" action.
   */
  #buildPresetPicker() {
    const presets = this.#presets;
    if (!presets) return this.#el("div", "");
    const wrapper = this.#el("div", "lector-dropdown lector-preset-picker");
    const trigger = this.#btn("lector-btn lector-preset-picker__trigger");
    trigger.appendChild(this.#icon("palette"));
    const triggerLabel = this.#el("span", "lector-preset-picker__trigger-label");
    triggerLabel.textContent = this.#t("presets.none");
    trigger.appendChild(triggerLabel);
    this.#tip(trigger, this.#t("presets.tooltip"));
    const chevron = this.#btn("lector-preset-picker__chevron");
    chevron.setAttribute("aria-label", this.#t("presets.tooltip"));
    chevron.innerHTML = resolveIcon("chevron-down") ?? "";
    const menu = this.#el("div", "lector-dropdown__menu lector-preset-picker__menu");
    const renderMenu = () => {
      menu.innerHTML = "";
      const list = presets.presets.peek();
      const active = presets.activePreset.peek();
      if (list.length === 0) {
        const empty = this.#el("div", "lector-preset-picker__empty");
        empty.textContent = this.#t("presets.empty");
        menu.appendChild(empty);
      }
      for (const preset of list) {
        const item = this.#btn("lector-preset-picker__item");
        if (preset.name === active) item.classList.add("lector-preset-picker__item--active");
        const swatch = this.#el("span", "lector-preset-picker__swatch");
        if (preset.color) {
          const { r, g, b, a } = preset.color;
          swatch.style.background = `rgba(${r},${g},${b},${(a ?? 255) / 255})`;
        } else {
          swatch.classList.add("lector-preset-picker__swatch--none");
        }
        item.appendChild(swatch);
        const lbl = this.#el("span", "lector-preset-picker__label");
        lbl.textContent = preset.label ?? preset.name;
        item.appendChild(lbl);
        if (!preset.builtin) {
          const del = this.#el("span", "lector-preset-picker__delete");
          del.innerHTML = resolveIcon("x") ?? "";
          del.title = this.#t("common.delete");
          del.addEventListener("click", (e) => {
            e.stopPropagation();
            if (presets.deletePreset(preset.name)) {
              this.#showToast(this.#t("toast.presetDeleted"));
              renderMenu();
              updateTriggerLabel();
            }
          });
          item.appendChild(del);
        }
        item.addEventListener("click", (e) => {
          e.stopPropagation();
          if (presets.activePreset.peek() === preset.name) {
            presets.setActivePreset(null);
          } else {
            presets.setActivePreset(preset.name);
          }
          renderMenu();
          updateTriggerLabel();
          menu.classList.remove("lector-dropdown__menu--open");
        });
        menu.appendChild(item);
      }
      const footer = this.#el("div", "lector-preset-picker__footer");
      const saveBtn = this.#btn("lector-preset-picker__save");
      saveBtn.appendChild(this.#icon("plus"));
      const saveLabel = this.#el("span", "");
      saveLabel.textContent = this.#t("presets.saveCurrent");
      saveBtn.appendChild(saveLabel);
      saveBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        menu.classList.remove("lector-dropdown__menu--open");
        this.#showPresetNameDialog((trimmed) => {
          const created = presets.saveCurrentAsPreset(trimmed);
          if (created) {
            presets.setActivePreset(created.name);
            this.#showToast(this.#t("toast.presetSaved"));
            renderMenu();
            updateTriggerLabel();
          }
        });
      });
      footer.appendChild(saveBtn);
      menu.appendChild(footer);
    };
    const updateTriggerLabel = () => {
      const name = presets.activePreset.peek();
      if (name === null) {
        triggerLabel.textContent = this.#t("presets.none");
        trigger.classList.remove("lector-preset-picker__trigger--active");
      } else {
        const p = presets.getPreset(name);
        triggerLabel.textContent = p?.label ?? name;
        trigger.classList.add("lector-preset-picker__trigger--active");
      }
    };
    renderMenu();
    updateTriggerLabel();
    trigger.addEventListener("click", (e) => {
      e.stopPropagation();
      this.#hideTooltip();
      menu.classList.toggle("lector-dropdown__menu--open");
    });
    chevron.addEventListener("click", (e) => {
      e.stopPropagation();
      this.#hideTooltip();
      menu.classList.toggle("lector-dropdown__menu--open");
    });
    const close = (e) => {
      if (!wrapper.contains(e.target)) {
        menu.classList.remove("lector-dropdown__menu--open");
      }
    };
    document.addEventListener("click", close);
    this.#pushCleanup(() => document.removeEventListener("click", close));
    this.#wireMenu(chevron, menu, "lector-dropdown__menu--open");
    const offChanged = this.#engine.plugins.events.on("annotation-presets:changed", () => {
      renderMenu();
      for (const child of menu.children) {
        if (child.tagName === "BUTTON" && !child.getAttribute("role")) {
          child.setAttribute("role", "menuitem");
          child.tabIndex = -1;
        }
      }
      updateTriggerLabel();
    });
    const offActive = this.#engine.plugins.events.on("annotation-presets:active-changed", () => {
      renderMenu();
      for (const child of menu.children) {
        if (child.tagName === "BUTTON" && !child.getAttribute("role")) {
          child.setAttribute("role", "menuitem");
          child.tabIndex = -1;
        }
      }
      updateTriggerLabel();
    });
    this.#pushCleanup(offChanged, offActive);
    const triggerWrap = this.#el("div", "lector-preset-picker__triggerWrap");
    triggerWrap.appendChild(trigger);
    triggerWrap.appendChild(chevron);
    wrapper.appendChild(triggerWrap);
    wrapper.appendChild(menu);
    return wrapper;
  }
  /**
   * Show a custom modal asking the user to name a preset, then call
   * `onConfirm` with the trimmed name. Used by the preset picker's
   * "Save current style…" footer action.
   *
   * Replaces an earlier `window.prompt` because native dialogs are
   * blocked in some embed contexts (sandboxed iframes, fullscreen
   * compositors) and don't honour the viewer's theme.
   */
  #showPresetNameDialog(onConfirm) {
    const overlay = this.#el("div", "lector-modal-overlay");
    const dialog = this.#el("div", "lector-modal");
    const title = this.#el("h3", "lector-modal__title");
    title.textContent = this.#t("presets.dialogTitle");
    dialog.appendChild(title);
    const body = this.#el("div", "lector-modal__body");
    const label = this.#el("label", "lector-cal-dialog__label");
    label.textContent = this.#t("presets.namePrompt");
    body.appendChild(label);
    const input = document.createElement("input");
    input.type = "text";
    input.className = "lector-modal__input";
    input.placeholder = this.#t("presets.namePlaceholder");
    input.style.width = "100%";
    body.appendChild(input);
    const errorEl = this.#el("div", "lector-modal__error");
    errorEl.id = "lector-preset-error";
    errorEl.setAttribute("aria-live", "assertive");
    errorEl.style.display = "none";
    errorEl.style.color = "var(--lector-danger, rgb(220, 38, 38))";
    errorEl.style.fontSize = "12px";
    errorEl.style.marginTop = "6px";
    input.setAttribute("aria-describedby", "lector-preset-error");
    body.appendChild(errorEl);
    dialog.appendChild(body);
    const actions = this.#el("div", "lector-modal__actions");
    const cancelBtn = this.#btn("lector-modal__btn");
    cancelBtn.textContent = this.#t("common.cancel");
    cancelBtn.addEventListener("click", () => overlay.remove());
    const saveBtn = this.#btn("lector-modal__btn lector-modal__btn--primary");
    saveBtn.textContent = this.#t("common.save");
    const submit = () => {
      const trimmed = input.value.trim();
      if (trimmed.length === 0) {
        errorEl.textContent = this.#t("presets.nameRequired");
        errorEl.style.display = "";
        input.setAttribute("aria-invalid", "true");
        input.focus();
        return;
      }
      overlay.remove();
      onConfirm(trimmed);
    };
    saveBtn.addEventListener("click", submit);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        submit();
      }
    });
    actions.appendChild(cancelBtn);
    actions.appendChild(saveBtn);
    dialog.appendChild(actions);
    overlay.appendChild(dialog);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.remove();
    });
    this.#openModal(overlay, dialog, null, input);
  }
  /**
   * Build a tool group dropdown: a primary button (showing the last-used tool's icon)
   * with a chevron that opens a dropdown panel with all tools in the group.
   * Clicking the primary button activates the last-used tool directly.
   * Clicking a tool in the dropdown activates it and updates the primary icon.
   */
  #buildAnnotToolGroup(defaultIcon, tools) {
    const group = this.#el("div", "lector-tool-group");
    let activeIcon = defaultIcon;
    const trigger = this.#btn("lector-tool-group__trigger");
    const iconSpan = this.#icon(activeIcon);
    trigger.appendChild(iconSpan);
    trigger.dataset["annotToolGroup"] = tools[0].tool;
    this.#tip(trigger, tools[0].tooltip);
    const chevronBtn = this.#btn("lector-tool-group__chevron");
    chevronBtn.setAttribute("aria-label", this.#t("toolbar.moreTools"));
    chevronBtn.innerHTML = resolveIcon("chevron-down") ?? "";
    chevronBtn.tabIndex = -1;
    const panel = this.#el("div", "lector-tool-group__panel");
    for (const t of tools) {
      const btn = this.#btn("lector-btn");
      btn.appendChild(this.#icon(t.icon));
      this.#tip(btn, t.tooltip);
      btn.dataset["annotTool"] = t.tool;
      btn.addEventListener("click", () => {
        if (!this.#annotation) return;
        this.#annotation.setActiveTool(t.tool);
        activeIcon = t.icon;
        iconSpan.innerHTML = resolveIcon(t.icon) ?? "";
        trigger.dataset["annotToolGroup"] = t.tool;
        trigger.setAttribute("aria-label", t.tooltip);
        panel.classList.remove("lector-tool-group__panel--open");
      });
      panel.appendChild(btn);
    }
    const closeAllPanels = () => {
      for (const p of this.#annotToolbar.querySelectorAll(".lector-tool-group__panel--open")) {
        p.classList.remove("lector-tool-group__panel--open");
      }
    };
    trigger.addEventListener("click", (e) => {
      if (!this.#annotation) return;
      e.stopPropagation();
      closeAllPanels();
      const currentTool = trigger.dataset["annotToolGroup"];
      const current = this.#annotation.activeTool.peek();
      if (current === currentTool) {
        this.#annotation.setActiveTool(null);
      } else {
        this.#annotation.setActiveTool(currentTool);
      }
    });
    chevronBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const wasOpen = panel.classList.contains("lector-tool-group__panel--open");
      closeAllPanels();
      if (!wasOpen) {
        panel.classList.add("lector-tool-group__panel--open");
      }
    });
    const close = (e) => {
      if (!group.contains(e.target)) {
        panel.classList.remove("lector-tool-group__panel--open");
      }
    };
    document.addEventListener("click", close);
    this.#pushCleanup(() => document.removeEventListener("click", close));
    this.#wireMenu(chevronBtn, panel, "lector-tool-group__panel--open");
    group.appendChild(trigger);
    group.appendChild(chevronBtn);
    group.appendChild(panel);
    return group;
  }
  #addAnnotToolBtn(parent, icon, tooltip, tool) {
    const btn = this.#btn("lector-btn");
    btn.appendChild(this.#icon(icon));
    this.#tip(btn, tooltip);
    btn.dataset["annotTool"] = tool;
    btn.addEventListener("click", () => {
      if (!this.#annotation) return;
      const current = this.#annotation.activeTool.peek();
      if (current === tool) {
        this.#annotation.setActiveTool(null);
      } else {
        this.#annotation.setActiveTool(tool);
      }
    });
    parent.appendChild(btn);
  }
  /**
   * Build the "Apply Redactions" toolbar button. Shows a count of
   * pending (unapplied) redaction marks, disables when zero, opens
   * the bulk-apply confirmation dialog on click. Reactively refreshes
   * on annotation create/delete/update and on redaction:applied.
   */
  #buildApplyRedactionsBtn() {
    const btn = this.#btn("lector-btn lector-btn--apply-redactions");
    const label = this.#el("span", "lector-btn__label");
    btn.appendChild(label);
    this.#tip(btn, this.#t("redact.applyButton"));
    const refresh = () => {
      if (!this.#redaction) return;
      const doc = this.#document.activeDocument.peek();
      const count = doc ? this.#redaction.getMarkedRedactions(doc.id).length : 0;
      label.textContent = count > 0 ? `${this.#t("redact.applyButton")} (${count})` : this.#t("redact.applyButton");
      btn.disabled = count === 0;
      btn.style.display = count === 0 ? "none" : "";
    };
    btn.addEventListener("click", () => {
      this.#showApplyRedactionsDialog();
    });
    const events = this.#engine.plugins.events;
    const off1 = events.on("annotation:created", refresh);
    const off2 = events.on("annotation:deleted", refresh);
    const off3 = events.on("annotation:updated", refresh);
    const off4 = events.on("redaction:applied", refresh);
    const off5 = events.on("annotation:page-loaded", refresh);
    const off6 = this.#document.activeDocument.subscribe(refresh);
    this.#pushCleanup(off1, off2, off3, off4, off5, off6);
    refresh();
    return btn;
  }
  // ─── Annotation Properties Popover ─────────────────────
  #showAnnotPopover(annotId) {
    this.#annotPopoverAnnotId = annotId;
    if (!this.#annotation) return;
    let doc = null;
    let tracked;
    for (const id of this.#allOpenDocIds()) {
      const handle = this.#document.getHandle(id);
      if (!handle) continue;
      const found = this.#annotation.getForDocument(id).find((t) => t.id === annotId);
      if (found) {
        doc = handle;
        tracked = found;
        break;
      }
    }
    if (!doc || !tracked) {
      this.#hideAnnotPopover();
      return;
    }
    if (tracked.data.subtype === 1) {
      this.#hideAnnotPopover();
      return;
    }
    const annot = tracked.data;
    const pop = this.#annotPopover;
    pop.innerHTML = "";
    if (annot.subtype === 28 || annot.tag === "redaction") {
      const actions2 = this.#el("div", "lector-annot-popover__actions");
      actions2.style.marginTop = "0";
      actions2.style.borderTop = "none";
      const delBtn2 = this.#btn("lector-annot-popover__delete");
      delBtn2.textContent = this.#t("annotation.delete");
      delBtn2.addEventListener("click", () => {
        void this.#annotation.delete(doc.id, annotId);
        this.#hideAnnotPopover();
      });
      actions2.appendChild(delBtn2);
      pop.appendChild(actions2);
      pop.style.display = "";
      pop.style.visibility = "hidden";
      pop.style.left = "-9999px";
      pop.style.top = "-9999px";
      pop.offsetHeight;
      this.#positionAnnotPopover(annot, doc);
      pop.style.visibility = "";
      return;
    }
    const COLORS = [
      { n: "Red", r: 228, g: 66, b: 52 },
      { n: "Orange", r: 245, g: 158, b: 11 },
      { n: "Yellow", r: 255, g: 205, b: 69 },
      { n: "Green", r: 34, g: 197, b: 94 },
      { n: "Blue", r: 59, g: 130, b: 246 },
      { n: "Purple", r: 139, g: 92, b: 246 },
      { n: "Black", r: 0, g: 0, b: 0 },
      { n: "White", r: 255, g: 255, b: 255 }
    ];
    const buildColorRow = (label, currentColor, onSelect, showNone) => {
      const lbl = this.#el("div", "lector-annot-popover__label");
      lbl.textContent = label;
      pop.appendChild(lbl);
      const row = this.#el("div", "lector-annot-popover__colors");
      if (showNone) {
        const noneBtn = this.#btn("lector-annot-popover__color lector-annot-popover__color--none");
        noneBtn.title = this.#t("misc.none");
        if (!currentColor || currentColor.a === 0) noneBtn.classList.add("lector-annot-popover__color--active");
        noneBtn.addEventListener("click", () => {
          onSelect({ r: 0, g: 0, b: 0, a: 0 });
        });
        row.appendChild(noneBtn);
      }
      for (const c of COLORS) {
        const btn = this.#btn("lector-annot-popover__color");
        btn.style.background = `rgb(${c.r}, ${c.g}, ${c.b})`;
        if (c.n === "White") btn.style.border = "1px solid var(--lector-border)";
        if (currentColor && currentColor.a > 0 && currentColor.r === c.r && currentColor.g === c.g && currentColor.b === c.b) {
          btn.classList.add("lector-annot-popover__color--active");
        }
        btn.addEventListener("click", () => {
          onSelect({ r: c.r, g: c.g, b: c.b, a: 255 });
        });
        row.appendChild(btn);
      }
      pop.appendChild(row);
    };
    buildColorRow(this.#t("annotation.color"), annot.color, (c) => {
      void this.#annotation.update(doc.id, annotId, { color: c });
    }, true);
    const subtype = annot.subtype;
    const isFreeText = subtype === 3 && annot.freeText !== void 0;
    const isLineAnnot = subtype === 4 || annot.tag === "line" || annot.tag === "arrow" || annot.tag === "arrow-start" || annot.tag === "arrow-both";
    const hasInterior = (subtype === 5 || subtype === 6) && !isLineAnnot;
    if (isFreeText) {
      const currentFreeText = () => {
        const t = this.#annotation.getForDocument(doc.id).find((a) => a.id === annotId);
        return t?.data.freeText ?? annot.freeText;
      };
      const fc = annot.freeText.fontColor;
      buildColorRow(this.#t("annotation.fontColor"), fc ? { ...fc, a: 255 } : void 0, (c) => {
        void this.#annotation.update(doc.id, annotId, {
          freeText: { ...currentFreeText(), fontColor: { r: c.r, g: c.g, b: c.b } }
        });
      }, false);
      const fsLabel = this.#el("div", "lector-annot-popover__label");
      fsLabel.textContent = this.#t("annotation.fontSize");
      pop.appendChild(fsLabel);
      const currentFs = annot.freeText.fontSize || 14;
      const fsRow = this.#el("div", "lector-annot-popover__slider-row");
      const fsSlider = document.createElement("input");
      fsSlider.type = "range";
      fsSlider.className = "lector-annot-popover__slider";
      fsSlider.setAttribute("aria-label", this.#t("annotation.fontSize"));
      fsSlider.min = "8";
      fsSlider.max = "48";
      fsSlider.step = "1";
      fsSlider.value = String(Math.round(currentFs));
      const fsValue = this.#el("span", "lector-annot-popover__slider-value");
      fsValue.textContent = `${Math.round(currentFs)}pt`;
      fsSlider.addEventListener("input", () => {
        const fs = parseInt(fsSlider.value, 10);
        fsValue.textContent = `${fs}pt`;
        void this.#annotation.update(doc.id, annotId, {
          freeText: { ...currentFreeText(), fontSize: fs }
        });
      });
      fsRow.appendChild(fsSlider);
      fsRow.appendChild(fsValue);
      pop.appendChild(fsRow);
      const alignLabel = this.#el("div", "lector-annot-popover__label");
      alignLabel.textContent = this.#t("annotation.textAlign");
      pop.appendChild(alignLabel);
      const alignRow = this.#el("div", "lector-annot-popover__widths");
      const currentAlign = annot.freeText.textAlign ?? "left";
      for (const al of [
        { key: "left", icon: "align-left", label: this.#t("annotation.alignLeft") },
        { key: "center", icon: "align-center", label: this.#t("annotation.alignCenter") },
        { key: "right", icon: "align-right", label: this.#t("annotation.alignRight") }
      ]) {
        const btn = this.#btn("lector-annot-popover__width");
        btn.innerHTML = resolveIcon(al.icon) ?? al.label;
        btn.title = al.label;
        if (currentAlign === al.key) btn.classList.add("lector-annot-popover__width--active");
        btn.addEventListener("click", () => {
          void this.#annotation.update(doc.id, annotId, {
            freeText: { ...currentFreeText(), textAlign: al.key }
          });
        });
        alignRow.appendChild(btn);
      }
      pop.appendChild(alignRow);
    }
    if (hasInterior) {
      buildColorRow(this.#t("annotation.fill"), annot.interiorColor, (c) => {
        void this.#annotation.update(doc.id, annotId, { interiorColor: c });
      }, true);
    }
    const hasBorder = !isLineAnnot && (subtype === 5 || subtype === 6 || subtype === 15);
    if (hasBorder) {
      const wLabel = this.#el("div", "lector-annot-popover__label");
      wLabel.textContent = this.#t("annotation.borderWidth");
      pop.appendChild(wLabel);
      const currentW = annot.border?.width ?? 2;
      const wRow = this.#el("div", "lector-annot-popover__slider-row");
      const wSlider = document.createElement("input");
      wSlider.type = "range";
      wSlider.className = "lector-annot-popover__slider";
      wSlider.setAttribute("aria-label", this.#t("annotation.borderWidth"));
      wSlider.min = "1";
      wSlider.max = "10";
      wSlider.step = "1";
      wSlider.value = String(Math.round(currentW));
      const wValue = this.#el("span", "lector-annot-popover__slider-value");
      wValue.textContent = `${Math.round(currentW)}px`;
      wSlider.addEventListener("input", () => {
        const w = parseInt(wSlider.value, 10);
        wValue.textContent = `${w}px`;
        void this.#annotation.update(doc.id, annotId, {
          border: { horizontalRadius: 0, verticalRadius: 0, width: w }
        });
      });
      wRow.appendChild(wSlider);
      wRow.appendChild(wValue);
      pop.appendChild(wRow);
    }
    if (isLineAnnot) {
      const styleLabel = this.#el("div", "lector-annot-popover__label");
      styleLabel.textContent = this.#t("annotation.lineStyle");
      pop.appendChild(styleLabel);
      const styleRow = this.#el("div", "lector-annot-popover__widths");
      const currentDash = annot.border?.horizontalRadius ?? 0;
      for (const ds of [
        { label: this.#t("lineStyle.solid"), dash: 0, svg: "M2 12h20" },
        { label: this.#t("lineStyle.dashed"), dash: 1, svg: "M2 12h4M10 12h4M18 12h4" },
        { label: this.#t("lineStyle.dotted"), dash: 2, svg: "M3 12h1M7 12h1M11 12h1M15 12h1M19 12h1" }
      ]) {
        const btn = this.#btn("lector-annot-popover__width");
        const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        icon.setAttribute("viewBox", "0 0 24 24");
        icon.setAttribute("width", "24");
        icon.setAttribute("height", "12");
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", ds.svg);
        path.setAttribute("stroke", "currentColor");
        path.setAttribute("stroke-width", "2");
        path.setAttribute("stroke-linecap", "round");
        path.setAttribute("fill", "none");
        icon.appendChild(path);
        btn.appendChild(icon);
        btn.title = ds.label;
        if (currentDash === ds.dash) btn.classList.add("lector-annot-popover__width--active");
        btn.addEventListener("click", () => {
          void this.#annotation.update(doc.id, annotId, {
            border: {
              horizontalRadius: ds.dash,
              verticalRadius: annot.border?.verticalRadius ?? 0,
              width: annot.border?.width ?? 2
            }
          });
        });
        styleRow.appendChild(btn);
      }
      pop.appendChild(styleRow);
      const thickLabel = this.#el("div", "lector-annot-popover__label");
      thickLabel.textContent = this.#t("annotation.thickness");
      pop.appendChild(thickLabel);
      const currentThick = annot.border?.width ?? 2;
      const thickRow = this.#el("div", "lector-annot-popover__slider-row");
      const thickSlider = document.createElement("input");
      thickSlider.type = "range";
      thickSlider.className = "lector-annot-popover__slider";
      thickSlider.setAttribute("aria-label", this.#t("annotation.thickness"));
      thickSlider.min = "1";
      thickSlider.max = "10";
      thickSlider.step = "1";
      thickSlider.value = String(Math.round(currentThick));
      const thickValue = this.#el("span", "lector-annot-popover__slider-value");
      thickValue.textContent = `${Math.round(currentThick)}px`;
      thickSlider.addEventListener("input", () => {
        const w = parseInt(thickSlider.value, 10);
        thickValue.textContent = `${w}px`;
        void this.#annotation.update(doc.id, annotId, {
          border: {
            horizontalRadius: annot.border?.horizontalRadius ?? 0,
            verticalRadius: annot.border?.verticalRadius ?? 0,
            width: w
          }
        });
      });
      thickRow.appendChild(thickSlider);
      thickRow.appendChild(thickValue);
      pop.appendChild(thickRow);
    }
    {
      const isStampOrImage = annot.subtype === 13;
      const opLabel = this.#el("div", "lector-annot-popover__label");
      opLabel.textContent = this.#t("annotation.opacity");
      pop.appendChild(opLabel);
      const currentOp = isStampOrImage ? Math.round((annot.opacity ?? 1) * 100) : annot.color ? Math.round(annot.color.a / 255 * 100) : 100;
      const opRow = this.#el("div", "lector-annot-popover__slider-row");
      const opSlider = document.createElement("input");
      opSlider.type = "range";
      opSlider.className = "lector-annot-popover__slider";
      opSlider.setAttribute("aria-label", this.#t("annotation.opacity"));
      opSlider.min = "10";
      opSlider.max = "100";
      opSlider.step = "5";
      opSlider.value = String(currentOp);
      const opValue = this.#el("span", "lector-annot-popover__slider-value");
      opValue.textContent = `${currentOp}%`;
      opSlider.addEventListener("input", () => {
        const pct = parseInt(opSlider.value, 10);
        opValue.textContent = `${pct}%`;
        if (isStampOrImage) {
          void this.#annotation.update(doc.id, annotId, {
            opacity: pct / 100
          });
        } else if (annot.color) {
          void this.#annotation.update(doc.id, annotId, {
            color: { ...annot.color, a: Math.round(pct / 100 * 255) }
          });
        }
      });
      opRow.appendChild(opSlider);
      opRow.appendChild(opValue);
      pop.appendChild(opRow);
    }
    if (annot.measurement) {
      const measLabel = this.#el("div", "lector-annot-popover__label");
      measLabel.textContent = this.#t("measurement.unit");
      pop.appendChild(measLabel);
      const unitRow = this.#el("div", "lector-annot-popover__widths");
      const UNITS_FOR_POPOVER = [
        MeasurementUnit.MM,
        MeasurementUnit.CM,
        MeasurementUnit.M,
        MeasurementUnit.IN,
        MeasurementUnit.FT,
        MeasurementUnit.YD
      ];
      for (const u of UNITS_FOR_POPOVER) {
        const btn = this.#btn("lector-annot-popover__width");
        btn.textContent = u;
        btn.style.fontSize = "var(--lector-font-size-sm)";
        btn.style.minWidth = "32px";
        if (annot.measurement.unit === u) {
          btn.classList.add("lector-annot-popover__width--active");
        }
        btn.addEventListener("click", () => {
          if (!annot.measurement) return;
          void this.#annotation.update(doc.id, annotId, {
            measurement: { ...annot.measurement, unit: u }
          });
        });
        unitRow.appendChild(btn);
      }
      pop.appendChild(unitRow);
      const precLabel = this.#el("div", "lector-annot-popover__label");
      precLabel.textContent = this.#t("measurement.precision");
      pop.appendChild(precLabel);
      const precRow = this.#el("div", "lector-annot-popover__widths");
      for (let i = 0; i <= 4; i++) {
        const btn = this.#btn("lector-annot-popover__width");
        btn.textContent = String(i);
        btn.style.fontSize = "var(--lector-font-size-sm)";
        btn.style.minWidth = "28px";
        if ((annot.measurement.precision ?? 2) === i) {
          btn.classList.add("lector-annot-popover__width--active");
        }
        const p = i;
        btn.addEventListener("click", () => {
          if (!annot.measurement) return;
          void this.#annotation.update(doc.id, annotId, {
            measurement: { ...annot.measurement, precision: p }
          });
        });
        precRow.appendChild(btn);
      }
      pop.appendChild(precRow);
    }
    const actions = this.#el("div", "lector-annot-popover__actions");
    const frontBtn = this.#btn("lector-btn lector-annot-popover__action");
    frontBtn.appendChild(this.#icon("bring-to-front"));
    this.#tip(frontBtn, this.#t("annotation.bringToFront"));
    frontBtn.addEventListener("click", () => {
      void this.#annotation.bringToFront(doc.id, annotId);
    });
    actions.appendChild(frontBtn);
    const backBtn = this.#btn("lector-btn lector-annot-popover__action");
    backBtn.appendChild(this.#icon("send-to-back"));
    this.#tip(backBtn, this.#t("annotation.sendToBack"));
    backBtn.addEventListener("click", () => {
      void this.#annotation.sendToBack(doc.id, annotId);
    });
    actions.appendChild(backBtn);
    if (annot.groupId) {
      const ungroupBtn = this.#btn("lector-btn lector-annot-popover__action");
      ungroupBtn.appendChild(this.#icon("ungroup"));
      this.#tip(ungroupBtn, this.#t("annotation.ungroup"));
      ungroupBtn.addEventListener("click", () => {
        void this.#annotation.ungroupAnnotations(doc.id, annot.groupId);
      });
      actions.appendChild(ungroupBtn);
    }
    const delBtn = this.#btn("lector-annot-popover__delete");
    delBtn.textContent = this.#t("annotation.delete");
    delBtn.addEventListener("click", () => {
      void this.#annotation.delete(doc.id, annotId);
      this.#hideAnnotPopover();
    });
    actions.appendChild(delBtn);
    pop.appendChild(actions);
    pop.style.display = "";
    pop.style.visibility = "hidden";
    pop.style.left = "-9999px";
    pop.style.top = "-9999px";
    pop.offsetHeight;
    this.#positionAnnotPopover(annot, doc);
    pop.style.visibility = "";
  }
  #positionAnnotPopover(annot, doc) {
    const pane = this.#findPaneForDoc(doc.id);
    const sourceViewport = pane.viewport;
    const sourceCanvas = pane.canvas;
    const pos = sourceViewport.pagePositions.peek().find((p) => p.pageIndex === annot.pageIndex);
    if (!pos) return;
    const scale = sourceViewport.scale.peek();
    const pageH = doc.pageSizes[annot.pageIndex]?.height ?? 792;
    const canvasWrap = this.#canvasWrap;
    let pdfLeft = annot.rect.left;
    let pdfRight = annot.rect.right;
    let pdfTop = Math.max(annot.rect.top, annot.rect.bottom);
    let pdfBottom = Math.min(annot.rect.top, annot.rect.bottom);
    if (annot.ink && annot.ink.strokes.length > 0) {
      let sMinX = Infinity, sMinY = Infinity, sMaxX = -Infinity, sMaxY = -Infinity;
      for (const stroke of annot.ink.strokes) {
        for (const p of stroke) {
          if (p.x < sMinX) sMinX = p.x;
          if (p.y < sMinY) sMinY = p.y;
          if (p.x > sMaxX) sMaxX = p.x;
          if (p.y > sMaxY) sMaxY = p.y;
        }
      }
      pdfLeft = sMinX;
      pdfRight = sMaxX;
      pdfTop = sMaxY;
      pdfBottom = sMinY;
    }
    const wrapRect = canvasWrap.getBoundingClientRect();
    const canvasRect = sourceCanvas.getBoundingClientRect();
    const canvasOffsetX = canvasRect.left - wrapRect.left;
    const canvasOffsetY = canvasRect.top - wrapRect.top;
    const vxInCanvas = pos.x + pdfLeft * scale - sourceCanvas.scrollLeft;
    const vyInCanvas = pos.y + (pageH - pdfTop) * scale - sourceCanvas.scrollTop;
    const vx = canvasOffsetX + vxInCanvas;
    const vy = canvasOffsetY + vyInCanvas;
    const vw = (pdfRight - pdfLeft) * scale;
    const vh = (pdfTop - pdfBottom) * scale;
    const popW = this.#annotPopover.offsetWidth || 220;
    const popH = this.#annotPopover.offsetHeight || 200;
    const paneLeft = canvasOffsetX;
    const paneTop = canvasOffsetY;
    const paneW = sourceCanvas.clientWidth;
    const paneH = sourceCanvas.clientHeight;
    const paneRight = paneLeft + paneW;
    const paneBottom = paneTop + paneH;
    const margin = 6;
    const gap = 8;
    const spaceRight = paneRight - (vx + vw + gap);
    const spaceLeft = vx - paneLeft - gap;
    const spaceBelow = paneBottom - (vy + vh + gap);
    const spaceAbove = vy - paneTop - gap;
    let left;
    let top;
    if (spaceRight >= popW) {
      left = vx + vw + gap;
      top = vy;
    } else if (spaceLeft >= popW) {
      left = vx - gap - popW;
      top = vy;
    } else if (spaceBelow >= popH) {
      left = vx + vw / 2 - popW / 2;
      top = vy + vh + gap;
    } else if (spaceAbove >= popH) {
      left = vx + vw / 2 - popW / 2;
      top = vy - gap - popH;
    } else {
      if (spaceRight >= spaceLeft) {
        left = paneRight - popW - margin;
        top = vy;
      } else {
        left = paneLeft + margin;
        top = vy;
      }
    }
    left = Math.max(paneLeft + margin, Math.min(left, paneRight - popW - margin));
    top = Math.max(paneTop + margin, Math.min(top, paneBottom - popH - margin));
    this.#annotPopover.style.left = `${left}px`;
    this.#annotPopover.style.top = `${top}px`;
  }
  #hideAnnotPopover() {
    this.#annotPopoverAnnotId = null;
    this.#annotPopover.style.display = "none";
    this.#annotPopover.innerHTML = "";
  }
  /** Reposition the popover if it's visible — called on scroll/zoom/layout changes. */
  #repositionAnnotPopover() {
    const annotId = this.#annotPopoverAnnotId;
    if (!annotId || !this.#annotation) return;
    let annotData = null;
    let doc = null;
    for (const docId of this.#allOpenDocIds()) {
      const found = this.#annotation.getForDocument(docId).find((t) => t.id === annotId);
      if (found) {
        annotData = found.data;
        doc = this.#document.getHandle(docId) ?? null;
        break;
      }
    }
    if (!annotData || !doc) return;
    const visiblePages = this.#viewportInstance.visiblePages.peek();
    if (!visiblePages.includes(annotData.pageIndex)) {
      this.#annotPopover.style.display = "none";
      return;
    }
    this.#annotPopover.style.display = "";
    this.#positionAnnotPopover(annotData, doc);
  }
  // ─── Multi-select action bar ──────────────────────────
  /**
   * Build (lazily) and show the floating action bar for a multi-
   * selection of annotations. Anchored to the bottom-centre of the
   * canvas wrapper. Updates its content on every call so the count
   * label and the Group/Ungroup choice stay in sync as the user
   * shift-clicks more annotations.
   *
   * The bar offers:
   *   - Group: only enabled when none of the selected annotations are
   *     already grouped (or all are in the same group), so we can
   *     create a single new group atomically.
   *   - Ungroup: only enabled when at least one selected annotation
   *     belongs to a group; ungroups every distinct group ID in one
   *     pass.
   *   - Delete: removes every selected annotation.
   */
  #showMultiSelectBar(ids) {
    if (!this.#annotation) return;
    this.#hideAnnotPopover();
    if (!this.#multiSelectBar) {
      const bar2 = this.#el("div", "lector-multi-select-bar");
      this.#canvasWrap.appendChild(bar2);
      this.#multiSelectBar = bar2;
    }
    const bar = this.#multiSelectBar;
    bar.innerHTML = "";
    bar.style.display = "";
    const owned = [];
    for (const id of ids) {
      for (const docId of this.#allOpenDocIds()) {
        const found = this.#annotation.getForDocument(docId).find((t) => t.id === id);
        if (found) {
          owned.push({ docId, id, data: found.data });
          break;
        }
      }
    }
    if (owned.length === 0) {
      bar.style.display = "none";
      return;
    }
    const count = this.#el("span", "lector-multi-select-bar__count");
    count.textContent = this.#t("annotation.multiSelectCount", { count: owned.length });
    bar.appendChild(count);
    const sep = this.#el("span", "lector-multi-select-bar__sep");
    bar.appendChild(sep);
    const anyAlreadyGrouped = owned.some((o) => o.data.groupId !== void 0);
    const groupBtn = this.#btn("lector-btn lector-multi-select-bar__btn");
    groupBtn.appendChild(this.#icon("group"));
    const groupLabel = this.#el("span", "lector-btn__label");
    groupLabel.textContent = this.#t("annotation.group");
    groupBtn.appendChild(groupLabel);
    groupBtn.disabled = anyAlreadyGrouped || owned.length < 2;
    groupBtn.addEventListener("click", () => {
      const byDoc = /* @__PURE__ */ new Map();
      for (const o of owned) {
        const list = byDoc.get(o.docId) ?? [];
        list.push(o.id);
        byDoc.set(o.docId, list);
      }
      for (const [docId, idList] of byDoc) {
        if (idList.length >= 2) {
          void this.#annotation.groupAnnotations(docId, idList);
        }
      }
    });
    bar.appendChild(groupBtn);
    const ungroupBtn = this.#btn("lector-btn lector-multi-select-bar__btn");
    ungroupBtn.appendChild(this.#icon("ungroup"));
    const ungroupLabel = this.#el("span", "lector-btn__label");
    ungroupLabel.textContent = this.#t("annotation.ungroup");
    ungroupBtn.appendChild(ungroupLabel);
    ungroupBtn.disabled = !anyAlreadyGrouped;
    ungroupBtn.addEventListener("click", () => {
      const byDoc = /* @__PURE__ */ new Map();
      for (const o of owned) {
        if (!o.data.groupId) continue;
        let s = byDoc.get(o.docId);
        if (!s) {
          s = /* @__PURE__ */ new Set();
          byDoc.set(o.docId, s);
        }
        s.add(o.data.groupId);
      }
      for (const [docId, groupIds] of byDoc) {
        for (const gid of groupIds) {
          void this.#annotation.ungroupAnnotations(docId, gid);
        }
      }
    });
    bar.appendChild(ungroupBtn);
    const delBtn = this.#btn("lector-btn lector-multi-select-bar__btn lector-multi-select-bar__btn--danger");
    delBtn.appendChild(this.#icon("trash"));
    const delLabel = this.#el("span", "lector-btn__label");
    delLabel.textContent = this.#t("annotation.delete");
    delBtn.appendChild(delLabel);
    delBtn.addEventListener("click", () => {
      for (const o of owned) {
        void this.#annotation.delete(o.docId, o.id);
      }
      this.#annotation.clearAnnotationSelection();
    });
    bar.appendChild(delBtn);
  }
  #hideMultiSelectBar() {
    if (this.#multiSelectBar) {
      this.#multiSelectBar.style.display = "none";
      this.#multiSelectBar.innerHTML = "";
    }
  }
  // ─── Color Picker ──────────────────────────────────────
  // ─── Zoom control ─────────────────────────────────────
  #buildZoomControl() {
    const wrap3 = this.#el("div", "lector-zoom");
    const valueWrap = this.#el("div", "lector-zoom__value");
    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = "numeric";
    input.className = "lector-zoom__input";
    input.setAttribute("aria-label", this.#t("toolbar.zoom"));
    input.value = "100";
    const pct = this.#el("span", "lector-zoom__pct");
    pct.textContent = "%";
    valueWrap.appendChild(input);
    valueWrap.appendChild(pct);
    const chevronWrap = this.#el("div", "lector-dropdown");
    chevronWrap.style.display = "flex";
    chevronWrap.style.alignItems = "center";
    chevronWrap.style.height = "100%";
    const chevron = this.#btn("lector-zoom__chevron");
    chevron.setAttribute("aria-label", this.#t("toolbar.zoomPresets"));
    chevron.innerHTML = resolveIcon("chevron-down") ?? "";
    const zoomMenu = this.#el("div", "lector-dropdown__menu");
    const presets = [25, 50, 75, 100, 125, 150, 200, 400];
    for (const pv of presets) {
      const item = this.#btn("lector-dropdown__item");
      const lbl = this.#el("span", "lector-dropdown__item-label");
      lbl.textContent = `${pv}%`;
      item.appendChild(lbl);
      item.addEventListener("click", () => {
        this.#zoom.setLevel(pv / 100);
        zoomMenu.classList.remove("lector-dropdown__menu--open");
      });
      zoomMenu.appendChild(item);
    }
    zoomMenu.appendChild(this.#el("div", "lector-dropdown__separator"));
    for (const [label, cmd] of [[this.#t("toolbar.fitPage"), "fitPage"], [this.#t("toolbar.fitWidth"), "fitWidth"]]) {
      const item = this.#btn("lector-dropdown__item");
      const lbl = this.#el("span", "lector-dropdown__item-label");
      lbl.textContent = label;
      item.appendChild(lbl);
      item.addEventListener("click", () => {
        if (cmd === "fitPage") this.#zoom.fitPage();
        else this.#zoom.fitWidth();
        zoomMenu.classList.remove("lector-dropdown__menu--open");
      });
      zoomMenu.appendChild(item);
    }
    chevron.addEventListener("click", (e) => {
      e.stopPropagation();
      zoomMenu.classList.toggle("lector-dropdown__menu--open");
    });
    const closeZoomMenu = (e) => {
      if (!chevronWrap.contains(e.target)) zoomMenu.classList.remove("lector-dropdown__menu--open");
    };
    document.addEventListener("click", closeZoomMenu);
    this.#pushCleanup(() => document.removeEventListener("click", closeZoomMenu));
    this.#wireMenu(chevron, zoomMenu, "lector-dropdown__menu--open");
    chevronWrap.appendChild(chevron);
    chevronWrap.appendChild(zoomMenu);
    const outBtn = this.#btn("lector-zoom__btn");
    outBtn.innerHTML = resolveIcon("zoom-out") ?? "";
    this.#tip(outBtn, this.#t("zoom.zoomOut"));
    outBtn.addEventListener("click", () => this.#zoom.zoomOut());
    const inBtn = this.#btn("lector-zoom__btn");
    inBtn.innerHTML = resolveIcon("zoom-in") ?? "";
    this.#tip(inBtn, this.#t("zoom.zoomIn"));
    inBtn.addEventListener("click", () => this.#zoom.zoomIn());
    const unsub = this.#zoom.level.subscribe((level) => {
      input.value = String(Math.round(level * 100));
    });
    this.#pushCleanup(unsub);
    input.addEventListener("focus", () => setTimeout(() => input.select(), 0));
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const v = parseInt(input.value, 10);
        if (!isNaN(v) && v > 0) this.#zoom.setLevel(v / 100);
        input.blur();
      }
      if (e.key === "Escape") input.blur();
    });
    input.addEventListener("blur", () => {
      input.value = String(Math.round(this.#zoom.level.peek() * 100));
    });
    wrap3.appendChild(valueWrap);
    wrap3.appendChild(chevronWrap);
    wrap3.appendChild(outBtn);
    wrap3.appendChild(inBtn);
    return wrap3;
  }
  // ─── Floating page controls ────────────────────────────
  #buildPageControls() {
    this.#runSection("page-controls", () => this.#buildPageControlsImpl());
  }
  #buildPageControlsImpl() {
    this.#pageControlsEl.innerHTML = "";
    const inner = this.#el("div", "lector-page-controls__inner");
    const prevBtn = this.#btn("lector-btn");
    prevBtn.appendChild(this.#icon("chevron-left"));
    this.#tip(prevBtn, this.#t("toolbar.prevPage"));
    prevBtn.addEventListener("click", () => {
      try {
        void this.#engine.plugins.commands.execute("navigation.previous-page");
      } catch {
      }
    });
    const pageWrap = this.#el("div", "lector-page-input");
    const pageInput = document.createElement("input");
    pageInput.type = "text";
    pageInput.inputMode = "numeric";
    pageInput.className = "lector-page-input__field";
    pageInput.setAttribute("aria-label", this.#t("page.goToPage"));
    const sizePageInput = (val) => {
      const w = Math.max(1.5, val.length * 0.65 + 0.5);
      pageInput.style.setProperty("--lector-page-input-w", `${w}em`);
    };
    pageInput.value = "1";
    sizePageInput("1");
    const totalSpan = this.#el("span", "lector-page-input__total");
    totalSpan.textContent = "";
    const nextBtn = this.#btn("lector-btn");
    nextBtn.appendChild(this.#icon("chevron-right"));
    this.#tip(nextBtn, this.#t("toolbar.nextPage"));
    nextBtn.addEventListener("click", () => {
      try {
        void this.#engine.plugins.commands.execute("navigation.next-page");
      } catch {
      }
    });
    const pageStatus = this.#el("span", "lector-sr-only");
    pageStatus.setAttribute("aria-live", "polite");
    pageStatus.setAttribute("role", "status");
    pageWrap.appendChild(pageInput);
    pageWrap.appendChild(totalSpan);
    pageWrap.appendChild(pageStatus);
    inner.appendChild(prevBtn);
    inner.appendChild(pageWrap);
    inner.appendChild(nextBtn);
    this.#pageControlsEl.appendChild(inner);
    let pinnedPage = null;
    let lastDisplayedPage = 1;
    const setDisplayedPage = (pg) => {
      if (pg === lastDisplayedPage) return;
      lastDisplayedPage = pg;
      pageInput.value = String(pg);
      sizePageInput(pageInput.value);
      const total = this.#document.activeDocument.peek()?.pageCount ?? 0;
      pageStatus.textContent = total ? `Page ${pg} of ${total}` : `Page ${pg}`;
    };
    const unsubNav = this.#engine.plugins.events.on("viewport:scroll-to-page", (...args) => {
      const p = args[0] + 1;
      pinnedPage = p;
      setDisplayedPage(p);
    });
    this.#pushCleanup(unsubNav);
    const clearPin = () => {
      pinnedPage = null;
    };
    this.#canvas.addEventListener("wheel", clearPin, { passive: true });
    this.#canvas.addEventListener("touchmove", clearPin, { passive: true });
    this.#canvas.addEventListener("pointerdown", clearPin);
    const unsub1 = this.#viewportInstance.visiblePages.subscribe((pages) => {
      if (document.activeElement === pageInput) return;
      if (pinnedPage !== null) return;
      if (pages.length > 0) {
        setDisplayedPage(pages[0] + 1);
      }
    });
    this.#pushCleanup(unsub1);
    const unsub2 = this.#document.activeDocument.subscribe((handle) => {
      totalSpan.textContent = handle ? `\xA0${handle.pageCount}` : "";
    });
    this.#pushCleanup(unsub2);
    pageInput.addEventListener("focus", () => setTimeout(() => pageInput.select(), 0));
    const goToPage = (pg) => {
      const doc = this.#document.activeDocument.peek();
      if (isNaN(pg) || pg < 1 || !doc || pg > doc.pageCount) return;
      pinnedPage = pg;
      setDisplayedPage(pg);
      this.#viewportInstance.scrollToPage(pg - 1, false);
    };
    pageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        goToPage(parseInt(pageInput.value, 10));
        pageInput.blur();
      }
      if (e.key === "Escape") pageInput.blur();
    });
  }
  // ─── Sidebar ───────────────────────────────────────────
  #buildSidebar() {
    this.#runSection("sidebar", () => this.#buildSidebarImpl());
  }
  #buildSidebarImpl() {
    this.#sidebarEl.innerHTML = "";
    const panels = this.#ui.state.sidebar.panels;
    if (panels.length === 0) {
      this.#sidebarEl.style.display = "none";
      return;
    }
    const content = this.#el("div", "lector-sidebar__content");
    const tabs = this.#el("div", "lector-sidebar__tabs");
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", this.#t("toolbar.sidebar"));
    for (let i = 0; i < panels.length; i++) {
      const panel = panels[i];
      const tab = this.#btn("lector-sidebar__tab");
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", "false");
      tab.id = `lector-tab-${panel.id}`;
      tab.setAttribute("aria-controls", `lector-tabpanel-${panel.id}`);
      tab.tabIndex = i === 0 ? 0 : -1;
      tab.dataset["panelId"] = panel.id;
      const iconSvg = resolveIcon(panel.icon);
      if (iconSvg) {
        const ic = this.#el("span", "lector-sidebar__tab-icon");
        ic.innerHTML = iconSvg;
        tab.appendChild(ic);
      }
      const tabLabel = panel.labelKey ? this.#t(panel.labelKey) : panel.label;
      this.#tip(tab, tabLabel);
      tab.addEventListener("click", () => this.#ui.setActivePanel(panel.id));
      tabs.appendChild(tab);
    }
    tabs.addEventListener("keydown", (e) => {
      const tabEls = Array.from(tabs.querySelectorAll('[role="tab"]'));
      const idx = tabEls.indexOf(e.target);
      if (idx < 0) return;
      let next = -1;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (idx + 1) % tabEls.length;
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (idx - 1 + tabEls.length) % tabEls.length;
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = tabEls.length - 1;
      if (next >= 0) {
        e.preventDefault();
        for (const t of tabEls) t.tabIndex = -1;
        tabEls[next].tabIndex = 0;
        tabEls[next].focus();
        tabEls[next].click();
      }
    });
    const body = this.#el("div", "lector-sidebar__body");
    body.setAttribute("role", "tabpanel");
    body.id = "lector-tabpanel-sidebar";
    if (panels.length > 1) {
      content.appendChild(tabs);
    }
    content.appendChild(body);
    const handle = this.#el("div", "lector-sidebar__resize-handle");
    this.#wireResizeHandle(handle);
    this.#sidebarEl.appendChild(content);
    this.#sidebarEl.appendChild(handle);
  }
  #wireResizeHandle(handle) {
    handle.tabIndex = 0;
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-orientation", "vertical");
    handle.setAttribute("aria-label", this.#t("toolbar.sidebar"));
    handle.setAttribute("aria-valuenow", String(this.#sidebarEl.getBoundingClientRect().width));
    handle.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        const current = this.#sidebarEl.getBoundingClientRect().width;
        const step = e.shiftKey ? 40 : 10;
        const delta = e.key === "ArrowRight" ? step : -step;
        const newW = Math.max(160, Math.min(480, current + delta));
        this.#sidebarEl.style.width = `${newW}px`;
        handle.setAttribute("aria-valuenow", String(Math.round(newW)));
      }
    });
    let startX = 0, startW = 0, dragging = false;
    const down = (e) => {
      e.preventDefault();
      dragging = true;
      startX = e.clientX;
      startW = this.#sidebarEl.getBoundingClientRect().width;
      handle.setPointerCapture(e.pointerId);
      handle.classList.add("lector-sidebar__resize-handle--dragging");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      this.#sidebarEl.style.transition = "none";
      this.#viewportInstance.setResizeObserverPaused(true);
    };
    const move = (e) => {
      if (!dragging) return;
      this.#sidebarEl.style.width = `${Math.max(160, Math.min(480, startW + e.clientX - startX))}px`;
    };
    const up = () => {
      if (!dragging) return;
      dragging = false;
      handle.classList.remove("lector-sidebar__resize-handle--dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      this.#sidebarEl.style.transition = "";
      this.#viewportInstance.setResizeObserverPaused(false);
    };
    handle.addEventListener("pointerdown", down);
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
    handle.addEventListener("pointercancel", up);
  }
  #updateSidebarActiveTab() {
    const activeId = this.#ui.state.sidebar.activePanel.value;
    for (const tab of this.#sidebarEl.querySelectorAll(".lector-sidebar__tab")) {
      const t = tab;
      const isActive = t.dataset["panelId"] === activeId;
      t.classList.toggle("lector-sidebar__tab--active", isActive);
      t.setAttribute("aria-selected", String(isActive));
      t.tabIndex = isActive ? 0 : -1;
    }
    const body = this.#sidebarEl.querySelector(".lector-sidebar__body");
    if (!body) return;
    if (activeId) {
      body.setAttribute("aria-labelledby", `lector-tab-${activeId}`);
    }
    body.innerHTML = "";
    if (!activeId) return;
    this.#runSection("sidebar-panel", () => {
      switch (activeId) {
        case "thumbnails":
          this.#buildThumbnails(body);
          break;
        case "bookmarks":
          this.#buildBookmarksPanel(body);
          break;
        case "annotations":
          this.#buildAnnotationsPanel(body);
          break;
        case "attachments":
          this.#buildAttachmentsPanel(body);
          break;
        case "signatures":
          this.#buildSignaturesPanel(body);
          break;
        case "layers":
          this.#buildLayersPanel(body);
          break;
        case "comparison":
          this.#buildComparisonPanel(body);
          break;
      }
    });
  }
  // ─── Thumbnails ────────────────────────────────────────
  #buildThumbnails(container) {
    const handle = this.#document.activeDocument.peek();
    if (!handle) return;
    const tc = this.#el("div", "lector-thumbnails");
    container.appendChild(tc);
    const TW = 150;
    for (let i = 0; i < handle.pageCount; i++) {
      const ps = handle.pageSizes[i];
      const th = Math.round(TW * (ps.height / ps.width));
      const wrap3 = this.#el("div", "lector-thumbnail");
      wrap3.dataset["pageIndex"] = String(i);
      const cw = this.#el("div", "lector-thumbnail__canvas-wrap");
      const canvas = document.createElement("canvas");
      canvas.className = "lector-thumbnail__canvas";
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(TW * dpr);
      canvas.height = Math.round(th * dpr);
      canvas.style.width = `${TW}px`;
      canvas.style.height = `${th}px`;
      cw.appendChild(canvas);
      wrap3.appendChild(cw);
      const label = this.#el("div", "lector-thumbnail__label");
      label.textContent = String(i + 1);
      wrap3.appendChild(label);
      wrap3.addEventListener("click", () => this.#viewportInstance.scrollToPage(i, false));
      wrap3.addEventListener("contextmenu", (e) => this.#showThumbnailContextMenu(e, i));
      wrap3.addEventListener("keydown", (e) => {
        if (e.key === "F10" && e.shiftKey || e.key === "ContextMenu") {
          e.preventDefault();
          const r = wrap3.getBoundingClientRect();
          this.#showThumbnailContextMenu(
            new MouseEvent("contextmenu", { clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 }),
            i
          );
        }
      });
      if (this.#pageOps) {
        wrap3.draggable = true;
        wrap3.addEventListener("dragstart", (e) => {
          e.dataTransfer?.setData("text/plain", String(i));
          wrap3.classList.add("lector-thumbnail--dragging");
        });
        wrap3.addEventListener("dragend", () => {
          wrap3.classList.remove("lector-thumbnail--dragging");
          tc.querySelectorAll(".lector-thumbnail--drop-target").forEach((el) => el.classList.remove("lector-thumbnail--drop-target"));
        });
        wrap3.addEventListener("dragover", (e) => {
          e.preventDefault();
          wrap3.classList.add("lector-thumbnail--drop-target");
        });
        wrap3.addEventListener("dragleave", () => {
          wrap3.classList.remove("lector-thumbnail--drop-target");
        });
        wrap3.addEventListener("drop", (e) => {
          e.preventDefault();
          wrap3.classList.remove("lector-thumbnail--drop-target");
          const fromIdx = parseInt(e.dataTransfer?.getData("text/plain") ?? "-1", 10);
          if (fromIdx >= 0 && fromIdx !== i) {
            void this.#pageOps.movePage(handle.id, fromIdx, i).then(() => {
              setTimeout(() => this.#updateSidebarActiveTab(), 200);
            });
          }
        });
      }
      tc.appendChild(wrap3);
      this.#renderThumbnail(handle, i, canvas);
    }
    const unsub = this.#viewportInstance.visiblePages.subscribe((pages) => {
      for (const t of tc.querySelectorAll(".lector-thumbnail")) {
        const el = t;
        el.classList.toggle("lector-thumbnail--active", pages.includes(parseInt(el.dataset["pageIndex"] ?? "-1", 10)));
      }
    });
    this.#pushCleanup(unsub);
  }
  async #renderThumbnail(handle, idx, canvas) {
    try {
      const bmp = await this.#engine.renderPage(handle.id, idx, canvas.width, canvas.height, {
        priority: RenderPriority.LOW
      });
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
      bmp.close();
    } catch {
    }
  }
  /**
   * Re-render the thumbnail for a single page after its content changed
   * (e.g. a redaction was applied). The thumbnails panel is built lazily, so
   * when it isn't mounted this is a no-op — it will render fresh content the
   * next time the panel is opened. No-op when the changed document isn't the
   * active one.
   */
  #refreshThumbnail(docId, pageIndex) {
    const handle = this.#document.activeDocument.peek();
    if (!handle || handle.id !== docId) return;
    const canvas = this.#root.querySelector(
      `.lector-thumbnail[data-page-index="${pageIndex}"] .lector-thumbnail__canvas`
    );
    if (canvas) void this.#renderThumbnail(handle, pageIndex, canvas);
  }
  #rerenderScheduled = false;
  /**
   * Re-render the visible pages on the next animation frame, coalescing
   * multiple invalidations (e.g. a bulk redaction across several pages) into
   * a single render pass.
   */
  #scheduleVisibleRerender() {
    if (this.#rerenderScheduled) return;
    this.#rerenderScheduled = true;
    requestAnimationFrame(() => {
      this.#rerenderScheduled = false;
      const vis = this.#viewportInstance.visiblePages.peek();
      if (vis.length > 0) void this.#renderVisiblePages(vis);
    });
  }
  // ─── Bookmarks Panel ────────────────────────────────────
  #buildBookmarksPanel(container) {
    if (!this.#navigation) {
      container.textContent = this.#t("bookmarks.navigationPluginMissing");
      return;
    }
    const handle = this.#document.activeDocument.peek();
    if (!handle) return;
    const wrap3 = this.#el("div", "lector-bookmark-panel");
    container.appendChild(wrap3);
    const addRow = this.#el("div", "lector-attachment-panel__actions");
    const addBtn = this.#btn("lector-attachment-panel__add");
    addBtn.appendChild(this.#icon("plus"));
    const addLabel = this.#el("span", "");
    addLabel.textContent = ` ${this.#t("bookmarks.add")}`;
    addLabel.style.fontSize = "var(--lector-font-size-sm)";
    addBtn.appendChild(addLabel);
    addBtn.addEventListener("click", () => {
      const visiblePages = this.#viewportInstance.visiblePages.peek();
      const pageIdx = visiblePages[0] ?? 0;
      const title = this.#t("annotations.pageHeader", { page: pageIdx + 1 });
      void this.#engine.workerProxy.addBookmark(handle.id, title, pageIdx, -1).then((ok) => {
        if (ok) {
          this.#navigation.clearCache(handle.id);
          this.#showToast(this.#t("toast.bookmarkAdded"));
          this.#updateSidebarActiveTab();
        }
      });
    });
    addRow.appendChild(addBtn);
    wrap3.appendChild(addRow);
    void this.#navigation.getBookmarks(handle.id).then((bookmarks) => {
      if (bookmarks.length === 0) {
        const empty = this.#el("div", "lector-sidebar-empty");
        empty.textContent = this.#t("bookmarks.empty");
        wrap3.appendChild(empty);
        return;
      }
      const tree = this.#buildBookmarkTree(bookmarks, handle.id);
      wrap3.appendChild(tree);
    });
  }
  #buildBookmarkTree(nodes, docId) {
    const ul = this.#el("ul", "lector-bookmark-tree");
    const isTopLevel = docId !== void 0;
    for (let nodeIdx = 0; nodeIdx < nodes.length; nodeIdx++) {
      const node = nodes[nodeIdx];
      const li = this.#el("li", "lector-bookmark-node");
      const hasChildren = node.children.length > 0;
      const row = this.#el("div", "lector-bookmark-node__row");
      if (isTopLevel) {
        const grip = this.#el("span", "lector-bookmark-node__grip");
        grip.innerHTML = resolveIcon("grip") ?? "";
        grip.title = this.#t("bookmarks.dragHint");
        row.appendChild(grip);
      }
      if (hasChildren) {
        const toggle = this.#btn("lector-bookmark-node__toggle");
        toggle.innerHTML = resolveIcon("chevron-right-sm") ?? "";
        row.appendChild(toggle);
        toggle.addEventListener("click", (e) => {
          e.stopPropagation();
          li.classList.toggle("lector-bookmark-node--expanded");
        });
      } else {
        const spacer = this.#el("span", "lector-bookmark-node__spacer");
        row.appendChild(spacer);
      }
      const label = this.#el("span", "lector-bookmark-node__label");
      label.textContent = node.title;
      label.title = node.title;
      row.appendChild(label);
      if (isTopLevel && docId) {
        const idx = nodeIdx;
        const beginRename = () => {
          if (label.querySelector("input")) return;
          const input = document.createElement("input");
          input.type = "text";
          input.className = "lector-bookmark-node__rename";
          input.value = node.title;
          input.setAttribute("aria-label", this.#t("bookmarks.rename"));
          label.textContent = "";
          label.appendChild(input);
          input.focus();
          input.select();
          let committed = false;
          const cancel = () => {
            if (committed) return;
            committed = true;
            label.textContent = node.title;
            label.title = node.title;
          };
          const commit = () => {
            if (committed) return;
            const next = input.value.trim();
            if (next.length === 0 || next === node.title) {
              cancel();
              return;
            }
            committed = true;
            label.textContent = next;
            label.title = next;
            void this.#engine.workerProxy.setBookmarkTitle(docId, idx, next).then((ok) => {
              if (ok) {
                this.#navigation.clearCache(docId);
                this.#showToast(this.#t("toast.bookmarkRenamed"));
                this.#updateSidebarActiveTab();
              }
            });
          };
          input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              commit();
            } else if (e.key === "Escape") {
              e.preventDefault();
              cancel();
            }
          });
          input.addEventListener("blur", commit);
          input.addEventListener("click", (e) => {
            e.stopPropagation();
          });
          input.addEventListener("dblclick", (e) => {
            e.stopPropagation();
          });
        };
        label.addEventListener("dblclick", (e) => {
          e.stopPropagation();
          beginRename();
        });
        const editBtn = this.#btn("lector-bookmark-node__edit");
        editBtn.innerHTML = resolveIcon("pencil") ?? "";
        editBtn.title = this.#t("bookmarks.rename");
        editBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          beginRename();
        });
        row.appendChild(editBtn);
        const delBtn = this.#btn("lector-bookmark-node__delete");
        delBtn.innerHTML = resolveIcon("x") ?? "";
        delBtn.title = this.#t("common.delete");
        delBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          void this.#engine.workerProxy.deleteBookmark(docId, idx).then((ok) => {
            if (ok) {
              this.#navigation.clearCache(docId);
              this.#showToast(this.#t("toast.bookmarkDeleted"));
              this.#updateSidebarActiveTab();
            }
          });
        });
        row.appendChild(delBtn);
        li.draggable = true;
        li.dataset.bookmarkIndex = String(idx);
        li.addEventListener("dragstart", (e) => {
          if (!e.dataTransfer) return;
          e.dataTransfer.effectAllowed = "move";
          e.dataTransfer.setData("application/x-lector-bookmark", String(idx));
          li.classList.add("lector-bookmark-node--dragging");
        });
        li.addEventListener("dragend", () => {
          li.classList.remove("lector-bookmark-node--dragging");
          li.classList.remove("lector-bookmark-node--drop-before");
          li.classList.remove("lector-bookmark-node--drop-after");
        });
        li.addEventListener("dragover", (e) => {
          if (!e.dataTransfer?.types.includes("application/x-lector-bookmark")) return;
          e.preventDefault();
          e.dataTransfer.dropEffect = "move";
          const rect = li.getBoundingClientRect();
          const before = e.clientY < rect.top + rect.height / 2;
          li.classList.toggle("lector-bookmark-node--drop-before", before);
          li.classList.toggle("lector-bookmark-node--drop-after", !before);
        });
        li.addEventListener("dragleave", (e) => {
          const related = e.relatedTarget;
          if (related && li.contains(related)) return;
          li.classList.remove("lector-bookmark-node--drop-before");
          li.classList.remove("lector-bookmark-node--drop-after");
        });
        li.addEventListener("drop", (e) => {
          if (!e.dataTransfer?.types.includes("application/x-lector-bookmark")) return;
          e.preventDefault();
          e.stopPropagation();
          const fromStr = e.dataTransfer.getData("application/x-lector-bookmark");
          const from = Number.parseInt(fromStr, 10);
          const before = li.classList.contains("lector-bookmark-node--drop-before");
          li.classList.remove("lector-bookmark-node--drop-before");
          li.classList.remove("lector-bookmark-node--drop-after");
          if (Number.isNaN(from) || from === idx) return;
          let to = before ? idx : idx + 1;
          if (from < to) to -= 1;
          if (to === from) return;
          void this.#engine.workerProxy.moveBookmark(docId, from, to).then((ok) => {
            if (ok) {
              this.#navigation.clearCache(docId);
              this.#showToast(this.#t("toast.bookmarkMoved"));
              this.#updateSidebarActiveTab();
            }
          });
        });
      }
      if (node.pageIndex !== null) {
        const pageNum = this.#el("span", "lector-bookmark-node__page");
        pageNum.textContent = String(node.pageIndex + 1);
        row.appendChild(pageNum);
        row.addEventListener("click", () => {
          if (node.pageIndex !== null) {
            this.#viewportInstance.scrollToPage(node.pageIndex, false);
          }
        });
        row.style.cursor = "pointer";
      }
      li.appendChild(row);
      if (hasChildren) {
        const childTree = this.#buildBookmarkTree(node.children);
        li.appendChild(childTree);
      }
      ul.appendChild(li);
    }
    return ul;
  }
  // ─── Annotations/Comments Panel ────────────────────────
  #buildAnnotationsPanel(container) {
    if (!this.#annotation) {
      container.textContent = this.#t("annotations.pluginMissing");
      return;
    }
    const handle = this.#document.activeDocument.peek();
    if (!handle) return;
    const wrap3 = this.#el("div", "lector-annot-panel");
    container.appendChild(wrap3);
    const allAnnots = this.#annotation.getForDocument(handle.id);
    const visible = allAnnots.filter((t) => isUserAnnotation(t.data.subtype));
    if (visible.length === 0) {
      const empty = this.#el("div", "lector-sidebar-empty");
      empty.textContent = this.#t("annotations.empty");
      wrap3.appendChild(empty);
      return;
    }
    const byPage = /* @__PURE__ */ new Map();
    for (const t of visible) {
      let arr = byPage.get(t.data.pageIndex);
      if (!arr) {
        arr = [];
        byPage.set(t.data.pageIndex, arr);
      }
      arr.push(t);
    }
    const sortedPages = [...byPage.keys()].sort((a, b) => a - b);
    for (const pageIdx of sortedPages) {
      const pageAnnots = byPage.get(pageIdx);
      const pageGroup = this.#el("div", "lector-annot-panel__page-group");
      const pageHeader = this.#el("div", "lector-annot-panel__page-header");
      pageHeader.textContent = this.#t("annotations.pageHeader", { page: pageIdx + 1 });
      pageGroup.appendChild(pageHeader);
      for (const tracked of pageAnnots) {
        const annot = tracked.data;
        const card = this.#el("div", "lector-annot-panel__card");
        if (annot.readAt === void 0) card.classList.add("lector-annot-panel__card--unread");
        const header = this.#el("div", "lector-annot-panel__card-header");
        const typeLabel = this.#el("span", "lector-annot-panel__type");
        typeLabel.textContent = this.#annotSubtypeName(annot.subtype, annot.tag);
        header.appendChild(typeLabel);
        if (annot.author) {
          const authorEl = this.#el("span", "lector-annot-panel__author");
          authorEl.textContent = annot.author;
          header.appendChild(authorEl);
        }
        if (annot.createdDate) {
          const timeEl = this.#el("span", "lector-annot-panel__time");
          timeEl.textContent = this.#formatDate(annot.createdDate);
          header.appendChild(timeEl);
        }
        card.appendChild(header);
        if (annot.commentStatus && annot.commentStatus !== "open") {
          const badge = this.#el("span", `lector-annot-panel__status lector-annot-panel__status--${annot.commentStatus}`);
          badge.textContent = annot.commentStatus.charAt(0).toUpperCase() + annot.commentStatus.slice(1);
          card.appendChild(badge);
        }
        if (annot.resolved) {
          const badge = this.#el("span", "lector-annot-panel__status lector-annot-panel__status--resolved");
          badge.textContent = this.#t("commentStatus.resolved");
          card.appendChild(badge);
        }
        if (annot.contents) {
          const preview = this.#el("div", "lector-annot-panel__text");
          preview.textContent = annot.contents.length > 120 ? annot.contents.substring(0, 120) + "..." : annot.contents;
          card.appendChild(preview);
        }
        if (annot.comments && annot.comments.length > 0) {
          const replies = this.#el("span", "lector-annot-panel__replies");
          replies.textContent = `${annot.comments.length} ${annot.comments.length === 1 ? "reply" : "replies"}`;
          card.appendChild(replies);
        }
        card.addEventListener("click", () => {
          this.#viewportInstance.scrollToPage(annot.pageIndex, false);
          if (this.#annotation) {
            this.#annotation.selectAnnotation(tracked.id);
          }
        });
        pageGroup.appendChild(card);
      }
      wrap3.appendChild(pageGroup);
    }
  }
  #annotSubtypeName(subtype, tag) {
    if (tag === "line") return "Line";
    if (tag === "arrow" || tag === "arrow-start" || tag === "arrow-both") return "Arrow";
    if (tag === "polygon") return "Polygon";
    if (tag === "polyline") return this.#t("annotation.polyline");
    const names = {
      1: this.#t("annotation.stickyNote"),
      2: "Link",
      3: this.#t("annotation.freetext"),
      4: this.#t("annotation.line"),
      5: this.#t("annotation.rectangle"),
      6: this.#t("annotation.circle"),
      7: this.#t("annotation.polygon"),
      8: this.#t("annotation.polyline"),
      9: this.#t("annotation.highlight"),
      10: this.#t("annotation.underline"),
      11: this.#t("annotation.squiggly"),
      12: this.#t("annotation.strikeout"),
      13: this.#t("annotation.stamp"),
      14: this.#t("annotation.insertText"),
      15: this.#t("annotation.ink"),
      28: this.#t("annotation.redaction")
    };
    return names[subtype] ?? this.#t("annotation.defaultLabel");
  }
  // ─── Comments Sidebar ──────────────────────────────────
  //
  // The right-hand persistent panel for review-style commenting. Built
  // as static chrome (header + filter row + scrollable body) once in
  // #buildCommentsSidebarShell, then refreshed on annotation changes
  // via #refreshCommentsSidebar. Auto-opens on annotation selection
  // and stays open until the user closes it (toggle button or X).
  //
  // The active-pane document tracking is inherited from the existing
  // viewport.activeViewport effect — when the user clicks into the
  // other pane in split mode, document.activeDocument changes, the
  // refresh effect fires, and the sidebar swaps to that pane's doc.
  #buildCommentsSidebarShell() {
    const shell = this.#commentsSidebarEl;
    shell.innerHTML = "";
    const header = this.#el("div", "lector-comments-sidebar__header");
    const title = this.#el("div", "lector-comments-sidebar__title");
    title.textContent = this.#t("comments.title") || "Comments";
    header.appendChild(title);
    const closeBtn = this.#btn("lector-comments-sidebar__close");
    closeBtn.innerHTML = resolveIcon("x") ?? "";
    closeBtn.setAttribute("aria-label", this.#t("common.close") || "Close");
    closeBtn.addEventListener("click", () => this.#setCommentsSidebarCollapsed(true));
    header.appendChild(closeBtn);
    shell.appendChild(header);
    const filters = this.#el("div", "lector-comments-sidebar__filters");
    const filterSelect = document.createElement("select");
    filterSelect.className = "lector-comments-sidebar__select";
    filterSelect.setAttribute("aria-label", this.#t("comments.filter.all"));
    for (const opt of [
      { v: "all", l: this.#t("comments.filter.all") || "All" },
      { v: "open", l: this.#t("comments.filter.open") || "Open" },
      { v: "resolved", l: this.#t("comments.filter.resolved") || "Resolved" },
      { v: "mine", l: this.#t("comments.filter.mine") || "Mine" },
      { v: "mentions", l: this.#t("comments.filter.mentions") || "@ me" }
    ]) {
      const o = document.createElement("option");
      o.value = opt.v;
      o.textContent = opt.l;
      if (opt.v === this.#commentsFilter) o.selected = true;
      filterSelect.appendChild(o);
    }
    filterSelect.addEventListener("change", () => {
      this.#commentsFilter = filterSelect.value;
      this.#refreshCommentsSidebar();
    });
    filters.appendChild(filterSelect);
    const sortSelect = document.createElement("select");
    sortSelect.className = "lector-comments-sidebar__select";
    sortSelect.setAttribute("aria-label", this.#t("comments.sort.page"));
    for (const opt of [
      { v: "page", l: this.#t("comments.sort.page") || "By page" },
      { v: "date", l: this.#t("comments.sort.date") || "By date" },
      { v: "author", l: this.#t("comments.sort.author") || "By author" }
    ]) {
      const o = document.createElement("option");
      o.value = opt.v;
      o.textContent = opt.l;
      if (opt.v === this.#commentsSort) o.selected = true;
      sortSelect.appendChild(o);
    }
    sortSelect.addEventListener("change", () => {
      this.#commentsSort = sortSelect.value;
      this.#refreshCommentsSidebar();
    });
    filters.appendChild(sortSelect);
    shell.appendChild(filters);
    this.#commentsSidebarBody = this.#el("div", "lector-comments-sidebar__body");
    shell.appendChild(this.#commentsSidebarBody);
  }
  /** Open or close the comments sidebar. */
  #setCommentsSidebarCollapsed(collapsed) {
    this.#commentsSidebarCollapsed.value = collapsed;
    this.#commentsSidebarEl.classList.toggle(
      "lector-comments-sidebar--collapsed",
      collapsed
    );
  }
  /**
   * Debounced sidebar refresh. Used by selection / annotation event
   * handlers so a burst of changes only triggers a single rebuild,
   * AND so the rebuild doesn't run synchronously during a pointerdown
   * (which would block subsequent pointermove events and make
   * annotation drags feel sluggish).
   *
   * If a drag is in progress when the timer fires, the refresh is
   * skipped — drag-end re-schedules it.
   */
  #scheduleSidebarRefresh(delayMs = 80) {
    if (this.#commentsRefreshTimer) {
      clearTimeout(this.#commentsRefreshTimer);
    }
    this.#commentsRefreshTimer = setTimeout(() => {
      this.#commentsRefreshTimer = null;
      if (this.#commentsSidebarCollapsed.peek()) return;
      if (this.#annotDragging) return;
      const active = document.activeElement;
      if (active instanceof HTMLTextAreaElement && this.#commentsSidebarEl.contains(active) && active.value.length > 0) {
        return;
      }
      this.#refreshCommentsSidebar();
    }, delayMs);
  }
  /** Cancel any pending sidebar refresh (called when a drag starts). */
  #cancelPendingSidebarRefresh() {
    if (this.#commentsRefreshTimer) {
      clearTimeout(this.#commentsRefreshTimer);
      this.#commentsRefreshTimer = null;
    }
  }
  /** Toggle the comments sidebar (toolbar button + keyboard handler). */
  toggleCommentsSidebar() {
    this.#setCommentsSidebarCollapsed(!this.#commentsSidebarCollapsed.peek());
  }
  /**
   * Re-render the comments sidebar body. Cheap because the body is
   * isolated DOM — header + filter row are not touched.
   *
   * Driven by an effect on annotation store events (created / updated /
   * deleted) plus the active document signal so split-pane switching
   * swaps the list.
   */
  #refreshCommentsSidebar() {
    const body = this.#commentsSidebarBody;
    if (!body || !this.#annotation) return;
    body.innerHTML = "";
    const handle = this.#document.activeDocument.peek();
    if (!handle) {
      this.#renderCommentsEmpty(body, this.#t("comments.noDocument") || "No document open");
      return;
    }
    const all = this.#annotation.getForDocument(handle.id);
    const activeId = this.#activeThreadAnnotId;
    const hasUserComments = (a) => {
      if (a.comments && a.comments.length > 0) return true;
      if (a.contents && !(a.subtype === 13 && a.stamp?.name === a.contents)) return true;
      return false;
    };
    let threads = all.filter((t) => {
      const a = t.data;
      if (!isUserAnnotation(a.subtype)) return false;
      if (a.subtype === 1) return true;
      if (isToolOutputAnnotation(a.tag, a.subtype)) return hasUserComments(a);
      if (a.subtype === 13 && !hasUserComments(a)) return false;
      if (t.id === activeId) return true;
      return hasUserComments(a);
    });
    const userId = this.#engine.user?.id;
    threads = threads.filter((t) => {
      const a = t.data;
      switch (this.#commentsFilter) {
        case "open":
          return !a.resolved && (a.commentStatus ?? "open") === "open";
        case "resolved":
          return a.resolved === true;
        case "mine":
          return userId !== void 0 && a.authorId === userId;
        case "mentions":
          if (userId === void 0) return false;
          if (a.comments?.some((c) => c.mentions?.some((m) => m.userId === userId))) return true;
          return false;
        case "all":
        default:
          return true;
      }
    });
    if (threads.length === 0) {
      this.#renderCommentsEmpty(
        body,
        this.#t("comments.empty") || "No comments yet.\nClick any annotation in the page to start a thread."
      );
      return;
    }
    if (this.#commentsSort === "date") {
      threads = [...threads].sort((a, b) => {
        const da2 = a.data.createdDate ?? "";
        const db = b.data.createdDate ?? "";
        return db.localeCompare(da2);
      });
    } else if (this.#commentsSort === "author") {
      threads = [...threads].sort(
        (a, b) => (a.data.author ?? "").localeCompare(b.data.author ?? "")
      );
    } else {
      threads = [...threads].sort((a, b) => a.data.pageIndex - b.data.pageIndex);
    }
    if (this.#commentsSort === "page") {
      let lastPage = -1;
      for (const t of threads) {
        if (t.data.pageIndex !== lastPage) {
          lastPage = t.data.pageIndex;
          const groupHeader = this.#el("div", "lector-comments-sidebar__page-header");
          groupHeader.textContent = `${this.#t("common.page") || "Page"} ${lastPage + 1}`;
          body.appendChild(groupHeader);
        }
        body.appendChild(this.#buildCommentsThreadCard(handle.id, t.id, t.data));
      }
    } else {
      for (const t of threads) {
        body.appendChild(this.#buildCommentsThreadCard(handle.id, t.id, t.data));
      }
    }
  }
  #renderCommentsEmpty(container, message) {
    const empty = this.#el("div", "lector-comments-sidebar__empty");
    const iconWrap = this.#el("div", "lector-comments-sidebar__empty-icon");
    iconWrap.innerHTML = resolveIcon("message-square") ?? "";
    empty.appendChild(iconWrap);
    const text = this.#el("div", "");
    text.style.whiteSpace = "pre-line";
    text.textContent = message;
    empty.appendChild(text);
    container.appendChild(empty);
  }
  /**
   * Render one thread card. Two visual states with a continuous look:
   *   - collapsed: header (type · author · time + status label),
   *                2-line body preview, optional reply count.
   *   - active:    same header, sticky-note properties strip (sticky
   *                notes only), comment entries (no avatars), compose
   *                textarea, single footer row with Resolve / Delete
   *                links on the left and Cancel / Reply buttons on
   *                the right.
   *
   * No popovers. No ⋯ menu. No status pill button. No card chrome
   * (no color stripe, no background tint on active). Threads read as
   * a flat vertical conversation list.
   */
  #buildCommentsThreadCard(docId, annotId, annot) {
    const isActive = this.#activeThreadAnnotId === annotId;
    const card = this.#el("div", "lector-cmt-thread");
    if (isActive) card.classList.add("lector-cmt-thread--active");
    card.dataset["annotId"] = annotId;
    const header = this.#el("div", "lector-cmt-thread__header");
    const topRow = this.#el("div", "lector-cmt-thread__header-top");
    const type = this.#el("span", "lector-cmt-thread__type");
    type.textContent = this.#annotSubtypeName(annot.subtype, annot.tag);
    topRow.appendChild(type);
    const topRight = this.#el("div", "lector-cmt-thread__header-actions");
    if (isActive && annot.subtype === 1 && this.#annotation) {
      const gearBtn = this.#btn("lector-cmt-thread__gear");
      gearBtn.appendChild(this.#icon("settings-2"));
      this.#tip(gearBtn, this.#t("annotation.icon"));
      const propsPanel2 = this.#el("div", "lector-cmt-thread__props-dropdown");
      propsPanel2.style.display = "none";
      this.#buildSidebarStickyProps(propsPanel2, docId, annotId, annot);
      gearBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const open = propsPanel2.style.display !== "none";
        propsPanel2.style.display = open ? "none" : "";
      });
      topRight.appendChild(gearBtn);
      header.__propsPanel = propsPanel2;
    }
    topRight.appendChild(this.#buildSidebarStatusLabel(annot));
    topRow.appendChild(topRight);
    header.appendChild(topRow);
    const metaRow = this.#el("div", "lector-cmt-thread__header-meta");
    if (annot.author) {
      const author = this.#el("span", "lector-cmt-thread__author");
      author.textContent = annot.author;
      metaRow.appendChild(author);
    }
    if (annot.author && annot.createdDate) {
      const sep = this.#el("span", "lector-cmt-thread__sep");
      sep.textContent = "\xB7";
      metaRow.appendChild(sep);
    }
    if (annot.createdDate) {
      const time = this.#el("span", "lector-cmt-thread__time");
      time.textContent = this.#formatDate(annot.createdDate);
      metaRow.appendChild(time);
    }
    if (metaRow.childElementCount > 0) header.appendChild(metaRow);
    card.appendChild(header);
    const propsPanel = header.__propsPanel;
    if (propsPanel) card.appendChild(propsPanel);
    const isStamp = annot.subtype === 13;
    const contentsIsMetadata = isStamp && annot.stamp?.name === annot.contents;
    if (!isActive) {
      const preview = this.#el("div", "lector-cmt-thread__preview");
      const hasUserContent = annot.contents && !contentsIsMetadata;
      if (hasUserContent) {
        preview.textContent = annot.contents;
      } else {
        preview.classList.add("lector-cmt-thread__preview--empty");
        preview.textContent = this.#t("comments.noBody") || "No comment yet \u2014 click to add";
      }
      card.appendChild(preview);
      const replyCount = annot.comments?.length ?? 0;
      if (replyCount > 0) {
        const footer = this.#el("div", "lector-cmt-thread__footer");
        const rc = this.#el("span", "lector-cmt-thread__reply-count");
        rc.innerHTML = resolveIcon("message-square") ?? "";
        const num = document.createElement("span");
        num.textContent = String(replyCount);
        rc.appendChild(num);
        footer.appendChild(rc);
        card.appendChild(footer);
      }
      card.addEventListener("click", (e) => {
        if (!this.#annotation) return;
        e.stopPropagation();
        this.#annotation.selectAnnotation(annotId);
        this.#viewportInstance.scrollToPage(annot.pageIndex, false);
      });
      return card;
    }
    const expanded = this.#el("div", "lector-cmt-thread__expanded");
    expanded.addEventListener("click", (e) => e.stopPropagation());
    const entries = this.#el("div", "lector-cmt-entries");
    if (annot.contents && !contentsIsMetadata) {
      this.#buildSidebarComment(
        entries,
        {
          id: "initial",
          authorId: annot.authorId ?? "",
          authorName: annot.author ?? "Unknown",
          text: annot.contents,
          timestamp: annot.createdDate ?? ""
        },
        docId,
        annotId,
        true
      );
    }
    if (annot.comments) {
      for (const c of annot.comments) {
        this.#buildSidebarComment(entries, c, docId, annotId, false);
      }
    }
    if (entries.children.length > 0) {
      expanded.appendChild(entries);
    }
    if (this.#engine.user && this.#annotation) {
      const ann = this.#annotation;
      setTimeout(() => ann.markAsRead(docId, annotId), 0);
    }
    this.#buildSidebarCompose(expanded, docId, annotId, annot);
    card.appendChild(expanded);
    return card;
  }
  /**
   * Render a single comment entry inside the expanded thread:
   *   author · time   [Edit  Delete on hover, owner only]
   *   body
   *
   * No avatars — they take 28px of horizontal space and add visual
   * noise on a narrow sidebar. The bold name + muted time line is
   * enough to anchor each entry. Replies are rendered with the same
   * structure as the original (no indentation).
   *
   * `isInitial=true` means this is the original comment stored on
   * annot.contents (rather than annot.comments[]). Editing the
   * initial comment patches `contents`; deleting it clears `contents`
   * but keeps the annotation. Replies use editComment / deleteComment.
   */
  #buildSidebarComment(parent, comment, docId, annotId, isInitial) {
    const user = this.#engine.user;
    const isOwner = user !== void 0 && comment.authorId === user.id;
    const entry = this.#el("div", "lector-cmt-entry");
    if (isInitial) entry.classList.add("lector-cmt-entry--initial");
    const body = this.#el("div", "lector-cmt-entry__body");
    body.textContent = comment.text;
    const head = this.#el("div", "lector-cmt-entry__head");
    if (!isInitial) {
      const author = this.#el("span", "lector-cmt-entry__author");
      author.textContent = comment.authorName;
      head.appendChild(author);
      const sep = this.#el("span", "lector-cmt-entry__sep");
      sep.textContent = "\xB7";
      head.appendChild(sep);
      const time = this.#el("span", "lector-cmt-entry__time");
      let timeText = comment.timestamp ? this.#formatDate(comment.timestamp) : "";
      if (comment.edited) {
        timeText = (timeText.length > 0 ? `${timeText} ` : "") + (this.#t("comment.edited") || "(edited)");
      }
      if (timeText.length > 0) time.textContent = timeText;
      head.appendChild(time);
    }
    if (isOwner && this.#annotation) {
      const actions = this.#el("span", "lector-cmt-entry__actions");
      const editBtn = this.#btn("lector-cmt-link");
      editBtn.textContent = this.#t("comment.edit") || "Edit";
      editBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        body.innerHTML = "";
        const edit = document.createElement("textarea");
        edit.className = "lector-cmt-compose__textarea";
        edit.value = comment.text;
        edit.rows = 3;
        edit.addEventListener("keydown", (ke) => ke.stopPropagation());
        body.appendChild(edit);
        const editActions = this.#el("div", "lector-cmt-compose__actions");
        const cancelBtn = this.#btn("lector-cmt-btn");
        cancelBtn.textContent = this.#t("comment.cancel") || "Cancel";
        cancelBtn.addEventListener("click", () => {
          body.innerHTML = "";
          body.textContent = comment.text;
        });
        const saveBtn = this.#btn("lector-cmt-btn lector-cmt-btn--primary");
        saveBtn.textContent = this.#t("comment.save") || "Save";
        saveBtn.addEventListener("click", () => {
          const newText = edit.value.trim();
          if (newText && newText !== comment.text) {
            if (isInitial) {
              void this.#annotation.update(docId, annotId, { contents: newText });
            } else {
              void this.#annotation.editComment(docId, annotId, comment.id, newText);
            }
          } else {
            body.innerHTML = "";
            body.textContent = comment.text;
          }
        });
        editActions.appendChild(cancelBtn);
        editActions.appendChild(saveBtn);
        body.appendChild(editActions);
        requestAnimationFrame(() => edit.focus());
      });
      actions.appendChild(editBtn);
      const delBtn = this.#btn("lector-cmt-link lector-cmt-link--danger");
      delBtn.textContent = this.#t("comment.delete") || "Delete";
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (isInitial) {
          void this.#annotation.update(docId, annotId, { contents: "" });
        } else {
          void this.#annotation.deleteComment(docId, annotId, comment.id);
        }
      });
      actions.appendChild(delBtn);
      head.appendChild(actions);
    }
    if (head.childElementCount > 0) entry.appendChild(head);
    entry.appendChild(body);
    parent.appendChild(entry);
  }
  /**
   * Read-only status label rendered in the thread header. Right-aligned
   * via `margin-left: auto`. Shows the current commentStatus, with
   * "Resolved" overriding everything when annot.resolved is true.
   * Not clickable — status changes via the Resolve link in the footer.
   */
  #buildSidebarStatusLabel(annot) {
    const status = annot.resolved ? "resolved" : annot.commentStatus ?? "open";
    const label = this.#el("span", `lector-cmt-thread__status lector-cmt-thread__status--${status}`);
    const dot = this.#el("span", "lector-cmt-thread__status-dot");
    label.appendChild(dot);
    const text = document.createElement("span");
    text.textContent = this.#t(`commentStatus.${status}`) || status.charAt(0).toUpperCase() + status.slice(1);
    label.appendChild(text);
    return label;
  }
  /**
   * Sticky-note properties strip — icon row + color row — rendered
   * inline at the top of the expanded thread with small ICON / COLOR
   * labels. Sticky notes have no popover, so this is their only
   * properties surface.
   */
  #buildSidebarStickyProps(parent, docId, annotId, annot) {
    if (!this.#annotation) return;
    const props = this.#el("div", "lector-cmt-thread__props");
    const iconGroup = this.#el("div", "lector-cmt-thread__props-group");
    const iconLabel = this.#el("div", "lector-cmt-thread__props-label");
    iconLabel.textContent = this.#t("annotation.icon").toUpperCase();
    iconGroup.appendChild(iconLabel);
    const iconRow = this.#el("div", "lector-cmt-thread__props-row");
    const NOTE_ICONS = [
      { key: "Comment", icon: "message-square" },
      { key: "Note", icon: "sticky-note" },
      { key: "Help", icon: "help-circle" },
      { key: "Insert", icon: "plus" },
      { key: "Check", icon: "check" },
      { key: "Cross", icon: "x" },
      { key: "Star", icon: "star" },
      { key: "Circle", icon: "circle" },
      { key: "Key", icon: "key" }
    ];
    const currentIcon = annot.noteIcon ?? "Comment";
    for (const ni of NOTE_ICONS) {
      const btn = this.#btn("lector-cmt-thread__props-icon");
      btn.appendChild(this.#icon(ni.icon));
      btn.title = ni.key;
      btn.setAttribute("aria-label", ni.key);
      if (currentIcon === ni.key) btn.classList.add("lector-cmt-thread__props-icon--active");
      btn.addEventListener("click", () => {
        void this.#annotation.update(docId, annotId, { noteIcon: ni.key }).then(() => {
          for (const b of iconRow.querySelectorAll(".lector-cmt-thread__props-icon")) {
            b.classList.toggle("lector-cmt-thread__props-icon--active", b === btn);
          }
          this.#overlays.rebuildOverlays();
        });
      });
      iconRow.appendChild(btn);
    }
    iconGroup.appendChild(iconRow);
    props.appendChild(iconGroup);
    const colorGroup = this.#el("div", "lector-cmt-thread__props-group");
    const colorLabel = this.#el("div", "lector-cmt-thread__props-label");
    colorLabel.textContent = this.#t("annotation.color").toUpperCase();
    colorGroup.appendChild(colorLabel);
    const colorRow = this.#el("div", "lector-cmt-thread__props-row");
    const COLORS = [
      { r: 228, g: 66, b: 52 },
      { r: 245, g: 158, b: 11 },
      { r: 255, g: 205, b: 69 },
      { r: 34, g: 197, b: 94 },
      { r: 59, g: 130, b: 246 },
      { r: 139, g: 92, b: 246 },
      { r: 0, g: 0, b: 0 }
    ];
    const currentColor = annot.color;
    for (const c of COLORS) {
      const btn = this.#btn("lector-cmt-thread__props-color");
      btn.setAttribute("aria-label", `${this.#t("annotation.color")}: rgb(${c.r}, ${c.g}, ${c.b})`);
      btn.style.background = `rgb(${c.r}, ${c.g}, ${c.b})`;
      if (currentColor && currentColor.a > 0 && currentColor.r === c.r && currentColor.g === c.g && currentColor.b === c.b) {
        btn.classList.add("lector-cmt-thread__props-color--active");
      }
      btn.addEventListener("click", () => {
        void this.#annotation.update(docId, annotId, {
          color: { r: c.r, g: c.g, b: c.b, a: 255 }
        }).then(() => {
          for (const b of colorRow.querySelectorAll(".lector-cmt-thread__props-color")) {
            b.classList.toggle("lector-cmt-thread__props-color--active", b === btn);
          }
          this.#overlays.rebuildOverlays();
        });
      });
      colorRow.appendChild(btn);
    }
    colorGroup.appendChild(colorRow);
    props.appendChild(colorGroup);
    parent.appendChild(props);
  }
  /**
   * Compose section inside the expanded thread:
   *   [textarea]
   *   Resolve · Delete                                Cancel  Reply
   *
   * Includes @-mention autocomplete (resolved via engine.mentionUsers)
   * with a dropdown anchored under the textarea. Posting either
   * patches `contents` (first comment) or appends to `comments[]`
   * (replies). The footer row also holds the thread-level Resolve and
   * Delete actions as plain text links — no popover, no menu.
   */
  #buildSidebarCompose(parent, docId, annotId, annot) {
    if (!this.#annotation) return;
    const user = this.#engine.user;
    const compose = this.#el("div", "lector-cmt-compose");
    const taWrap = this.#el("div", "lector-cmt-compose__textarea-wrap");
    const textarea = document.createElement("textarea");
    textarea.className = "lector-cmt-compose__textarea";
    textarea.placeholder = annot.contents ? this.#t("comment.replyOrMention") || "Reply or use @ to mention\u2026" : this.#t("comment.commentOrMention") || "Comment or use @ to mention\u2026";
    textarea.rows = 3;
    textarea.addEventListener("keydown", (e) => e.stopPropagation());
    taWrap.appendChild(textarea);
    compose.appendChild(taWrap);
    const mentionResolver = this.#engine.mentionUsers;
    let mentionDropdown = null;
    let mentionQuery = "";
    let mentionStart = -1;
    const pendingMentions = [];
    const closeMentionDropdown = () => {
      if (mentionDropdown) {
        mentionDropdown.remove();
        mentionDropdown = null;
      }
      mentionStart = -1;
      mentionQuery = "";
    };
    const insertMention = (mentionUser) => {
      const before = textarea.value.substring(0, mentionStart);
      const after = textarea.value.substring(mentionStart + 1 + mentionQuery.length);
      const mentionText = `@${mentionUser.name}`;
      textarea.value = `${before}${mentionText} ${after}`;
      pendingMentions.push({
        userId: mentionUser.id,
        userName: mentionUser.name,
        offset: mentionStart,
        length: mentionText.length
      });
      closeMentionDropdown();
      textarea.focus();
      const cursorPos = mentionStart + mentionText.length + 1;
      textarea.setSelectionRange(cursorPos, cursorPos);
    };
    const showMentionDropdown = async (query) => {
      if (!mentionResolver) return;
      const results = typeof mentionResolver === "function" ? await mentionResolver(query) : mentionResolver.filter((u) => u.name.toLowerCase().includes(query.toLowerCase()));
      if (results.length === 0) {
        closeMentionDropdown();
        return;
      }
      if (!mentionDropdown) {
        mentionDropdown = this.#el("div", "lector-cmt-mention-dropdown");
        taWrap.appendChild(mentionDropdown);
      }
      mentionDropdown.innerHTML = "";
      for (const u of results.slice(0, 8)) {
        const item = this.#btn("lector-cmt-mention-item");
        const av = this.#el("div", "lector-cmt-avatar lector-cmt-avatar--sm");
        av.textContent = (u.name.charAt(0) || "?").toUpperCase();
        item.appendChild(av);
        const nameEl = this.#el("span", "lector-cmt-mention-item__name");
        nameEl.textContent = u.name;
        item.appendChild(nameEl);
        item.addEventListener("click", (e) => {
          e.stopPropagation();
          insertMention(u);
        });
        mentionDropdown.appendChild(item);
      }
    };
    if (mentionResolver) {
      textarea.addEventListener("input", () => {
        const pos = textarea.selectionStart;
        const text = textarea.value;
        let atIdx = -1;
        for (let i = pos - 1; i >= 0; i--) {
          const ch = text[i];
          if (ch === "@") {
            atIdx = i;
            break;
          }
          if (ch === " " || ch === "\n") break;
        }
        if (atIdx >= 0 && (atIdx === 0 || /\s/.test(text[atIdx - 1] ?? ""))) {
          mentionStart = atIdx;
          mentionQuery = text.substring(atIdx + 1, pos);
          void showMentionDropdown(mentionQuery);
        } else {
          closeMentionDropdown();
        }
      });
      textarea.addEventListener("blur", () => {
        setTimeout(closeMentionDropdown, 200);
      });
      textarea.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && mentionDropdown) {
          e.stopPropagation();
          closeMentionDropdown();
        }
      });
    }
    const actions = this.#el("div", "lector-cmt-compose__actions");
    const resolveLink = this.#btn("lector-cmt-link");
    resolveLink.textContent = annot.resolved ? this.#t("comment.reopen") || "Reopen" : this.#t("comment.resolve") || "Resolve";
    resolveLink.addEventListener("click", (e) => {
      e.stopPropagation();
      void this.#annotation.toggleResolved(docId, annotId);
    });
    actions.appendChild(resolveLink);
    const deleteLink = this.#btn("lector-cmt-link lector-cmt-link--danger");
    deleteLink.textContent = this.#t("annotation.delete") || "Delete";
    deleteLink.addEventListener("click", (e) => {
      e.stopPropagation();
      void this.#annotation.delete(docId, annotId);
      this.#activeThreadAnnotId = null;
    });
    actions.appendChild(deleteLink);
    const spacer = this.#el("span", "lector-cmt-compose__actions-spacer");
    actions.appendChild(spacer);
    const cancelBtn = this.#btn("lector-cmt-btn");
    cancelBtn.textContent = this.#t("comment.cancel") || "Cancel";
    cancelBtn.addEventListener("click", () => {
      textarea.value = "";
      pendingMentions.length = 0;
      closeMentionDropdown();
    });
    actions.appendChild(cancelBtn);
    const postBtn = this.#btn("lector-cmt-btn lector-cmt-btn--primary");
    postBtn.textContent = annot.contents ? this.#t("comment.reply") || "Reply" : this.#t("comment.post") || "Comment";
    postBtn.addEventListener("click", () => {
      const text = textarea.value.trim();
      if (!text) return;
      const mentions = pendingMentions.filter((m) => text.substring(m.offset, m.offset + m.length) === `@${m.userName}`).map((m) => ({
        userId: m.userId,
        userName: m.userName,
        offset: m.offset,
        length: m.length
      }));
      if (!annot.contents) {
        void this.#annotation.update(docId, annotId, { contents: text });
      } else {
        const newComment = {
          id: uuid(),
          authorId: user?.id ?? "",
          authorName: user?.name ?? "Unknown",
          text,
          timestamp: (/* @__PURE__ */ new Date()).toISOString(),
          mentions: mentions.length > 0 ? mentions : void 0
        };
        const existing = annot.comments ?? [];
        void this.#annotation.update(docId, annotId, {
          comments: [...existing, newComment]
        });
      }
      for (const m of mentions) {
        this.#engine.plugins.events.emit("comment:mention", {
          annotationId: annotId,
          documentId: docId,
          mentionedUserId: m.userId,
          mentionedUserName: m.userName,
          authorId: user?.id,
          authorName: user?.name,
          text
        });
      }
      textarea.value = "";
      pendingMentions.length = 0;
    });
    actions.appendChild(postBtn);
    compose.appendChild(actions);
    parent.appendChild(compose);
  }
  /** Scroll the active thread card into view inside the sidebar. */
  #scrollSidebarToActiveThread() {
    if (!this.#activeThreadAnnotId) return;
    const card = this.#commentsSidebarBody.querySelector(
      `[data-annot-id="${this.#activeThreadAnnotId}"]`
    );
    if (card) {
      card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }
  // ─── Attachments Panel ─────────────────────────────────
  #buildAttachmentsPanel(container) {
    if (!this.#attachment) {
      container.textContent = this.#t("attachments.pluginMissing");
      return;
    }
    const handle = this.#document.activeDocument.peek();
    if (!handle) return;
    const wrap3 = this.#el("div", "lector-attachment-panel");
    container.appendChild(wrap3);
    const addRow = this.#el("div", "lector-attachment-panel__actions");
    const addBtn = this.#btn("lector-attachment-panel__add");
    addBtn.appendChild(this.#icon("plus"));
    const addLabel = this.#el("span", "");
    addLabel.textContent = ` ${this.#t("attachment.addFile")}`;
    addLabel.style.fontSize = "var(--lector-font-size-sm)";
    addBtn.appendChild(addLabel);
    addBtn.addEventListener("click", () => {
      const fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.addEventListener("change", () => {
        const file = fileInput.files?.[0];
        if (!file || !this.#attachment) return;
        void file.arrayBuffer().then((data) => {
          void this.#attachment.add(handle.id, file.name, data).then(() => {
            this.#showToast(this.#t("toast.attached", { name: file.name }));
            this.#updateSidebarActiveTab();
          });
        });
      });
      fileInput.click();
    });
    addRow.appendChild(addBtn);
    wrap3.appendChild(addRow);
    void this.#attachment.list(handle.id).then((attachments) => {
      if (attachments.length === 0) {
        const empty = this.#el("div", "lector-sidebar-empty");
        empty.textContent = this.#t("attachments.empty");
        wrap3.appendChild(empty);
        return;
      }
      for (const att of attachments) {
        const card = this.#el("div", "lector-attachment-card");
        const icon = this.#el("span", "lector-attachment-card__icon");
        icon.innerHTML = resolveIcon("paperclip") ?? "";
        card.appendChild(icon);
        const info = this.#el("div", "lector-attachment-card__info");
        const name = this.#el("div", "lector-attachment-card__name");
        name.textContent = att.name;
        name.title = att.name;
        info.appendChild(name);
        const meta = this.#el("div", "lector-attachment-card__meta");
        meta.textContent = this.#formatFileSize(att.size);
        if (att.modDate) {
          meta.textContent += ` \xB7 ${this.#formatDate(att.modDate)}`;
        }
        info.appendChild(meta);
        card.appendChild(info);
        const actionsWrap = this.#el("div", "lector-attachment-card__actions");
        const dlBtn = this.#btn("lector-attachment-card__download");
        dlBtn.innerHTML = resolveIcon("download") ?? "";
        dlBtn.title = this.#t("common.download");
        dlBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          if (!this.#attachment) return;
          void this.#attachment.download(handle.id, att.index).then(({ name: fname, data }) => {
            const blob = new Blob([data]);
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = fname;
            a.click();
            URL.revokeObjectURL(url);
          });
        });
        actionsWrap.appendChild(dlBtn);
        const delBtn = this.#btn("lector-attachment-card__delete");
        delBtn.innerHTML = resolveIcon("trash") ?? "";
        delBtn.title = this.#t("common.delete");
        delBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          if (!this.#attachment) return;
          void this.#attachment.delete(handle.id, att.index).then(() => {
            this.#showToast(this.#t("toast.removed", { name: att.name }));
            this.#updateSidebarActiveTab();
          });
        });
        actionsWrap.appendChild(delBtn);
        card.appendChild(actionsWrap);
        wrap3.appendChild(card);
      }
    });
  }
  #formatDate(input) {
    if (this.#formatting) return this.#formatting.formatDateTime(input) || input;
    try {
      let d;
      const pdfDate = input.replace(/^D:/, "");
      const pdfMatch = /^(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?([Z+-])?(\d{2})?'?(\d{2})?'?$/.exec(pdfDate);
      if (pdfMatch) {
        const [, year, month, day, hh = "00", mm = "00", ss = "00", tzSign, tzH = "00", tzM = "00"] = pdfMatch;
        const tz = tzSign === "Z" || !tzSign ? "Z" : `${tzSign}${tzH}:${tzM}`;
        d = /* @__PURE__ */ new Date(`${year}-${month}-${day}T${hh}:${mm}:${ss}${tz}`);
      } else {
        d = new Date(input);
      }
      if (isNaN(d.getTime())) return input;
      const locale = typeof navigator !== "undefined" ? navigator.language : "en-US";
      return new Intl.DateTimeFormat(locale, { dateStyle: "medium", timeStyle: "short" }).format(d);
    } catch {
      return input;
    }
  }
  #formatFileSize(bytes) {
    if (this.#formatting) return this.#formatting.formatFileSize(bytes);
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  // ─── Signatures Panel ──────────────────────────────────
  #buildSignaturesPanel(container) {
    if (!this.#signature) {
      container.textContent = this.#t("signatures.pluginMissing");
      return;
    }
    const handle = this.#document.activeDocument.peek();
    if (!handle) return;
    const wrap3 = this.#el("div", "lector-signature-panel");
    container.appendChild(wrap3);
    const renderPanel = (sigs, validationResults, validationError, validationPending) => {
      wrap3.innerHTML = "";
      if (sigs.length === 0) {
        const empty = this.#el("div", "lector-sidebar-empty");
        empty.textContent = this.#t("signatures.empty");
        wrap3.appendChild(empty);
        return;
      }
      if (validationPending) {
        const banner = this.#el("div", "lector-signature-panel__loading");
        banner.style.cssText = "padding:8px 12px;font-size:12px;color:var(--lector-fg-muted);font-style:italic";
        banner.textContent = this.#t("signatures.validating");
        wrap3.appendChild(banner);
      }
      this.#renderSignatureCards(wrap3, sigs, validationResults, validationError);
    };
    const cachedSigs = this.#sigInfoCache.get(handle.id);
    const sigsPromise = cachedSigs !== void 0 ? Promise.resolve(cachedSigs) : this.#signature.getAllInfo(handle.id).then((sigs) => {
      this.#sigInfoCache.set(handle.id, sigs);
      return sigs;
    });
    void sigsPromise.then((sigs) => {
      if (sigs.length === 0) {
        renderPanel(sigs, null, null, false);
        return;
      }
      if (!this.#sigValidation) {
        renderPanel(sigs, null, "signature-validation plugin not registered", false);
        return;
      }
      const cached = this.#validationCache.get(handle.id);
      if (Array.isArray(cached)) {
        renderPanel(sigs, cached, null, false);
        return;
      }
      if (cached && typeof cached === "object" && "error" in cached) {
        renderPanel(sigs, null, cached.error, false);
        return;
      }
      renderPanel(sigs, null, null, true);
      if (cached === "pending") return;
      this.#validationCache.set(handle.id, "pending");
      this.#updateSignatureStatusBadge();
      void this.#sigValidation.validateAll(handle.id).then((results) => {
        this.#validationCache.set(handle.id, results);
        this.#updateSignatureStatusBadge();
        if (this.#ui.state.sidebar.activePanel.peek() === "signatures") {
          renderPanel(sigs, results, null, false);
        }
      }).catch((err) => {
        const msg = err instanceof Error ? err.message : typeof err === "object" && err !== null && "message" in err ? String(err.message) : String(err);
        this.#validationCache.set(handle.id, { error: msg });
        this.#updateSignatureStatusBadge();
        if (this.#ui.state.sidebar.activePanel.peek() === "signatures") {
          renderPanel(sigs, null, msg, false);
        }
      });
    });
  }
  /**
   * Render the signature card list into the panel wrap. Extracted from
   * #buildSignaturesPanel so the panel can be re-rendered cheaply when
   * validation results arrive asynchronously.
   */
  #renderSignatureCards(wrap3, sigs, validationResults, validationError) {
    for (let i = 0; i < sigs.length; i++) {
      const sig = sigs[i];
      const validation = validationResults?.[i] ?? null;
      const card = this.#el("div", "lector-signature-card");
      const icon = this.#el("div", "lector-signature-card__icon");
      if (validation) {
        const statusIcon = validation.status === "valid" ? "check-circle" : validation.status === "invalid" ? "x-circle" : "alert-circle";
        const statusColor = validation.status === "valid" ? "var(--lector-success, #16a34a)" : validation.status === "invalid" ? "var(--lector-danger, #ef4444)" : "var(--lector-fg-muted)";
        icon.innerHTML = resolveIcon(statusIcon) ?? resolveIcon("shield") ?? "";
        icon.style.color = statusColor;
      } else {
        icon.innerHTML = resolveIcon("shield") ?? "";
      }
      card.appendChild(icon);
      const info = this.#el("div", "lector-signature-card__info");
      const title = this.#el("div", "lector-signature-card__title");
      if (validation?.signerCertificate?.subject) {
        title.textContent = validation.signerCertificate.subject;
      } else {
        title.textContent = this.#t("signatures.itemTitle", { index: sig.index + 1 });
      }
      info.appendChild(title);
      if (validation) {
        const badge = this.#el("span", "lector-signature-card__badge");
        if (validation.status === "valid") {
          badge.textContent = this.#t("signatures.badgeValid");
          badge.classList.add("lector-signature-card__badge--valid");
        } else if (validation.status === "invalid") {
          badge.textContent = this.#t("signatures.badgeInvalid");
          badge.classList.add("lector-signature-card__badge--invalid");
        } else if (validation.status === "error") {
          badge.textContent = this.#t("signatures.badgeError");
          badge.classList.add("lector-signature-card__badge--error");
        } else {
          badge.textContent = this.#t("signatures.badgeUnknown");
          badge.classList.add("lector-signature-card__badge--unknown");
        }
        info.appendChild(badge);
      }
      if (validation) {
        if (validation.integrityValid) {
          const row = this.#el("div", "lector-signature-card__detail lector-signature-card__detail--ok");
          row.textContent = this.#t("signatures.integrityVerified");
          info.appendChild(row);
        } else if (validation.status !== "error") {
          const row = this.#el("div", "lector-signature-card__detail lector-signature-card__detail--fail");
          row.textContent = this.#t("signatures.integrityModified");
          info.appendChild(row);
        }
        if (validation.signatureValid) {
          const row = this.#el("div", "lector-signature-card__detail lector-signature-card__detail--ok");
          row.textContent = this.#t("signatures.cryptoValid");
          info.appendChild(row);
        }
        if (validation.signerCertificate) {
          const cert = validation.signerCertificate;
          if (cert.isExpired) {
            const row = this.#el("div", "lector-signature-card__detail lector-signature-card__detail--fail");
            row.textContent = this.#t("signatures.certExpired");
            info.appendChild(row);
          }
          if (cert.isSelfSigned) {
            const row = this.#el("div", "lector-signature-card__detail lector-signature-card__detail--warn");
            row.textContent = this.#t("signatures.certSelfSigned");
            info.appendChild(row);
          }
          if (cert.issuer && cert.issuer !== cert.subject) {
            const row = this.#el("div", "lector-signature-card__meta");
            row.textContent = this.#t("signatures.certIssuer", { issuer: cert.issuer });
            info.appendChild(row);
          }
        }
        if (validation.hashAlgorithm) {
          const row = this.#el("div", "lector-signature-card__meta");
          row.textContent = this.#t("signatures.algorithm", {
            algorithm: validation.hashAlgorithm
          });
          info.appendChild(row);
        }
        if (validation.errorMessage) {
          const row = this.#el("div", "lector-signature-card__detail lector-signature-card__detail--fail");
          row.textContent = validation.errorMessage;
          info.appendChild(row);
        }
      }
      if (sig.subFilter) {
        const sfKeys = {
          "adbe.pkcs7.detached": "signatures.format.pkcs7Detached",
          "adbe.pkcs7.sha1": "signatures.format.pkcs7Sha1",
          "adbe.x509.rsa_sha1": "signatures.format.x509RsaSha1",
          "ETSI.CAdES.detached": "signatures.format.cadesDetached",
          "ETSI.RFC3161": "signatures.format.rfc3161"
        };
        const formatKey = sfKeys[sig.subFilter];
        const formatLabel = formatKey ? this.#t(formatKey) : sig.subFilter;
        const subRow = this.#el("div", "lector-signature-card__meta");
        subRow.textContent = this.#t("signatures.format", { format: formatLabel });
        info.appendChild(subRow);
      }
      if (sig.reason) {
        const reason = this.#el("div", "lector-signature-card__meta");
        reason.textContent = this.#t("signatures.reason", { reason: sig.reason });
        info.appendChild(reason);
      }
      if (sig.time) {
        const time = this.#el("div", "lector-signature-card__meta");
        time.textContent = this.#t("signatures.signedAt", {
          time: this.#formatDate(sig.time)
        });
        info.appendChild(time);
      }
      if (sig.byteRange && sig.byteRange.length >= 4) {
        const totalCovered = sig.byteRange[1] + sig.byteRange[3];
        const range = this.#el("div", "lector-signature-card__meta");
        range.textContent = this.#t("signatures.covers", {
          size: this.#formatFileSize(totalCovered)
        });
        info.appendChild(range);
      }
      const permKeys = {
        0: "signatures.perm.approval",
        1: "signatures.perm.noChanges",
        2: "signatures.perm.formSigning",
        3: "signatures.perm.formCommenting"
      };
      const perm = this.#el("div", "lector-signature-card__meta");
      const permKey = permKeys[sig.docMDPPermission];
      perm.textContent = permKey ? this.#t(permKey) : this.#t("signatures.perm.unknown", { level: sig.docMDPPermission });
      info.appendChild(perm);
      if (!validation && validationError) {
        const hint = this.#el("div", "lector-signature-card__meta");
        hint.style.cssText = "color:var(--lector-fg-muted);font-style:italic;margin-top:4px";
        hint.textContent = this.#t("signatures.validationHint", { error: validationError });
        info.appendChild(hint);
      }
      card.appendChild(info);
      wrap3.appendChild(card);
    }
  }
  /**
   * Refresh the toolbar's Compare button state. Driven reactively from
   * `#wireEffects` whenever the comparison plugin's state, the active
   * tab, or the active document set changes.
   */
  #updateCompareButton() {
    const btn = this.#compareBtn;
    if (!btn || !this.#comparison) return;
    const state = this.#comparison.state.peek();
    const pair = this.#activeComparisonPair();
    btn.classList.remove("lector-btn--active");
    btn.disabled = false;
    const label = btn.querySelector(".lector-btn__label");
    if (state === "active" || state === "computing") {
      btn.classList.add("lector-btn--active");
      if (label) label.textContent = this.#t("comparison.exit");
    } else {
      if (label) label.textContent = this.#t("comparison.compare");
      if (!pair) {
        btn.disabled = true;
      }
    }
  }
  // ─── Comparison ────────────────────────────────────────
  /**
   * Resolve the active split tab's two doc ids, or null if the active
   * tab is not a split with two distinct documents loaded. The Compare
   * button uses this to gate its enabled state, and `#startComparison`
   * uses it to know which docs to diff.
   */
  #activeComparisonPair() {
    const tab = this.#activeTab();
    if (!tab || tab.kind !== "split") return null;
    if (!tab.right) return null;
    if (tab.left.docId === tab.right.docId) return null;
    return { docA: tab.left.docId, docB: tab.right.docId };
  }
  /**
   * Start comparing the two docs in the current split tab. Wires up the
   * sync-scroll listeners, opens the Changes sidebar panel, and pushes
   * the result onto both overlay managers when it resolves.
   */
  #startComparison() {
    if (!this.#comparison) return;
    const pair = this.#activeComparisonPair();
    if (!pair) return;
    void this.#comparison.enter(pair.docA, pair.docB);
  }
  /** Exit compare mode. Idempotent. */
  #exitComparison() {
    if (!this.#comparison) return;
    this.#comparison.exit();
  }
  /**
   * Find the LectorPane currently displaying `docId` (the split right
   * pane), if any. Returns null when no extra pane has that doc pinned —
   * e.g. when the active tab is a single doc, or while the dynamic
   * import is still resolving.
   */
  #findExtraPaneForDoc(docId) {
    for (const pane of this.#extraPanes.values()) {
      if (pane.viewport.docId.peek() === docId) return pane;
    }
    return null;
  }
  /**
   * Push the active comparison result onto both panes' overlay managers
   * (or clear them when `result` is null). Side A is always the primary
   * canvas (left pane), side B is the right pane.
   */
  #applyComparisonOverlays(result) {
    const pair = this.#comparison?.activePair.peek();
    if (!result || !pair) {
      this.#overlays.setComparison("A", null);
      for (const pane of this.#extraPanes.values()) {
        pane.overlays.setComparison("B", null);
      }
      return;
    }
    this.#overlays.setComparison("A", result.pageDiffs, this.#activeChangeIndex);
    const rightPane = this.#findExtraPaneForDoc(pair.docB);
    if (rightPane) {
      rightPane.overlays.setComparison("B", result.pageDiffs, this.#activeChangeIndex);
    }
  }
  /**
   * Scroll both panes so the change at `flatIndex` is in view, and
   * mark it active so its highlight pulses on both sides.
   */
  #focusChange(flatIndex) {
    if (!this.#comparison) return;
    const result = this.#comparison.result.peek();
    if (!result) return;
    let i = 0;
    let target = null;
    for (const diff of result.pageDiffs) {
      for (const change of diff.changes) {
        if (i === flatIndex) {
          target = { diff, change };
          break;
        }
        i++;
      }
      if (target) break;
    }
    if (!target) return;
    this.#activeChangeIndex = flatIndex;
    this.#overlays.setComparisonActiveIndex(flatIndex);
    for (const pane of this.#extraPanes.values()) {
      pane.overlays.setComparisonActiveIndex(flatIndex);
    }
    this.#syncScrollLock = true;
    try {
      const pageA = target.diff.pageA;
      if (pageA != null) {
        this.#viewportInstance.scrollToPage(pageA, false);
      }
      const pageB = target.diff.pageB;
      if (pageB != null) {
        const pair = this.#comparison.activePair.peek();
        if (pair) {
          const rightPane = this.#findExtraPaneForDoc(pair.docB);
          if (rightPane) rightPane.viewport.scrollToPage(pageB);
        }
      }
    } finally {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          this.#syncScrollLock = false;
        });
      });
    }
    this.#refreshComparisonPanelActive();
  }
  /**
   * Filtered list of flat change indices according to `#compareFilter`.
   * Used by both the sidebar panel and prev/next navigation.
   */
  #filteredChangeIndices() {
    if (!this.#comparison) return [];
    const result = this.#comparison.result.peek();
    if (!result) return [];
    const out = [];
    let i = 0;
    for (const diff of result.pageDiffs) {
      for (const change of diff.changes) {
        const t = change.type === "region" ? "replace" : change.type;
        if (this.#compareFilter === "all" || this.#compareFilter === t) {
          out.push(i);
        }
        i++;
      }
    }
    return out;
  }
  #nextChange() {
    const filtered = this.#filteredChangeIndices();
    if (filtered.length === 0) return;
    const cur = filtered.indexOf(this.#activeChangeIndex);
    const nextIdx = cur < 0 ? filtered[0] : filtered[(cur + 1) % filtered.length];
    this.#focusChange(nextIdx);
  }
  #prevChange() {
    const filtered = this.#filteredChangeIndices();
    if (filtered.length === 0) return;
    const cur = filtered.indexOf(this.#activeChangeIndex);
    const prevIdx = cur < 0 ? filtered[filtered.length - 1] : filtered[(cur - 1 + filtered.length) % filtered.length];
    this.#focusChange(prevIdx);
  }
  /**
   * Wire pane-to-pane scroll sync. When the user scrolls one pane, the
   * other follows to the matched page from the diff alignment. Cleared
   * via `#unwireComparisonSyncScroll()` on exit.
   */
  #wireComparisonSyncScroll() {
    this.#unwireComparisonSyncScroll();
    if (!this.#comparison) return;
    const pair = this.#comparison.activePair.peek();
    if (!pair) return;
    const result = this.#comparison.result.peek();
    if (!result) return;
    const aToB = /* @__PURE__ */ new Map();
    const bToA = /* @__PURE__ */ new Map();
    for (const d of result.pageDiffs) {
      if (d.pageA != null && d.pageB != null) {
        aToB.set(d.pageA, d.pageB);
        bToA.set(d.pageB, d.pageA);
      }
    }
    const rightPane = this.#findExtraPaneForDoc(pair.docB);
    if (!rightPane) return;
    const leftVp = this.#viewportInstance;
    const rightVp = rightPane.viewport;
    this.#compareSyncCleanups.push(effect2(() => {
      const vis = leftVp.visiblePages.value;
      if (this.#syncScrollLock || vis.length === 0) return;
      const top = vis[0];
      const matched = aToB.get(top);
      if (matched == null || matched < 0) return;
      const rightVis = rightVp.visiblePages.peek();
      if (rightVis[0] === matched) return;
      this.#syncScrollLock = true;
      try {
        rightVp.scrollToPage(matched);
      } finally {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            this.#syncScrollLock = false;
          });
        });
      }
    }));
    this.#compareSyncCleanups.push(effect2(() => {
      const vis = rightVp.visiblePages.value;
      if (this.#syncScrollLock || vis.length === 0) return;
      const top = vis[0];
      const matched = bToA.get(top);
      if (matched == null || matched < 0) return;
      const leftVis = leftVp.visiblePages.peek();
      if (leftVis[0] === matched) return;
      this.#syncScrollLock = true;
      try {
        leftVp.scrollToPage(matched);
      } finally {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            this.#syncScrollLock = false;
          });
        });
      }
    }));
  }
  #unwireComparisonSyncScroll() {
    for (const u of this.#compareSyncCleanups) u();
    this.#compareSyncCleanups.length = 0;
  }
  /**
   * Build the Changes sidebar panel. Lists every comparison change
   * grouped by page, with type chips for filtering. Clicking a row
   * focuses the change in both panes.
   */
  #buildComparisonPanel(container) {
    container.innerHTML = "";
    const wrap3 = this.#el("div", "lector-compare-panel");
    container.appendChild(wrap3);
    if (!this.#comparison) {
      const empty = this.#el("div", "lector-sidebar-empty");
      empty.textContent = this.#t("comparison.requireSplit");
      wrap3.appendChild(empty);
      return;
    }
    const state = this.#comparison.state.peek();
    const result = this.#comparison.result.peek();
    const header = this.#el("div", "lector-compare-panel__header");
    wrap3.appendChild(header);
    const title = this.#el("div", "lector-compare-panel__title");
    title.textContent = this.#t("comparison.title");
    header.appendChild(title);
    if (state === "active" && result) {
      const navWrap = this.#el("div", "lector-compare-panel__nav");
      const prevBtn = this.#btn("lector-btn");
      prevBtn.appendChild(this.#icon("chevron-up"));
      this.#tip(prevBtn, this.#t("comparison.prevChange"));
      prevBtn.addEventListener("click", () => this.#prevChange());
      navWrap.appendChild(prevBtn);
      const nextBtn = this.#btn("lector-btn");
      nextBtn.appendChild(this.#icon("chevron-down"));
      this.#tip(nextBtn, this.#t("comparison.nextChange"));
      nextBtn.addEventListener("click", () => this.#nextChange());
      navWrap.appendChild(nextBtn);
      const exitBtn = this.#btn("lector-btn");
      exitBtn.appendChild(this.#icon("x"));
      this.#tip(exitBtn, this.#t("comparison.exit"));
      exitBtn.addEventListener("click", () => this.#exitComparison());
      navWrap.appendChild(exitBtn);
      header.appendChild(navWrap);
    }
    if (state === "computing") {
      const msg = this.#el("div", "lector-sidebar-empty");
      msg.textContent = this.#t("comparison.computing");
      wrap3.appendChild(msg);
      return;
    }
    if (state === "error") {
      const err = this.#el("div", "lector-sidebar-empty");
      err.textContent = this.#t("comparison.error", {
        error: this.#comparison.error.peek() ?? "unknown error"
      });
      wrap3.appendChild(err);
      return;
    }
    if (state === "inactive" || !result) {
      const empty = this.#el("div", "lector-sidebar-empty");
      empty.textContent = this.#t("comparison.requireSplit");
      wrap3.appendChild(empty);
      const pair = this.#activeComparisonPair();
      if (pair) {
        const cta = this.#btn("lector-btn");
        cta.appendChild(this.#icon("git-compare"));
        const lbl = this.#el("span", "lector-btn__label");
        lbl.textContent = this.#t("comparison.compare");
        cta.appendChild(lbl);
        cta.style.marginTop = "12px";
        cta.style.alignSelf = "center";
        cta.addEventListener("click", () => this.#startComparison());
        wrap3.appendChild(cta);
      }
      return;
    }
    const totalCount = result.totalChanges;
    const summary = this.#el("div", "lector-compare-panel__summary");
    if (totalCount === 0) {
      summary.textContent = this.#t("comparison.identical");
    } else {
      summary.textContent = totalCount === 1 ? this.#t("comparison.totalChanges", { count: totalCount }) : this.#t("comparison.totalChangesPlural", { count: totalCount });
    }
    wrap3.appendChild(summary);
    if (totalCount === 0) return;
    const chips = this.#el("div", "lector-compare-panel__chips");
    const chipDefs = [
      { id: "all", label: this.#t("comparison.filterAll") },
      { id: "insert", label: this.#t("comparison.filterInsert") },
      { id: "delete", label: this.#t("comparison.filterDelete") },
      { id: "replace", label: this.#t("comparison.filterReplace") }
    ];
    for (const def of chipDefs) {
      const c = this.#btn("lector-compare-chip");
      c.classList.toggle("lector-compare-chip--active", this.#compareFilter === def.id);
      c.classList.add(`lector-compare-chip--${def.id}`);
      c.textContent = def.label;
      c.addEventListener("click", () => {
        this.#compareFilter = def.id;
        this.#buildComparisonPanel(container);
      });
      chips.appendChild(c);
    }
    wrap3.appendChild(chips);
    const list = this.#el("div", "lector-compare-panel__list");
    wrap3.appendChild(list);
    let flatIdx = 0;
    for (const diff of result.pageDiffs) {
      const groupIndices = [];
      let probe = flatIdx;
      for (const change of diff.changes) {
        const t = change.type === "region" ? "replace" : change.type;
        if (this.#compareFilter === "all" || this.#compareFilter === t) {
          groupIndices.push(probe);
        }
        probe++;
      }
      if (groupIndices.length === 0) {
        flatIdx += diff.changes.length;
        continue;
      }
      const heading = this.#el("div", "lector-compare-group__heading");
      heading.textContent = this.#comparisonHeadingForDiff(diff);
      list.appendChild(heading);
      if (diff.mode === "region" || diff.mode === "mismatched") {
        const badge = this.#el("span", `lector-compare-group__badge lector-compare-group__badge--${diff.mode}`);
        badge.textContent = diff.mode === "region" ? this.#t("comparison.changeRegion") : this.#t("comparison.pageHeadingMismatch", { a: "", b: "" }).split("(")[1]?.replace(")", "") ?? "mixed";
        heading.appendChild(badge);
      }
      for (const change of diff.changes) {
        const t = change.type === "region" ? "replace" : change.type;
        if (this.#compareFilter !== "all" && this.#compareFilter !== t) {
          flatIdx++;
          continue;
        }
        const row = this.#el("button", "lector-compare-row");
        row.type = "button";
        row.classList.add(`lector-compare-row--${change.type}`);
        if (flatIdx === this.#activeChangeIndex) {
          row.classList.add("lector-compare-row--active");
        }
        row.dataset["flatIdx"] = String(flatIdx);
        const iconName = change.type === "insert" ? "plus-square" : change.type === "delete" ? "minus-square" : change.type === "region" ? "git-compare" : "replace";
        const icnSpan = this.#el("span", "lector-compare-row__icon");
        icnSpan.innerHTML = resolveIcon(iconName) ?? "";
        row.appendChild(icnSpan);
        const text = this.#el("div", "lector-compare-row__text");
        if (change.type === "replace") {
          const before = this.#el("div", "lector-compare-row__before");
          before.textContent = change.textBefore ?? "";
          text.appendChild(before);
          const after = this.#el("div", "lector-compare-row__after");
          after.textContent = change.textAfter ?? "";
          text.appendChild(after);
        } else if (change.type === "insert") {
          const after = this.#el("div", "lector-compare-row__after");
          after.textContent = change.textAfter ?? "";
          text.appendChild(after);
        } else if (change.type === "delete") {
          const before = this.#el("div", "lector-compare-row__before");
          before.textContent = change.textBefore ?? "";
          text.appendChild(before);
        } else {
          const region = this.#el("div", "lector-compare-row__region");
          const pct = change.pixelDelta != null ? ` (${Math.round(change.pixelDelta * 100)}%)` : "";
          region.textContent = `${this.#t("comparison.changeRegion")}${pct}`;
          text.appendChild(region);
        }
        row.appendChild(text);
        const captured = flatIdx;
        row.addEventListener("click", () => this.#focusChange(captured));
        list.appendChild(row);
        flatIdx++;
      }
    }
  }
  /** Map a PageDiff to a localized heading string. */
  #comparisonHeadingForDiff(diff) {
    if (diff.mode === "inserted" && diff.pageB != null) {
      return this.#t("comparison.pageHeadingInsert", { b: diff.pageB + 1 });
    }
    if (diff.mode === "deleted" && diff.pageA != null) {
      return this.#t("comparison.pageHeadingDelete", { a: diff.pageA + 1 });
    }
    if (diff.mode === "mismatched" && diff.pageA != null && diff.pageB != null) {
      return this.#t("comparison.pageHeadingMismatch", { a: diff.pageA + 1, b: diff.pageB + 1 });
    }
    if (diff.mode === "region" && diff.pageA != null && diff.pageB != null) {
      return this.#t("comparison.pageHeadingRegion", { a: diff.pageA + 1, b: diff.pageB + 1 });
    }
    return this.#t("comparison.pageHeading", {
      a: (diff.pageA ?? 0) + 1,
      b: (diff.pageB ?? 0) + 1
    });
  }
  /**
   * Update only the active-row marker inside the change list, without
   * rebuilding the whole panel. Cheaper for prev/next navigation than
   * re-running `#buildComparisonPanel`.
   */
  #refreshComparisonPanelActive() {
    const body = this.#sidebarEl.querySelector(".lector-sidebar__body");
    if (!body) return;
    for (const el of body.querySelectorAll(".lector-compare-row")) {
      const e = el;
      const idx = parseInt(e.dataset["flatIdx"] ?? "-1", 10);
      e.classList.toggle("lector-compare-row--active", idx === this.#activeChangeIndex);
    }
  }
  // ─── Layers Panel ──────────────────────────────────────
  #buildLayersPanel(container) {
    if (!this.#layer) {
      container.textContent = this.#t("layers.pluginMissing");
      return;
    }
    const handle = this.#document.activeDocument.peek();
    if (!handle) return;
    const wrap3 = this.#el("div", "lector-layer-panel");
    container.appendChild(wrap3);
    void this.#layer.loadLayers(handle.id).then(() => {
      const layers = this.#layer.layers.peek();
      if (layers.length === 0) {
        const empty = this.#el("div", "lector-sidebar-empty");
        empty.textContent = this.#t("layers.empty");
        wrap3.appendChild(empty);
        return;
      }
      for (const layer of layers) {
        const row = this.#el("div", "lector-layer-row");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "lector-layer-row__checkbox";
        checkbox.checked = layer.visible;
        checkbox.addEventListener("change", () => {
          if (this.#layer && handle) {
            void this.#layer.setVisible(handle.id, layer.index, checkbox.checked);
          }
        });
        row.appendChild(checkbox);
        const label = this.#el("span", "lector-layer-row__name");
        label.textContent = layer.name || `Layer ${layer.index + 1}`;
        label.title = layer.intent || "";
        row.appendChild(label);
        wrap3.appendChild(row);
      }
    });
  }
  // ─── Stamp Picker ──────────────────────────────────────
  /** Predefined PDF stamp names (standard + custom). */
  static STAMPS = [
    { name: "Approved", label: "Approved", color: "#22c55e" },
    { name: "NotApproved", label: "Not Approved", color: "#ef4444" },
    { name: "Draft", label: "Draft", color: "#f59e0b" },
    { name: "Final", label: "Final", color: "#3b82f6" },
    { name: "Confidential", label: "Confidential", color: "#ef4444" },
    { name: "ForPublicRelease", label: "For Public Release", color: "#22c55e" },
    { name: "NotForPublicRelease", label: "Not For Public Release", color: "#ef4444" },
    { name: "ForComment", label: "For Comment", color: "#f59e0b" },
    { name: "Void", label: "Void", color: "#6b7280" },
    { name: "AsIs", label: "As Is", color: "#6b7280" },
    { name: "Expired", label: "Expired", color: "#ef4444" },
    { name: "Completed", label: "Completed", color: "#22c55e" },
    { name: "InformationOnly", label: "Information Only", color: "#3b82f6" },
    { name: "PreliminaryResults", label: "Preliminary Results", color: "#f59e0b" }
  ];
  #activeStampName = "Approved";
  #buildStampPicker() {
    const wrapper = this.#el("div", "lector-dropdown");
    const trigger = this.#btn("lector-btn");
    trigger.appendChild(this.#icon("stamp"));
    const label = this.#el("span", "lector-btn__label");
    label.textContent = "";
    trigger.appendChild(label);
    this.#tip(trigger, this.#t("stamp.pickerTooltip"));
    trigger.dataset["annotTool"] = "stamp";
    trigger.addEventListener("click", () => {
      if (!this.#annotation) return;
      const current = this.#annotation.activeTool.peek();
      if (current === "stamp") {
        this.#annotation.setActiveTool(null);
      } else {
        this.#annotation.setActiveTool("stamp");
      }
    });
    const menu = this.#el("div", "lector-dropdown__menu lector-stamp-picker");
    for (const s of _LectorViewer.STAMPS) {
      const item = this.#btn("lector-stamp-picker__item");
      const swatch = this.#el("span", "lector-stamp-picker__swatch");
      swatch.style.background = s.color;
      item.appendChild(swatch);
      const lbl = this.#el("span", "lector-stamp-picker__label");
      lbl.textContent = s.label;
      item.appendChild(lbl);
      if (s.name === this.#activeStampName) item.classList.add("lector-stamp-picker__item--active");
      item.addEventListener("click", (e) => {
        e.stopPropagation();
        this.#activeStampName = s.name;
        if (this.#annotation) this.#annotation.activeStampName = s.name;
        for (const el of menu.querySelectorAll(".lector-stamp-picker__item")) {
          el.classList.remove("lector-stamp-picker__item--active");
        }
        item.classList.add("lector-stamp-picker__item--active");
        menu.classList.remove("lector-dropdown__menu--open");
        if (this.#annotation) this.#annotation.setActiveTool("stamp");
      });
      menu.appendChild(item);
    }
    trigger.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      e.stopPropagation();
      menu.classList.toggle("lector-dropdown__menu--open");
    });
    const chevron = this.#btn("lector-stamp-picker__chevron");
    chevron.setAttribute("aria-label", this.#t("stamp.pickerTooltip"));
    chevron.innerHTML = resolveIcon("chevron-down") ?? "";
    chevron.addEventListener("click", (e) => {
      e.stopPropagation();
      this.#hideTooltip();
      menu.classList.toggle("lector-dropdown__menu--open");
    });
    const close = (e) => {
      if (!wrapper.contains(e.target)) menu.classList.remove("lector-dropdown__menu--open");
    };
    document.addEventListener("click", close);
    this.#pushCleanup(() => document.removeEventListener("click", close));
    this.#wireMenu(chevron, menu, "lector-dropdown__menu--open");
    const btnWrap = this.#el("div", "lector-stamp-picker__trigger");
    btnWrap.appendChild(trigger);
    btnWrap.appendChild(chevron);
    wrapper.appendChild(btnWrap);
    wrapper.appendChild(menu);
    return wrapper;
  }
  /** Get the currently selected stamp name for placement. */
  get activeStampName() {
    return this.#activeStampName;
  }
  /**
   * Build the image-annotation toolbar button.
   *
   * Clicking the button:
   *   1. Opens a hidden `<input type="file" accept="image/*">` picker
   *   2. On file pick, decodes the file to a base64 data URI via FileReader
   *   3. Probes natural pixel dimensions via `Image.decode()` (cheaper
   *      than appending to the DOM and waiting for `load`)
   *   4. Stages the decoded image on the annotation plugin
   *   5. Activates the `image` tool so the next page click places it
   *
   * If the user cancels the file picker, no tool activation happens.
   * If the file is too large, we surface a toast and abort. We treat
   * 4 MB as the rough upper bound — anything larger ends up in a PDF
   * dictionary string the worker has to copy on every annotation
   * read, which gets expensive past a few megabytes.
   */
  #buildImageButton() {
    const wrapper = this.#el("div", "lector-toolbar__group");
    const btn = this.#btn("lector-btn");
    btn.appendChild(this.#icon("image-plus"));
    this.#tip(btn, this.#t("annotation.imagePickFile"));
    btn.dataset["annotTool"] = "image";
    btn.addEventListener("click", () => {
      if (!this.#annotation) return;
      if (this.#annotation.activeTool.peek() === "image") {
        this.#annotation.setActiveTool(null);
        this.#annotation.setStagedImage(null);
        return;
      }
      const input = document.createElement("input");
      input.type = "file";
      input.accept = "image/png,image/jpeg,image/webp,image/gif,image/svg+xml";
      input.style.display = "none";
      document.body.appendChild(input);
      input.addEventListener("change", async () => {
        const file = input.files?.[0] ?? null;
        document.body.removeChild(input);
        if (!file) return;
        const FOUR_MB = 4 * 1024 * 1024;
        if (file.size > FOUR_MB) {
          this.#showToast(this.#t("annotation.imageTooLarge"));
          return;
        }
        let dataUri;
        try {
          dataUri = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error ?? new Error("FileReader failed"));
            reader.readAsDataURL(file);
          });
        } catch {
          this.#showToast(this.#t("annotation.imageInvalid"));
          return;
        }
        const img = new Image();
        img.src = dataUri;
        try {
          await img.decode();
        } catch {
          this.#showToast(this.#t("annotation.imageInvalid"));
          return;
        }
        if (!this.#annotation) return;
        this.#annotation.setStagedImage({
          dataUri,
          naturalWidth: img.naturalWidth || img.width || 1,
          naturalHeight: img.naturalHeight || img.height || 1
        });
        this.#annotation.setActiveTool("image");
        this.#showToast(this.#t("annotation.imageHelp"));
      }, { once: true });
      input.addEventListener("cancel", () => {
        if (input.parentNode) input.parentNode.removeChild(input);
      }, { once: true });
      input.click();
    });
    wrapper.appendChild(btn);
    return wrapper;
  }
  // ─── Redaction Dialog ─────────────────────────────────
  #showApplyRedactionsDialog() {
    if (!this.#redaction) return;
    const doc = this.#document.activeDocument.peek();
    if (!doc) return;
    const pending = this.#redaction.getMarkedRedactions(doc.id);
    if (pending.length === 0) {
      this.#showToast(this.#t("toast.noRedactions"));
      return;
    }
    const overlay = this.#el("div", "lector-modal-overlay");
    const dialog = this.#el("div", "lector-modal");
    const title = this.#el("h3", "lector-modal__title");
    title.textContent = this.#t("redact.dialogTitle");
    dialog.appendChild(title);
    const body = this.#el("div", "lector-modal__body");
    const summaryP = this.#el("p", "");
    summaryP.style.margin = "0 0 8px";
    summaryP.textContent = this.#t(
      pending.length === 1 ? "redact.summarySingle" : "redact.summaryPlural",
      { count: pending.length }
    );
    const warningP = this.#el("p", "");
    warningP.style.margin = "0";
    warningP.style.color = "var(--lector-fg-muted)";
    warningP.textContent = this.#t("redact.warning");
    body.appendChild(summaryP);
    body.appendChild(warningP);
    dialog.appendChild(body);
    const actions = this.#el("div", "lector-modal__actions");
    const cancelBtn = this.#btn("lector-modal__btn");
    cancelBtn.textContent = this.#t("common.cancel");
    cancelBtn.addEventListener("click", () => overlay.remove());
    const applyBtn = this.#btn("lector-modal__btn lector-modal__btn--danger");
    applyBtn.textContent = this.#t("redact.applyButton");
    applyBtn.addEventListener("click", async () => {
      applyBtn.disabled = true;
      applyBtn.textContent = this.#t("redact.applying");
      try {
        const pages = new Set(pending.map((t) => t.data.pageIndex));
        for (const pageIdx of pages) {
          await this.#redaction.applyRedactions(doc.id, pageIdx);
        }
        this.#showToast(
          this.#t(
            pending.length === 1 ? "toast.redactionsApplied" : "toast.redactionsAppliedPlural",
            { count: pending.length }
          )
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : JSON.stringify(err);
        this.#showToast(this.#t("toast.redactionsFailed", { error: msg }));
      }
      overlay.remove();
    });
    actions.appendChild(cancelBtn);
    actions.appendChild(applyBtn);
    dialog.appendChild(actions);
    overlay.appendChild(dialog);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.remove();
    });
    this.#openModal(overlay, dialog);
  }
  // ─── Measurement Calibration Dialog ──────────────────
  /**
   * Show the measurement calibration dialog. Lets the user define a
   * scale ratio (PDF distance ↔ real-world distance) plus the active
   * display unit and decimal precision. Calls into the measurement
   * plugin so future measurement annotations embed this scale.
   */
  #showCalibrationDialog() {
    if (!this.#measurement) return;
    const overlay = this.#el("div", "lector-modal-overlay");
    const dialog = this.#el("div", "lector-modal");
    const title = this.#el("h3", "lector-modal__title");
    title.textContent = this.#t("measurement.calibrateTitle");
    dialog.appendChild(title);
    const body = this.#el("div", "lector-modal__body");
    const desc = this.#el("p", "");
    desc.style.margin = "0 0 12px";
    desc.style.color = "var(--lector-fg-secondary)";
    desc.textContent = this.#t("measurement.calibrateDesc");
    body.appendChild(desc);
    const current = this.#measurement.getScale();
    if (current) {
      const cur = this.#el("div", "lector-cal-dialog__current");
      cur.textContent = this.#t("measurement.currentScale", {
        source: String(current.source),
        sourceUnit: current.sourceUnit,
        target: String(current.target),
        targetUnit: current.targetUnit
      });
      cur.style.fontSize = "var(--lector-font-size-sm)";
      cur.style.color = "var(--lector-fg-muted)";
      cur.style.marginBottom = "12px";
      body.appendChild(cur);
    }
    const ALL_UNITS = [
      { value: MeasurementUnit.PT, label: "pt" },
      { value: MeasurementUnit.IN, label: "in" },
      { value: MeasurementUnit.MM, label: "mm" },
      { value: MeasurementUnit.CM, label: "cm" },
      { value: MeasurementUnit.M, label: "m" },
      { value: MeasurementUnit.FT, label: "ft" },
      { value: MeasurementUnit.YD, label: "yd" }
    ];
    const buildField = (label, defaultValue, defaultUnit) => {
      const wrap3 = this.#el("div", "lector-cal-dialog__row");
      const lbl = this.#el("label", "lector-cal-dialog__label");
      lbl.textContent = label;
      wrap3.appendChild(lbl);
      const fieldRow = this.#el("div", "lector-cal-dialog__field");
      const input = document.createElement("input");
      input.type = "number";
      input.min = "0";
      input.step = "any";
      input.value = String(defaultValue);
      input.className = "lector-modal__input lector-cal-dialog__input";
      fieldRow.appendChild(input);
      const select = document.createElement("select");
      select.className = "lector-cal-dialog__select";
      for (const u of ALL_UNITS) {
        const opt = document.createElement("option");
        opt.value = u.value;
        opt.textContent = u.label;
        if (u.value === defaultUnit) opt.selected = true;
        select.appendChild(opt);
      }
      fieldRow.appendChild(select);
      wrap3.appendChild(fieldRow);
      return { wrap: wrap3, input, select };
    };
    const sourceField = buildField(
      this.#t("measurement.pdfDistance"),
      current?.source ?? 1,
      current?.sourceUnit ?? MeasurementUnit.IN
    );
    body.appendChild(sourceField.wrap);
    const targetField = buildField(
      this.#t("measurement.realDistance"),
      current?.target ?? 1,
      current?.targetUnit ?? MeasurementUnit.CM
    );
    body.appendChild(targetField.wrap);
    const precWrap = this.#el("div", "lector-cal-dialog__row");
    const precLbl = this.#el("label", "lector-cal-dialog__label");
    precLbl.textContent = this.#t("measurement.precision");
    precWrap.appendChild(precLbl);
    const precSelect = document.createElement("select");
    precSelect.className = "lector-cal-dialog__select";
    const currentPrec = this.#measurement.precision.peek();
    for (let i = 0; i <= 4; i++) {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = String(i);
      if (i === currentPrec) opt.selected = true;
      precSelect.appendChild(opt);
    }
    precWrap.appendChild(precSelect);
    body.appendChild(precWrap);
    dialog.appendChild(body);
    const actions = this.#el("div", "lector-modal__actions");
    const cancelBtn = this.#btn("lector-modal__btn");
    cancelBtn.textContent = this.#t("common.cancel");
    cancelBtn.addEventListener("click", () => overlay.remove());
    const clearBtn = this.#btn("lector-modal__btn");
    clearBtn.textContent = this.#t("measurement.clearScale");
    clearBtn.addEventListener("click", () => {
      this.#measurement.setScale({
        source: 1,
        sourceUnit: MeasurementUnit.IN,
        target: 1,
        targetUnit: MeasurementUnit.IN
      });
      this.#showToast(this.#t("toast.calibrationCleared"));
      overlay.remove();
    });
    const saveBtn = this.#btn("lector-modal__btn lector-modal__btn--primary");
    saveBtn.textContent = this.#t("common.save");
    saveBtn.addEventListener("click", () => {
      const sourceVal = parseFloat(sourceField.input.value);
      const targetVal = parseFloat(targetField.input.value);
      if (!Number.isFinite(sourceVal) || sourceVal <= 0 || !Number.isFinite(targetVal) || targetVal <= 0) {
        this.#showToast(this.#t("toast.calibrationInvalid"));
        return;
      }
      const sourceUnit = sourceField.select.value;
      const targetUnit = targetField.select.value;
      const scale = {
        source: sourceVal,
        sourceUnit,
        target: targetVal,
        targetUnit
      };
      this.#measurement.setScale(scale);
      this.#measurement.setActiveUnit(targetUnit);
      this.#measurement.setPrecision(parseInt(precSelect.value, 10));
      this.#showToast(this.#t("toast.calibrationSaved"));
      overlay.remove();
    });
    actions.appendChild(cancelBtn);
    actions.appendChild(clearBtn);
    actions.appendChild(saveBtn);
    dialog.appendChild(actions);
    overlay.appendChild(dialog);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.remove();
    });
    this.#openModal(overlay, dialog);
  }
  // ─── Password Dialog ──────────────────────────────────
  #showPasswordModal(data, name) {
    const overlay = this.#el("div", "lector-modal-overlay");
    const dialog = this.#el("div", "lector-modal");
    const title = this.#el("h3", "lector-modal__title");
    title.textContent = this.#t("password.openTitle");
    dialog.appendChild(title);
    const body = this.#el("div", "lector-modal__body");
    const desc = this.#el("p", "");
    desc.style.margin = "0 0 12px";
    desc.style.color = "var(--lector-fg-secondary)";
    desc.textContent = name ? this.#t("password.descNamed", { name }) : this.#t("password.desc");
    body.appendChild(desc);
    const hiddenUser = document.createElement("input");
    hiddenUser.type = "text";
    hiddenUser.autocomplete = "username";
    hiddenUser.tabIndex = -1;
    hiddenUser.setAttribute("aria-hidden", "true");
    hiddenUser.style.cssText = "position:absolute;width:0;height:0;overflow:hidden;opacity:0";
    body.appendChild(hiddenUser);
    const input = document.createElement("input");
    input.type = "password";
    input.className = "lector-modal__input";
    input.placeholder = this.#t("password.openPlaceholder");
    input.setAttribute("aria-label", this.#t("password.openPlaceholder"));
    input.autocomplete = "current-password";
    body.appendChild(input);
    const errorMsg = this.#el("div", "");
    errorMsg.style.color = "var(--lector-danger, #ef4444)";
    errorMsg.style.fontSize = "var(--lector-font-size-sm)";
    errorMsg.style.marginTop = "8px";
    errorMsg.style.display = "none";
    body.appendChild(errorMsg);
    const form = document.createElement("form");
    form.addEventListener("submit", (e) => e.preventDefault());
    form.autocomplete = "off";
    form.appendChild(body);
    const actions = this.#el("div", "lector-modal__actions");
    const cancelBtn = this.#btn("lector-modal__btn");
    cancelBtn.textContent = this.#t("common.cancel");
    const openBtn = this.#btn("lector-modal__btn lector-modal__btn--primary");
    openBtn.textContent = this.#t("common.open");
    const tryOpen = async () => {
      const pw = input.value;
      if (!pw) {
        input.focus();
        return;
      }
      openBtn.textContent = this.#t("password.opening");
      openBtn.setAttribute("disabled", "");
      errorMsg.style.display = "none";
      try {
        await this.loadDocument(data, pw, name);
        overlay.remove();
      } catch (err) {
        const isPasswordErr = typeof err === "object" && err !== null && "code" in err && err["code"] === 4;
        if (isPasswordErr) {
          errorMsg.textContent = this.#t("password.incorrect");
          errorMsg.style.display = "";
          input.value = "";
          input.focus();
        } else {
          const errStr = err instanceof Error ? err.message : typeof err === "object" && err !== null && "message" in err ? String(err.message) : String(err);
          errorMsg.textContent = this.#t("password.openFailed", { error: errStr });
          errorMsg.style.display = "";
        }
        openBtn.textContent = this.#t("common.open");
        openBtn.removeAttribute("disabled");
      }
    };
    cancelBtn.addEventListener("click", () => overlay.remove());
    openBtn.addEventListener("click", () => void tryOpen());
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") void tryOpen();
    });
    actions.appendChild(cancelBtn);
    actions.appendChild(openBtn);
    form.appendChild(actions);
    dialog.appendChild(form);
    overlay.appendChild(dialog);
    this.#openModal(overlay, dialog, null, input);
  }
  // ─── Password Protect Dialog ──────────────────────────
  #showProtectDialog() {
    const doc = this.#document.activeDocument.peek();
    if (!doc) {
      this.#showToast(this.#t("toast.noDocument"));
      return;
    }
    const overlay = this.#el("div", "lector-modal-overlay");
    const dialog = this.#el("div", "lector-modal");
    const title = this.#el("h3", "lector-modal__title");
    title.textContent = this.#t("protect.dialogTitle");
    dialog.appendChild(title);
    const form = document.createElement("form");
    form.addEventListener("submit", (e) => e.preventDefault());
    form.autocomplete = "off";
    const body = this.#el("div", "lector-modal__body");
    const hiddenUser = document.createElement("input");
    hiddenUser.type = "text";
    hiddenUser.autocomplete = "username";
    hiddenUser.tabIndex = -1;
    hiddenUser.setAttribute("aria-hidden", "true");
    hiddenUser.style.cssText = "position:absolute;width:0;height:0;overflow:hidden;opacity:0";
    body.appendChild(hiddenUser);
    const userLabel = this.#el("label", "lector-modal__label");
    userLabel.textContent = this.#t("protect.userLabel");
    body.appendChild(userLabel);
    const userPw = document.createElement("input");
    userPw.type = "password";
    userPw.autocomplete = "new-password";
    userPw.className = "lector-modal__input";
    userPw.placeholder = this.#t("protect.userPlaceholder");
    body.appendChild(userPw);
    const confirmLabel = this.#el("label", "lector-modal__label");
    confirmLabel.textContent = this.#t("protect.confirmLabel");
    confirmLabel.style.marginTop = "8px";
    body.appendChild(confirmLabel);
    const confirmPw = document.createElement("input");
    confirmPw.type = "password";
    confirmPw.autocomplete = "new-password";
    confirmPw.className = "lector-modal__input";
    confirmPw.placeholder = this.#t("protect.confirmPlaceholder");
    body.appendChild(confirmPw);
    const ownerLabel = this.#el("label", "lector-modal__label");
    ownerLabel.textContent = this.#t("protect.ownerLabel");
    ownerLabel.style.marginTop = "12px";
    body.appendChild(ownerLabel);
    const ownerPw = document.createElement("input");
    ownerPw.type = "password";
    ownerPw.autocomplete = "new-password";
    ownerPw.className = "lector-modal__input";
    ownerPw.placeholder = this.#t("protect.ownerPlaceholder");
    body.appendChild(ownerPw);
    const permLabel = this.#el("div", "lector-modal__label");
    permLabel.textContent = this.#t("protect.permissionsLabel");
    permLabel.style.marginTop = "12px";
    body.appendChild(permLabel);
    const perms = [
      { id: "print", label: this.#t("protect.perm.print"), checked: true },
      { id: "extract", label: this.#t("protect.perm.extract"), checked: true },
      { id: "annotate", label: this.#t("protect.perm.annotate"), checked: true },
      { id: "modify", label: this.#t("protect.perm.modify"), checked: false }
    ];
    const permChecks = {};
    for (const p of perms) {
      const row = this.#el("label", "lector-modal__checkbox-label");
      row.style.display = "flex";
      row.style.alignItems = "center";
      row.style.gap = "6px";
      row.style.fontSize = "var(--lector-font-size-sm)";
      row.style.marginTop = "4px";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = p.checked;
      permChecks[p.id] = cb;
      row.appendChild(cb);
      const span = this.#el("span", "");
      span.textContent = p.label;
      row.appendChild(span);
      body.appendChild(row);
    }
    const errorMsg = this.#el("div", "");
    errorMsg.style.color = "var(--lector-danger, #ef4444)";
    errorMsg.style.fontSize = "var(--lector-font-size-sm)";
    errorMsg.style.marginTop = "8px";
    errorMsg.style.display = "none";
    body.appendChild(errorMsg);
    form.appendChild(body);
    const actions = this.#el("div", "lector-modal__actions");
    const cancelBtn = this.#btn("lector-modal__btn");
    cancelBtn.textContent = this.#t("common.cancel");
    cancelBtn.addEventListener("click", () => overlay.remove());
    const protectBtn = this.#btn("lector-modal__btn lector-modal__btn--primary");
    protectBtn.textContent = this.#t("protect.applyButton");
    protectBtn.addEventListener("click", () => {
      const pw = userPw.value;
      if (!pw) {
        errorMsg.textContent = this.#t("protect.errorRequired");
        errorMsg.style.display = "";
        return;
      }
      if (pw !== confirmPw.value) {
        errorMsg.textContent = this.#t("protect.errorMismatch");
        errorMsg.style.display = "";
        return;
      }
      protectBtn.textContent = this.#t("protect.applying");
      protectBtn.setAttribute("disabled", "");
      void (async () => {
        try {
          await this.#engine.workerProxy.setDocumentPassword(doc.id, {
            userPassword: pw,
            ownerPassword: ownerPw.value || void 0,
            allowPrint: permChecks["print"].checked,
            allowModify: permChecks["modify"].checked,
            allowExtract: permChecks["extract"].checked,
            allowAnnotate: permChecks["annotate"].checked
          });
          const bytes = await this.#engine.workerProxy.saveAsCopy(doc.id);
          const blob = new Blob([bytes], { type: "application/pdf" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          const baseName = (this.#getDocName(doc.id) ?? "document").replace(/\.pdf$/i, "");
          a.download = `${baseName}-protected.pdf`;
          a.click();
          URL.revokeObjectURL(url);
          this.#showToast(this.#t("toast.protectedSaved"));
          overlay.remove();
        } catch (err) {
          const msg = err instanceof Error ? err.message : typeof err === "object" && err !== null && "message" in err ? String(err.message) : String(err);
          errorMsg.textContent = this.#t("protect.errorFailed", { error: msg });
          errorMsg.style.display = "";
          protectBtn.textContent = this.#t("protect.applyButton");
          protectBtn.removeAttribute("disabled");
        }
      })();
    });
    actions.appendChild(cancelBtn);
    actions.appendChild(protectBtn);
    form.appendChild(actions);
    dialog.appendChild(form);
    overlay.appendChild(dialog);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.remove();
    });
    this.#openModal(overlay, dialog, null, userPw);
  }
  // ─── Toast Notifications ──────────────────────────────
  #showToast(message) {
    const toast = this.#el("div", "lector-toast");
    toast.setAttribute("role", "status");
    toast.setAttribute("aria-live", "polite");
    toast.textContent = message;
    const ws = this.#root.querySelector(".lector-workspace") ?? this.#root;
    ws.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("lector-toast--visible"));
    setTimeout(() => {
      toast.classList.remove("lector-toast--visible");
      setTimeout(() => toast.remove(), 300);
    }, 3e3);
  }
  // ─── Marquee Capture ──────────────────────────────────
  async #handleCaptureRegion(rect, sourceDocId, sourceViewportId) {
    if (!this.#capture) return;
    const docId = sourceDocId ?? this.#document.activeDocument.peek()?.id ?? null;
    if (!docId) return;
    let result;
    try {
      result = await this.#capture.captureRegion(docId, rect, { dpi: 300 });
    } catch {
      this.#showToast(this.#t("capture.failed"));
      return;
    }
    this.#showCaptureActionBar(result, sourceViewportId);
    result.bitmap.close();
  }
  /**
   * Resolve a viewport instance and its containing canvas element from
   * a viewport id. Returns the primary viewport when no match is found
   * (e.g. the source pane has been destroyed before the popover lands).
   */
  #resolveViewportContext(viewportId) {
    if (this.#viewportInstance.id === viewportId) {
      return { viewport: this.#viewportInstance, canvas: this.#canvas };
    }
    for (const pane of this.#extraPanes.values()) {
      if (pane.viewport.id === viewportId) {
        return { viewport: pane.viewport, canvas: pane.canvas };
      }
    }
    return { viewport: this.#viewportInstance, canvas: this.#canvas };
  }
  #showCaptureActionBar(result, sourceViewportId) {
    this.#hideCaptureActionBar();
    const ws = this.#root.querySelector(".lector-workspace");
    if (!ws) return;
    const sourceCtx = this.#resolveViewportContext(sourceViewportId);
    const sourceVp = sourceCtx.viewport;
    const sourceCanvas = sourceCtx.canvas;
    const positions = sourceVp.pagePositions.peek();
    const scale = sourceVp.scale.peek();
    const pos = positions.find((p) => p.pageIndex === result.rect.pageIndex);
    const offset = sourceVp.scrollOffset.peek();
    const bar = this.#el("div", "lector-capture-action-bar");
    bar.setAttribute("role", "toolbar");
    bar.setAttribute("aria-label", this.#t("capture.actions") || "Capture actions");
    const preview = window.document.createElement("img");
    preview.className = "lector-capture-action-bar__preview";
    preview.src = URL.createObjectURL(result.blob);
    preview.alt = "";
    bar.appendChild(preview);
    const copyBtn = this.#btn("lector-capture-action-bar__btn");
    copyBtn.appendChild(this.#icon("copy"));
    const copyLbl = this.#el("span", "lector-capture-action-bar__btn-label");
    copyLbl.textContent = this.#t("capture.copy") || "Copy";
    copyBtn.appendChild(copyLbl);
    copyBtn.addEventListener("click", () => {
      void this.#copyCaptureToClipboard(result);
    });
    bar.appendChild(copyBtn);
    const saveBtn = this.#btn("lector-capture-action-bar__btn");
    saveBtn.appendChild(this.#icon("download"));
    const saveLbl = this.#el("span", "lector-capture-action-bar__btn-label");
    saveLbl.textContent = this.#t("capture.save") || "Save";
    saveBtn.appendChild(saveLbl);
    saveBtn.addEventListener("click", () => {
      this.#downloadCapture(result);
    });
    bar.appendChild(saveBtn);
    const closeBtn = this.#btn("lector-capture-action-bar__close");
    closeBtn.appendChild(this.#icon("x"));
    closeBtn.setAttribute("aria-label", this.#t("common.close") || "Close");
    closeBtn.addEventListener("click", () => this.#hideCaptureActionBar());
    bar.appendChild(closeBtn);
    ws.appendChild(bar);
    this.#captureActionBar = bar;
    if (pos) {
      const wsRect = ws.getBoundingClientRect();
      const canvasRect = sourceCanvas.getBoundingClientRect();
      const rectXCanvas = pos.x + result.rect.x * scale - offset.x;
      const rectYBottomCanvas = pos.y + (result.rect.y + result.rect.height) * scale - offset.y;
      const rectWidthCanvas = result.rect.width * scale;
      requestAnimationFrame(() => {
        const barRect = bar.getBoundingClientRect();
        const left = canvasRect.left - wsRect.left + rectXCanvas + rectWidthCanvas / 2 - barRect.width / 2;
        const top = canvasRect.top - wsRect.top + rectYBottomCanvas + 8;
        const maxLeft = canvasRect.left - wsRect.left + canvasRect.width - barRect.width - 8;
        const minLeft = canvasRect.left - wsRect.left + 8;
        const maxTop = canvasRect.top - wsRect.top + canvasRect.height - barRect.height - 8;
        bar.style.left = `${Math.max(minLeft, Math.min(left, maxLeft))}px`;
        bar.style.top = `${Math.min(top, maxTop)}px`;
      });
    }
    const onDocClick = (e) => {
      if (this.#captureActionBar && !this.#captureActionBar.contains(e.target)) {
        this.#hideCaptureActionBar();
      }
    };
    setTimeout(() => window.document.addEventListener("mousedown", onDocClick, { once: false }), 0);
    bar.dataset["cleanupListener"] = "1";
    bar.__cleanup = () => {
      window.document.removeEventListener("mousedown", onDocClick);
      URL.revokeObjectURL(preview.src);
    };
  }
  #hideCaptureActionBar() {
    if (!this.#captureActionBar) return;
    const cleanup = this.#captureActionBar.__cleanup;
    cleanup?.();
    this.#captureActionBar.remove();
    this.#captureActionBar = null;
  }
  async #copyCaptureToClipboard(result) {
    try {
      const item = new ClipboardItem({ "image/png": result.blob });
      await navigator.clipboard.write([item]);
      this.#showToast(this.#t("capture.copiedToClipboard"));
      this.#hideCaptureActionBar();
    } catch {
      this.#showToast(this.#t("capture.copyFailed"));
    }
  }
  #downloadCapture(result) {
    const doc = this.#document.activeDocument.peek();
    const baseName = (doc?.id ?? "document").replace(/\.pdf$/i, "");
    const filename = `${baseName}-capture-p${result.rect.pageIndex + 1}.png`;
    const url = URL.createObjectURL(result.blob);
    const a = window.document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1e3);
    this.#hideCaptureActionBar();
  }
  // ─── Print Dialog ─────────────────────────────────────
  #showPrintDialog() {
    const doc = this.#document.activeDocument.peek();
    if (!doc) return;
    const overlay = this.#el("div", "lector-modal-overlay");
    const dialog = this.#el("div", "lector-modal");
    const title = this.#el("h3", "lector-modal__title");
    title.textContent = this.#t("print.dialogTitle");
    dialog.appendChild(title);
    const body = this.#el("div", "lector-modal__body");
    const rangeLabel = this.#el("label", "lector-modal__label");
    rangeLabel.textContent = this.#t("print.pagesLabel");
    body.appendChild(rangeLabel);
    const rangeGroup = this.#el("div", "lector-modal__radio-group");
    const allRadio = document.createElement("input");
    allRadio.type = "radio";
    allRadio.name = "print-range";
    allRadio.value = "all";
    allRadio.checked = true;
    allRadio.id = "lector-print-all";
    const allLabel = this.#el("label", "lector-modal__radio-label");
    allLabel.htmlFor = "lector-print-all";
    allLabel.textContent = this.#t("print.allPages", { total: doc.pageCount });
    rangeGroup.appendChild(allRadio);
    rangeGroup.appendChild(allLabel);
    const customRadio = document.createElement("input");
    customRadio.type = "radio";
    customRadio.name = "print-range";
    customRadio.value = "custom";
    customRadio.id = "lector-print-custom";
    const customLabel = this.#el("label", "lector-modal__radio-label");
    customLabel.htmlFor = "lector-print-custom";
    customLabel.textContent = this.#t("print.customRange");
    rangeGroup.appendChild(customRadio);
    rangeGroup.appendChild(customLabel);
    const rangeInput = document.createElement("input");
    rangeInput.type = "text";
    rangeInput.className = "lector-modal__input";
    rangeInput.placeholder = this.#t("print.rangePlaceholder");
    rangeInput.disabled = true;
    rangeGroup.appendChild(rangeInput);
    allRadio.addEventListener("change", () => {
      rangeInput.disabled = true;
    });
    customRadio.addEventListener("change", () => {
      rangeInput.disabled = false;
      rangeInput.focus();
    });
    body.appendChild(rangeGroup);
    dialog.appendChild(body);
    const actions = this.#el("div", "lector-modal__actions");
    const cancelBtn = this.#btn("lector-modal__btn");
    cancelBtn.textContent = this.#t("common.cancel");
    cancelBtn.addEventListener("click", () => overlay.remove());
    const printBtn = this.#btn("lector-modal__btn lector-modal__btn--primary");
    printBtn.textContent = this.#t("print.printButton");
    printBtn.addEventListener("click", async () => {
      printBtn.disabled = true;
      printBtn.textContent = this.#t("print.rendering");
      const pages = this.#parsePageRange(
        allRadio.checked ? `1-${doc.pageCount}` : rangeInput.value,
        doc.pageCount
      );
      if (pages.length === 0) {
        this.#showToast(this.#t("toast.invalidPageRange"));
        overlay.remove();
        return;
      }
      try {
        await this.#executePrint(doc, pages);
      } catch (err) {
        this.#showToast(this.#t("toast.printFailed", { error: String(err) }));
      }
      overlay.remove();
    });
    actions.appendChild(cancelBtn);
    actions.appendChild(printBtn);
    dialog.appendChild(actions);
    overlay.appendChild(dialog);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.remove();
    });
    const ws2 = this.#root.querySelector(".lector-workspace") ?? this.#root;
    ws2.appendChild(overlay);
  }
  /** Parse a page range string like "1-3, 5, 8-10" into zero-based page indices. */
  #parsePageRange(input, pageCount) {
    const pages = /* @__PURE__ */ new Set();
    for (const part of input.split(",")) {
      const trimmed = part.trim();
      if (!trimmed) continue;
      const match = trimmed.match(/^(\d+)\s*-\s*(\d+)$/);
      if (match) {
        const start = Math.max(1, parseInt(match[1], 10));
        const end = Math.min(pageCount, parseInt(match[2], 10));
        for (let i = start; i <= end; i++) pages.add(i - 1);
      } else {
        const pg = parseInt(trimmed, 10);
        if (!isNaN(pg) && pg >= 1 && pg <= pageCount) pages.add(pg - 1);
      }
    }
    return [...pages].sort((a, b) => a - b);
  }
  /** Render pages at high DPI and trigger browser print. */
  async #executePrint(doc, pages) {
    const dpi = 300;
    const ptsToPixels = dpi / 72;
    const iframe = document.createElement("iframe");
    iframe.style.cssText = "position:fixed;top:-10000px;left:-10000px;width:0;height:0;border:none";
    document.body.appendChild(iframe);
    const iframeDoc = iframe.contentDocument ?? iframe.contentWindow?.document;
    if (!iframeDoc) {
      iframe.remove();
      return;
    }
    iframeDoc.open();
    iframeDoc.write(`<!DOCTYPE html><html><head><style>
      @media print {
        * { margin: 0; padding: 0; }
        @page { margin: 0; }
        body { margin: 0; }
        img { display: block; width: 100%; height: auto; page-break-after: always; }
        img:last-child { page-break-after: avoid; }
      }
      body { margin: 0; }
    </style></head><body></body></html>`);
    iframeDoc.close();
    for (const pageIdx of pages) {
      const ps = doc.pageSizes[pageIdx];
      if (!ps) continue;
      const w = Math.round(ps.width * ptsToPixels);
      const h = Math.round(ps.height * ptsToPixels);
      const bmp = await this.#engine.renderPage(doc.id, pageIdx, w, h, {
        priority: RenderPriority.VISIBLE,
        flags: 2048
        // PRINTING flag
      });
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.drawImage(bmp, 0, 0, w, h);
      bmp.close();
      const img = iframeDoc.createElement("img");
      img.src = canvas.toDataURL("image/png");
      iframeDoc.body.appendChild(img);
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
    iframe.contentWindow?.print();
    setTimeout(() => iframe.remove(), 1e3);
  }
  // ─── Export/Save ──────────────────────────────────────
  /** Save the current document as PDF (download). */
  async #saveDocument() {
    const doc = this.#document.activeDocument.peek();
    if (!doc) return;
    try {
      const bytes = await this.#engine.workerProxy.saveAsCopy(doc.id);
      const blob = new Blob([bytes], { type: "application/pdf" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = this.#getDocName(doc.id) ?? "document.pdf";
      a.click();
      URL.revokeObjectURL(url);
      this.#showToast(this.#t("toast.documentSaved"));
    } catch (err) {
      this.#showToast(this.#t("toast.saveFailed", { error: String(err) }));
    }
  }
  /** Export the current visible page as a PNG image. */
  async #exportPageAsImage() {
    const doc = this.#document.activeDocument.peek();
    if (!doc) return;
    const visiblePages = this.#viewportInstance.visiblePages.peek();
    const pageIdx = visiblePages[0] ?? 0;
    const ps = doc.pageSizes[pageIdx];
    if (!ps) return;
    const dpi = 300;
    const scale = dpi / 72;
    const w = Math.round(ps.width * scale);
    const h = Math.round(ps.height * scale);
    try {
      const bmp = await this.#engine.renderPage(doc.id, pageIdx, w, h, {
        priority: RenderPriority.VISIBLE
      });
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.drawImage(bmp, 0, 0, w, h);
      bmp.close();
      canvas.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `page-${pageIdx + 1}.png`;
        a.click();
        URL.revokeObjectURL(url);
        this.#showToast(this.#t("toast.pageExported", { page: pageIdx + 1 }));
      }, "image/png");
    } catch (err) {
      this.#showToast(this.#t("toast.exportFailed", { error: String(err) }));
    }
  }
  // ─── Thumbnail Context Menu ───────────────────────────
  #showThumbnailContextMenu(e, pageIndex) {
    e.preventDefault();
    this.#root.querySelector(".lector-context-menu")?.remove();
    const handle = this.#document.activeDocument.peek();
    if (!handle) return;
    const pageOps = this.#pageOps;
    const menu = this.#el("div", "lector-context-menu");
    menu.style.left = `${e.clientX}px`;
    menu.style.top = `${e.clientY}px`;
    const items = [];
    if (pageOps) {
      items.push(
        { label: this.#t("contextMenu.insertPageBefore"), icon: "file-plus", action: () => {
          const ps = handle.pageSizes[pageIndex];
          void pageOps.insertBlankPage(handle.id, pageIndex, ps?.width ?? 612, ps?.height ?? 792);
        } },
        { label: this.#t("contextMenu.insertPageAfter"), icon: "file-plus", action: () => {
          const ps = handle.pageSizes[pageIndex];
          void pageOps.insertBlankPage(handle.id, pageIndex + 1, ps?.width ?? 612, ps?.height ?? 792);
        } },
        { label: this.#t("contextMenu.duplicatePage"), icon: "copy-plus", action: () => {
          void pageOps.duplicatePage(handle.id, pageIndex);
        } }
      );
      items.push(
        { label: this.#t("contextMenu.rotateCW"), icon: "rotate-cw", action: () => {
          void pageOps.rotatePage(handle.id, pageIndex, 90);
        } },
        { label: this.#t("contextMenu.rotateCCW"), icon: "rotate-ccw", action: () => {
          void pageOps.rotatePage(handle.id, pageIndex, 270);
        } }
      );
      if (handle.pageCount > 1) {
        items.push(
          { label: this.#t("contextMenu.deletePage"), icon: "trash", danger: true, action: () => {
            void pageOps.deletePage(handle.id, pageIndex);
          } }
        );
      }
    }
    if (items.length === 0) return;
    if (items.length === 0) return;
    for (const item of items) {
      const btn = this.#btn("lector-context-menu__item" + (item.danger ? " lector-context-menu__item--danger" : ""));
      btn.setAttribute("role", "menuitem");
      btn.tabIndex = -1;
      const ic = this.#el("span", "lector-context-menu__icon");
      ic.innerHTML = resolveIcon(item.icon) ?? "";
      btn.appendChild(ic);
      const lbl = this.#el("span", "lector-context-menu__label");
      lbl.textContent = item.label;
      btn.appendChild(lbl);
      btn.addEventListener("click", () => {
        menu.remove();
        item.action();
        setTimeout(() => this.#updateSidebarActiveTab(), 300);
      });
      menu.appendChild(btn);
    }
    menu.setAttribute("role", "menu");
    this.#wireContextMenuKeyboard(menu);
    const close = (ev) => {
      if (!menu.contains(ev.target)) {
        menu.remove();
        document.removeEventListener("click", close);
      }
    };
    setTimeout(() => document.addEventListener("click", close), 0);
    const ws = this.#root.querySelector(".lector-workspace") ?? this.#root;
    ws.appendChild(menu);
    const firstItem = menu.querySelector('[role="menuitem"]');
    if (firstItem) requestAnimationFrame(() => firstItem.focus());
  }
  // ─── Signature Dialog ──────────────────────────────────
  #showSignatureDialog(detail) {
    const overlay = this.#el("div", "lector-modal-overlay");
    const dialog = this.#el("div", "lector-modal");
    const title = this.#el("h3", "lector-modal__title");
    title.textContent = this.#t("sign.dialogTitle");
    dialog.appendChild(title);
    const tabs = this.#el("div", "lector-sig-dialog__tabs");
    let activeTab = "draw";
    const body = this.#el("div", "lector-modal__body");
    const tabDefs = [
      { id: "draw", label: this.#t("sign.tab.draw") },
      { id: "type", label: this.#t("sign.tab.type") },
      { id: "image", label: this.#t("sign.tab.image") }
    ];
    for (const tab of tabDefs) {
      const btn = this.#btn("lector-sig-dialog__tab");
      btn.textContent = tab.label;
      if (tab.id === activeTab) btn.classList.add("lector-sig-dialog__tab--active");
      btn.addEventListener("click", () => {
        activeTab = tab.id;
        for (const t of tabs.querySelectorAll(".lector-sig-dialog__tab")) {
          t.classList.toggle("lector-sig-dialog__tab--active", t === btn);
        }
        renderTabContent();
      });
      tabs.appendChild(btn);
    }
    dialog.appendChild(tabs);
    dialog.appendChild(body);
    let signatureDataUrl = "";
    const renderTabContent = () => {
      body.innerHTML = "";
      if (activeTab === "draw") {
        const canvas = document.createElement("canvas");
        canvas.className = "lector-sig-dialog__canvas";
        canvas.width = 400;
        canvas.height = 150;
        body.appendChild(canvas);
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.strokeStyle = "#1a1a1a";
          ctx.lineWidth = 2;
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          let drawing = false;
          const getPos = (e) => {
            const r = canvas.getBoundingClientRect();
            return { x: (e.clientX - r.left) * (canvas.width / r.width), y: (e.clientY - r.top) * (canvas.height / r.height) };
          };
          canvas.addEventListener("pointerdown", (e) => {
            drawing = true;
            canvas.setPointerCapture(e.pointerId);
            const p = getPos(e);
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
          });
          canvas.addEventListener("pointermove", (e) => {
            if (!drawing) return;
            const p = getPos(e);
            ctx.lineTo(p.x, p.y);
            ctx.stroke();
          });
          canvas.addEventListener("pointerup", () => {
            drawing = false;
            signatureDataUrl = canvas.toDataURL("image/png");
          });
        }
        const clearBtn = this.#btn("lector-modal__btn");
        clearBtn.textContent = this.#t("sign.clearButton");
        clearBtn.style.marginTop = "8px";
        clearBtn.addEventListener("click", () => {
          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
          }
          signatureDataUrl = "";
        });
        body.appendChild(clearBtn);
      } else if (activeTab === "type") {
        const input = document.createElement("input");
        input.type = "text";
        input.className = "lector-sig-dialog__type-input";
        input.placeholder = this.#t("sign.typedPlaceholder");
        input.value = this.#engine.user?.name ?? "";
        input.addEventListener("input", () => {
          const tmpCanvas = document.createElement("canvas");
          tmpCanvas.width = 400;
          tmpCanvas.height = 150;
          const tmpCtx = tmpCanvas.getContext("2d");
          if (tmpCtx) {
            tmpCtx.fillStyle = "#1a1a1a";
            tmpCtx.font = '36px "Brush Script MT", "Segoe Script", cursive';
            tmpCtx.textAlign = "center";
            tmpCtx.textBaseline = "middle";
            tmpCtx.fillText(input.value, 200, 75);
            signatureDataUrl = tmpCanvas.toDataURL("image/png");
          }
        });
        body.appendChild(input);
        input.dispatchEvent(new Event("input"));
      } else {
        const preview = this.#el("div", "lector-sig-dialog__preview");
        preview.textContent = this.#t("sign.uploadPrompt");
        body.appendChild(preview);
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "image/*";
        fileInput.style.display = "none";
        fileInput.addEventListener("change", () => {
          const file = fileInput.files?.[0];
          if (!file) return;
          const reader = new FileReader();
          reader.onload = () => {
            signatureDataUrl = reader.result;
            preview.innerHTML = "";
            const img = document.createElement("img");
            img.src = signatureDataUrl;
            preview.appendChild(img);
          };
          reader.readAsDataURL(file);
        });
        body.appendChild(fileInput);
        preview.addEventListener("click", () => fileInput.click());
      }
    };
    renderTabContent();
    const actions = this.#el("div", "lector-modal__actions");
    const cancelBtn = this.#btn("lector-modal__btn");
    cancelBtn.textContent = this.#t("common.cancel");
    cancelBtn.addEventListener("click", () => overlay.remove());
    const certForm = document.createElement("form");
    certForm.autocomplete = "off";
    certForm.addEventListener("submit", (e) => e.preventDefault());
    const certHiddenUser = document.createElement("input");
    certHiddenUser.type = "text";
    certHiddenUser.autocomplete = "username";
    certHiddenUser.tabIndex = -1;
    certHiddenUser.setAttribute("aria-hidden", "true");
    certHiddenUser.style.cssText = "position:absolute;width:0;height:0;overflow:hidden;opacity:0";
    certForm.appendChild(certHiddenUser);
    const certSection = this.#el("div", "lector-sig-dialog__cert-section");
    const certLabel = this.#el("div", "lector-sig-dialog__cert-label");
    certLabel.textContent = this.#t("sign.certLabel");
    certSection.appendChild(certLabel);
    const certRow = this.#el("div", "lector-sig-dialog__cert-row");
    const certFileBtn = this.#btn("lector-modal__btn");
    certFileBtn.textContent = this.#t("sign.certUpload");
    const certFileName = this.#el("span", "lector-sig-dialog__cert-name");
    certFileName.textContent = this.#t("sign.certNone");
    let certData = null;
    const certFileInput = document.createElement("input");
    certFileInput.type = "file";
    certFileInput.accept = ".pfx,.p12";
    certFileInput.style.display = "none";
    certFileInput.addEventListener("change", () => {
      const file = certFileInput.files?.[0];
      if (!file) return;
      certFileName.textContent = file.name;
      void file.arrayBuffer().then((d) => {
        certData = d;
      });
    });
    certFileBtn.addEventListener("click", () => certFileInput.click());
    certRow.appendChild(certFileBtn);
    certRow.appendChild(certFileName);
    certSection.appendChild(certRow);
    certSection.appendChild(certFileInput);
    const certPwInput = document.createElement("input");
    certPwInput.type = "password";
    certPwInput.autocomplete = "off";
    certPwInput.className = "lector-modal__input";
    certPwInput.placeholder = this.#t("sign.certPasswordPlaceholder");
    certPwInput.style.marginTop = "8px";
    certSection.appendChild(certPwInput);
    const reasonInput = document.createElement("input");
    reasonInput.type = "text";
    reasonInput.className = "lector-modal__input";
    reasonInput.placeholder = this.#t("sign.reasonPlaceholder");
    reasonInput.style.marginTop = "8px";
    certSection.appendChild(reasonInput);
    const mdpSelect = document.createElement("select");
    mdpSelect.className = "lector-modal__input";
    mdpSelect.style.marginTop = "8px";
    mdpSelect.style.marginBottom = "12px";
    const mdpOptions = [
      { value: "0", label: this.#t("sign.mdp.approval") },
      { value: "3", label: this.#t("sign.mdp.formCommenting") },
      { value: "2", label: this.#t("sign.mdp.formSigning") },
      { value: "1", label: this.#t("sign.mdp.noChanges") }
    ];
    for (const opt of mdpOptions) {
      const o = document.createElement("option");
      o.value = opt.value;
      o.textContent = opt.label;
      mdpSelect.appendChild(o);
    }
    certSection.appendChild(mdpSelect);
    const tsaInput = document.createElement("input");
    tsaInput.type = "url";
    tsaInput.className = "lector-modal__input";
    tsaInput.placeholder = this.#t("sign.tsaPlaceholder");
    tsaInput.style.marginTop = "8px";
    tsaInput.style.marginBottom = "12px";
    certSection.appendChild(tsaInput);
    certForm.appendChild(certSection);
    dialog.appendChild(certForm);
    const signBtn = this.#btn("lector-modal__btn lector-modal__btn--primary");
    signBtn.textContent = this.#t("sign.applyButton");
    const errorMsg = this.#el("div", "lector-modal__error");
    errorMsg.style.cssText = "color:var(--lector-danger,#dc3545);margin-top:8px;display:none;font-size:13px";
    certSection.appendChild(errorMsg);
    signBtn.addEventListener("click", () => {
      if (!signatureDataUrl && !certData) {
        this.#showToast(this.#t("toast.signatureMissing"));
        return;
      }
      if (!certData) {
        this.#engine.plugins.events.emit("signature:applied", {
          ...detail,
          signatureImage: signatureDataUrl,
          signerName: this.#engine.user?.name ?? "Unknown",
          timestamp: (/* @__PURE__ */ new Date()).toISOString(),
          reason: reasonInput.value || void 0
        });
        this.#showToast(this.#t("toast.signatureApplied"));
        overlay.remove();
        return;
      }
      const doc = this.#document.activeDocument.peek();
      if (!doc) {
        this.#showToast(this.#t("toast.noDocument"));
        return;
      }
      signBtn.textContent = this.#t("sign.applying");
      signBtn.setAttribute("disabled", "");
      errorMsg.style.display = "none";
      const rectObj = detail.rect;
      void (async () => {
        try {
          let appearanceJpeg;
          let appearanceWidth;
          let appearanceHeight;
          if (signatureDataUrl) {
            const img = new Image();
            img.src = signatureDataUrl;
            await img.decode();
            const canvas = document.createElement("canvas");
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            const blob2 = await new Promise((resolve, reject) => {
              canvas.toBlob(
                (b) => b ? resolve(b) : reject(new Error("toBlob failed")),
                "image/jpeg",
                0.92
              );
            });
            appearanceJpeg = await blob2.arrayBuffer();
            appearanceWidth = img.naturalWidth;
            appearanceHeight = img.naturalHeight;
          }
          const result = await this.#engine.workerProxy.signDocument(doc.id, {
            pfxData: certData,
            pfxPassword: certPwInput.value,
            pageIndex: detail.pageIndex,
            rectLeft: rectObj?.left ?? 0,
            rectBottom: rectObj?.bottom ?? 0,
            rectRight: rectObj?.right ?? 200,
            rectTop: rectObj?.top ?? 50,
            reason: reasonInput.value || void 0,
            signerName: this.#engine.user?.name,
            mdpLevel: parseInt(mdpSelect.value, 10),
            tsaUrl: tsaInput.value.trim() || void 0,
            appearanceJpeg,
            appearanceWidth,
            appearanceHeight
          });
          const blob = new Blob([result.signedPdf], { type: "application/pdf" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          const baseName = (this.#getDocName(doc.id) ?? "document").replace(/\.pdf$/i, "");
          a.download = `${baseName}-signed.pdf`;
          a.click();
          URL.revokeObjectURL(url);
          this.#showToast(this.#t("toast.signedSaved"));
          overlay.remove();
        } catch (err) {
          const msg = err instanceof Error ? err.message : typeof err === "object" && err !== null && "message" in err ? String(err.message) : String(err);
          errorMsg.textContent = msg;
          errorMsg.style.display = "";
          signBtn.textContent = this.#t("sign.applyButton");
          signBtn.removeAttribute("disabled");
        }
      })();
    });
    actions.appendChild(cancelBtn);
    actions.appendChild(signBtn);
    dialog.appendChild(actions);
    overlay.appendChild(dialog);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.remove();
    });
    this.#openModal(overlay, dialog);
  }
  // ─── Page Context Menu ────────────────────────────────
  #showPageContextMenu(e, pageIndex) {
    e.preventDefault();
    this.#root.querySelector(".lector-context-menu")?.remove();
    const handle = this.#document.activeDocument.peek();
    if (!handle) return;
    const menu = this.#el("div", "lector-context-menu");
    menu.style.left = `${e.clientX}px`;
    menu.style.top = `${e.clientY}px`;
    const items = [];
    const picked = this.#textLayer?.selection.peek()?.text?.trim();
    if (picked) {
      items.push({ label: this.#t("contextMenu.copyText"), icon: "copy", action: () => {
        void copyText(picked);
        this.#showToast(this.#t("capture.copiedToClipboard"));
      } });
    }
    if (this.#annotation) {
      const pageEl = this.#pageElements.get(pageIndex);
      const pageRect = pageEl?.getBoundingClientRect();
      const ps = handle.pageSizes[pageIndex];
      const scale = this.#viewportInstance.scale.peek();
      if (pageRect && ps) {
        const cssX = e.clientX - pageRect.left;
        const cssY = e.clientY - pageRect.top;
        const clickVp = PageViewport.fromRotatedSize(
          ps.width,
          ps.height,
          this.#engine.pageRotation.get(handle.id, pageIndex),
          scale
        );
        const { x: pdfX, y: pdfY } = clickVp.cssPointToPdf(cssX, cssY);
        items.push(
          { label: this.#t("contextMenu.addNote"), icon: "message-square", action: () => {
            const noteSize = 24;
            void this.#annotation.create(handle.id, pageIndex, {
              subtype: 1,
              // TEXT (sticky note)
              pageIndex,
              rect: { left: pdfX, bottom: pdfY, right: pdfX + noteSize, top: pdfY - noteSize },
              color: { r: 255, g: 205, b: 69, a: 255 },
              contents: ""
            }).then((t) => this.#annotation.selectAnnotation(t.id));
          } },
          { label: this.#t("contextMenu.highlight"), icon: "highlighter", action: () => {
            this.#annotation.setActiveTool("highlight");
          } }
        );
      }
    }
    if (this.#pageOps) {
      items.push(
        { label: this.#t("contextMenu.rotateCW"), icon: "rotate-cw", action: () => {
          void this.#pageOps.rotatePage(handle.id, pageIndex, 90);
        } },
        { label: this.#t("contextMenu.rotateCCW"), icon: "rotate-ccw", action: () => {
          void this.#pageOps.rotatePage(handle.id, pageIndex, 270);
        } }
      );
    }
    if (items.length === 0) return;
    for (const item of items) {
      const btn = this.#btn("lector-context-menu__item" + (item.danger ? " lector-context-menu__item--danger" : ""));
      btn.setAttribute("role", "menuitem");
      btn.tabIndex = -1;
      const ic = this.#el("span", "lector-context-menu__icon");
      ic.innerHTML = resolveIcon(item.icon) ?? "";
      btn.appendChild(ic);
      const lbl = this.#el("span", "lector-context-menu__label");
      lbl.textContent = item.label;
      btn.appendChild(lbl);
      btn.addEventListener("click", () => {
        menu.remove();
        item.action();
      });
      menu.appendChild(btn);
    }
    menu.setAttribute("role", "menu");
    this.#wireContextMenuKeyboard(menu);
    const close = (ev) => {
      if (!menu.contains(ev.target)) {
        menu.remove();
        document.removeEventListener("click", close);
      }
    };
    setTimeout(() => document.addEventListener("click", close), 0);
    const ws = this.#root.querySelector(".lector-workspace") ?? this.#root;
    ws.appendChild(menu);
    const firstItem = menu.querySelector('[role="menuitem"]');
    if (firstItem) requestAnimationFrame(() => firstItem.focus());
  }
  // ─── Search Bar ─────────────────────────────────────────
  #buildSearchBar() {
    if (!this.#search) return;
    const canvasWrap = this.#canvasWrap;
    const bar = this.#el("div", "lector-search-bar lector-search-bar--hidden");
    this.#searchBarEl = bar;
    const input = document.createElement("input");
    input.type = "text";
    input.className = "lector-search-bar__input";
    input.placeholder = this.#t("search.placeholder");
    input.setAttribute("aria-label", this.#t("toolbar.search"));
    input.spellcheck = false;
    input.autocomplete = "off";
    const count = this.#el("span", "lector-search-bar__count");
    count.setAttribute("aria-live", "polite");
    count.setAttribute("role", "status");
    count.textContent = "";
    const caseBtn = this.#btn("lector-search-bar__toggle");
    caseBtn.textContent = "Aa";
    caseBtn.title = this.#t("search.matchCase");
    caseBtn.setAttribute("aria-label", this.#t("search.matchCase"));
    let matchCase = false;
    caseBtn.setAttribute("aria-pressed", "false");
    caseBtn.addEventListener("click", () => {
      matchCase = !matchCase;
      caseBtn.classList.toggle("lector-search-bar__toggle--active", matchCase);
      caseBtn.setAttribute("aria-pressed", String(matchCase));
      doSearch();
    });
    const wordBtn = this.#btn("lector-search-bar__toggle");
    wordBtn.textContent = "Ab";
    wordBtn.title = this.#t("search.matchWholeWord");
    wordBtn.setAttribute("aria-label", this.#t("search.matchWholeWord"));
    let matchWholeWord = false;
    wordBtn.setAttribute("aria-pressed", "false");
    wordBtn.addEventListener("click", () => {
      matchWholeWord = !matchWholeWord;
      wordBtn.classList.toggle("lector-search-bar__toggle--active", matchWholeWord);
      wordBtn.setAttribute("aria-pressed", String(matchWholeWord));
      doSearch();
    });
    const divider1 = this.#el("div", "lector-search-bar__divider");
    const prevBtn = this.#btn("lector-search-bar__btn");
    prevBtn.innerHTML = resolveIcon("chevron-up") ?? "&#x25B2;";
    prevBtn.title = this.#t("search.previous");
    prevBtn.disabled = true;
    prevBtn.addEventListener("click", () => this.#search.previousMatch());
    const nextBtn = this.#btn("lector-search-bar__btn");
    nextBtn.innerHTML = resolveIcon("chevron-down") ?? "&#x25BC;";
    nextBtn.title = this.#t("search.next");
    nextBtn.disabled = true;
    nextBtn.addEventListener("click", () => this.#search.nextMatch());
    const divider2 = this.#el("div", "lector-search-bar__divider");
    const closeBtn = this.#btn("lector-search-bar__btn");
    closeBtn.innerHTML = resolveIcon("x") ?? "&#x2715;";
    closeBtn.title = this.#t("search.close");
    closeBtn.addEventListener("click", () => this.#hideSearchBar());
    bar.appendChild(input);
    bar.appendChild(caseBtn);
    bar.appendChild(wordBtn);
    bar.appendChild(count);
    bar.appendChild(divider1);
    bar.appendChild(prevBtn);
    bar.appendChild(nextBtn);
    bar.appendChild(divider2);
    bar.appendChild(closeBtn);
    canvasWrap.appendChild(bar);
    let searchTimer = null;
    const search = this.#search;
    const doSearch = () => {
      if (searchTimer) clearTimeout(searchTimer);
      const query = input.value;
      const doc = this.#document.activeDocument.peek();
      if (!doc || !query) {
        search.clear();
        count.textContent = "";
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        return;
      }
      searchTimer = setTimeout(() => {
        void search.search(doc.id, query, { matchCase, matchWholeWord });
      }, 200);
    };
    input.addEventListener("input", doSearch);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (e.shiftKey) {
          search.previousMatch();
        } else {
          search.nextMatch();
        }
      }
      if (e.key === "Escape") {
        e.preventDefault();
        this.#hideSearchBar();
      }
    });
    this.#pushCleanup(effect2(() => {
      const res = search.result.value;
      const idx = search.activeMatchIndex.value;
      if (!res || res.totalCount === 0) {
        count.textContent = input.value ? this.#t("search.noResults") : "";
        prevBtn.disabled = true;
        nextBtn.disabled = true;
      } else {
        count.textContent = `${idx + 1} / ${res.totalCount}`;
        prevBtn.disabled = false;
        nextBtn.disabled = false;
      }
    }));
    this.#pushCleanup(effect2(() => {
      const res = search.result.value;
      const idx = search.activeMatchIndex.value;
      if (res && idx >= 0 && idx < res.matches.length) {
        const match = res.matches[idx];
        this.#viewportInstance.scrollToPage(match.pageIndex, false);
      }
    }));
  }
  #showSearchBar() {
    if (!this.#searchBarEl) return;
    this.#searchBarEl.classList.remove("lector-search-bar--hidden");
    const input = this.#searchBarEl.querySelector(".lector-search-bar__input");
    if (input) {
      input.focus();
      input.select();
    }
  }
  #hideSearchBar() {
    if (!this.#searchBarEl) return;
    this.#searchBarEl.classList.add("lector-search-bar--hidden");
    this.#search?.clear();
    const input = this.#searchBarEl.querySelector(".lector-search-bar__input");
    if (input) input.value = "";
    this.#canvas.focus();
  }
  // ─── Text Selection Toolbar ───────────────────────────
  #textSelToolbar = null;
  #showTextSelectionToolbar() {
    const sel = this.#textLayer?.selection.peek();
    const text = sel?.text?.trim();
    if (!sel || !text) {
      this.#hideTextSelectionToolbar();
      return;
    }
    const canvasWrap = this.#canvasWrap;
    const marks = canvasWrap.querySelectorAll(".lector-text-highlight");
    const anchor = marks[marks.length - 1];
    if (!anchor) {
      this.#hideTextSelectionToolbar();
      return;
    }
    const rect = anchor.getBoundingClientRect();
    const wrapRect = canvasWrap.getBoundingClientRect();
    if (!this.#textSelToolbar) {
      this.#textSelToolbar = this.#el("div", "lector-text-sel-toolbar");
      canvasWrap.appendChild(this.#textSelToolbar);
    }
    const tb = this.#textSelToolbar;
    tb.innerHTML = "";
    const tools = [
      { icon: "copy", tooltip: "Copy", action: () => {
        void copyText(text);
        this.#showToast(this.#t("toast.copied"));
        this.#hideTextSelectionToolbar();
      } }
    ];
    if (this.#annotation) {
      tools.push(
        { icon: "highlighter", tooltip: this.#t("annotation.highlight"), action: () => {
          this.#annotation.setActiveTool("highlight");
        } },
        { icon: "underline-text", tooltip: this.#t("annotation.underline"), action: () => {
          this.#annotation.setActiveTool("underline");
        } },
        { icon: "strikethrough", tooltip: this.#t("annotation.strikeout"), action: () => {
          this.#annotation.setActiveTool("strikeout");
        } }
      );
    }
    tools.push({ icon: "search", tooltip: "Search", action: () => {
      try {
        void this.#engine.plugins.commands.execute("search.open");
      } catch {
      }
    } });
    for (const t of tools) {
      const btn = this.#btn("lector-text-sel-toolbar__btn");
      btn.innerHTML = resolveIcon(t.icon) ?? "";
      btn.title = t.tooltip;
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        t.action();
      });
      tb.appendChild(btn);
    }
    tb.style.left = `${rect.left + rect.width / 2 - wrapRect.left}px`;
    tb.style.top = `${rect.top - wrapRect.top - 40}px`;
    tb.style.display = "";
  }
  #hideTextSelectionToolbar() {
    if (this.#textSelToolbar) {
      this.#textSelToolbar.style.display = "none";
    }
  }
  // ─── Reactive Effects ──────────────────────────────────
  #wireEffects() {
    this.#buildToolbar();
    this.#buildSidebar();
    this.#buildPageControls();
    this.#buildSearchBar();
    const ws = this.#root.querySelector(".lector-workspace");
    if (ws) {
      let prevTier = "";
      let lastCollapseWasAuto = false;
      this.#pushCleanup(effect2(() => {
        const tier = this.#ui.state.breakpoint.value;
        ws.classList.toggle("lector-workspace--compact", tier === "compact");
        ws.classList.toggle("lector-workspace--medium", tier === "medium");
        ws.classList.toggle("lector-workspace--wide", tier === "wide");
        if (tier === "compact" && prevTier !== "compact") {
          if (!this.#ui.state.sidebar.collapsed.peek()) {
            this.#ui.setSidebarCollapsed(true);
            lastCollapseWasAuto = true;
          }
        }
        if (tier !== "compact" && prevTier === "compact" && lastCollapseWasAuto) {
          this.#ui.setSidebarCollapsed(false);
          lastCollapseWasAuto = false;
        }
        prevTier = tier;
      }));
      this.#pushCleanup(effect2(() => {
        const theme = this.#ui.effectiveTheme.value;
        ws.classList.toggle("lector-workspace--dark", theme === "dark");
        ws.classList.toggle("lector-workspace--light", theme === "light");
      }));
    }
    this.#pushCleanup(effect2(() => {
      void this.#ui.state.toolbar.items.value;
      this.#buildToolbar();
    }));
    this.#pushCleanup(effect2(() => {
      void this.#document.activeDocument.value;
      this.#updateSignatureStatusBadge();
    }));
    this.#pushCleanup(effect2(() => {
      const collapsed = this.#ui.state.sidebar.collapsed.value;
      this.#sidebarEl.classList.toggle("lector-sidebar--collapsed", collapsed);
      const sidebarBtn = this.#toolbar.querySelector('[data-action="ui.toggle-sidebar"]');
      if (sidebarBtn) sidebarBtn.setAttribute("aria-expanded", String(!collapsed));
      const backdrop = this.#root.querySelector(".lector-sidebar-backdrop");
      if (backdrop) {
        const tier = this.#ui.state.breakpoint.peek();
        backdrop.classList.toggle("lector-sidebar-backdrop--visible", !collapsed && tier === "compact");
      }
    }));
    this.#pushCleanup(effect2(() => {
      void this.#ui.state.sidebar.activePanel.value;
      void this.#document.activeDocument.value;
      this.#updateSidebarActiveTab();
    }));
    if (this.#i18n) {
      let prevLocale = this.#i18n.locale.peek();
      this.#pushCleanup(effect2(() => {
        const locale = this.#i18n.locale.value;
        const ws2 = this.#root.querySelector(".lector-workspace");
        if (ws2) ws2.setAttribute("lang", locale);
        if (locale !== prevLocale) {
          prevLocale = locale;
          this.#buildToolbar();
          this.#buildAnnotToolbar();
          this.#buildSidebar();
          this.#updateSidebarActiveTab();
          this.#buildPageControls();
        }
      }));
    }
    {
      const HIDE_DELAY = 4e3;
      let hideTimer = null;
      const pc = this.#pageControlsEl;
      const show = () => {
        pc.classList.remove("lector-page-controls--hidden");
        if (hideTimer) clearTimeout(hideTimer);
        hideTimer = setTimeout(hide, HIDE_DELAY);
      };
      const hide = () => {
        if (pc.contains(document.activeElement)) return;
        pc.classList.add("lector-page-controls--hidden");
      };
      this.#canvasWrap.addEventListener("mousemove", show);
      this.#canvasWrap.addEventListener("scroll", show, true);
      this.#canvas.addEventListener("scroll", show);
      pc.addEventListener("mouseenter", () => {
        if (hideTimer) clearTimeout(hideTimer);
      });
      pc.addEventListener("mouseleave", () => {
        hideTimer = setTimeout(hide, HIDE_DELAY);
      });
      pc.addEventListener("focusin", () => {
        if (hideTimer) clearTimeout(hideTimer);
      });
      pc.addEventListener("focusout", () => {
        hideTimer = setTimeout(hide, HIDE_DELAY);
      });
      hideTimer = setTimeout(hide, HIDE_DELAY);
      this.#pushCleanup(() => {
        if (hideTimer) clearTimeout(hideTimer);
      });
    }
    const interactionCap = this.#engine.plugins.tryGet("interaction");
    if (interactionCap) {
      this.#pushCleanup(effect2(() => {
        const mode = interactionCap.mode.value;
        if (this.#annotation) {
          void this.#annotation.activeTool.value;
        }
        void this.#viewport.viewports.value;
        const cursor = interactionCap.cursor.value;
        const allCanvases = [this.#canvas];
        for (const pane of this.#extraPanes.values()) {
          allCanvases.push(pane.canvas);
        }
        for (const c of allCanvases) {
          c.style.cursor = cursor || "auto";
          c.classList.toggle("lector-canvas--drawing", mode === "draw");
        }
        for (const btn of this.#toolbar.querySelectorAll("[data-tool]")) {
          const el = btn;
          el.classList.toggle("lector-btn--active", el.dataset["tool"] === mode);
        }
      }));
    }
    if (this.#annotation) {
      this.#buildAnnotToolbar();
      this.#pushCleanup(effect2(() => {
        const tool = this.#annotation.activeTool.value;
        for (const btn of this.#annotToolbar.querySelectorAll("[data-annot-tool]")) {
          const el = btn;
          el.classList.toggle("lector-btn--active", el.dataset["annotTool"] === tool);
        }
        for (const trigger of this.#annotToolbar.querySelectorAll("[data-annot-tool-group]")) {
          const el = trigger;
          const groupTool = el.dataset["annotToolGroup"];
          const panel = el.parentElement?.querySelector(".lector-tool-group__panel");
          let groupActive = groupTool === tool;
          if (!groupActive && panel) {
            for (const child of panel.querySelectorAll("[data-annot-tool]")) {
              if (child.dataset["annotTool"] === tool) {
                groupActive = true;
                break;
              }
            }
          }
          el.classList.toggle("lector-tool-group__trigger--active", groupActive);
        }
      }));
      this.#pushCleanup(effect2(() => {
        const selId = this.#annotation.selectedAnnotation.value;
        if (!selId) {
          this.#hideAnnotPopover();
          if (this.#activeThreadAnnotId !== null) {
            this.#activeThreadAnnotId = null;
            this.#refreshCommentsSidebar();
          }
          return;
        }
        if (selId === this.#activeThreadAnnotId) {
          return;
        }
        let annotData = null;
        for (const id of this.#allOpenDocIds()) {
          const found = this.#annotation.getForDocument(id).find((t) => t.id === selId);
          if (found) {
            annotData = found.data;
            break;
          }
        }
        if (!annotData) {
          this.#hideAnnotPopover();
          return;
        }
        const isStickyNote = annotData.subtype === 1;
        if (isStickyNote) {
          this.#hideAnnotPopover();
        } else {
          this.#showAnnotPopover(selId);
        }
        if (isToolOutputAnnotation(annotData.tag, annotData.subtype)) return;
        this.#activeThreadAnnotId = selId;
        this.#setCommentsSidebarCollapsed(false);
        this.#scheduleSidebarRefresh();
        setTimeout(() => {
          if (this.#activeThreadAnnotId !== selId) return;
          this.#scrollSidebarToActiveThread();
          const annotEl = this.#canvas.querySelector(`[data-annot-id="${selId}"]`);
          if (annotEl) {
            annotEl.scrollIntoView({ block: "nearest", inline: "nearest" });
          }
        }, 200);
      }));
      this.#pushCleanup(effect2(() => {
        const ids = this.#annotation.selectedAnnotations.value;
        if (ids.length < 2) {
          this.#hideMultiSelectBar();
        } else {
          this.#showMultiSelectBar(ids);
        }
      }));
      this.#pushCleanup(this.#annotation.subscribe((event) => {
        if (event.type === "updated" && event.patch !== void 0) {
          const patchKeys = Object.keys(event.patch);
          if (patchKeys.length === 1 && patchKeys[0] === "readAt") return;
        }
        this.#scheduleSidebarRefresh();
      }));
      this.#pushCleanup(effect2(() => {
        void this.#document.activeDocument.value;
        if (!this.#commentsSidebarCollapsed.peek()) {
          this.#scheduleSidebarRefresh();
        }
      }));
      const events2 = this.#engine.plugins.events;
      this.#pushCleanup(events2.on("annotation:updated", () => {
        const selId = this.#annotation.selectedAnnotation.peek();
        if (selId && this.#annotPopover.style.display !== "none") {
          const active = document.activeElement;
          if (active instanceof HTMLInputElement && active.type === "range" && this.#annotPopover.contains(active)) {
            return;
          }
          this.#showAnnotPopover(selId);
        }
      }));
      this.#pushCleanup(events2.on("annotation:drag-start", () => {
        this.#annotPopover.style.display = "none";
        this.#annotDragging = true;
        this.#cancelPendingSidebarRefresh();
      }));
      this.#pushCleanup(events2.on("annotation:drag-end", (...args) => {
        const annotId = args[0];
        this.#annotDragging = false;
        if (annotId) this.#showAnnotPopover(annotId);
        this.#scheduleSidebarRefresh();
      }));
      let annotSidebarTimer = null;
      this.#pushCleanup(this.#annotation.subscribe((event) => {
        if (event.type === "updated" && event.patch !== void 0) {
          const patchKeys = Object.keys(event.patch);
          if (patchKeys.length === 1 && patchKeys[0] === "readAt") return;
        }
        const activePanel = this.#ui.state.sidebar.activePanel.peek();
        if (activePanel === "annotations") {
          if (annotSidebarTimer) clearTimeout(annotSidebarTimer);
          annotSidebarTimer = setTimeout(() => {
            annotSidebarTimer = null;
            this.#updateSidebarActiveTab();
          }, 150);
        }
      }));
    }
    let lastScale = this.#viewportInstance.scale.peek();
    this.#pushCleanup(effect2(() => {
      const pos = this.#viewportInstance.pagePositions.value;
      const h = this.#viewportInstance.totalHeight.value;
      const newScale = this.#viewportInstance.scale.value;
      this.#updatePages(pos, h);
      if (newScale !== lastScale) {
        lastScale = newScale;
        this.#renderedPages.clear();
        this.#overlays.rebuildOverlays();
      }
    }));
    this.#pushCleanup(effect2(() => {
      const vis = this.#viewportInstance.visiblePages.value;
      void this.#viewportInstance.scale.value;
      void this.#viewportInstance.docId.value;
      void this.#renderVisiblePages(vis);
    }));
    const events = this.#engine.plugins.events;
    this.#pushCleanup(events.on("layer:visibility-changed", () => {
      this.#renderedPages.clear();
      const vis = this.#viewportInstance.visiblePages.peek();
      void this.#renderVisiblePages(vis);
    }));
    if (this.#capture) {
      this.#pushCleanup(events.on("capture:region-selected", (...args) => {
        const payload = args[0];
        void this.#handleCaptureRegion(payload.rect, payload.docId, payload.viewportId);
      }));
      this.#pushCleanup(events.on("capture:cancelled", () => {
        this.#hideCaptureActionBar();
      }));
      this.#pushCleanup(events.on("capture:mode-disabled", () => {
        this.#hideCaptureActionBar();
      }));
    }
    if (this.#docManager) {
      this.#pushCleanup(events.on("document-manager:opened", (...args) => {
        const info = args[0];
        this.#registerLoadedHandle(info.handle, info.name);
      }));
      this.#pushCleanup(events.on("document-manager:open-failed", (...args) => {
        const payload = args[0];
        const name = payload?.source?.fileName ?? payload?.source?.url ?? "document";
        const errMsg = payload?.error instanceof Error ? payload.error.message : payload?.error != null ? String(payload.error) : "unknown error";
        this.#showToast(this.#t("toast.openNamedFailed", { name, error: errMsg }));
      }));
      this.#pushCleanup(events.on("document-manager:drop-rejected", (...args) => {
        const payload = args[0];
        if (payload?.reason !== "not-pdf") return;
        if ((payload.files?.length ?? 0) === 0) return;
        const key = payload.partial ? "dropzone.partialNotPdf" : "dropzone.notPdf";
        const fallback = payload.partial ? "Some files were ignored \u2014 only PDF files can be opened" : "Only PDF files can be opened";
        this.#showToast(this.#t(key) || fallback);
      }));
      const enableDropZone = this.#engine.enableViewerDropZone ?? true;
      if (enableDropZone) {
        const workspace = this.#root.querySelector(".lector-workspace");
        if (workspace) {
          this.#pushCleanup(
            this.#docManager.registerDropZone(workspace, {
              multiple: true,
              promptText: this.#t("dropzone.releaseToOpen") || "Release to load PDF"
            })
          );
        }
      }
    }
    this.#pushCleanup(events.on("document:closed", (...args) => {
      const closedId = args[0];
      this.#validationCache.delete(closedId);
      this.#sigInfoCache.delete(closedId);
    }));
    this.#pushCleanup(events.on("page-ops:pages-changed", () => {
      this.#renderedPages.clear();
      this.#tileManager.clearAll();
      const pos = this.#viewportInstance.pagePositions.peek();
      const h = this.#viewportInstance.totalHeight.peek();
      this.#updatePages(pos, h);
      requestAnimationFrame(() => {
        const vis = this.#viewportInstance.visiblePages.peek();
        if (vis.length > 0) void this.#renderVisiblePages(vis);
      });
    }));
    this.#pushCleanup(events.on("redaction:applied", (...args) => {
      const docId = args[0];
      const pageIndex = args[1];
      const active = this.#document.activeDocument.peek();
      if (!active || active.id !== docId) return;
      this.#renderedPages.delete(pageIndex);
      this.#tileManager.clearPage(pageIndex);
      this.#refreshThumbnail(docId, pageIndex);
      this.#scheduleVisibleRerender();
    }));
    this.#pushCleanup(events.on("ui:document-open", () => this.#openFileDialog()));
    this.#pushCleanup(events.on("ui:document-close", () => {
      const id = this.#getActiveDocId();
      if (id) void this.closeDocument(id);
    }));
    this.#pushCleanup(events.on("ui:document-save", () => void this.#saveDocument()));
    this.#pushCleanup(events.on("ui:document-export", () => void this.#saveDocument()));
    this.#pushCleanup(events.on("ui:document-print", () => this.#showPrintDialog()));
    this.#pushCleanup(events.on("ui:toggle-comments-sidebar", () => {
      this.#refreshCommentsSidebar();
      this.toggleCommentsSidebar();
    }));
    this.#pushCleanup(events.on("ui:split-horizontal", () => {
      this.#splitToEmpty("horizontal");
    }));
    this.#pushCleanup(events.on("ui:split-vertical", () => {
      this.#splitToEmpty("vertical");
    }));
    this.#pushCleanup(events.on("ui:close-extra-panes", () => {
      const tab = this.#activeTab();
      if (tab && tab.kind === "split") {
        void this.closeTabSide(this.#activeTabIndex, "right").then(() => {
          this.#buildToolbar();
        });
      }
    }));
    if (this.#comparison) {
      this.#pushCleanup(effect2(() => {
        const state = this.#comparison.state.value;
        const result = this.#comparison.result.value;
        void this.#viewport.viewports.value;
        if (state === "active" && result) {
          this.#applyComparisonOverlays(result);
          this.#wireComparisonSyncScroll();
        } else {
          this.#unwireComparisonSyncScroll();
          this.#applyComparisonOverlays(null);
          this.#activeChangeIndex = -1;
        }
        this.#updateCompareButton();
        const activePanel = this.#ui.state.sidebar.activePanel.peek();
        if (activePanel === "comparison") {
          this.#updateSidebarActiveTab();
        }
      }));
      this.#pushCleanup(events.on("comparison:entered", () => {
        if (this.#ui.state.sidebar.collapsed.peek()) {
          this.#ui.setSidebarCollapsed(false);
        }
        if (this.#ui.state.sidebar.activePanel.peek() !== "comparison") {
          this.#ui.setActivePanel("comparison");
        }
      }));
      this.#pushCleanup(effect2(() => {
        void this.#viewport.viewports.value;
        void this.#document.activeDocument.value;
        this.#updateCompareButton();
      }));
    }
    this.#pushCleanup(effect2(() => {
      const active = this.#viewport.activeViewport.value;
      const activeContainer = active?.container ?? null;
      const isPrimaryActive = activeContainer === this.#canvas;
      this.#canvas.classList.toggle("lector-canvas--active", isPrimaryActive);
      this.#canvasHost.classList.toggle(
        "lector-canvas-host--active",
        isPrimaryActive
      );
      for (const pane of this.#extraPanes.values()) {
        pane.container.classList.toggle(
          "lector-pane--active",
          pane.canvas === activeContainer
        );
      }
      if (active) {
        const pinned = active.docId.peek();
        if (pinned !== null) {
          const current = this.#document.activeDocument.peek();
          if (current?.id !== pinned) {
            this.#document.setActive(pinned);
          }
          const tab = this.#activeTab();
          if (tab && tab.kind === "split") {
            const newSide = tab.right?.docId === pinned ? "right" : "left";
            if (this.#activeTabSide !== newSide) {
              this.#activeTabSide = newSide;
              this.#updateDocTabs();
            }
          }
        }
      }
    }));
    this.#pushCleanup(events.on("ui:document-screenshot", () => void this.#exportPageAsImage()));
    this.#pushCleanup(events.on("ui:document-protect", () => this.#showProtectDialog()));
    this.#pushCleanup(events.on("ui:document-sign", () => {
      const doc = this.#document.activeDocument.peek();
      if (!doc) {
        this.#showToast(this.#t("toast.noDocument"));
        return;
      }
      this.#showSignatureDialog({ docId: doc.id, pageIndex: 0, fieldName: "Signature1", rect: void 0 });
    }));
    const keyHandler = (e) => {
      const tag = e.target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || e.target?.isContentEditable) return;
      if (e.ctrlKey && !e.shiftKey && e.key === "p") {
        e.preventDefault();
        this.#showPrintDialog();
        return;
      }
      if (e.ctrlKey && !e.shiftKey && e.key === "s") {
        e.preventDefault();
        void this.#saveDocument();
        return;
      }
      if (e.ctrlKey && !e.shiftKey && e.key === "f") {
        e.preventDefault();
        this.#showSearchBar();
        return;
      }
      const commands = this.#engine.plugins.commands;
      for (const cmd of commands.getAll().values()) {
        if (!cmd.shortcut) continue;
        if (cmd.enabled !== void 0 && !cmd.enabled.value) continue;
        if (this.#matchShortcut(e, cmd.shortcut)) {
          e.preventDefault();
          void cmd.execute();
          return;
        }
      }
    };
    document.addEventListener("keydown", keyHandler);
    this.#pushCleanup(() => document.removeEventListener("keydown", keyHandler));
    this.#pushCleanup(events.on("search:open", () => this.#showSearchBar()));
    this.#pushCleanup(events.on("signature:field-click", (...args) => {
      const detail = args[0];
      this.#showSignatureDialog(detail);
    }));
    const contextHandler = (e) => {
      const target = e.target;
      if (!this.#canvas.contains(target)) return;
      const pageEl = target.closest(".lector-page");
      if (pageEl) {
        const idx = [...this.#pageElements.entries()].find(([, el]) => el === pageEl)?.[0] ?? 0;
        this.#showPageContextMenu(e, idx);
      }
    };
    this.#canvas.addEventListener("contextmenu", contextHandler);
    this.#pushCleanup(() => this.#canvas.removeEventListener("contextmenu", contextHandler));
    const kbContextHandler = (e) => {
      if (e.key === "F10" && e.shiftKey || e.key === "ContextMenu") {
        e.preventDefault();
        const doc = this.#document.activeDocument.peek();
        if (!doc) return;
        const visiblePage = this.#viewportInstance.visiblePages.peek()[0] ?? 0;
        const r = this.#canvas.getBoundingClientRect();
        this.#showPageContextMenu(
          new MouseEvent("contextmenu", { clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 }),
          visiblePage
        );
      }
    };
    this.#canvas.addEventListener("keydown", kbContextHandler);
    this.#pushCleanup(() => this.#canvas.removeEventListener("keydown", kbContextHandler));
    if (this.#textLayer) {
      this.#pushCleanup(
        effect2(() => {
          const sel = this.#textLayer.selection.value;
          if (!sel) {
            this.#hideTextSelectionToolbar();
            return;
          }
          setTimeout(() => this.#showTextSelectionToolbar(), 100);
        })
      );
    }
    this.#canvas.addEventListener("scroll", () => {
      this.#hideTextSelectionToolbar();
      this.#repositionAnnotPopover();
    });
    this.#wirePinchToZoom();
  }
  #wirePinchToZoom() {
    const canvas = this.#canvas;
    let initialDistance = 0;
    let initialZoom = 1;
    let active = false;
    const getDistance = (t1, t2) => {
      const dx = t2.clientX - t1.clientX;
      const dy = t2.clientY - t1.clientY;
      return Math.sqrt(dx * dx + dy * dy);
    };
    const onTouchStart = (e) => {
      if (e.touches.length === 2) {
        active = true;
        initialDistance = getDistance(e.touches[0], e.touches[1]);
        initialZoom = this.#zoom.level.peek();
        e.preventDefault();
      }
    };
    const onTouchMove = (e) => {
      if (!active || e.touches.length !== 2) return;
      e.preventDefault();
      const currentDistance = getDistance(e.touches[0], e.touches[1]);
      const scale = currentDistance / initialDistance;
      const newLevel = Math.max(0.1, Math.min(10, initialZoom * scale));
      this.#zoom.setLevel(newLevel);
    };
    const onTouchEnd = (e) => {
      if (e.touches.length < 2) {
        active = false;
      }
    };
    canvas.addEventListener("touchstart", onTouchStart, { passive: false });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    canvas.addEventListener("touchend", onTouchEnd);
    canvas.addEventListener("touchcancel", onTouchEnd);
    this.#pushCleanup(() => {
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("touchend", onTouchEnd);
      canvas.removeEventListener("touchcancel", onTouchEnd);
    });
    const onWheel = (e) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      const absDelta = Math.abs(e.deltaY);
      const factor = absDelta >= 50 ? e.deltaY > 0 ? 1 / 1.1 : 1.1 : 1 + -e.deltaY * 5e-3;
      const current = this.#zoom.level.peek();
      const newLevel = Math.max(0.1, Math.min(10, current * factor));
      if (Math.abs(newLevel - current) < 1e-6) return;
      const rect = canvas.getBoundingClientRect();
      const cursorX = e.clientX - rect.left;
      const cursorY = e.clientY - rect.top;
      const docX = (canvas.scrollLeft + cursorX) / current;
      const docY = (canvas.scrollTop + cursorY) / current;
      this.#zoom.setLevel(newLevel);
      canvas.scrollLeft = docX * newLevel - cursorX;
      canvas.scrollTop = docY * newLevel - cursorY;
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    this.#pushCleanup(() => canvas.removeEventListener("wheel", onWheel));
  }
  // ─── Page Rendering ────────────────────────────────────
  /**
   * Resolve the document handle that the primary canvas is currently
   * showing. In split-tab mode the primary viewport is pinned to the
   * left doc, so the left canvas's render loop must read its OWN
   * pinned doc rather than the engine's active document — otherwise
   * clicking the right pane (which switches active document) would
   * cause the left canvas to re-render with the right doc's pages,
   * making both panes show the same content.
   */
  /**
   * Match a KeyboardEvent against a shortcut string like `'Ctrl+Shift+0'`
   * or `'Alt+ArrowLeft'`. Supports Ctrl, Shift, Alt modifiers and any
   * key name (case-insensitive match against `e.key`).
   */
  #matchShortcut(e, shortcut) {
    const parts = shortcut.split("+");
    let wantCtrl = false;
    let wantShift = false;
    let wantAlt = false;
    let key = "";
    for (const part of parts) {
      const lower = part.toLowerCase();
      if (lower === "ctrl") wantCtrl = true;
      else if (lower === "shift") wantShift = true;
      else if (lower === "alt") wantAlt = true;
      else key = part;
    }
    if (e.ctrlKey !== wantCtrl) return false;
    if (e.shiftKey !== wantShift) return false;
    if (e.altKey !== wantAlt) return false;
    const eKey = e.key.toLowerCase();
    const wantKey = key.toLowerCase();
    if (eKey === wantKey) return true;
    if (wantKey === "=" && eKey === "=") return true;
    if (wantKey === "-" && eKey === "-") return true;
    if (/^\d$/.test(key) && e.code === `Digit${key}`) return true;
    if (eKey === wantKey) return true;
    return false;
  }
  #resolveCanvasDoc() {
    const pinnedId = this.#viewportInstance.docId.peek();
    if (pinnedId !== null) {
      return this.#document.getHandle(pinnedId) ?? null;
    }
    return this.#document.activeDocument.peek();
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
        this.#tileManager.clearPage(idx);
        this.#pendingResizes.delete(idx);
      }
    }
    for (const pos of positions) {
      let pe = this.#pageElements.get(pos.pageIndex);
      let cv = this.#pageCanvases.get(pos.pageIndex);
      if (!pe) {
        pe = this.#el("div", "lector-page lector-page--loading");
        cv = document.createElement("canvas");
        cv.className = "lector-page__canvas";
        pe.appendChild(cv);
        this.#scrollArea.appendChild(pe);
        this.#pageElements.set(pos.pageIndex, pe);
        this.#pageCanvases.set(pos.pageIndex, cv);
      }
      const doc = this.#resolveCanvasDoc();
      if (doc) {
        const ps = doc.pageSizes[pos.pageIndex];
        if (ps) this.#overlays.attachPage(pos.pageIndex, pe, ps.width, ps.height);
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
          this.#schedulePixelResize(pos.pageIndex, cv, cw, ch, pe);
        }
      }
    }
  }
  /**
   * Schedule a debounced canvas pixel-buffer resize. During rapid zoom,
   * only CSS dimensions change (instant GPU scaling). When zoom settles
   * (~150ms idle), all pending resizes fire and sharp re-renders start.
   */
  #schedulePixelResize(pageIndex, cv, w, h, pe) {
    this.#pendingResizes.set(pageIndex, { cv, w, h, pe });
    if (this.#pixelResizeTimer) clearTimeout(this.#pixelResizeTimer);
    this.#pixelResizeTimer = setTimeout(() => {
      this.#pixelResizeTimer = null;
      this.#flushPixelResizes();
    }, 150);
  }
  /** Flush all pending pixel-buffer resizes and trigger re-render. */
  #flushPixelResizes() {
    for (const [pageIndex, { cv, w, h }] of this.#pendingResizes) {
      if (cv.width !== w || cv.height !== h) {
        cv.width = w;
        cv.height = h;
        this.#renderedPages.delete(pageIndex);
      }
    }
    this.#pendingResizes.clear();
    const vis = this.#viewportInstance.visiblePages.peek();
    if (vis.length > 0) void this.#renderVisiblePages(vis);
  }
  /** The docId that #renderedPages belongs to. When it changes, the set is stale. */
  #renderedForDoc = null;
  async #renderVisiblePages(visible) {
    const doc = this.#resolveCanvasDoc();
    if (!doc) return;
    if (this.#renderedForDoc !== doc.id) {
      this.#renderedPages.clear();
      this.#tileManager.clearAll();
      this.#renderedForDoc = doc.id;
    }
    for (const idx of visible) {
      const cv = this.#pageCanvases.get(idx);
      if (!cv) continue;
      const fullW = cv.width;
      const fullH = cv.height;
      if (fullW === 0 || fullH === 0) continue;
      if (this.#tileManager.shouldTile(fullW, fullH)) {
        const pe = this.#pageElements.get(idx);
        if (!pe) continue;
        const scrollEl = this.#canvas;
        const dpr = window.devicePixelRatio || 1;
        const scale = this.#viewportInstance.scale.peek();
        const pageRect = pe.getBoundingClientRect();
        const scrollRect = scrollEl.getBoundingClientRect();
        const vpLeft = Math.max(0, scrollRect.left - pageRect.left) * dpr;
        const vpTop = Math.max(0, scrollRect.top - pageRect.top) * dpr;
        const vpRight = Math.min(pageRect.width, scrollRect.right - pageRect.left) * dpr;
        const vpBottom = Math.min(pageRect.height, scrollRect.bottom - pageRect.top) * dpr;
        const viewportRect = {
          x: vpLeft,
          y: vpTop,
          w: Math.max(0, vpRight - vpLeft),
          h: Math.max(0, vpBottom - vpTop)
        };
        const capturedDocId = doc.id;
        const renderFn = async (req) => {
          const currentDoc = this.#resolveCanvasDoc();
          if (!currentDoc || currentDoc.id !== capturedDocId) {
            throw new Error("Document changed \u2014 tile stale");
          }
          return this.#engine.renderPageTile(
            capturedDocId,
            req.pageIndex,
            req.tileX,
            req.tileY,
            req.tileW,
            req.tileH,
            req.fullW,
            req.fullH
          );
        };
        const tiles = this.#tileManager.updateVisibleTiles(
          doc.id,
          idx,
          fullW,
          fullH,
          viewportRect,
          scale,
          renderFn
        );
        const ctx = cv.getContext("2d");
        if (ctx) {
          for (const tile of tiles) {
            if (tile.status === "ready" && tile.bitmap) {
              ctx.drawImage(tile.bitmap, tile.x, tile.y, tile.w, tile.h);
            }
          }
        }
        const allReady = tiles.every((t) => t.status === "ready");
        if (allReady && tiles.length > 0) {
          this.#renderedPages.add(idx);
          if (pe) pe.classList.remove("lector-page--loading");
        } else {
          this.#renderedPages.delete(idx);
        }
        continue;
      }
      if (this.#renderedPages.has(idx)) continue;
      this.#renderedPages.add(idx);
      try {
        const capturedDocId = doc.id;
        const bmp = await this.#engine.renderPage(doc.id, idx, fullW, fullH, { priority: RenderPriority.VISIBLE });
        const currentDoc = this.#resolveCanvasDoc();
        if (!currentDoc || currentDoc.id !== capturedDocId) {
          bmp.close();
          this.#renderedPages.delete(idx);
          continue;
        }
        const stale = cv.width !== fullW || cv.height !== fullH;
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
  // ─── Tab + split machinery ─────────────────────────────
  //
  // A tab is either a single document or a side-by-side split of two
  // documents. The viewer hosts one chrome (toolbar, sidebar, doctabs)
  // and reuses its primary canvas (#canvas + #viewportInstance) as the
  // "left" pane of a split tab. The "right" pane is mounted as an
  // extra LectorPane sibling inside #canvasWrap with a divider between
  // them. Switching tabs unmounts whichever extra pane (if any) the
  // previous tab had and mounts whatever the new tab needs.
  //
  // All panes share one engine, one document store, one annotation
  // store — they only differ in viewport state (scroll, scale, layout)
  // and which doc id they pin. Click in any pane to make it the active
  // viewport for the toolbar and sidebar.
  /** True if the active tab is a split. */
  get isSplit() {
    const tab = this.#activeTab();
    return tab !== null && tab.kind === "split";
  }
  /** Read-only view of the current tab list. */
  get tabs() {
    return this.#tabs;
  }
  #activeTab() {
    return this.#tabs[this.#activeTabIndex] ?? null;
  }
  #findTabIndexByDoc(docId) {
    for (let i = 0; i < this.#tabs.length; i++) {
      const tab = this.#tabs[i];
      if (tab.kind === "single" && tab.docId === docId) return i;
      if (tab.kind === "split" && (tab.left.docId === docId || tab.right?.docId === docId)) {
        return i;
      }
    }
    return -1;
  }
  /** All doc ids the viewer currently has loaded across all tabs. */
  #allOpenDocIds() {
    const ids = [];
    for (const tab of this.#tabs) {
      if (tab.kind === "single") ids.push(tab.docId);
      else {
        ids.push(tab.left.docId);
        if (tab.right) ids.push(tab.right.docId);
      }
    }
    return ids;
  }
  /**
   * Resolve the viewport instance and canvas element that currently
   * displays the given doc id. Returns the primary viewport when no
   * pane is showing this doc (e.g. the doc is in an inactive tab).
   */
  #findPaneForDoc(docId) {
    if (this.#viewportInstance.docId.peek() === docId) {
      return { viewport: this.#viewportInstance, canvas: this.#canvas };
    }
    for (const pane of this.#extraPanes.values()) {
      if (pane.viewport.docId.peek() === docId) {
        return { viewport: pane.viewport, canvas: pane.canvas };
      }
    }
    return { viewport: this.#viewportInstance, canvas: this.#canvas };
  }
  #getDocName(docId) {
    for (const tab of this.#tabs) {
      if (tab.kind === "single" && tab.docId === docId) return tab.name;
      if (tab.kind === "split") {
        if (tab.left.docId === docId) return tab.left.name;
        if (tab.right?.docId === docId) return tab.right.name;
      }
    }
    return null;
  }
  /**
   * The doc id currently in front for toolbar / sidebar context. For
   * single tabs this is the only doc; for split tabs this is the doc
   * pinned to the active side (which follows whichever pane the user
   * last clicked into). When the active side is the empty placeholder
   * we fall back to the loaded side so the chrome stays operable.
   */
  #getActiveDocId() {
    const tab = this.#activeTab();
    if (!tab) return null;
    if (tab.kind === "single") return tab.docId;
    if (this.#activeTabSide === "right" && tab.right) return tab.right.docId;
    return tab.left.docId;
  }
  /**
   * Make the canvas wrap match the active tab's structure. Mounts or
   * unmounts the extra pane, pins the primary viewport to the
   * appropriate doc, syncs CSS classes, refreshes the doctab bar, and
   * activates whichever side should have focus.
   *
   * Idempotent: safe to call after any tab-list mutation.
   */
  #applyActiveTab() {
    const tab = this.#activeTab();
    for (const pane of this.#extraPanes.values()) {
      const host2 = pane.container;
      pane.destroy();
      const prev = host2.previousElementSibling;
      if (prev?.classList.contains("lector-pane-divider")) prev.remove();
      host2.remove();
    }
    this.#extraPanes.clear();
    this.#canvasWrap.querySelectorAll(
      ":scope > .lector-pane--empty, :scope > .lector-pane-divider"
    ).forEach((el) => el.remove());
    this.#canvasHost.style.flex = "";
    this.#canvasHost.style.width = "";
    this.#canvasHost.style.height = "";
    if (!tab) {
      this.#canvasWrap.classList.remove(
        "lector-canvas-wrap--split",
        "lector-canvas-wrap--horizontal",
        "lector-canvas-wrap--vertical"
      );
      this.#viewportInstance.setDocument(null);
      this.#updateDocTabs();
      return;
    }
    if (tab.kind === "single") {
      this.#canvasWrap.classList.remove(
        "lector-canvas-wrap--split",
        "lector-canvas-wrap--horizontal",
        "lector-canvas-wrap--vertical"
      );
      this.#renderedPages.clear();
      this.#tileManager.clearAll();
      this.#viewportInstance.setDocument(tab.docId);
      this.#document.setActive(tab.docId);
      this.#viewport.setActiveViewport(this.#viewportInstance.id);
      this.#updateDocTabs();
      this.#zoom.fitPage();
      requestAnimationFrame(() => {
        const vis = this.#viewportInstance.visiblePages.peek();
        if (vis.length > 0) void this.#renderVisiblePages(vis);
        this.#overlays.rebuildOverlays();
      });
      return;
    }
    this.#canvasWrap.classList.add("lector-canvas-wrap--split");
    this.#canvasWrap.classList.toggle(
      "lector-canvas-wrap--horizontal",
      tab.orientation === "horizontal"
    );
    this.#canvasWrap.classList.toggle(
      "lector-canvas-wrap--vertical",
      tab.orientation === "vertical"
    );
    this.#renderedPages.clear();
    this.#tileManager.clearAll();
    this.#viewportInstance.setDocument(tab.left.docId);
    const divider = this.#el(
      "div",
      `lector-pane-divider lector-pane-divider--${tab.orientation}`
    );
    divider.setAttribute("role", "separator");
    divider.setAttribute(
      "aria-orientation",
      tab.orientation === "horizontal" ? "vertical" : "horizontal"
    );
    this.#wirePaneDivider(divider, tab.orientation);
    this.#canvasHost.insertAdjacentElement("afterend", divider);
    const host = this.#el("div", "lector-pane");
    host.dataset["paneId"] = `right-${++this.#nextExtraPaneId}`;
    divider.insertAdjacentElement("afterend", host);
    const tabIndex = this.#activeTabIndex;
    const rightInfo = tab.right;
    if (!rightInfo) {
      this.#activeTabSide = "left";
      this.#viewport.setActiveViewport(this.#viewportInstance.id);
      this.#document.setActive(tab.left.docId);
      host.classList.add("lector-pane--empty");
      this.#buildEmptyPanePlaceholder(host, tabIndex, tab.orientation);
      this.#updateDocTabs();
      this.#zoom.fitPage();
      requestAnimationFrame(() => {
        const vis = this.#viewportInstance.visiblePages.peek();
        if (vis.length > 0) void this.#renderVisiblePages(vis);
      });
      return;
    }
    void import("./lector-pane-CHK6NFAQ.js").then(({ LectorPane: LectorPane2 }) => {
      const pane = new LectorPane2({
        engine: this.#engine,
        container: host,
        docId: rightInfo.docId
      });
      this.#extraPanes.set(host.dataset["paneId"], pane);
      const targetSide = this.#activeTabSide;
      if (targetSide === "right") {
        this.#viewport.setActiveViewport(pane.viewport.id);
        this.#document.setActive(rightInfo.docId);
      } else {
        this.#viewport.setActiveViewport(this.#viewportInstance.id);
        this.#document.setActive(tab.left.docId);
      }
      this.#updateDocTabs();
      this.#zoom.fitPage();
      requestAnimationFrame(() => {
        const vis = this.#viewportInstance.visiblePages.peek();
        if (vis.length > 0) void this.#renderVisiblePages(vis);
      });
    });
    this.#updateDocTabs();
  }
  /**
   * Convert the active single tab into a split tab with an empty
   * placeholder on the right. The placeholder shows a centered "Open
   * PDF to compare" button and accepts file drops directly. The user
   * picks the second document from inside the empty pane — no surprise
   * OS file picker on click.
   */
  #splitToEmpty(orientation) {
    const tab = this.#activeTab();
    if (!tab || tab.kind !== "single") return;
    const tabIndex = this.#activeTabIndex;
    this.#tabs[tabIndex] = {
      kind: "split",
      orientation,
      left: { docId: tab.docId, name: tab.name },
      right: null
    };
    this.#activeTabSide = "left";
    this.#applyActiveTab();
  }
  /**
   * Build the empty-pane placeholder DOM into `host`. Renders a centered
   * call-to-action button + drop hint, attaches per-pane file drop
   * handlers (with stopPropagation so the workspace-level dropzone never
   * sees them), and a click handler that opens the file picker scoped
   * to filling this pane.
   */
  #buildEmptyPanePlaceholder(host, tabIndex, orientation) {
    host.innerHTML = "";
    const inner = this.#el("div", "lector-pane-empty__inner");
    host.appendChild(inner);
    const btn = this.#btn("lector-pane-empty__btn");
    btn.type = "button";
    btn.appendChild(this.#icon("plus"));
    const lbl = this.#el("span", "lector-pane-empty__label");
    lbl.textContent = this.#t("split.openToCompare") || "Open PDF to compare";
    btn.appendChild(lbl);
    btn.addEventListener("click", () => {
      this.#pickFileForEmptyPane(tabIndex, orientation);
    });
    inner.appendChild(btn);
    const hint = this.#el("div", "lector-pane-empty__hint");
    hint.textContent = this.#t("split.dropHint") || "or drop a PDF here";
    inner.appendChild(hint);
    let dragDepth = 0;
    const isFileDrag = (e) => e.dataTransfer?.types.includes("Files") === true;
    const onDragEnter = (e) => {
      if (!isFileDrag(e)) return;
      e.preventDefault();
      e.stopPropagation();
      dragDepth++;
      if (dragDepth === 1) host.classList.add("lector-pane-empty--over");
    };
    const onDragOver = (e) => {
      if (!isFileDrag(e)) return;
      e.preventDefault();
      e.stopPropagation();
      if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
    };
    const onDragLeave = (e) => {
      if (!isFileDrag(e)) return;
      e.stopPropagation();
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) host.classList.remove("lector-pane-empty--over");
    };
    const onDrop = (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragDepth = 0;
      host.classList.remove("lector-pane-empty--over");
      const file = Array.from(e.dataTransfer?.files ?? []).find(
        (f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf")
      );
      if (!file) {
        this.#showToast(this.#t("dropzone.notPdf"));
        return;
      }
      void file.arrayBuffer().then((data) => {
        this.#fillEmptyPane(tabIndex, orientation, data, file.name);
      });
    };
    host.addEventListener("dragenter", onDragEnter);
    host.addEventListener("dragover", onDragOver);
    host.addEventListener("dragleave", onDragLeave);
    host.addEventListener("drop", onDrop);
  }
  /**
   * Open the OS file picker scoped to filling the right pane of a split
   * tab. The split must already exist (and have an empty right side).
   */
  #pickFileForEmptyPane(tabIndex, orientation) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".pdf,application/pdf";
    input.addEventListener("change", () => {
      const f = input.files?.[0];
      if (!f) return;
      void f.arrayBuffer().then((data) => {
        this.#fillEmptyPane(tabIndex, orientation, data, f.name);
      });
    });
    input.click();
  }
  /**
   * Load `data` as a new document and pin it into the right side of an
   * existing split tab whose right side is currently empty. Handles
   * password-protected files by routing them through a toast (per the
   * existing limitation in `#showPasswordModalForSplit`).
   */
  #fillEmptyPane(tabIndex, orientation, data, name) {
    const tab = this.#tabs[tabIndex];
    if (!tab || tab.kind !== "split" || tab.right !== null) return;
    void this.#document.load(data).then((handle) => {
      const current = this.#tabs[tabIndex];
      if (!current || current.kind !== "split" || current.right !== null) {
        void this.#document.close(handle.id).catch(() => {
        });
        return;
      }
      this.#tabs[tabIndex] = {
        kind: "split",
        orientation,
        left: current.left,
        right: { docId: handle.id, name }
      };
      if (this.#activeTabIndex === tabIndex) {
        this.#activeTabSide = "right";
      }
      this.#applyActiveTab();
      this.#prefetchSignatures(handle);
    }).catch((err) => {
      const isPasswordErr = typeof err === "object" && err !== null && "code" in err && err["code"] === 4;
      if (isPasswordErr) {
        this.#showPasswordModalForSplit(orientation, data, name);
      } else {
        const msg = err instanceof Error ? err.message : typeof err === "object" && err !== null && "message" in err ? String(err.message) : String(err);
        this.#showToast(this.#t("toast.openFailed", { error: msg }));
      }
    });
  }
  /**
   * Variant of `#showPasswordModal` that wires the resulting decrypted
   * doc into a split-tab flow instead of `loadDocument`. Uses the same
   * UI by delegating to `#showPasswordModal` and intercepting via the
   * document plugin's load result.
   *
   * The split-creation flow does not currently support password-
   * protected second files end-to-end via the modal — for v1 we just
   * surface a toast pointing the user at the regular Open flow. A
   * future iteration can plumb a custom password modal here.
   */
  #showPasswordModalForSplit(_orientation, _data, name) {
    this.#showToast(
      `${name} is password-protected \u2014 open it as its own tab first, then split.`
    );
  }
  /**
   * Make a divider between two panes draggable, redistributing space
   * between the previous-sibling pane and the next-sibling pane via
   * flex-grow on each side.
   */
  #wirePaneDivider(divider, direction) {
    divider.tabIndex = 0;
    divider.setAttribute("role", "separator");
    divider.setAttribute("aria-orientation", direction);
    divider.setAttribute("aria-label", direction === "horizontal" ? "Resize panes horizontally" : "Resize panes vertically");
    divider.addEventListener("keydown", (e) => {
      const relevantKeys = direction === "horizontal" ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
      if (!relevantKeys.includes(e.key)) return;
      e.preventDefault();
      const prev = divider.previousElementSibling;
      const next = divider.nextElementSibling;
      if (!prev || !next) return;
      const step = e.shiftKey ? 40 : 10;
      const prevRect = prev.getBoundingClientRect();
      const nextRect = next.getBoundingClientRect();
      const prevSize2 = direction === "horizontal" ? prevRect.width : prevRect.height;
      const nextSize2 = direction === "horizontal" ? nextRect.width : nextRect.height;
      const grow = e.key === "ArrowRight" || e.key === "ArrowDown";
      const delta = grow ? step : -step;
      const minSize = 120;
      const newPrev = Math.max(minSize, prevSize2 + delta);
      const newNext = Math.max(minSize, nextSize2 - delta);
      prev.style.flex = `0 0 ${newPrev}px`;
      next.style.flex = `0 0 ${newNext}px`;
    });
    let dragging = false;
    let startCoord = 0;
    let prevSize = 0;
    let nextSize = 0;
    let prevEl = null;
    let nextEl = null;
    const onDown = (e) => {
      e.preventDefault();
      prevEl = divider.previousElementSibling;
      nextEl = divider.nextElementSibling;
      if (!prevEl || !nextEl) return;
      dragging = true;
      divider.setPointerCapture(e.pointerId);
      const prevRect = prevEl.getBoundingClientRect();
      const nextRect = nextEl.getBoundingClientRect();
      if (direction === "horizontal") {
        startCoord = e.clientX;
        prevSize = prevRect.width;
        nextSize = nextRect.width;
      } else {
        startCoord = e.clientY;
        prevSize = prevRect.height;
        nextSize = nextRect.height;
      }
      divider.classList.add("lector-pane-divider--dragging");
    };
    const onMove = (e) => {
      if (!dragging || !prevEl || !nextEl) return;
      const coord = direction === "horizontal" ? e.clientX : e.clientY;
      const delta = coord - startCoord;
      const minSize = 120;
      const newPrev = Math.max(minSize, prevSize + delta);
      const newNext = Math.max(minSize, nextSize - delta);
      if (direction === "horizontal") {
        prevEl.style.flex = `0 0 ${newPrev}px`;
        nextEl.style.flex = `0 0 ${newNext}px`;
      } else {
        prevEl.style.flex = `0 0 ${newPrev}px`;
        nextEl.style.flex = `0 0 ${newNext}px`;
      }
    };
    const onUp = (e) => {
      if (!dragging) return;
      dragging = false;
      try {
        divider.releasePointerCapture(e.pointerId);
      } catch {
      }
      divider.classList.remove("lector-pane-divider--dragging");
    };
    divider.addEventListener("pointerdown", onDown);
    divider.addEventListener("pointermove", onMove);
    divider.addEventListener("pointerup", onUp);
    divider.addEventListener("pointercancel", onUp);
    this.#pushCleanup(() => {
      divider.removeEventListener("pointerdown", onDown);
      divider.removeEventListener("pointermove", onMove);
      divider.removeEventListener("pointerup", onUp);
      divider.removeEventListener("pointercancel", onUp);
    });
  }
  // ─── Public API ────────────────────────────────────────
  /**
   * Load a PDF document from raw bytes into a new tab.
   *
   * @param data   - The complete PDF file as an ArrayBuffer.
   * @param password - Optional document-open password for encrypted PDFs.
   * @param name   - Display name shown in the tab bar (defaults to "Document N").
   */
  async loadDocument(data, password, name) {
    const handle = await this.#document.load(data, password);
    this.#registerLoadedHandle(handle, name);
  }
  /**
   * Load a PDF from a URL with optional custom headers.
   *
   * @example
   * ```ts
   * await viewer.loadDocumentFromUrl('/api/docs/report.pdf', {
   *   headers: { Authorization: `Bearer ${token}` },
   *   credentials: 'include',
   * });
   * ```
   */
  async loadDocumentFromUrl(url, options) {
    const { password, name, ...fetchOptions } = options ?? {};
    const handle = await this.#engine.openDocumentFromUrl(url, { ...fetchOptions, password });
    this.#registerLoadedHandle(handle, name ?? url.split("/").pop()?.split("?")[0]);
  }
  /**
   * Wire an already-loaded `DocumentHandle` into the viewer's tabs,
   * render queue, and signature pre-fetch. Used both by `loadDocument`
   * (which loads from raw bytes) and by the `document-manager:opened`
   * event listener (which receives docs opened via drag-and-drop, the
   * file dialog, or recent files).
   *
   * Idempotent: a no-op if the handle is already tracked in any tab.
   * Always creates a new single-doc tab and switches to it; the new
   * tab does not auto-merge with the active tab.
   */
  #registerLoadedHandle(handle, name) {
    if (this.#findTabIndexByDoc(handle.id) !== -1) return;
    const displayName = name ?? `Document ${this.#tabs.length + 1}`;
    this.#tabs.push({ kind: "single", docId: handle.id, name: displayName });
    this.#activeTabIndex = this.#tabs.length - 1;
    this.#activeTabSide = "left";
    this.#applyActiveTab();
    this.#prefetchSignatures(handle);
    if (this.#pendingInitialZoom !== void 0) {
      const z = this.#pendingInitialZoom;
      this.#pendingInitialZoom = void 0;
      requestAnimationFrame(() => {
        if (z === "fit-width") this.#zoom.fitWidth();
        else if (z === "fit-page") this.#zoom.fitPage();
        else this.#zoom.setLevel(z);
      });
    }
  }
  /**
   * Pre-fetch signature info + validation for a document so the first
   * click on the signatures tab is instant. Cheap RPC; never blocks the
   * caller.
   */
  #prefetchSignatures(handle) {
    if (!this.#signature) return;
    void this.#signature.getAllInfo(handle.id).then((sigs) => {
      this.#sigInfoCache.set(handle.id, sigs);
      this.#updateSignatureStatusBadge();
      if (sigs.length > 0 && this.#sigValidation && !this.#validationCache.has(handle.id)) {
        this.#validationCache.set(handle.id, "pending");
        this.#updateSignatureStatusBadge();
        void this.#sigValidation.validateAll(handle.id).then((results) => {
          this.#validationCache.set(handle.id, results);
          this.#updateSignatureStatusBadge();
        }).catch((err) => {
          const msg = err instanceof Error ? err.message : String(err);
          this.#validationCache.set(handle.id, { error: msg });
          this.#updateSignatureStatusBadge();
        });
      }
    }).catch(() => {
    });
  }
  /**
   * Switch to whichever tab contains the given doc id. Backward-compat
   * shim around the new tab model — prefer `switchTab(index)` for
   * direct tab navigation.
   */
  switchDocument(docId) {
    const idx = this.#findTabIndexByDoc(docId);
    if (idx === -1) return;
    const tab = this.#tabs[idx];
    let side = "left";
    if (tab.kind === "split" && tab.right?.docId === docId) side = "right";
    this.switchTab(idx, side);
  }
  /**
   * Switch to a tab by index. For split tabs, optionally focus a
   * specific side. For same-tab side switches inside a split, the
   * extra pane is reused (no rebuild) — only the active-viewport
   * pointer is updated, which lets the effect downstream update the
   * active document, the doctab highlight, and the chrome.
   */
  switchTab(index, side) {
    if (index < 0 || index >= this.#tabs.length) return;
    const sameIndex = index === this.#activeTabIndex;
    const sameSide = !side || side === this.#activeTabSide;
    if (sameIndex && sameSide) return;
    if (sameIndex && side) {
      const tab = this.#activeTab();
      if (!tab || tab.kind !== "split") return;
      if (side === "left") {
        this.#viewport.setActiveViewport(this.#viewportInstance.id);
      } else {
        const rightPane = this.#extraPanes.values().next().value;
        if (rightPane) {
          this.#viewport.setActiveViewport(rightPane.viewport.id);
        }
      }
      return;
    }
    this.#activeTabIndex = index;
    this.#activeTabSide = side ?? "left";
    this.#applyActiveTab();
  }
  /**
   * Close the document with the given id. Backward-compat shim — for
   * split tabs this closes only the matching side and demotes the tab
   * back to a single. To close the whole tab use `closeTab(index)`.
   */
  async closeDocument(docId) {
    const idx = this.#findTabIndexByDoc(docId);
    if (idx === -1) return;
    const tab = this.#tabs[idx];
    if (tab.kind === "single") {
      await this.closeTab(idx);
    } else {
      const side = tab.left.docId === docId ? "left" : "right";
      await this.closeTabSide(idx, side);
    }
  }
  /** Close an entire tab. For split tabs this closes both documents. */
  async closeTab(index) {
    if (index < 0 || index >= this.#tabs.length) return;
    const tab = this.#tabs[index];
    const docs = tab.kind === "single" ? [tab.docId] : tab.right ? [tab.left.docId, tab.right.docId] : [tab.left.docId];
    this.#tabs.splice(index, 1);
    if (this.#activeTabIndex === index) {
      if (this.#tabs.length === 0) {
        this.#activeTabIndex = -1;
      } else {
        this.#activeTabIndex = Math.min(index, this.#tabs.length - 1);
      }
      this.#activeTabSide = "left";
    } else if (this.#activeTabIndex > index) {
      this.#activeTabIndex--;
    }
    this.#applyActiveTab();
    for (const id of docs) {
      this.#validationCache.delete(id);
      this.#sigInfoCache.delete(id);
      await this.#document.close(id);
    }
  }
  /**
   * Close one side of a split tab, demoting it to a single tab. The
   * remaining doc continues to be the tab's content. No-op if the
   * tab is not a split or the index is out of bounds.
   *
   * Special cases:
   *  - Closing the empty placeholder side (right === null on the side
   *    being closed): nothing to free in the engine; just demote.
   *  - Closing the loaded side while the other side is still empty: the
   *    tab would have no content left, so close the whole tab.
   */
  async closeTabSide(index, side) {
    if (index < 0 || index >= this.#tabs.length) return;
    const tab = this.#tabs[index];
    if (tab.kind !== "split") return;
    const closing = side === "left" ? tab.left : tab.right;
    const remaining = side === "left" ? tab.right : tab.left;
    if (!closing) {
      if (!remaining) {
        await this.closeTab(index);
        return;
      }
      this.#tabs[index] = {
        kind: "single",
        docId: remaining.docId,
        name: remaining.name
      };
      if (this.#activeTabIndex === index) this.#activeTabSide = "left";
      this.#applyActiveTab();
      return;
    }
    if (!remaining) {
      await this.closeTab(index);
      return;
    }
    this.#tabs[index] = {
      kind: "single",
      docId: remaining.docId,
      name: remaining.name
    };
    if (this.#activeTabIndex === index) {
      this.#activeTabSide = "left";
    }
    this.#applyActiveTab();
    this.#validationCache.delete(closing.docId);
    this.#sigInfoCache.delete(closing.docId);
    await this.#document.close(closing.docId);
  }
  /**
   * Render the doctab bar. Single tabs show as `[ name (x) ]`. Split
   * tabs show as `[ left | right (x) ]` where each side is clickable
   * to focus that pane and has a hover-revealed per-side close button.
   */
  #updateDocTabs() {
    this.#docTabs.innerHTML = "";
    if (this.#tabs.length === 0) return;
    this.#docTabs.setAttribute("role", "tablist");
    this.#docTabs.setAttribute("aria-label", this.#t("page.documents"));
    const isSingle = this.#tabs.length === 1 && !this.#allowLocalOpen;
    this.#docTabs.classList.toggle("lector-doctabs--single", isSingle);
    for (let i = 0; i < this.#tabs.length; i++) {
      const tab = this.#tabs[i];
      const isActive = i === this.#activeTabIndex;
      if (tab.kind === "single") {
        const tabEl = this.#el("div", "lector-doctab" + (isActive ? " lector-doctab--active" : ""));
        tabEl.setAttribute("role", "tab");
        tabEl.setAttribute("aria-selected", String(isActive));
        tabEl.tabIndex = isActive ? 0 : -1;
        const lbl = this.#el("span", "lector-doctab__label");
        lbl.textContent = tab.name;
        tabEl.appendChild(lbl);
        const closeBtn = this.#btn("lector-doctab__close");
        closeBtn.setAttribute("aria-label", this.#t("toolbar.close"));
        closeBtn.innerHTML = resolveIcon("x") ?? "";
        const tabIndex = i;
        closeBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          void this.closeTab(tabIndex);
        });
        tabEl.appendChild(closeBtn);
        tabEl.addEventListener("click", () => this.switchTab(tabIndex));
        this.#docTabs.appendChild(tabEl);
      } else {
        const tabEl = this.#el(
          "div",
          "lector-doctab lector-doctab--split" + (isActive ? " lector-doctab--active" : "")
        );
        const tabIndex = i;
        const makeSide = (which, info) => {
          const isEmpty = info === null;
          const sideEl = this.#el(
            "span",
            "lector-doctab__side lector-doctab__side--" + which + (isActive && this.#activeTabSide === which ? " lector-doctab__side--active" : "") + (isEmpty ? " lector-doctab__side--empty" : "")
          );
          const sideLbl = this.#el("span", "lector-doctab__side-label");
          sideLbl.textContent = isEmpty ? this.#t("split.emptySideLabel") || "Empty" : info.name;
          sideEl.appendChild(sideLbl);
          const sideClose = this.#btn("lector-doctab__side-close");
          sideClose.innerHTML = resolveIcon("x") ?? "";
          this.#tip(sideClose, this.#t("split.closeThisPane"));
          sideClose.addEventListener("click", (e) => {
            e.stopPropagation();
            void this.closeTabSide(tabIndex, which);
          });
          sideEl.appendChild(sideClose);
          sideEl.addEventListener("click", (e) => {
            e.stopPropagation();
            if (isEmpty) return;
            this.switchTab(tabIndex, which);
          });
          return sideEl;
        };
        tabEl.appendChild(makeSide("left", tab.left));
        const sep = this.#el("span", "lector-doctab__sep");
        sep.textContent = "|";
        tabEl.appendChild(sep);
        tabEl.appendChild(makeSide("right", tab.right));
        const closeBtn = this.#btn("lector-doctab__close");
        closeBtn.innerHTML = resolveIcon("x") ?? "";
        this.#tip(closeBtn, this.#t("split.closeSplit"));
        closeBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          void this.closeTab(tabIndex);
        });
        tabEl.appendChild(closeBtn);
        tabEl.addEventListener("click", () => this.switchTab(tabIndex));
        this.#docTabs.appendChild(tabEl);
      }
    }
    if (this.#allowLocalOpen) {
      const add = this.#btn("lector-doctab lector-doctab--add");
      add.innerHTML = resolveIcon("plus") ?? "+";
      this.#tip(add, this.#t("split.openPdf"));
      add.addEventListener("click", () => this.#openFileDialog());
      this.#docTabs.appendChild(add);
    }
  }
  #openFileDialog() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".pdf,application/pdf";
    input.addEventListener("change", () => {
      const f = input.files?.[0];
      if (f) void f.arrayBuffer().then((d) => {
        void this.loadDocument(d, void 0, f.name).catch((err) => {
          const isPasswordErr = typeof err === "object" && err !== null && "code" in err && err["code"] === 4;
          if (isPasswordErr) {
            this.#showPasswordModal(d, f.name);
          } else {
            this.#showToast(
              this.#t("toast.openFailed", {
                error: err instanceof Error ? err.message : typeof err === "object" && err !== null && "message" in err ? String(err.message) : String(err)
              })
            );
          }
        });
      });
    });
    input.click();
  }
  /** The {@link LectorEngine} instance powering this viewer. */
  get engine() {
    return this.#engine;
  }
  /** The UI capability for programmatic sidebar / theme / schema control. */
  get ui() {
    return this.#ui;
  }
  /**
   * Tear down the viewer: remove DOM, disconnect observers, and free
   * all resources. The engine is NOT destroyed — the consumer owns it
   * and may attach another viewer or call `engine.destroy()` separately.
   */
  destroy() {
    this.#tileManager.destroy();
    this.#overlays.destroy();
    for (const u of this.#cleanups) u();
    this.#cleanups.length = 0;
    for (const [, list] of this.#sections) {
      for (const u of list) {
        try {
          u();
        } catch {
        }
      }
    }
    this.#sections.clear();
    if (this.#tooltipTimer) clearTimeout(this.#tooltipTimer);
    if (this.#pixelResizeTimer) clearTimeout(this.#pixelResizeTimer);
    if (this.#commentsRefreshTimer) {
      clearTimeout(this.#commentsRefreshTimer);
      this.#commentsRefreshTimer = null;
    }
    this.#unwireComparisonSyncScroll();
    this.#tooltipEl.remove();
    for (const pane of this.#extraPanes.values()) pane.destroy();
    this.#extraPanes.clear();
    this.#viewportInstance.destroy();
    this.#pageElements.clear();
    this.#pageCanvases.clear();
    this.#renderedPages.clear();
    this.#root.innerHTML = "";
    const allDocIds = [];
    for (const tab of this.#tabs) {
      if (tab.kind === "single") allDocIds.push(tab.docId);
      else {
        allDocIds.push(tab.left.docId);
        if (tab.right) allDocIds.push(tab.right.docId);
      }
    }
    this.#tabs = [];
    this.#activeTabIndex = -1;
    for (const id of allDocIds) void this.#document.close(id).catch(() => {
    });
  }
  [Symbol.dispose]() {
    this.destroy();
  }
};

// src/index.ts
import { FpdfRenderFlag } from "@truespar/lector-pdfium-wasm";
export {
  ANNOTATION_TOOL_DEFAULTS,
  AnnotationStore,
  BlendMode,
  BreakpointObserver,
  CommandRegistry,
  DEFAULT_BREAKPOINTS,
  DEFAULT_UI_SCHEMA,
  DirtyTracker,
  EngineError,
  EngineErrorCode,
  EventBus,
  FormStore,
  FormattingManager,
  FpdfRenderFlag,
  I18nManager,
  LectorEngine,
  LectorPane,
  LectorViewer,
  LineCap,
  MeasurementUnit,
  NoteIcon,
  OperationLog,
  PageOverlayManager,
  PluginRegistry,
  RenderPriority,
  TOOL_TO_SUBTYPE,
  TileManager,
  UIManager,
  acroformJSPlugin,
  annotationPlugin,
  annotationPresetsPlugin,
  attachmentPlugin,
  buildViewerClass,
  capturePlugin,
  comparisonPlugin,
  da as daTranslations,
  de as deTranslations,
  definePlugin,
  documentManagerPlugin,
  documentPlugin,
  en as enTranslations,
  es as esTranslations,
  fi as fiTranslations,
  formPlugin,
  formattingPlugin,
  getIcon,
  historyPlugin,
  i18nPlugin,
  imageBitmapToPngBlob,
  injectLectorStyles,
  interactionPlugin,
  isCategoryVisible,
  isEraserTool,
  isInkTool,
  isInlineSvg,
  isMarkupTool,
  isMeasurementTool,
  isPlacementTool,
  isPolygonTool,
  isRedactionTool,
  isShapeTool,
  isStampTool,
  isToolOutputAnnotation,
  isToolOutputTool,
  isUserAnnotation,
  layerPlugin,
  linearizationPlugin,
  measurementPlugin,
  mergeSchema,
  mergeSplitPlugin,
  navigationPlugin,
  nb as nbTranslations,
  pageOpsPlugin,
  performancePlugin,
  redactionPlugin,
  renderPlugin,
  resolveIcon,
  searchPlugin,
  signaturePlugin,
  signatureSigningPlugin,
  signatureValidationPlugin,
  sv as svTranslations,
  textLayerPlugin,
  uiPlugin,
  viewportPlugin,
  zoomPlugin
};
