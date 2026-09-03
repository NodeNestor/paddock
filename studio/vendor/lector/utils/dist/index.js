// src/reactive/signal.ts
var activeEffect = null;
var batchDepth = 0;
var batchQueue = /* @__PURE__ */ new Set();
function signal(initial) {
  let value = initial;
  const subs = /* @__PURE__ */ new Set();
  const self = {
    get value() {
      if (activeEffect !== null) {
        const eff = activeEffect;
        subs.add(eff);
      }
      return value;
    },
    set value(next) {
      if (Object.is(value, next)) return;
      value = next;
      if (batchDepth > 0) {
        batchQueue.add(subs);
      } else {
        notify(subs, value);
      }
    },
    peek() {
      return value;
    },
    update(fn) {
      self.value = fn(value);
    },
    subscribe(fn) {
      subs.add(fn);
      return () => {
        subs.delete(fn);
      };
    }
  };
  return self;
}
function computed(fn) {
  let value;
  let dirty = true;
  const subs = /* @__PURE__ */ new Set();
  const tracker = () => {
    dirty = true;
    if (subs.size > 0) {
      const prev = value;
      recompute();
      if (!Object.is(prev, value)) {
        notify(subs, value);
      }
    }
  };
  function recompute() {
    const prevEffect = activeEffect;
    activeEffect = tracker;
    try {
      value = fn();
      dirty = false;
    } finally {
      activeEffect = prevEffect;
    }
  }
  const self = {
    get value() {
      if (activeEffect !== null) {
        const eff = activeEffect;
        subs.add(eff);
      }
      if (dirty) recompute();
      return value;
    },
    peek() {
      if (dirty) recompute();
      return value;
    },
    subscribe(fn2) {
      subs.add(fn2);
      if (dirty) recompute();
      return () => {
        subs.delete(fn2);
      };
    }
  };
  return self;
}
function effect(fn) {
  let cleanup;
  let disposed = false;
  function run() {
    if (disposed) return;
    if (typeof cleanup === "function") cleanup();
    const prevEffect = activeEffect;
    activeEffect = run;
    try {
      cleanup = fn();
    } finally {
      activeEffect = prevEffect;
    }
  }
  run();
  return () => {
    disposed = true;
    if (typeof cleanup === "function") cleanup();
  };
}
function batch(fn) {
  batchDepth++;
  try {
    fn();
  } finally {
    batchDepth--;
    if (batchDepth === 0) {
      const queued = [...batchQueue];
      batchQueue.clear();
      for (const subs of queued) {
        for (const sub of subs) {
          sub(void 0);
        }
      }
    }
  }
}
function notify(subs, value) {
  for (const fn of [...subs]) {
    fn(value);
  }
}
export {
  batch,
  computed,
  effect,
  signal
};
