import { type DeepReadonly, type Ref } from 'vue';
import type { ReadonlySignal } from '@truespar/lector-utils';
/**
 * Bridge a Lector signal to a Vue ref.
 *
 * Returns a readonly Vue ref that stays in sync with the signal. The
 * subscription is cleaned up when the calling component/scope is disposed.
 *
 * Uses `shallowRef` to avoid deep reactivity overhead — the signal's value
 * is treated as an opaque snapshot.
 *
 * @typeParam T - The value type held by the signal.
 * @param sig - Any Lector `ReadonlySignal`.
 * @returns A readonly Vue ref tracking the signal's value.
 */
export declare function useSignalRef<T>(sig: ReadonlySignal<T>): DeepReadonly<Ref<T>>;
//# sourceMappingURL=signal-bridge.d.ts.map