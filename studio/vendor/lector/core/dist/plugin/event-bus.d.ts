import type { Unsubscribe } from '@truespar/lector-utils';
/**
 * Typed event bus for inter-plugin communication.
 *
 * Plugins subscribe to named events via `on()` and broadcast via `emit()`.
 * Disposing the bus clears all listeners, preventing memory leaks.
 */
export declare class EventBus implements Disposable {
    #private;
    /**
     * Subscribe to an event. Returns an unsubscribe function.
     *
     * @param event - The event name to listen for.
     * @param handler - Callback invoked when the event is emitted.
     * @returns A function that removes this listener.
     */
    on(event: string, handler: (...args: unknown[]) => void): Unsubscribe;
    /**
     * Emit an event, invoking all registered listeners synchronously.
     *
     * @param event - The event name to emit.
     * @param args - Arguments forwarded to each listener.
     */
    emit(event: string, ...args: unknown[]): void;
    /** Dispose: clear all listeners. */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=event-bus.d.ts.map