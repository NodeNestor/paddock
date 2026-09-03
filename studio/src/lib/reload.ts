// Who would lose something if this tab reloaded right now.
//
// The Studio can be running last BUILD'S JavaScript: swapping the exe changes
// nothing in an already-open tab, and AppShell notices by comparing the served
// bundle hash to the running one. That staleness cannot be fixed by refetching
// - the new components are not in memory, so the only way to get them is to
// load them.
//
// What can be fixed is the interruption. A page reload is invisible if it
// happens where a repaint was going to happen anyway: at a navigation. So the
// shell waits for the next route change and turns it into a real one, and the
// user simply arrives on the new build.
//
// Except when something would be thrown away by it. That is what these holds
// are: a component that owns unsaved or in-flight work says so, and the swap
// waits for the banner and a deliberate click instead. Conversations live in
// SQLite and the URL carries the route, so those need no hold - the things
// that do are the ones that exist only in this tab's memory: typed-but-unsent
// text, staged files, a stream in progress, a recording being made.

import { reactive } from 'vue'

const holds = reactive(new Set<string>())

/** Claim (or release) a reason not to reload. Keyed so a component can call
 *  this on every change without counting. */
export function holdReload(key: string, held: boolean): void {
  if (held) holds.add(key)
  else holds.delete(key)
}

/** True when reloading would cost the user something they cannot get back. */
export function reloadHeld(): boolean {
  return holds.size > 0
}
