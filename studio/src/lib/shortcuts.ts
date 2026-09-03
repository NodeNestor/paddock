// The Studio's keyboard shortcuts, in one file so the label a tooltip prints
// and the keys a handler tests cannot drift apart. A shortcut nobody can see
// is not a shortcut, so every chord here is expected to appear in the UI that
// triggers the same action by mouse.
//
// Both chords are ones a web page can still claim. Chrome reserves a short
// list (Ctrl+T/N/W and their Shift variants, Ctrl+Tab, Alt+F4) and hands
// everything else to the renderer first, so preventDefault genuinely wins;
// Firefox's own Ctrl+B (bookmarks sidebar) is preventable too. If a browser
// ever does eat one, the click target is right there and nothing breaks.

/** The chord prefix for LABELS. Never used for matching - see `mod()`. */
export const MOD = /Mac|iPhone|iPad/.test(navigator.userAgent) ? '⌘' : 'Ctrl'

export const NEW_CHAT = `${MOD}+Shift+O`
export const TOGGLE_CHATS = `${MOD}+B`
export const SEARCH_CHATS = `${MOD}+K`

/** Cmd on Apple, Ctrl elsewhere - but accept either, so a Windows keyboard on
 *  a Mac (and the reverse) still works rather than silently doing nothing. */
function mod(e: KeyboardEvent): boolean {
  return e.ctrlKey || e.metaKey
}

/** The pressed letter, lowercased. `key` rather than `code` deliberately: it
 *  follows the user's keyboard LAYOUT, which is what someone on a Dvorak or
 *  AZERTY board expects from a text app. Ctrl+Shift+O arrives as 'O'. */
function letter(e: KeyboardEvent): string {
  return e.key.length === 1 ? e.key.toLowerCase() : ''
}

export function isNewChat(e: KeyboardEvent): boolean {
  return mod(e) && e.shiftKey && !e.altKey && letter(e) === 'o'
}

export function isToggleChats(e: KeyboardEvent): boolean {
  return mod(e) && !e.shiftKey && !e.altKey && letter(e) === 'b'
}

export function isSearchChats(e: KeyboardEvent): boolean {
  return mod(e) && !e.shiftKey && !e.altKey && letter(e) === 'k'
}
