import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

type Theme = 'light' | 'dark'

function initialTheme(): Theme {
  const stored = localStorage.getItem('pk_theme')
  if (stored === 'light' || stored === 'dark') return stored
  // First-visit fallback: follow the system preference.
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

/** null = "model maximum": send whatever the context window has left after the
 *  prompt, rather than a fixed number. A local GGUF has no output limit of its
 *  own - the context is the ceiling - so a constant here was only ever a guess,
 *  and at 8192 it truncated artifacts mid-tool-call.
 *
 *  The stored 8192 is cleared once: nobody chose it, it was the default, and
 *  leaving it would mean the fix reached only fresh installs. An explicit
 *  choice (any other value) is the user's and survives. */
function initialMaxTokens(): number | null {
  if (!localStorage.getItem('pk_max_tokens_v2')) {
    localStorage.setItem('pk_max_tokens_v2', '1')
    if (localStorage.getItem('pk_max_tokens') === '8192') {
      localStorage.removeItem('pk_max_tokens')
    }
  }
  const raw = localStorage.getItem('pk_max_tokens')
  if (raw == null || raw === 'max') return null
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? n : null
}

/** null = send no `max_tool_calls` at all, so the server's own budget applies.
 *  Anything else is a number the user chose deliberately. */
function initialMaxToolCalls(): number | null {
  const raw = localStorage.getItem('pk_max_tool_calls')
  if (raw == null || raw === 'server') return null
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : null
}

/** UI/user preferences. Each persists to localStorage on change. */
export const useSettingsStore = defineStore('settings', () => {
  const theme = ref<Theme>(initialTheme())
  // Cap on one reply (thinking + answer share it), or null for "model maximum"
  // - the window minus the prompt, computed per send. Slider in Settings.
  const maxTokens = ref<number | null>(initialMaxTokens())
  /** Ceiling on how many tools one reply may run, or null for "whatever the
   * server allows" - the Responses API's own `max_tool_calls`,
   *  ridden as a request parameter rather than kept as model config, because
   *  it is a property of this ask and not of the endpoint.
   *
   *  It exists because the server's budget was previously unreachable: a turn
   *  that spent it printed "[tool budget spent: 8 rounds...]" and there was
   * nowhere to change the 8. Null is the default and sends
   *  nothing, so the server's own bounds apply untouched; a number both caps
   *  the calls AND raises the server's round ceiling to match, which is what
   *  makes this a real knob rather than a way to ask for less. */
  const maxToolCalls = ref<number | null>(initialMaxToolCalls())
  // Context compaction: summarize older messages when a chat outgrows the
  // window (on by default); off = drop the oldest messages, the old behavior.
  const summarize = ref<boolean>(localStorage.getItem('pk_summarize') !== 'off')
  // Mark the words a speech model was least sure of. A VIEWER preference, the
  // way Rev's "show low confidence words" is: the marks help you find what to
  // check, and they are noise once you have. On by default because the Studio
  // is for judging models, not for producing a clean read.
  const markUnsure = ref<boolean>(localStorage.getItem('pk_mark_unsure') !== 'off')
  /** Which transcriber the composer's mic dictates with, by model id.
   *  A user who picks their good multilingual model once should not
   *  re-pick it every turn - and it is a PREFERENCE, not conversation state:
   *  the same ears follow you into a new chat. Empty = "whichever is running",
   *  which is also what a stale id falls back to. */
  const dictateWith = ref<string>(localStorage.getItem('pk_dictate_with') ?? '')
  /** Which microphone every mic path opens. Empty = the system default, which
   *  is also what a box with one input never has to think about.
   *
   *  A PREFERENCE and not conversation state, for the same reason `dictateWith`
   *  is: it is a fact about this machine's hardware, so it follows you into
   *  every chat and outlives all of them. The id is opaque and origin-scoped -
   *  it means nothing on another browser or machine, which is fine, since
   *  neither does the hardware it names. */
  const micDeviceId = ref<string>(localStorage.getItem('pk_mic_device') ?? '')
  /** ...and what it was CALLED when it was chosen. Stored beside the id purely
   *  so a device that is unplugged can still be named: `enumerateDevices` only
   *  lists what is connected, so without this the honest report degrades from
   *  "your Jabra headset isn't here" to "the microphone you chose isn't
   *  here" - which is the difference between knowing what to plug in and
   *  guessing. */
  const micDeviceLabel = ref<string>(localStorage.getItem('pk_mic_device_label') ?? '')
  /** Raster tile template for the interactive map a geotagged photo can open
   *  (layer 3). Empty = OSM's own tile server, named in the UI.
   *
   *  It is a SETTING and not a constant for a reason that is about the user,
   *  not about taste. Fetching a tile tells that host, request by request,
   *  where the photos in front of you were taken - so the destination has to
   *  be the user's to choose, and to change to their own server or a paid
   *  one. It is also what OSM's tile policy asks of anyone whose users would
   *  pull from it systematically. The map itself stays behind a click; this
   *  decides where that click goes.
   *
   *  A preference of this BROWSER, like the theme: nothing about it belongs
   *  in a server's config or the manager's DB. */
  const mapTiles = ref<string>(localStorage.getItem('pk_map_tiles') ?? '')

  watch(theme, (v) => {
    localStorage.setItem('pk_theme', v)
    document.documentElement.setAttribute('data-theme', v)
  })
  watch(maxTokens, (v) => {
    if (v == null) localStorage.setItem('pk_max_tokens', 'max')
    else localStorage.setItem('pk_max_tokens', String(v))
  })
  watch(maxToolCalls, (v) => {
    localStorage.setItem('pk_max_tool_calls', v == null ? 'server' : String(v))
  })
  watch(summarize, (v) => {
    localStorage.setItem('pk_summarize', v ? 'on' : 'off')
  })
  watch(markUnsure, (v) => {
    localStorage.setItem('pk_mark_unsure', v ? 'on' : 'off')
  })

  watch(dictateWith, (v) => {
    if (v) localStorage.setItem('pk_dictate_with', v)
    else localStorage.removeItem('pk_dictate_with')
  })
  watch(micDeviceId, (v) => {
    if (v) localStorage.setItem('pk_mic_device', v)
    else localStorage.removeItem('pk_mic_device')
  })
  watch(micDeviceLabel, (v) => {
    if (v) localStorage.setItem('pk_mic_device_label', v)
    else localStorage.removeItem('pk_mic_device_label')
  })
  watch(mapTiles, (v) => {
    const t = v.trim()
    if (t) localStorage.setItem('pk_map_tiles', t)
    else localStorage.removeItem('pk_map_tiles')
  })

  return {
    theme,
    maxTokens,
    maxToolCalls,
    summarize,
    markUnsure,
    dictateWith,
    micDeviceId,
    micDeviceLabel,
    mapTiles,
  }
})
