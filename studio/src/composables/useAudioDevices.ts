// Which microphone the Studio listens on.
//
// A box has more than one. A laptop with a headset plugged in has three (the
// built-in array, the headset, and whatever the OS calls "default"), a desk
// with an interface has as many as the interface has inputs, and a meeting
// room has the one nobody remembers is still selected. Until now every mic
// path here asked for `audio: true` and took whatever the browser handed back,
// which is the browser's idea of default and not necessarily the user's.
//
// The FAILURE this EXISTS to PREVENT is not "the wrong device is convenient to
// change". It is recording ten minutes on the laptop lid while wearing a
// headset and finding out from the transcript. So the two hard parts here are
// both about honesty rather than about picking:
//
//  - `deviceId: { exact }`, never a bare hint. A plain `deviceId: id` is a
//    PREFERENCE the browser may silently ignore; `exact` makes it refuse. A
//    refusal we can report; a substitution we cannot even see.
//  - a chosen device that has gone away is SAID, not swallowed. The open is
//    retried on the system default so recording still works, and `micLost`
//    names what was missing - before the first word, not after the tenth
//    minute.
//
// The stored choice SURVIVES a disappearance. A headset unplugged for a
// meeting comes back, and quietly forgetting the choice on a transient
// disconnect would be its own small betrayal - so the preference is kept, the
// picker shows the missing device as not connected, and the next open uses it
// again the moment it returns.
//
// LABELS are A PERMISSION, not a device property: before the page has been
// granted a microphone once, `enumerateDevices()` answers with entries whose
// `label` is the empty string (and, in Chrome, one blank placeholder per
// kind). There is therefore no picker to draw until then - which is why the
// composer's menu shows this section only after the first recording, and why
// Settings offers `reveal()` instead of an empty list.

import { readonly, ref } from 'vue'

import { useSettingsStore } from '@/stores/settings'

/** One microphone the browser is willing to name. */
export interface AudioDevice {
  /** `MediaDeviceInfo.deviceId` - stable for this origin until site data is
   *  cleared, and not stable across browsers or machines. */
  id: string
  label: string
}

/** Chrome synthesises two aliases that are not devices: `default` (the OS
 *  default, labelled "Default - Something") and `communications`. Listing them
 *  beside the real entry they alias makes one microphone look like three, and
 *  Firefox has neither - so they are dropped and the picker offers its own
 *  "System default" row, which means the same thing in every browser and means
 *  it by sending no constraint at all. */
const ALIASES = new Set(['default', 'communications'])

const devices = ref<AudioDevice[]>([])
/** The device the last open ASKED for and did not find, by label. Null is the
 *  normal state and the one a clean open restores. */
const lost = ref<string | null>(null)
let watching = false

function supported(): boolean {
  return !!navigator.mediaDevices?.enumerateDevices
}

/** Re-read the device list. Cheap, and safe to call often - it is one
 *  browser call and it settles the same way whether or not permission has been
 *  granted. */
export async function refreshMicDevices(): Promise<void> {
  if (!supported()) {
    devices.value = []
    return
  }
  try {
    const all = await navigator.mediaDevices.enumerateDevices()
    devices.value = all
      .filter((d) => d.kind === 'audioinput' && d.deviceId && !ALIASES.has(d.deviceId))
      .map((d) => ({ id: d.deviceId, label: d.label }))
  } catch {
    // Absence over invention: a browser that refuses to enumerate has no list,
    // and the mic still works on whatever it picks.
    devices.value = []
  }
}

/** Start listening for hardware changes, once per page. Without it the list is
 *  stale the moment somebody plugs in a headset - which is exactly the moment
 *  they open the picker. */
function ensure(): void {
  if (watching || !supported()) return
  watching = true
  navigator.mediaDevices.addEventListener?.('devicechange', () => void refreshMicDevices())
  void refreshMicDevices()
}

/** Ask for the microphone once purely so the browser will name the devices,
 *  then let it go immediately. This is what Settings offers before the first
 *  recording: there is no other way to learn the labels, and a list of three
 *  blanks is not a choice. Returns whether it worked. */
export async function revealMicDevices(): Promise<boolean> {
  if (!navigator.mediaDevices?.getUserMedia) return false
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true })
    s.getTracks().forEach((t) => t.stop())
  } catch {
    return false
  }
  await refreshMicDevices()
  return true
}

/** The processing every microphone path asks for, in one place.
 *
 *  Shared deliberately rather than repeated: a recording and a dictation of
 *  the same sentence must not arrive at the model having been through
 *  different front ends, or a comparison between them is measuring the
 *  browser. */
function constraints(deviceId: string): MediaTrackConstraints {
  return {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
  }
}

/** The exact-constraint refusal, under both spellings. Chrome throws an
 *  `OverconstrainedError` (not a DOMException, and it carries `.constraint`);
 *  a device that vanished between enumeration and open can also come back as
 *  `NotFoundError`. Everything else - permission, no hardware at all, an
 *  insecure origin - is a different fact and belongs to the caller. */
function isMissingDevice(e: unknown): boolean {
  const name = (e as { name?: string } | null)?.name
  return name === 'OverconstrainedError' || name === 'NotFoundError'
}

/** Open the microphone the user chose.
 *
 *  Throws exactly what `getUserMedia` throws, so callers keep their own
 *  wording for permission and missing-hardware - this adds the device layer
 *  and nothing else. */
export async function openMic(): Promise<MediaStream> {
  const settings = useSettingsStore()
  const want = settings.micDeviceId
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: constraints(want) })
    lost.value = null
    // Permission has just been granted (or re-confirmed), so the labels are
    // readable now even if they were blank a moment ago.
    void refreshMicDevices()
    return stream
  } catch (e) {
    if (!want || !isMissingDevice(e)) throw e
    // The chosen device is gone. Record anyway - refusing outright would mean
    // an unplugged headset breaks recording until someone finds a settings
    // page - but never quietly: the note is what the composer shows before the
    // first word, and the preference stays put for when the device comes back.
    const stream = await navigator.mediaDevices.getUserMedia({ audio: constraints('') })
    lost.value = settings.micDeviceLabel || 'The microphone you chose'
    void refreshMicDevices()
    return stream
  }
}

export function useAudioDevices() {
  ensure()
  return {
    /** Whether this page can reach the microphone APIs at all. False on a
     *  plain-http LAN origin, where `navigator.mediaDevices` does not exist -
     *  a browser rule, not a missing device, and worth saying as itself. */
    supported: () => !!navigator.mediaDevices?.getUserMedia,
    /** Every named input the browser will admit to, aliases removed. */
    devices: readonly(devices),
    /** Whether the browser is naming them yet - false until this page has been
     *  granted the microphone once, when every label is the empty string. */
    named: () => devices.value.some((d) => d.label !== ''),
    /** The chosen device is not in the current list. Distinct from `lost`:
     *  this is "it is not plugged in", `lost` is "we tried and it was not
     *  there", and the picker wants the first while the composer wants the
     *  second. */
    missing: (id: string) => !!id && devices.value.every((d) => d.id !== id),
    lost: readonly(lost),
    refresh: refreshMicDevices,
    reveal: revealMicDevices,
  }
}
