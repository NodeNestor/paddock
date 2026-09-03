// One transport per CLIP, shared by everything that displays it.
//
// A transcription compare shows the same audio N times. The bytes are
// identical across lanes - only the SEGMENTATION differs - so N players is N
// copies of one thing, and playing the same eight seconds twice to check two
// readings of it is the opposite of what a compare is for. One player and one
// clock drive every lane at once: you hear the word once and watch each lane
// light its own span.
//
// Keyed by ATTACHMENT ID rather than by message or turn, because that is the
// identity of the audio itself. Two lanes answering one clip resolve to the
// same key without anything having to pass a prop down through the compare
// block's structure, and a conversation with several audio turns gets one
// transport each for free.
//
// Deliberately module-level rather than a store: this is transient view state
// (where the playhead is right now), it dies with the page, and nothing wants
// it persisted or in the DB.

import { reactive } from 'vue'

import type { TranscriptGuard } from '@/types/chat'

/** Where each clip's playhead is, in seconds. Reactive so every lane's
 *  highlight follows the one player without a subscription of its own. */
const heads = reactive(new Map<string, number>())

/** How to move each clip's playhead - registered by whoever mounts the actual
 *  transport. Not reactive: it is a callback table, and nothing renders off it. */
const seekers = new Map<string, (t: number) => void>()

/** The clip's real length once something has measured it. The attachment part
 *  carries a duration read in the browser, which some containers refuse to
 *  answer (a MediaRecorder blob has no duration in its header, so Chrome says
 *  Infinity for a clip recorded on this very page). The SERVER measured it to
 *  transcribe it, so once any lane has a transcript the number is known - and
 *  this is how it reaches the player, which sits on the user's turn and has no
 *  transcript of its own. */
const durations = reactive(new Map<string, number>())

/** Spans the decode refused or cut, per clip - the same crossing as the
 *  duration and for the same reason. They are a fact the LANE learned (the
 *  runner reports them with the transcript) about audio the USER's turn owns,
 *  and the player that draws them sits on the user's turn with no transcript
 *  of its own. Painting them on the waveform is what turns "a span of your
 *  audio produced nothing trustworthy" from a line of prose into a place. */
const guards = reactive(new Map<string, TranscriptGuard[]>())

export function useClipPlayback() {
  return {
    /** Current playhead, 0 when this clip has no transport yet. */
    timeOf(clip: string | undefined): number {
      return (clip && heads.get(clip)) || 0
    },
    /** Called by the transport as it plays. */
    publishTime(clip: string | undefined, t: number): void {
      if (clip) heads.set(clip, t)
    },
    /** Longest duration anyone has measured, or 0. */
    durationOf(clip: string | undefined): number {
      return (clip && durations.get(clip)) || 0
    },
    publishDuration(clip: string | undefined, secs: number | undefined): void {
      if (clip && secs && secs > 0 && Number.isFinite(secs)) durations.set(clip, secs)
    },
    /** What any lane reported about this clip's decode. Empty is the normal
     *  answer and the one a clean clip gives. */
    guardsOf(clip: string | undefined): TranscriptGuard[] {
      return (clip && guards.get(clip)) || []
    },
    /** Merged across lanes rather than last-writer-wins: in a compare each
     *  model reports its own refusals, and a span only one of them dropped is
     *  still a span of that audio worth seeing. Deduped on the triple that
     *  identifies a span, so two lanes agreeing draws one band. */
    publishGuards(clip: string | undefined, list: TranscriptGuard[] | undefined): void {
      if (!clip || !list?.length) return
      const have = guards.get(clip) ?? []
      const key = (g: TranscriptGuard): string => `${g.start}|${g.end}|${g.reason}`
      const seen = new Set(have.map(key))
      const add = list.filter((g) => !seen.has(key(g)))
      if (add.length) guards.set(clip, [...have, ...add])
    },
    /** Mount-time registration from the component owning the transport. The
     *  callback reads its own player ref lazily, so registering before the ref
     *  fills in is fine. */
    registerPlayer(clip: string | undefined, seek: (t: number) => void): void {
      if (clip) seekers.set(clip, seek)
    },
    /** Only clears the entry if it is still OURS. A lane re-rendering while
     *  another instance holds the transport must not leave the clip with no
     *  seeker at all. */
    releasePlayer(clip: string | undefined, seek: (t: number) => void): void {
      if (clip && seekers.get(clip) === seek) {
        seekers.delete(clip)
        heads.delete(clip)
      }
    },
    /** Move the clip's playhead.
     *
     *  Loud when it cannot, because the two ways this fails are both invisible
     *  and both look identical to the user: a click that does nothing. The
     *  chain from a word to the audio crosses two message turns and a
     *  by-id registry, and every link in it - the id on the lane, the id on
     *  the user turn, the mounted player, the exposed method - used to be an
     *  `if` or a `?.` that swallowed the miss. A control that a person clicked
     *  and that decided to do nothing has to say so somewhere.
     *
     *  Returns whether it moved, so a caller can react rather than assume. */
    seekClip(clip: string | undefined, t: number): boolean {
      if (!clip) {
        console.warn(
          '[clip] seek ignored: this turn has no clip id - the lane and the ' +
            'user turn disagree about which attachment they are about',
        )
        return false
      }
      const seek = seekers.get(clip)
      if (!seek) {
        console.warn(
          `[clip] seek ignored: no player is mounted for ${clip} ` +
            `(registered: ${[...seekers.keys()].join(', ') || 'none'})`,
        )
        return false
      }
      seek(t)
      return true
    },
  }
}
