//! Frame-level speech detection over raw PCM - the DSP half of server VAD
//!
//! This answers exactly one question, once per 20 ms frame: *is there speech
//! energy here?* Everything that turns those verdicts into a TURN - how long
//! silence must last before an utterance is over, how much audio to keep from
//! before speech started - is API semantics and lives with the session that
//! serves them (`paddock-runner`'s realtime lane). The split is the same one
//! the mel frontend has: signal work here, policy there.
//!
//! ## What this is, and what the state of the art is
//!
//! This is an energy detector with an adaptive noise floor: RMS per frame in
//! dBFS, a floor read off the MINIMUM of the last five seconds, and a frame
//! counts as speech when it stands far enough above that floor. Minimum
//! statistics (Martin 2001, the standard noise-floor estimator) is what makes
//! it level-independent - a quiet mic and a hot one both work - and what
//! learns a steady hum or fan as room rather than transcribing it forever:
//! noise has no pauses, so its minimum is its level, while speech dips between
//! words and its minimum sits far below its peaks.
//!
//! The SOTA is not this. Since ~2021 the field runs small neural detectors -
//! Silero VAD (the de facto open default), TEN VAD, FireRedVAD - which are far
//! better than any energy rule in noise, on music beds, and against non-speech
//! transients. They cost 1-2 MB and a few hundred microseconds a frame.
//!
//! We do not ship one, and the reason is structural rather than a shortcut:
//! those are MODELS, and paddock runs models on the GPU only - there is no
//! host inference path here to hang one on, by design. Serving a neural VAD
//! means loading it as an engine model with its own kernels, which is a real
//! project and a different task. Until then this detector is honest about
//! being what it is: it will fire on a slammed door and it will miss a whisper
//! under traffic noise, and the session it drives always leaves the client the
//! manual `input_audio_buffer.commit` that needs no detector at all.
//!
//! The threshold knob is the OpenAI one (0..1, 0.5 default, "higher requires
//! louder audio"), mapped onto the SNR margin below.

/// Frame length. 20 ms is the standard VAD frame - long enough for a stable
/// RMS at speech frequencies, short enough that a turn boundary lands within
/// one frame of where a listener would put it.
pub const FRAME_MS: usize = 20;

/// How far back the minimum looks. Long enough that a talker's pauses fall
/// inside it (so speech never becomes the floor), short enough to follow a
/// room that changes - a fan starting, a window opening.
const WINDOW_FRAMES: usize = 5000 / FRAME_MS;

/// The minimum of a noisy signal sits below its average level, so a floor read
/// straight off the minimum is pessimistic. Martin's estimator corrects this
/// analytically; a flat few dB is the honest cheap version.
const BIAS_DB: f32 = 3.0;

/// Until the window has filled, the floor is not allowed to sit above this.
/// A session that opens with the speaker MID-WORD would otherwise learn speech
/// as its room and then hear nothing at all - while the minimum is still
/// guesswork, assume a quiet room. Once five seconds are in, the measurement
/// stands on its own and a genuinely loud room is allowed to be one.
const FLOOR_MAX_DB: f32 = -40.0;
/// The quietest floor worth tracking. Below this it is digital silence and the
/// absolute gate decides anyway.
const FLOOR_MIN_DB: f32 = -90.0;
/// Past this the "room" is loud enough to be clipping, and treating it as
/// floor would mean hearing nothing at all.
const CEIL_DB: f32 = -12.0;

/// A frame this quiet is silence whatever the floor says. Stops a muted mic's
/// dither - which sits 30 dB above a floor of -90 - from reading as speech.
const ABS_GATE_DB: f32 = -60.0;

/// One frame's verdict, with absolute sample positions counted from the first
/// sample this detector ever saw. Absolute so a caller that trims its buffer
/// can still line frames up with wall-clock offsets into the session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub start: usize,
    pub end: usize,
    pub speech: bool,
    /// the frame's RMS in dBFS, for logging and tests
    pub db: f32,
}

pub struct Vad {
    frame: usize,
    /// samples of a partial frame held over from the last `feed`
    carry: Vec<f32>,
    /// absolute index of the next frame's first sample
    pos: usize,
    /// dB a frame must stand above the floor to count as speech
    margin_db: f32,
    /// Sliding-window minimum as a monotonic deque of `(frame index, dB)`,
    /// increasing front to back - the front is the window's minimum, and
    /// keeping it this way costs O(1) a frame instead of rescanning 250.
    mins: std::collections::VecDeque<(usize, f32)>,
    /// frames judged so far, which is also the current frame's index
    n: usize,
}

impl Vad {
    /// `threshold` is OpenAI's `server_vad.threshold`: 0..1, higher demands
    /// louder audio. It maps to the SNR margin - 6 dB at 0 (twitchy), 15 dB at
    /// the 0.5 default (a normal room), 24 dB at 1 (shout only).
    pub fn new(rate: u32, threshold: f32) -> Self {
        let frame = (rate as usize * FRAME_MS / 1000).max(1);
        let t = threshold.clamp(0.0, 1.0);
        Self {
            frame,
            carry: Vec::with_capacity(frame),
            pos: 0,
            margin_db: 6.0 + 18.0 * t,
            mins: std::collections::VecDeque::with_capacity(WINDOW_FRAMES),
            n: 0,
        }
    }

    /// Samples per frame - what a caller needs to size a pre-roll.
    pub fn frame_len(&self) -> usize {
        self.frame
    }

    /// Absolute index one past the last sample this detector has consumed.
    /// Samples in `carry` are not counted: they have not been judged yet.
    pub fn consumed(&self) -> usize {
        self.pos
    }

    /// Feed newly arrived PCM; get back a verdict for every whole frame that
    /// completed. A partial frame is held until the rest arrives, so a caller
    /// may append in whatever chunk sizes the wire gives it.
    pub fn feed(&mut self, pcm: &[f32]) -> Vec<Frame> {
        let mut out = Vec::with_capacity((self.carry.len() + pcm.len()) / self.frame + 1);
        let mut src = pcm;
        // finish the carried frame first, then walk whole frames in place
        if !self.carry.is_empty() {
            let need = self.frame - self.carry.len();
            let take = need.min(src.len());
            self.carry.extend_from_slice(&src[..take]);
            src = &src[take..];
            if self.carry.len() < self.frame {
                return out;
            }
            let carry = std::mem::take(&mut self.carry);
            out.push(self.judge(&carry));
            self.carry = carry;
            self.carry.clear();
        }
        let whole = src.len() / self.frame;
        for i in 0..whole {
            out.push(self.judge(&src[i * self.frame..(i + 1) * self.frame]));
        }
        self.carry.extend_from_slice(&src[whole * self.frame..]);
        out
    }

    /// One frame: level, floor, verdict.
    ///
    /// The current frame goes into the window before the floor is read off it.
    /// A minimum can only fall, so a loud frame cannot raise its own bar - but
    /// a drop to silence lowers the floor on the very frame that dropped,
    /// which is what makes the end of a turn land where a listener puts it.
    fn judge(&mut self, f: &[f32]) -> Frame {
        let sum: f64 = f.iter().map(|&s| (s as f64) * (s as f64)).sum();
        let rms = (sum / f.len() as f64).sqrt().max(1e-10);
        let db = (20.0 * rms.log10()) as f32;

        while self.mins.back().is_some_and(|&(_, v)| v >= db) {
            self.mins.pop_back();
        }
        self.mins.push_back((self.n, db));
        while self
            .mins
            .front()
            .is_some_and(|&(i, _)| i + WINDOW_FRAMES <= self.n)
        {
            self.mins.pop_front();
        }
        let mut floor = self.mins.front().map_or(db, |&(_, v)| v) + BIAS_DB;
        if self.n + 1 < WINDOW_FRAMES {
            floor = floor.min(FLOOR_MAX_DB);
        }
        let floor = floor.clamp(FLOOR_MIN_DB, CEIL_DB);
        self.n += 1;

        let speech = db > floor + self.margin_db && db > ABS_GATE_DB;
        let start = self.pos;
        self.pos += f.len();
        Frame {
            start,
            end: self.pos,
            speech,
            db,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16000;

    /// A tone at `amp`, `n` samples.
    fn tone(n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / RATE as f32).sin())
            .collect()
    }

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    #[test]
    fn frames_come_out_whole_and_in_order_whatever_the_chunking() {
        let mut v = Vad::new(RATE, 0.5);
        let f = v.frame_len();
        assert_eq!(f, 320);
        // a chunk smaller than a frame yields nothing yet
        assert!(v.feed(&silence(100)).is_empty());
        // ... and the rest of that frame completes it
        let out = v.feed(&silence(220));
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].start, out[0].end), (0, 320));
        // two and a half frames: two out, half carried
        let out = v.feed(&silence(f * 2 + 160));
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].start, out[1].end), (320, 960));
        assert_eq!(v.consumed(), 960);
    }

    #[test]
    fn quiet_room_then_speech_flips_and_flips_back() {
        let mut v = Vad::new(RATE, 0.5);
        // room tone at about -66 dBFS
        let room = tone(RATE as usize, 0.0007);
        assert!(
            v.feed(&room).iter().all(|f| !f.speech),
            "room tone read as speech"
        );
        // a talker at about -20 dBFS
        let loud = v.feed(&tone(RATE as usize / 2, 0.14));
        assert!(loud.iter().filter(|f| f.speech).count() > loud.len() * 9 / 10);
        // back to the room: silence again within a frame or two
        let after = v.feed(&room);
        assert!(
            after[3..].iter().all(|f| !f.speech),
            "did not fall back to silence"
        );
    }

    #[test]
    fn a_session_that_opens_mid_word_still_hears_it() {
        // the floor clamp is what makes this work: without it the first frames
        // of speech would teach the detector that speech is the room
        let mut v = Vad::new(RATE, 0.5);
        let out = v.feed(&tone(RATE as usize / 2, 0.14));
        assert!(out[0].speech, "first frame missed");
        assert!(out.iter().filter(|f| f.speech).count() > out.len() * 9 / 10);
    }

    #[test]
    fn a_steady_hum_is_learned_as_the_room_and_stops_being_speech() {
        let mut v = Vad::new(RATE, 0.5);
        // twenty seconds of an unchanging tone: a fan, a fridge, mains hum. It
        // reads as speech at first - it is a level step, and five seconds of
        // history is exactly what the detector does not have yet - and then
        // the window fills with it and it becomes the room. That is what keeps
        // an open mic in a noisy office from transcribing the office forever.
        let out = v.feed(&tone(RATE as usize * 20, 0.25));
        assert!(out[0].speech, "the step itself should trip it");
        let tail = &out[out.len() - 50..];
        assert!(tail.iter().all(|f| !f.speech), "the floor never caught up");
    }

    #[test]
    fn the_threshold_knob_moves_where_the_line_sits() {
        // a signal 10 dB over the room: heard at a lax threshold, not at a
        // strict one. Same audio, same detector, only the knob differs.
        let room = tone(RATE as usize, 0.002);
        let quiet_talker = tone(RATE as usize / 4, 0.0063);
        let mut lax = Vad::new(RATE, 0.0);
        lax.feed(&room);
        let heard = lax.feed(&quiet_talker);
        assert!(heard.iter().any(|f| f.speech), "6 dB margin should hear it");

        let mut strict = Vad::new(RATE, 1.0);
        strict.feed(&room);
        let missed = strict.feed(&quiet_talker);
        assert!(missed.iter().all(|f| !f.speech), "24 dB margin should not");
    }

    #[test]
    fn digital_silence_is_never_speech_however_low_the_floor_goes() {
        let mut v = Vad::new(RATE, 0.0);
        // a long stretch of true zeros drags the floor to its minimum
        v.feed(&silence(RATE as usize * 2));
        // dither an LSB wide sits ~30 dB above that floor but is not speech
        let dither: Vec<f32> = (0..RATE as usize)
            .map(|i| {
                if i % 2 == 0 {
                    1.0 / 32768.0
                } else {
                    -1.0 / 32768.0
                }
            })
            .collect();
        assert!(
            v.feed(&dither).iter().all(|f| !f.speech),
            "dither read as speech"
        );
    }
}
