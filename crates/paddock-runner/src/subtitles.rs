//! SRT and WebVTT writers for `/v1/audio/transcriptions`
//! (`response_format=srt|vtt`).
//!
//! Both are OpenAI response formats, and both are the point of a transcript
//! feature for most callers: a subtitle file is what you drop next to a video.
//! They carry exactly what segments already know - a start, an end
//! and a line of text - so this file is formatting and nothing else.
//!
//! The two formats look alike and are not interchangeable:
//!
//! | | SRT | WebVTT |
//! |---|---|---|
//! | header | none | `WEBVTT` + a blank line |
//! | cue number | required, 1-based | optional (omitted here) |
//! | decimal separator | `,` | `.` |
//! | text | plain | HTML-ish: `&`, `<`, `>` must be escaped |
//!
//! Getting the separator wrong is the classic bug and it fails loudly in a
//! player (the whole file is rejected), so both are pinned by test.

/// One subtitle cue: seconds from the start of the clip, and its line.
pub struct Cue<'a> {
    pub start: f64,
    pub end: f64,
    pub text: &'a str,
}

/// `HH:MM:SS<sep>mmm`. Hours are always written, which both formats accept and
/// which keeps a >1 h transcript from silently changing field count mid-file.
fn stamp(t: f64, sep: char) -> String {
    let t = if t.is_finite() && t > 0.0 { t } else { 0.0 };
    // round to ms first: formatting 59.9996 s as {:06.3} would print "60.000"
    // inside the minute and produce 00:00:60.000, which players reject
    let ms_total = (t * 1000.0).round() as u64;
    let (h, m, s, ms) = (
        ms_total / 3_600_000,
        ms_total / 60_000 % 60,
        ms_total / 1000 % 60,
        ms_total % 1000,
    );
    format!("{h:02}:{m:02}:{s:02}{sep}{ms:03}")
}

/// Cue text as one paragraph: a BLANK line inside a cue terminates it in both
/// formats, so an embedded one would silently split the file's structure.
fn body(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// WebVTT cue payload is parsed as markup, so these three are mandatory.
fn vtt_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub fn srt(cues: &[Cue]) -> String {
    let mut out = String::new();
    // counted on EMITTED cues, not on the input index: a skipped blank one
    // would otherwise leave a gap, and a file numbered 1, 3 is malformed
    let mut n = 0usize;
    for c in cues {
        let line = body(c.text);
        if line.is_empty() {
            continue;
        }
        n += 1;
        out.push_str(&format!(
            "{n}\n{} --> {}\n{line}\n\n",
            stamp(c.start, ','),
            stamp(c.end, ','),
        ));
    }
    out
}

pub fn vtt(cues: &[Cue]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for c in cues {
        let line = body(c.text);
        if line.is_empty() {
            continue;
        }
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            stamp(c.start, '.'),
            stamp(c.end, '.'),
            vtt_escape(&line)
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cues() -> Vec<Cue<'static>> {
        vec![
            Cue {
                start: 0.0,
                end: 2.48,
                text: "Hej och välkommen",
            },
            Cue {
                start: 2.48,
                end: 3661.5,
                text: "till sändningen",
            },
        ]
    }

    #[test]
    fn srt_uses_comma_and_numbers_from_one() {
        let s = srt(&cues());
        assert_eq!(
            s,
            "1\n00:00:00,000 --> 00:00:02,480\nHej och välkommen\n\n\
             2\n00:00:02,480 --> 01:01:01,500\ntill sändningen\n\n"
        );
    }

    #[test]
    fn vtt_uses_a_dot_and_a_header() {
        let s = vtt(&cues());
        assert!(s.starts_with("WEBVTT\n\n"), "missing header: {s:?}");
        assert!(s.contains("00:00:00.000 --> 00:00:02.480"), "{s}");
        // the hour rolls over correctly rather than printing 61 minutes
        assert!(s.contains("01:01:01.500"), "{s}");
        assert!(
            !s.contains(','),
            "a comma separator would be an SRT stamp: {s}"
        );
    }

    #[test]
    fn a_stamp_never_rolls_a_field_past_its_range() {
        // 59.9996 s formatted naively as {:06.3} prints "60.000" inside the
        // minute - the file is then rejected whole by most players
        assert_eq!(stamp(59.9996, ','), "00:01:00,000");
        assert_eq!(stamp(0.0004, ','), "00:00:00,000");
        // and a negative or non-finite time clamps rather than printing junk
        assert_eq!(stamp(-1.0, '.'), "00:00:00.000");
        assert_eq!(stamp(f64::NAN, '.'), "00:00:00.000");
    }

    #[test]
    fn vtt_escapes_markup_and_srt_does_not() {
        let c = [Cue {
            start: 0.0,
            end: 1.0,
            text: "a < b & c > d",
        }];
        assert!(vtt(&c).contains("a &lt; b &amp; c &gt; d"));
        // SRT is not markup - escaping there would corrupt the caption
        assert!(srt(&c).contains("a < b & c > d"));
    }

    #[test]
    fn an_embedded_blank_line_cannot_split_a_cue() {
        // a blank line terminates a cue in both formats, so one arriving in
        // the text would silently restructure the file
        let c = [Cue {
            start: 0.0,
            end: 1.0,
            text: "first\n\nsecond",
        }];
        let s = srt(&c);
        assert!(s.contains("first second"), "{s}");
        // exactly one cue: index 1 present, index 2 absent
        assert!(s.starts_with("1\n") && !s.contains("\n2\n"), "{s}");
    }

    #[test]
    fn empty_cues_are_skipped_and_numbering_stays_dense() {
        let c = [
            Cue {
                start: 0.0,
                end: 1.0,
                text: "one",
            },
            Cue {
                start: 1.0,
                end: 2.0,
                text: "   ",
            },
            Cue {
                start: 2.0,
                end: 3.0,
                text: "two",
            },
        ];
        let s = srt(&c);
        // a subtitle numbered 1, 3 is malformed; the blank one must not
        // consume an index
        assert!(s.contains("1\n00:00:00,000"), "{s}");
        assert!(s.contains("2\n00:00:02,000"), "{s}");
        assert!(!s.contains("3\n"), "{s}");
        assert_eq!(vtt(&c).matches("-->").count(), 2);
    }

    #[test]
    fn no_cues_still_produces_a_valid_file() {
        assert_eq!(srt(&[]), "");
        // a VTT without its header is not a VTT, even when empty
        assert_eq!(vtt(&[]), "WEBVTT\n\n");
    }
}
