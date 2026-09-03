//! The builtin `current_time` tool: a clock the agent loop answers itself,
//! instantly and without touching the network. Declared in a request's `tools`
//! as `{"type": "current_time", "timezone": "Europe/Stockholm"}` (paddock
//! extension, same family as `{"type": "forensics"}`).
//!
//! Why a tool and not prompt injection: an injected clock freezes SEND time -
//! stale by the end of a long generation, absurd in an hours-old tab - and a
//! minute-stamp in the system prompt re-tokenizes the prompt head every send,
//! voiding the conversation's radix prefix. The tool is correct at the moment
//! of use and costs the cache nothing. The Studio pairs it with a date-only
//! line in `instructions` (passive day-scale grounding); this covers the rest.
//!
//! The DECLARATION carries the user's IANA timezone because that is the one
//! fact the server lacks: its clock is NTP-correct about the instant, but the
//! box may sit in a UTC rack while the user is in Stockholm. The model may
//! still override per call ("what time is it in Tokyo?") via the `timezone`
//! argument; an unqualified call answers in the declared zone, and with no
//! declaration at all the server's own local zone is the honest default (a
//! direct API caller on their own box wants their box's time).

use chrono::{Datelike, Local, Offset, TimeZone, Timelike, Utc};
pub use chrono_tz;
use chrono_tz::Tz;
use serde_json::{Value, json};

pub const TOOL_NAME: &str = "get_current_time";

const TOOL_DESC: &str = "Get the current date and time, rendered in the user's timezone \
    (or a requested one). Use it whenever the answer depends on the time of day right now \
    - the current time is NOT in your context and must be fetched, never guessed.";

/// The gathered spec: the declaration's zone, validated at request time.
/// `None` = no zone declared; execution falls back to the server's local zone.
#[derive(Clone, Copy, Debug)]
pub struct ClockSpec {
    pub tz: Option<Tz>,
}

/// Parse the tool declaration's optional `timezone`. A present-but-invalid
/// zone is a clear 400 at request time, never a silent fallback the user
/// discovers as wrong clock answers.
pub fn parse_spec(t: &Value) -> Result<ClockSpec, String> {
    let tz = match t
        .get("timezone")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(s) => Some(s.parse::<Tz>().map_err(|_| {
            format!(
                "current_time: unknown IANA timezone {s:?} (expected e.g. \"Europe/Stockholm\")"
            )
        })?),
        None => None,
    };
    Ok(ClockSpec { tz })
}

/// The tool schema disclosed to the model (chat/responses nested-function
/// shape, same as web_search and forensics).
pub fn tool_def() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": TOOL_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone to render in (e.g. \"Asia/Tokyo\"). \
                                        Omit for the user's own timezone."
                    }
                },
                "additionalProperties": false
            }
        }
    })
}

/// The flat OpenAI Responses-wire def ({type, name, parameters}) - what the
/// manager's cloud loop declares to a Responses provider, exactly as it
/// flattens the tool_search pair.
pub fn responses_tool_def() -> Value {
    let f = tool_def();
    let func = &f["function"];
    json!({
        "type": "function",
        "name": func["name"],
        "description": func["description"],
        "parameters": func["parameters"],
    })
}

/// The Anthropic `/v1/messages` def (`{name, description, input_schema}`) -
/// same identity and schema, only the envelope differs, exactly as forensics.
pub fn anthropic_tool_def() -> Value {
    let f = tool_def();
    let func = &f["function"];
    json!({
        "name": func["name"],
        "description": func["description"],
        "input_schema": func["parameters"],
    })
}

/// Execute a call. Returns the forensics-shaped quad
/// `(model_content, output, error, status)` so both agent loops consume it
/// through the same seams. Zone precedence: the call's own `timezone`
/// argument -> the declaration's -> the server's local zone.
pub fn run(
    spec: ClockSpec,
    arguments: &str,
) -> (String, Option<String>, Option<String>, &'static str) {
    let args: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
    let tz = match args
        .get("timezone")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(s) => match s.parse::<Tz>() {
            Ok(z) => Some(z),
            Err(_) => {
                let m = format!(
                    "unknown IANA timezone {s:?} - use e.g. \"Europe/Stockholm\", or omit for the user's timezone"
                );
                return (m.clone(), None, Some(m), "failed");
            }
        },
        None => spec.tz,
    };
    let now = Utc::now();
    let body = match tz {
        Some(z) => render(now.with_timezone(&z), z.name()),
        // no declared zone: the server's local clock, named by offset since
        // chrono::Local has no IANA identity to report
        None => {
            let local = now.with_timezone(&Local);
            let off = local.offset().fix().to_string();
            render(local, &format!("UTC{off}"))
        }
    };
    let s = body.to_string();
    (s.clone(), Some(s), None, "completed")
}

/// One result shape for every zone: unambiguous ISO instant plus the pieces a
/// model actually reasons with (weekday, date, wall-clock, offset).
fn render<T: TimeZone>(t: chrono::DateTime<T>, zone: &str) -> Value
where
    T::Offset: std::fmt::Display,
{
    json!({
        "iso": t.to_rfc3339_opts(chrono::SecondsFormat::Secs, false),
        "timezone": zone,
        "weekday": t.weekday().to_string(),
        "date": format!("{:04}-{:02}-{:02}", t.year(), t.month(), t.day()),
        "time": format!("{:02}:{:02}", t.hour(), t.minute()),
        "utc_offset": t.offset().fix().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn def_shapes_agree() {
        let f = tool_def();
        assert_eq!(f["function"]["name"], TOOL_NAME);
        let a = anthropic_tool_def();
        assert_eq!(a["name"], TOOL_NAME);
        assert_eq!(a["input_schema"], f["function"]["parameters"]);
    }

    #[test]
    fn spec_validates_the_zone_at_request_time() {
        let ok = parse_spec(&json!({"type": "current_time", "timezone": "Europe/Stockholm"}))
            .expect("valid zone");
        assert_eq!(ok.tz, Some(chrono_tz::Europe::Stockholm));
        assert!(
            parse_spec(&json!({"type": "current_time"}))
                .expect("optional")
                .tz
                .is_none()
        );
        let err = parse_spec(&json!({"type": "current_time", "timezone": "Mars/Olympus"}))
            .expect_err("junk zone is a 400");
        assert!(err.contains("Mars/Olympus"));
    }

    #[test]
    fn runs_in_the_declared_zone() {
        let spec = ClockSpec {
            tz: Some(chrono_tz::Europe::Stockholm),
        };
        let (content, output, error, status) = run(spec, "{}");
        assert_eq!(status, "completed");
        assert!(error.is_none());
        let v: Value = serde_json::from_str(&content).expect("json content");
        assert_eq!(v["timezone"], "Europe/Stockholm");
        // Stockholm is UTC+1 or UTC+2, never UTC
        let off = v["utc_offset"].as_str().expect("offset");
        assert!(off == "+01:00" || off == "+02:00", "offset was {off}");
        assert_eq!(output.as_deref(), Some(content.as_str()));
    }

    #[test]
    fn call_argument_overrides_the_declaration() {
        let spec = ClockSpec {
            tz: Some(chrono_tz::Europe::Stockholm),
        };
        let (content, _, _, status) = run(spec, r#"{"timezone": "Asia/Tokyo"}"#);
        assert_eq!(status, "completed");
        let v: Value = serde_json::from_str(&content).expect("json");
        assert_eq!(v["timezone"], "Asia/Tokyo");
        assert_eq!(v["utc_offset"], "+09:00");
    }

    #[test]
    fn empty_timezone_argument_reads_as_omitted() {
        // observed live: qwen3.5 sent {"timezone":""} on its first call - an
        // empty string is "no preference", never an unknown-zone error
        let spec = ClockSpec {
            tz: Some(chrono_tz::Europe::Stockholm),
        };
        let (content, _, error, status) = run(spec, r#"{"timezone": ""}"#);
        assert_eq!(status, "completed");
        assert!(error.is_none());
        let v: Value = serde_json::from_str(&content).expect("json");
        assert_eq!(v["timezone"], "Europe/Stockholm");
    }

    #[test]
    fn bad_call_argument_fails_the_call_not_the_turn() {
        let (content, output, error, status) =
            run(ClockSpec { tz: None }, r#"{"timezone": "nope"}"#);
        assert_eq!(status, "failed");
        assert!(output.is_none());
        assert!(error.is_some());
        assert!(content.contains("nope"));
    }
}
