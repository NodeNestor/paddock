//! The agent loop's budget - what stops a loop that has stopped working.
//!
//! Making a bad tool call cheap did nothing about the loop spending eight
//! rounds discovering that it was bad. Observed live on gpt-5.6: four dead
//! `mcp_call_tool` attempts, one round of 4096 tokens
//! and 75 s of GPU that produced nothing at all, and a turn that recovered
//! only because the model eventually found the tool by itself.
//!
//! Three levers, and they compose into one story - a loop repeating itself is
//! told so, a round cannot outspend the turn, and a turn that runs out of
//! budget still ANSWERS instead of handing back the empty tail of a tool
//! round:
//!
//! 1. [`CallLedger`] - one record per turn of what has already run. A second
//!    identical call is answered from it; a third is refused.
//! 2. [`turn_output_cap`] and [`round_cap`] - the tool half of a turn is
//!    bounded as a whole, and no single round may spend past what is left.
//! 3. [`Stop`] + [`ANSWER_ONLY_NUDGE`] - when either bound is reached, one
//!    more round runs with the tools taken away and an instruction to answer.
//!
//! The caller gets a bound of their own on top: the Responses API's
//! `max_tool_calls` ([`CallLedger::with_limit`]) counts dispatched calls and
//! ends the tool half at the caller's number. It is the same machinery - the
//! ledger already sees every call, and reaching the limit is just a third
//! [`Stop`] into the same answer round, which is exactly the spec's own
//! reading of it ("further attempts to call a tool will be ignored").
//!
//! The shapes follow where the field landed. Vercel's AI SDK, when a
//! `stopWhen` condition fires while the model is still calling tools, makes
//! one more turn with `toolChoice:'none'` "so the run ends with a
//! natural-language answer instead of a half-finished tool call"; OpenRouter's
//! `stop_server_tools_when` says the same thing in its own words ("the model
//! is asked to produce its final answer with the context gathered so far");
//! and runtimes that fight stuck agents converge on a per-turn signature set
//! over (tool, canonical arguments) with a small repeat threshold. Ours
//! differs deliberately in one place: a repeat is ANSWERED from the ledger
//! before it is ever refused, because the second identical call is usually a
//! model that lost the result, not one that is stuck - and handing the result
//! back costs nothing and is very often what it actually wanted.
//!
//! Everything here is a decision plus the words for it. No dialect, no I/O:
//! the four runner loops (Responses × stream/non-stream, Anthropic ×
//! stream/non-stream) and both manager cloud lanes share this file, because
//! the unwrapping that unified had already drifted across three copies
//! and this is the same shape of thing.

use std::collections::HashMap;

use serde_json::Value;

/// Rounds a loop may spend calling tools before the answer round, when the
/// caller names no budget of their own. See [`rounds_cap`] for the one they can.
///
/// Was 8 - the number all four loops happened to have independently, never
/// measured. Raised to 16 for two reasons:
///
/// 1. **The field sits higher.** Our "round" is the same unit as the OpenAI
///    Agents SDK's `max_turns` (one model call; tool execution happens inside
///    it) and LangChain's `max_iterations`, which defaults to 15. LangGraph
///    ships a 25-super-step recursion limit. 8 was the outlier.
/// 2. **Progressive disclosure taxes the budget, permanently.** A server past
///    [`SEARCH_DISCLOSURE_THRESHOLD`](crate::tool_search::SEARCH_DISCLOSURE_THRESHOLD)
///    is hidden whole and stays hidden - `disclose_servers` skips any server
///    that alone exceeds the threshold, so a 40-tool connector is behind
///    `mcp_search_tools` on every request by design. Measured on one live turn
///    (gpt-5.6 + a 40-tool connector): 19 calls over 8 rounds, 4 of them searches
///    and 2 were a failed call plus its retry - a third of the budget spent
///    before any answer. That overhead is structural, not a bug awaiting a fix,
///    so the budget has to be sized with it in it.
///
/// Raising this is cheap in the direction that matters: exhausting the budget
/// degrades an answer, it never errors ([`Stop`] always yields an answer
/// round), while the token bound below is what actually protects spend.
pub const MAX_ROUNDS: usize = 16;

/// The most rounds any request may take, whatever it asks for - the runaway
/// backstop that [`rounds_cap`] clamps to. Deliberately far above any real
/// agentic turn: it exists to bound a bug, not to shape a workload.
pub const HARD_MAX_ROUNDS: usize = 64;

/// The round ceiling for one request, given the caller's `max_tool_calls`.
///
/// Their number governs in both directions, which is the fix for a real
/// conformance gap: `max_tool_calls` could previously only lower the budget,
/// so a caller who asked for 100 tool calls was still cut at 8 rounds by a
/// bound we had never told them about. Reaching a limit is not an error in the
/// spec's reading ("further attempts to call a tool will be ignored"), so the
/// number they set has to be the one that decides.
///
/// Rounds and calls are different units, and there is no need to convert
/// between them: a tool round emits at least one call, so a round budget above
/// the call budget is unreachable anyway. Handing rounds the caller's own
/// number is therefore exact rather than a heuristic - the ledger stops the
/// turn on calls, and this stops it on rounds only if refusals and replays
/// (which dispatch nothing, and so spend no call) somehow got there first.
pub fn rounds_cap(max_tool_calls: Option<usize>) -> usize {
    match max_tool_calls {
        Some(n) => n.min(HARD_MAX_ROUNDS),
        None => MAX_ROUNDS,
    }
}

/// What the TOOL half of a turn may generate across all its rounds.
///
/// Each round re-spends the request's own cap, so without this a tool turn's
/// total is unbounded in rounds: eight rounds of a 32k request is a quarter of
/// a million tokens. Four times the request's cap, floored so a small cap
/// cannot strangle a legitimate multi-tool turn - the manager's cloud loop has
/// had exactly this since it was written, and the runner had no equivalent at
/// all. The answer round is deliberately outside it (see [`round_cap`]).
pub fn turn_output_cap(request_cap: usize) -> usize {
    request_cap.saturating_mul(4).max(16_384)
}

/// Ceiling on one tool round: the request's own cap, or whatever is left of
/// the turn's tool budget, whichever is smaller.
///
/// This is the per-round lever, and its shape is deliberate. The obvious
/// version - "a tool round is emitting a call, not an essay, so give it a
/// tighter fixed cap" - is wrong here, and would break shipped behaviour two
/// ways: the answer can arrive in any round (the usual agentic shape is call,
/// result, answer), and a tool round's own output can legitimately be huge,
/// because `artifact_create`'s page is the arguments of a tool call. A fixed
/// squeeze truncates both, and a truncated call is the failure the Studio
/// already has an apology string for.
///
/// Deriving the ceiling from what the turn has left cuts a round only when the
/// budget is genuinely gone, which is a thing worth saying out loud rather
/// than a guess about what the round was going to do. It also makes the turn
/// cap exact: checking the total only between rounds lets one round overshoot
/// it by its whole cap.
///
/// The answer round does not take this - it gets the request's cap in full, on
/// the far side of the budget, because a turn that has run out of tool budget
/// still owes the user a complete answer.
///
/// Never returns 0, and never a number too small to be a request: the last
/// sliver of a budget would otherwise become `max_tokens: 3`, which providers
/// reject outright (OpenAI's floor is 16) and which cannot produce anything
/// anyway. The turn can therefore overshoot its budget by up to
/// [`MIN_ROUND_TOKENS`], which is the right way round.
pub fn round_cap(request_cap: usize, spent: usize, turn_cap: usize) -> usize {
    request_cap.min(turn_cap.saturating_sub(spent).max(MIN_ROUND_TOKENS))
}

/// The smallest generation worth asking for - see [`round_cap`].
pub const MIN_ROUND_TOKENS: usize = 256;

/// Why the tool half of a turn ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// This request's round budget went by, carrying the number that applied -
    /// [`rounds_cap`], not necessarily [`MAX_ROUNDS`]. It has to be the real
    /// one: a caller who set `max_tool_calls` and then read "16 rounds" in the
    /// transcript would be looking for a knob that was not the one that bit.
    Rounds(usize),
    /// The turn's tool budget ([`turn_output_cap`]) is spent - either counted
    /// between rounds, or hit inside one and cut there by [`round_cap`].
    Output,
    /// The CALLER's own ceiling: the Responses API's `max_tool_calls`, carrying
    /// the number they asked for. Ours are the two above; this one is theirs,
    /// and the notice has to say so or the reader blames the server.
    ToolCalls(usize),
}

impl Stop {
    /// The line the user sees, ahead of the answer the model then gives.
    ///
    /// Loud deliberately: a turn that quietly stops calling tools and answers
    /// anyway is the no-silent-failures principle broken, and "it answered
    /// without finishing its tool work" is exactly what a reader has to be
    /// able to see in the transcript.
    pub fn notice(self) -> String {
        match self {
            Stop::Rounds(n) => format!(
                "[tool budget spent: {n} rounds of tool calls - answering with what has been gathered]"
            ),
            Stop::Output => {
                "[tool budget spent: this turn's output budget - answering with what has been gathered]"
                    .to_owned()
            }
            // Nothing was spent and nothing was gathered when the limit is 0,
            // so that one gets its own words rather than a line that is false.
            Stop::ToolCalls(0) => {
                "[no tool may run: this request set max_tool_calls: 0 - answering directly]"
                    .to_owned()
            }
            Stop::ToolCalls(n) => format!(
                "[tool budget spent: this request's max_tool_calls limit of {n} - answering with what has been gathered]"
            ),
        }
    }
}

/// The turn the loop appends before the tools-off round.
///
/// It has to say all three things or the round is wasted: no tool will run,
/// none are listed any more (so "I'll call X" is not an option), and a
/// half-answer naming what is missing beats a promise to go and get it.
pub const ANSWER_ONLY_NUDGE: &str = "The tool budget for this turn is spent: no tool will run again on this turn, and the tools \
     are no longer listed. Answer now from what the results above already give you. Where \
     something could not be established, say so plainly - do not describe the call you would \
     have made.";

/// What to do with a call the model just emitted.
pub enum Verdict {
    /// Not seen this turn, or a failure that has earned one retry: run it,
    /// then hand the outcome to [`CallLedger::record`].
    Fresh,
    /// Seen, and it succeeded: this text goes back as the result. Nothing runs.
    Replay(String),
    /// Seen enough. The message says why nothing ran, addressed to the model.
    Refuse(String),
}

/// One call's identity within a turn: the RESOLVED tool name (post-`mcp_call_tool`
/// unwrap, so the envelope and the direct call are one thing) plus its
/// arguments in canonical form.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Signature(String);

/// What one signature has done so far this turn.
struct Entry {
    /// How many times the model has EMITTED it (not how many times it ran).
    emitted: usize,
    /// `None` while a call is out; `Some((ok, output))` once it comes back.
    outcome: Option<(bool, String)>,
}

/// Every call this turn, keyed by signature. One per turn, not per round: a
/// loop that repeats itself does it across rounds.
///
/// It also counts, because it is the one place that sees every call: the
/// Responses API's `max_tool_calls` is exactly a ceiling on that count, and
/// [`limit_reached`](Self::limit_reached) turns it into the same [`Stop`] the
/// other two bounds produce.
#[derive(Default)]
pub struct CallLedger {
    seen: HashMap<Signature, Entry>,
    /// Calls actually DISPATCHED this turn. A replay and a refusal both run
    /// nothing, so neither counts - which is also the spec's reading ("calls
    /// to built-in tools that can be **processed**").
    ran: usize,
    /// The caller's `max_tool_calls`; `None` = only our own bounds apply.
    limit: Option<usize>,
    /// Catalog searches this turn - see [`Self::search_budget_spent`].
    searches: usize,
    /// Calls refused before dispatch, and how often - see [`Self::note_refused`].
    /// Kept apart from `seen`, which is about calls that RAN.
    refused: HashMap<Signature, usize>,
}

/// Catalog searches one turn may run before the loop stops answering them.
///
/// Discovery is not free: every `mcp_search_tools` costs a whole round, and a
/// round is the scarcest thing a tool turn has. The number is small deliberately,
/// because our search already
/// returns `all_tool_names`, the COMPLETE catalog index, with every single
/// result. After one search the model holds every name there is; a second is a
/// reasonable "now fetch that one's schema by exact name", and a third is a
/// model re-asking a question it is already holding the answer to.
///
/// Observed live (gpt-5.6): four searches in one turn, two of them after
/// fourteen successful direct calls - pure re-asking.
pub const SEARCH_BUDGET: usize = 3;

impl CallLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// A ledger bounded by the caller's own `max_tool_calls`.
    ///
    /// OpenAI: "the maximum number of total calls to built-in tools that can be
    /// processed in a response... any further attempts to call a tool by the
    /// model will be ignored" - so reaching it is not an error, it is the end
    /// of the tool half of the turn, and the answer round follows. `Some(0)` is
    /// a legitimate ask (tools declared, none may run) and behaves as one: the
    /// turn goes straight to the answer round.
    pub fn with_limit(limit: Option<usize>) -> Self {
        Self {
            limit,
            ..Self::default()
        }
    }

    /// Calls dispatched so far this turn.
    pub fn ran(&self) -> usize {
        self.ran
    }

    /// Count one catalog search, and say when the turn has had enough of them.
    ///
    /// `None` = run it. `Some(message)` = do not; the message goes back as the
    /// search's result and is written for the model. It does not say "no" and
    /// stop there - a bare refusal would leave a model that is genuinely lost
    /// with nothing - it says the thing that is actually true: the complete
    /// index has been in every result already, so the next move is to pick a
    /// name from it, not to ask again.
    pub fn search_budget_spent(&mut self) -> Option<String> {
        self.searches += 1;
        if self.searches <= SEARCH_BUDGET {
            return None;
        }
        Some(format!(
            "No search was run: this turn has already searched the tool catalog {} times, and \
             every one of those results carried `all_tool_names` - the COMPLETE list of tools \
             available to you. Searching again cannot return a name that list does not already \
             have. Scroll back to it, pick the exact name you need, and call it with \
             mcp_call_tool. If nothing in it fits, say so plainly instead of searching again.",
            SEARCH_BUDGET
        ))
    }

    /// Catalog searches run so far this turn.
    pub fn searches(&self) -> usize {
        self.searches
    }

    /// `Some` once the caller's `max_tool_calls` is used up - the loops set
    /// this as their [`Stop`] at the top of a round, which is what makes
    /// `max_tool_calls: 0` stop the turn before it calls anything.
    pub fn limit_reached(&self) -> Option<Stop> {
        match self.limit {
            Some(n) if self.ran >= n => Some(Stop::ToolCalls(n)),
            _ => None,
        }
    }

    /// Decide what happens to a call, and reserve its signature.
    ///
    /// The rules, in the order they bite:
    /// * first emission - run it;
    /// * emitted again while the first is still out (the model wrote the same
    ///   call twice in one round) - refuse the duplicate, the original is
    ///   already running and its result lands alongside;
    /// * second emission of one that SUCCEEDED - replay the result;
    /// * second emission of one that FAILED - run it again. Transient errors
    ///   and polls are real, and a tool that failed has not been "used" yet;
    /// * third emission, either way - refuse, and say which case it is.
    ///
    /// On [`Verdict::Fresh`] the caller must follow up with [`Self::record`],
    /// or the signature stays reserved and later repeats read as same-round
    /// duplicates.
    ///
    /// The caller's `max_tool_calls` is checked first and leaves no trace: a
    /// call that is never going to run should not colour what the ledger says
    /// about a later one either.
    pub fn check(&mut self, name: &str, arguments: &str) -> (Signature, Verdict) {
        let sig = Signature(format!("{name}\u{1}{}", canonical(arguments)));
        if let Some(limit) = self.limit
            && self.ran >= limit
        {
            let spent = match limit {
                0 => "no tool call is allowed on this turn".to_owned(),
                1 => "1 tool call has already run".to_owned(),
                n => format!("{n} tool calls have already run"),
            };
            return (
                sig,
                Verdict::Refuse(format!(
                    "{name} was NOT called - this request set max_tool_calls: {limit}, and \
                     {spent}. No further tool will run: answer from the results above."
                )),
            );
        }
        let e = self.seen.entry(sig.clone()).or_insert(Entry {
            emitted: 0,
            outcome: None,
        });
        e.emitted += 1;
        let verdict = match (e.emitted, &e.outcome) {
            (1, _) => Verdict::Fresh,
            (_, None) => Verdict::Refuse(format!(
                "{name} was NOT called - you emitted this identical call twice in the same round. \
                 The first one is running; its result arrives next to this message. Use that."
            )),
            (2, Some((true, out))) => Verdict::Replay(format!(
                "(you already called {name} with these exact arguments in this turn - this is \
                 that result again, nothing ran a second time)\n{out}"
            )),
            (2, Some((false, _))) => {
                // One retry, and it is a genuine re-run: clear the outcome so
                // the entry is "out" again.
                e.outcome = None;
                Verdict::Fresh
            }
            (_, Some((true, _))) => Verdict::Refuse(format!(
                "{name} was NOT called - you have already run it twice this turn with these exact \
                 arguments and its result is above. Use that result, or do something different."
            )),
            (_, Some((false, err))) => Verdict::Refuse(format!(
                "{name} was NOT called - it has already failed twice this turn with these exact \
                 arguments: {err}. Change the arguments or take a different approach; the same \
                 call will not be run a third time."
            )),
        };
        if matches!(verdict, Verdict::Fresh) {
            self.ran += 1;
        }
        (sig, verdict)
    }

    /// Remember a call refused before dispatch - a schema check, not a tool
    /// failure - and escalate the wording when the same one comes back.
    ///
    /// This is not [`Self::check`], deliberately: nothing was processed, so a
    /// refusal must spend none of the caller's `max_tool_calls`. But identity
    /// and budget are different questions, and leaving refusals entirely
    /// untracked meant a model could resend a byte-identical impossible call
    /// forever, getting the same sentence back each time - which is exactly
    /// what happened: `tic__get_company_bankruptcies` was
    /// called twice with `{"companyId": ...}` against a schema that requires
    /// `initiatedDate` and has no `companyId` at all, and the second attempt
    /// read like the first because nothing remembered the first.
    ///
    /// Returns the message to actually send back.
    pub fn note_refused(&mut self, name: &str, arguments: &str, message: &str) -> String {
        let sig = Signature(format!("{name}\u{1}{}", canonical(arguments)));
        let n = self.refused.entry(sig).or_insert(0);
        *n += 1;
        match *n {
            1 => message.to_owned(),
            2 => format!(
                "{message}\n\nThis is the SAME call you just sent, refused for the same reason - \
                 sending it again unchanged will not work. Fix the field named above, or use a \
                 different tool."
            ),
            n => format!(
                "{name} was NOT called. You have now sent this identical call {n} times and it \
                 has been refused every time, for this reason: {message}\nStop resending it. \
                 Call it with different arguments, use a different tool, or say plainly what you \
                 could not find out."
            ),
        }
    }

    /// File what a [`Verdict::Fresh`] call actually returned.
    pub fn record(&mut self, sig: &Signature, ok: bool, output: &str) {
        if let Some(e) = self.seen.get_mut(sig) {
            e.outcome = Some((ok, output.to_owned()));
        }
    }
}

/// Arguments in canonical form: object keys sorted, insignificant whitespace
/// gone, so `{"a":1,"b":2}` and `{ "b": 2, "a": 1 }` are one call and not two.
///
/// Written out rather than leaning on `serde_json`'s map ordering, which is
/// only sorted while nothing anywhere in the build tree turns on
/// `preserve_order` - a feature-unification landmine, and this is a
/// correctness key, not a display string.
fn canonical(arguments: &str) -> String {
    match serde_json::from_str::<Value>(arguments) {
        Ok(v) => {
            let mut s = String::new();
            write_canonical(&v, &mut s);
            s
        }
        // Not JSON at all: the raw text is its own identity. Two identical
        // malformed calls are still the same call.
        Err(_) => arguments.trim().to_owned(),
    }
}

fn write_canonical(v: &Value, out: &mut String) {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String((*k).to_string()).to_string());
                out.push(':');
                write_canonical(&map[*k], out);
            }
            out.push('}');
        }
        // Array order is meaning, not formatting - never sorted.
        Value::Array(items) => {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(it, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(v: &Verdict) -> &str {
        match v {
            Verdict::Fresh => "fresh",
            Verdict::Replay(s) | Verdict::Refuse(s) => s,
        }
    }

    #[test]
    fn the_same_call_is_answered_once_then_replayed_then_refused() {
        let mut l = CallLedger::new();
        let (sig, v) = l.check("artifact_read", r#"{"artifact_id":"a1"}"#);
        assert!(matches!(v, Verdict::Fresh));
        l.record(&sig, true, "the page");

        // Second: the result comes back for free, marked as a replay so the
        // model knows nothing re-ran.
        let (_, v) = l.check("artifact_read", r#"{"artifact_id":"a1"}"#);
        let Verdict::Replay(out) = &v else {
            panic!("expected a replay: {}", text(&v))
        };
        assert!(
            out.ends_with("the page"),
            "the original result must survive verbatim: {out}"
        );
        assert!(out.contains("nothing ran a second time"), "{out}");

        // Third: it is looping.
        let (_, v) = l.check("artifact_read", r#"{"artifact_id":"a1"}"#);
        let Verdict::Refuse(m) = &v else {
            panic!("expected a refusal: {}", text(&v))
        };
        assert!(m.contains("already run it twice"), "{m}");
    }

    /// The carve-out the task called for: a poll or a retry after a transient
    /// error is legitimate, so a FAILED call is not cached - it is re-run.
    #[test]
    fn a_failure_earns_one_retry_and_no_more() {
        let mut l = CallLedger::new();
        let (sig, v) = l.check("tic__search", r#"{"q":"acme"}"#);
        assert!(matches!(v, Verdict::Fresh));
        l.record(&sig, false, "upstream timed out");

        let (sig, v) = l.check("tic__search", r#"{"q":"acme"}"#);
        assert!(
            matches!(v, Verdict::Fresh),
            "a failed call must be retryable: {}",
            text(&v)
        );
        l.record(&sig, false, "upstream timed out");

        let (_, v) = l.check("tic__search", r#"{"q":"acme"}"#);
        let Verdict::Refuse(m) = &v else {
            panic!("expected a refusal: {}", text(&v))
        };
        assert!(m.contains("failed twice"), "{m}");
        assert!(
            m.contains("upstream timed out"),
            "the error must be quoted back: {m}"
        );
    }

    /// A retry that SUCCEEDS is a normal success from then on - the next
    /// repeat replays rather than refusing.
    #[test]
    fn a_retry_that_works_makes_the_call_replayable() {
        let mut l = CallLedger::new();
        let (sig, _) = l.check("t", "{}");
        l.record(&sig, false, "boom");
        let (sig, v) = l.check("t", "{}");
        assert!(matches!(v, Verdict::Fresh));
        l.record(&sig, true, "fine");
        let (_, v) = l.check("t", "{}");
        assert!(
            matches!(v, Verdict::Refuse(_)),
            "third emission is a loop either way"
        );
    }

    #[test]
    fn key_order_and_whitespace_do_not_make_a_new_call() {
        let mut l = CallLedger::new();
        let (sig, _) = l.check("t", r#"{"a":1,"b":[2,3]}"#);
        l.record(&sig, true, "done");
        let (_, v) = l.check("t", "{ \"b\" : [2, 3] ,\n  \"a\": 1 }");
        assert!(
            matches!(v, Verdict::Replay(_)),
            "reordered keys are the same call: {}",
            text(&v)
        );
    }

    #[test]
    fn array_order_is_meaning_and_stays_a_different_call() {
        let mut l = CallLedger::new();
        let (sig, _) = l.check("t", r#"{"ids":[1,2]}"#);
        l.record(&sig, true, "done");
        let (_, v) = l.check("t", r#"{"ids":[2,1]}"#);
        assert!(
            matches!(v, Verdict::Fresh),
            "a reordered array is a different request"
        );
    }

    #[test]
    fn a_different_tool_with_the_same_arguments_is_a_different_call() {
        let mut l = CallLedger::new();
        let (sig, _) = l.check("a", r#"{"x":1}"#);
        l.record(&sig, true, "done");
        let (_, v) = l.check("b", r#"{"x":1}"#);
        assert!(matches!(v, Verdict::Fresh));
    }

    /// Two identical calls in one round: the second cannot replay a result
    /// that does not exist yet, so it is refused and pointed at its twin.
    #[test]
    fn a_duplicate_inside_one_round_is_refused_not_replayed() {
        let mut l = CallLedger::new();
        let (_, v) = l.check("t", r#"{"x":1}"#);
        assert!(matches!(v, Verdict::Fresh));
        let (_, v) = l.check("t", r#"{"x":1}"#);
        let Verdict::Refuse(m) = &v else {
            panic!("expected a refusal: {}", text(&v))
        };
        assert!(m.contains("same round"), "{m}");
    }

    /// Malformed arguments are still an identity - a model repeating the same
    /// broken call is the loop this exists to catch.
    #[test]
    fn unparseable_arguments_still_have_an_identity() {
        let mut l = CallLedger::new();
        let (sig, v) = l.check("t", "{not json");
        assert!(matches!(v, Verdict::Fresh));
        l.record(&sig, false, "bad");
        let (_, v) = l.check("t", "  {not json  ");
        assert!(
            matches!(v, Verdict::Fresh),
            "a failure retries once, whatever the text"
        );
        let (_, v) = l.check("t", "{not json");
        assert!(matches!(v, Verdict::Refuse(_)));
    }

    /// A fresh turn's round may spend everything the caller asked for - the
    /// ceiling is the turn's REMAINDER, never a guess about what the round is
    /// for. (A fixed squeeze would truncate both a final answer and an
    /// `artifact_create` page, which is why it is not one.)
    #[test]
    fn a_round_is_only_capped_by_what_the_turn_has_left() {
        let turn = turn_output_cap(8_000); // 32_000
        assert_eq!(
            round_cap(8_000, 0, turn),
            8_000,
            "nothing spent, nothing withheld"
        );
        assert_eq!(
            round_cap(8_000, 24_000, turn),
            8_000,
            "still exactly enough"
        );
        assert_eq!(
            round_cap(8_000, 28_000, turn),
            4_000,
            "only the remainder is left"
        );
        // Spent - but never 0 or some unaskable sliver; a provider rejects
        // `max_tokens: 3` outright, and the between-round check has already
        // decided this is the last tool round either way.
        assert_eq!(round_cap(8_000, 32_000, turn), MIN_ROUND_TOKENS, "spent");
        assert_eq!(
            round_cap(8_000, 99_999, turn),
            MIN_ROUND_TOKENS,
            "and it does not wrap"
        );
        assert_eq!(
            round_cap(64, 99_999, turn),
            64,
            "...but a tiny request is still honoured"
        );
    }

    #[test]
    fn the_turn_cap_has_a_floor_so_a_tiny_request_can_still_use_tools() {
        // 4x, but never so small that a legitimate multi-round turn dies.
        assert_eq!(turn_output_cap(256), 16_384);
        assert_eq!(turn_output_cap(8_000), 32_000);
        // and it cannot overflow on an absurd request
        assert!(turn_output_cap(usize::MAX) > 0);
    }

    #[test]
    fn every_stop_says_what_happened_and_what_it_is_doing_about_it() {
        for s in [Stop::Rounds(MAX_ROUNDS), Stop::Output, Stop::ToolCalls(3)] {
            let n = s.notice();
            assert!(n.starts_with('[') && n.ends_with(']'), "{n}");
            assert!(n.contains("answering with what has been gathered"), "{n}");
        }
        // The round notice quotes the cap that actually applied, not the
        // default - a caller who raised it via max_tool_calls must not read a
        // number they never set and go looking for the wrong knob.
        assert!(Stop::Rounds(MAX_ROUNDS).notice().contains("16 rounds"));
        assert!(Stop::Rounds(40).notice().contains("40 rounds"));
        // The caller's limit must be named as theirs, with their number: a
        // reader who set it has to recognise their own knob in the transcript.
        let n = Stop::ToolCalls(3).notice();
        assert!(n.contains("max_tool_calls") && n.contains('3'), "{n}");
        // ...and a limit of 0 spent no budget and gathered nothing, so it says
        // something true instead of the same sentence.
        let z = Stop::ToolCalls(0).notice();
        assert!(z.starts_with('[') && z.ends_with(']'), "{z}");
        assert!(z.contains("max_tool_calls: 0"), "{z}");
        assert!(!z.contains("gathered"), "nothing was gathered: {z}");
    }

    /// The caller's `max_tool_calls` counts what ran, and stops the turn at
    /// their number rather than erroring - the spec's "further attempts to
    /// call a tool by the model will be ignored".
    #[test]
    fn the_callers_tool_call_limit_ends_the_tool_half() {
        let mut l = CallLedger::with_limit(Some(2));
        assert!(l.limit_reached().is_none(), "nothing has run yet");

        let (sig, v) = l.check("a", "{}");
        assert!(matches!(v, Verdict::Fresh));
        l.record(&sig, true, "one");
        assert!(l.limit_reached().is_none(), "one of two");

        let (sig, v) = l.check("b", "{}");
        assert!(matches!(v, Verdict::Fresh));
        l.record(&sig, true, "two");
        assert_eq!(l.ran(), 2);
        assert_eq!(l.limit_reached(), Some(Stop::ToolCalls(2)), "spent");

        // Anything the model emits after that is refused, by name, saying
        // whose limit it was and that the answer comes next.
        let (_, v) = l.check("c", "{}");
        let Verdict::Refuse(m) = &v else {
            panic!("expected a refusal: {}", text(&v))
        };
        assert!(m.contains("max_tool_calls: 2"), "{m}");
        assert!(m.contains("answer from the results above"), "{m}");
    }

    /// A replay and a refusal both run nothing, so neither may spend the
    /// caller's budget - the spec counts calls that are PROCESSED.
    #[test]
    fn only_dispatched_calls_spend_the_limit() {
        let mut l = CallLedger::with_limit(Some(2));
        let (sig, _) = l.check("a", "{}");
        l.record(&sig, true, "one");
        // repeat -> replayed from the ledger, nothing ran
        assert!(matches!(l.check("a", "{}").1, Verdict::Replay(_)));
        // and a third -> refused, still nothing ran
        assert!(matches!(l.check("a", "{}").1, Verdict::Refuse(_)));
        assert_eq!(l.ran(), 1, "one dispatch, whatever the model emitted");
        assert!(
            l.limit_reached().is_none(),
            "the caller still has one call left"
        );
        let (_, v) = l.check("b", "{}");
        assert!(
            matches!(v, Verdict::Fresh),
            "and it is spendable: {}",
            text(&v)
        );
    }

    /// A retry after a failure is a second real call, and it costs a second
    /// slot - otherwise a flapping tool could outspend any limit.
    #[test]
    fn a_retry_spends_a_second_slot() {
        let mut l = CallLedger::with_limit(Some(2));
        let (sig, _) = l.check("a", "{}");
        l.record(&sig, false, "boom");
        let (sig, v) = l.check("a", "{}");
        assert!(matches!(v, Verdict::Fresh));
        l.record(&sig, true, "fine");
        assert_eq!(l.ran(), 2);
        assert_eq!(l.limit_reached(), Some(Stop::ToolCalls(2)));
    }

    /// The live case, reproduced against a real server:
    /// `tic__get_company_bankruptcies` takes `initiatedDate` and
    /// has no `companyId` property, while its fourteen `get_company_*`
    /// siblings all take `companyId` - so the model sent `companyId`, twice.
    /// The refusal was correct both times and read identically both times,
    /// which is what makes a model retry instead of rethink.
    #[test]
    fn the_same_refused_call_does_not_get_the_same_sentence_twice() {
        let mut l = CallLedger::new();
        let why = "get_company_bankruptcies was NOT called: missing `initiatedDate`.";
        let args = r#"{"companyId":3129783}"#;

        let first = l.note_refused("tic__get_company_bankruptcies", args, why);
        assert_eq!(first, why, "the first refusal is just the reason");

        // key order must not disguise a repeat, same as the call ledger
        let second = l.note_refused(
            "tic__get_company_bankruptcies",
            r#"{ "companyId" : 3129783 }"#,
            why,
        );
        assert!(second.contains(why), "it still says what is wrong");
        assert!(
            second.contains("SAME call"),
            "...and that it is a repeat: {second}"
        );

        let third = l.note_refused("tic__get_company_bankruptcies", args, why);
        assert!(third.contains("3 times"), "{third}");
        assert!(third.contains("Stop resending"), "{third}");

        // ...and none of it spent the caller's budget, because nothing ran
        assert_eq!(l.ran(), 0);
        // a different call is untouched by any of that
        assert_eq!(l.note_refused("t", "{}", "nope"), "nope");
    }

    /// The caller's number decides the ROUND ceiling too, in both directions.
    /// Before this, `max_tool_calls` could only tighten: a caller who asked for
    /// 100 calls was still cut at 8 rounds by a bound we never told them about,
    /// which is not what "further attempts will be ignored" means.
    #[test]
    fn the_callers_limit_governs_rounds_as_well() {
        assert_eq!(rounds_cap(None), MAX_ROUNDS, "no ask, our default");
        assert_eq!(rounds_cap(Some(40)), 40, "a bigger ask RAISES the ceiling");
        assert_eq!(rounds_cap(Some(3)), 3, "...and a smaller one lowers it");
        // never past the runaway backstop, whatever is asked for
        assert_eq!(rounds_cap(Some(10_000)), HARD_MAX_ROUNDS);
        // 0 is a legitimate ask and stays 0: the loop takes the answer round
        // immediately, which is what `limit_reached` already decided.
        assert_eq!(rounds_cap(Some(0)), 0);
    }

    /// Discovery costs rounds, so it gets a budget of its own - and spending it
    /// hands back the index rather than just saying no, because a model that
    /// keeps searching is one that has not noticed it already has the answer.
    #[test]
    fn searching_has_its_own_budget_and_the_refusal_is_useful() {
        let mut l = CallLedger::new();
        for i in 1..=SEARCH_BUDGET {
            assert!(
                l.search_budget_spent().is_none(),
                "search {i} of {SEARCH_BUDGET} must run"
            );
        }
        assert_eq!(l.searches(), SEARCH_BUDGET);
        let m = l.search_budget_spent().expect("past the budget");
        assert!(
            m.contains("all_tool_names"),
            "it must point at what the model already has: {m}"
        );
        assert!(m.contains("mcp_call_tool"), "...and at the next move: {m}");
        // and it stays spent
        assert!(l.search_budget_spent().is_some());
    }

    /// `max_tool_calls: 0` is a legitimate ask - tools declared, none may run
    /// - and the loops read it off `limit_reached` before round 0.
    #[test]
    fn a_zero_limit_stops_the_turn_before_it_calls_anything() {
        let l = CallLedger::with_limit(Some(0));
        assert_eq!(l.limit_reached(), Some(Stop::ToolCalls(0)));
        // ...and no limit at all leaves only our own bounds
        assert!(CallLedger::with_limit(None).limit_reached().is_none());
    }
}
