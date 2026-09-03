//! Constrained decoding: token-level grammar masking for
//! `response_format: json_object / json_schema` and `tool_choice:
//! "required"` / named function. The engine drives the `TokenConstraint`
//! seam; everything tokenizer-aware lives here.
//!
//! Mechanics: a byte-level pushdown machine (JSON, schema-directed or bare;
//! or a dialect's tool-call grammar) wrapped in a free-until-trigger gate so
//! reasoning stays unconstrained - qwen thinking runs free until `</think>`,
//! gpt-oss until the `final`-channel `<|message|>`. A candidate token is
//! legal iff its bytes advance a CLONE of the machine; stop tokens are legal
//! only when the machine says the output may end (the engine enforces that
//! via `may_stop`).
//!
//! Schema subset (strict, OpenAI-style): object (all properties required,
//! emitted in `required` order), array (`items`), string (+`enum`), number,
//! integer, boolean, null. Unsupported keywords are a 400 at compile, never
//! silently ignored.
//!
//! Tool `parameters` compile through a LENIENT sibling of that subset
//! (`compile_tool_args`): properties in any order, only the `required` ones
//! forced, and anything the subset can't model degraded to free JSON instead
//! of refused. A tool schema is the model's brief, not the caller's output
//! contract, so refusing the whole request over one `anyOf` would throw away
//! a call we can still make.

use std::sync::Arc;

use paddock_engine::sampler::TokenConstraint;
use paddock_tokenizer::GgufTokenizer;
use serde_json::Value;

/// Per-id raw byte table + special-token flags, built once per served model.
pub struct VocabBytes {
    data: Vec<u8>,
    offs: Vec<u32>,
    special: Vec<bool>,
}

impl VocabBytes {
    pub fn build(tok: &GgufTokenizer) -> VocabBytes {
        let n = tok.vocab_size;
        let mut data = Vec::with_capacity(n * 4);
        let mut offs = Vec::with_capacity(n + 1);
        let mut special = Vec::with_capacity(n);
        offs.push(0u32);
        for id in 0..n as u32 {
            let full = tok.decode(&[id], false).unwrap_or_default();
            let visible = tok.decode(&[id], true).unwrap_or_default();
            // control/special tokens vanish when specials are skipped
            special.push(visible.is_empty() && !full.is_empty());
            data.extend_from_slice(full.as_bytes());
            offs.push(data.len() as u32);
        }
        VocabBytes {
            data,
            offs,
            special,
        }
    }

    fn bytes(&self, id: u32) -> &[u8] {
        let i = id as usize;
        if i + 1 >= self.offs.len() {
            return &[];
        }
        &self.data[self.offs[i] as usize..self.offs[i + 1] as usize]
    }

    fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(true)
    }
}

mod json;

pub use json::{CompiledSchema, JsonMachine};
use json::{MAX_LAX_PROPS, compile_lax_schema, compile_tool_args};

// ----------------------------------------------------- forced-tool grammar --

/// Which family's tool-call syntax a forced call has to be spelled in. This
/// grammar is the dialect parser's mirror image - whatever it emits, that
/// parser must read back as a `tool_calls` entry - so the two move together
/// and a family only appears here once its parser exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolSyntax {
    /// qwen3.5/3.6: `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n
    /// </parameter>\n...</function>\n</tool_call>`. Parameter values are free
    /// text ended by the `\n</parameter>\n` terminator (mirrors the parser),
    /// so nothing about a value's TYPE is enforced.
    QwenXml,
    /// Laguna (poolside), GLM-shaped and with no padding anywhere:
    /// `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`.
    ///
    /// The name rides bare after the opener, which is the one structural
    /// difference that matters: `<tool_call>get` is a prefix of
    /// `<tool_call>get_weather`, so each function gets its opener extended
    /// through the FOLLOWING tag to keep the candidate set prefix-free.
    ///
    /// Values are typed here, unlike qwen: the template writes
    /// `v | tojson if v is not string else v`, so a declared string goes
    /// verbatim and everything else is real JSON - and the parser's `coerce`
    /// mirrors exactly that. So string parameters get free text and the rest
    /// run through the schema machine.
    LagunaXml,
    /// Hermes-style JSON, which is IBM Granite 4.1's shape:
    /// `<tool_call>\n{"name": "NAME", "arguments": {...}}\n</tool_call>`.
    ///
    /// Arguments are real JSON here, so unlike the XML syntax the argument
    /// object runs through the schema machine - which makes a forced call on
    /// this dialect genuine structured output. That is the whole point on
    /// `/v1/messages`: the Anthropic API defines no `response_format`, so a
    /// forced tool call is the only schema-shaped-output mechanism it has.
    Json,
    /// Muse Glimmer's ATEM markup, addressed by message recipient:
    ///
    /// ```text
    ///  to=NAME<|message|><atem:function_calls>
    /// <atem:invoke name="NAME">
    /// <atem:parameter name="KEY">VALUE</atem:parameter>
    /// </atem:invoke>
    /// </atem:function_calls>
    /// ```
    ///
    /// Two things are structurally unlike the other three.
    ///
    /// **The grammar spells SPECIAL tokens.** `<|message|>` is part of the
    /// call's opener and `<|eom|>` closes a thought, so this is the family
    /// that needs `Dialect::grammar_specials` - the blanket "no control tokens
    /// inside constrained output" rule would deadlock it. llama.cpp reaches
    /// the same conclusion from the other direction with its
    /// `preserved_tokens` list.
    ///
    /// **The turn may open with reasoning.** `render_reasoning()` in this
    /// model's template is unconditional - it always thinks first - so a
    /// forced call that started at token 0 would forbid the model its own
    /// thought process. The machine therefore carries an analysis branch
    /// (` to=self<|message|>...<|eom|>`, repeatable) ahead of the call, which is
    /// exactly the `zero_or_more(start + analysis) + start + tool_calls`
    /// shape llama.cpp's PEG builds for `tool_choice: "required"`.
    ///
    /// Values are typed like laguna's: `render_atem` writes `v | tojson` for
    /// mappings and iterables and everything else bare, so a declared string
    /// is free text and the rest is real JSON - and `muse::parse`'s `coerce`
    /// reads exactly that back.
    AtemXml,
}

/// Compiled tool set for `tool_choice: "required"` / named function. One call
/// per request, then only stop tokens - a forced choice means "at least one
/// call", and one is the shape every dialect's parser reads back cleanly.
pub struct ToolSet {
    syntax: ToolSyntax,
    fns: Vec<ToolFn>,
}

struct ToolFn {
    name: String,
    /// (key, required) - the XML syntaxes' parameter list
    params: Vec<(String, bool)>,
    /// per-parameter value grammar, index-parallel to `params`. Laguna only;
    /// empty for the other two (qwen carries every value as free text, and the
    /// JSON syntax puts the whole object through `args`).
    values: Vec<LagValue>,
    /// grammar for the `arguments` object. JSON syntax only.
    args: Option<Arc<CompiledSchema>>,
}

/// How one laguna parameter's value is spelled. The template's test is
/// `v is not string` on the VALUE, so this mirrors the declared type:
/// a string is written bare, anything else is `tojson`'d.
#[derive(Clone)]
enum LagValue {
    /// declared `string` (or a schema we can't read): bare text to the
    /// terminator. Also the safe answer for union/unknown types - the template
    /// decides by runtime type there, so we cannot know whether it would quote,
    /// and `parse_laguna_call`'s `coerce` accepts either way.
    Text,
    /// declared `string` with an `enum`: still bare, but only these
    Enum(Vec<String>),
    /// any other declared type: the value is JSON, so the schema applies
    Json(Arc<CompiledSchema>),
}

impl ToolSet {
    /// `tools` = request tool definitions; `only` = named-function filter.
    pub fn compile(
        syntax: ToolSyntax,
        tools: &[Value],
        only: Option<&str>,
    ) -> Result<Arc<ToolSet>, String> {
        let mut fns = Vec::new();
        for t in tools {
            let f = t.get("function").unwrap_or(t);
            let Some(name) = f.get("name").and_then(Value::as_str) else {
                continue;
            };
            if let Some(o) = only
                && o != name
            {
                continue;
            }
            let required: Vec<&str> = f
                .get("parameters")
                .and_then(|p| p.get("required"))
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let params: Vec<(String, bool)> = f
                .get("parameters")
                .and_then(|p| p.get("properties"))
                .and_then(Value::as_object)
                .map(|props| {
                    props
                        .keys()
                        .map(|k| (k.clone(), required.contains(&k.as_str())))
                        .collect()
                })
                .unwrap_or_default();
            // the function NAME is a literal in both syntaxes: a grammar byte
            // there (or one JSON would have to escape) makes candidate
            // matching ambiguous, and no real tool name contains any of them
            if name
                .bytes()
                .any(|b| matches!(b, b'<' | b'>' | b'"' | b'\\') || b < 0x20)
            {
                return Err(format!("tool name {name:?} contains grammar bytes"));
            }
            // parameter KEYS are literals only in the XML syntaxes; the JSON
            // syntax gets them from the compiled schema, which degrades an
            // unspellable key to free-form args rather than refusing
            // Muse addresses a call by RECIPIENT, and `self`/`user` are that
            // grammar's own two reserved addresses - a tool wearing either name
            // would make its call indistinguishable from a thought or an
            // answer, both in the grammar (` to=self<|message|>` is a prefix of
            // the call opener) and on the way back through the parser. That
            // ambiguity is the model's, not ours, so it gets named rather than
            // silently resolved one way.
            if syntax == ToolSyntax::AtemXml && matches!(name, "self" | "user") {
                return Err(format!(
                    "tool name {name:?} collides with a muse-glimmer channel address \
                     (`self` is its reasoning channel, `user` its answer channel)"
                ));
            }
            // parameter KEYS are literals only in the XML syntaxes; the JSON
            // syntax gets them from the compiled schema, which degrades an
            // unspellable key to free-form args rather than refusing
            let mut values = Vec::new();
            let args = match syntax {
                ToolSyntax::QwenXml | ToolSyntax::LagunaXml | ToolSyntax::AtemXml => {
                    if params.len() > MAX_LAX_PROPS {
                        return Err(format!(
                            "function {name:?} has more than {MAX_LAX_PROPS} parameters"
                        ));
                    }
                    for (k, _) in &params {
                        if k.bytes().any(|b| matches!(b, b'<' | b'>' | b'\n' | b'"')) {
                            return Err(format!("tool parameter {k:?} contains grammar bytes"));
                        }
                    }
                    // laguna and muse both spell values the way their template
                    // writes them: declared string = bare text, anything else
                    // `tojson`'d. Same decision function, different terminator.
                    if matches!(syntax, ToolSyntax::LagunaXml | ToolSyntax::AtemXml) {
                        let props = f
                            .get("parameters")
                            .and_then(|p| p.get("properties"))
                            .and_then(Value::as_object);
                        values = params
                            .iter()
                            .map(|(k, _)| lag_value_grammar(props.and_then(|p| p.get(k))))
                            .collect();
                    }
                    None
                }
                ToolSyntax::Json => Some(compile_tool_args(f.get("parameters"))),
            };
            fns.push(ToolFn {
                name: name.to_owned(),
                params,
                values,
                args,
            });
        }
        if fns.is_empty() {
            return Err(match only {
                Some(o) => format!("tool_choice names unknown function {o:?}"),
                None => "tool_choice \"required\" needs at least one tool".into(),
            });
        }
        Ok(Arc::new(ToolSet { syntax, fns }))
    }
}

/// Decide how one laguna (or muse ATEM) parameter's value is spelled, from its
/// declared type - both templates split values the same way.
///
/// The template writes `v | tojson(...) if v is not string else v`, so the
/// split is exactly "declared string or not". Where the declared type is a
/// union (`["string","null"]`), missing, or something we can't read, the
/// template's answer depends on the runtime value and we cannot predict it -
/// free text accepts both spellings and `coerce` sorts it out on the way back.
fn lag_value_grammar(schema: Option<&Value>) -> LagValue {
    let Some(obj) = schema.and_then(Value::as_object) else {
        return LagValue::Text;
    };
    match obj.get("type").and_then(Value::as_str) {
        Some("string") => match obj.get("enum").and_then(Value::as_array) {
            None => LagValue::Text,
            Some(vs) => {
                let variants: Option<Vec<String>> =
                    vs.iter().map(|v| v.as_str().map(str::to_owned)).collect();
                match variants {
                    // a variant carrying `<` could collide with the terminator
                    // it is glued to, so that set falls back to free text
                    Some(v) if !v.is_empty() && !v.iter().any(|s| s.bytes().any(|b| b == b'<')) => {
                        LagValue::Enum(v)
                    }
                    _ => LagValue::Text,
                }
            }
        },
        Some("number" | "integer" | "boolean" | "array" | "object" | "null") => {
            LagValue::Json(compile_lax_schema(schema.expect("checked")))
        }
        _ => LagValue::Text,
    }
}

const PARAM_END: &[u8] = b"\n</parameter>\n";
const ARG_VALUE_END: &[u8] = b"</arg_value>";
/// muse's `render_atem` writes `'</atem:parameter>\n'` after every value, and
/// `muse::parse` cuts the value at the tag - so the newline belongs to the
/// SEPARATOR, not the value, and is spelled here rather than left optional.
const ATEM_PARAM_END: &[u8] = b"</atem:parameter>\n";
/// muse's analysis message runs free until it closes. `<|eom|>` is a single
/// special id, which is why `Dialect::grammar_specials` has to name it.
const ATEM_EOM: &[u8] = b"<|eom|>";

/// The tag three of the four syntaxes open a call with.
const TOOL_CALL_TAG: &[u8] = b"<tool_call>";

/// Closes a JSON-syntax call: the `}` belongs to the `{"name":...}` wrapper the
/// template writes as a literal, not to the arguments object.
const JSON_TAIL: &[u8] = b"}\n</tool_call>";

/// The JSON syntax's UNWRAPPED opener - see `ToolSyntax::bare_trigger`. Kept
/// byte-identical to what follows `<tool_call>\n` in the wrapped candidates so
/// the two forms cannot drift apart, and mirrored on the read-back side by
/// `parsers::JSON_BARE_OPEN`.
const JSON_BARE_OPEN: &[u8] = b"{\"name\": \"";

/// Closes an unwrapped call: just the object's own brace, since there is no
/// tag to close. Pairing `JSON_TAIL` with a bare opener would make the model
/// emit a `</tool_call>` whose opener never existed, and `json_tool_parse`
/// would file the whole thing as content - the grammar and the parser have to
/// agree about the frame, not only the object.
const JSON_BARE_TAIL: &[u8] = b"}";

/// The literal that ends one parameter VALUE in this syntax - the terminator a
/// free-text value runs to, and the tail a typed one is glued to.
fn value_end(syntax: ToolSyntax) -> &'static [u8] {
    match syntax {
        ToolSyntax::LagunaXml => ARG_VALUE_END,
        ToolSyntax::AtemXml => ATEM_PARAM_END,
        _ => PARAM_END,
    }
}

/// muse: what the model may write at a message boundary - one more `to=self`
/// thought, or the tool call itself.
///
/// `first` distinguishes the two entry points, and it is the whole reason this
/// is a function rather than a fixed candidate list: the generation prompt
/// ends with `<|start|>assistant`, so the first header is typed bare, while
/// every later one has to write its own `<|start|>assistant` after the
/// `<|eom|>` that closed the previous message.
///
/// Prefix-freeness - which `Choice` relies on - holds because ` to=self` and
/// ` to=NAME` diverge inside the recipient, and `ToolSet::compile` refuses a
/// tool actually named `self`.
fn atem_heads(set: &ToolSet, first: bool) -> Vec<(Vec<u8>, TNext)> {
    let lead: &[u8] = if first { b"" } else { b"<|start|>assistant" };
    let mut cands: Vec<(Vec<u8>, TNext)> = Vec::new();

    // the reasoning branch: this model's template always asks it to think, so
    // a forced call that skipped this would be forcing a lobotomy
    let mut think = lead.to_vec();
    think.extend_from_slice(b" to=self<|message|>");
    cands.push((think, TNext::Analysis));

    for (i, f) in set.fns.iter().enumerate() {
        let mut lit = lead.to_vec();
        lit.extend_from_slice(b" to=");
        lit.extend_from_slice(f.name.as_bytes());
        lit.extend_from_slice(b"<|message|><atem:function_calls>\n<atem:invoke name=\"");
        lit.extend_from_slice(f.name.as_bytes());
        lit.extend_from_slice(b"\">\n");
        cands.push((lit, TNext::Boundary(i, 0)));
    }
    cands
}

#[derive(Clone)]
enum TState {
    /// matching a fixed set of candidate literals; on full match of candidate
    /// i, `next(i)` decides the follow-up state
    Choice {
        cands: Arc<Vec<(Vec<u8>, TNext)>>,
        alive: Vec<bool>,
        pos: usize,
    },
    /// free-text value; kmp = matched prefix of the syntax's terminator
    ParamValue {
        fn_idx: usize,
        emitted: u32,
        kmp: usize,
    },
    /// free text until `term` matches, then `next`. muse's reasoning message:
    /// anything at all until `<|eom|>` closes it. (Distinct from `ParamValue`
    /// because there is no parameter bookkeeping and the terminator is not the
    /// syntax's value terminator.)
    FreeUntil {
        term: &'static [u8],
        kmp: usize,
        next: TNext,
    },
    /// a schema-directed JSON value, followed by the fixed `tail` literal
    Args {
        json: JsonMachine,
        tail: &'static [u8],
        next: TNext,
    },
    /// inside `lit`, `pos` bytes matched, then `next`
    Tail {
        lit: &'static [u8],
        pos: usize,
        next: TNext,
    },
    Done,
}

#[derive(Clone, Copy)]
enum TNext {
    /// at a parameter boundary: (fn_idx, emitted-bitmask so far)
    Boundary(usize, u32),
    /// laguna: `<arg_key>` is open, choose which key (fn_idx, emitted so far)
    Keys(usize, u32),
    /// free-text value chosen: (fn_idx, emitted-bitmask after this param)
    Param(usize, u32),
    /// laguna: a typed value follows - (fn_idx, emitted after, param index)
    LagValue(usize, u32, usize),
    /// muse: a `to=self` analysis message was opened - free text to `<|eom|>`
    Analysis,
    /// muse: an analysis message closed, so the model is back at
    /// `<|start|>assistant` and picks again: another thought, or the call
    AtemHead,
    /// function selected, JSON syntax: its arguments object follows
    JsonArgs(usize),
    /// same, entered through the UNWRAPPED opener - identical but for the
    /// tail, which has no `</tool_call>` to close
    JsonArgsBare(usize),
    /// closing sequence fully matched
    Finish,
}

/// The tool-call grammar machine.
#[derive(Clone)]
pub struct ToolMachine {
    set: Arc<ToolSet>,
    state: TState,
}

impl ToolMachine {
    /// A machine for the CALL itself, entered at `syntax.trigger()`. Every
    /// candidate literal here starts with that trigger, which is what lets the
    /// dispatcher hand the trigger bytes straight to a fresh machine.
    pub fn new(set: Arc<ToolSet>) -> ToolMachine {
        // Opener literals, one or two per candidate function. Each runs one
        // byte past the name into something the name cannot contain, so the
        // candidate set is prefix-free even for `get` vs `get_weather` - that
        // is not decoration on laguna, whose name rides bare after the tag.
        let mut cands: Vec<(Vec<u8>, TNext)> = Vec::new();
        for (i, f) in set.fns.iter().enumerate() {
            match set.syntax {
                ToolSyntax::QwenXml => {
                    let mut lit = b"<tool_call>\n<function=".to_vec();
                    lit.extend_from_slice(f.name.as_bytes());
                    lit.extend_from_slice(b">\n");
                    cands.push((lit, TNext::Boundary(i, 0)));
                }
                ToolSyntax::LagunaXml => {
                    let head = |suffix: &[u8]| {
                        let mut lit = b"<tool_call>".to_vec();
                        lit.extend_from_slice(f.name.as_bytes());
                        lit.extend_from_slice(suffix);
                        lit
                    };
                    if !f.params.is_empty() {
                        cands.push((head(b"<arg_key>"), TNext::Keys(i, 0)));
                    }
                    // a zero-argument call closes right after the name; only
                    // legal when nothing is required
                    if f.params.iter().all(|(_, req)| !req) {
                        cands.push((head(b"</tool_call>"), TNext::Finish));
                    }
                }
                ToolSyntax::Json => {
                    let mut lit = b"<tool_call>\n{\"name\": \"".to_vec();
                    lit.extend_from_slice(f.name.as_bytes());
                    lit.extend_from_slice(b"\", \"arguments\": ");
                    cands.push((lit, TNext::JsonArgs(i)));
                }
                ToolSyntax::AtemXml => {
                    let mut lit = ToolSyntax::AtemXml.trigger().to_vec();
                    lit.extend_from_slice(f.name.as_bytes());
                    lit.extend_from_slice(b"\">\n");
                    cands.push((lit, TNext::Boundary(i, 0)));
                }
            }
        }
        ToolMachine {
            set,
            state: TState::Choice {
                cands: Arc::new(cands),
                alive: Vec::new(),
                pos: 0,
            },
        }
    }

    /// A machine entered at `syntax.bare_trigger()` - the same call, minus the
    /// wrapper tag. `None` for a syntax that has no bare form.
    ///
    /// Every candidate starts with the bare trigger for the same reason `new`'s
    /// start with the wrapped one: the dispatcher replays the trigger bytes
    /// into the fresh machine, so it has to land exactly where the model
    /// already is.
    pub fn bare(set: Arc<ToolSet>) -> Option<ToolMachine> {
        set.syntax.bare_trigger()?;
        debug_assert_eq!(set.syntax, ToolSyntax::Json);
        let cands: Vec<(Vec<u8>, TNext)> = set
            .fns
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut lit = JSON_BARE_OPEN.to_vec();
                lit.extend_from_slice(f.name.as_bytes());
                lit.extend_from_slice(b"\", \"arguments\": ");
                (lit, TNext::JsonArgsBare(i))
            })
            .collect();
        Some(ToolMachine {
            set,
            state: TState::Choice {
                cands: Arc::new(cands),
                alive: Vec::new(),
                pos: 0,
            },
        })
    }

    /// A machine for a FORCED call: it owns the region from the first sampled
    /// token to the end of the call, so `tool_choice: "required"` means the
    /// turn cannot end without one.
    ///
    /// Identical to `new` everywhere the call is the first thing the model
    /// writes. muse is the exception: its turn opens with a message HEADER,
    /// and its template always asks for a thought first, so the forced region
    /// has to spell the analysis messages too - see `atem_heads`.
    pub fn forced(set: Arc<ToolSet>) -> ToolMachine {
        if set.syntax != ToolSyntax::AtemXml {
            return Self::new(set);
        }
        let cands = atem_heads(&set, true);
        ToolMachine {
            set,
            state: TState::Choice {
                cands: Arc::new(cands),
                alive: Vec::new(),
                pos: 0,
            },
        }
    }

    /// The one place a `TNext` becomes the state it names - reached from a
    /// completed `Choice` candidate and from a completed `Tail`.
    fn follow(&self, next: TNext) -> TState {
        match next {
            TNext::Boundary(fn_idx, emitted) => self.param_boundary(fn_idx, emitted),
            TNext::Keys(fn_idx, emitted) => self.key_choice(fn_idx, emitted),
            TNext::Param(fn_idx, emitted) => TState::ParamValue {
                fn_idx,
                emitted,
                kmp: 0,
            },
            TNext::LagValue(fn_idx, emitted, param) => self.lag_value(fn_idx, emitted, param),
            TNext::JsonArgs(i) => self.json_args(i, JSON_TAIL),
            TNext::JsonArgsBare(i) => self.json_args(i, JSON_BARE_TAIL),
            TNext::Analysis => TState::FreeUntil {
                term: ATEM_EOM,
                kmp: 0,
                next: TNext::AtemHead,
            },
            TNext::AtemHead => Self::choice(atem_heads(&self.set, false)),
            TNext::Finish => TState::Done,
        }
    }

    /// The arguments object of function `i`, glued to whichever closing
    /// literal the opener that got us here committed to.
    fn json_args(&self, i: usize, tail: &'static [u8]) -> TState {
        TState::Args {
            json: JsonMachine::new(
                self.set.fns[i]
                    .args
                    .clone()
                    .expect("json syntax compiles args"),
            ),
            tail,
            next: TNext::Finish,
        }
    }

    /// Sit inside `lit` with `pos` of it already matched - and fall straight
    /// through to `next` when that is all of it.
    ///
    /// The pass-through is not a nicety: the byte that SELECTS a tail also
    /// consumes its first byte, so a one-byte tail is finished the moment it is
    /// entered. Parking in `Tail` at `pos == len` would leave a complete call
    /// looking unterminated forever - `may_stop` is only true in `Done`, so the
    /// turn could never end. That is exactly what the unwrapped JSON call's
    /// lone `}` does (`JSON_BARE_TAIL`), and the reason this is a helper rather
    /// than a check at the one site that needed it.
    fn enter_tail(&self, lit: &'static [u8], pos: usize, next: TNext) -> TState {
        if pos == lit.len() {
            self.follow(next)
        } else {
            TState::Tail { lit, pos, next }
        }
    }

    fn choice(cands: Vec<(Vec<u8>, TNext)>) -> TState {
        TState::Choice {
            cands: Arc::new(cands),
            alive: Vec::new(),
            pos: 0,
        }
    }

    /// Between values: open another parameter, or close the call.
    fn param_boundary(&self, fn_idx: usize, emitted: u32) -> TState {
        let f = &self.set.fns[fn_idx];
        let left = f.params.len() as u32 > emitted.count_ones();
        let all_required_done = f
            .params
            .iter()
            .enumerate()
            .all(|(i, (_, req))| !req || emitted & (1 << i) != 0);
        let mut cands: Vec<(Vec<u8>, TNext)> = Vec::new();
        match self.set.syntax {
            ToolSyntax::QwenXml => {
                for (i, (key, _)) in f.params.iter().enumerate() {
                    if emitted & (1 << i) == 0 {
                        let mut lit = b"<parameter=".to_vec();
                        lit.extend_from_slice(key.as_bytes());
                        lit.extend_from_slice(b">\n");
                        cands.push((lit, TNext::Param(fn_idx, emitted | (1 << i))));
                    }
                }
                if all_required_done {
                    cands.push((b"</function>\n</tool_call>".to_vec(), TNext::Finish));
                }
            }
            ToolSyntax::LagunaXml => {
                // the key literal comes after `<arg_key>` here, so the tag is
                // its own step and the keys are a separate choice
                if left {
                    cands.push((b"<arg_key>".to_vec(), TNext::Keys(fn_idx, emitted)));
                }
                if all_required_done {
                    cands.push((b"</tool_call>".to_vec(), TNext::Finish));
                }
            }
            ToolSyntax::AtemXml => {
                for (i, (key, _)) in f.params.iter().enumerate() {
                    if emitted & (1 << i) == 0 {
                        let mut lit = b"<atem:parameter name=\"".to_vec();
                        lit.extend_from_slice(key.as_bytes());
                        lit.extend_from_slice(b"\">");
                        cands.push((lit, TNext::LagValue(fn_idx, emitted | (1 << i), i)));
                    }
                }
                if all_required_done {
                    // exactly what `render_atem` writes to close a call
                    cands.push((
                        b"</atem:invoke>\n</atem:function_calls>".to_vec(),
                        TNext::Finish,
                    ));
                }
            }
            ToolSyntax::Json => unreachable!("the json syntax has no parameter boundary"),
        }
        Self::choice(cands)
    }

    /// laguna: `<arg_key>` is open - which of the remaining keys, and its
    /// `</arg_key><arg_value>` in the same literal so the set stays prefix-free.
    fn key_choice(&self, fn_idx: usize, emitted: u32) -> TState {
        let f = &self.set.fns[fn_idx];
        let mut cands: Vec<(Vec<u8>, TNext)> = Vec::new();
        for (i, (key, _)) in f.params.iter().enumerate() {
            if emitted & (1 << i) == 0 {
                let mut lit = key.as_bytes().to_vec();
                lit.extend_from_slice(b"</arg_key><arg_value>");
                cands.push((lit, TNext::LagValue(fn_idx, emitted | (1 << i), i)));
            }
        }
        Self::choice(cands)
    }

    /// laguna / muse: the value itself, spelled the way the template would
    /// write it. Same three cases, only the closing tag differs.
    fn lag_value(&self, fn_idx: usize, emitted: u32, param: usize) -> TState {
        let end = value_end(self.set.syntax);
        match &self.set.fns[fn_idx].values[param] {
            LagValue::Text => TState::ParamValue {
                fn_idx,
                emitted,
                kmp: 0,
            },
            LagValue::Enum(variants) => Self::choice(
                variants
                    .iter()
                    .map(|v| {
                        let mut lit = v.as_bytes().to_vec();
                        lit.extend_from_slice(end);
                        (lit, TNext::Boundary(fn_idx, emitted))
                    })
                    .collect(),
            ),
            LagValue::Json(schema) => TState::Args {
                json: JsonMachine::new(schema.clone()),
                tail: end,
                next: TNext::Boundary(fn_idx, emitted),
            },
        }
    }

    pub fn may_stop(&self) -> bool {
        matches!(self.state, TState::Done)
    }

    pub fn feed(&mut self, b: u8) -> bool {
        // carried across the Choice->ParamValue transition via self.state writes
        match &mut self.state {
            TState::Done => false,
            TState::Choice { cands, alive, pos } => {
                if alive.is_empty() {
                    *alive = vec![true; cands.len()];
                }
                let mut any = false;
                let mut full: Option<TNext> = None;
                for (i, a) in alive.iter_mut().enumerate() {
                    if !*a {
                        continue;
                    }
                    let lit = &cands[i].0;
                    if *pos < lit.len() && lit[*pos] == b {
                        any = true;
                        if *pos + 1 == lit.len() {
                            full = Some(cands[i].1);
                        }
                    } else {
                        *a = false;
                    }
                }
                if !any {
                    return false;
                }
                *pos += 1;
                if let Some(next) = full {
                    // a fully matched candidate wins - candidates are
                    // prefix-free by construction: names/keys are unique and
                    // each literal runs one byte past them into something the
                    // name cannot contain (`>` or `<` for XML, `"` for JSON)
                    self.state = self.follow(next);
                }
                true
            }
            TState::ParamValue {
                fn_idx,
                emitted,
                kmp,
            } => {
                // KMP over the terminator; any byte is legal value content
                let (fi, em) = (*fn_idx, *emitted);
                let end = value_end(self.set.syntax);
                let mut k = *kmp;
                loop {
                    if end[k] == b {
                        k += 1;
                        break;
                    }
                    if k == 0 {
                        break;
                    }
                    k = kmp_fail(end, k);
                }
                if k == end.len() {
                    self.state = self.param_boundary(fi, em);
                    return true;
                }
                *kmp = k;
                true
            }
            TState::FreeUntil { term, kmp, next } => {
                // same KMP shape as ParamValue, but the region has no owner:
                // this is muse's thought, and every byte in it is legal
                let (term, next) = (*term, *next);
                let mut k = *kmp;
                loop {
                    if term[k] == b {
                        k += 1;
                        break;
                    }
                    if k == 0 {
                        break;
                    }
                    k = kmp_fail(term, k);
                }
                if k == term.len() {
                    self.state = self.follow(next);
                    return true;
                }
                *kmp = k;
                true
            }
            TState::Args { json, tail, next } => {
                // Feed the value machine directly. A rejected byte can leave it
                // half-advanced (a number frame pops before saying no), but
                // that is safe here because both exits throw the machine away:
                // either the value just ended and the fixed tail takes over, or
                // this candidate token is illegal and the whole ToolMachine
                // clone is dropped by `allows`.
                if json.feed(b) {
                    return true;
                }
                let (stop, tail, next) = (json.may_stop(), *tail, *next);
                if stop && tail[0] == b {
                    self.state = self.enter_tail(tail, 1, next);
                    return true;
                }
                false
            }
            TState::Tail { lit, pos, next } => {
                if lit.get(*pos) != Some(&b) {
                    return false;
                }
                let (lit, pos, next) = (*lit, *pos + 1, *next);
                self.state = self.enter_tail(lit, pos, next);
                true
            }
        }
    }
}

/// KMP failure link: given that `pat[..k]` matched, fall back to the longest
/// proper prefix of it that is also a suffix. Brute force over a pattern of a
/// dozen bytes, recomputed per mismatch - cheaper than carrying a table for
/// two literals whose only self-overlap is a leading `\n` or `<`.
/// One KMP step: given `pat[..k]` matched, how much matches after `b`. Reaching
/// `pat.len()` is a complete match.
fn kmp_advance(pat: &[u8], k: usize, b: u8) -> usize {
    let mut k = k;
    loop {
        if pat[k] == b {
            return k + 1;
        }
        if k == 0 {
            return 0;
        }
        k = kmp_fail(pat, k);
    }
}

fn kmp_fail(pat: &[u8], k: usize) -> usize {
    debug_assert!(k > 0 && k <= pat.len());
    let m = &pat[..k];
    let mut b = k - 1;
    while b > 0 {
        if m[..b] == m[k - b..] {
            return b;
        }
        b -= 1;
    }
    0
}

// ------------------------------------------------------------- dispatch --

impl ToolSyntax {
    /// The opener that arms a call. Every candidate literal `ToolMachine::new`
    /// builds starts with this, which is what lets the dispatcher hand the
    /// trigger bytes straight to a fresh machine instead of compiling a second
    /// grammar that begins after the tag. A new syntax has to answer here.
    fn trigger(self) -> &'static [u8] {
        match self {
            ToolSyntax::QwenXml | ToolSyntax::LagunaXml | ToolSyntax::Json => TOOL_CALL_TAG,
            // muse addresses a call in its message HEADER, which the model has
            // already written by the time a body could arm anything - so the
            // arming tag is the ATEM opener, run through `name="` so the tool
            // name lands inside the constrained region. That is also where
            // `muse::parse` reads the name from, and where llama.cpp's PEG
            // pins it (`p.tool_name` sits inside `<atem:invoke name="`), so
            // header and body cannot disagree about which tool was called.
            ToolSyntax::AtemXml => b"<atem:function_calls>\n<atem:invoke name=\"",
        }
    }

    /// A SECOND opener for the same call, spelled without the dialect's
    /// wrapper tag - the shape a model produces when it means to call a tool
    /// and simply never writes the tag.
    ///
    /// Measured on granite-vision-4.1-4b: given an image
    /// and a tool it wants, it emits a perfectly-formed
    /// `{"name": ..., "arguments": {...}}` object as its whole answer, with no
    /// `<tool_call>` anywhere. The object is right; only the frame is missing.
    /// Left alone that lands in content and the user reads raw JSON.
    ///
    /// Arming the grammar here is not wrapping that text after the fact - the
    /// machine takes over GENERATION at the trigger, so what comes back is
    /// constrained output produced under the constraint, name from the literal
    /// alternation and arguments through the schema. Nothing the model wrote
    /// as prose is ever promoted to a call.
    ///
    /// `AtemXml` above is the precedent that a trigger need not be the tag;
    /// this is the same move for a tag the model omitted rather than one it
    /// writes early. Only the JSON syntax has a bare form worth catching: its
    /// wrapper is pure decoration around a self-describing object, where the
    /// XML syntaxes carry the tool name inside markup that is itself the call.
    ///
    /// The dispatcher arms this only at TURN START (`DState::Free.fresh`) -
    /// see `DispatchMachine::feed` for why that guard is what makes a second
    /// opener safe.
    fn bare_trigger(self) -> Option<&'static [u8]> {
        match self {
            ToolSyntax::Json => Some(JSON_BARE_OPEN),
            _ => None,
        }
    }
}

/// Free text <-> tool grammar, re-armable: the TagDispatch shape from
/// XGrammar-2.
///
/// This is what makes `tool_choice: "auto"` enforceable. A forced choice is one
/// constrained region running to the end of the turn, which `ToolMachine` alone
/// already spells; an auto turn is prose, then MAYBE a call, then prose again,
/// and only a machine that RELEASES can express that. Everything else in here
/// existed already - the missing piece was the return edge.
///
/// What it buys, in the order it matters: a call that starts cannot be
/// malformed (no unbalanced JSON, no half-closed tag, no invented parameter),
/// the tool NAME is a literal alternation so a hallucinated name is
/// unrepresentable, and `may_stop` is false inside a call so the turn cannot
/// end mid-call. Post-hoc repair layers fix some fraction of those after the
/// fact; this leaves no fraction to fix.
#[derive(Clone)]
pub struct DispatchMachine {
    set: Arc<ToolSet>,
    trigger: &'static [u8],
    /// the same call spelled without its wrapper tag, when the syntax has one
    /// (`ToolSyntax::bare_trigger`). Armed only at turn start - see `feed`.
    bare: Option<&'static [u8]>,
    /// `parallel_tool_calls: false` - stop re-arming after the first call
    single: bool,
    state: DState,
}

#[derive(Clone)]
enum DState {
    /// outside a call; `kmp` = matched prefix of the trigger so far
    Free {
        kmp: usize,
        /// matched prefix of the BARE trigger, tracked in parallel
        bare_kmp: usize,
        /// nothing but whitespace has been generated yet (counting a partial
        /// bare-trigger match as still-fresh, since those bytes are the
        /// trigger). Gates the bare opener and nothing else.
        fresh: bool,
    },
    InCall(ToolMachine),
    /// one call done and `single` was set: free forever, trigger disarmed
    Spent,
}

impl DState {
    /// Outside a call, at the top of a turn.
    fn opening() -> DState {
        DState::Free {
            kmp: 0,
            bare_kmp: 0,
            fresh: true,
        }
    }

    /// Outside a call, but past the point where a bare opener may arm - the
    /// state a completed call returns to.
    fn resumed() -> DState {
        DState::Free {
            kmp: 0,
            bare_kmp: 0,
            fresh: false,
        }
    }
}

impl DispatchMachine {
    pub fn new(set: Arc<ToolSet>, single: bool) -> DispatchMachine {
        let trigger = set.syntax.trigger();
        let bare = set.syntax.bare_trigger();
        DispatchMachine {
            set,
            trigger,
            bare,
            single,
            state: DState::opening(),
        }
    }

    fn feed(&mut self, b: u8) -> bool {
        // Free is read by value first: the arm replaces `self.state`, which it
        // cannot do while holding a borrow into it.
        if let DState::Free {
            kmp,
            bare_kmp,
            fresh,
        } = self.state
        {
            let k = kmp_advance(self.trigger, kmp, b);
            if k == self.trigger.len() {
                return self.arm(ToolMachine::new(self.set.clone()), self.trigger);
            }

            // The bare opener, tracked in parallel and armed only while the
            // turn is still `fresh`. Committing to a call on `{"name": "` is a
            // read of INTENT, unlike `<tool_call>`, which is unambiguous - so
            // it gets the one context where the read is safe: a turn whose
            // entire output so far is the opener itself. A model writing prose
            // and then showing a JSON example never reaches it; a model whose
            // whole answer is a call blob (the measured granite-vision case)
            // hits it on byte one.
            let bk = match self.bare {
                Some(t) if fresh => kmp_advance(t, bare_kmp, b),
                _ => 0,
            };
            if let Some(t) = self.bare
                && bk == t.len()
                && let Some(m) = ToolMachine::bare(self.set.clone())
            {
                return self.arm(m, t);
            }

            // Freshness survives whitespace and the trigger's own bytes;
            // anything else means the turn has started saying something, and
            // the bare opener is off the table for the rest of it.
            self.state = DState::Free {
                kmp: k,
                bare_kmp: bk,
                fresh: fresh && (b.is_ascii_whitespace() || bk > 0),
            };
            return true;
        }
        let done = match &mut self.state {
            DState::Spent => return true,
            DState::Free { .. } => unreachable!("handled above"),
            DState::InCall(m) => {
                if !m.feed(b) {
                    return false;
                }
                m.may_stop()
            }
        };
        if done {
            self.state = if self.single {
                DState::Spent
            } else {
                // `resumed`, not `opening`: a second call mid-turn is no
                // longer at turn start, so only the unambiguous tag can open it
                DState::resumed()
            };
        }
        true
    }

    /// Trigger complete: hand those exact bytes to the fresh machine - every
    /// opener candidate begins with them, so it lands precisely where the model
    /// already is.
    fn arm(&mut self, mut m: ToolMachine, trigger: &'static [u8]) -> bool {
        for &t in trigger {
            if !m.feed(t) {
                // Only reachable if a future syntax's candidates do not all
                // start with its trigger. Staying free degrades to
                // unconstrained decoding; engaging anyway would deadlock
                // the request.
                debug_assert!(false, "trigger is not a prefix of every opener candidate");
                self.state = DState::resumed();
                return true;
            }
        }
        self.state = DState::InCall(m);
        true
    }

    fn in_call(&self) -> bool {
        matches!(self.state, DState::InCall(_))
    }

    /// Outside a call the turn may end; inside one it may not - which is the
    /// whole truncated-tool-call failure mode gone.
    pub fn may_stop(&self) -> bool {
        !self.in_call()
    }
}

// --------------------------------------------------------- gated wrapper --

/// Free-until-trigger gate: reasoning tokens flow unconstrained, the grammar
/// activates after the dialect's content trigger.
pub enum Gate {
    Immediate,
    /// activate after this token id (qwen `</think>`)
    AfterToken(u32),
    /// activate at the `<|message|>` that follows a `final` channel header
    HarmonyFinal {
        channel: u32,
        message: u32,
        collecting: bool,
        header: Vec<u8>,
    },
    /// muse: activate at the `<|message|>` of the first message not addressed
    /// `to=self` - everything the model writes to itself is reasoning and runs
    /// free. Same shape as `HarmonyFinal`, with two differences that come
    /// straight from the grammar: the header is opened by `<|start|>` rather
    /// than a channel marker, and it starts open, because the generation
    /// prompt already spelled `<|start|>assistant` and the model types its
    /// recipient immediately.
    MuseContent {
        start: u32,
        message: u32,
        collecting: bool,
        header: Vec<u8>,
    },
}

#[derive(Clone)]
pub enum Machine {
    Json(JsonMachine),
    Tool(ToolMachine),
    Dispatch(DispatchMachine),
}

impl Machine {
    fn feed(&mut self, b: u8) -> bool {
        match self {
            Machine::Json(m) => m.feed(b),
            Machine::Tool(m) => m.feed(b),
            Machine::Dispatch(m) => m.feed(b),
        }
    }
    fn may_stop(&self) -> bool {
        match self {
            Machine::Json(m) => m.may_stop(),
            Machine::Tool(m) => m.may_stop(),
            Machine::Dispatch(m) => m.may_stop(),
        }
    }
    /// True when the machine constrains nothing right now: a dispatcher outside
    /// a call, where the model is writing its ordinary answer. Control tokens
    /// are legal there too (the engine's stop handling owns the stop tokens).
    ///
    /// This is also the hot path. `allows` runs per nucleus candidate, and
    /// without this the free region would clone the machine and replay the
    /// token's bytes thousands of times a step to learn what it already knows:
    /// outside a call, everything is legal.
    fn unconstrained(&self) -> bool {
        match self {
            Machine::Json(_) | Machine::Tool(_) => false,
            Machine::Dispatch(m) => !m.in_call(),
        }
    }
}

pub struct GatedConstraint {
    vocab: Arc<VocabBytes>,
    gate: Gate,
    active: bool,
    machine: Machine,
    /// Special ids the grammar itself may spell (`Dialect::grammar_specials`).
    /// Empty for every family whose tool syntax is plain text; muse's message
    /// envelope is made of control tokens, so its grammar cannot be written
    /// without them. Kept as a tiny sorted Vec - it holds three ids at most
    /// and `allows` runs per nucleus candidate, so a linear scan beats a hash.
    preserved: Vec<u32>,
}

impl GatedConstraint {
    pub fn new(
        vocab: Arc<VocabBytes>,
        gate: Gate,
        machine: Machine,
        preserved: Vec<u32>,
    ) -> GatedConstraint {
        let active = matches!(gate, Gate::Immediate);
        GatedConstraint {
            vocab,
            gate,
            active,
            machine,
            preserved,
        }
    }
}

impl TokenConstraint for GatedConstraint {
    fn allows(&self, id: u32) -> bool {
        if !self.active {
            return true; // free phase (stop tokens are the engine's call)
        }
        if self.machine.unconstrained() {
            // Dispatcher outside a call: ordinary content, nothing to check.
            // `accept` still runs the token's bytes through the trigger
            // matcher - load-bearing when a family tokenizes its own
            // `<tool_call>` as a single control id.
            return true;
        }
        if self.vocab.is_special(id) && !self.preserved.contains(&id) {
            // no control tokens inside constrained output, except the ones
            // this dialect's grammar is written out of (muse's envelope)
            return false;
        }
        let bytes = self.vocab.bytes(id);
        if bytes.is_empty() {
            return false;
        }
        let mut m = self.machine.clone();
        bytes.iter().all(|&b| m.feed(b))
    }

    fn accept(&mut self, id: u32) {
        if self.active {
            for &b in self.vocab.bytes(id) {
                let ok = self.machine.feed(b);
                debug_assert!(ok, "accept() of a token allows() rejected");
            }
            return;
        }
        match &mut self.gate {
            Gate::Immediate => unreachable!("immediate gate starts active"),
            Gate::AfterToken(t) => {
                if id == *t {
                    self.active = true;
                }
            }
            Gate::MuseContent {
                start,
                message,
                collecting,
                header,
            } => {
                if id == *start {
                    // a new message header opens; the role word lands in the
                    // buffer with it, which `to=self` never matches anyway
                    *collecting = true;
                    header.clear();
                } else if *collecting && id == *message {
                    // the header is ` to=RECIPIENT`, or `assistant to=...` once
                    // the model has written its own `<|start|>assistant`, or
                    // empty (the template makes ` to=user` optional)
                    let h = String::from_utf8_lossy(header);
                    let recipient = h.trim().rsplit_once("to=").map(|(_, r)| r.trim());
                    if recipient != Some("self") {
                        self.active = true;
                    }
                    *collecting = false;
                } else if *collecting {
                    header.extend_from_slice(self.vocab.bytes(id));
                }
            }
            Gate::HarmonyFinal {
                channel,
                message,
                collecting,
                header,
            } => {
                if id == *channel {
                    *collecting = true;
                    header.clear();
                } else if *collecting && id == *message {
                    let name = String::from_utf8_lossy(header);
                    if name.trim() == "final" {
                        self.active = true;
                    }
                    *collecting = false;
                } else if *collecting {
                    header.extend_from_slice(self.vocab.bytes(id));
                }
            }
        }
    }

    fn may_stop(&self) -> bool {
        // Pre-activation is the free phase: `allows` already hands every token
        // to the model there ("stop tokens are the engine's call"), and this
        // must agree. It didn't - `active &&` masked the stop ids out of the
        // nucleus for the whole thinking region, silently reweighting every
        // draw where a stop id carried mass. Measured on a seeded replay:
        // 3/8 seeds degenerated into parallel-call spam with
        // the grammar armed against 0/40 without it, and the trajectories
        // diverged from the first thinking token. Once the gate arms, the
        // machine owns the answer, exactly as before.
        !self.active || self.machine.may_stop()
    }

    fn free_now(&self) -> bool {
        // pre-activation reasoning, or an armed dispatcher between calls:
        // every token is legal, so spec rounds are exact here
        !self.active || self.machine.unconstrained()
    }
}

// ── thinking budget ──────────────────────────────────────────────────────

/// Where a [`BudgetGated`] wrapper is in its life.
enum BudgetPhase {
    /// Reasoning is running free; accepted tokens count against the budget.
    Counting,
    /// The budget is spent: the next legal token is `exit_ids[i]` and nothing
    /// else - the dialect's early-exit phrase plus its think-close marker is
    /// injected one token per step through the ordinary sampling seam.
    Forcing(usize),
    /// The think block closed (naturally or by injection); this wrapper is
    /// transparent and the inner grammar - if any - owns the stream.
    Done,
}

/// Thinking budget as a [`TokenConstraint`] wrapper (Anthropic
/// `thinking.budget_tokens` / OpenRouter `reasoning.max_tokens`): count the
/// reasoning tokens, and when the budget is spent force the model out of its
/// think block with the dialect's own budget-exhaustion recipe - the Qwen3
/// technical report's published mechanism (inject "Considering the limited
/// time by the user, I have to give the solution based on the thinking
/// directly now." and close the block), not a bare close-tag slammed
/// mid-word.
///
/// Composition rules, learned from the may_stop incident above:
/// - Counting and Done DELEGATE to the inner grammar rather than blanket-
///   allowing, so a budget composed with a forced-tool grammar can never
///   bypass it (pre-activation the inner allows everything anyway, so
///   delegation is behavior-identical for the free phase).
/// - `accept` always feeds the inner first: the forced close token is how the
///   inner's gate learns the think block ended, exactly as if the model had
///   sampled it.
/// - Forcing masks stops (`may_stop` = false): the injection is atomic.
///
/// The sampler side needs no changes: during Forcing the nucleus is usually
/// fully illegal and `sample_constrained` falls back to its whole-vocab
/// argmax-and-mask walk, which finds the single legal id. A slot carrying any
/// constraint already runs the dense decode path, so arming a budget costs
/// what arming a tool grammar costs - nothing new.
/// Past the budget, how many more tokens an open tool call may run before
/// the exit is injected anyway. Severing a call corrupts it, so the injection
/// defers to the call's close - but the bound has to bind, and a call that
/// never closes is already broken.
const CALL_DEFER_CAP: usize = 1024;

pub struct BudgetGated {
    vocab: Arc<VocabBytes>,
    inner: Option<GatedConstraint>,
    budget: usize,
    count: usize,
    /// The dialect's think-close id (`</think>`, gemma's `<channel|>`): seeing
    /// it sampled naturally disarms the budget.
    disarm: u32,
    /// The forced exit sequence, ending at the close id - no trailing text,
    /// so a grammar that activates on the close token owns everything after.
    exit_ids: Vec<u32>,
    /// The dialect's tool-call open/close markers, when it has any: qwen
    /// writes calls inside a still-open think block (seen live),
    /// and injecting the exit phrase mid-call severs the call - the model
    /// then re-derives it from corrupted context. Injection waits for the
    /// close marker (bounded by [`CALL_DEFER_CAP`]).
    call_markers: Option<(Vec<u8>, Vec<u8>)>,
    /// open-call depth inside the think region
    call_depth: usize,
    /// rolling byte tail so a marker split across tokens still matches
    tail: Vec<u8>,
    phase: BudgetPhase,
}

impl BudgetGated {
    pub fn new(
        vocab: Arc<VocabBytes>,
        inner: Option<GatedConstraint>,
        budget: usize,
        disarm: u32,
        exit_ids: Vec<u32>,
        call_markers: Option<(Vec<u8>, Vec<u8>)>,
    ) -> BudgetGated {
        debug_assert!(!exit_ids.is_empty(), "exit sequence must not be empty");
        debug_assert_eq!(
            exit_ids.last(),
            Some(&disarm),
            "exit must end at the close id"
        );
        BudgetGated {
            vocab,
            inner,
            budget,
            count: 0,
            disarm,
            exit_ids,
            call_markers,
            call_depth: 0,
            tail: Vec::new(),
            phase: BudgetPhase::Counting,
        }
    }

    /// Update `call_depth` from this token's bytes. The buffer keeps the last
    /// `max(marker)-1` bytes, so a marker split across token boundaries still
    /// matches and a fully-retained match cannot be counted twice (it would
    /// need one more byte than the tail can hold).
    fn track_calls(&mut self, id: u32) {
        let Some((open, close)) = &self.call_markers else {
            return;
        };
        self.tail.extend_from_slice(self.vocab.bytes(id));
        let (mut at, buf) = (0usize, std::mem::take(&mut self.tail));
        loop {
            let o = find(&buf[at..], open);
            let c = find(&buf[at..], close);
            match (o, c) {
                (Some(i), Some(j)) if i < j => {
                    self.call_depth += 1;
                    at += i + open.len();
                }
                (Some(_), Some(j)) | (None, Some(j)) => {
                    self.call_depth = self.call_depth.saturating_sub(1);
                    at += j + close.len();
                }
                (Some(i), None) => {
                    self.call_depth += 1;
                    at += i + open.len();
                }
                (None, None) => break,
            }
        }
        let keep = open.len().max(close.len()).saturating_sub(1);
        self.tail = buf[buf.len().saturating_sub(keep).max(at)..].to_vec();
    }
}

/// First occurrence of `needle` in `hay` (short markers, tiny windows - a
/// naive scan beats pulling in a search crate).
fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

impl TokenConstraint for BudgetGated {
    fn allows(&self, id: u32) -> bool {
        match self.phase {
            BudgetPhase::Forcing(i) => id == self.exit_ids[i],
            BudgetPhase::Counting | BudgetPhase::Done => {
                self.inner.as_ref().is_none_or(|c| c.allows(id))
            }
        }
    }

    fn accept(&mut self, id: u32) {
        // Inner first: its gate must see every token - including the forced
        // close - to arm at the think boundary like it always has.
        if let Some(c) = &mut self.inner {
            c.accept(id);
        }
        match self.phase {
            BudgetPhase::Counting => {
                if id == self.disarm {
                    self.phase = BudgetPhase::Done;
                    return;
                }
                self.track_calls(id);
                self.count += 1;
                if self.count >= self.budget
                    && (self.call_depth == 0 || self.count >= self.budget + CALL_DEFER_CAP)
                {
                    self.phase = BudgetPhase::Forcing(0);
                }
            }
            BudgetPhase::Forcing(i) => {
                debug_assert_eq!(id, self.exit_ids[i], "accept() of an id allows() rejected");
                self.phase = if i + 1 == self.exit_ids.len() {
                    BudgetPhase::Done
                } else {
                    BudgetPhase::Forcing(i + 1)
                };
            }
            BudgetPhase::Done => {}
        }
    }

    fn may_stop(&self) -> bool {
        match self.phase {
            BudgetPhase::Forcing(_) => false,
            BudgetPhase::Counting | BudgetPhase::Done => {
                self.inner.as_ref().is_none_or(|c| c.may_stop())
            }
        }
    }

    fn free_now(&self) -> bool {
        match self.phase {
            // the injection is a forced sequence - never speculate through it
            BudgetPhase::Forcing(_) => false,
            // counting/done delegate: free is the INNER grammar's question,
            // and the budget transition itself is caught by the caller's
            // post-accept free_now() re-check (count reaching the budget
            // flips Forcing inside accept(), turning this false)
            BudgetPhase::Counting | BudgetPhase::Done => {
                self.inner.as_ref().is_none_or(|c| c.free_now())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather_tool() -> Value {
        json!({"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"},
                "units": {"type": "string"}
            }, "required": ["city"]}
        }})
    }

    fn tool_set() -> Arc<ToolSet> {
        ToolSet::compile(ToolSyntax::QwenXml, &[weather_tool()], None).expect("compile tools")
    }

    fn json_tool_set() -> Arc<ToolSet> {
        tool_set_for(ToolSyntax::Json)
    }

    /// The same one tool compiled for whichever syntax - for the checks that
    /// are about the SYNTAX rather than about a particular schema.
    fn tool_set_for(syntax: ToolSyntax) -> Arc<ToolSet> {
        ToolSet::compile(syntax, &[weather_tool()], None).expect("compile tools")
    }

    /// feed a whole string; true = every byte was legal
    fn tool_feed(m: &mut ToolMachine, s: &str) -> bool {
        s.bytes().all(|b| m.feed(b))
    }

    #[test]
    fn tool_grammar_happy_path() {
        let mut m = ToolMachine::new(tool_set());
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        assert!(text.bytes().all(|b| m.feed(b)), "full call must be legal");
        assert!(m.may_stop());
        assert!(!m.feed(b'x'), "nothing after the call");
    }

    #[test]
    fn tool_grammar_requires_required_params() {
        let mut m = ToolMachine::new(tool_set());
        let prefix = "<tool_call>\n<function=get_weather>\n";
        assert!(prefix.bytes().all(|b| m.feed(b)));
        // closing before emitting required `city` must be impossible
        assert!(!"</".bytes().all(|b| m.feed(b)));
    }

    #[test]
    fn tool_grammar_value_may_contain_angle_brackets() {
        let mut m = ToolMachine::new(tool_set());
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\na <b> c\nd\n</parameter>\n</function>\n</tool_call>";
        assert!(text.bytes().all(|b| m.feed(b)));
        assert!(m.may_stop());
    }

    #[test]
    fn tool_grammar_no_duplicate_params() {
        let mut m = ToolMachine::new(tool_set());
        let text =
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nx\n</parameter>\n<parameter=c";
        assert!(
            !text.bytes().all(|b| m.feed(b)),
            "city already emitted; only units/close legal"
        );
    }

    // ------------------------------------------------- JSON (granite) syntax --

    /// The exact byte sequence granite's own chat template renders for an
    /// assistant tool call - the grammar and the template have to agree or the
    /// forced call is off-distribution from its first token.
    #[test]
    fn json_tool_grammar_matches_the_template_shape() {
        let mut m = ToolMachine::new(json_tool_set());
        assert!(tool_feed(
            &mut m,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"
        ));
        assert!(m.may_stop());
        assert!(!m.feed(b'x'), "nothing after the call");
    }

    #[test]
    fn json_tool_grammar_allows_optional_params_in_either_order() {
        for args in [
            "{\"city\": \"Paris\", \"units\": \"c\"}",
            "{\"units\": \"c\", \"city\": \"Paris\"}",
        ] {
            let mut m = ToolMachine::new(json_tool_set());
            let text = format!(
                "<tool_call>\n{{\"name\": \"get_weather\", \"arguments\": {args}}}\n</tool_call>"
            );
            assert!(tool_feed(&mut m, &text), "{args}");
            assert!(m.may_stop(), "{args}");
        }
    }

    #[test]
    fn json_tool_grammar_requires_required_params() {
        let mut m = ToolMachine::new(json_tool_set());
        assert!(tool_feed(
            &mut m,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {"
        ));
        // `city` is required, so neither an empty object nor a units-only one
        // may close
        assert!(!m.clone().feed(b'}'), "empty args must be illegal");
        let mut n = m.clone();
        assert!(tool_feed(&mut n, "\"units\": \"c\""));
        assert!(!n.feed(b'}'), "closing without city must be illegal");
        assert!(tool_feed(&mut m, "\"city\": \"Paris\"}"));
    }

    #[test]
    fn json_tool_grammar_rejects_undeclared_and_duplicate_keys() {
        let mut m = ToolMachine::new(json_tool_set());
        assert!(tool_feed(
            &mut m,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {"
        ));
        assert!(!m.clone().feed(b'z'), "undeclared key must be illegal");
        assert!(tool_feed(&mut m, "\"city\": \"Paris\", "));
        assert!(!tool_feed(&mut m.clone(), "\"city"), "city already emitted");
        assert!(tool_feed(&mut m, "\"units\": \"c\"}"));
    }

    /// Types are enforced, which is the point on `/v1/messages`: a forced call
    /// there is the Anthropic API's only schema-shaped output.
    #[test]
    fn json_tool_grammar_enforces_value_types() {
        let tools = [json!({"type": "function", "function": {
            "name": "chart",
            "parameters": {"type": "object", "properties": {
                "bars": {"type": "integer"},
                "kind": {"type": "string", "enum": ["bar", "line"]}
            }, "required": ["bars", "kind"]}
        }})];
        let set = ToolSet::compile(ToolSyntax::Json, &tools, None).expect("compile");
        let mut m = ToolMachine::new(set);
        assert!(tool_feed(
            &mut m,
            "<tool_call>\n{\"name\": \"chart\", \"arguments\": {\"bars\": 4"
        ));
        assert!(!m.clone().feed(b'.'), "integer must reject a fraction");
        assert!(tool_feed(&mut m, ", \"kind\": \"b"));
        assert!(
            !m.clone().feed(b'x'),
            "enum variant must stay on a listed value"
        );
        assert!(tool_feed(&mut m, "ar\"}}\n</tool_call>"));
        assert!(m.may_stop());
    }

    /// A parameter the strict subset can't model degrades to free JSON rather
    /// than 400-ing a request whose call we can still make.
    #[test]
    fn json_tool_grammar_degrades_unmodelable_params_to_free_json() {
        let tools = [json!({"type": "function", "function": {
            "name": "f",
            "parameters": {"type": "object", "properties": {
                "x": {"anyOf": [{"type": "string"}, {"type": "number"}]}
            }, "required": ["x"]}
        }})];
        let set = ToolSet::compile(ToolSyntax::Json, &tools, None).expect("must not refuse");
        for v in ["\"s\"", "3.5", "{\"a\": [1, null]}"] {
            let mut m = ToolMachine::new(set.clone());
            let text = format!(
                "<tool_call>\n{{\"name\": \"f\", \"arguments\": {{\"x\": {v}}}}}\n</tool_call>"
            );
            assert!(tool_feed(&mut m, &text), "{v}");
            assert!(m.may_stop(), "{v}");
        }
    }

    /// Both APIs spell a zero-argument tool as `parameters` with no
    /// properties; the only legal arguments value is then `{}`.
    #[test]
    fn json_tool_grammar_forces_an_empty_object_for_a_zero_arg_tool() {
        let tools = [json!({"type": "function", "function": {"name": "now"}})];
        let set = ToolSet::compile(ToolSyntax::Json, &tools, None).expect("compile");
        let mut m = ToolMachine::new(set);
        assert!(tool_feed(
            &mut m,
            "<tool_call>\n{\"name\": \"now\", \"arguments\": {"
        ));
        assert!(!m.clone().feed(b'"'), "no key may open");
        assert!(tool_feed(&mut m, "}}\n</tool_call>"));
        assert!(m.may_stop());
    }

    /// Names sharing a prefix must both stay reachable: the opener literal
    /// runs past the name into `", "` precisely so the candidate set is
    /// prefix-free.
    #[test]
    fn json_tool_grammar_disambiguates_prefix_named_functions() {
        let tools = [
            json!({"type": "function", "function": {"name": "get"}}),
            json!({"type": "function", "function": {"name": "get_weather"}}),
        ];
        let set = ToolSet::compile(ToolSyntax::Json, &tools, None).expect("compile");
        for name in ["get", "get_weather"] {
            let mut m = ToolMachine::new(set.clone());
            let text =
                format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {{}}}}\n</tool_call>");
            assert!(tool_feed(&mut m, &text), "{name}");
            assert!(m.may_stop(), "{name}");
        }
    }

    #[test]
    fn json_tool_grammar_honors_the_named_function_filter() {
        let tools = [
            json!({"type": "function", "function": {"name": "a"}}),
            json!({"type": "function", "function": {"name": "b"}}),
        ];
        let set = ToolSet::compile(ToolSyntax::Json, &tools, Some("b")).expect("compile");
        let mut m = ToolMachine::new(set);
        assert!(tool_feed(&mut m, "<tool_call>\n{\"name\": \""));
        assert!(
            !m.clone().feed(b'a'),
            "filtered-out function must be unreachable"
        );
        assert!(m.feed(b'b'));
    }

    // ------------------------------------------------- laguna (GLM) syntax --

    fn laguna_tool() -> Value {
        json!({"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "units": {"type": "string", "enum": ["c", "f"]}
            }, "required": ["city"]}
        }})
    }

    fn laguna_set() -> Arc<ToolSet> {
        ToolSet::compile(ToolSyntax::LagunaXml, &[laguna_tool()], None).expect("compile tools")
    }

    /// The exact byte sequence laguna's template renders: no padding anywhere,
    /// the name bare after the opener.
    #[test]
    fn laguna_tool_grammar_matches_the_template_shape() {
        let mut m = ToolMachine::new(laguna_set());
        assert!(tool_feed(
            &mut m,
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
        ));
        assert!(m.may_stop());
        assert!(!m.feed(b'x'), "nothing after the call");
    }

    /// The name rides BARE after `<tool_call>`, so `get` is a byte-prefix of
    /// `get_weather` - the opener has to run into the following tag or one of
    /// them becomes unreachable.
    #[test]
    fn laguna_tool_grammar_disambiguates_prefix_named_functions() {
        let tools = [
            json!({"type": "function", "function": {"name": "get"}}),
            json!({"type": "function", "function": {"name": "get_weather"}}),
        ];
        let set = ToolSet::compile(ToolSyntax::LagunaXml, &tools, None).expect("compile");
        for name in ["get", "get_weather"] {
            let mut m = ToolMachine::new(set.clone());
            assert!(
                tool_feed(&mut m, &format!("<tool_call>{name}</tool_call>")),
                "{name}"
            );
            assert!(m.may_stop(), "{name}");
        }
    }

    #[test]
    fn laguna_tool_grammar_requires_required_params() {
        let mut m = ToolMachine::new(laguna_set());
        assert!(tool_feed(&mut m, "<tool_call>get_weather"));
        // `city` is required, so the zero-argument close is not even offered
        assert!(
            !tool_feed(&mut m.clone(), "</tool_call>"),
            "closed without city"
        );
        // nor may it close after emitting only an optional one
        let mut n = m.clone();
        assert!(tool_feed(
            &mut n,
            "<arg_key>days</arg_key><arg_value>3</arg_value>"
        ));
        assert!(!tool_feed(&mut n, "</tool_call>"), "closed without city");
        assert!(tool_feed(
            &mut m,
            "<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
        ));
        assert!(m.may_stop());
    }

    #[test]
    fn laguna_tool_grammar_rejects_undeclared_and_duplicate_keys() {
        let mut m = ToolMachine::new(laguna_set());
        assert!(tool_feed(&mut m, "<tool_call>get_weather<arg_key>"));
        assert!(!m.clone().feed(b'z'), "undeclared key must be illegal");
        assert!(tool_feed(
            &mut m,
            "city</arg_key><arg_value>Paris</arg_value><arg_key>"
        ));
        assert!(!tool_feed(&mut m.clone(), "city"), "city already emitted");
        assert!(tool_feed(
            &mut m,
            "days</arg_key><arg_value>3</arg_value></tool_call>"
        ));
        assert!(m.may_stop());
    }

    /// Laguna values are TYPED, unlike qwen's: the template writes
    /// `tojson` for anything that is not a string, so a declared integer must
    /// be JSON and a declared string is bare text.
    #[test]
    fn laguna_tool_grammar_types_values_the_way_the_template_writes_them() {
        // a string value is bare - quotes would be part of the string
        let mut m = ToolMachine::new(laguna_set());
        assert!(tool_feed(
            &mut m,
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>a <b> c</arg_value></tool_call>"
        ));
        assert!(m.may_stop(), "free text must accept angle brackets");

        // an integer value must be JSON: no fraction, no bare word
        let mut m = ToolMachine::new(laguna_set());
        assert!(tool_feed(
            &mut m,
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
             <arg_key>days</arg_key><arg_value>3"
        ));
        assert!(!m.clone().feed(b'.'), "integer must reject a fraction");
        assert!(!m.clone().feed(b'x'), "integer must reject a word");
        assert!(tool_feed(&mut m, "</arg_value></tool_call>"));
        assert!(m.may_stop());

        // an enum stays on a listed variant, still unquoted
        let mut m = ToolMachine::new(laguna_set());
        assert!(tool_feed(
            &mut m,
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
             <arg_key>units</arg_key><arg_value>"
        ));
        assert!(
            !m.clone().feed(b'x'),
            "enum must reject an unlisted variant"
        );
        assert!(tool_feed(&mut m, "c</arg_value></tool_call>"));
        assert!(m.may_stop());
    }

    /// A union or unreadable type is the one case the template's own rule
    /// cannot be predicted from the schema (it tests the VALUE's runtime type),
    /// so those fall back to free text, which accepts either spelling.
    #[test]
    fn laguna_tool_grammar_falls_back_to_text_for_unreadable_types() {
        let tools = [json!({"type": "function", "function": {
            "name": "f",
            "parameters": {"type": "object", "properties": {
                "x": {"type": ["string", "null"]}
            }, "required": ["x"]}
        }})];
        let set = ToolSet::compile(ToolSyntax::LagunaXml, &tools, None).expect("must not refuse");
        for v in ["bare words", "null", "{\"a\": 1}"] {
            let mut m = ToolMachine::new(set.clone());
            assert!(
                tool_feed(
                    &mut m,
                    &format!(
                        "<tool_call>f<arg_key>x</arg_key><arg_value>{v}</arg_value></tool_call>"
                    )
                ),
                "{v}"
            );
            assert!(m.may_stop(), "{v}");
        }
    }

    #[test]
    fn laguna_tool_grammar_honors_the_named_function_filter() {
        let tools = [
            json!({"type": "function", "function": {"name": "a"}}),
            json!({"type": "function", "function": {"name": "b"}}),
        ];
        let set = ToolSet::compile(ToolSyntax::LagunaXml, &tools, Some("b")).expect("compile");
        let mut m = ToolMachine::new(set);
        assert!(tool_feed(&mut m, "<tool_call>"));
        assert!(
            !m.clone().feed(b'a'),
            "filtered-out function must be unreachable"
        );
        assert!(m.feed(b'b'));
    }

    #[test]
    fn laguna_tool_grammar_output_round_trips_through_the_parser() {
        let mut m = ToolMachine::new(laguna_set());
        let text = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
                    <arg_key>days</arg_key><arg_value>3</arg_value>\
                    <arg_key>units</arg_key><arg_value>c</arg_value></tool_call>";
        assert!(tool_feed(&mut m, text));
        assert!(m.may_stop());
        let hints = crate::parsers::tool_hints(Some(&[laguna_tool()]));
        let p = crate::parsers::parse(crate::parsers::Dialect::Laguna, text, false, hints.as_ref());
        assert_eq!(p.tool_calls.len(), 1, "{p:?}");
        assert_eq!(p.complete_calls, 1);
        assert_eq!(p.tool_calls[0].name, "get_weather");
        // and the typing survives the trip: `days` is a number, not "3"
        assert_eq!(
            serde_json::from_str::<Value>(&p.tool_calls[0].arguments).expect("args parse"),
            json!({"city": "Paris", "days": 3, "units": "c"})
        );
        assert_eq!(p.content, None, "the whole generation is the call");
    }

    /// Whatever the grammar emits, the dialect's own parser has to read back
    /// as a tool call - that agreement is the whole contract.
    #[test]
    fn json_tool_grammar_output_round_trips_through_the_parser() {
        let mut m = ToolMachine::new(json_tool_set());
        let text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"units\": \"c\"}}\n</tool_call>";
        assert!(tool_feed(&mut m, text));
        assert!(m.may_stop());
        let hints = crate::parsers::tool_hints(Some(&[weather_tool()]));
        let p = crate::parsers::parse(
            crate::parsers::Dialect::JsonToolCall,
            text,
            false,
            hints.as_ref(),
        );
        assert_eq!(p.tool_calls.len(), 1, "{p:?}");
        assert_eq!(p.complete_calls, 1);
        assert_eq!(p.tool_calls[0].name, "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(&p.tool_calls[0].arguments).expect("args parse"),
            json!({"city": "Paris", "units": "c"})
        );
        assert_eq!(p.content, None, "the whole generation is the call");
    }

    // ---------------------------------------------------- dispatch --

    /// feed a whole string; true = every byte was legal
    fn dfeed(m: &mut DispatchMachine, s: &str) -> bool {
        s.bytes().all(|b| m.feed(b))
    }

    fn dispatch(single: bool) -> DispatchMachine {
        DispatchMachine::new(tool_set(), single)
    }

    /// The free phase belongs to the engine: before the gate
    /// arms, `may_stop` must not veto stop ids. It did (`active &&`), which
    /// masked the stop ids out of the nucleus for the whole thinking region
    /// and silently reweighted every draw where one carried mass - measured
    /// as 3/8 replay seeds degenerating into parallel-call spam with the
    /// grammar armed against 0/40 without it. After arming, the machine owns
    /// the answer, as before.
    #[test]
    fn gated_may_stop_is_free_before_activation() {
        let vocab = Arc::new(VocabBytes {
            data: b"a".to_vec(),
            offs: vec![0, 1],
            special: vec![false],
        });
        let mk = || {
            GatedConstraint::new(
                vocab.clone(),
                Gate::AfterToken(7),
                Machine::Dispatch(DispatchMachine::new(tool_set(), false)),
                Vec::new(),
            )
        };
        let g = mk();
        assert!(
            g.may_stop(),
            "pre-activation: stop tokens are the engine's call"
        );
        let mut g = mk();
        g.accept(7); // the gate token arms the grammar
        assert!(
            g.may_stop(),
            "armed, outside a call: prose may end the turn"
        );
    }

    /// Ten one-byte tokens; ids 5 and 9 double as the exit phrase and the
    /// think-close marker in the budget tests below.
    fn budget_vocab() -> Arc<VocabBytes> {
        Arc::new(VocabBytes {
            data: b"a{cdefghij".to_vec(),
            offs: (0..=10).collect(),
            special: vec![false; 10],
        })
    }

    /// The core lifecycle: free until the budget is spent, then the exit
    /// sequence is the only legal path (stops masked), then transparent.
    #[test]
    fn budget_forces_the_exit_sequence() {
        let mut b = BudgetGated::new(budget_vocab(), None, 3, 9, vec![5, 9], None);
        for _ in 0..3 {
            assert!(b.allows(0) && b.allows(5), "counting phase is free");
            assert!(b.may_stop(), "counting phase: stops are the engine's call");
            b.accept(0);
        }
        assert!(
            b.allows(5) && !b.allows(0),
            "budget spent: only the exit id"
        );
        assert!(
            !b.may_stop(),
            "the injection is atomic - no stopping inside it"
        );
        b.accept(5);
        assert!(b.allows(9) && !b.allows(5), "second forced id");
        b.accept(9);
        assert!(b.allows(0) && b.may_stop(), "closed: transparent again");
    }

    /// A model that closes its think block on its own never meets the budget.
    #[test]
    fn natural_close_disarms_the_budget() {
        let mut b = BudgetGated::new(budget_vocab(), None, 2, 9, vec![5, 9], None);
        b.accept(0);
        b.accept(9); // the model closed the block itself
        for _ in 0..10 {
            assert!(b.allows(0), "disarmed: free past the budget");
            b.accept(0);
        }
        assert!(b.may_stop());
    }

    /// free_now() is what lets spec rounds run through constrained slots:
    /// true exactly where allows() admits everything (thinking, prose between
    /// calls), false the moment a machine owns the stream.
    #[test]
    fn free_now_tracks_the_free_phases() {
        // gated dispatch: free before the gate, free between calls, owned in one
        let mut g = GatedConstraint::new(
            budget_vocab(),
            Gate::AfterToken(9),
            Machine::Dispatch(DispatchMachine::new(tool_set(), false)),
            Vec::new(),
        );
        assert!(g.free_now(), "pre-activation reasoning is free");
        g.accept(9);
        assert!(g.free_now(), "armed dispatcher outside a call is free");

        // forced tool: the machine owns the stream from token 0
        let f = GatedConstraint::new(
            budget_vocab(),
            Gate::Immediate,
            Machine::Tool(ToolMachine::forced(tool_set())),
            Vec::new(),
        );
        assert!(!f.free_now(), "a forced call is never free");

        // budget: counting free, forcing not, done delegates
        let mut b = BudgetGated::new(budget_vocab(), None, 2, 9, vec![5, 9], None);
        assert!(b.free_now(), "counting is free");
        b.accept(0);
        b.accept(0); // budget met -> Forcing
        assert!(!b.free_now(), "the injection is never speculated through");
        b.accept(5);
        b.accept(9); // exit done
        assert!(b.free_now(), "closed budget with no grammar is free");
    }

    /// The budget met inside an open tool call defers the injection to the
    /// call's close: qwen writes calls inside a still-open think block, and
    /// severing one mid-JSON hands the model corrupted context to re-derive
    /// from. Seen live.
    #[test]
    fn budget_defers_inside_an_open_tool_call() {
        // one-byte tokens: 0='<', 1='t', 2='>', 3='/', 4='a'
        let vocab = Arc::new(VocabBytes {
            data: b"<t>/a".to_vec(),
            offs: (0..=5).collect(),
            special: vec![false; 5],
        });
        // markers "<t>" / "</t>" in this tiny alphabet
        let mk = |budget| {
            BudgetGated::new(
                vocab.clone(),
                None,
                budget,
                4, // 'a' stands in for </think>
                vec![4],
                Some((b"<t>".to_vec(), b"</t>".to_vec())),
            )
        };
        let mut b = mk(3);
        for id in [0u32, 1, 2] {
            b.accept(id); // "<t>" - the call opens exactly as budget (3) is met
        }
        assert!(b.allows(1), "mid-call: still free, injection deferred");
        b.accept(1); // call body
        for id in [0u32, 3, 1, 2] {
            assert!(b.allows(id), "closing marker tokens flow free");
            b.accept(id); // "</t>" - the call closes
        }
        assert!(
            !b.allows(1) && b.allows(4),
            "call closed: the exit forces now"
        );

        // and the deferral is BOUNDED - a call that never closes still exits
        let mut b = mk(3);
        b.accept(0);
        b.accept(1);
        b.accept(2); // open, budget met at the marker's last byte
        for _ in 0..CALL_DEFER_CAP {
            assert!(b.allows(1));
            b.accept(1);
        }
        assert!(
            !b.allows(1) && b.allows(4),
            "defer cap reached: forced anyway"
        );
    }

    /// The forced close is how the inner grammar's gate learns the think
    /// block ended: after the injection, the tool machine owns the stream
    /// exactly as if the model had sampled `</think>` itself.
    #[test]
    fn budget_hands_over_to_the_inner_grammar() {
        let inner = GatedConstraint::new(
            budget_vocab(),
            Gate::AfterToken(9),
            Machine::Tool(ToolMachine::forced(tool_set())),
            Vec::new(),
        );
        let mut b = BudgetGated::new(budget_vocab(), Some(inner), 2, 9, vec![5, 9], None);
        assert!(b.allows(0), "reasoning free: budget and inner both defer");
        b.accept(0);
        b.accept(0);
        assert!(!b.allows(0) && b.allows(5), "budget spent mid-think");
        b.accept(5);
        b.accept(9); // the forced close arms the inner gate
        assert!(
            !b.allows(0),
            "the forced tool grammar owns the stream now - 'a' is not how a call starts"
        );
        assert!(
            !b.may_stop(),
            "a forced call has not run yet: the turn may not end"
        );
    }

    /// The shape a `tool_choice: "auto"` turn actually has: prose, a call the
    /// grammar owns, then prose again. The one-shot `Gate` could not spell it.
    #[test]
    fn dispatch_constrains_only_the_call() {
        let mut m = dispatch(false);
        assert!(
            dfeed(&mut m, "Let me look that up for you.\n"),
            "prose is free"
        );
        assert!(m.may_stop(), "the turn may end in prose");
        assert!(dfeed(&mut m, "<tool_call>"), "the trigger arms the grammar");
        assert!(!m.may_stop(), "the turn may NOT end mid-call");
        assert!(
            !m.clone().feed(b'{'),
            "qwen's syntax has no JSON here - the grammar is live now"
        );
        assert!(dfeed(
            &mut m,
            "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>"
        ));
        assert!(m.may_stop(), "the call closed, the turn may end again");
        assert!(dfeed(&mut m, " Got it - 4 degrees."), "prose is free again");
    }

    /// The name is a literal alternation, so an invented tool cannot be
    /// sampled. No post-hoc repair layer can do this: it has no way to know
    /// which of the declared tools a hallucinated name meant.
    #[test]
    fn dispatch_makes_a_hallucinated_tool_name_unrepresentable() {
        let mut m = dispatch(false);
        assert!(dfeed(&mut m, "<tool_call>\n<function=get_"));
        assert!(!m.feed(b'f'), "only get_weather is declared");
    }

    #[test]
    fn dispatch_rearms_for_a_second_call() {
        let call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";
        let mut m = dispatch(false);
        assert!(dfeed(&mut m, call));
        assert!(dfeed(&mut m, "\n"));
        assert!(
            dfeed(&mut m, call),
            "a parallel second call is still constrained"
        );
        assert!(m.may_stop());
    }

    /// `parallel_tool_calls: false` is enforced by disarming, not by dropping
    /// extra calls after the fact - so the model never spends tokens on a call
    /// the API would throw away.
    #[test]
    fn dispatch_single_disarms_after_the_first_call() {
        let mut m = dispatch(true);
        assert!(dfeed(
            &mut m,
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>"
        ));
        assert!(
            dfeed(&mut m, "<tool_call>\n<function=nonsense>{{{"),
            "the trigger is spent: this is ordinary text now, not a grammar error"
        );
        assert!(m.may_stop());
    }

    /// A trigger that never completes must leave the text free - the model
    /// talking *about* `<tool_ca...` is not a call.
    #[test]
    fn dispatch_partial_trigger_stays_free() {
        let mut m = dispatch(false);
        assert!(dfeed(&mut m, "the <tool_ca tag, and <tool_call"));
        assert!(m.may_stop(), "still outside a call");
        assert!(dfeed(&mut m, " is how you write it"));
        assert!(m.may_stop());
    }

    /// Same machinery on the JSON syntax, where the arguments object runs
    /// through the schema - malformed JSON is unsamplable, not repaired.
    #[test]
    fn dispatch_over_the_json_syntax() {
        let mut m = DispatchMachine::new(json_tool_set(), false);
        assert!(dfeed(
            &mut m,
            "I'll check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": "
        ));
        assert!(!m.clone().feed(b'3'), "city is declared a string");
        assert!(dfeed(&mut m, "\"Oslo\"}}\n</tool_call>"));
        assert!(m.may_stop());
    }

    // --------------------------------- the unwrapped opener  --

    /// The granite-vision failure, caught at generation time: the model opens
    /// its whole answer with a bare `{"name": "` and the same grammar arms, so
    /// what comes out is a call rather than a JSON blob in the chat.
    ///
    /// Note the tail - a bare call closes on its own `}`, with no
    /// `</tool_call>` to close, because there is no opener to match it.
    #[test]
    fn bare_opener_arms_at_turn_start() {
        let mut m = DispatchMachine::new(json_tool_set(), false);
        assert!(dfeed(
            &mut m,
            "{\"name\": \"get_weather\", \"arguments\": {"
        ));
        assert!(!m.may_stop(), "inside a call the turn cannot end");
        assert!(
            !m.clone().feed(b'z'),
            "arguments run through the schema once armed"
        );
        assert!(dfeed(&mut m, "\"city\": \"Oslo\"}}"));
        assert!(m.may_stop());
    }

    /// Leading whitespace is still turn start - a model that opens with a
    /// newline has not said anything yet.
    #[test]
    fn bare_opener_survives_leading_whitespace() {
        let mut m = DispatchMachine::new(json_tool_set(), false);
        assert!(dfeed(
            &mut m,
            "\n  {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}"
        ));
        assert!(m.may_stop());
    }

    /// The guardrail. Prose first means the object is being written about, not
    /// emitted - the one read the bare trigger must not make. Committing here
    /// would turn a model explaining a tool into a model calling it.
    #[test]
    fn bare_opener_is_dead_after_prose() {
        let mut m = DispatchMachine::new(json_tool_set(), false);
        assert!(dfeed(
            &mut m,
            "To call it you would send {\"name\": \"get_weather\", \"arguments\": {\"city\": 3"
        ));
        assert!(
            m.may_stop(),
            "never armed, so the object is ordinary content - note `city: 3` \
             would be illegal under the schema and is fine here"
        );
    }

    /// A false start closes the door too: `{x` is not the opener, so the turn
    /// has said something and a later bare object is content.
    #[test]
    fn bare_opener_dies_on_a_mismatched_brace() {
        let mut m = DispatchMachine::new(json_tool_set(), false);
        assert!(dfeed(&mut m, "{x} "));
        assert!(dfeed(
            &mut m,
            "{\"name\": \"get_weather\", \"arguments\": {\"city\": 3}}"
        ));
        assert!(m.may_stop(), "no arming, so no schema");
    }

    /// After a wrapped call the turn is under way, so a following bare object
    /// gets no second opener - only the unambiguous tag can arm again.
    #[test]
    fn bare_opener_does_not_re_arm_mid_turn() {
        let mut m = DispatchMachine::new(json_tool_set(), false);
        assert!(dfeed(
            &mut m,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>"
        ));
        assert!(m.may_stop());
        assert!(dfeed(
            &mut m,
            "{\"name\": \"get_weather\", \"arguments\": {\"city\": 3}}"
        ));
        assert!(m.may_stop(), "content, not a second armed call");
    }

    /// The wrapped opener is unaffected: it arms anywhere, prose or not.
    #[test]
    fn wrapped_opener_still_arms_after_prose() {
        let mut m = DispatchMachine::new(json_tool_set(), false);
        assert!(dfeed(
            &mut m,
            "Let me look.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {"
        ));
        assert!(!m.may_stop(), "armed mid-turn, as it always has");
    }

    /// Only the JSON syntax has a bare form; the XML dialects carry the tool
    /// name inside markup that is the call, so there is nothing to catch.
    #[test]
    fn only_the_json_syntax_has_a_bare_opener() {
        assert!(ToolSyntax::Json.bare_trigger().is_some());
        for s in [
            ToolSyntax::QwenXml,
            ToolSyntax::LagunaXml,
            ToolSyntax::AtemXml,
        ] {
            assert!(s.bare_trigger().is_none(), "{s:?} must not arm bare");
            assert!(ToolMachine::bare(tool_set_for(s)).is_none());
        }
    }

    /// The dispatcher replays the trigger into the fresh machine, so the bare
    /// trigger has to be a prefix of every unwrapped candidate - the same
    /// invariant `atem_dispatch_arms_on_the_atem_opener` guards for the tag.
    #[test]
    fn bare_trigger_is_a_prefix_of_every_bare_opener() {
        let set = json_tool_set();
        let trigger = ToolSyntax::Json.bare_trigger().expect("json arms bare");
        let mut m = ToolMachine::bare(set).expect("json builds a bare machine");
        assert!(
            trigger.iter().all(|&b| m.feed(b)),
            "every bare candidate must start with the bare trigger"
        );
    }

    // ------------------------------------- muse-glimmer / ATEM  --

    fn atem_tool() -> Value {
        json!({"type": "function", "function": {
            "name": "weather.get",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "units": {"type": "string", "enum": ["c", "f"]}
            }, "required": ["city"]}
        }})
    }

    fn atem_set() -> Arc<ToolSet> {
        ToolSet::compile(ToolSyntax::AtemXml, &[atem_tool()], None).expect("compile tools")
    }

    /// The call body, entered at the dispatch trigger - and typed on the way
    /// through: `days` is a number, `units` is one of two literals.
    #[test]
    fn atem_grammar_matches_the_template_shape() {
        let mut m = ToolMachine::new(atem_set());
        let text = "<atem:function_calls>\n<atem:invoke name=\"weather.get\">\n\
                    <atem:parameter name=\"city\">Paris</atem:parameter>\n\
                    <atem:parameter name=\"days\">3</atem:parameter>\n\
                    </atem:invoke>\n</atem:function_calls>";
        assert!(
            tool_feed(&mut m, text),
            "the template's own spelling must be legal"
        );
        assert!(m.may_stop());
    }

    /// A forced call owns the turn from token 0 - but this model always thinks
    /// first, so the analysis message has to be inside the grammar, repeatable,
    /// and unable to end the turn on its own.
    #[test]
    fn atem_forced_grammar_lets_the_model_think_first() {
        let mut m = ToolMachine::forced(atem_set());
        assert!(tool_feed(
            &mut m,
            " to=self<|message|>I should look this up."
        ));
        assert!(
            !m.may_stop(),
            "a forced call cannot be satisfied by a thought"
        );
        assert!(tool_feed(&mut m, "<|eom|>"));
        // a second thought is allowed, and it introduces itself this time
        assert!(tool_feed(
            &mut m,
            "<|start|>assistant to=self<|message|>Paris then.<|eom|>"
        ));
        assert!(!m.may_stop());
        assert!(tool_feed(
            &mut m,
            "<|start|>assistant to=weather.get<|message|><atem:function_calls>\n\
             <atem:invoke name=\"weather.get\">\n\
             <atem:parameter name=\"city\">Paris</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls>"
        ));
        assert!(m.may_stop(), "the call closed");
    }

    /// The first header rides bare (the prompt already wrote
    /// `<|start|>assistant`); a machine that demanded one would deadlock the
    /// very first token.
    #[test]
    fn atem_forced_grammar_calls_without_thinking_at_all() {
        let mut m = ToolMachine::forced(atem_set());
        assert!(
            !m.clone().feed(b'<'),
            "the first message introduces itself in the PROMPT"
        );
        assert!(tool_feed(
            &mut m,
            " to=weather.get<|message|><atem:function_calls>\n\
             <atem:invoke name=\"weather.get\">\n\
             <atem:parameter name=\"city\">Oslo</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls>"
        ));
        assert!(m.may_stop());
    }

    #[test]
    fn atem_grammar_types_values_the_way_render_atem_writes_them() {
        // integer: bare JSON, no quotes
        let mut m = ToolMachine::new(atem_set());
        assert!(tool_feed(
            &mut m,
            "<atem:function_calls>\n<atem:invoke name=\"weather.get\">\n\
             <atem:parameter name=\"days\">"
        ));
        assert!(!m.clone().feed(b'"'), "an integer is not quoted");
        assert!(tool_feed(&mut m, "3</atem:parameter>\n"));

        // enum string: bare, and only the declared variants
        let mut m2 = m.clone();
        assert!(tool_feed(&mut m2, "<atem:parameter name=\"units\">"));
        assert!(!m2.clone().feed(b'x'), "units is c or f");
        assert!(tool_feed(&mut m2, "c</atem:parameter>\n"));

        // `city` is required, so the call cannot close without it: the close
        // literal is not even a candidate at this boundary
        assert!(!tool_feed(
            &mut m.clone(),
            "</atem:invoke>\n</atem:function_calls>"
        ));
    }

    #[test]
    fn atem_grammar_rejects_undeclared_and_duplicate_params() {
        let mut m = ToolMachine::new(atem_set());
        assert!(tool_feed(
            &mut m,
            "<atem:function_calls>\n<atem:invoke name=\"weather.get\">\n\
             <atem:parameter name=\"city\">Paris</atem:parameter>\n"
        ));
        let mut dup = m.clone();
        assert!(tool_feed(&mut dup, "<atem:parameter name=\""));
        assert!(!dup.clone().feed(b'c'), "city was already emitted");
        assert!(!dup.feed(b'z'), "no parameter named z");
    }

    /// A tool named `self` would make its call unreadable - the analysis
    /// branch and the call branch would share a full literal. Refuse, loudly.
    #[test]
    fn atem_refuses_a_tool_that_shadows_a_channel_address() {
        for name in ["self", "user"] {
            let t = json!({"type": "function", "function": {"name": name, "parameters": {}}});
            let e = ToolSet::compile(ToolSyntax::AtemXml, &[t], None)
                .err()
                .unwrap_or_else(|| panic!("{name} must be refused"));
            assert!(e.contains("channel address"), "{name}: {e}");
        }
        // the same names are ordinary tools on every other syntax
        let t = json!({"type": "function", "function": {"name": "self", "parameters": {}}});
        assert!(ToolSet::compile(ToolSyntax::QwenXml, &[t], None).is_ok());
    }

    /// Whatever the grammar emits, `muse::parse` has to read back as a call -
    /// that agreement is the whole contract between the two.
    #[test]
    fn atem_grammar_output_round_trips_through_the_parser() {
        let mut m = ToolMachine::forced(atem_set());
        let text = " to=self<|message|>need the weather<|eom|>\
                    <|start|>assistant to=weather.get<|message|>\
                    <atem:function_calls>\n<atem:invoke name=\"weather.get\">\n\
                    <atem:parameter name=\"city\">Paris</atem:parameter>\n\
                    <atem:parameter name=\"days\">3</atem:parameter>\n\
                    <atem:parameter name=\"units\">c</atem:parameter>\n\
                    </atem:invoke>\n</atem:function_calls>";
        assert!(tool_feed(&mut m, text));
        assert!(m.may_stop());
        let hints = crate::parsers::tool_hints(Some(&[atem_tool()]));
        let p = crate::parsers::parse(
            crate::parsers::Dialect::MuseChannel,
            text,
            false,
            hints.as_ref(),
        );
        assert_eq!(p.tool_calls.len(), 1, "{p:?}");
        assert_eq!(p.complete_calls, 1);
        assert_eq!(p.tool_calls[0].name, "weather.get");
        assert_eq!(
            serde_json::from_str::<Value>(&p.tool_calls[0].arguments).expect("args parse"),
            json!({"city": "Paris", "days": 3, "units": "c"})
        );
        assert_eq!(p.reasoning.as_deref(), Some("need the weather"));
        assert_eq!(p.content, None, "the markup is a call, not text");
    }

    /// The dispatcher's trigger has to be a prefix of every opener `new`
    /// builds - otherwise arming hands the fresh machine bytes it rejects and
    /// the request decodes unconstrained (or worse, deadlocks).
    #[test]
    fn atem_dispatch_arms_on_the_atem_opener() {
        let mut m = DispatchMachine::new(atem_set(), false);
        assert!(m.may_stop(), "free text before any call");
        assert!(dfeed(&mut m, " to=self<|message|>thinking out loud<|eom|>"));
        assert!(m.may_stop(), "a thought is not a call");
        assert!(dfeed(
            &mut m,
            "<|start|>assistant to=weather.get<|message|>"
        ));
        assert!(m.may_stop(), "the header alone still commits to nothing");
        assert!(dfeed(&mut m, "<atem:function_calls>\n<atem:invoke name=\""));
        assert!(!m.may_stop(), "inside a call the turn may not end");
        assert!(
            !m.clone().feed(b'z'),
            "a hallucinated tool name is unrepresentable"
        );
        assert!(dfeed(
            &mut m,
            "weather.get\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls>"
        ));
        assert!(m.may_stop(), "and it releases for the next message");
    }
}
