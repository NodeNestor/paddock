//! Can this computer run models on its own? - asked once at startup, so the
//! product can say so before anybody tries.
//!
//! The engine already refuses honestly, but it refuses inside a runner at
//! model load, which means a machine that can never serve looks completely
//! normal until you pick a model and press start. Somebody copied paddock to a
//! box with no usable card, opened the Manager, and nothing anywhere said a
//! word. This is the answer to that.
//!
//! It PUBLISHES FACTS, not SENTENCES. What we found, what this build needs,
//! and which OS is asking - the words live in the Studio, where they can be
//! written and rewritten without a rebuild, and where the rule that the
//! Manager speaks no jargon is enforceable by reading it.

use paddock_models::gpu_support::{self, Arch};
use serde::Serialize;

/// The CUDA version this build's kernels are compiled against. A driver older
/// than this cannot load them, and that is a fixable problem - which is the
/// whole reason it gets its own state rather than being lumped in with "no
/// card".
const CUDA_NEEDED: (u32, u32) = (13, 0);

/// What the probe found, in the order of what the user can do about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum State {
    /// A card we have finished validating. Serve, and say nothing further.
    Ready,
    /// An NVIDIA card we can see and have not finished testing. It may well
    /// work; we will not claim it does.
    Untested,
    /// An NVIDIA card, and a driver too old to load this build's kernels.
    /// The one state with a fix the user can apply in five minutes.
    DriverTooOld,
    // There was a NeedsSetup state here, for "a card we serve, a driver new
    // enough, and the engine's CUDA libraries are not installed yet". It is
    // gone with the setup that produced it: paddock ships no
    // NVIDIA redistributable and fetches none, so a supported card with a
    // current driver is simply Ready. Should a required library ever return,
    // the no-bundling rule still applies - fetch
    // hardware-matched, never bundle - but it comes back as new code, not as a
    // variant nothing could reach.
    /// No NVIDIA card, or no NVIDIA driver - from here the two are the same
    /// question ("is there anything to talk to"), and the guidance is the same
    /// either way, so the copy covers both rather than guessing between them.
    NoCard,
}

/// One generation we serve, named the way a person shops for it.
#[derive(Debug, Clone, Serialize)]
pub struct SupportedGen {
    /// "Ampere"
    pub name: String,
    /// the cards, NVIDIA's own names
    pub cards: Vec<String>,
}

/// One CARD, for the sheet a person searches to find their own.
///
/// A row per card rather than per generation because the question being asked
/// is "will mine work" - and nobody knows which generation their card is. The
/// generation rides along as context, never as the key.
#[derive(Debug, Clone, Serialize)]
pub struct CardRow {
    /// NVIDIA's own name, e.g. "NVIDIA RTX A6000"
    pub card: &'static str,
    /// "Ampere"
    pub generation: &'static str,
    /// "workstation" | "datacenter" | "jetson" - the Studio words them.
    pub kind: &'static str,
    /// "supported" | "testing" | "planned" | "too-old"
    pub status: &'static str,
}

/// Every card this build knows about, in table order.
///
/// Honest about all four states, not just the good one: a person whose card is
/// too old is owed that answer as plainly as one whose card works, and a
/// generation still being brought up is worth saying out loud, not omitting.
pub fn card_sheet() -> Vec<CardRow> {
    let mut rows = Vec::new();
    for a in gpu_support::ALL {
        let status = match a.status {
            gpu_support::Status::Supported => "supported",
            gpu_support::Status::Bringup => "testing",
            gpu_support::Status::Built => "planned",
            gpu_support::Status::TooOld => "too-old",
        };
        let groups: [(&'static str, &[&'static str]); 3] = [
            ("workstation", a.workstation),
            ("datacenter", a.datacenter),
            ("jetson", a.jetson),
        ];
        for (kind, cards) in groups {
            for card in cards {
                rows.push(CardRow {
                    card,
                    generation: a.name,
                    kind,
                    status,
                });
            }
        }
    }
    rows
}

/// Everything the Studio needs to say something true and act on it.
#[derive(Debug, Clone, Serialize)]
pub struct Readiness {
    pub state: State,
    /// The card we looked at, as the driver names it ("NVIDIA RTX A6000").
    /// Present whenever there was one to look at, including when it is too
    /// old or untested - being told which card was rejected is most of the
    /// value of being told.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub card: Option<String>,
    /// Its generation ("Ampere"), when we recognise the silicon at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<String>,
    /// Its compute capability as `[major, minor]`. The generation NAME is what
    /// a person shops by and stays the headline; the numbers are what code
    /// compares against, and some artifacts carry a floor (`min_cc`) - NVFP4's
    /// W4A16 kernels are sm_120a-only, so the Studio needs to know 8.6 from
    /// 12.0 to grey that choice out instead of selling a download that would
    /// quietly serve the base build instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cc: Option<[u32; 2]>,
    /// The driver's own version string, for the "yours is X" half of an
    /// update prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
    /// The CUDA version that driver speaks, and the one we need.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda: Option<String>,
    pub cuda_needed: String,
    /// Which OS is asking - updating a driver is a different errand on
    /// Windows than on Linux, and guidance for the wrong one is worse than
    /// none.
    pub os: &'static str,
    /// Every card this build serves, for the sheet. Always present: a user
    /// deciding what to buy needs it most when their own machine cannot run
    /// anything.
    pub supported: Vec<SupportedGen>,
}

impl Readiness {
    /// Whether local serving is possible at all - what the Studio gates its
    /// GPU instruments and its local-model vocabulary on.
    ///
    /// The boot verdict is the current one: hardware does not change under a
    /// running manager. There used to be a `current()` beside this that
    /// re-stat'd the CUDA libraries, because we fetched those ourselves and a
    /// verdict frozen at boot would have left the Studio on a progress bar
    /// forever after a successful setup. Nothing is fetched any more, so
    /// nothing can change, so the refresh is gone with it.
    pub fn can_serve(&self) -> bool {
        self.state == State::Ready
    }
}

fn os_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    }
}

fn supported_gens() -> Vec<SupportedGen> {
    gpu_support::supported()
        .map(|a: &Arch| SupportedGen {
            name: a.name.to_owned(),
            cards: a.cards().map(str::to_owned).collect(),
        })
        .collect()
}

/// NVML packs a CUDA version as `major * 1000 + minor * 10`.
fn cuda_parts(v: i32) -> (u32, u32) {
    let v = v.max(0) as u32;
    (v / 1000, (v % 1000) / 10)
}

/// Look once. Never fails: "we could not tell" is an answer here, and it is
/// the same answer as "there is nothing to tell about".
pub fn probe() -> Readiness {
    let base = |state: State| Readiness {
        state,
        card: None,
        generation: None,
        cc: None,
        driver: None,
        cuda: None,
        cuda_needed: format!("{}.{}", CUDA_NEEDED.0, CUDA_NEEDED.1),
        os: os_name(),
        supported: supported_gens(),
    };

    // No NVML means no NVIDIA driver to ask. That is the same user-visible
    // situation as no NVIDIA card, and we deliberately do not guess which:
    // both answers lead to the same two options.
    let Ok(nvml) = crate::nvml::init() else {
        return base(State::NoCard);
    };
    let count = nvml.device_count().unwrap_or(0);
    if count == 0 {
        return base(State::NoCard);
    }

    let driver = nvml.sys_driver_version().ok();
    let cuda = nvml.sys_cuda_driver_version().ok().map(cuda_parts);

    // This one stops every card on the box, so it is answered before asking
    // which card it is - telling somebody their RTX 4090 is untested when the
    // real problem is a driver from 2023 sends them down the wrong road
    // entirely. Nothing we download fixes it either: CUDA 13 needs a driver
    // that speaks CUDA 13, whoever ships the libraries.
    if cuda.is_some_and(|v| v.0 < CUDA_NEEDED.0) {
        let (name, arch) = first_card(&nvml);
        return finish(base(State::DriverTooOld), name, arch, driver, cuda);
    }

    // Prefer a card we can actually serve on: a box with a supported card and
    // a spare old one is a supported box, and reporting the wrong device would
    // turn it into a refusal.
    let mut fallback: Option<(String, Option<&Arch>)> = None;
    for i in 0..count {
        let Ok(d) = nvml.device_by_index(i) else {
            continue;
        };
        let name = d.name().unwrap_or_default();
        let arch = d
            .cuda_compute_capability()
            .ok()
            .and_then(|cc| gpu_support::find((cc.major.max(0) as u32, cc.minor.max(0) as u32)));
        if arch.is_some_and(|a| a.status.serves()) {
            // A card we serve and a driver new enough is ready. There used to
            // be a library question here; paddock ships no NVIDIA
            // redistributable, so there is nothing left to be missing.
            return finish(base(State::Ready), name, arch, driver, cuda);
        }
        if fallback.is_none() {
            fallback = Some((name, arch));
        }
    }

    let (name, arch) = fallback.unwrap_or_default();
    finish(base(State::Untested), name, arch, driver, cuda)
}

/// Device 0, for the states that are about the machine rather than the card -
/// naming what you have is still most of the value of being told.
fn first_card(nvml: &nvml_wrapper::Nvml) -> (String, Option<&'static Arch>) {
    let Ok(d) = nvml.device_by_index(0) else {
        return (String::new(), None);
    };
    let arch = d
        .cuda_compute_capability()
        .ok()
        .and_then(|cc| gpu_support::find((cc.major.max(0) as u32, cc.minor.max(0) as u32)));
    (d.name().unwrap_or_default(), arch)
}

fn finish(
    mut r: Readiness,
    card: String,
    arch: Option<&Arch>,
    driver: Option<String>,
    cuda: Option<(u32, u32)>,
) -> Readiness {
    r.card = Some(card).filter(|s| !s.is_empty());
    r.generation = arch.map(|a| a.name.to_owned());
    // Only for silicon we recognise. An unknown card leaves this absent, and
    // absent must read as "no claim" rather than "too old" - the engine refuses
    // an unvalidated arch outright anyway, so a per-artifact floor is moot
    // there and greying one out would just be a second, wronger refusal.
    r.cc = arch.map(|a| [a.cc.0, a.cc.1]);
    r.driver = driver;
    r.cuda = cuda.map(|(a, b)| format!("{a}.{b}"));
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe always answers, and always carries the sheet - a machine
    /// that cannot serve is exactly where a user most needs to know what does.
    #[test]
    fn probing_always_answers_and_always_carries_the_supported_list() {
        let r = probe();
        assert!(!r.supported.is_empty(), "the sheet must never be empty");
        assert!(r.supported.iter().all(|g| !g.cards.is_empty()));
        assert_eq!(r.cuda_needed, "13.0");
        assert!(matches!(r.os, "windows" | "linux" | "macos"));
        // Whatever the machine is, the state and the card agree about whether
        // there was anything to look at.
        if r.state == State::NoCard {
            assert!(r.card.is_none());
        }
    }

    /// Whatever this box is, the verdict has to hang together: a state that
    /// implies a card must name one, and a card we serve must come back Ready
    /// rather than merely present. Runs everywhere and asserts only what is
    /// true everywhere - and prints the verdict under `--nocapture`, which is
    /// the cheapest way to ask a new machine what it thinks it is.
    #[test]
    fn the_verdict_hangs_together() {
        let r = probe();
        eprintln!(
            "readiness: {:?} card={:?} generation={:?} driver={:?} cuda={:?} os={}",
            r.state, r.card, r.generation, r.driver, r.cuda, r.os
        );
        match r.state {
            State::NoCard => assert!(r.card.is_none() && r.generation.is_none()),
            State::Ready => {
                assert!(r.card.is_some(), "ready without naming the card");
                assert!(
                    r.generation.is_some(),
                    "ready on silicon we do not recognise"
                );
            }
            State::Untested | State::DriverTooOld => {
                assert!(
                    r.card.is_some(),
                    "a machine-level problem must still name the card"
                )
            }
        }
    }

    /// Serving is gated on Ready and nothing else. This replaces a test that
    /// checked a finished CUDA setup flipped the verdict without a restart -
    /// there is no setup to finish any more, so the property worth pinning is
    /// the simpler one it degenerated into.
    #[test]
    fn only_a_ready_verdict_can_serve() {
        let base = Readiness {
            state: State::Ready,
            card: Some("NVIDIA RTX A6000".to_owned()),
            generation: Some("Ampere".to_owned()),
            cc: Some([8, 6]),
            driver: None,
            cuda: None,
            cuda_needed: "13.0".to_owned(),
            os: os_name(),
            supported: supported_gens(),
        };
        assert!(base.can_serve());
        for s in [State::Untested, State::DriverTooOld, State::NoCard] {
            let r = Readiness {
                state: s,
                ..base.clone()
            };
            assert!(!r.can_serve(), "{s:?} must not read as servable");
        }
    }

    /// The sheet is what a person searches for their own card, so every card
    /// has to be there exactly once, with an honest status - including the
    /// ones that will never work. Omitting those would answer "is mine
    /// supported?" with silence, which reads as "not listed yet".
    #[test]
    fn the_card_sheet_names_every_card_once_with_an_honest_status() {
        let sheet = card_sheet();
        assert!(sheet.len() > 30, "only {} cards", sheet.len());

        let mut seen = std::collections::HashSet::new();
        for r in &sheet {
            assert!(seen.insert(r.card), "{} listed twice", r.card);
            assert!(
                matches!(r.status, "supported" | "testing" | "planned" | "too-old"),
                "{r:?}"
            );
            assert!(
                matches!(r.kind, "workstation" | "datacenter" | "jetson"),
                "{r:?}"
            );
            assert!(!r.generation.is_empty());
            // the Manager's vocabulary rule, enforced on the widest surface
            assert!(
                !r.card.contains("sm_"),
                "card names must not carry sm_ numbers: {}",
                r.card
            );
        }

        // Not merely non-empty: the supported rows have to be exactly the
        // generations the engine serves, or the page would promise something
        // the engine refuses.
        let listed: std::collections::HashSet<&str> = sheet
            .iter()
            .filter(|r| r.status == "supported")
            .map(|r| r.generation)
            .collect();
        let serving: std::collections::HashSet<&str> =
            gpu_support::supported().map(|a| a.name).collect();
        assert_eq!(listed, serving);
    }

    #[test]
    fn nvml_packs_cuda_versions_as_major_thousands() {
        assert_eq!(cuda_parts(13000), (13, 0));
        assert_eq!(cuda_parts(12080), (12, 8));
        assert_eq!(cuda_parts(0), (0, 0));
    }

    /// Ampere is the generation this file was written on, so its presence in
    /// the sheet is the cheapest possible check that the data reached here.
    #[test]
    fn the_sheet_names_cards_not_capabilities() {
        let gens = supported_gens();
        let all: Vec<&str> = gens
            .iter()
            .flat_map(|g| g.cards.iter().map(String::as_str))
            .collect();
        assert!(all.contains(&"NVIDIA RTX A6000"), "{all:?}");
        // and never leaks the vocabulary the Manager is not allowed to use
        assert!(
            !all.iter().any(|c| c.contains("sm_")),
            "card names must not carry sm_ numbers"
        );
    }
}
