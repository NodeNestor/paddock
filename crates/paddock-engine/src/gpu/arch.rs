//! Validated-arch allowlist - "supported" is a TESTED claim, not a compile
//! flag.
//!
//! Paddock is a specialized engine: a compute generation is served only after
//! its bring-up is complete (kernels tuned on the real die, parity-gated,
//! throughput measured). Everything else gets an honest refusal at
//! startup - never a half-serve at unknown performance. CUDA's own rules make
//! the half-serve the default failure mode without this gate: plain sm_120
//! SASS forward-loads onto any 12.x minor (so a GB10 / DGX Spark "works"
//! until the first `sm_120a`-only tensor-core lane launches and dies
//! mid-request), and preflight's trial launch only proves the BASELINE image.
//! An exact-(major,minor) allowlist is the only honest check.
//!
//! Lifecycle per generation: unknown -> refused; in bring-up -> serves only
//! under `PADDOCK_UNVALIDATED_ARCH=1`, stamped UNVALIDATED in every log so a
//! number measured mid-bring-up can never masquerade as a supported result;
//! validated -> listed below, with the campaign that closed it.

// The lists themselves are DATA, not code: `gpu-support.toml` at the repo
// root, parsed once by paddock-models and read here and by the manager alike.
// They used to be consts in this file, which meant the same fact also lived
// in the manager and in the Studio's prose - and all three managed to
// disagree at once. One file, several readers.
use paddock_models::gpu_support::{self, Status};

/// Capabilities whose bring-up campaign has closed, as `(major, minor, why)`.
fn validated() -> Vec<(u32, u32, &'static str)> {
    rows(Status::Supported)
}

/// Capabilities with an open campaign - named in the refusal so the state is
/// visible, and expected to run under the override meanwhile.
fn in_bring_up() -> Vec<(u32, u32, &'static str)> {
    rows(Status::Bringup)
}

fn rows(want: Status) -> Vec<(u32, u32, &'static str)> {
    gpu_support::ALL
        .iter()
        .filter(|a| a.status == want)
        .map(|a| (a.cc.0, a.cc.1, a.campaign.unwrap_or(a.name)))
        .collect()
}

pub(super) enum Gate {
    /// Campaign closed - serve normally.
    Validated,
    /// Unvalidated silicon, operator override - serve, WARN with the stamp.
    Overridden(String),
    /// Unvalidated silicon, no override - refuse with the full sentence.
    Refused(String),
}

/// Pure decision (env read stays at the call site so this is testable).
pub(super) fn gate(cc: (u32, u32), device: &str, override_set: bool) -> Gate {
    gate_in(cc, device, override_set, &validated(), &in_bring_up())
}

/// The decision against GIVEN lists, so the bring-up branch stays under test
/// while `IN_BRING_UP` is empty. A branch nobody exercises is a branch that
/// rots, and this one only wakes up when a new generation opens - exactly
/// when it is least convenient to discover it stopped working.
fn gate_in(
    cc: (u32, u32),
    device: &str,
    override_set: bool,
    validated: &[(u32, u32, &str)],
    in_bring_up: &[(u32, u32, &str)],
) -> Gate {
    let (maj, min) = cc;
    if validated.iter().any(|&(a, b, _)| (a, b) == (maj, min)) {
        return Gate::Validated;
    }
    if override_set {
        return Gate::Overridden(format!(
            "SERVING ON UNVALIDATED ARCH sm_{maj}{min} ({device}) - \
             PADDOCK_UNVALIDATED_ARCH override is set. Numbers from this \
             machine are bring-up data, NOT supported results; label them so."
        ));
    }
    let validated_list = validated
        .iter()
        .map(|&(a, b, _)| format!("sm_{a}{b}"))
        .collect::<Vec<_>>()
        .join(", ");
    let bring_up = in_bring_up
        .iter()
        .find(|&&(a, b, _)| (a, b) == (maj, min))
        .map(|&(.., note)| format!(" This generation's bring-up is IN PROGRESS ({note})."))
        .unwrap_or_default();
    Gate::Refused(format!(
        "{device} is sm_{maj}{min}, which this engine build has not validated \
         (validated: {validated_list}).{bring_up} Paddock serves a compute \
         generation only after its bring-up validation is complete - an unvalidated \
         die would serve at unknown performance or fail mid-request, and an \
         honest \"not yet\" beats both. Set PADDOCK_UNVALIDATED_ARCH=1 to \
         serve anyway for bring-up/testing; every log is then stamped \
         UNVALIDATED."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validated_dies_serve() {
        assert!(matches!(gate((8, 6), "A6000", false), Gate::Validated));
        assert!(matches!(
            gate((12, 0), "RTX PRO 6000", false),
            Gate::Validated
        ));
        // sm_100 joined when its campaign closed - before that it sat in
        // IN_BRING_UP, serving only under the override.
        assert!(matches!(gate((10, 0), "B200", false), Gate::Validated));
    }

    /// The GB10 / DGX Spark case: same major as the validated consumer die,
    /// different minor - exact matching must refuse it (plain-SASS forward
    /// compatibility is precisely what made it half-serve before this gate).
    #[test]
    fn same_major_different_minor_is_refused() {
        let Gate::Refused(msg) = gate((12, 1), "GB10", false) else {
            panic!("sm_121 must be refused without the override");
        };
        assert!(msg.contains("sm_121"));
        assert!(msg.contains("PADDOCK_UNVALIDATED_ARCH"));
    }

    /// A generation with an open campaign: refused by default with the
    /// campaign named, served with the stamp under the override.
    ///
    /// Driven through a synthetic list because `IN_BRING_UP` is empty today -
    /// sm_100 was its last occupant and its campaign has closed. The behaviour
    /// has to keep working for whatever opens next, and an untested branch
    /// would not.
    #[test]
    fn bring_up_arch_names_its_campaign_and_overrides_with_stamp() {
        const NEXT: &[(u32, u32, &str)] = &[(13, 0, "Rubin - campaign open")];
        let Gate::Refused(msg) = gate_in((13, 0), "Rubin", false, &validated(), NEXT) else {
            panic!("an open campaign must still be refused by default");
        };
        assert!(msg.contains("IN PROGRESS"), "{msg}");
        let Gate::Overridden(warn) = gate_in((13, 0), "Rubin", true, &validated(), NEXT) else {
            panic!("override must serve");
        };
        assert!(warn.contains("UNVALIDATED"), "{warn}");
    }

    /// A future major (Rubin-class) gets the same honest refusal - no PTX
    /// limp mode, no cryptic driver error.
    #[test]
    fn future_major_is_refused() {
        assert!(matches!(gate((13, 0), "Rubin", false), Gate::Refused(_)));
    }
}
