//! The TLS identity has to build **in the manager's feature set**, not
//! just in its own crate's.
//!
//! `paddock-tls` has unit tests that build an `Identity`, and they passed while
//! the shipped binary panicked on startup. Cargo unifies features
//! across the graph: paddock-tls alone enables only rustls' `aws-lc-rs`, but
//! the manager also links reqwest, which brings `ring` - and rustls, offered
//! two providers and no instruction, panics at the first builder call rather
//! than picking one. A crash, not a fallback, and invisible to every test that
//! runs in the smaller feature set.
//!
//! So this lives here, in a crate whose dependency graph is the real one.

#[test]
fn the_identity_builds_under_the_managers_feature_unification() {
    let dir = std::env::temp_dir().join(format!("pd-mgr-tls-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();

    let id = paddock_tls::Identity::load_or_create(&dir)
        .expect("the manager must be able to establish an identity");

    assert!(id.root_pem.starts_with("-----BEGIN CERTIFICATE-----"));
    // A sha256 as colon-separated hex: 32 bytes, 31 separators.
    assert_eq!(id.fingerprint.len(), 95, "fingerprint: {}", id.fingerprint);
    assert!(id.names.iter().any(|n| n == "localhost"));
    // Reaching here at all is the point: `server` is the field whose
    // construction paniced.
    assert!(!id.server.alpn_protocols.is_empty());

    std::fs::remove_dir_all(&dir).ok();
}
