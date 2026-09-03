//! The box's own TLS identity, and a listener that speaks both schemes.
//!
//! # Why this exists
//!
//! Browsers gate a whole class of APIs on the origin being a **secure
//! context**: https, or the `localhost` / `127.0.0.1` carve-out. A LAN address
//! like `http://10.10.0.189:11500` is neither, and the gated APIs are not
//! degraded there - they are *absent*. Three of them the Studio needs:
//!
//! - `navigator.mediaDevices` - the whole object is `[SecureContext]`, so there
//!   is no microphone and no fallback that could produce one. Dictation, live
//!   transcription and the compare lanes stop existing off-box.
//! - `navigator.clipboard` - every copy button threw (shimmed in
//!   `studio/src/lib/clipboard.ts` with the deprecated `execCommand` path).
//! - `crypto.randomUUID` - every chat action threw and the Studio broke
//!   outright (shimmed in `studio/src/lib/uuid.ts`).
//!
//! Two of those bought a shim. The microphone cannot: no non-secure API hands
//! you an audio input. The only real fix is to stop serving the Studio over
//! plain http to anyone who is not sitting at the box.
//!
//! Encryption is the other half, and it was overdue on its own terms: a
//! non-loopback bind already auto-generates an API key, and then sent that key
//! - and every prompt, document and answer - in cleartext across the LAN.
//!
//! # What it does
//!
//! A **per-box CA** signs a **leaf** covering every name and address this box
//! answers to. Both live under `<data>/tls/`, generated on first run, renewed
//! without asking. No configuration: `paddock` comes up on https the first
//! time it is started, because a security property nobody switches on is one
//! that is off.
//!
//! A CA rather than a bare self-signed leaf, for two reasons:
//!
//! 1. The root can be installed once per client device and the warnings stop
//!    for good. A bare leaf can only ever be clicked through.
//! 2. The root is stable, so reissuing the leaf - a renewal, a new address -
//!    does not invalidate trust anyone has already granted.
//!
//! Not installing the root still works. The interstitial is ugly, but clicking
//! through it yields a real secure context, so the microphone comes back either
//! way. Install the root and it comes back without the ugliness.
//!
//! # One port
//!
//! [`serve`] sniffs the first byte of each connection: `0x16` is a TLS
//! handshake record, anything else is plain HTTP. So the same port keeps
//! answering `http://127.0.0.1:11500` for the CLI, the health check and the
//! Studio's own artifacts callback, while a browser on the LAN gets TLS - and
//! a LAN client that spoke http anyway is redirected rather than shown the
//! binary soup of a TLS response it did not ask for.

pub mod cert;
mod perms;
pub mod serve;

pub use cert::{Identity, box_names, fingerprint_hex, primary_address};
pub use serve::serve;

/// Anything that can stop a TLS identity from being established.
///
/// Every variant is fatal to *https*, never to the manager: [`Identity::load_or_create`]
/// is expected to be called by something that logs the error and serves plain
/// HTTP instead. A box that cannot write a key file is still a box that should
/// come up.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read or write the TLS identity at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("certificate generation failed: {0}")]
    Gen(#[from] rcgen::Error),
    #[error("rustls rejected the generated certificate: {0}")]
    Rustls(#[from] rustls::Error),
    #[error("the stored TLS identity at {path} is not readable as PEM: {detail}")]
    Corrupt {
        path: std::path::PathBuf,
        detail: String,
    },
}
