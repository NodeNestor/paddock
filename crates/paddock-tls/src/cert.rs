//! The box's certificate authority and the leaf it signs.
//!
//! Four files under `<data>/tls/`, and nothing else:
//!
//! | file | what it is |
//! |---|---|
//! | `ca.crt` | the root a user installs on their laptop, once |
//! | `ca.key` | the key that signs leaves - the only real secret here |
//! | `server.crt` | the leaf actually presented on the wire |
//! | `server.key` | its key |
//!
//! The split matters. A bare self-signed leaf could only ever be clicked
//! through, and would have to be *replaced* on every renewal or address
//! change - invalidating whatever trust the user had granted it. A root that
//! outlives its leaves can be installed once and stays correct.

use std::collections::BTreeSet;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, GeneralSubtree, IsCa,
    Issuer, KeyPair, KeyUsagePurpose, NameConstraints, SanType,
};
use rustls_pki_types::pem::PemObject;
use rustls_pki_types::{CertificateDer, PrivateKeyDer};
use time::{Duration, OffsetDateTime};

use crate::{Error, perms};

/// How long a root is good for. Long, because reissuing it is the one event
/// that costs every user who trusted it a second trip through their OS trust
/// settings.
const CA_YEARS: i64 = 10;

/// Leaf lifetime. **398 days is a ceiling, not a preference**: Apple platforms
/// reject any TLS server certificate valid for longer, and they enforce it for
/// locally-installed roots too - a 10-year leaf would simply fail on Safari
/// and every iPhone on the LAN. A year, minus a fortnight of slack.
const LEAF_DAYS: i64 = 380;

/// Reissue once the leaf has this long left. Generous, because reissuing is
/// milliseconds and costs nobody anything (the ROOT is what trust is pinned
/// to), while an expired leaf is a scary red page.
const RENEW_WITHIN_DAYS: i64 = 30;

/// Backdate `not_before`. Clock skew between the box and a client is normal on
/// a LAN, and a certificate that is not valid *yet* fails exactly as hard as
/// one that has expired.
const BACKDATE_HOURS: i64 = 24;

/// A ready-to-serve TLS identity for this box.
pub struct Identity {
    /// Handed to rustls for every accepted connection.
    pub server: Arc<rustls::ServerConfig>,
    /// The root, PEM-encoded - what `GET /tls/root.crt` returns.
    pub root_pem: String,
    /// SHA-256 of the root, colon-separated hex. Printed at startup so a user
    /// installing the root can check they are trusting the box in front of
    /// them and not something that answered first.
    pub fingerprint: String,
    /// Everything the leaf covers, for the banner and the trust page.
    pub names: Vec<String>,
    /// True when this run had to write new files (first run, renewal, or the
    /// box's addresses changed). Worth a log line; nothing else reads it.
    pub issued: bool,
}

impl Identity {
    /// Load the identity from `dir`, creating or renewing whatever is missing,
    /// expired, or no longer matches the box's addresses.
    ///
    /// Every failure is the caller's cue to serve plain HTTP and say so - a
    /// box that cannot write a key file is still a box that should come up.
    pub fn load_or_create(dir: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(dir).map_err(|e| Error::Io {
            path: dir.into(),
            source: e,
        })?;
        perms::restrict_to_owner(dir);

        let (ca_pem, ca_key_pem, ca_fresh) = load_or_create_ca(dir)?;
        let ca_der = der_of_pem(dir.join("ca.crt"), &ca_pem)?;

        let wanted = box_names();
        let leaf = dir.join("server.crt");
        let leaf_key = dir.join("server.key");

        // Reuse only a leaf that is (a) parseable, (b) not about to expire,
        // and (c) still covers exactly the names this box answers to. A new
        // network address is as good a reason to reissue as a near-expiry -
        // an address the leaf does not name is an address the browser refuses.
        let reuse = if ca_fresh {
            // A new root means every existing leaf is signed by a key nothing
            // trusts any more.
            None
        } else {
            match (
                std::fs::read_to_string(&leaf),
                std::fs::read_to_string(&leaf_key),
            ) {
                (Ok(crt), Ok(key)) => match inspect(&leaf, &crt) {
                    Ok(found) => (found.expires - OffsetDateTime::now_utc()
                        > Duration::days(RENEW_WITHIN_DAYS)
                        && found.names == wanted)
                        .then_some((crt, key)),
                    Err(_) => None,
                },
                _ => None,
            }
        };

        let (leaf_pem, leaf_key_pem, issued) = match reuse {
            Some((crt, key)) => (crt, key, false),
            None => {
                let ca_key = KeyPair::from_pem(&ca_key_pem)?;
                let issuer = Issuer::from_ca_cert_pem(&ca_pem, ca_key)?;
                let key = KeyPair::generate()?;
                let cert = leaf_params(&wanted)?.signed_by(&key, &issuer)?;
                let (crt_pem, key_pem) = (cert.pem(), key.serialize_pem());
                write_secret(&leaf_key, &key_pem)?;
                write_public(&leaf, &crt_pem)?;
                (crt_pem, key_pem, true)
            }
        };

        let chain = vec![
            CertificateDer::from_pem_slice(leaf_pem.as_bytes()).map_err(|e| Error::Corrupt {
                path: leaf,
                detail: e.to_string(),
            })?,
        ];
        let key =
            PrivateKeyDer::from_pem_slice(leaf_key_pem.as_bytes()).map_err(|e| Error::Corrupt {
                path: leaf_key,
                detail: e.to_string(),
            })?;

        // NAME the provider rather than take the ambient default. `rustls`
        // features unify across the whole workspace, and something else in the
        // graph already turns on `ring` beside our `aws-lc-rs` - with both
        // present rustls refuses to guess and PANICS at the first builder
        // call, which is a crash on startup, not a fallback (found the first
        // time this ran). Installing a process-wide default from
        // a library would also be the wrong shape: this is one connection
        // listener's business, not the process's.
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let mut server = rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()?
            .with_no_client_auth()
            .with_single_cert(chain, key)?;
        // **http/1.1 only, deliberately.** Offer h2 over ALPN and a browser
        // takes it - and WebSockets over h2 need RFC 8441 extended CONNECT,
        // which is a different upgrade path than the one `/api/gpu/stream` and
        // the realtime transcription relay use. Cleartext http already resolves
        // to http/1.1 in practice, so pinning it here means turning TLS on
        // changes encryption and nothing else.
        server.alpn_protocols = vec![b"http/1.1".to_vec()];

        Ok(Identity {
            server: Arc::new(server),
            fingerprint: fingerprint_hex(&ca_der),
            root_pem: ca_pem,
            names: wanted.into_iter().collect(),
            issued: issued || ca_fresh,
        })
    }
}

/// SHA-256 over the certificate's DER, colon-separated uppercase hex - the
/// form every OS certificate viewer shows, so the two can be compared by eye.
pub fn fingerprint_hex(der: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(der);
    digest
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(":")
}

/// Returns (cert PEM, key PEM, freshly generated).
fn load_or_create_ca(dir: &Path) -> Result<(String, String, bool), Error> {
    let crt = dir.join("ca.crt");
    let key = dir.join("ca.key");
    if let (Ok(c), Ok(k)) = (std::fs::read_to_string(&crt), std::fs::read_to_string(&key))
        && KeyPair::from_pem(&k).is_ok()
    {
        return Ok((c, k, false));
    }
    let kp = KeyPair::generate()?;
    let cert = ca_params()?.self_signed(&kp)?;
    let (c, k) = (cert.pem(), kp.serialize_pem());
    write_secret(&key, &k)?;
    write_public(&crt, &c)?;
    Ok((c, k, true))
}

fn ca_params() -> Result<CertificateParams, Error> {
    let mut p = CertificateParams::default();
    let now = OffsetDateTime::now_utc();
    p.not_before = now - Duration::hours(BACKDATE_HOURS);
    p.not_after = now + Duration::days(365 * CA_YEARS);
    p.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
    p.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
    p.distinguished_name
        .push(DnType::CommonName, format!("Paddock on {}", host_label()));
    p.distinguished_name
        .push(DnType::OrganizationName, "Paddock");
    p.use_authority_key_identifier_extension = true;

    // **Name constraints, DNS only.** This key sits on the user's own box, and
    // installing its root into an OS trust store is the one genuinely
    // consequential thing we ask of anyone. An unconstrained root that leaks
    // could mint a certificate for any bank in the world; constrained to these
    // subtrees it can only ever impersonate this machine.
    //
    // IP SANs are left deliberately unconstrained. RFC 5280 constrains only
    // the name types that appear in permittedSubtrees, and pinning IP ranges
    // here would mean a box on an address we did not anticipate - a public
    // static IP, an unusual private range - failing verification outright.
    // A DNS constraint blocks the dangerous case; an IP constraint would only
    // narrow an attack that already requires being on the wire.
    let mut permitted = vec![
        GeneralSubtree::DnsName("localhost".to_owned()),
        // mDNS: covers `<host>.local`, the name a Mac or an iPhone resolves.
        GeneralSubtree::DnsName("local".to_owned()),
    ];
    let host = host_label();
    if host != "localhost" {
        permitted.push(GeneralSubtree::DnsName(host));
    }
    p.name_constraints = Some(NameConstraints {
        permitted_subtrees: permitted,
        excluded_subtrees: vec![],
    });
    Ok(p)
}

fn leaf_params(names: &BTreeSet<String>) -> Result<CertificateParams, Error> {
    let mut p = CertificateParams::default();
    let now = OffsetDateTime::now_utc();
    p.not_before = now - Duration::hours(BACKDATE_HOURS);
    p.not_after = now + Duration::days(LEAF_DAYS);
    p.is_ca = IsCa::NoCa;
    p.key_usages = vec![
        KeyUsagePurpose::DigitalSignature,
        KeyUsagePurpose::KeyEncipherment,
    ];
    p.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    p.use_authority_key_identifier_extension = true;
    p.distinguished_name.push(DnType::CommonName, host_label());
    for n in names {
        p.subject_alt_names.push(match n.parse::<IpAddr>() {
            Ok(ip) => SanType::IpAddress(ip),
            Err(_) => SanType::DnsName(n.clone().try_into()?),
        });
    }
    Ok(p)
}

/// Every name and address a browser might reasonably use to reach this box.
///
/// A `BTreeSet` because it is compared against the stored leaf's SAN list to
/// decide on reissue, and that comparison has to be order-insensitive.
///
/// Loopback is included even though loopback is already a secure context: a
/// user who trusts the root and then visits `https://localhost:11500` should
/// not meet a name mismatch.
/// Every name this box legitimately answers to: loopback, its hostname (plus
/// `.local`), and every non-loopback interface address. Public because the TLS
/// certificate is not the only thing that needs it - rmcp's Streamable HTTP
/// server validates the inbound `Host` header against an allow-list, and the
/// honest list is exactly this one.
pub fn box_names() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    out.insert("localhost".to_owned());
    out.insert("127.0.0.1".to_owned());
    out.insert("::1".to_owned());

    let host = host_label();
    if host != "localhost" {
        out.insert(format!("{host}.local"));
        out.insert(host);
    }

    if let Ok(ifaces) = if_addrs::get_if_addrs() {
        for i in ifaces {
            let ip = i.addr.ip();
            if ip.is_loopback() {
                continue;
            }
            // Link-local IPv6 carries a zone index that has no meaning on
            // another host, and no browser will ever be pointed at one.
            if matches!(ip, IpAddr::V6(v6) if (v6.segments()[0] & 0xffc0) == 0xfe80) {
                continue;
            }
            out.insert(ip.to_string());
        }
    }
    out
}

/// The address another device would actually reach this box on - the one to
/// print when telling someone where to go.
///
/// Enumerating interfaces is not enough to CHOOSE. A Windows box routinely has
/// a Hyper-V or WSL virtual switch alongside the real network card, and its
/// address routinely sorts ahead of the one that actually works; pointing a
/// person at that is pointing them at nothing.
///
/// So ask the routing table instead: which local address would be used to
/// reach the outside world. The UDP `connect` transmits nothing - it only
/// resolves the route - and the target is TEST-NET-3, which is reserved for
/// documentation and routed nowhere even if something did leak.
///
/// `None` on a box with no default route at all, where there is no better
/// answer than "look at the list".
pub fn primary_address() -> Option<IpAddr> {
    let sock = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("203.0.113.1:80").ok()?;
    let ip = sock.local_addr().ok()?.ip();
    (!ip.is_loopback() && !ip.is_unspecified()).then_some(ip)
}

/// This box's short hostname, lowercased, sanitised to what a DNS label may
/// hold. Windows machine names in particular can carry characters that are
/// not valid in a certificate name.
fn host_label() -> String {
    let raw = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .or_else(|| {
            std::process::Command::new("hostname")
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        })
        .unwrap_or_default();
    let cleaned: String = raw
        .split('.')
        .next()
        .unwrap_or("")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect::<String>()
        .to_ascii_lowercase();
    if cleaned.is_empty() {
        "localhost".to_owned()
    } else {
        cleaned
    }
}

struct Stored {
    expires: OffsetDateTime,
    names: BTreeSet<String>,
}

/// What a stored leaf actually says - read from the certificate rather than a
/// sidecar file, so the two can never disagree.
fn inspect(path: &Path, pem: &str) -> Result<Stored, Error> {
    use x509_parser::certificate::X509Certificate;
    use x509_parser::extensions::GeneralName;
    use x509_parser::prelude::FromDer;

    let der = der_of_pem(path.to_path_buf(), pem)?;
    let (_, x509) = X509Certificate::from_der(&der).map_err(|e| Error::Corrupt {
        path: path.into(),
        detail: e.to_string(),
    })?;

    let mut names = BTreeSet::new();
    if let Ok(Some(san)) = x509.subject_alternative_name() {
        for gn in &san.value.general_names {
            match gn {
                GeneralName::DNSName(d) => {
                    names.insert((*d).to_owned());
                }
                GeneralName::IPAddress(raw) => {
                    // x509 stores addresses as raw octets, not text.
                    let ip = match raw.len() {
                        4 => Some(IpAddr::from(<[u8; 4]>::try_from(*raw).unwrap_or_default())),
                        16 => Some(IpAddr::from(<[u8; 16]>::try_from(*raw).unwrap_or_default())),
                        _ => None,
                    };
                    if let Some(ip) = ip {
                        names.insert(ip.to_string());
                    }
                }
                _ => {}
            }
        }
    }

    let secs = x509.validity().not_after.timestamp();
    let expires = OffsetDateTime::from_unix_timestamp(secs).map_err(|e| Error::Corrupt {
        path: path.into(),
        detail: e.to_string(),
    })?;
    Ok(Stored { expires, names })
}

fn der_of_pem(path: PathBuf, pem: &str) -> Result<Vec<u8>, Error> {
    CertificateDer::from_pem_slice(pem.as_bytes())
        .map(|d| d.as_ref().to_vec())
        .map_err(|e| Error::Corrupt {
            path,
            detail: e.to_string(),
        })
}

/// A private key: written, then narrowed to this user alone.
fn write_secret(path: &Path, pem: &str) -> Result<(), Error> {
    std::fs::write(path, pem).map_err(|e| Error::Io {
        path: path.into(),
        source: e,
    })?;
    perms::restrict_to_owner(path);
    Ok(())
}

/// A certificate: public by definition - the root is meant to be handed out.
fn write_public(path: &Path, pem: &str) -> Result<(), Error> {
    std::fs::write(path, pem).map_err(|e| Error::Io {
        path: path.into(),
        source: e,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pd-tls-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        d
    }

    #[test]
    fn first_run_creates_an_identity_and_the_second_reuses_it() {
        let dir = tmp("reuse");
        let a = Identity::load_or_create(&dir).expect("first run");
        assert!(a.issued, "a first run has nothing to reuse");
        assert!(a.root_pem.starts_with("-----BEGIN CERTIFICATE-----"));

        let b = Identity::load_or_create(&dir).expect("second run");
        assert!(
            !b.issued,
            "a fresh leaf must not be reissued on every start"
        );
        assert_eq!(
            a.fingerprint, b.fingerprint,
            "the ROOT is what trust is pinned to"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The reason a stale leaf is detected at all: a box that gained or lost an
    /// address must reissue, or the browser meets a name it does not cover.
    #[test]
    fn a_leaf_that_no_longer_covers_the_box_is_reissued() {
        let dir = tmp("names");
        Identity::load_or_create(&dir).expect("first run");

        let leaf = dir.join("server.crt");
        let pem = std::fs::read_to_string(&leaf).expect("leaf");
        let found = inspect(&leaf, &pem).expect("parse");
        assert_eq!(
            found.names,
            box_names(),
            "what we asked for is what got signed"
        );
        assert!(found.names.contains("localhost"));
        assert!(found.names.contains("127.0.0.1"));

        // Sign a leaf for a box that answers to something else entirely.
        let ca_pem = std::fs::read_to_string(dir.join("ca.crt")).expect("ca");
        let ca_key = KeyPair::from_pem(&std::fs::read_to_string(dir.join("ca.key")).expect("k"))
            .expect("ca key");
        let issuer = Issuer::from_ca_cert_pem(&ca_pem, ca_key).expect("issuer");
        let key = KeyPair::generate().expect("key");
        let other = BTreeSet::from(["localhost".to_owned()]);
        let cert = leaf_params(&other)
            .expect("params")
            .signed_by(&key, &issuer)
            .expect("sign");
        std::fs::write(&leaf, cert.pem()).expect("write");
        std::fs::write(dir.join("server.key"), key.serialize_pem()).expect("write");

        let again = Identity::load_or_create(&dir).expect("third run");
        assert!(
            again.issued,
            "a leaf missing the box's own addresses must be replaced"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Apple rejects TLS server certificates valid for more than 398 days, and
    /// enforces it for privately-trusted roots too. Getting this wrong means
    /// every iPhone and every Safari on the LAN refuses the Studio, which is
    /// not something a Windows dev box would ever notice.
    #[test]
    fn the_leaf_stays_under_apples_398_day_ceiling() {
        let dir = tmp("398");
        Identity::load_or_create(&dir).expect("run");
        let leaf = dir.join("server.crt");
        let pem = std::fs::read_to_string(&leaf).expect("leaf");
        let found = inspect(&leaf, &pem).expect("parse");
        let span = found.expires - (OffsetDateTime::now_utc() - Duration::hours(BACKDATE_HOURS));
        assert!(
            span < Duration::days(398),
            "leaf valid for {} days",
            span.whole_days()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A hostname is not a DNS label. Windows in particular allows characters
    /// (and a length) that no certificate name may carry.
    #[test]
    fn the_host_label_is_always_a_usable_dns_label() {
        let h = host_label();
        assert!(!h.is_empty());
        assert!(
            h.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            "unusable label: {h}"
        );
    }
}
