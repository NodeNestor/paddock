//! OAuth 2.1 authorization-code + PKCE for remote (Streamable HTTP) MCP servers.
//!
//! This is the browser-consent flow behind Claude.ai's "custom connector" UX: a
//! server registered with an optional OAuth Client ID/Secret is *authorized* by
//! sending the user to the provider's consent page, then exchanging the returned
//! code (with the PKCE verifier) for an access token. The token is persisted and
//! later injected as `Authorization: Bearer ...` on the existing HTTP transport -
//! so connection itself needs no new transport, just a header.
//!
//! Two client-identity paths, both PKCE:
//!  * **pre-registered** - a `client_id` (+ optional `client_secret`) the user
//!    pasted; we `configure_client` with it directly.
//!  * **dynamic** - no `client_id`; we run RFC 7591 Dynamic Client Registration
//!    against the provider's `registration_endpoint` (public client, PKCE).
//!
//! rmcp owns the OAuth crypto (`oauth2` under the hood); this module is the thin
//! paddock-facing surface - `begin()` -> `PendingAuth::finish()` -> `OAuthTokens`.

use std::time::{SystemTime, UNIX_EPOCH};

use rmcp::transport::auth::{AuthorizationManager, OAuthClientConfig};

use crate::{McpError, Result};

/// Inputs for starting an authorization: the MCP server URL (also the OAuth
/// resource + discovery base), an optional pre-registered client, requested
/// scopes, and the redirect URI our callback listens on.
#[derive(Clone, Debug)]
pub struct OAuthConfig {
    pub server_url: String,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub scopes: Vec<String>,
    pub redirect_uri: String,
}

/// The tokens minted by a completed authorization. `expires_at` is an absolute
/// unix-seconds deadline (computed from the response's `expires_in`), so a
/// connect-time check can tell "expired" without tracking when it was issued.
#[derive(Clone, Debug)]
pub struct OAuthTokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: Option<u64>,
}

/// A live authorization in flight: holds the `AuthorizationManager` (which owns
/// the PKCE verifier, stored keyed by CSRF `state`) between the authorize request
/// and the browser callback. The same instance must handle both halves.
pub struct PendingAuth {
    manager: AuthorizationManager,
    /// The CSRF `state` the provider will echo back - how the callback finds us.
    pub state: String,
}

fn oauth_err(e: impl std::fmt::Display) -> McpError {
    McpError::Oauth(e.to_string())
}

/// Pull the `state` (CSRF) query param out of the authorization URL rmcp built.
fn state_from_url(auth_url: &str) -> Option<String> {
    url::Url::parse(auth_url)
        .ok()?
        .query_pairs()
        .find(|(k, _)| k == "state")
        .map(|(_, v)| v.into_owned())
}

/// Begin an authorization: discover the provider, establish a client identity
/// (pre-registered or via DCR), and build the PKCE authorization URL the user is
/// sent to. Returns `(authorization_url, pending)`; stash `pending` keyed by
/// `pending.state` until the callback arrives.
pub async fn begin(cfg: &OAuthConfig) -> Result<(String, PendingAuth)> {
    let mut manager = AuthorizationManager::new(cfg.server_url.as_str())
        .await
        .map_err(oauth_err)?;

    // OAuth endpoints must be *discovered*, never guessed (rmcp: protected-resource
    // metadata first, then the authorization-server well-known). rmcp 3.x renamed
    // this to resolve_metadata and now also reports which of the two well-knowns
    // answered; we only need the metadata itself.
    let resolution = manager.resolve_metadata().await.map_err(oauth_err)?;
    manager.set_metadata(resolution.metadata);

    match &cfg.client_id {
        Some(client_id) if !client_id.trim().is_empty() => {
            // OAuthClientConfig went #[non_exhaustive] in rmcp 3.x (it grew
            // `application_type` for SEP-837), so it is built through its
            // constructor now rather than a struct literal. `new` seeds the
            // application type; leaving that default is what we want.
            let mut client = OAuthClientConfig::new(client_id.clone(), cfg.redirect_uri.clone())
                .with_scopes(cfg.scopes.clone());
            if let Some(secret) = cfg.client_secret.as_ref().filter(|s| !s.trim().is_empty()) {
                client = client.with_client_secret(secret.clone());
            }
            manager.configure_client(client).map_err(oauth_err)?;
        }
        // No pre-registered client -> Dynamic Client Registration (public + PKCE).
        // register_client() configures the manager's client internally on success.
        _ => {
            // rmcp 3.x takes the requested scopes at registration time as well,
            // so the DCR request advertises what we will actually ask for
            // instead of registering blank and widening at authorize time.
            let scope_refs: Vec<&str> = cfg.scopes.iter().map(String::as_str).collect();
            manager
                .register_client("Paddock", &cfg.redirect_uri, &scope_refs)
                .await
                .map_err(oauth_err)?;
        }
    }

    let scope_refs: Vec<&str> = cfg.scopes.iter().map(String::as_str).collect();
    let auth_url = manager
        .get_authorization_url(&scope_refs)
        .await
        .map_err(oauth_err)?;
    let state = state_from_url(&auth_url)
        .ok_or_else(|| McpError::Oauth("authorization url had no state parameter".into()))?;

    Ok((auth_url, PendingAuth { manager, state }))
}

impl PendingAuth {
    /// Complete the authorization: exchange `code` (+ the stored PKCE verifier,
    /// looked up by our `state`) for tokens. Consumes the pending flow.
    pub async fn finish(self, code: &str) -> Result<OAuthTokens> {
        let token = self
            .manager
            .exchange_code_for_token(code, &self.state)
            .await
            .map_err(oauth_err)?;

        // Extract via serde rather than the oauth2 TokenResponse trait so this
        // crate needs no direct oauth2 dependency (StandardTokenResponse is
        // Serialize: access_token/refresh_token/expires_in).
        let v = serde_json::to_value(&token).unwrap_or_default();
        let access_token = v
            .get("access_token")
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| McpError::Oauth("token response had no access_token".into()))?
            .to_string();
        let refresh_token = v
            .get("refresh_token")
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from);
        let expires_at = v
            .get("expires_in")
            .and_then(|x| x.as_u64())
            .map(|secs| now_secs().saturating_add(secs));

        Ok(OAuthTokens {
            access_token,
            refresh_token,
            expires_at,
        })
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
