//! Re-export of the shared web-search provider client (`paddock-websearch`).
//! The runner executes the server-side `web_search` tool with provider config
//! declared at launch; the manager owns the stored settings and the Studio
//! surface. Kept as a module re-export so the dialect handlers keep their
//! `crate::websearch::` paths.

pub use paddock_websearch::*;
