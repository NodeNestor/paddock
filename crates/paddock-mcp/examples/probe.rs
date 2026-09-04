//! P0 gate: connect to the reference MCP server over stdio, list its tools, and
//! call one - end-to-end through the paddock-mcp client.
//!
//!   cargo run -p paddock-mcp --example probe
//!
//! Spawns `npx -y @modelcontextprotocol/server-everything` (downloads on first
//! run), so it needs node + network.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::collections::HashMap;
use std::time::Duration;

use paddock_mcp::{McpManager, ServerConfig, Transport};

#[tokio::main]
async fn main() {
    let cfg = ServerConfig {
        id: "probe".into(),
        label: "everything".into(),
        transport: Transport::Stdio {
            // `cmd /c` so Windows launches the npx shim reliably.
            command: "cmd".into(),
            args: vec![
                "/c".into(),
                "npx".into(),
                "-y".into(),
                "@modelcontextprotocol/server-everything".into(),
            ],
            env: HashMap::new(),
        },
    };
    let mgr = McpManager::new();

    let run = async {
        let tools = mgr.list_tools(&cfg).await?;
        println!("PROBE_OK discovered {} tools", tools.len());
        for t in tools.iter().take(20) {
            println!("  tool: {}", t.name);
        }
        if tools.iter().any(|t| t.name == "echo") {
            let res = mgr
                .call_tool(
                    &cfg,
                    "echo",
                    serde_json::json!({"message": "hello from paddock"}),
                )
                .await?;
            println!(
                "PROBE_ECHO is_error={} content={}",
                res.is_error, res.content
            );
        }
        Ok::<(), paddock_mcp::McpError>(())
    };

    match tokio::time::timeout(Duration::from_secs(90), run).await {
        Ok(Ok(())) => println!("PROBE_DONE"),
        Ok(Err(e)) => println!("PROBE_ERR {e}"),
        Err(_) => println!("PROBE_TIMEOUT"),
    }
    mgr.shutdown().await;
    // A lingering stdio child can hold the runtime open; exit deterministically.
    std::process::exit(0);
}
