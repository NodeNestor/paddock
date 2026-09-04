//! Tiny admin-surface CLI for debugging: `adminctl <port> <verb> [arg]` with
//! verbs identify | health | stats | events | drain | shutdown | list. Talks
//! the same pipe/socket the manager uses - handy for poking a live runner.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use paddock_admin::client::AdminClient;

fn usage() -> ! {
    eprintln!("usage: adminctl <port> identify|health|stats|drain|shutdown [timeout_ms]");
    eprintln!("       adminctl <port> events [since]");
    eprintln!("       adminctl list");
    std::process::exit(2);
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("list") {
        println!("{:?}", paddock_admin::enumerate());
        return;
    }
    let (Some(port), Some(verb)) = (args.first(), args.get(1)) else {
        usage()
    };
    let Ok(port) = port.parse::<u16>() else {
        usage()
    };
    let timeout = args.get(2).and_then(|t| t.parse::<u64>().ok());
    let c = AdminClient::new(port);
    let out = match verb.as_str() {
        "identify" => serde_json::to_value(c.identify().await.expect("identify")),
        "health" => serde_json::to_value(c.health().await.expect("health")),
        "stats" => Ok(c.stats().await.expect("stats")),
        "events" => {
            let since = timeout.unwrap_or(0); // third arg doubles as the cursor
            serde_json::to_value(c.events(since, 512, 0).await.expect("events"))
        }
        "drain" => serde_json::to_value(c.drain(timeout).await.expect("drain")),
        "shutdown" => serde_json::to_value(c.shutdown(timeout).await.expect("shutdown")),
        _ => usage(),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&out.expect("serialize")).expect("print")
    );
}
