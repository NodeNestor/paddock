// static_assets.rs embeds the built Studio SPA from `static/` via rust-embed.
// That folder is the gitignored vite build output - on a checkout that never
// built the Studio (CI, a fresh clone, the Linux bench box) it does not exist,
// and #[derive(RustEmbed)] hard-fails on a MISSING folder even though the
// serving code already handles an empty one ("Studio UI is not built into
// this binary"). Guarantee the folder here so every checkout compiles; the
// rerun hint also picks up a later `vite build` without a manual touch.
fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("static");
    std::fs::create_dir_all(&dir).expect("create static/ for rust-embed");
    println!("cargo:rerun-if-changed={}", dir.display());

    #[cfg(windows)]
    windows_resource();
}

// Windows VERSIONINFO + the taskbar icon. The consumer binary - it is what
// shows up in Task Manager, in the firewall prompt on first bind, and in the
// SmartScreen dialog before anyone has ever run it. A blank ProductName there
// undoes the work the Authenticode signature is doing.
//
// The runner's build.rs carries the same block; see it for why a failure here
// panics rather than warns.
#[cfg(windows)]
fn windows_resource() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/<crate> is two levels below the repo root");
    let icon = repo.join("assets").join("paddock.ico");
    println!("cargo:rerun-if-changed={}", icon.display());

    let mut res = winresource::WindowsResource::new();
    res.set_icon(icon.to_str().expect("icon path is UTF-8"));
    res.set("ProductName", "Paddock");
    res.set("FileDescription", "Paddock - local AI serving");
    res.set("CompanyName", "Truespar");
    res.set(
        "LegalCopyright",
        "Copyright (c) 2026 Truespar. MIT OR Apache-2.0.",
    );
    res.set("OriginalFilename", "paddock.exe");
    res.set("InternalName", "paddock");

    if let Err(e) = res.compile() {
        panic!(
            "could not compile the Windows resource ({e}).\n\
             rc.exe comes with the Windows SDK - build from a VS2022 vcvars64 \
             environment."
        );
    }
}
