//! OS-service integration: a model server the OS
//! supervises - no paddock manager required, no wrapper (NSSM/WinSW)
//! either. The per-endpoint config FILE is the whole configuration, so a
//! service is just "run the binary with `--config <file>` at boot":
//!
//!   Windows: `paddock-runner service install --config servers\11540.toml`
//!            registers a native auto-start Windows service (elevated
//!            terminal required). SCM Stop drains in-flight requests via the
//!            runner's own admin shutdown, then exits.
//!   Linux:   the same verb writes + enables a systemd unit - a system unit
//!            when root, else a user unit (with the linger note printed, so
//!            "starts at boot" is true and not just "starts at login").
//!
//! The manager stays what it is - an editor/launcher of config files. A
//! manager started next to service-managed runners adopts them over the
//! admin pipe and never double-spawns a serving port; per model, pick one
//! boot owner (the unit or a manager election).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;

use crate::config::Config;
use crate::startup::ServiceAction;

/// Service name for a config: stable, port-derived, collision-free per
/// endpoint (one port = one server = one service).
fn service_name(cfg_path: &Path, explicit: Option<&str>) -> Result<String, String> {
    if let Some(n) = explicit {
        return Ok(n.to_string());
    }
    let cfg = Config::from_toml(&cfg_path.to_path_buf()).map_err(|e| e.to_string())?;
    Ok(format!("paddock-runner-{}", cfg.port))
}

/// A canonical, absolute config path - services run with an arbitrary
/// working directory (System32 on Windows), so a relative path would break
/// the moment the SCM launches us.
fn abs_config(p: &Path) -> Result<PathBuf, String> {
    std::fs::canonicalize(p).map_err(|e| format!("config file {}: {e}", p.display()))
}

pub fn dispatch(action: ServiceAction) -> ExitCode {
    let result = match action {
        ServiceAction::Install { config, name } => install(&config, name.as_deref()),
        ServiceAction::Uninstall { config, name } => uninstall(config.as_deref(), name.as_deref()),
        ServiceAction::Run { config } => return run_service(&config),
    };
    match result {
        Ok(msg) => {
            println!("{msg}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

// ─── Windows: a native service via the SCM ──────────────────────────────────

#[cfg(windows)]
fn install(config: &Path, name: Option<&str>) -> Result<String, String> {
    use windows_service::service::{
        ServiceAccess, ServiceErrorControl, ServiceInfo, ServiceStartType, ServiceType,
    };
    use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};

    let config = abs_config(config)?;
    let name = service_name(&config, name)?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let manager = ServiceManager::local_computer(
        None::<&str>,
        ServiceManagerAccess::CONNECT | ServiceManagerAccess::CREATE_SERVICE,
    )
    .map_err(|e| format!("cannot open the service manager ({e}) - run this from an elevated (Administrator) terminal"))?;
    let info = ServiceInfo {
        name: name.clone().into(),
        display_name: format!("Paddock model server ({name})").into(),
        service_type: ServiceType::OWN_PROCESS,
        start_type: ServiceStartType::AutoStart,
        error_control: ServiceErrorControl::Normal,
        executable_path: exe,
        launch_arguments: vec![
            "service".into(),
            "run".into(),
            "--config".into(),
            config.clone().into(),
        ],
        dependencies: vec![],
        account_name: None, // LocalSystem
        account_password: None,
    };
    let service = manager
        .create_service(&info, ServiceAccess::CHANGE_CONFIG | ServiceAccess::START)
        .map_err(|e| format!("create service {name:?}: {e}"))?;
    let _ = service.set_description(format!(
        "Paddock model server - config: {}",
        config.display()
    ));
    // Start it now too: "install" should end with a serving endpoint, not a
    // second command to remember. Failure to start is reported, not hidden.
    match service.start::<&str>(&[]) {
        Ok(()) => Ok(format!(
            "service {name:?} installed (auto-start at boot) and starting.\n\
             Logs: {} (or the log_file set in the config).\n\
             Manage it with: sc stop {name} / sc start {name} / paddock-runner service uninstall --config {}",
            config.with_extension("toml.log").display(),
            config.display(),
        )),
        Err(e) => Ok(format!(
            "service {name:?} installed (auto-start at boot), but starting it now failed: {e}\n\
             Start it with: sc start {name}"
        )),
    }
}

#[cfg(windows)]
fn uninstall(config: Option<&Path>, name: Option<&str>) -> Result<String, String> {
    use windows_service::service::ServiceAccess;
    use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};

    let name = match (name, config) {
        (Some(n), _) => n.to_string(),
        (None, Some(c)) => service_name(&abs_config(c)?, None)?,
        (None, None) => return Err("pass --config or --name to identify the service".into()),
    };
    let manager =
        ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)
            .map_err(|e| format!("cannot open the service manager ({e}) - run this from an elevated (Administrator) terminal"))?;
    let service = manager
        .open_service(
            &name,
            ServiceAccess::STOP | ServiceAccess::DELETE | ServiceAccess::QUERY_STATUS,
        )
        .map_err(|e| format!("open service {name:?}: {e}"))?;
    // Best-effort stop first - deleting a running service only marks it for
    // deletion, which reads as "uninstall did nothing" until a reboot.
    let _ = service.stop();
    service
        .delete()
        .map_err(|e| format!("delete service {name:?}: {e}"))?;
    Ok(format!("service {name:?} removed"))
}

#[cfg(windows)]
fn run_service(config: &Path) -> ExitCode {
    // The SCM invoked us: hand the config to the service entry via a global
    // (define_windows_service's ffi entry takes no captured state).
    let Ok(config) = abs_config(config) else {
        return ExitCode::FAILURE;
    };
    let _ = SERVICE_CONFIG.set(config);
    match windows_service::service_dispatcher::start("paddock-runner", ffi_service_main) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // Not under the SCM (someone ran `service run` by hand): honest
            // pointer to the plain form instead of a cryptic 1063.
            eprintln!(
                "not running under the service manager ({e}) - run the server directly with\n  paddock-runner --config <file>"
            );
            ExitCode::FAILURE
        }
    }
}

#[cfg(windows)]
static SERVICE_CONFIG: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

#[cfg(windows)]
windows_service::define_windows_service!(ffi_service_main, service_main);

#[cfg(windows)]
fn service_main(_args: Vec<std::ffi::OsString>) {
    use std::time::Duration;
    use windows_service::service::{
        ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus,
        ServiceType,
    };
    use windows_service::service_control_handler::{self, ServiceControlHandlerResult};

    let Some(config) = SERVICE_CONFIG.get().cloned() else {
        return;
    };

    // File logging: a service has no console. `log_file` from the config
    // wins; default is a .log sibling of the config file.
    let cfg_for_log = Config::from_toml(&config).ok();
    let log_path = cfg_for_log
        .as_ref()
        .and_then(|c| c.log_file.clone())
        .unwrap_or_else(|| config.with_extension("toml.log"));
    paddock_admin::logging::init(Some(&log_path));

    let port = cfg_for_log.map(|c| c.port).unwrap_or(0);
    let status_handle = match service_control_handler::register("paddock-runner", move |control| {
        match control {
            ServiceControl::Stop | ServiceControl::Shutdown => {
                // Drain-then-exit through the runner's own admin surface -
                // the same graceful stop the manager uses. The process exits
                // when drained; the SCM reads process exit as stopped.
                std::thread::spawn(move || {
                    if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        let _ = rt.block_on(
                            paddock_admin::client::AdminClient::new(port).shutdown(Some(25_000)),
                        );
                    }
                    // Belt and braces: if the admin path didn't take us down,
                    // don't leave the SCM hanging in STOP_PENDING forever.
                    std::thread::sleep(Duration::from_secs(30));
                    std::process::exit(0);
                });
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        }
    }) {
        Ok(h) => h,
        Err(_) => return,
    };
    let report = |state: ServiceState, wait: Duration| {
        let _ = status_handle.set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: state,
            controls_accepted: ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: wait,
            process_id: None,
        });
    };
    // RUNNING immediately: the model load takes minutes and the SCM's start
    // window doesn't. "Running" = the service process is up; readiness is
    // what /healthz is for.
    report(ServiceState::Running, Duration::ZERO);

    // The normal serve path, config layered exactly like a CLI launch
    // (file + env; there are no flags here).
    let cli = crate::startup::Cli::parse_from([
        std::ffi::OsString::from("paddock-runner"),
        "--config".into(),
        config.into(),
    ]);
    let code = match crate::startup::resolve(&cli) {
        Ok((cfg, banner)) => match tokio::runtime::Runtime::new() {
            Ok(rt) => match rt.block_on(crate::run(cfg, banner)) {
                Ok(()) => 0u32,
                Err(e) => {
                    tracing::error!(%e, "server error");
                    1
                }
            },
            Err(e) => {
                tracing::error!(%e, "failed to start async runtime");
                1
            }
        },
        Err(e) => {
            tracing::error!(%e, "config error");
            2
        }
    };
    let _ = status_handle.set_service_status(ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: ServiceState::Stopped,
        controls_accepted: ServiceControlAccept::empty(),
        exit_code: ServiceExitCode::Win32(code),
        checkpoint: 0,
        wait_hint: Duration::ZERO,
        process_id: None,
    });
}

// ─── Linux/macOS: systemd units ─────────────────────────────────────────────

#[cfg(unix)]
fn install(config: &Path, name: Option<&str>) -> Result<String, String> {
    let config = abs_config(config)?;
    let name = service_name(&config, name)?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let unit = format!(
        "[Unit]\nDescription=Paddock model server ({name})\nAfter=network.target\n\n\
         [Service]\nType=simple\nExecStart={} --config {}\nRestart=on-failure\nRestartSec=3\n\n\
         [Install]\nWantedBy=default.target\n",
        exe.display(),
        config.display(),
    );
    // Root = a system unit; anyone else = a user unit (no sudo needed). The
    // linger note keeps "starts at boot" honest for user units.
    let root = unsafe { libc_geteuid() } == 0;
    let (dir, ctl): (PathBuf, &[&str]) = if root {
        (PathBuf::from("/etc/systemd/system"), &["systemctl"])
    } else {
        let home = std::env::var("HOME").map_err(|_| "HOME is not set".to_string())?;
        (
            PathBuf::from(home).join(".config/systemd/user"),
            &["systemctl", "--user"],
        )
    };
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let unit_path = dir.join(format!("{name}.service"));
    std::fs::write(&unit_path, unit).map_err(|e| format!("write {}: {e}", unit_path.display()))?;
    for args in [vec!["daemon-reload"], vec!["enable", "--now", &name]] {
        let st = std::process::Command::new(ctl[0])
            .args(&ctl[1..])
            .args(&args)
            .status()
            .map_err(|e| format!("systemctl: {e}"))?;
        if !st.success() {
            return Err(format!("systemctl {} failed ({st})", args.join(" ")));
        }
    }
    let linger = if root {
        String::new()
    } else {
        "\nNote: a user unit starts at LOGIN; for boot-without-login run: loginctl enable-linger $USER".to_string()
    };
    Ok(format!(
        "unit {} written, enabled, and starting.{linger}",
        unit_path.display()
    ))
}

#[cfg(unix)]
fn uninstall(config: Option<&Path>, name: Option<&str>) -> Result<String, String> {
    let name = match (name, config) {
        (Some(n), _) => n.to_string(),
        (None, Some(c)) => service_name(&abs_config(c)?, None)?,
        (None, None) => return Err("pass --config or --name to identify the service".into()),
    };
    let root = unsafe { libc_geteuid() } == 0;
    let ctl: &[&str] = if root {
        &["systemctl"]
    } else {
        &["systemctl", "--user"]
    };
    let _ = std::process::Command::new(ctl[0])
        .args(&ctl[1..])
        .args(["disable", "--now", &name])
        .status();
    let dir = if root {
        PathBuf::from("/etc/systemd/system")
    } else {
        PathBuf::from(std::env::var("HOME").map_err(|_| "HOME is not set".to_string())?)
            .join(".config/systemd/user")
    };
    let unit_path = dir.join(format!("{name}.service"));
    std::fs::remove_file(&unit_path).map_err(|e| format!("remove {}: {e}", unit_path.display()))?;
    Ok(format!("unit {} removed", unit_path.display()))
}

#[cfg(unix)]
fn run_service(config: &Path) -> ExitCode {
    // On systemd the unit runs the plain form; `service run` is just an alias
    // so a copied Windows invocation still works.
    let cli = crate::startup::Cli::parse_from([
        std::ffi::OsString::from("paddock-runner"),
        "--config".into(),
        config.as_os_str().to_owned(),
    ]);
    // systemd captures stdout into the journal, so terminal-only is right here
    // and the unit decides where it lands. What was wrong was the filter: this
    // arm carried its own `"info,paddock=debug"` and so missed the rmcp/hyper/
    // h2/tower_http caps, meaning a service-run runner logged the full JSON-RPC
    // body of every MCP call while the same binary run by hand did not.
    paddock_admin::logging::init(None);
    match crate::startup::resolve(&cli) {
        Ok((cfg, banner)) => match tokio::runtime::Runtime::new() {
            Ok(rt) => match rt.block_on(crate::run(cfg, banner)) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    // Same reasoning as the startup path: a service has no
                    // console to catch a stray eprintln, so the failure has to
                    // go where the operator will actually look.
                    tracing::error!(error = %e, "server error");
                    ExitCode::FAILURE
                }
            },
            Err(e) => {
                eprintln!("failed to start async runtime: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("config error: {e}");
            ExitCode::from(2)
        }
    }
}

#[cfg(unix)]
unsafe fn libc_geteuid() -> u32 {
    // no libc dependency needed for one syscall wrapper
    unsafe extern "C" {
        fn geteuid() -> u32;
    }
    unsafe { geteuid() }
}
