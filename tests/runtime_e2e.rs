//! Live end-to-end tests against a real container runtime.
//!
//! These tests need a running Docker or Podman daemon, so they are gated on the
//! `DEV_E2E_RUNTIME` environment variable (`docker` or `podman`). When it is
//! unset every test returns early, which keeps the default `cargo test` run and
//! the fast CI jobs green on machines without a runtime. The CI matrix sets it
//! per leg:
//!
//! ```text
//! DEV_E2E_RUNTIME=docker cargo test --test runtime_e2e -- --test-threads=1
//! DEV_E2E_RUNTIME=podman cargo test --test runtime_e2e -- --test-threads=1
//! ```
//!
//! Two layers of coverage live here. The binary layer drives the built `dev`
//! executable through a full lifecycle, so it exercises argument parsing,
//! runtime selection, readiness, and label filtering together. The trait layer
//! calls [`ContainerRuntime`] directly to reach the daemon-compatibility surface
//! the in-crate mock-server tests cannot: whether the daemon actually accepts
//! the create body and whether Podman's Docker-compat layer agrees with Docker's.

use std::path::PathBuf;
use std::process::{Command, Output};

use devcontainer::runtime::docker::DockerRuntime;
use devcontainer::runtime::podman::PodmanRuntime;
use devcontainer::runtime::{ContainerConfig, ContainerRuntime, ContainerState};
use tempfile::TempDir;

/// Selects which runtime this test process targets. Unset means skip.
const RUNTIME_ENV: &str = "DEV_E2E_RUNTIME";

/// Small public image. One pull per test process, so registry rate limits are
/// a non-issue, and `alpine` ships both `sh` and `sleep` for the keep-alive
/// command the runtime injects.
const TEST_IMAGE: &str = "alpine:3.20";

/// The runtime this process was launched against, or `None` when the gate is
/// unset. Anything other than `docker`/`podman` is a configuration error and
/// panics rather than silently skipping.
fn selected_runtime() -> Option<&'static str> {
    match std::env::var(RUNTIME_ENV).ok().as_deref() {
        Some("docker") => Some("docker"),
        Some("podman") => Some("podman"),
        Some("") | None => None,
        Some(other) => panic!("{RUNTIME_ENV} must be `docker` or `podman`, got `{other}`"),
    }
}

/// Skip the calling test unless a runtime is selected.
macro_rules! require_runtime {
    () => {
        match selected_runtime() {
            Some(runtime) => runtime,
            None => {
                eprintln!("skipping: set {RUNTIME_ENV}=docker|podman to run live e2e tests");
                return;
            }
        }
    };
}

fn connect(runtime: &str) -> Box<dyn ContainerRuntime> {
    match runtime {
        "docker" => Box::new(DockerRuntime::connect().expect("connect to the Docker daemon")),
        "podman" => Box::new(PodmanRuntime::connect().expect("connect to the Podman socket")),
        other => panic!("unknown runtime `{other}`"),
    }
}

/// A workspace with a minimal image-based devcontainer config, plus an isolated
/// `HOME` so the child never reads the host's `~/.dev` base config.
struct E2e {
    workspace: TempDir,
    home: TempDir,
    runtime: &'static str,
}

impl E2e {
    fn new(runtime: &'static str) -> Self {
        let workspace = TempDir::new().expect("temp workspace");
        let home = TempDir::new().expect("temp home");
        let devcontainer = workspace.path().join(".devcontainer");
        std::fs::create_dir_all(&devcontainer).expect("create .devcontainer");
        std::fs::write(
            devcontainer.join("devcontainer.json"),
            format!("{{\n  \"name\": \"runtime-e2e\",\n  \"image\": \"{TEST_IMAGE}\"\n}}\n"),
        )
        .expect("write devcontainer.json");
        Self {
            workspace,
            home,
            runtime,
        }
    }

    fn dev(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_dev"))
            .arg("--workspace")
            .arg(self.workspace.path())
            .arg("--runtime")
            .arg(self.runtime)
            .args(args)
            .env("HOME", self.home.path())
            // Deterministic non-interactive run: no rebuild prompt, no TTY
            // probing against the test harness's inherited stdin.
            .env("DEV_FORCE_TTY", "0")
            .output()
            .expect("run the dev binary")
    }
}

/// Best-effort teardown so a failing assertion does not leak a container into
/// the next test.
struct Cleanup<'a>(&'a E2e);

impl Drop for Cleanup<'_> {
    fn drop(&mut self) {
        let _ = self.0.dev(&["down", "--remove"]);
    }
}

fn assert_success(output: &Output, what: &str) {
    assert!(
        output.status.success(),
        "{what} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// The whole CLI against this backend: bring a container up, confirm `status`
/// sees it, run a command in it, tear it down.
#[test]
fn binary_lifecycle_up_status_exec_down() {
    let runtime = require_runtime!();
    let e2e = E2e::new(runtime);
    let _cleanup = Cleanup(&e2e);

    // Start from a clean slate in case an earlier run leaked a container.
    let _ = e2e.dev(&["down", "--remove"]);

    let up = e2e.dev(&["up"]);
    assert_success(&up, "dev up");

    assert_running(&e2e, "after up");

    let exec = e2e.dev(&["exec", "--", "echo", "e2e-ok"]);
    assert_success(&exec, "dev exec");
    assert!(
        String::from_utf8_lossy(&exec.stdout).contains("e2e-ok"),
        "exec output should carry the command result, got:\n{}",
        String::from_utf8_lossy(&exec.stdout)
    );

    let down = e2e.dev(&["down", "--remove"]);
    assert_success(&down, "dev down --remove");

    let items = status_items(&e2e);
    assert!(
        items.is_empty(),
        "workspace should have no containers after down, got {items:?}"
    );
}

fn status_items(e2e: &E2e) -> Vec<serde_json::Value> {
    let status = e2e.dev(&["status", "--json"]);
    assert_success(&status, "dev status --json");
    serde_json::from_slice(&status.stdout).expect("status --json must be valid JSON")
}

fn assert_running(e2e: &E2e, context: &str) {
    let items = status_items(e2e);
    assert_eq!(
        items.len(),
        1,
        "expected one container {context}, got {items:?}"
    );
    assert_eq!(
        items[0]["state"], "Running",
        "container should be running {context}, got {items:?}"
    );
}

/// Build a create request that keeps the container alive and carries a label
/// `list_containers` can filter on, mirroring what `dev up` sends.
fn test_config(workspace: &std::path::Path) -> ContainerConfig {
    let local_folder = workspace
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(workspace));
    let name = format!(
        "dev-e2e-{}",
        workspace
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "ws".to_string())
    );

    ContainerConfig {
        image: TEST_IMAGE.to_string(),
        name,
        labels: std::collections::HashMap::from([(
            "devcontainer.local_folder".to_string(),
            local_folder.to_string_lossy().to_string(),
        )]),
        env: std::collections::HashMap::new(),
        mounts: vec![],
        volumes: vec![],
        ports: vec![],
        workspace_mount: None,
        workspace_folder: None,
        extra_args: vec![],
        entrypoint: None,
        init: false,
        privileged: false,
        cap_add: vec![],
        security_opt: vec![],
        userns_mode: None,
        devices: vec![],
        group_add: vec![],
        device_cgroup_rules: vec![],
    }
}

/// The daemon-compatibility surface: does this backend accept the body `dev`
/// builds, and does a full create/start/exec/list/stop/remove round trip work?
#[tokio::test]
async fn runtime_trait_lifecycle() {
    let runtime = require_runtime!();
    let rt = connect(runtime);

    // Pull first: it is the reachability check, and it exercises the empty
    // registry-credential path when the host has no ~/.docker/config.json
    // entry, the case Podman's Docker-compat layer used to reject as invalid
    // JSON.
    rt.pull_image(TEST_IMAGE)
        .await
        .expect("pull the test image");

    let workspace = TempDir::new().expect("temp workspace");
    let config = test_config(workspace.path());

    let id = rt
        .create_container(&config)
        .await
        .expect("create_container");
    rt.start_container(&id).await.expect("start_container");

    let info = rt.inspect_container(&id).await.expect("inspect_container");
    assert_eq!(info.state, ContainerState::Running);

    let exec = rt
        .exec(&id, &["echo".to_string(), "e2e-ok".to_string()], None, None)
        .await
        .expect("exec");
    assert_eq!(exec.exit_code, 0, "exec stderr: {}", exec.stderr);
    assert!(
        exec.stdout.contains("e2e-ok"),
        "exec stdout should carry the command result, got: {}",
        exec.stdout
    );

    let filters: Vec<String> = config
        .labels
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    let listed = rt.list_containers(&filters).await.expect("list_containers");
    assert!(
        listed.iter().any(|c| c.id == id || c.name == config.name),
        "label filter should find the container we created, got {listed:?}"
    );

    rt.stop_container(&id).await.expect("stop_container");
    rt.remove_container(&id).await.expect("remove_container");
}

/// Podman-specific divergence: `--userns keep-id` is accepted by Podman's
/// Docker-compat create endpoint and has no Docker equivalent. The mock-server
/// test proves `dev` serializes the field; this proves the daemon takes it.
#[tokio::test]
async fn podman_accepts_keep_id_userns() {
    let runtime = require_runtime!();
    if runtime != "podman" {
        eprintln!("skipping: keep-id userns is Podman-specific");
        return;
    }

    let rt = connect(runtime);
    rt.pull_image(TEST_IMAGE)
        .await
        .expect("pull the test image");

    let workspace = TempDir::new().expect("temp workspace");
    let mut config = test_config(workspace.path());
    config.userns_mode = Some("keep-id".to_string());

    let id = rt
        .create_container(&config)
        .await
        .expect("Podman should accept keep-id at create time");
    rt.start_container(&id).await.expect("start_container");
    rt.stop_container(&id).await.expect("stop_container");
    rt.remove_container(&id).await.expect("remove_container");
}
