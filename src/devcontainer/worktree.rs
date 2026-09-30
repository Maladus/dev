//! Keep git usable inside the container when the workspace is a linked worktree.
//!
//! A linked worktree's `.git` is a file (`gitdir: <path>`) that points into the
//! main repository's `.git/worktrees/<name>`, which in turn points at the shared
//! object store through its `commondir` file. Only the workspace folder is
//! bind-mounted, so the pointer does not resolve in the container and every git
//! command fails with "not a git repository".
//!
//! The mounts that fix this are derived from the host at `dev up` time, so no
//! user- or machine-specific path ever has to be written into a devcontainer.json.
//! Both absolute pointers (git's default) and relative ones
//! (`git worktree add --relative-paths`, git 2.48+) are handled: the shared git
//! directory is mounted wherever the pointer resolves *inside* the container.

use std::path::{Component, Path, PathBuf};

use crate::runtime::BindMount;

/// Extra bind mounts that make git work in a container whose workspace is a
/// linked worktree. Empty for a normal clone, a non-git folder, or when the
/// shared git directory is already inside the workspace.
///
/// `workspace_target` is the container path the workspace is mounted at.
pub fn worktree_mounts(workspace: &Path, workspace_target: &str) -> Vec<BindMount> {
    let Some(pointer) = read_gitdir_pointer(workspace) else {
        return Vec::new();
    };
    let workspace_target = Path::new(workspace_target);

    // The per-worktree admin dir (`<common>/worktrees/<name>`), on both sides.
    let host_admin = normalize(&workspace.join(&pointer));
    let container_admin = normalize(&workspace_target.join(&pointer));

    // The shared git dir. A gitfile without `commondir` (a submodule) points at
    // a complete git dir, which is then the thing to mount.
    let commondir = read_trimmed(&host_admin.join("commondir")).map(PathBuf::from);
    let (host_common, container_common) = match &commondir {
        Some(c) => (
            normalize(&host_admin.join(c)),
            normalize(&container_admin.join(c)),
        ),
        None => (host_admin.clone(), container_admin.clone()),
    };

    if host_common.starts_with(workspace) || !host_common.is_dir() {
        return Vec::new();
    }

    let mut mounts = vec![BindMount {
        source: host_common,
        target: container_common.to_string_lossy().into_owned(),
        readonly: false,
    }];

    // The admin dir's `gitdir` file points back at the worktree's `.git`. If
    // that path does not exist in the container, `git worktree prune` run there
    // treats the worktree as deleted and removes its admin dir, which breaks the
    // worktree on the host too. Mount the workspace a second time at the path
    // the back-pointer expects so prune sees it as alive.
    if let Some(back) = read_trimmed(&host_admin.join("gitdir")) {
        let container_back = normalize(&container_admin.join(back));
        let expected = normalize(&workspace_target.join(".git"));
        if container_back != expected
            && let Some(parent) = container_back.parent()
            && parent != Path::new("/")
        {
            mounts.push(BindMount {
                source: workspace.to_path_buf(),
                target: parent.to_string_lossy().into_owned(),
                readonly: false,
            });
        }
    }

    mounts
}

/// The `gitdir:` value of `<workspace>/.git` when it is a gitfile.
fn read_gitdir_pointer(workspace: &Path) -> Option<PathBuf> {
    let dot_git = workspace.join(".git");
    if !dot_git.is_file() {
        return None;
    }
    let content = std::fs::read_to_string(dot_git).ok()?;
    content
        .lines()
        .find_map(|l| l.strip_prefix("gitdir:"))
        .map(|p| PathBuf::from(p.trim()))
        .filter(|p| !p.as_os_str().is_empty())
}

fn read_trimmed(path: &Path) -> Option<String> {
    let s = std::fs::read_to_string(path).ok()?;
    let s = s.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// Lexical normalization (`.` and `..`), without touching the filesystem: the
/// container-side paths do not exist on the host.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// A main repo and one linked worktree as git lays them out on disk.
    struct Layout {
        _tmp: TempDir,
        main_git: PathBuf,
        worktree: PathBuf,
    }

    fn layout(relative: bool) -> Layout {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let main_git = root.join("repo/.git");
        let admin = main_git.join("worktrees/feat");
        let worktree = root.join("repo__worktrees/feat");
        fs::create_dir_all(&admin).unwrap();
        fs::create_dir_all(&worktree).unwrap();
        fs::write(admin.join("commondir"), "../..\n").unwrap();
        if relative {
            fs::write(
                worktree.join(".git"),
                "gitdir: ../../repo/.git/worktrees/feat\n",
            )
            .unwrap();
            // As written by `git worktree add --relative-paths` (git 2.53).
            fs::write(
                admin.join("gitdir"),
                "../../../../repo__worktrees/feat/.git\n",
            )
            .unwrap();
        } else {
            fs::write(
                worktree.join(".git"),
                format!("gitdir: {}\n", admin.display()),
            )
            .unwrap();
            fs::write(
                admin.join("gitdir"),
                format!("{}\n", worktree.join(".git").display()),
            )
            .unwrap();
        }
        Layout {
            _tmp: tmp,
            main_git,
            worktree,
        }
    }

    #[test]
    fn absolute_pointer_mounts_common_dir_and_workspace_at_host_paths() {
        let l = layout(false);
        let mounts = worktree_mounts(&l.worktree, "/workspace");
        assert_eq!(mounts.len(), 2);
        assert_eq!(mounts[0].source, l.main_git);
        assert_eq!(mounts[0].target, l.main_git.to_string_lossy());
        assert_eq!(mounts[1].source, l.worktree);
        assert_eq!(mounts[1].target, l.worktree.to_string_lossy());
    }

    #[test]
    fn absolute_pointer_with_same_path_workspace_needs_no_second_mount() {
        let l = layout(false);
        let target = l.worktree.to_string_lossy().into_owned();
        let mounts = worktree_mounts(&l.worktree, &target);
        assert_eq!(mounts.len(), 1);
        assert_eq!(mounts[0].target, l.main_git.to_string_lossy());
    }

    #[test]
    fn relative_pointer_resolves_against_the_container_workspace() {
        let l = layout(true);
        let mounts = worktree_mounts(&l.worktree, "/workspaces/feat");
        assert_eq!(mounts.len(), 2);
        assert_eq!(mounts[0].source, l.main_git);
        assert_eq!(mounts[0].target, "/repo/.git");
        // The relative back-pointer expects the sibling layout.
        assert_eq!(mounts[1].source, l.worktree);
        assert_eq!(mounts[1].target, "/repo__worktrees/feat");
    }

    #[test]
    fn relative_pointer_with_mirrored_layout_needs_only_the_common_dir() {
        let l = layout(true);
        let mounts = worktree_mounts(&l.worktree, "/src/repo__worktrees/feat");
        assert_eq!(mounts.len(), 1);
        assert_eq!(mounts[0].target, "/src/repo/.git");
    }

    #[test]
    fn normal_clone_needs_nothing() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join(".git")).unwrap();
        assert!(worktree_mounts(tmp.path(), "/workspace").is_empty());
    }

    #[test]
    fn non_git_folder_needs_nothing() {
        let tmp = TempDir::new().unwrap();
        assert!(worktree_mounts(tmp.path(), "/workspace").is_empty());
    }

    #[test]
    fn missing_common_dir_is_skipped() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join(".git"),
            "gitdir: /nonexistent/.git/worktrees/x\n",
        )
        .unwrap();
        assert!(worktree_mounts(tmp.path(), "/workspace").is_empty());
    }

    #[test]
    fn common_dir_inside_workspace_is_skipped() {
        let tmp = TempDir::new().unwrap();
        let ws = tmp.path().canonicalize().unwrap();
        let admin = ws.join("inner/.git");
        fs::create_dir_all(&admin).unwrap();
        fs::write(ws.join(".git"), "gitdir: inner/.git\n").unwrap();
        assert!(worktree_mounts(&ws, "/workspace").is_empty());
    }

    #[test]
    fn normalize_resolves_parent_components() {
        assert_eq!(
            normalize(Path::new("/workspace/../../repo/./.git")),
            PathBuf::from("/repo/.git")
        );
    }
}
