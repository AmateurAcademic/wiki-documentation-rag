# utils/git_handler.py
import os
import json
import glob
import subprocess
import time
from typing import List, Tuple, Optional

class GitHandler:
    """
    Encapsulates Git operations and state for the markdown repo.
    Responsible for:
      - verifying Git availability and repo status
      - configuring safe.directory
      - reading current commit
      - diffing between commits
      - persisting last processed commit
      - (optionally) full-file fallback listing
    """
    def __init__(self, repo_dir: str, state_file: str):
        self.repo_dir = repo_dir          # e.g. "/app/data/markdown"
        self.state_file = state_file      # e.g. "/app/state/.git_processing_state.json"
        self._branch_name: Optional[str] = None

    # --------- public API ---------
    def verify_git_installed(self) -> bool:
        """Check if git is installed and callable."""
        try:
            subprocess.run(
                ["git", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def configure_safe_directory(self) -> bool:
        """Configure Git to trust this repo directory (CVE-2022-24765)."""
        try:
            self._run_git_command(
                "config", "--global", "--add", "safe.directory", self.repo_dir
            )
            print(f"Configured Git safe directory: {self.repo_dir}")
            return True
        except Exception as e:
            print(f"Failed to configure Git safe directory: {e}")
            return False

    def is_git_repo(self) -> bool:
        """Return True if repo_dir is inside a Git repository."""
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_current_commit(self) -> str:
        """Return the current HEAD commit hash for detected branch."""
        branch = self._detect_git_branch()
        result = self._run_git_command("rev-parse", branch)
        return result.stdout.strip()

    def get_changed_files(
        self, old_commit: str, new_commit: str
    ) -> Tuple[List[str], List[str]]:
        """
        Get changed files using Git diff with proper status handling.
        Returns: (changed_files, deleted_files)
        Each list contains **absolute paths** under repo_dir to .md files.
        If diff fails (index.lock, etc.), falls back to listing all markdown files
        as "changed" and an empty deleted_files list.
        """
        try:
            result = self._run_git_command(
                "diff",
                "--name-status",
                "--find-renames",
                f"{old_commit}..{new_commit}",
            )
        except Exception as exc:
            print(f"Error getting changed files: {exc}")
            # Fallback: treat all markdown files as changed
            return self._list_all_markdown_files(), []
        
        changed_files: List[str] = []
        deleted_files: List[str] = []
        stdout = result.stdout.strip()
        
        if not stdout:
            print("Git diff found 0 changed files, 0 deleted files")
            return changed_files, deleted_files
            
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            status = parts[0]
            file_path = parts[1]  # relative path from repo root
            
            # Deleted
            if status.startswith("D"):
                if file_path.endswith(".md"):
                    deleted_files.append(
                        os.path.join(self.repo_dir, file_path)
                    )
                continue
                
            # Renamed (RXXX where XXX is similarity %)
            if status.startswith("R"):
                old_path = parts[1]
                new_path = parts[2] if len(parts) > 2 else file_path
                if old_path.endswith(".md"):
                    deleted_files.append(os.path.join(self.repo_dir, old_path))
                if new_path.endswith(".md"):
                    changed_files.append(os.path.join(self.repo_dir, new_path))
                continue
                
            # Modified (M), Added (A), Copied (C), etc.
            if file_path.endswith(".md"):
                changed_files.append(os.path.join(self.repo_dir, file_path))
                
        print(
            f"Git diff found {len(changed_files)} changed files, "
            f"{len(deleted_files)} deleted files"
        )
        return changed_files, deleted_files

    def save_last_processed_commit(self, commit_hash: str) -> None:
        """Persist last processed commit (atomic write to avoid corruption)."""
        tmp = self.state_file + ".tmp"
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"last_processed_commit": commit_hash}, f)
            os.replace(tmp, self.state_file)
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

    def load_last_processed_commit(self) -> Optional[str]:
        """Load last processed commit hash from state file, if present."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    return state.get("last_processed_commit")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading state file: {e}")
        return None

    def _run_git_command(self, *args, max_retries: int = 3):
        """Run a Git command in repo_dir with retry on index.lock."""
        delay = 1
        for attempt in range(max_retries):
            try:
                return subprocess.run(
                    ["git", *map(str, args)],
                    cwd=self.repo_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
            except subprocess.CalledProcessError as e:
                if "index.lock" in e.stderr.lower() and attempt < max_retries - 1:
                    print(
                        f"Git index.lock detected, retrying in {delay} seconds "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 10)
                    continue
                raise
            except subprocess.TimeoutExpired as exc:
                cmd_str = " ".join(map(str, args))
                raise RuntimeError(f"Git command timed out: {cmd_str}") from exc
        raise RuntimeError("Unexpected error in Git command execution")

    def _detect_git_branch(self) -> str:
        """Auto-detect Git branch (main or master)."""
        if self._branch_name:
            return self._branch_name
        for branch in ("main", "master"):
            try:
                self._run_git_command("rev-parse", "--verify", branch)
                self._branch_name = branch
                return branch
            except subprocess.CalledProcessError:
                continue
        raise ValueError("Could not detect Git branch (tried 'main' and 'master')")

    def _list_all_markdown_files(self) -> List[str]:
        """List all markdown files under repo_dir (for fallback)."""
        files: List[str] = []
        pattern = os.path.join(self.repo_dir, "**", "*.md")
        for filepath in glob.glob(pattern, recursive=True):
            if os.path.exists(filepath) and os.path.isfile(filepath):
                files.append(filepath)
        return files

    def git_add_and_commit(self, file_path: str, commit_message: str) -> None:
        """
        Add a file to the git staging area and commit it with the provided message.
        """
        try:
            self._run_git_command("add", file_path)
            self._run_git_command("commit", "-m", commit_message)
            print(f"Successfully added and committed {file_path}")
        except Exception as e:
            print(f"Error adding and committing {file_path}: {e}")