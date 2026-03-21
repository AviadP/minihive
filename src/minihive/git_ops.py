"""
git_ops.py — Auto-commit module. The ONLY place where commits are created.

Agents never commit directly — their system prompts explicitly forbid it.
This module is the single point of truth for all git operations.
"""

from __future__ import annotations

import asyncio
import logging
from fnmatch import fnmatch
from pathlib import Path

from minihive.config import SUBPROCESS_MEDIUM_TIMEOUT
from minihive.contracts import TaskOutput, TaskStatus

logger = logging.getLogger(__name__)

# Patterns that must never be staged by the auto-committer.
_SENSITIVE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*secret*",
    "*credential*",
    "*credentials*",
    "*password*",
    "*.aws/credentials",
    "*.ssh/id_*",
    "id_rsa",
    "id_ed25519",
    ".netrc",
    # Minihive agent metadata — must never enter project commits
    ".minihive/*",
    ".minihive/**",
    "me_file*",
    # Agent-generated reports/reviews — work products, not source code
    "*REVIEW*",
    "*_REPORT*",
    "*_report*",
    "reviews/*",
    "reviews/**",
    "REVIEW_*.md",
    "*.review.md",
    # Orchestration metadata — plans, tasks, notes
    "*.plan.md",
    "notes.json",
    ".notes.json",
    "NOTES.md",
    "task_*.json",
    "plans/*",
    "plans/**",
    "tasks/*",
    "tasks/**",
)


def _is_sensitive(filepath: str) -> bool:
    """Return True if *filepath* matches any known sensitive file pattern."""
    name = Path(filepath).name
    normalized = filepath.replace("\\", "/")
    if normalized.startswith(".minihive/") or normalized == ".minihive":
        return True
    if normalized.startswith("plans/") or normalized.startswith("tasks/"):
        return True
    return any(fnmatch(filepath, pat) or fnmatch(name, pat) for pat in _SENSITIVE_PATTERNS)


# ── Per-project git lock ─────────────────────────────────────────────────────

_git_locks: dict[str, asyncio.Lock] = {}


def _git_lock(project_dir: str) -> asyncio.Lock:
    """Get or create a per-project git lock to prevent concurrent commits."""
    return _git_locks.setdefault(project_dir, asyncio.Lock())


# ── Staging helpers ──────────────────────────────────────────────────────────


async def _stage_scoped_files(project_dir: str, scoped_files: list[str]) -> None:
    """Stage only the files the task claims to have changed.

    Each file is validated against _SENSITIVE_PATTERNS before staging.
    Files that don't exist or have no changes are silently skipped by git.
    """
    staged = 0
    for filepath in scoped_files:
        if _is_sensitive(filepath):
            logger.warning("[git] Skipping sensitive scoped file: %s", filepath)
            continue
        result = await _run(["git", "add", "--", filepath], cwd=project_dir)
        if result is not None:
            staged += 1
    if staged:
        logger.debug("[git] Scoped staging: %d/%d files staged", staged, len(scoped_files))


async def _stage_files_safely(project_dir: str) -> None:
    """Stage project changes while excluding known-sensitive file patterns.

    1. ``git add -u`` — stages modifications/deletions of tracked files.
    2. Unstage any tracked sensitive files from the index.
    3. Stage safe untracked files individually.
    """
    # Stage tracked changes
    await _run(["git", "add", "-u"], cwd=project_dir)

    # Unstage any sensitive tracked files that got staged
    staged = await _run(["git", "diff", "--cached", "--name-only"], cwd=project_dir)
    if staged.strip():
        for filepath in staged.strip().splitlines():
            filepath = filepath.strip()
            if filepath and _is_sensitive(filepath):
                await _run(["git", "reset", "HEAD", "--", filepath], cwd=project_dir)
                logger.info("[git] Unstaged tracked metadata file: %s", filepath)

    # Stage safe untracked files
    raw = await _run(["git", "status", "--porcelain", "-z"], cwd=project_dir)
    entries = [e.strip() for e in raw.split("\0") if e.strip()]
    for entry in entries:
        if not entry.startswith("?? "):
            continue
        filepath = entry[3:]
        if _is_sensitive(filepath):
            logger.warning("[git] Skipping sensitive file from auto-commit: %s", filepath)
        else:
            await _run(["git", "add", "--", filepath], cwd=project_dir)


# ── Commit message builder ───────────────────────────────────────────────────


def _build_commit_message(
    task_id: str, role: str, summary: str, files: list[str]
) -> str:
    """Build a structured commit message for a single task."""
    first_line = f"feat: {summary[:72]}"
    body_lines = [f"\nTask: {task_id}", f"Role: {role}"]
    if files:
        unique = list(dict.fromkeys(files[:5]))
        body_lines.append(f"Files: {', '.join(unique)}")
    return first_line + "\n" + "\n".join(body_lines)


# ── Public API ───────────────────────────────────────────────────────────────


async def commit_single_task(
    project_dir: str,
    output: TaskOutput,
) -> str | None:
    """Commit changes after a single task completes.

    Stages only files in output.artifacts, filters sensitive files.
    Returns the short commit hash, or None if nothing to commit.
    """
    if not output or not output.is_successful():
        return None

    scoped_files = [f for f in (output.artifacts or []) if not _is_sensitive(f)] or None

    async with _git_lock(project_dir):
        proj = Path(project_dir)
        if not (proj / ".git").exists():
            logger.debug("[git] No .git directory, skipping auto-commit")
            return None

        status = await _run(["git", "status", "--porcelain"], cwd=project_dir)
        if not status.strip():
            return None

        if scoped_files:
            await _stage_scoped_files(project_dir, scoped_files)
        else:
            await _stage_files_safely(project_dir)

        # Check if anything was actually staged
        staged = await _run(["git", "diff", "--cached", "--name-only"], cwd=project_dir)
        if not staged.strip():
            logger.debug("[git] All changes were sensitive files — nothing to commit")
            return None

        role = getattr(output, "role", "agent")
        message = _build_commit_message(
            output.task_id, str(role), output.summary, output.artifacts or []
        )
        await _run(["git", "commit", "-m", message], cwd=project_dir)

        hash_result = await _run(["git", "rev-parse", "--short", "HEAD"], cwd=project_dir)
        short_hash = hash_result.strip()
        logger.info("[git] Auto-committed task %s: %s", output.task_id, short_hash)
        return short_hash


async def executor_commit(
    project_dir: str,
    round_outputs: dict[str, TaskOutput],
    round_num: int,
) -> str | None:
    """Round-level fallback commit for any remaining unstaged changes.

    In normal flow, commit_single_task handles per-task commits.
    This catches anything that slipped through.
    """
    if not round_outputs:
        return None

    async with _git_lock(project_dir):
        proj = Path(project_dir)
        if not (proj / ".git").exists():
            return None

        status = await _run(["git", "status", "--porcelain"], cwd=project_dir)
        if not status.strip():
            return None

        await _stage_files_safely(project_dir)

        staged = await _run(["git", "diff", "--cached", "--name-only"], cwd=project_dir)
        if not staged.strip():
            return None

        # Build round-level commit message
        outputs = list(round_outputs.values())
        successful = [o for o in outputs if o.status == TaskStatus.COMPLETED]
        failed = [o for o in outputs if o.status == TaskStatus.FAILED]

        if len(successful) == 1:
            first_line = f"feat: {successful[0].summary[:72]}"
        elif successful:
            first_line = f"feat: complete round {round_num} — {len(successful)} tasks"
        else:
            first_line = f"wip: round {round_num} (partial — {len(failed)} failed)"

        body_lines: list[str] = []
        for o in successful:
            body_lines.append(f"  - [{o.task_id}] {o.summary[:100]}")
        for o in failed:
            body_lines.append(f"  - [{o.task_id}] FAILED: {'; '.join(o.issues[:2])[:80]}")

        all_artifacts: list[str] = []
        for o in successful:
            all_artifacts.extend(o.artifacts[:3])
        if all_artifacts:
            unique = list(dict.fromkeys(all_artifacts))[:10]
            body_lines.append(f"\nFiles: {', '.join(unique)}")

        total_cost = sum(o.cost_usd for o in outputs)
        body_lines.append(f"Cost: ${total_cost:.4f}")

        message = first_line + "\n" + "\n".join(body_lines)
        await _run(["git", "commit", "-m", message], cwd=project_dir)

        hash_result = await _run(["git", "rev-parse", "--short", "HEAD"], cwd=project_dir)
        short_hash = hash_result.strip()
        logger.info("[git] Auto-committed round %d: %s", round_num, short_hash)
        return short_hash


# ── Subprocess runner ────────────────────────────────────────────────────────


async def _run(cmd: list[str], cwd: str) -> str:
    """Run a git subprocess command and return stdout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=SUBPROCESS_MEDIUM_TIMEOUT
        )
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            logger.debug("[git] Command %s failed: %s", cmd, err)
            return ""
        return stdout.decode(errors="replace")
    except asyncio.TimeoutError:
        logger.debug("[git] Command %s timed out after %.1fs", cmd, SUBPROCESS_MEDIUM_TIMEOUT)
        return ""
    except OSError as exc:
        logger.debug("[git] Command %s exception: %s", cmd, exc)
        return ""
