"""Minihive CLI — multi-agent orchestrator."""

import argparse
import asyncio
import os
import shutil
import sys
import time
from pathlib import Path


class _TeeWriter:
    """Write to both the original stream and a log file."""

    def __init__(self, original: object, log_file: object) -> None:
        self.original = original
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.original.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        self.original.flush()
        self.log_file.flush()


def _check_claude_cli() -> str:
    """Return path to claude CLI or exit with error."""
    from minihive.config import CLAUDE_CLI_PATH
    path = shutil.which(CLAUDE_CLI_PATH)
    if not path:
        print(f"Error: Claude CLI not found ('{CLAUDE_CLI_PATH}'). Install with: npm install -g @anthropic-ai/claude-code", file=sys.stderr)
        sys.exit(1)
    return path


def _check_git_repo(project_dir: str) -> None:
    """Verify project_dir is a git repo."""
    if not os.path.isdir(os.path.join(project_dir, ".git")):
        print(f"Error: {project_dir} is not a git repository. Run 'git init' first.", file=sys.stderr)
        sys.exit(1)


async def _run(args: argparse.Namespace) -> None:
    cli_path = _check_claude_cli()
    project_dir = os.path.abspath(args.project_dir)
    _check_git_repo(project_dir)

    # Tee stdout to .minihive/run.log so background runs are reviewable
    log_dir = Path(project_dir) / ".minihive"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    log_file = open(log_path, "a")  # noqa: SIM115
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"  minihive run — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"  project: {project_dir}\n")
    log_file.write(f"  args: {' '.join(sys.argv[1:])}\n")
    log_file.write(f"{'='*60}\n")
    log_file.flush()
    original_stdout = sys.stdout
    sys.stdout = _TeeWriter(original_stdout, log_file)
    try:
        from minihive.orchestrator import run
        await run(args, cli_path, project_dir)
    finally:
        sys.stdout = original_stdout
        log_file.close()
        print(f"Log saved to: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="minihive",
        description="Multi-agent orchestrator — PM plans, agents build in parallel, test, review, commit",
    )
    parser.add_argument("task", nargs="*", help="Task description")
    parser.add_argument("--project-dir", default=".", help="Project directory (default: cwd)")
    parser.add_argument("--plan-file", help="Markdown plan file to convert and execute")
    parser.add_argument("--budget", type=float, help="Max budget in USD")
    parser.add_argument("--max-parallel", type=int, default=4, help="Max parallel agents")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only, don't execute")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
