"""Minihive CLI — multi-agent orchestrator."""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minihive.contracts import TaskGraph
    from minihive.dag_executor import ExecutionResult


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


def _get_file_tree(project_dir: str, max_files: int = 50) -> str:
    """Get a short file tree for project context."""
    try:
        result = subprocess.run(
            ["find", ".", "-type", "f", "-not", "-path", "./.git/*", "-not", "-path", "./node_modules/*", "-not", "-path", "./.venv/*", "-not", "-path", "./__pycache__/*"],
            cwd=project_dir, capture_output=True, text=True, timeout=5
        )
        files = result.stdout.strip().split("\n")[:max_files]
        return "\n".join(files) if files[0] else "(empty project)"
    except (subprocess.SubprocessError, OSError, ValueError):
        return "(could not read file tree)"


def _print_task_graph(graph: "TaskGraph") -> None:
    """Print the TaskGraph as a readable table."""
    print(f"\n{'='*60}")
    print(f"  Vision: {graph.vision}")
    if graph.epic_breakdown:
        print(f"  Epics: {', '.join(graph.epic_breakdown[:5])}")
    print(f"  Tasks: {len(graph.tasks)}")
    print(f"{'='*60}")
    for t in graph.tasks:
        deps = f" (after: {', '.join(t.depends_on)})" if t.depends_on else ""
        print(f"  [{t.id}] {t.role.value}: {t.goal[:70]}...{deps}")
    print()


def _print_result(result: "ExecutionResult") -> None:
    """Print short completion line (dag_executor prints the full summary)."""
    print("Done.")


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
        await _run_inner(args, cli_path, project_dir)
    finally:
        sys.stdout = original_stdout
        log_file.close()
        print(f"Log saved to: {log_path}")


async def _run_inner(args: argparse.Namespace, cli_path: str, project_dir: str) -> None:
    from minihive.config import MAX_BUDGET_USD
    from minihive.sdk_client import ClaudeSDKManager

    sdk = ClaudeSDKManager(cli_path=cli_path, max_concurrent=args.max_parallel)
    budget = args.budget or MAX_BUDGET_USD

    # Resume from checkpoint -- skip PM planning entirely
    if args.resume:
        from minihive.contracts import TaskGraph
        from minihive.dag_executor import execute_graph
        from minihive.prompts import PROMPT_REGISTRY

        ckpt_path = os.path.join(project_dir, ".minihive", "checkpoint.json")
        if not os.path.isfile(ckpt_path):
            print("No checkpoint found, starting fresh")
        else:
            print(f"Resuming from checkpoint: {ckpt_path}")

        # Provide a dummy graph -- execute_graph will load the real one from checkpoint
        dummy_graph = TaskGraph(
            project_id=os.path.basename(project_dir),
            user_message="(resumed)",
            vision="(resumed from checkpoint)",
            tasks=[],
        )
        result = await execute_graph(
            graph=dummy_graph,
            project_dir=project_dir,
            sdk=sdk,
            prompts=PROMPT_REGISTRY,
            max_budget_usd=budget,
            max_concurrent=args.max_parallel,
            resume=True,
        )
        _print_result(result)
        return

    from minihive.pm_agent import create_task_graph
    from minihive.project_context import build_project_header

    # Build context
    project_context = build_project_header(project_dir)
    file_tree = _get_file_tree(project_dir)
    if file_tree:
        project_context += f"\n\nFile tree:\n{file_tree}"

    # Get task description
    plan_content = ""
    if args.plan_file:
        plan_path = os.path.abspath(args.plan_file)
        if not os.path.isfile(plan_path):
            print(f"Error: plan file not found: {plan_path}", file=sys.stderr)
            sys.exit(1)
        with open(plan_path) as f:
            plan_content = f.read()
        task_desc = f"Execute the plan from: {os.path.basename(plan_path)}"
        print(f"Loading plan from: {plan_path}")
    else:
        task_desc = " ".join(args.task)
        if not task_desc.strip():
            print("Error: provide a task description or --plan-file", file=sys.stderr)
            sys.exit(1)

    project_id = os.path.basename(project_dir)

    # PM planning phase
    print(f"\nPlanning: {task_desc[:80]}...")
    t0 = time.time()
    graph = await create_task_graph(
        sdk=sdk,
        user_message=task_desc,
        project_id=project_id,
        project_context=project_context,
        plan_file_content=plan_content,
    )
    print(f"Plan created in {time.time() - t0:.1f}s")

    if args.dry_run:
        _print_task_graph(graph)
        print("(dry run — not executing)")
        return

    if not args.yes:
        answer = input("Execute this plan? [Y/n] ").strip().lower()
        if answer and answer != "y":
            print("Aborted.")
            return

    # DAG execution phase
    from minihive.dag_executor import execute_graph
    from minihive.prompts import PROMPT_REGISTRY
    print("Executing...\n")
    result = await execute_graph(
        graph=graph,
        project_dir=project_dir,
        sdk=sdk,
        prompts=PROMPT_REGISTRY,
        max_budget_usd=budget,
        max_concurrent=args.max_parallel,
    )
    _print_result(result)


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
