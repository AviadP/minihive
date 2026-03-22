"""Orchestrator -- stripped-down wrapper around the DAG executor.

Provides project scanning, validation, experience extraction, and the
main ``run()`` entry point that replaces ``_run_inner`` in ``__main__.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minihive.contracts import TaskGraph
    from minihive.dag_executor import ExecutionResult
    from minihive.sdk_client import ClaudeSDKManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key-file detection patterns
# ---------------------------------------------------------------------------

_ENTRY_POINT_NAMES = frozenset({
    "__main__.py", "main.py", "app.py", "cli.py", "manage.py", "server.py",
})

_MODEL_KEYWORDS = frozenset({"model", "type", "schema"})

_CONFIG_NAMES = frozenset({
    "config.py", "config.ts", "settings.py", "pyproject.toml", "package.json",
})

_EXCLUDED_DIRS = frozenset({
    ".git", ".venv", "__pycache__", "node_modules", ".minihive",
})

_MAX_KEY_FILES = 10
_SNIPPET_MAX_LINES = 80


# ===================================================================
# 1. scan_project() -- Deep project scan
# ===================================================================


def _find_key_files(project_dir: str, file_list: list[str]) -> list[str]:
    """Filter *file_list* for entry points, models/types, and config files."""
    result: list[str] = []
    for rel in file_list:
        basename = os.path.basename(rel).lower()
        if basename in _ENTRY_POINT_NAMES:
            result.append(rel)
            continue
        if basename in _CONFIG_NAMES:
            result.append(rel)
            continue
        stem = os.path.splitext(basename)[0]
        if any(kw in stem for kw in _MODEL_KEYWORDS):
            result.append(rel)
    return result[:_MAX_KEY_FILES]


def _read_snippet(filepath: str, max_lines: int = _SNIPPET_MAX_LINES) -> str:
    """Read the first *max_lines* of *filepath* safely."""
    try:
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            lines = []
            for i, line in enumerate(fh):
                if i >= max_lines:
                    break
                lines.append(line)
            return "".join(lines)
    except OSError:
        return ""


def _load_prior_context(project_dir: str) -> str:
    """Read .minihive/experience.md + todo.md if they exist."""
    parts: list[str] = []
    for name in ("experience.md", "todo.md"):
        path = os.path.join(project_dir, ".minihive", name)
        content = _read_snippet(path, max_lines=120)
        if content:
            parts.append(f"### {name}\n{content}")
    return "\n\n".join(parts)


def scan_project(project_dir: str) -> str:
    """Build rich codebase summary for PM planning context."""
    # 1. File tree
    try:
        result = subprocess.run(
            [
                "find", ".", "-type", "f",
                *[arg for d in _EXCLUDED_DIRS for arg in ("-not", "-path", f"./{d}/*")],
            ],
            cwd=project_dir, capture_output=True, text=True, timeout=10,
        )
        all_files = [f for f in result.stdout.strip().split("\n") if f]
    except (subprocess.SubprocessError, OSError):
        all_files = []

    tree_str = "\n".join(all_files[:80]) if all_files else "(empty project)"

    # 2. Key files
    key_files = _find_key_files(project_dir, all_files)
    snippets: list[str] = []
    for rel in key_files:
        abs_path = os.path.join(project_dir, rel.lstrip("./"))
        content = _read_snippet(abs_path)
        if content:
            snippets.append(f"### {rel}\n```\n{content}```")

    # 3. Prior context
    prior = _load_prior_context(project_dir)

    # Assemble
    sections = [f"## File Tree\n{tree_str}"]
    if snippets:
        sections.append("## Key Files\n" + "\n\n".join(snippets))
    if prior:
        sections.append(f"## Prior Context\n{prior}")
    return "\n\n".join(sections)


# ===================================================================
# 2. RunningContext + callbacks
# ===================================================================


@dataclass
class RunningContext:
    """Mutable state accumulated during DAG execution."""

    completed_summaries: list[str] = field(default_factory=list)
    files_changed: set[str] = field(default_factory=set)
    issues_found: list[str] = field(default_factory=list)
    total_cost: float = 0.0
    budget_warned: bool = False


async def _on_task_done(
    task: "TaskInput",
    output: "TaskOutput",
    ctx: RunningContext,
    project_dir: str,
    max_budget: float,
) -> None:
    """Callback invoked after each task completes."""
    from minihive.contracts import TaskInput, TaskOutput  # noqa: F811

    # One-liner summary
    ctx.completed_summaries.append(f"[{task.id}] {output.summary[:120]}")

    # Track changed files via git
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--name-only", "HEAD",
            cwd=project_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        for line in stdout.decode().strip().split("\n"):
            if line:
                ctx.files_changed.add(line)
    except (OSError, ValueError):
        pass

    # Track cost
    ctx.total_cost += output.cost_usd

    # Budget warning at 80%
    if ctx.total_cost > max_budget * 0.8 and not ctx.budget_warned:
        print(f"  WARNING: Budget usage at {ctx.total_cost / max_budget * 100:.0f}% "
              f"(${ctx.total_cost:.2f} / ${max_budget:.2f})")
        ctx.budget_warned = True

    # Track issues
    if output.issues:
        for issue in output.issues:
            ctx.issues_found.append(f"[{task.id}] {issue}")


async def _on_round_done(
    round_num: int,
    completed: dict,
    total_cost: float,
    ctx: RunningContext,
) -> None:
    """Callback invoked after each execution round."""
    total = len(completed)
    successful = sum(1 for o in completed.values() if o.is_successful())
    print(f"  Round {round_num} complete: {successful}/{total} tasks succeeded "
          f"(${total_cost:.2f})")


# ===================================================================
# 3. validate_completion()
# ===================================================================


@dataclass
class ValidationResult:
    """Result of post-execution validation."""

    passed: bool
    checks: list[tuple[str, bool, str]]  # (name, passed, detail)
    fixable_issues: list[str] = field(default_factory=list)


def _check_tasks_completed(result: "ExecutionResult") -> tuple[bool, str]:
    """All tasks in result.completed_tasks succeeded."""
    failed = [
        tid for tid, out in result.completed_tasks.items()
        if not out.is_successful()
    ]
    if not failed:
        return True, "All tasks completed successfully"
    return False, f"{len(failed)} task(s) failed: {', '.join(failed[:5])}"


def _check_writers_produced_files(
    graph: "TaskGraph", result: "ExecutionResult", project_dir: str,
) -> tuple[bool, str]:
    """Writer agents have artifacts AND files exist on disk."""
    from minihive.contracts import WRITER_ROLES

    missing: list[str] = []
    for task in graph.tasks:
        if task.role not in WRITER_ROLES:
            continue
        output = result.completed_tasks.get(task.id)
        if output is None or not output.is_successful():
            continue
        if not output.artifacts:
            missing.append(f"{task.id}: no artifacts listed")
            continue
        for artifact_path in output.artifacts:
            full_path = os.path.join(project_dir, artifact_path)
            if not os.path.exists(full_path):
                missing.append(f"{task.id}: {artifact_path} missing on disk")

    if not missing:
        return True, "All writer artifacts present on disk"
    return False, "; ".join(missing[:5])


def _check_reviewer_included(graph: "TaskGraph") -> tuple[bool, str]:
    """At least one task with role REVIEWER."""
    from minihive.contracts import AgentRole

    has_reviewer = any(t.role == AgentRole.REVIEWER for t in graph.tasks)
    if has_reviewer:
        return True, "Reviewer task present"
    return False, "No reviewer task in graph"


def _check_tester_included(graph: "TaskGraph") -> tuple[bool, str]:
    """If len(graph.tasks) >= 5, at least one TEST_ENGINEER."""
    from minihive.contracts import AgentRole

    if len(graph.tasks) < 5:
        return True, "Small graph -- tester not required"
    has_tester = any(t.role == AgentRole.TEST_ENGINEER for t in graph.tasks)
    if has_tester:
        return True, "Test engineer task present"
    return False, "No test engineer in graph with 5+ tasks"


def _check_no_duplicate_models(result: "ExecutionResult") -> tuple[bool, str]:
    """No two tasks created files with model/type/schema in name."""
    model_owners: dict[str, str] = {}  # filename -> task_id
    duplicates: list[str] = []
    for tid, output in result.completed_tasks.items():
        if not output.is_successful():
            continue
        for artifact_path in output.artifacts:
            basename = os.path.basename(artifact_path).lower()
            stem = os.path.splitext(basename)[0]
            if any(kw in stem for kw in _MODEL_KEYWORDS):
                if artifact_path in model_owners:
                    duplicates.append(
                        f"{artifact_path} owned by {model_owners[artifact_path]} and {tid}"
                    )
                else:
                    model_owners[artifact_path] = tid
    if not duplicates:
        return True, "No duplicate model files"
    return False, "; ".join(duplicates[:3])


def _check_imports(project_dir: str) -> tuple[bool, str]:
    """Try to import the main package via subprocess."""
    project_name = os.path.basename(project_dir)
    # Look for a src/<name> or <name> package
    src_init = os.path.join(project_dir, "src", project_name, "__init__.py")
    root_init = os.path.join(project_dir, project_name, "__init__.py")

    if not os.path.exists(src_init) and not os.path.exists(root_init):
        return True, "No importable package detected -- skipped"

    try:
        result = subprocess.run(
            ["python", "-c", f"import {project_name}"],
            cwd=project_dir,
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "PYTHONPATH": os.path.join(project_dir, "src")},
        )
        if result.returncode == 0:
            return True, f"import {project_name} succeeded"
        return False, f"import {project_name} failed: {result.stderr[:200]}"
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"Import check error: {exc}"


def validate_completion(
    graph: "TaskGraph",
    result: "ExecutionResult",
    project_dir: str,
) -> ValidationResult:
    """Run post-execution validation checks."""
    checks: list[tuple[str, bool, str]] = []
    fixable: list[str] = []

    validators = [
        ("tasks_completed", _check_tasks_completed, (result,)),
        ("writers_produced_files", _check_writers_produced_files, (graph, result, project_dir)),
        ("reviewer_included", _check_reviewer_included, (graph,)),
        ("tester_included", _check_tester_included, (graph,)),
        ("no_duplicate_models", _check_no_duplicate_models, (result,)),
        ("imports_ok", _check_imports, (project_dir,)),
    ]

    for name, fn, args in validators:
        passed, detail = fn(*args)
        checks.append((name, passed, detail))
        if not passed and name in ("tester_included", "imports_ok"):
            fixable.append(f"{name}: {detail}")

    all_passed = all(passed for _, passed, _ in checks)

    # Print report
    print(f"\n{'='*60}")
    print("  VALIDATION REPORT")
    print(f"{'='*60}")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")
    if fixable:
        print(f"\n  Fixable issues: {len(fixable)}")
        for issue in fixable:
            print(f"    - {issue}")
    print(f"{'='*60}\n")

    return ValidationResult(passed=all_passed, checks=checks, fixable_issues=fixable)


# ===================================================================
# 4. write_task_ledger() + extract_experience()
# ===================================================================


def write_task_ledger(
    project_dir: str,
    graph: "TaskGraph",
    result: "ExecutionResult",
    validation: ValidationResult,
) -> None:
    """Write .minihive/todo.md with execution summary."""
    ledger_dir = Path(project_dir) / ".minihive"
    ledger_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# Task Ledger -- {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Vision: {graph.vision}",
        "",
    ]

    # Completed
    completed = [
        (tid, out) for tid, out in result.completed_tasks.items()
        if out.is_successful()
    ]
    lines.append("## Completed")
    for tid, out in completed:
        lines.append(f"- [{tid}] {out.summary[:120]}")
    if not completed:
        lines.append("- (none)")
    lines.append("")

    # Failed
    failed = [
        (tid, out) for tid, out in result.completed_tasks.items()
        if not out.is_successful()
    ]
    lines.append("## Failed")
    for tid, out in failed:
        detail = out.failure_details or out.summary
        lines.append(f"- [{tid}] {detail[:150]}")
    if not failed:
        lines.append("- (none)")
    lines.append("")

    # Validation
    lines.append("## Validation")
    for name, passed, detail in validation.checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"- [{status}] {name}: {detail}")
    lines.append("")

    # Pending
    if validation.fixable_issues:
        lines.append("## Pending")
        for issue in validation.fixable_issues:
            lines.append(f"- {issue}")
        lines.append("")

    (ledger_dir / "todo.md").write_text("\n".join(lines), encoding="utf-8")


async def extract_experience(
    sdk: "ClaudeSDKManager",
    project_dir: str,
    graph: "TaskGraph",
    result: "ExecutionResult",
) -> None:
    """Use an LLM call to distill 3-5 lessons from the execution."""
    from minihive.isolated_query import isolated_query

    # Build a compact execution summary
    summary_parts: list[str] = [f"Vision: {graph.vision}"]
    for tid, out in result.completed_tasks.items():
        status = "OK" if out.is_successful() else "FAIL"
        summary_parts.append(f"[{status}] {tid}: {out.summary[:100]}")
    summary_parts.append(f"Total cost: ${result.total_cost_usd:.2f}")
    exec_summary = "\n".join(summary_parts)

    prompt = (
        "Given this execution summary, extract 3-5 concrete lessons learned "
        "(what worked, what failed, patterns to reuse/avoid). "
        "Be specific -- include file names and error types.\n\n"
        f"{exec_summary}"
    )

    response = await isolated_query(
        sdk,
        prompt=prompt,
        system_prompt="You are a senior engineering retrospective facilitator.",
        cwd=project_dir,
        allowed_tools=[],
        max_turns=1,
        max_budget_usd=2.0,
    )

    if response.is_error:
        logger.warning("Experience extraction failed: %s", response.error_message)
        return

    exp_path = Path(project_dir) / ".minihive" / "experience.md"
    exp_path.parent.mkdir(parents=True, exist_ok=True)

    existing = ""
    if exp_path.exists():
        existing = exp_path.read_text(encoding="utf-8")

    entry = (
        f"\n\n---\n## {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"{response.text.strip()}\n"
    )
    exp_path.write_text(existing + entry, encoding="utf-8")


# ===================================================================
# 5. _try_replan()
# ===================================================================


async def _try_replan(
    sdk: "ClaudeSDKManager",
    project_dir: str,
    graph: "TaskGraph",
    result: "ExecutionResult",
    validation: ValidationResult,
    prompts: dict[str, str] | None,
    max_budget: float,
    max_concurrent: int,
) -> "ExecutionResult | None":
    """Attempt a one-shot replan to fix fixable_issues."""
    if not validation.fixable_issues:
        return None

    from minihive.dag_executor import execute_graph
    from minihive.pm_agent import create_task_graph
    from minihive.project_context import build_project_header

    # Build context from completed work
    completed_context = "\n".join(
        f"- [{tid}] {out.summary[:100]}"
        for tid, out in result.completed_tasks.items()
    )
    failure_context = "\n".join(f"- {issue}" for issue in validation.fixable_issues)

    fix_message = (
        f"Fix these issues from the previous run:\n{failure_context}\n\n"
        f"Already completed:\n{completed_context}"
    )

    project_context = build_project_header(project_dir)
    project_id = os.path.basename(project_dir)

    print("\nRe-planning to fix validation issues...")
    fix_graph = await create_task_graph(
        sdk=sdk,
        user_message=fix_message,
        project_id=project_id,
        project_context=project_context,
        project_dir=project_dir,
    )

    print("Executing fix-up graph...")
    fix_result = await execute_graph(
        graph=fix_graph,
        project_dir=project_dir,
        sdk=sdk,
        prompts=prompts,
        max_budget_usd=max_budget,
        max_concurrent=max_concurrent,
    )
    return fix_result


# ===================================================================
# Helpers moved from __main__.py
# ===================================================================


def _get_file_tree(project_dir: str, max_files: int = 50) -> str:
    """Get a short file tree for project context."""
    try:
        result = subprocess.run(
            [
                "find", ".", "-type", "f",
                "-not", "-path", "./.git/*",
                "-not", "-path", "./node_modules/*",
                "-not", "-path", "./.venv/*",
                "-not", "-path", "./__pycache__/*",
            ],
            cwd=project_dir, capture_output=True, text=True, timeout=5,
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


# ===================================================================
# 6. run() -- Main entry point
# ===================================================================


async def run(args: argparse.Namespace, cli_path: str, project_dir: str) -> None:
    """Main orchestration entry point -- replaces _run_inner in __main__.py."""
    from minihive.config import MAX_BUDGET_USD
    from minihive.sdk_client import ClaudeSDKManager

    sdk = ClaudeSDKManager(cli_path=cli_path, max_concurrent=args.max_parallel)
    budget = args.budget or MAX_BUDGET_USD

    # --- Resume path ---
    if args.resume:
        from minihive.contracts import TaskGraph as TG
        from minihive.dag_executor import execute_graph
        from minihive.prompts import PROMPT_REGISTRY

        ckpt_path = os.path.join(project_dir, ".minihive", "checkpoint.json")
        if not os.path.isfile(ckpt_path):
            print("No checkpoint found, starting fresh")
        else:
            print(f"Resuming from checkpoint: {ckpt_path}")

        dummy_graph = TG(
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
        # Skip to validation
        validation = validate_completion(dummy_graph, result, project_dir)
        write_task_ledger(project_dir, dummy_graph, result, validation)
        print("Done.")
        return

    # --- Normal path ---
    # 1. Scan project
    rich_context = scan_project(project_dir)

    # 2. Build project header
    from minihive.project_context import build_project_header
    project_context = build_project_header(project_dir)
    if rich_context:
        project_context += f"\n\n{rich_context}"

    # 3. Get task description
    plan_content = ""
    if args.plan_file:
        plan_path = os.path.abspath(args.plan_file)
        if not os.path.isfile(plan_path):
            print(f"Error: plan file not found: {plan_path}")
            return
        with open(plan_path, encoding="utf-8") as f:
            plan_content = f.read()
        task_desc = f"Execute the plan from: {os.path.basename(plan_path)}"
        print(f"Loading plan from: {plan_path}")
    else:
        task_desc = " ".join(args.task)
        if not task_desc.strip():
            print("Error: provide a task description or --plan-file")
            return

    project_id = os.path.basename(project_dir)

    # 4. PM planning
    from minihive.pm_agent import create_task_graph

    print(f"\nPlanning: {task_desc[:80]}...")
    t0 = time.time()
    graph = await create_task_graph(
        sdk=sdk,
        user_message=task_desc,
        project_id=project_id,
        project_context=project_context,
        plan_file_content=plan_content,
        project_dir=project_dir,
    )
    print(f"Plan created in {time.time() - t0:.1f}s")

    # 5. Dry run
    if args.dry_run:
        _print_task_graph(graph)
        print("(dry run -- not executing)")
        return

    # 6. Confirmation
    _print_task_graph(graph)
    if not args.yes:
        answer = input("Execute this plan? [Y/n] ").strip().lower()
        if answer and answer != "y":
            print("Aborted.")
            return

    # 7. Execute
    from minihive.dag_executor import execute_graph
    from minihive.prompts import PROMPT_REGISTRY

    ctx = RunningContext()

    async def task_done_cb(task: object, output: object) -> None:
        await _on_task_done(task, output, ctx, project_dir, budget)

    async def round_done_cb(round_num: int, completed: dict, total_cost: float) -> None:
        await _on_round_done(round_num, completed, total_cost, ctx)

    print("Executing...\n")
    result = await execute_graph(
        graph=graph,
        project_dir=project_dir,
        sdk=sdk,
        prompts=PROMPT_REGISTRY,
        max_budget_usd=budget,
        max_concurrent=args.max_parallel,
        on_task_done=task_done_cb,
        on_round_done=round_done_cb,
    )

    # 8. Validate
    validation = validate_completion(graph, result, project_dir)

    # 9. Try replan if fixable issues
    if not validation.passed and validation.fixable_issues:
        fix_result = await _try_replan(
            sdk, project_dir, graph, result, validation,
            PROMPT_REGISTRY, budget, args.max_parallel,
        )
        if fix_result is not None:
            # Re-validate after fix
            validation = validate_completion(graph, fix_result, project_dir)
            # Merge costs
            result.total_cost_usd += fix_result.total_cost_usd

    # 10. Write ledger
    write_task_ledger(project_dir, graph, result, validation)

    # 11. Extract experience (non-critical)
    try:
        await extract_experience(sdk, project_dir, graph, result)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Experience extraction failed: %s", exc)

    # 12. Final summary
    print(f"\n{'='*60}")
    print("  ORCHESTRATION COMPLETE")
    print(f"  Total cost: ${result.total_cost_usd:.2f}")
    if ctx.files_changed:
        print(f"  Files changed: {len(ctx.files_changed)}")
    print(f"  Validation: {'PASSED' if validation.passed else 'FAILED'}")
    print(f"{'='*60}")
