"""DAG Executor -- core execution engine with self-healing.

Runs a TaskGraph to completion using round-based scheduling:
- Writers with overlapping file scopes run sequentially
- Readers run in parallel
- Failed tasks get retried, then remediated via auto-generated fix tasks

Simplified from Hivemind's ~2900-line dag_executor.py.  Removed:
SQLite checkpoints, watchdog, blackboard, skills, reflexion, dynamic
spawner, dashboard events, agent log files, commit approval callbacks.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path as _Path
from typing import Any

from minihive.config import (
    DAG_MAX_CONCURRENT_NODES,
    MAX_DAG_ROUNDS,
    MAX_REMEDIATION_DEPTH,
    MAX_TASK_RETRIES,
    MAX_TOTAL_REMEDIATIONS,
    get_agent_budget,
    get_agent_timeout,
    get_agent_turns,
)
from minihive.contracts import (
    FailureCategory,
    TaskGraph,
    TaskInput,
    TaskOutput,
    TaskStatus,
    WRITER_ROLES,
    READER_ROLES,
    classify_failure,
    create_remediation_task,
    extract_task_output,
    task_input_to_prompt,
)
from minihive.file_context import ArtifactRegistry
from minihive.project_context import build_project_header
from minihive.git_ops import commit_single_task
from minihive.sdk_client import ClaudeSDKManager

logger = logging.getLogger(__name__)

# Categories where retrying wastes money
_NO_RETRY_CATEGORIES: frozenset[FailureCategory] = frozenset({
    FailureCategory.UNCLEAR_GOAL,
    FailureCategory.PERMISSION,
    FailureCategory.EXTERNAL,
})

# Two-phase execution constants
_SUMMARY_PHASE_TURNS = 5
_MIN_WORK_TURNS_FOR_SUMMARY = 3


def _print_progress(completed: dict, total: int, round_num: int, cost: float) -> None:
    done = len(completed)
    success = sum(1 for o in completed.values() if o.is_successful())
    failed = done - success
    bar_len = 30
    filled = int(bar_len * done / total) if total else 0
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    print(f"\n  [{bar}] {done}/{total} tasks | Round {round_num} | ${cost:.2f}")
    if failed:
        print(f"  ({failed} failed)")


# ---------------------------------------------------------------------------
# Checkpoint — save/load/clear execution state
# ---------------------------------------------------------------------------

_CHECKPOINT_FILE = ".minihive/checkpoint.json"


def _save_checkpoint(project_dir: str, graph: TaskGraph, completed: dict[str, TaskOutput],
                     retries: dict[str, int], total_cost: float, round_num: int,
                     healing_history: list[dict]) -> None:
    """Save execution state to .minihive/checkpoint.json for resume."""
    ckpt_dir = _Path(project_dir) / ".minihive"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "graph": graph.model_dump(mode="json"),
        "completed": {tid: out.model_dump(mode="json") for tid, out in completed.items()},
        "retries": retries,
        "total_cost": total_cost,
        "round_num": round_num,
        "healing_history": healing_history,
    }
    ckpt_path = ckpt_dir / "checkpoint.json"
    ckpt_path.write_text(_json.dumps(data, indent=2, default=str))
    logger.info("Checkpoint saved: %d/%d tasks, round %d", len(completed), len(graph.tasks), round_num)


def _load_checkpoint(project_dir: str) -> dict | None:
    """Load checkpoint from .minihive/checkpoint.json, or None if not found."""
    ckpt_path = _Path(project_dir) / _CHECKPOINT_FILE
    if not ckpt_path.exists():
        return None
    try:
        data = _json.loads(ckpt_path.read_text())
        return data
    except (OSError, _json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to load checkpoint: %s", exc)
        return None


def _clear_checkpoint(project_dir: str) -> None:
    """Remove checkpoint file after successful completion."""
    ckpt_path = _Path(project_dir) / _CHECKPOINT_FILE
    if ckpt_path.exists():
        ckpt_path.unlink()
        logger.info("Checkpoint cleared")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Result of a full DAG execution."""

    completed_tasks: dict[str, TaskOutput] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    healing_history: list[dict[str, str]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        success = sum(1 for o in self.completed_tasks.values() if o.is_successful())
        failed = sum(1 for o in self.completed_tasks.values() if not o.is_successful())
        lines = [
            f"Tasks: {success + failed} total, {success} succeeded, {failed} failed",
            f"Total cost: ${self.total_cost_usd:.4f}",
        ]
        if self.healing_history:
            lines.append("Self-healing actions:")
            for h in self.healing_history:
                lines.append(f"  - {h.get('action', 'unknown')}: {h.get('detail', '')}")
        return "\n".join(lines)


class FileLockManager:
    """Per-file asyncio locks to prevent writer conflicts.

    Sorted acquisition order prevents ABBA deadlocks.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._meta_lock: asyncio.Lock | None = None

    def _get_meta_lock(self) -> asyncio.Lock:
        """Lazily create meta lock inside a running event loop."""
        if self._meta_lock is None:
            self._meta_lock = asyncio.Lock()
        return self._meta_lock

    async def acquire(self, file_paths: list[str], timeout: float = 120.0) -> bool:
        """Acquire locks for all paths atomically.  Returns False on timeout."""
        if not file_paths:
            return True

        sorted_paths = sorted(set(file_paths))

        async with self._get_meta_lock():
            for fp in sorted_paths:
                if fp not in self._locks:
                    self._locks[fp] = asyncio.Lock()

        acquired: list[str] = []
        deadline = asyncio.get_event_loop().time() + timeout

        try:
            for fp in sorted_paths:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    self._release_paths(acquired)
                    return False
                try:
                    await asyncio.wait_for(self._locks[fp].acquire(), timeout=remaining)
                    acquired.append(fp)
                except TimeoutError:
                    self._release_paths(acquired)
                    return False
        except (RuntimeError, OSError):
            self._release_paths(acquired)
            raise

        return True

    def release(self, file_paths: list[str]) -> None:
        """Release locks for the given file paths."""
        self._release_paths(sorted(set(file_paths)))

    def _release_paths(self, paths: list[str]) -> None:
        for fp in paths:
            lock = self._locks.get(fp)
            if lock is not None and lock.locked():
                lock.release()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def execute_graph(
    graph: TaskGraph,
    project_dir: str,
    sdk: ClaudeSDKManager,
    prompts: dict[str, str] | None = None,
    max_budget_usd: float = 50.0,
    max_concurrent: int = DAG_MAX_CONCURRENT_NODES,
    resume: bool = False,
) -> ExecutionResult:
    """Execute a TaskGraph to completion with self-healing.

    Args:
        graph:          The PM's execution plan.
        project_dir:    Working directory for all agents.
        sdk:            ClaudeSDKManager instance.
        prompts:        role_name -> system prompt.
        max_budget_usd: Hard budget cap across the entire graph.
        max_concurrent: Max DAG nodes running simultaneously.
        resume:         If True, attempt to restore from checkpoint.

    Returns:
        ExecutionResult with all outputs, cost, and healing history.
    """
    completed: dict[str, TaskOutput] = {}
    retries: dict[str, int] = {}
    total_cost = 0.0
    healing_history: list[dict] = []
    start_round = 0

    # Restore from checkpoint if requested
    if resume:
        ckpt = _load_checkpoint(project_dir)
        if ckpt is not None:
            graph = TaskGraph(**ckpt["graph"])
            completed = {tid: TaskOutput(**v) for tid, v in ckpt["completed"].items()}
            retries = ckpt.get("retries", {})
            total_cost = ckpt.get("total_cost", 0.0)
            start_round = ckpt.get("round_num", 0)
            healing_history = ckpt.get("healing_history", [])
            print(f"  Resuming from checkpoint: {len(completed)}/{len(graph.tasks)} tasks done, round {start_round}")
        else:
            print("  No checkpoint found, starting fresh")

    # Validate DAG structure
    errors = graph.validate_dag()
    if errors:
        raise ValueError(f"Invalid DAG: {errors}")

    concurrency = max(1, max_concurrent)

    file_lock_manager = FileLockManager()

    ctx: dict[str, Any] = {
        "graph": graph,
        "project_dir": project_dir,
        "sdk": sdk,
        "prompts": prompts,
        "max_budget_usd": max_budget_usd,
        "completed": completed,
        "retries": retries,
        "total_cost": total_cost,
        "remediation_count": 0,
        "healing_history": healing_history,
        "task_counter": len(graph.tasks),
        "graph_lock": asyncio.Lock(),
        "session_ids": {},         # session_key -> session_id
        "semaphore": asyncio.Semaphore(concurrency),
        "artifact_registry": ArtifactRegistry(project_dir),
        "concurrency": concurrency,
        "file_lock_manager": file_lock_manager,
        "start_round": start_round,
    }

    print(f"\n{'='*60}")
    print(f"  MINIHIVE \u2014 Executing {len(graph.tasks)} tasks")
    print(f"  Vision: {graph.vision}")
    print(f"{'='*60}")
    for t in graph.tasks:
        deps = f" (after: {', '.join(t.depends_on)})" if t.depends_on else ""
        print(f"  [{t.id}] {t.role.value}: {t.goal[:70]}{deps}")
    print(f"{'='*60}\n")

    result = await _execute_graph_inner(ctx)
    return result


# ---------------------------------------------------------------------------
# Core execution loop
# ---------------------------------------------------------------------------


async def _execute_graph_inner(ctx: dict[str, Any]) -> ExecutionResult:
    """Round-based execution loop with self-healing."""
    graph: TaskGraph = ctx["graph"]
    completed: dict[str, TaskOutput] = ctx["completed"]
    max_budget: float = ctx["max_budget_usd"]
    project_dir: str = ctx["project_dir"]
    round_num = ctx.get("start_round", 0)

    while not graph.is_complete(completed):
        round_num += 1

        if round_num > MAX_DAG_ROUNDS:
            pending = [t.id for t in graph.tasks if t.id not in completed]
            print(f"[DAG] Safety limit: exceeded {MAX_DAG_ROUNDS} rounds. Pending: {pending}")
            break

        ready = graph.ready_tasks(completed)

        if not ready:
            if graph.has_failed(completed):
                healed = await _try_self_heal(ctx)
                if healed:
                    continue
            # Deadlock -- no ready tasks, graph not complete
            pending = [t.id for t in graph.tasks if t.id not in completed]
            print(f"[DAG] Deadlock: no ready tasks. Pending: {pending}")
            break

        if ctx["total_cost"] >= max_budget:
            print(f"[DAG] Budget exhausted (${ctx['total_cost']:.2f} >= ${max_budget:.2f})")
            break

        # Plan and execute batches
        batches = _plan_batches(ready)
        print(f"\n--- Round {round_num} ---")

        for batch in batches:
            subtasks = [
                asyncio.create_task(
                    _run_with_semaphore(task, ctx),
                    name=f"dag-{task.id}",
                )
                for task in batch
            ]
            done, _ = await asyncio.wait(subtasks, return_when=asyncio.ALL_COMPLETED)

            # Collect results
            for task, subtask in zip(batch, subtasks, strict=False):
                if subtask.cancelled():
                    output = TaskOutput(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        summary="Task cancelled",
                        confidence=0.0,
                    )
                elif subtask.exception() is not None:
                    exc = subtask.exception()
                    output = TaskOutput(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        summary=f"Exception: {type(exc).__name__}: {str(exc)[:200]}",
                        issues=[str(exc)[:300]],
                        confidence=0.0,
                    )
                    output.failure_category = classify_failure(output)
                else:
                    output = subtask.result()

                completed[task.id] = output
                ctx["total_cost"] += output.cost_usd

                if output.is_successful():
                    print(f"  \u2713 [{task.id}] completed (${output.cost_usd:.2f})")
                else:
                    print(f"  \u2717 [{task.id}] FAILED: {output.summary[:80]}")

                # Successful remediation unblocks downstream
                if output.is_successful() and task.is_remediation and task.original_task_id:
                    completed[task.original_task_id] = output

                # Register artifacts for downstream context
                if output.is_successful():
                    try:
                        ctx["artifact_registry"].register(task.id, output)
                    except (ValueError, TypeError, OSError) as exc:
                        logger.warning("[DAG] Artifact registration failed: %s", exc)

                    # Per-task git commit
                    try:
                        sha = await commit_single_task(ctx["project_dir"], output)
                        if sha:
                            logger.info("[DAG] Task %s committed: %s", task.id, sha)
                    except (OSError, RuntimeError) as exc:
                        logger.warning("[DAG] Per-task commit failed: %s", exc)

                # Handle failures
                if not output.is_successful():
                    await _handle_failure(task, output, ctx)

        _print_progress(ctx["completed"], len(graph.tasks), round_num, ctx["total_cost"])

        # Save checkpoint after each round
        _save_checkpoint(
            project_dir, graph, completed, ctx["retries"],
            ctx["total_cost"], round_num, ctx["healing_history"],
        )

    # Clear checkpoint on successful completion
    if graph.is_complete(completed):
        _clear_checkpoint(project_dir)

    healing_history = ctx["healing_history"]
    print(f"\n{'='*60}")
    print(f"  EXECUTION COMPLETE")
    print(f"  Tasks: {len(ctx['completed'])}/{len(graph.tasks)}")
    print(f"  Cost: ${ctx['total_cost']:.2f}")
    if healing_history:
        print(f"  Self-healed: {len(healing_history)} tasks")
    print(f"{'='*60}")

    return ExecutionResult(
        completed_tasks=completed,
        total_cost_usd=ctx["total_cost"],
        healing_history=ctx["healing_history"],
    )


# ---------------------------------------------------------------------------
# Semaphore + file-lock wrapper
# ---------------------------------------------------------------------------


async def _run_with_semaphore(task: TaskInput, ctx: dict[str, Any]) -> TaskOutput:
    """Acquire semaphore slot, optionally file locks, then run the task."""
    is_writer = task.role in WRITER_ROLES
    locked_files: list[str] = []

    file_lock_manager: FileLockManager = ctx["file_lock_manager"]

    async with ctx["semaphore"]:
        if is_writer and task.files_scope:
            locked_files = list(task.files_scope)
            acquired = await file_lock_manager.acquire(locked_files)
            if not acquired:
                return TaskOutput(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    summary="Timed out waiting for file locks",
                    issues=["Could not acquire file-level locks within timeout"],
                    confidence=0.0,
                )

        try:
            return await _run_single_task(task, ctx)
        finally:
            if locked_files:
                file_lock_manager.release(locked_files)


# ---------------------------------------------------------------------------
# Single task execution (two-phase)
# ---------------------------------------------------------------------------


async def _run_single_task(task: TaskInput, ctx: dict[str, Any]) -> TaskOutput:
    """Two-phase execution: WORK then SUMMARY.

    Phase 1 (WORK): full tools, does the actual work.
    Phase 2 (SUMMARY): tools disabled, extracts structured JSON output.
    """
    from minihive.isolated_query import isolated_query

    role_name = task.role.value
    max_turns = get_agent_turns(role_name)
    retry_count = ctx["retries"].get(task.id, 0)
    task_timeout = get_agent_timeout(role_name, retry_attempt=retry_count)

    print(f"  \u25b6 [{task.id}] {role_name}: {task.goal[:60]}...")

    # Gather upstream context
    context_outputs = {
        tid: ctx["completed"][tid]
        for tid in task.context_from
        if tid in ctx["completed"]
    }

    # Build prompt
    graph: TaskGraph = ctx["graph"]
    prompt = task_input_to_prompt(
        task,
        context_outputs,
        graph_vision=graph.vision,
        graph_epics=graph.epic_breakdown,
    )

    # Inject project CLAUDE.md rules at the top of the prompt
    project_context = build_project_header(ctx["project_dir"])
    if project_context:
        prompt = (
            "<project_rules>\n"
            "MANDATORY — The project owner defined these rules. You MUST follow them:\n"
            f"{project_context}\n"
            "</project_rules>\n\n"
        ) + prompt

    # Get system prompt
    prompts = ctx["prompts"]
    if prompts is not None:
        system_prompt = prompts.get(
            role_name,
            prompts.get("backend_developer", "You are an expert software engineer."),
        )
    else:
        from minihive.prompts import get_specialist_prompt
        system_prompt = get_specialist_prompt(role_name)

    # Session resume
    session_key = f"{graph.project_id}:{role_name}:{task.id}"
    session_id = ctx["session_ids"].get(session_key)

    # Phase 1: WORK
    work_turns = max(max_turns - _SUMMARY_PHASE_TURNS, max_turns // 2)
    work_timeout = max(task_timeout - 90, task_timeout // 2)

    t0 = time.monotonic()
    response = None
    try:
        response = await asyncio.wait_for(
            isolated_query(
                ctx["sdk"],
                prompt=prompt,
                system_prompt=system_prompt,
                cwd=ctx["project_dir"],
                session_id=session_id,
                max_turns=work_turns,
                max_budget_usd=get_agent_budget(role_name),
            ),
            timeout=work_timeout,
        )
    except TimeoutError:
        logger.warning("[DAG] Task %s: WORK phase timed out after %.0fs", task.id, time.monotonic() - t0)
    except asyncio.CancelledError:
        logger.warning("[DAG] Task %s: WORK phase cancelled", task.id)

    work_elapsed = time.monotonic() - t0

    # Process work phase response
    work_session_id: str | None = None
    work_cost = 0.0
    work_input_tokens = 0
    work_output_tokens = 0
    work_turns_used = 0
    work_text = ""
    work_had_error = False

    if response is not None:
        work_session_id = response.session_id or None
        work_cost = response.cost_usd
        work_input_tokens = response.input_tokens
        work_output_tokens = response.output_tokens
        work_text = response.text
        work_had_error = response.is_error
        work_turns_used = len(response.tool_uses) if response.tool_uses else (1 if len(response.text) > 500 else 0)

        if response.session_id:
            ctx["session_ids"][session_key] = response.session_id

        logger.info(
            "[DAG] Task %s: WORK done in %.1fs — error=%s cost=$%.4f len=%d",
            task.id, work_elapsed, response.is_error, work_cost, len(work_text),
        )

    # Try to parse JSON from work phase
    output: TaskOutput | None = None
    if work_text:
        tool_uses = response.tool_uses if response is not None else None
        output = extract_task_output(work_text, task.id, role_name, tool_uses=tool_uses)
        output.cost_usd = work_cost
        output.input_tokens = work_input_tokens
        output.output_tokens = work_output_tokens

    # Phase 2: SUMMARY (mandatory if work did meaningful turns)
    needs_summary = (
        work_session_id
        and (work_turns_used >= _MIN_WORK_TURNS_FOR_SUMMARY or work_had_error)
        and (output is None or not output.is_successful() or output.confidence <= 0.90)
    )

    if needs_summary:
        summary_output = await _run_summary_phase(
            task.id, work_session_id, ctx["sdk"], ctx["project_dir"],
            system_prompt, work_cost, work_input_tokens, work_output_tokens,
            task.id, role_name,
            tool_uses=response.tool_uses if response else None,
        )
        if summary_output is not None:
            if output is None or summary_output.confidence > output.confidence:
                output = summary_output

    if output is None:
        output = TaskOutput(
            task_id=task.id,
            status=TaskStatus.FAILED,
            summary=f"Agent produced no output (elapsed={work_elapsed:.0f}s, error={work_had_error})",
            issues=["No output from work phase and no session for summary phase"],
            cost_usd=work_cost,
            input_tokens=work_input_tokens,
            output_tokens=work_output_tokens,
            confidence=0.0,
        )

    # Classify unclassified failures
    if not output.is_successful() and not output.failure_category:
        output.failure_category = classify_failure(output)

    return output


async def _run_summary_phase(
    task_id: str,
    session_id: str,
    sdk: ClaudeSDKManager,
    cwd: str,
    system_prompt: str,
    work_cost: float,
    work_input_tokens: int,
    work_output_tokens: int,
    orig_task_id: str,
    role_name: str,
    tool_uses: list[str] | None = None,
) -> TaskOutput | None:
    """Phase 2: cheap follow-up to extract structured JSON output.

    Resumes the same session with tools disabled.
    Cost: ~$0.01-0.05 per call.
    """
    from minihive.isolated_query import isolated_query

    summary_prompt = (
        "Your work phase is complete. Now produce ONLY the required JSON output block.\n"
        "Do NOT do any more work. Do NOT use any tools.\n\n"
        "Reflect on everything you did and produce an accurate JSON summary:\n\n"
        "```json\n"
        "{\n"
        f'  "task_id": "{orig_task_id}",\n'
        '  "status": "completed",\n'
        '  "summary": "what you did in 2-3 sentences",\n'
        '  "artifacts": ["list/of/files/created/or/modified.py"],\n'
        '  "issues": [],\n'
        '  "blockers": [],\n'
        '  "followups": ["any remaining work"],\n'
        '  "confidence": 0.95\n'
        "}\n"
        "```\n\n"
        "IMPORTANT: Output ONLY the JSON block above. No explanations, no tools."
    )

    try:
        response = await asyncio.wait_for(
            isolated_query(
                sdk,
                prompt=summary_prompt,
                system_prompt=system_prompt,
                cwd=cwd,
                session_id=session_id,
                max_turns=_SUMMARY_PHASE_TURNS,
                max_budget_usd=5.0,
            ),
            timeout=180,
        )
    except (TimeoutError, asyncio.CancelledError):
        logger.warning("[DAG] Task %s: SUMMARY phase timed out", task_id)
        return None

    if response.is_error:
        logger.warning("[DAG] Task %s: SUMMARY error: %s", task_id, response.error_message[:100])
        return None

    output = extract_task_output(response.text, orig_task_id, role_name, tool_uses=tool_uses)
    output.cost_usd = work_cost + response.cost_usd
    output.input_tokens = work_input_tokens + response.input_tokens
    output.output_tokens = work_output_tokens + response.output_tokens

    if output.is_successful() and output.confidence > 0.0:
        return output

    return None


# ---------------------------------------------------------------------------
# Batch planning
# ---------------------------------------------------------------------------


def _plan_batches(tasks: list[TaskInput]) -> list[list[TaskInput]]:
    """Split ready tasks into sequential batches.

    Writers with overlapping file scopes go in separate batches.
    Readers can go in any batch (parallel-safe).
    Writers run first -- they produce code that readers verify.
    """
    if not tasks:
        return []

    readers = [t for t in tasks if t.role in READER_ROLES]
    writers = [t for t in tasks if t.role in WRITER_ROLES]
    others = [t for t in tasks if t.role not in READER_ROLES and t.role not in WRITER_ROLES]

    batches: list[list[TaskInput]] = []

    if writers:
        batches.extend(_split_writers_by_conflicts(writers))

    parallel_batch = readers + others
    if parallel_batch:
        batches.append(parallel_batch)

    return batches


def _split_writers_by_conflicts(writers: list[TaskInput]) -> list[list[TaskInput]]:
    """Group writers into non-conflicting batches based on files_scope overlap."""
    batches: list[list[TaskInput]] = []
    claimed_files: set[str] = set()
    current_batch: list[TaskInput] = []

    for task in writers:
        if not task.files_scope:
            # No scope declared -- isolate to its own batch
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                claimed_files = set()
            batches.append([task])
            continue

        scope = set(task.files_scope)
        if scope & claimed_files:
            # Conflict -- start a new batch
            if current_batch:
                batches.append(current_batch)
            current_batch = [task]
            claimed_files = scope
        else:
            current_batch.append(task)
            claimed_files |= scope

    if current_batch:
        batches.append(current_batch)

    return batches


# ---------------------------------------------------------------------------
# Self-healing: retry + remediation
# ---------------------------------------------------------------------------


async def _handle_failure(
    task: TaskInput,
    output: TaskOutput,
    ctx: dict[str, Any],
) -> None:
    """Decide between retry, remediation, or give up."""
    category = output.failure_category or classify_failure(output)

    if category in _NO_RETRY_CATEGORIES:
        logger.warning("[DAG] Task %s failed with %s -- not retryable", task.id, category.value)
        return

    retry_count = ctx["retries"].get(task.id, 0)

    # Retry if budget allows
    if retry_count < MAX_TASK_RETRIES and not output.is_terminal():
        ctx["retries"][task.id] = retry_count + 1
        print(f"  [{task.id}] Retrying ({ctx['retries'][task.id]}/{MAX_TASK_RETRIES})")
        ctx["total_cost"] -= output.cost_usd
        del ctx["completed"][task.id]
        return

    # Retries exhausted -- try remediation
    if ctx["remediation_count"] < MAX_TOTAL_REMEDIATIONS:
        depth = _remediation_depth(task, ctx["graph"].tasks)
        if depth < MAX_REMEDIATION_DEPTH:
            await _create_remediation(task, output, ctx)


def _remediation_depth(task: TaskInput, graph_tasks: list[TaskInput]) -> int:
    """Count how deep in the remediation chain this task is."""
    if not task.is_remediation:
        return 0
    depth = 1
    task_map = {t.id: t for t in graph_tasks}
    current_id = task.original_task_id
    seen: set[str] = {task.id}
    while current_id in task_map and current_id not in seen:
        parent = task_map[current_id]
        seen.add(current_id)
        if parent.is_remediation:
            depth += 1
            current_id = parent.original_task_id
        else:
            break
    return depth


async def _create_remediation(
    failed_task: TaskInput,
    failed_output: TaskOutput,
    ctx: dict[str, Any],
) -> bool:
    """Create and inject a remediation task into the graph.

    Uses graph_lock for atomic cap enforcement.
    """
    async with ctx["graph_lock"]:
        if ctx["remediation_count"] >= MAX_TOTAL_REMEDIATIONS:
            logger.warning(
                "[DAG] Remediation cap (%d) reached for %s",
                MAX_TOTAL_REMEDIATIONS, failed_task.id,
            )
            return False

        ctx["task_counter"] += 1
        remediation = create_remediation_task(
            failed_task=failed_task,
            failed_output=failed_output,
            task_counter=ctx["task_counter"],
        )

        if remediation is None:
            return False

        ctx["graph"].add_task(remediation)
        ctx["remediation_count"] += 1

    ctx["healing_history"].append({
        "action": "remediation_created",
        "failed_task": failed_task.id,
        "failure_category": (failed_output.failure_category or FailureCategory.UNKNOWN).value,
        "remediation_task": remediation.id,
        "detail": (
            f"Auto-created {remediation.id} ({remediation.role.value}) to fix "
            f"{failed_task.id}: {(failed_output.failure_details or '')[:100]}"
        ),
    })

    print(f"  \u21bb Self-healing: {remediation.id} to fix {failed_task.id}")
    return True


async def _try_self_heal(ctx: dict[str, Any]) -> bool:
    """Last-resort: scan all failed tasks for possible remediation.

    Returns True if at least one remediation task was created.
    """
    graph: TaskGraph = ctx["graph"]
    completed: dict[str, TaskOutput] = ctx["completed"]
    healed = False

    for task in graph.tasks:
        if task.id not in completed:
            continue
        output = completed[task.id]
        if output.is_successful() or output.is_terminal():
            continue
        if task.is_remediation:
            continue

        # Skip if we already created a remediation for this task
        already_has_fix = any(
            t.is_remediation and t.original_task_id == task.id
            for t in graph.tasks
        )
        if already_has_fix:
            continue

        created = await _create_remediation(task, output, ctx)
        if created:
            healed = True

    return healed
