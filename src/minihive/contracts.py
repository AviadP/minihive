"""Typed contracts for the multi-agent system.

Every agent receives a TaskInput and must return a TaskOutput.
No free text, no regex parsing — pure structured contracts.

Extracted from Hivemind's contracts.py — stripped of subcategories,
checkpoints, memory snapshots, and blackboard dependencies.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_FOLLOWUP = "needs_followup"
    REMEDIATION = "remediation"


class AgentRole(str, Enum):
    PM = "pm"
    FRONTEND_DEVELOPER = "frontend_developer"
    BACKEND_DEVELOPER = "backend_developer"
    DATABASE_EXPERT = "database_expert"
    DEVOPS = "devops"
    SECURITY_AUDITOR = "security_auditor"
    TEST_ENGINEER = "test_engineer"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"


class ArtifactType(str, Enum):
    API_CONTRACT = "api_contract"
    SCHEMA = "schema"
    COMPONENT_MAP = "component_map"
    TEST_REPORT = "test_report"
    SECURITY_REPORT = "security_report"
    REVIEW_REPORT = "review_report"
    ARCHITECTURE = "architecture"
    RESEARCH = "research"
    DEPLOYMENT = "deployment"
    FILE_MANIFEST = "file_manifest"
    CUSTOM = "custom"


class FailureCategory(str, Enum):
    DEPENDENCY_MISSING = "dependency_missing"
    API_MISMATCH = "api_mismatch"
    TEST_FAILURE = "test_failure"
    BUILD_ERROR = "build_error"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    UNCLEAR_GOAL = "unclear_goal"
    MISSING_CONTEXT = "missing_context"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Role classification (replaces complexity.py + blackboard dependency)
# ---------------------------------------------------------------------------

WRITER_ROLES: frozenset[AgentRole] = frozenset({
    AgentRole.FRONTEND_DEVELOPER,
    AgentRole.BACKEND_DEVELOPER,
    AgentRole.DATABASE_EXPERT,
    AgentRole.DEVOPS,
})

READER_ROLES: frozenset[AgentRole] = frozenset({
    AgentRole.RESEARCHER,
    AgentRole.REVIEWER,
    AgentRole.SECURITY_AUDITOR,
    AgentRole.TEST_ENGINEER,
})

WRITER_ROLE_NAMES: frozenset[str] = frozenset(r.value for r in WRITER_ROLES)


# ---------------------------------------------------------------------------
# Artifact — structured knowledge transfer between agents
# ---------------------------------------------------------------------------


class Artifact(BaseModel):
    """A structured piece of knowledge produced by an agent."""

    type: ArtifactType
    title: str = Field(..., description="Human-readable title")
    file_path: str = Field(default="", description="Path relative to project root")
    data: dict[str, Any] = Field(default_factory=dict)
    summary: str = Field(default="")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        if len(v.strip()) < 1:
            raise ValueError("Artifact title must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------


class TaskInput(BaseModel):
    """What an agent receives — the contract going IN."""

    id: str = Field(..., description="Unique task ID, e.g. 'task_001'")
    role: AgentRole
    goal: str = Field(..., description="Clear, measurable objective")
    constraints: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    context_from: list[str] = Field(default_factory=list)
    files_scope: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    required_artifacts: list[ArtifactType] = Field(default_factory=list)
    input_artifacts: list[str] = Field(default_factory=list)
    is_remediation: bool = Field(default=False)
    original_task_id: str = Field(default="")
    failure_context: str = Field(default="")
    expected_input_artifact_types: list[ArtifactType] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", v):
            raise ValueError(f"Invalid task id '{v}': use letters, digits, _ or - only")
        return v

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Task goal must be at least 10 characters")
        return v.strip()


class TaskOutput(BaseModel):
    """What an agent returns — the contract coming OUT."""

    model_config = {"extra": "allow"}

    task_id: str
    status: TaskStatus
    summary: str = Field(..., description="2-3 sentences describing what was done")
    artifacts: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list, max_length=50)
    blockers: list[str] = Field(default_factory=list, max_length=50)
    followups: list[str] = Field(default_factory=list, max_length=50)
    cost_usd: float = Field(default=0.0, ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    turns_used: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    structured_artifacts: list[Artifact] = Field(default_factory=list, max_length=20)
    failure_category: FailureCategory | None = Field(default=None)
    failure_details: str = Field(default="")

    def is_successful(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.BLOCKED)


# ---------------------------------------------------------------------------
# TaskGraph — the full execution plan
# ---------------------------------------------------------------------------


class TaskGraph(BaseModel):
    """The full execution plan produced by the PM Agent."""

    project_id: str
    user_message: str
    vision: str = Field(..., description="One-sentence mission statement")
    epic_breakdown: list[str] = Field(default_factory=list)
    tasks: list[TaskInput] = Field(..., description="All tasks with dependency wiring")

    def get_task(self, task_id: str) -> TaskInput | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def ready_tasks(self, completed: dict[str, TaskOutput] | set[str]) -> list[TaskInput]:
        """Return tasks whose dependencies are all successfully completed."""
        is_dict = isinstance(completed, dict)
        result = []
        for task in self.tasks:
            if task.id in completed:
                continue
            deps_ok = True
            for dep in task.depends_on:
                if dep not in completed:
                    deps_ok = False
                    break
                if is_dict and not completed[dep].is_successful():
                    deps_ok = False
                    break
            if deps_ok:
                result.append(task)
        return result

    def is_complete(self, completed: dict[str, TaskOutput]) -> bool:
        return all(t.id in completed for t in self.tasks)

    def has_failed(self, completed: dict[str, TaskOutput]) -> bool:
        """True if a blocked/failed task has no downstream path to completion."""
        blocked = {
            t.id
            for t in self.tasks
            if t.id in completed
            and completed[t.id].status in (TaskStatus.FAILED, TaskStatus.BLOCKED)
        }
        if not blocked:
            return False
        pending_ids = {t.id for t in self.tasks if t.id not in completed}
        for tid in pending_ids:
            task = self.get_task(tid)
            if task and any(dep in blocked for dep in task.depends_on):
                return True
        return False

    def add_task(self, task: TaskInput) -> None:
        self.tasks.append(task)

    def validate_dag(self) -> list[str]:
        """Check for cycles, self-deps, duplicate IDs, and missing deps."""
        errors: list[str] = []
        task_ids = {t.id for t in self.tasks}

        seen_ids: set[str] = set()
        for task in self.tasks:
            if task.id in seen_ids:
                errors.append(f"Duplicate task ID: '{task.id}'")
            seen_ids.add(task.id)

        for task in self.tasks:
            if task.id in task.depends_on:
                errors.append(f"Task '{task.id}' depends on itself")
            for dep in task.depends_on:
                if dep not in task_ids:
                    errors.append(f"Task '{task.id}' depends on unknown task '{dep}'")

        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            task = self.get_task(node)
            if task:
                for dep in task.depends_on:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            rec_stack.discard(node)
            return False

        for task in self.tasks:
            if task.id not in visited:
                if has_cycle(task.id):
                    errors.append(f"Cycle detected involving task '{task.id}'")
                    break

        return errors


# ---------------------------------------------------------------------------
# Failure Classification
# ---------------------------------------------------------------------------

_FAILURE_PATTERNS: list[tuple[FailureCategory, list[str]]] = [
    (
        FailureCategory.DEPENDENCY_MISSING,
        [
            "import error", "importerror", "module not found", "modulenotfounderror",
            "no such file", "dependency", "not installed", "missing module",
            "cannot find module", "no module named", "package not found",
            "could not resolve", "unresolved import",
        ],
    ),
    (
        FailureCategory.API_MISMATCH,
        [
            "404", "endpoint not found", "api mismatch", "contract",
            "expected response", "schema mismatch", "property does not exist",
            "undefined is not", "missing field", "wrong status code",
        ],
    ),
    (
        FailureCategory.TEST_FAILURE,
        [
            "test failed", "assertion error", "assertionerror", "expected",
            "assert", "pytest", "test_", "FAILED", "failures=",
        ],
    ),
    (
        FailureCategory.BUILD_ERROR,
        [
            "syntax error", "syntaxerror", "compilation", "build failed", "tsc",
            "cannot compile", "parse error", "unexpected token", "indentation",
            "unterminated", "invalid syntax", "typeerror", "nameerror",
            "referenceerror", "type mismatch", "incompatible type",
        ],
    ),
    (
        FailureCategory.TIMEOUT,
        [
            "timeout", "timed out", "max turns", "budget exceeded",
            "too many iterations", "deadline",
        ],
    ),
    (
        FailureCategory.PERMISSION,
        [
            "permission denied", "permissionerror", "access denied",
            "forbidden", "eacces", "read-only", "not writable",
        ],
    ),
    (
        FailureCategory.MISSING_CONTEXT,
        [
            "filenotfounderror", "file not found", "no such file or directory",
            "missing context", "dependency not completed", "upstream task",
            "context_from", "required artifact missing",
        ],
    ),
    (
        FailureCategory.UNCLEAR_GOAL,
        [
            "unclear", "ambiguous", "not sure what", "need clarification",
            "insufficient context", "cannot determine",
        ],
    ),
    (
        FailureCategory.EXTERNAL,
        [
            "connection refused", "network error", "dns", "502", "503",
            "service unavailable", "rate limit", "api key", "429",
            "too many requests", "throttled", "quota exceeded",
        ],
    ),
]


def classify_failure(output: TaskOutput) -> FailureCategory:
    """Auto-classify a failed task's failure category from its output text."""
    if output.failure_category and output.failure_category != FailureCategory.UNKNOWN:
        return output.failure_category

    search_text = " ".join([
        output.summary, output.failure_details,
        " ".join(output.issues), " ".join(output.blockers),
    ]).lower()

    if not search_text.strip():
        return FailureCategory.UNKNOWN

    scores: dict[FailureCategory, int] = {}
    for category, patterns in _FAILURE_PATTERNS:
        score = sum(1 for p in patterns if p in search_text)
        if score > 0:
            scores[category] = score

    if not scores:
        return FailureCategory.UNKNOWN

    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Remediation — auto-generate fix tasks
# ---------------------------------------------------------------------------

_REMEDIATION_STRATEGIES: dict[FailureCategory, dict[str, Any]] = {
    FailureCategory.DEPENDENCY_MISSING: {
        "goal_template": (
            "Fix dependency issue from task {task_id}: {failure_details}. "
            "Install missing packages, fix import paths, or create missing files."
        ),
        "constraints": ["Only fix the dependency issue — do not refactor unrelated code"],
    },
    FailureCategory.API_MISMATCH: {
        "goal_template": (
            "Fix API contract mismatch from task {task_id}: {failure_details}. "
            "Read the API contract artifact from upstream tasks, then align the "
            "implementation to match the contract exactly."
        ),
        "constraints": [
            "Read the api_contract artifact before making changes",
            "Do not change the contract — change the implementation",
        ],
    },
    FailureCategory.TEST_FAILURE: {
        "goal_template": (
            "Fix failing tests from task {task_id}: {failure_details}. "
            "Run the tests first to reproduce, then fix the code (not the tests) "
            "to make them pass. Run tests again to verify."
        ),
        "constraints": [
            "Fix the implementation, not the test assertions",
            "Run pytest -x --tb=short before and after changes",
        ],
    },
    FailureCategory.BUILD_ERROR: {
        "goal_template": (
            "Fix build/compilation error from task {task_id}: {failure_details}. "
            "Read the error output carefully, fix the syntax or type errors, "
            "and verify the build passes cleanly."
        ),
        "constraints": ["Run the build command after fixing to verify"],
    },
    FailureCategory.TIMEOUT: {
        "goal_template": (
            "Complete the work that timed out in task {task_id}: {failure_details}. "
            "The previous agent ran out of turns. Pick up where it left off — "
            "check git diff to see what was already done, then complete the remaining work."
        ),
        "constraints": ["Check git status first to understand what was already done"],
    },
    FailureCategory.MISSING_CONTEXT: {
        "goal_template": (
            "Fix missing file/context issue from task {task_id}: {failure_details}. "
            "A required file or upstream dependency was not found. Check if the file "
            "needs to be created, or if an upstream task failed to produce it."
        ),
        "constraints": [
            "Check if the missing file should exist from an upstream task",
            "Create the file if it's a new requirement, or fix the import path",
        ],
    },
}


def create_remediation_task(
    failed_task: TaskInput,
    failed_output: TaskOutput,
    task_counter: int,
) -> TaskInput | None:
    """Create a remediation task to fix a failure, or None if not remediable."""
    category = classify_failure(failed_output)

    strategy = _REMEDIATION_STRATEGIES.get(category)
    if strategy is None:
        return None

    role = failed_task.role

    failure_details = failed_output.failure_details or failed_output.summary
    goal = strategy["goal_template"].format(
        task_id=failed_task.id,
        failure_details=failure_details[:300],
    )

    prefix = f"fix_{task_counter:03d}_"
    max_suffix_len = 64 - len(prefix)
    suffix = failed_task.id[:max_suffix_len]
    remediation_id = prefix + suffix

    return TaskInput(
        id=remediation_id,
        role=role,
        goal=goal,
        constraints=strategy.get("constraints", []) + failed_task.constraints,
        depends_on=failed_task.depends_on,
        context_from=list(dict.fromkeys([*failed_task.context_from, failed_task.id])),
        files_scope=failed_task.files_scope,
        acceptance_criteria=[
            *failed_task.acceptance_criteria,
            f"The issue from {failed_task.id} is resolved",
            "All related tests pass (if applicable)",
        ],
        input_artifacts=[a.file_path for a in failed_output.structured_artifacts if a.file_path],
        is_remediation=True,
        original_task_id=failed_task.id,
        failure_context=f"[{category.value}] {failure_details[:500]}",
    )


# ---------------------------------------------------------------------------
# JSON Output Extraction
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_FILE_PATH_RE = re.compile(
    r"[\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml|css|html|sql|sh|env|toml|cfg)",
)
_CODE_BLOCK_RE = re.compile(r"```(?:python|typescript|javascript|bash|sql|\w+)?\n")


def extract_task_output(
    raw_text: str, task_id: str, task_role: str = "", tool_uses: list[str] | None = None
) -> TaskOutput:
    """Parse a TaskOutput from an agent's raw text response.

    Tries: 1) fenced JSON block, 2) last JSON object, 3) multi-signal work detection.
    """
    # Step 1: Try fenced JSON block
    for match in _JSON_BLOCK_RE.finditer(raw_text):
        try:
            data = json.loads(match.group(1).strip())
            data.setdefault("task_id", task_id)
            return TaskOutput(**data)
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            continue

    # Step 2: Try last JSON object in text
    start = raw_text.rfind("{")
    if start != -1:
        depth = 0
        for i in range(start, len(raw_text)):
            if raw_text[i] == "{":
                depth += 1
            elif raw_text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(raw_text[start : i + 1])
                        data.setdefault("task_id", task_id)
                        return TaskOutput(**data)
                    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                        break

    # Step 3: Multi-signal work detection
    logger.warning(
        f"[extract] No JSON found for {task_id}. len={len(raw_text)}. Using work detection."
    )

    score = 0.0
    signals: list[str] = []
    lower = raw_text.lower() if raw_text else ""

    # Signal 1: Tool use indicators
    _TOOL_PATTERNS = [
        (r"\$ .+", "shell"), (r"running:? `[^`]+`", "tool_exec"),
        (r"reading:? .+\.\w+", "reads"), (r"writing:? .+\.\w+", "writes"),
        (r"editing:? .+\.\w+", "edits"),
    ]
    tool_hits = 0
    for pattern, label in _TOOL_PATTERNS:
        matches = re.findall(pattern, lower)
        if matches:
            tool_hits += len(matches)
            signals.append(f"{label}({len(matches)})")
    if tool_hits >= 3:
        score += 0.4
    elif tool_hits >= 1:
        score += 0.2

    # Signal 2: File paths mentioned
    file_paths = _FILE_PATH_RE.findall(raw_text)
    unique_files = list(dict.fromkeys(file_paths))[:30]
    if len(unique_files) >= 5:
        score += 0.3
        signals.append(f"files({len(unique_files)})")
    elif len(unique_files) >= 2:
        score += 0.15
        signals.append(f"files({len(unique_files)})")

    # Signal 3: Action verbs
    _ACTION_VERBS = [
        "created ", "modified ", "updated ", "wrote ", "implemented", "fixed ",
        "added ", "refactored", "installed ", "configured", "tested ", "verified",
        "i've ", "successfully",
    ]
    verb_hits = sum(1 for v in _ACTION_VERBS if v in lower)
    if verb_hits >= 4:
        score += 0.3
        signals.append(f"verbs({verb_hits})")
    elif verb_hits >= 2:
        score += 0.15
        signals.append(f"verbs({verb_hits})")

    # Signal 4: Report sections
    _REPORT_MARKERS = ["## summary", "## files changed", "## status", "# summary"]
    report_hits = sum(1 for m in _REPORT_MARKERS if m in lower)
    if report_hits >= 2:
        score += 0.3
    elif report_hits >= 1:
        score += 0.1

    # Signal 5: Git activity
    if "git commit" in lower or "git add" in lower:
        score += 0.3

    # Signal 6: Text volume
    if len(raw_text) >= 2000:
        score += 0.15
    elif len(raw_text) >= 500:
        score += 0.05

    # Signal 7: Code blocks
    code_blocks = _CODE_BLOCK_RE.findall(raw_text)
    if len(code_blocks) >= 2:
        score += 0.2
    elif len(code_blocks) >= 1:
        score += 0.1

    # Signal 8: Write operations
    _WRITE_TOOLS = {"Write", "write_file", "create_file", "Edit", "edit_file"}
    _EXEC_TOOLS = {"Bash", "execute_bash", "bash"}
    write_hits = 0
    if tool_uses:
        for t in tool_uses:
            if t in _WRITE_TOOLS:
                write_hits += 1
            elif t in _EXEC_TOOLS:
                write_hits += 0.5
    else:
        for pattern in [r"writing:? .+\.\w+", r"editing:? .+\.\w+", r"created? .+\.\w+"]:
            write_hits += len(re.findall(pattern, lower))
    if write_hits >= 1:
        score += 0.3

    # Execution agents without writes → cap score
    if task_role in WRITER_ROLE_NAMES and write_hits == 0:
        score = min(score, 0.35)

    logger.info(f"[extract] {task_id}: score={score:.2f} signals={signals}")

    # Extract summary
    inferred_summary = ""
    for marker in ["## SUMMARY", "## Summary", "# Summary"]:
        idx = raw_text.find(marker)
        if idx != -1:
            after = raw_text[idx + len(marker):].strip()
            end = after.find("\n\n")
            inferred_summary = after[:end].strip() if end != -1 else after[:300].strip()
            break
    if not inferred_summary and raw_text:
        inferred_summary = raw_text[-300:].strip()

    WORK_THRESHOLD = 0.4

    if score >= WORK_THRESHOLD:
        confidence = min(0.5 + score * 0.3, 0.85)
        return TaskOutput(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            summary=f"Agent completed work (inferred, score={score:.2f}). {inferred_summary[:200]}",
            artifacts=unique_files,
            issues=["Output inferred via multi-signal detection (no JSON from agent)"],
            confidence=confidence,
        )

    fallback = TaskOutput(
        task_id=task_id,
        status=TaskStatus.FAILED,
        summary=(
            f"Agent output could not be parsed and work score too low "
            f"({score:.2f} < {WORK_THRESHOLD}). Last output: {inferred_summary[:200]}"
        ),
        issues=[f"No JSON output and low work score ({score:.2f}). Signals: {signals}"],
        failure_details=raw_text[-500:] if raw_text else "",
        confidence=0.0,
    )
    fallback.failure_category = classify_failure(fallback)
    return fallback


# ---------------------------------------------------------------------------
# Prompt Serialisation — Artifact-aware context passing
# ---------------------------------------------------------------------------


def _truncate_json_safely(data_str: str, max_len: int) -> str:
    if len(data_str) <= max_len:
        return data_str
    truncated = data_str[:max_len]
    for i in range(len(truncated) - 1, 0, -1):
        if truncated[i] in (",", "{", "[", "\n"):
            truncated = truncated[: i + 1]
            break
    return truncated + "\n    ... (truncated — read the file for full data)"


def task_input_to_prompt(
    task: TaskInput,
    context_outputs: dict[str, TaskOutput],
    graph_vision: str = "",
    graph_epics: list[str] | None = None,
) -> str:
    """Serialise a TaskInput into a structured XML prompt for the agent."""
    parts: list[str] = []

    if graph_vision or graph_epics:
        parts.append("<mission>")
        if graph_vision:
            parts.append(f"  <vision>{graph_vision}</vision>")
        if graph_epics:
            parts.append("  <epics>")
            for i, epic in enumerate(graph_epics, 1):
                parts.append(f"    <epic id='{i}'>{epic}</epic>")
            parts.append("  </epics>")
        parts.append("</mission>\n")

    parts.append("<task_assignment>")
    parts.append(f"  <task_id>{task.id}</task_id>")
    parts.append(f"  <role>{task.role.value}</role>")
    parts.append(f"  <goal>{task.goal}</goal>")

    if task.is_remediation:
        parts.append(f"  <remediation original_task='{task.original_task_id}'>")
        parts.append(f"    {task.failure_context}")
        parts.append("  </remediation>")

    if task.acceptance_criteria:
        parts.append("  <acceptance_criteria>")
        for c in task.acceptance_criteria:
            parts.append(f"    <criterion>{c}</criterion>")
        parts.append("  </acceptance_criteria>")

    if task.constraints:
        parts.append("  <constraints>")
        for c in task.constraints:
            parts.append(f"    <constraint>{c}</constraint>")
        parts.append("  </constraints>")

    if task.files_scope:
        parts.append(f"  <files_scope>{', '.join(task.files_scope)}</files_scope>")

    if task.required_artifacts:
        parts.append("  <required_artifacts>")
        for art_type in task.required_artifacts:
            parts.append(f"    <artifact_type>{art_type.value}</artifact_type>")
        parts.append("  </required_artifacts>")

    if task.input_artifacts:
        parts.append("  <input_artifacts>")
        for path in task.input_artifacts:
            parts.append(f"    <file>cat {path}</file>")
        parts.append("  </input_artifacts>")

    parts.append("</task_assignment>\n")

    if context_outputs:
        parts.append("<upstream_context>")
        for tid, output in context_outputs.items():
            parts.append(f"  <task_result id='{tid}' status='{output.status.value}'>")
            parts.append(f"    <summary>{output.summary}</summary>")
            if output.artifacts:
                parts.append(
                    f"    <files_changed>{', '.join(output.artifacts[:15])}</files_changed>"
                )
            if output.issues:
                parts.append("    <issues>")
                for issue in output.issues[:5]:
                    parts.append(f"      <issue>{issue}</issue>")
                parts.append("    </issues>")

            if output.structured_artifacts:
                parts.append("    <artifacts>")
                for art in output.structured_artifacts:
                    parts.append(f"      <artifact type='{art.type.value}'>")
                    parts.append(f"        <title>{art.title}</title>")
                    if art.file_path:
                        parts.append(f"        <file_path>{art.file_path}</file_path>")
                    if art.summary:
                        parts.append(f"        <summary>{art.summary}</summary>")
                    if art.data:
                        data_str = json.dumps(art.data, indent=2)
                        data_str = _truncate_json_safely(data_str, 1200)
                        parts.append(f"        <data>\n{data_str}\n        </data>")
                    parts.append("      </artifact>")
                parts.append("    </artifacts>")

            parts.append("  </task_result>")
        parts.append("</upstream_context>\n")

    parts.append(
        "<thinking_protocol>\n"
        "Before starting your work, think step-by-step inside <thinking> tags:\n"
        "1. What exactly is being asked? What does 'done' look like?\n"
        "2. What files/systems are involved? What do I need to read first?\n"
        "3. What are the constraints I must respect?\n"
        "4. What is my plan of action? (ordered steps)\n"
        "5. What could go wrong? How will I verify success?\n\n"
        "Only AFTER completing your <thinking> block, begin the actual work.\n"
        "</thinking_protocol>\n"
    )

    parts.append(
        "---\n"
        "After completing your work, briefly list what files you created or modified.\n"
        f"Include your task_id: {task.id}\n"
    )
    return "\n".join(parts)


def task_graph_schema() -> dict[str, Any]:
    """JSON schema for the PM agent's TaskGraph output."""
    return {
        "type": "object",
        "required": ["project_id", "user_message", "vision", "tasks"],
        "properties": {
            "project_id": {"type": "string"},
            "user_message": {"type": "string"},
            "vision": {"type": "string", "description": "One-sentence mission"},
            "epic_breakdown": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-7 high-level epics",
            },
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "role", "goal"],
                    "properties": {
                        "id": {"type": "string"},
                        "role": {"type": "string", "enum": [r.value for r in AgentRole]},
                        "goal": {"type": "string"},
                        "constraints": {"type": "array", "items": {"type": "string"}},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "context_from": {"type": "array", "items": {"type": "string"}},
                        "files_scope": {"type": "array", "items": {"type": "string"}},
                        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                        "required_artifacts": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [a.value for a in ArtifactType],
                            },
                        },
                        "input_artifacts": {"type": "array", "items": {"type": "string"}},
                        "expected_input_artifact_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [a.value for a in ArtifactType],
                            },
                        },
                    },
                },
            },
        },
    }
