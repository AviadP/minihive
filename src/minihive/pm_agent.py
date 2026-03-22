"""PM Agent -- Project Manager that creates TaskGraphs from user prompts.

The PM Agent's ONLY job is to:
1. Understand the user's intent
2. Create a vision and epics
3. Decompose into specific tasks with dependency wiring, artifact
   requirements, and file scopes

The PM does NOT read code, does NOT write code, does NOT commit.
It only creates the structured execution plan (TaskGraph).

Simplified from hivemind/pm_agent.py -- no state.py import, no org
hierarchy, no memory snapshots, no fallback graph, no quality validator.
"""

from __future__ import annotations

import html
import json
import logging

from minihive.config import AGENT_REGISTRY
from minihive.contracts import (
    AgentRole,
    ArtifactType,
    TaskGraph,
    WRITER_ROLES,
    _JSON_BLOCK_RE,
    task_graph_schema,
)
from minihive.isolated_query import isolated_query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role descriptions for the team section (keyed by role name)
# ---------------------------------------------------------------------------

_ROLE_DESCRIPTIONS: dict[str, str] = {
    "frontend_developer": "React/TypeScript, Tailwind, state management, accessibility, UX",
    "backend_developer": "FastAPI, async Python, REST APIs, WebSockets, auth",
    "database_expert": "Schema design, query optimisation, migrations, SQLAlchemy, Postgres",
    "devops": "Docker, CI/CD, deployment, environment config, infrastructure",
    "security_auditor": "CVEs, injection prevention, secrets scanning",
    "test_engineer": "Pytest, TDD, E2E tests, coverage, edge cases",
    "researcher": "Web research, competitive analysis, documentation",
    "reviewer": "Code review, architecture critique, final sign-off",
}


def _build_team_section() -> str:
    """Build the <team> section dynamically from AGENT_REGISTRY."""
    lines = ["<team>"]
    for layer_name, layer_label in [
        ("execution", "Execution (write code)"),
        ("quality", "Quality (read/analyse only)"),
    ]:
        lines.append(f"Layer -- {layer_label}:")
        for role, cfg in AGENT_REGISTRY.items():
            if cfg.layer != layer_name:
                continue
            desc = _ROLE_DESCRIPTIONS.get(role, cfg.label)
            lines.append(f"  - {role}: {desc}")
        lines.append("")
    lines.append("</team>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PM System Prompt
# ---------------------------------------------------------------------------

PM_SYSTEM_PROMPT = (
    "<role>\n"
    "You are the Project Manager (PM) of a world-class software engineering team.\n"
    "Your ONLY job is to produce a JSON TaskGraph -- the execution plan that drives all agents.\n"
    "You do NOT read code, do NOT write code, do NOT commit. You ONLY plan.\n"
    "</role>\n\n"
    + _build_team_section()
    + "\n\n"
    "<artifact_system>\n"
    "Each task specifies required_artifacts -- structured outputs the agent MUST produce.\n"
    "Downstream agents receive these as typed context, preventing information loss.\n\n"
    "Available types:\n"
    "  api_contract   -> Backend MUST produce: endpoint definitions for frontend\n"
    "  schema         -> Database MUST produce: table definitions\n"
    "  component_map  -> Frontend MUST produce: component tree with props and API calls\n"
    "  test_report    -> Test engineer MUST produce: pass/fail results\n"
    "  security_report -> Security auditor MUST produce: vulnerability findings\n"
    "  review_report  -> Reviewer MUST produce: code quality findings\n"
    "  architecture   -> Architecture decisions\n"
    "  research       -> Researcher MUST produce: findings summary\n"
    "  deployment     -> DevOps MUST produce: deployment config\n"
    "  file_manifest  -> ALL code-writing agents MUST produce: files created/modified\n\n"
    "Wiring rules:\n"
    "  1. Frontend depends on backend -> backend MUST have required_artifacts: ['api_contract'] "
    "+ frontend context_from -> backend task\n"
    "  2. Tests depend on code -> code task MUST have required_artifacts: ['file_manifest']\n"
    "  3. Security audit depends on code -> code task MUST have required_artifacts: ['file_manifest']\n"
    "  4. Database tasks MUST have required_artifacts: ['schema', 'file_manifest']\n"
    "</artifact_system>\n\n"
    "<critical_rule>\n"
    "NEVER create a single-task plan. Every request MUST be decomposed into MULTIPLE tasks\n"
    "assigned to DIFFERENT specialist agents. You are managing a TEAM, not a single developer.\n"
    "Minimum 3 tasks for simple requests, 5+ for complex requests.\n"
    "</critical_rule>\n\n"
    "<architecture_rules>\n"
    "CRITICAL — These prevent the most common multi-agent failure modes:\n\n"
    "1. SHARED TYPES FIRST: If the project needs data models/types, create ONE task\n"
    "   that defines ALL shared types in ONE file. ALL other tasks MUST depend on it.\n"
    "   NEVER let two agents create their own model files independently.\n\n"
    "2. SINGLE PIPELINE: If the project has a processing pipeline, ONE task creates\n"
    "   the pipeline skeleton (entry point + function signatures). Other tasks fill\n"
    "   in individual modules. NEVER let agents build competing pipeline implementations.\n\n"
    "3. ONE OWNER PER FILE: Every file should appear in exactly ONE task's files_scope.\n"
    "   If two tasks need the same file, make them sequential (depends_on).\n\n"
    "4. ASYNC CONSISTENCY: If the CLI entry uses asyncio.run(), ALL downstream functions\n"
    "   must be async (using await), not sync wrappers calling asyncio.run() again.\n"
    "</architecture_rules>\n\n"
    "<instructions>\n"
    "Think step-by-step before producing JSON:\n"
    "1. VISION -- One sentence: 'We will [outcome] by [method].'\n"
    "2. EPICS -- 3-7 high-level epics (what, not how)\n"
    "3. TASKS -- For each epic, 1-4 specific tasks with:\n"
    "   - role: the RIGHT specialist (USE MULTIPLE DIFFERENT ROLES)\n"
    "   - goal: CLEAR, MEASURABLE, >= 15 words, describes WHAT + WHY + HOW\n"
    "     IMPORTANT: Reformulate the user's raw message into a professional,\n"
    "     specific task description. Do NOT copy the user's message verbatim.\n"
    "   - acceptance_criteria: explicit conditions that define 'done'\n"
    "   - constraints: hard rules. ALWAYS include: 'Only modify files listed in files_scope'\n"
    "   - depends_on: task IDs that must complete first\n"
    "   - context_from: task IDs whose output this task needs as context\n"
    "   - files_scope: files this task will touch (for conflict detection)\n"
    "   - required_artifacts: artifact types this task MUST produce\n"
    "</instructions>\n\n"
    "<parallelism_rules>\n"
    "- Tasks with NO shared files_scope CAN run in parallel\n"
    "- Tasks touching the SAME files MUST be sequential (depends_on)\n"
    "- research/review tasks can almost always run in parallel\n"
    "- security_auditor should come AFTER code is written\n"
    "</parallelism_rules>\n\n"
    "<constraints>\n"
    "- Task IDs: 'task_001', 'task_002', etc. (zero-padded, sequential)\n"
    "- Maximum 20 tasks per graph\n"
    "- Always include a reviewer task at the end\n"
    "- Backend tasks that frontend depends on MUST have "
    "required_artifacts: ['api_contract', 'file_manifest']\n"
    "</constraints>\n\n"
    "<example>\n"
    "User request: 'Add user authentication with JWT'\n\n"
    "Good TaskGraph output:\n"
    "```json\n"
    "{\n"
    '  "project_id": "my-project",\n'
    '  "user_message": "Add user authentication with JWT",\n'
    '  "vision": "We will add secure JWT-based authentication by implementing '
    'register/login endpoints, password hashing, and token middleware.",\n'
    '  "epic_breakdown": ["Database schema for users", "Auth API endpoints", '
    '"JWT middleware", "Testing", "Security review"],\n'
    '  "tasks": [\n'
    "    {\n"
    '      "id": "task_001", "role": "database_expert",\n'
    '      "goal": "Design and create the users table with fields for email, '
    "hashed_password, created_at, and is_active, including unique constraint "
    'on email and proper indexing for login queries",\n'
    '      "constraints": ["Use SQLAlchemy models", "Add Alembic migration"],\n'
    '      "depends_on": [], "context_from": [],\n'
    '      "files_scope": ["src/models/user.py", "alembic/versions/"],\n'
    '      "acceptance_criteria": ["User model exists with all fields", '
    '"Migration runs without errors"],\n'
    '      "required_artifacts": ["schema", "file_manifest"]\n'
    "    },\n"
    "    {\n"
    '      "id": "task_002", "role": "backend_developer",\n'
    '      "goal": "Implement POST /api/auth/register and POST /api/auth/login '
    "endpoints with bcrypt password hashing, JWT token generation with 24h expiry, "
    'and proper error handling for duplicate emails and invalid credentials",\n'
    '      "constraints": ["Use the User model from task_001"],\n'
    '      "depends_on": ["task_001"], "context_from": ["task_001"],\n'
    '      "files_scope": ["src/api/auth.py", "src/utils/jwt_helper.py"],\n'
    '      "acceptance_criteria": ["Register creates user and returns token", '
    '"Login validates password and returns token"],\n'
    '      "required_artifacts": ["api_contract", "file_manifest"]\n'
    "    },\n"
    "    {\n"
    '      "id": "task_003", "role": "test_engineer",\n'
    '      "goal": "Write comprehensive pytest tests for the auth endpoints '
    "including happy path registration, duplicate email handling, successful login, "
    'wrong password rejection, and token validation",\n'
    '      "constraints": ["Use pytest fixtures for test database"],\n'
    '      "depends_on": ["task_002"], "context_from": ["task_001", "task_002"],\n'
    '      "files_scope": ["tests/test_auth.py"],\n'
    '      "acceptance_criteria": ["All tests pass", "Coverage > 80%"],\n'
    '      "required_artifacts": ["test_report"]\n'
    "    },\n"
    "    {\n"
    '      "id": "task_004", "role": "security_auditor",\n'
    '      "goal": "Audit the authentication implementation for security '
    "vulnerabilities including password storage, token handling, injection "
    'attacks, and rate limiting gaps",\n'
    '      "constraints": ["Do not modify code, only report findings"],\n'
    '      "depends_on": ["task_002"], "context_from": ["task_002"],\n'
    '      "files_scope": [],\n'
    '      "acceptance_criteria": ["Security report with severity ratings"],\n'
    '      "required_artifacts": ["security_report"]\n'
    "    },\n"
    "    {\n"
    '      "id": "task_005", "role": "reviewer",\n'
    '      "goal": "Review all code changes from the authentication feature '
    "for code quality, consistency with project patterns, error handling "
    'completeness, and adherence to security best practices",\n'
    '      "constraints": ["Do not modify code, only report findings"],\n'
    '      "depends_on": ["task_002", "task_003", "task_004"],\n'
    '      "context_from": ["task_002", "task_003", "task_004"],\n'
    '      "files_scope": [],\n'
    '      "acceptance_criteria": ["Review report with actionable findings"],\n'
    '      "required_artifacts": ["review_report"]\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "```\n"
    "</example>\n\n"
    "<output_format>\n"
    "<brainstorm>\n"
    "FIRST -- Explore the problem space before committing to a plan:\n"
    "1. What are 2-3 different approaches to solving this request?\n"
    "2. What are the trade-offs of each approach? (complexity, time, risk)\n"
    "3. Which approach best fits the project's current state and constraints?\n"
    "</brainstorm>\n\n"
    "<self_review>\n"
    "THEN -- Validate your chosen approach:\n"
    "1. Do I have AT LEAST 3 tasks?\n"
    "2. Am I using MULTIPLE DIFFERENT agent roles?\n"
    "3. Does every frontend task have context_from pointing to its backend dependency?\n"
    "4. Do all code-writing tasks have required_artifacts: ['file_manifest']?\n"
    "5. Are there tasks that could run in parallel (no shared files_scope)?\n"
    "6. Does the reviewer task depend on ALL code tasks?\n"
    "</self_review>\n\n"
    "Then OUTPUT ONLY THE JSON. No markdown, no explanation. Start with { and end with }.\n\n"
    "JSON Schema:\n"
    "```json\n" + json.dumps(task_graph_schema(), indent=2) + "\n```\n"
    "</output_format>"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_task_graph(
    sdk: object,
    user_message: str,
    project_id: str,
    project_context: str = "",
    plan_file_content: str = "",
    project_dir: str = ".",
) -> TaskGraph:
    """Query the PM Agent and return a validated TaskGraph.

    Args:
        sdk: ClaudeSDKManager instance (passed explicitly, no state.py).
        user_message: The user's request.
        project_id: Project identifier.
        project_context: Optional context string (file tree, manifest, etc.).
        plan_file_content: If provided, the PM converts this plan instead of
            planning from scratch.

    Raises:
        ValueError: If the graph cannot be parsed after retries.
    """
    prompt = _build_pm_prompt(user_message, project_id, project_context, plan_file_content)

    response = await isolated_query(
        sdk,
        prompt=prompt,
        system_prompt=PM_SYSTEM_PROMPT,
        cwd=project_dir,
        allowed_tools=[],
        max_turns=3,
    )

    if response.is_error:
        raise ValueError(f"PM Agent SDK error: {response.error_message}")

    graph = _parse_task_graph(response.text, project_id, user_message)
    graph = _enforce_artifact_requirements(graph)

    roles_used = list({t.role.value for t in graph.tasks})
    logger.info(
        "[PM] TaskGraph: %d tasks, %d roles, vision='%s'",
        len(graph.tasks), len(roles_used), graph.vision[:80],
    )
    return graph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_pm_prompt(
    user_message: str,
    project_id: str,
    project_context: str,
    plan_file_content: str,
) -> str:
    """Assemble the user-facing prompt sent to the PM agent."""
    parts = [
        f"<project_id>{html.escape(project_id)}</project_id>",
        f"<user_request>{html.escape(user_message)}</user_request>",
    ]
    if project_context:
        parts.append(f"<project_context>\n{project_context[:4000]}\n</project_context>")
    if plan_file_content:
        parts.append(
            "<plan_file>\n"
            "Convert this existing plan into a TaskGraph (do NOT re-plan from scratch):\n"
            f"{plan_file_content[:6000]}\n"
            "</plan_file>"
        )
    parts.append("\nCreate the TaskGraph JSON now. Output ONLY the JSON object.")
    return "\n\n".join(parts)


def _parse_task_graph(
    raw_text: str,
    project_id: str,
    user_message: str,
) -> TaskGraph:
    """Extract and validate a TaskGraph from the PM's raw response.

    Raises ValueError if no valid JSON is found.
    """
    candidates: list[str] = []

    # Try fenced JSON blocks first
    for match in _JSON_BLOCK_RE.finditer(raw_text):
        candidates.append(match.group(1).strip())

    # Try raw JSON (first top-level object)
    start = raw_text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(raw_text)):
            if raw_text[i] == "{":
                depth += 1
            elif raw_text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw_text[start : i + 1])
                    break

    for candidate in candidates:
        if len(candidate) > 500_000:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        data.setdefault("project_id", project_id)
        data.setdefault("user_message", user_message)

        try:
            graph = TaskGraph(**data)
        except (ValueError, TypeError) as exc:
            logger.debug("[PM] Pydantic validation failed: %s", exc)
            continue

        if not graph.tasks:
            continue

        # Reject vague one-liner goals
        for t in graph.tasks:
            if len(t.goal.split()) < 8:
                raise ValueError(
                    f"Task {t.id} goal too vague ({len(t.goal.split())} words): '{t.goal}'. "
                    "Each goal must be >= 8 words."
                )

        errors = graph.validate_dag()
        if errors:
            raise ValueError(f"DAG validation errors: {'; '.join(errors)}")

        return graph

    raise ValueError(
        f"No valid JSON TaskGraph found in PM response (length={len(raw_text)})"
    )


# ---------------------------------------------------------------------------
# Post-processing: enforce artifact requirements
# ---------------------------------------------------------------------------

_ROLE_DEFAULT_ARTIFACTS: dict[AgentRole, list[ArtifactType]] = {
    AgentRole.BACKEND_DEVELOPER: [ArtifactType.API_CONTRACT, ArtifactType.FILE_MANIFEST],
    AgentRole.FRONTEND_DEVELOPER: [ArtifactType.COMPONENT_MAP, ArtifactType.FILE_MANIFEST],
    AgentRole.DATABASE_EXPERT: [ArtifactType.SCHEMA, ArtifactType.FILE_MANIFEST],
    AgentRole.DEVOPS: [ArtifactType.DEPLOYMENT, ArtifactType.FILE_MANIFEST],
    AgentRole.TEST_ENGINEER: [ArtifactType.TEST_REPORT],
    AgentRole.SECURITY_AUDITOR: [ArtifactType.SECURITY_REPORT],
    AgentRole.REVIEWER: [ArtifactType.REVIEW_REPORT],
    AgentRole.RESEARCHER: [ArtifactType.RESEARCH],
}

def _enforce_artifact_requirements(graph: TaskGraph) -> TaskGraph:
    """Ensure every task has appropriate required_artifacts based on its role.

    If the PM forgot to add required_artifacts, we add sensible defaults.
    """
    for task in graph.tasks:
        defaults = _ROLE_DEFAULT_ARTIFACTS.get(task.role, [])
        if not task.required_artifacts and defaults:
            task.required_artifacts = list(defaults)
            logger.debug("[PM] Auto-added artifacts to %s: %s", task.id, defaults)

        if task.role in WRITER_ROLES:
            if ArtifactType.FILE_MANIFEST not in task.required_artifacts:
                task.required_artifacts.append(ArtifactType.FILE_MANIFEST)

    return graph
