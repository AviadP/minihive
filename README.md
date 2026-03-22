# Minihive

Multi-agent orchestrator for Claude Code. Describe a feature — a PM plans, specialist agents build in parallel, tests run, code is reviewed, and everything is committed.

Extracted from [Hivemind](https://github.com/cohen-liel/hivemind) — same core engine, no dashboard, no database, no web server. Just the orchestration.

## How It Works

```
You: "Add user authentication with JWT tokens and a login page"
                    |
                    v
         +------------------+
         |    PM Agent       |  Analyzes request, creates TaskGraph (DAG)
         |    (Planning)     |  with dependencies and file scopes
         +--------+---------+
                  |
         +--------v---------+
         |   DAG Executor    |  Launches agents in parallel
         |   (Orchestration) |  where dependencies allow
         +--------+---------+
                  |
    +-------------+-------------+
    v             v             v
+--------+  +--------+  +--------+
|Backend |  |Frontend|  |Database|   Agents work in parallel,
|  Dev   |  |  Dev   |  | Expert |   passing typed artifacts downstream
+---+----+  +---+----+  +---+----+
    |           |           |
    +-----+-----+-----------+
          v
    +------------------+
    |   Test Engineer   |   Tests the combined output
    +--------+---------+
             v
    +------------------+
    |    Reviewer       |   Quality gate — checks correctness,
    |  (Code Review)    |   consistency, and code quality
    +--------+---------+
             v
        Committed & Ready
```

## Install

Requires Python 3.11+ and Claude Code CLI (`claude` command).

```bash
git clone <this-repo> ~/minihive
cd ~/minihive
uv venv && uv pip install -e .

# Install as Claude Code slash command (available globally)
mkdir -p ~/.claude/commands
cp ~/minihive/commands/minihive.md ~/.claude/commands/minihive.md
```

## Usage

### CLI

```bash
# Basic — describe what you want
minihive "Add JWT authentication with a login page and protected routes"

# Specify project directory and budget
minihive --project-dir ~/myapp --budget 30 "Refactor the auth module"

# Dry run — see the plan without executing
minihive --dry-run "Add dark mode support"

# Skip confirmation prompt
minihive --yes "Add a health check endpoint"

# Execute from a plan file (Claude Code plan, markdown task list, etc.)
minihive --plan-file .claude/plans/my-plan.md

# Combine: convert plan to TaskGraph and preview it
minihive --plan-file plan.md --dry-run

# Resume after interruption (picks up from .minihive/checkpoint.json)
minihive --resume --project-dir ~/myapp
```

### Claude Code Slash Command

After installing the command file, use `/minihive` in any Claude Code session:

```
/minihive Add JWT authentication with a login page
/minihive --plan-file .claude/plans/my-plan.md
```

### Logs and Checkpoints

All output is saved to `.minihive/run.log` in the project directory — readable after background runs. Execution state is checkpointed to `.minihive/checkpoint.json` after each round, so interrupted runs can be resumed with `--resume`.

## What It Does

1. **PM Agent** reads your prompt (and optionally a plan file), analyzes the codebase context (CLAUDE.md, README.md, file tree), and produces a structured TaskGraph — a DAG of tasks with dependencies, role assignments, file scopes, and artifact contracts.

2. **DAG Executor** runs tasks in parallel where dependencies allow. Writer agents (backend, frontend, database, devops) are serialized when their file scopes overlap. Reader agents (tester, reviewer, security) run in parallel.

3. **Each task** spawns a Claude Code subprocess with a specialized system prompt. Agents produce typed artifacts (API contracts, schemas, test reports) that are passed to downstream tasks — no "telephone game" information loss.

4. **Self-healing**: when a task fails, the system classifies WHY (build error, test failure, missing dependency, etc.) and either retries or injects a targeted remediation task into the DAG.

5. **Git discipline**: only the DAG executor commits — never individual agents. Each successful task gets its own commit. Sensitive files (.env, *.key) are automatically excluded.

## Agent Roster

| Role | Layer | Specialty |
|------|-------|-----------|
| PM | Brain | Task planning, DAG creation |
| Frontend Developer | Execution | React, TypeScript, Tailwind, accessibility |
| Backend Developer | Execution | FastAPI, async Python, REST APIs, auth |
| Database Expert | Execution | Schema design, migrations, query optimization |
| DevOps | Execution | Docker, CI/CD, deployment |
| Test Engineer | Quality | pytest, TDD, verification loops |
| Security Auditor | Quality | OWASP Top 10, dependency scanning |
| Reviewer | Quality | Code review, architecture, quality gates |
| Researcher | Quality | Web research, documentation |

## Configuration

All settings via environment variables (no config files needed):

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_CLI_PATH` | `claude` | Path to Claude CLI |
| `MINIHIVE_MAX_BUDGET` | `50.0` | Max budget per run (USD) |
| `MINIHIVE_MAX_CONCURRENT` | `4` | Max parallel agents |
| `MINIHIVE_MAX_RETRIES` | `2` | Retries per failed task |
| `MINIHIVE_MAX_REMEDIATIONS` | `5` | Max auto-fix tasks per run |

## Project Structure

```
~/minihive/
├── plugin.json              # Claude Code plugin manifest
├── commands/minihive.md     # /minihive slash command
├── src/minihive/
│   ├── __main__.py          # CLI entry point
│   ├── contracts.py         # TaskInput, TaskOutput, TaskGraph, failure classification
│   ├── config.py            # Agent registry, execution limits
│   ├── prompts.py           # System prompts for all agent roles
│   ├── sdk_client.py        # Claude SDK wrapper, error handling, PID tracking
│   ├── isolated_query.py    # Thread-isolated event loops (anyio bug workaround)
│   ├── pm_agent.py          # PM creates TaskGraph from user prompt or plan file
│   ├── dag_executor.py      # DAG execution, parallel batching, self-healing
│   ├── git_ops.py           # Auto-commit after each task
│   ├── file_context.py      # Artifact registry for inter-agent context passing
│   └── project_context.py   # Load CLAUDE.md / README.md for agent context
└── tests/
```

## How It Differs from Hivemind

Minihive is the orchestration core only. Removed:

- Web dashboard and React frontend
- FastAPI server and WebSocket streaming
- SQLAlchemy database and Alembic migrations
- Device authentication and QR codes
- Scheduler, templates, and marketplace
- Memory agent, architect agent, debate engine, reflexion
- Blackboard shared memory, cross-project knowledge
- Active escalation, round-based orchestrator
- 70+ bundled skills

What remains: PM planning, DAG execution, parallel agents, typed artifact passing, self-healing, and git discipline. ~4,100 lines vs ~10,300.

## Auth

Works with whatever Claude authentication you have configured — OAuth subscription (`claude login`) or API key. Minihive doesn't force either method.

## License

Apache 2.0 (same as Hivemind)
