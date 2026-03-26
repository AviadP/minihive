---
description: Run multi-agent orchestrator — PM plans, agents build in parallel, test, review, commit
argument: task description (or --plan-file path)
---

Run the minihive multi-agent orchestrator on the current project.

IMPORTANT: Before running, determine how the user provided their input:
- If $ARGUMENTS is a file path (ends in .md, .txt, or starts with / or ./): use `--plan-file`
- If $ARGUMENTS is a task description (plain English): pass as quoted positional argument

First, check if minihive venv exists. If not, set it up:
```bash
cd ~/minihive && uv venv && uv pip install -e . 2>&1 | tail -3
```

Then run:
```bash
# Plan file mode (argument is a file path):
~/minihive/.venv/bin/python -m minihive --project-dir "$PWD" --yes --plan-file /path/to/plan.md

# Task description mode (argument is plain English):
~/minihive/.venv/bin/python -m minihive --project-dir "$PWD" --yes "Add JWT authentication"

# Dry run (add --dry-run to see the plan without executing):
~/minihive/.venv/bin/python -m minihive --project-dir "$PWD" --dry-run "describe the task"
```

Stream the output to the user and report the final summary when done.
