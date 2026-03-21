---
description: Run multi-agent orchestrator — PM plans, agents build in parallel, test, review, commit
argument: task description (or --plan-file path)
---

Run the minihive multi-agent orchestrator on the current project.

Execute this command in the user's project directory:

```bash
python -m minihive --project-dir "$PWD" --yes $ARGUMENTS
```

Stream the output to the user and report the final summary when done.
If the user provides a plan file path, pass it as: `--plan-file <path>`
