"""
file_context.py — JIT context passing between agents via artifact references.

Instead of passing full text summaries between agents (which degrades like
a game of telephone), this module maintains a registry of real files
produced by each task. Downstream agents receive lightweight file-path
references and read the source of truth directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from minihive.contracts import TaskInput, TaskOutput, TaskStatus

logger = logging.getLogger(__name__)

# ── File-type inference ──────────────────────────────────────────────────────

_EXT_MAP: dict[str, str] = {
    ".ts": "code",
    ".tsx": "code",
    ".js": "code",
    ".jsx": "code",
    ".py": "code",
    ".go": "code",
    ".rs": "code",
    ".java": "code",
    ".sql": "code",
    ".sh": "code",
    ".css": "code",
    ".scss": "code",
    ".html": "markup",
    ".xml": "markup",
    ".svg": "markup",
    ".json": "data",
    ".yaml": "data",
    ".yml": "data",
    ".toml": "data",
    ".csv": "data",
    ".env": "data",
    ".md": "doc",
    ".txt": "doc",
    ".rst": "doc",
    ".png": "asset",
    ".jpg": "asset",
    ".gif": "asset",
    ".ico": "asset",
    ".woff": "asset",
    ".woff2": "asset",
    ".ttf": "asset",
    ".lock": "lockfile",
}


def infer_file_type(path: str) -> str:
    """Return a human-friendly file type label based on extension."""
    ext = Path(path).suffix.lower()
    return _EXT_MAP.get(ext, "file")


# ── Artifact Reference ───────────────────────────────────────────────────────


@dataclass
class ArtifactRef:
    """A lightweight pointer to a file produced by a task."""

    task_id: str
    path: str  # relative to project root
    file_type: str  # code | data | doc | asset | ...
    description: str


# ── Artifact Registry ────────────────────────────────────────────────────────


class ArtifactRegistry:
    """Tracks file artifacts produced during DAG execution.

    Lifecycle:
        1. Created once per execution.
        2. After each successful task, ``register(task_id, output)`` is called.
        3. Before each task prompt is built, ``get_context_for_task(task)``
           returns upstream file references.
    """

    def __init__(self, project_dir: str) -> None:
        self._project_dir = project_dir
        self._refs: dict[str, list[ArtifactRef]] = {}  # task_id -> refs

    def register(self, task_id: str, output: TaskOutput) -> int:
        """Extract file references from a completed task output.

        Returns the number of artifacts registered.
        """
        if output.status != TaskStatus.COMPLETED:
            return 0

        refs: list[ArtifactRef] = []
        seen_paths: set[str] = set()

        # Structured artifacts (typed, with metadata)
        for art in output.structured_artifacts:
            path = art.file_path
            if not path or path in seen_paths:
                continue
            resolved = self._resolve(path)
            if resolved and os.path.exists(resolved):
                refs.append(
                    ArtifactRef(
                        task_id=task_id,
                        path=path,
                        file_type=infer_file_type(path),
                        description=art.title,
                    )
                )
                seen_paths.add(path)

        # Plain artifact paths (list[str])
        for path in output.artifacts:
            if path in seen_paths:
                continue
            resolved = self._resolve(path)
            if resolved and os.path.exists(resolved):
                refs.append(
                    ArtifactRef(
                        task_id=task_id,
                        path=path,
                        file_type=infer_file_type(path),
                        description=f"File produced by task {task_id}",
                    )
                )
                seen_paths.add(path)

        self._refs[task_id] = refs
        if refs:
            logger.info(
                "[file_context] Registered %d artifacts from task %s: %s",
                len(refs),
                task_id,
                [r.path for r in refs],
            )
        return len(refs)

    def get_context_for_task(self, task: TaskInput) -> list[ArtifactRef]:
        """Get upstream artifacts for a task based on context_from."""
        refs: list[ArtifactRef] = []
        seen: set[str] = set()

        for upstream_id in task.context_from:
            for ref in self._refs.get(upstream_id, []):
                if ref.path not in seen:
                    refs.append(ref)
                    seen.add(ref.path)

        # Also include input_artifacts declared on the task
        for path in task.input_artifacts:
            if path in seen:
                continue
            resolved = self._resolve(path)
            if resolved and os.path.exists(resolved):
                refs.append(
                    ArtifactRef(
                        task_id="input",
                        path=path,
                        file_type=infer_file_type(path),
                        description=f"Input artifact: {Path(path).name}",
                    )
                )
                seen.add(path)

        return refs

    def _resolve(self, path: str) -> str:
        """Resolve a path relative to project_dir if not absolute."""
        if os.path.isabs(path):
            return path
        return os.path.join(self._project_dir, path)
