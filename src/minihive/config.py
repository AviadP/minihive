"""Minimal configuration for Minihive.

Exposes agent registry, per-role helpers, and operational constants.
All values can be overridden via environment variables with sensible defaults.
No dotenv, no JSON overrides, no imports from other minihive modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ── Claude CLI path ──────────────────────────────────────────────────
CLAUDE_CLI_PATH: str = os.getenv("CLAUDE_CLI_PATH", "claude")

# ── Agent configuration ─────────────────────────────────────────────


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for a single agent role."""

    timeout: int = 900       # seconds
    turns: int = 100         # max_turns
    budget: float = 50.0     # USD per task
    layer: str = "execution" # brain | execution | quality
    emoji: str = "\U0001f527"  # default: wrench
    label: str = ""


AGENT_REGISTRY: dict[str, AgentConfig] = {
    # ── Brain ─────────────────────────────────────────────────────
    "pm": AgentConfig(
        timeout=600, turns=10, budget=10.0,
        layer="brain", emoji="\U0001f9e0", label="PM",
    ),
    # ── Execution ─────────────────────────────────────────────────
    "frontend_developer": AgentConfig(
        timeout=1800, turns=200, budget=50.0,
        layer="execution", emoji="\U0001f3a8", label="Frontend",
    ),
    "backend_developer": AgentConfig(
        timeout=1800, turns=200, budget=50.0,
        layer="execution", emoji="\u26a1", label="Backend",
    ),
    "database_expert": AgentConfig(
        timeout=900, turns=150, budget=50.0,
        layer="execution", emoji="\U0001f5c4\ufe0f", label="Database",
    ),
    "devops": AgentConfig(
        timeout=900, turns=150, budget=50.0,
        layer="execution", emoji="\U0001f680", label="DevOps",
    ),
    # ── Quality ───────────────────────────────────────────────────
    "security_auditor": AgentConfig(
        timeout=600, turns=50, budget=50.0,
        layer="quality", emoji="\U0001f510", label="Security",
    ),
    "test_engineer": AgentConfig(
        timeout=900, turns=100, budget=50.0,
        layer="quality", emoji="\U0001f9ea", label="Tester",
    ),
    "reviewer": AgentConfig(
        timeout=600, turns=50, budget=50.0,
        layer="quality", emoji="\U0001f50d", label="Reviewer",
    ),
    "researcher": AgentConfig(
        timeout=1200, turns=75, budget=50.0,
        layer="quality", emoji="\U0001f50e", label="Researcher",
    ),
}

# ── Operational constants (env-overridable) ──────────────────────────
MAX_TASK_RETRIES: int = int(os.getenv("MAX_TASK_RETRIES", "2"))
MAX_REMEDIATION_DEPTH: int = int(os.getenv("MAX_REMEDIATION_DEPTH", "2"))
MAX_TOTAL_REMEDIATIONS: int = int(os.getenv("MAX_TOTAL_REMEDIATIONS", "5"))
MAX_DAG_ROUNDS: int = int(os.getenv("MAX_DAG_ROUNDS", "30"))
DAG_MAX_CONCURRENT_NODES: int = int(os.getenv("DAG_MAX_CONCURRENT_NODES", "4"))
SUBPROCESS_MEDIUM_TIMEOUT: float = float(os.getenv("SUBPROCESS_MEDIUM_TIMEOUT", "30.0"))
MAX_BUDGET_USD: float = float(os.getenv("MAX_BUDGET_USD", "50.0"))
TIMEOUT_ESCALATION_FACTOR: float = float(os.getenv("TIMEOUT_ESCALATION_FACTOR", "1.5"))

# ── Registry helpers ─────────────────────────────────────────────────


def get_agent_config(role: str) -> AgentConfig:
    """Return the AgentConfig for *role*, falling back to defaults."""
    return AGENT_REGISTRY.get(role, AgentConfig())


def get_agent_timeout(role: str | None = None, retry_attempt: int = 0) -> int:
    """Return timeout (seconds) for *role* with escalation on retry."""
    base = get_agent_config(role).timeout if role else 900
    if retry_attempt >= 1:
        base = int(base * TIMEOUT_ESCALATION_FACTOR)
    return max(base, 30)


def get_agent_turns(role: str) -> int:
    """Return the max_turns for *role*."""
    return get_agent_config(role).turns


def get_agent_budget(role: str) -> float:
    """Return the per-task budget (USD) for *role*."""
    return get_agent_config(role).budget


