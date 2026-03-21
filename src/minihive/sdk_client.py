"""Claude Agent SDK wrapper — simplified from Hivemind.

Wraps the Claude Agent SDK with concurrency limiting (asyncio.Semaphore),
error classification, orphan process cleanup, and project-directory sandboxing.

Removed vs Hivemind: _ConnectionPool, CircuitBreaker, query_with_retry,
_GranularEventEmitter, performance tracking, ETA estimation, threading locks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# SDK environment setup
# ---------------------------------------------------------------------------

_SUBPROCESS_TIMEOUT = 5


_SDK_ENV_CLEANED = False

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # noqa: E402
from claude_agent_sdk._internal.message_parser import parse_message  # noqa: E402
from claude_agent_sdk.types import (  # noqa: E402
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)


# ============================================================
# Error Classification
# ============================================================


class ErrorCategory(Enum):
    """Classification of SDK errors for retry/handling decisions."""

    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    SESSION = "session"
    AUTH = "auth"
    BUDGET = "budget"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


def classify_error(message: str) -> ErrorCategory:
    """Classify an error message to determine retry strategy."""
    if not message:
        return ErrorCategory.UNKNOWN

    lower = message.lower()

    # Timeout — transient
    if any(kw in lower for kw in ("timeout", "timed out", "deadline exceeded")):
        return ErrorCategory.TRANSIENT

    # Connection — transient
    if any(
        kw in lower
        for kw in (
            "connection", "connect", "network", "dns", "econnrefused",
            "econnreset", "broken pipe", "eof", "socket", "unavailable",
            "502", "503", "504",
        )
    ):
        return ErrorCategory.TRANSIENT

    # Rate limiting
    if any(
        kw in lower
        for kw in ("rate limit", "rate_limit", "429", "too many requests", "throttl")
    ):
        return ErrorCategory.RATE_LIMIT

    # Session/resume errors
    if any(kw in lower for kw in ("session", "resume", "invalid session", "expired session")):
        return ErrorCategory.SESSION

    # Authentication — permanent
    if any(
        kw in lower
        for kw in (
            "authentication", "unauthorized", "401", "403", "forbidden",
            "permission denied", "not logged in", "login required",
            "invalid api key", "api key",
        )
    ):
        return ErrorCategory.AUTH

    # Budget — permanent
    if any(kw in lower for kw in ("budget", "spending limit", "insufficient funds", "quota")):
        return ErrorCategory.BUDGET

    # Process spawn — transient (except macOS sandbox exit code 71)
    if any(kw in lower for kw in ("process", "spawn", "enoent", "exited with")):
        if "exit code 71" in lower or "exit code: 71" in lower:
            return ErrorCategory.PERMANENT
        return ErrorCategory.TRANSIENT

    # Content/validation — permanent
    if any(kw in lower for kw in ("invalid", "malformed", "bad request", "400")):
        return ErrorCategory.PERMANENT

    return ErrorCategory.UNKNOWN


# ============================================================
# SDK Response
# ============================================================


@dataclass
class SDKResponse:
    text: str
    session_id: str = ""
    cost_usd: float = 0.0
    is_error: bool = False
    error_message: str = ""
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    input_tokens: int = 0
    output_tokens: int = 0
    tool_uses: list[str] = field(default_factory=list)


# ============================================================
# Process Management
# ============================================================


def _snapshot_claude_pids() -> set[int]:
    """Return the set of PIDs for currently running Claude CLI processes."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "claude.*--output-format"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
        if result.returncode != 0:
            return set()
        pids: set[int] = set()
        for line in result.stdout.strip().splitlines():
            try:
                pids.add(int(line.strip()))
            except ValueError:
                continue
        return pids
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return set()


def _kill_specific_pids(pids: set[int], grace_period: float = 3.0) -> int:
    """Kill specific PIDs: SIGTERM, then SIGKILL after grace_period.

    Returns the number of processes that received SIGTERM.
    """
    if not pids:
        return 0

    killed = 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to Claude PID %d", pid)
            killed += 1
        except ProcessLookupError:
            logger.debug("SIGTERM: PID %d already gone", pid)
        except PermissionError:
            logger.warning("No permission to kill PID %d", pid)

    if killed == 0:
        return 0

    time.sleep(grace_period)

    for pid in pids:
        try:
            os.kill(pid, 0)  # Check if still alive
            os.kill(pid, signal.SIGKILL)
            logger.warning("Sent SIGKILL to stubborn Claude PID %d", pid)
        except ProcessLookupError:
            logger.debug("SIGKILL: PID %d already terminated", pid)
        except PermissionError:
            logger.debug("SIGKILL: no permission to signal PID %d", pid)

    return killed


# ============================================================
# Project Guard (file-access sandbox)
# ============================================================


def _make_project_guard(project_dir: str) -> Callable:
    """Return a ``can_use_tool`` callback that blocks access outside *project_dir*.

    Intercepts file-access tools (Read/Write/Edit/Glob/Grep) AND command
    execution (Bash) at the SDK level.  Denies any path that resolves
    outside the project directory and blocks known exfiltration commands.
    """
    from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny

    project_resolved = str(Path(project_dir).resolve())

    _PATH_PARAMS: dict[str, list[str]] = {
        "Read": ["file_path", "path"],
        "Write": ["file_path", "path"],
        "Edit": ["file_path", "path"],
        "Glob": ["path"],
        "Grep": ["path"],
    }

    _BASH_TOOL_NAMES = frozenset({"Bash", "execute_bash", "bash"})

    _EXFIL_COMMANDS = frozenset({
        "curl", "wget", "nc", "ncat", "netcat", "ssh", "scp",
        "rsync", "ftp", "sftp", "telnet",
    })

    _SAFE_PREFIXES = ("/usr/", "/bin/", "/sbin/", "/opt/", "/tmp/", "/dev/")

    def _check_bash_command(
        command: str,
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Validate a Bash command against the project boundary."""
        import shlex

        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()

        for token in tokens:
            bare = token.lstrip("-")
            if bare in _EXFIL_COMMANDS:
                logger.critical(
                    "[PROJECT GUARD] BLOCKED Bash exfiltration command: %s", bare,
                )
                return PermissionResultDeny(
                    message=f"Command '{bare}' is blocked — potential data exfiltration.",
                )

        for token in tokens:
            if not token.startswith("/"):
                continue
            if any(token.startswith(p) for p in _SAFE_PREFIXES):
                continue
            try:
                resolved = str(Path(token).resolve())
            except (OSError, ValueError):
                continue
            if resolved != project_resolved and not resolved.startswith(project_resolved + "/"):
                logger.critical(
                    "[PROJECT GUARD] BLOCKED Bash path escape: %r -> %r",
                    token, resolved,
                )
                return PermissionResultDeny(
                    message=f"Access denied: command references path outside project: {token}",
                )

        return PermissionResultAllow()

    async def _guard(
        tool_name: str, tool_input: dict, _ctx: object,
    ) -> PermissionResultAllow | PermissionResultDeny:
        # Bash tool — command-level validation
        if tool_name in _BASH_TOOL_NAMES:
            command = tool_input.get("command", "")
            if not command:
                return PermissionResultAllow()
            return _check_bash_command(command)

        # File-access tools — path validation
        params = _PATH_PARAMS.get(tool_name)
        if params is None:
            return PermissionResultAllow()

        raw_path = ""
        for param in params:
            raw_path = tool_input.get(param) or ""
            if raw_path:
                break

        if not raw_path:
            return PermissionResultAllow()

        try:
            resolved = str(Path(raw_path).resolve())
        except (OSError, ValueError):
            logger.warning(
                "[PROJECT GUARD] Cannot resolve path %r -- denying %s", raw_path, tool_name
            )
            return PermissionResultDeny(message=f"Cannot resolve path: {raw_path}")

        if resolved == project_resolved or resolved.startswith(project_resolved + "/"):
            return PermissionResultAllow()

        logger.critical(
            "[PROJECT GUARD] BLOCKED %s(%r) -- resolved to %r, outside %r",
            tool_name, raw_path, resolved, project_resolved,
        )
        return PermissionResultDeny(
            message=(
                f"Access denied: {raw_path!r} is outside the project directory. "
                f"You may only access files within {project_resolved}."
            )
        )

    return _guard


# ============================================================
# Claude SDK Manager
# ============================================================


class ClaudeSDKManager:
    """Thin wrapper over claude-agent-sdk with semaphore concurrency control,
    error classification, and orphan process cleanup.
    """

    def __init__(self, cli_path: str = "claude", max_concurrent: int = 5) -> None:
        self._cli_path = cli_path
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Remove CLAUDECODE env var so the SDK can spawn claude subprocesses.
        # Without this, the SDK refuses with 'cannot launch inside another
        # Claude Code session'.  Done here (not at import time) so the env
        # is only modified when a manager is explicitly created.
        global _SDK_ENV_CLEANED
        if not _SDK_ENV_CLEANED:
            os.environ.pop("CLAUDECODE", None)
            _SDK_ENV_CLEANED = True

    async def query(
        self,
        prompt: str,
        system_prompt: str,
        cwd: str,
        session_id: str | None = None,
        max_turns: int = 10,
        max_budget_usd: float = 2.0,
        allowed_tools: list[str] | None = None,
        on_stream: Callable | None = None,
        agent_role: str | None = None,
        timeout: float = 300.0,
    ) -> SDKResponse:
        """Send a query to Claude Agent SDK.

        Acquires a semaphore slot, snapshots PIDs, runs the SDK stream,
        and cleans up orphan processes on error/timeout.
        """
        request_id = f"req_{int(time.monotonic() * 1000) % 100000}"
        pids_before = _snapshot_claude_pids()

        await self._semaphore.acquire()
        try:
            result = await asyncio.wait_for(
                self._run_query(
                    prompt, system_prompt, cwd, session_id, max_turns,
                    max_budget_usd, allowed_tools, on_stream,
                    request_id,
                ),
                timeout=timeout,
            )

            if result.is_error:
                result.error_category = classify_error(result.error_message)

            return result

        except TimeoutError:
            logger.warning(
                "[%s] SDK query TIMEOUT after %.0fs (role=%s)",
                request_id, timeout, agent_role or "default",
            )
            await asyncio.to_thread(
                self._kill_orphans, pids_before
            )
            return SDKResponse(
                text=f"Error: Agent timed out after {timeout} seconds",
                is_error=True,
                error_message=f"Timeout after {timeout}s",
                error_category=ErrorCategory.TRANSIENT,
            )

        except asyncio.CancelledError:
            logger.info("[%s] SDK query CANCELLED", request_id)
            await asyncio.to_thread(self._kill_orphans, pids_before)
            raise

        # SDK boundary catch-all: covers network, runtime, type, and timeout
        # errors that may surface from the SDK or its subprocess management.
        except (OSError, RuntimeError, ValueError, TypeError, asyncio.TimeoutError) as exc:
            logger.error("[%s] SDK query EXCEPTION: %s", request_id, exc, exc_info=True)
            await asyncio.to_thread(self._kill_orphans, pids_before)
            category = classify_error(str(exc))
            return SDKResponse(
                text=f"Error: {exc}",
                is_error=True,
                error_message=str(exc),
                error_category=category,
            )

        finally:
            self._semaphore.release()

    async def _run_query(
        self,
        prompt: str,
        system_prompt: str,
        cwd: str,
        session_id: str | None,
        max_turns: int,
        max_budget_usd: float,
        allowed_tools: list[str] | None,
        on_stream: Callable | None,
        request_id: str,
    ) -> SDKResponse:
        """Create SDK client, consume message stream, return SDKResponse."""
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            cwd=cwd,
            cli_path=self._cli_path,
            include_partial_messages=True,
            # Security: bypassPermissions is required for automated agents
            # (no human to click "allow").  _make_project_guard compensates
            # by intercepting Read/Write/Edit/Glob/Grep AND Bash at the
            # SDK level, blocking any access outside the project directory.
            sandbox={"enabled": False},
            permission_mode="bypassPermissions",
            can_use_tool=_make_project_guard(cwd),
        )
        if allowed_tools is not None:
            options.allowed_tools = allowed_tools
        if session_id:
            options.resume = session_id

        text_parts: list[str] = []
        result_session_id = ""
        cost_usd = 0.0
        input_tokens = 0
        output_tokens = 0
        last_seen_text = ""
        tool_uses: list[str] = []

        client = ClaudeSDKClient(options)
        try:
            await client.connect()
            await client.query(prompt)

            async for raw_data in client._query.receive_messages():
                try:
                    message = parse_message(raw_data)
                except (ValueError, TypeError, KeyError) as parse_err:
                    logger.debug("[%s] Skipping unparseable message: %s", request_id, parse_err)
                    continue
                if message is None:
                    continue

                if isinstance(message, AssistantMessage):
                    turn_text, tool_info = self._extract_assistant_content(
                        message, tool_uses
                    )

                    if on_stream and (turn_text != last_seen_text or tool_info):
                        await self._fire_stream(on_stream, turn_text, last_seen_text, tool_info)
                        last_seen_text = turn_text

                    self._collect_text(text_parts, turn_text)

                elif isinstance(message, ResultMessage):
                    return self._build_result(
                        message, text_parts, tool_uses, cost_usd,
                        input_tokens, output_tokens,
                    )

        except asyncio.CancelledError:
            raise
        except RuntimeError as exc:
            if "cancel scope" in str(exc):
                logger.warning("[%s] anyio cancel scope error (suppressed)", request_id)
                combined = "\n\n".join(text_parts).strip()
                return SDKResponse(
                    text=combined or "Agent interrupted (anyio cleanup error).",
                    is_error=True,
                    error_message="anyio cancel scope error",
                    error_category=ErrorCategory.TRANSIENT,
                )
            raise
        finally:
            try:
                await client.disconnect()
            except RuntimeError as exc:
                if "cancel scope" not in str(exc):
                    logger.warning("[%s] RuntimeError during disconnect: %s", request_id, exc)
            except (OSError, ConnectionError) as dc_err:
                logger.debug("[%s] Disconnect error: %s", request_id, dc_err)

        # Stream ended without ResultMessage
        combined = "\n\n".join(text_parts).strip()
        return SDKResponse(
            text=combined or "Agent produced no output (stream ended unexpectedly).",
            is_error=not bool(combined),
            error_message="" if combined else "Stream ended without ResultMessage",
            error_category=ErrorCategory.UNKNOWN if combined else ErrorCategory.TRANSIENT,
            tool_uses=tool_uses,
        )

    # ---- Private helpers ----

    @staticmethod
    def _extract_assistant_content(
        message: AssistantMessage, tool_uses: list[str],
    ) -> tuple[str, str]:
        """Extract text and tool info from an AssistantMessage."""
        turn_text = ""
        tool_info = ""
        for block in message.content:
            if isinstance(block, TextBlock):
                turn_text += block.text
            elif isinstance(block, ToolUseBlock):
                tool_name = block.name
                tool_input = block.input or {}
                tool_uses.append(tool_name)

                if tool_name in ("Read", "read_file"):
                    path = tool_input.get("file_path") or tool_input.get("path", "")
                    tool_info = f"Reading: {path}"
                elif tool_name in ("Write", "write_file", "create_file"):
                    path = tool_input.get("file_path") or tool_input.get("path", "")
                    tool_info = f"Writing: {path}"
                elif tool_name in ("Edit", "edit_file"):
                    path = tool_input.get("file_path") or tool_input.get("path", "")
                    tool_info = f"Editing: {path}"
                elif tool_name in ("Bash", "execute_bash", "bash"):
                    cmd = str(tool_input.get("command", ""))[:100]
                    tool_info = f"Running: `{cmd}`"
                elif tool_name in ("Glob", "glob", "ListFiles"):
                    pattern = tool_input.get("pattern", "")
                    tool_info = f"Searching: {pattern}"
                elif tool_name in ("Grep", "grep", "SearchFiles"):
                    pattern = tool_input.get("pattern", "")
                    tool_info = f"Grep: {pattern}"
                else:
                    tool_info = f"Tool: {tool_name}"

        return turn_text, tool_info

    @staticmethod
    async def _fire_stream(
        on_stream: Callable, turn_text: str, last_seen: str, tool_info: str,
    ) -> None:
        """Fire the on_stream callback with meaningful updates."""
        update = ""
        if tool_info:
            update = tool_info
        if turn_text and turn_text != last_seen:
            new_text = turn_text[len(last_seen):]
            preview = new_text[-300:] if len(new_text) > 300 else new_text
            update = f"{update}\n\n{preview}" if update else preview
        if update:
            try:
                await on_stream(update)
            except (TypeError, ValueError, RuntimeError) as exc:
                logger.error("Stream callback error: %s", exc)

    @staticmethod
    def _collect_text(text_parts: list[str], turn_text: str) -> None:
        """Append turn_text to text_parts, deduplicating streaming partials."""
        if not turn_text:
            return
        if text_parts and turn_text.startswith(text_parts[-1]):
            text_parts[-1] = turn_text
        elif not text_parts or turn_text != text_parts[-1]:
            text_parts.append(turn_text)

    @staticmethod
    def _build_result(
        message: ResultMessage,
        text_parts: list[str],
        tool_uses: list[str],
        cost_usd: float,
        input_tokens: int,
        output_tokens: int,
    ) -> SDKResponse:
        """Build an SDKResponse from a ResultMessage."""
        session_id = message.session_id or ""
        cost_usd = message.total_cost_usd or 0.0

        usage = message.usage or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)

        if message.result and message.result not in text_parts:
            text_parts.append(message.result)

        combined = "\n\n".join(text_parts).strip()

        if not combined and not message.is_error:
            tools_summary = ", ".join(set(tool_uses)) if tool_uses else "unknown"
            combined = (
                f"Task completed via tool use (tools: {tools_summary}). "
                "No text output -- work was done directly."
            )

        error_msg = ""
        category = ErrorCategory.UNKNOWN
        if message.is_error:
            error_msg = message.result or "Unknown error"
            category = classify_error(error_msg)

        return SDKResponse(
            text=combined,
            session_id=session_id,
            cost_usd=cost_usd,
            is_error=message.is_error,
            error_message=error_msg,
            error_category=category,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_uses=tool_uses,
        )

    @staticmethod
    def _kill_orphans(pids_before: set[int]) -> int:
        """Find and kill Claude processes spawned after pids_before snapshot."""
        try:
            pids_after = _snapshot_claude_pids()
            orphans = pids_after - pids_before
            if not orphans:
                return 0
            logger.warning("Found %d orphan Claude process(es): %s", len(orphans), orphans)
            return _kill_specific_pids(orphans)
        except (subprocess.SubprocessError, OSError) as exc:
            logger.debug("Error during orphan cleanup: %s", exc)
            return 0
