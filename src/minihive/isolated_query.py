"""Thread-isolated event loop runner for Claude Agent SDK queries.

Solves the anyio cancel-scope bug by running each SDK query() call inside a
**dedicated asyncio event loop on a separate thread**.  If the SDK's internal
anyio cleanup leaks a cancel-scope into the event loop, only that throwaway
loop is poisoned -- the main application loop stays clean.

Architecture
------------
Main event loop (orchestrator / CLI)
  +-- calls ``isolated_query()``
       +-- spawns a **thread** via ThreadPoolExecutor
            +-- ``asyncio.run(_inner())`` with async-gen finalizer disabled
                 +-- calls ``sdk._run_query()`` (bypasses loop-bound semaphore)

The thread pool size (8 workers) IS the concurrency limiter -- no need for
the asyncio.Semaphore pool inside the isolated thread.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any

from minihive.sdk_client import (
    ClaudeSDKManager,
    ErrorCategory,
    SDKResponse,
    classify_error,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread pool for isolated queries.  Each thread gets its own event loop.
# Size caps concurrent agents.  This IS the concurrency limiter -- the
# asyncio.Semaphore inside ClaudeSDKManager is bypassed (loop-bound).
# ---------------------------------------------------------------------------
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="isolated-sdk",
)

# Retry configuration
_MAX_RETRIES = 2
_RETRY_BACKOFF_BASE = 2  # seconds; exponential: 2, 4
_RATE_LIMIT_BACKOFF_BASE = 5  # seconds; exponential: 5, 15


# ============================================================
# Fresh-loop runner (THE cancel-scope fix)
# ============================================================


def _run_in_fresh_loop(
    coro_factory: Callable[[], Awaitable[SDKResponse]],
) -> SDKResponse:
    """Run *coro_factory()* in a brand-new event loop on the current thread.

    This is the function that executes inside the thread pool.  It creates a
    fresh ``asyncio.run()`` so the anyio cancel-scope leak cannot infect the
    caller's event loop.

    ROOT CAUSE FIX: We disable the async generator finalizer on the isolated
    loop.  Without this, Python's GC can finalize the SDK's async generator
    *after* the isolated loop has closed, which triggers anyio's cancel-scope
    cleanup on the MAIN event loop -- injecting a CancelledError into unrelated
    tasks.  ``asyncio.run()`` still calls ``shutdown_asyncgens()`` as part of
    its normal teardown, which handles cleanup safely within the same task
    context.
    """

    async def _inner() -> SDKResponse:
        # Disable async generator finalizer for THIS loop.
        # NOTE: finalizer=None means "don't change" -- use a no-op lambda.
        sys.set_asyncgen_hooks(firstiter=None, finalizer=lambda agen: None)
        return await coro_factory()

    try:
        return asyncio.run(_inner())
    except RuntimeError as exc:
        if "cancel scope" in str(exc):
            logger.warning(
                "Isolated loop caught anyio cancel-scope error (contained): %s",
                exc,
            )
            return SDKResponse(
                text="Agent completed but cleanup had an anyio error (contained).",
                is_error=True,
                error_message=f"anyio cancel scope (isolated): {exc}",
                error_category=ErrorCategory.TRANSIENT,
            )
        raise


# ============================================================
# Backoff helper
# ============================================================


def _backoff_seconds(category: ErrorCategory, attempt: int) -> float:
    """Return the sleep duration for a retry attempt based on error category."""
    if category == ErrorCategory.RATE_LIMIT:
        return min(_RATE_LIMIT_BACKOFF_BASE * (3 ** (attempt - 1)), 30.0)
    if category == ErrorCategory.SESSION:
        return 0.5
    # TRANSIENT / UNKNOWN
    return min(_RETRY_BACKOFF_BASE * (2 ** (attempt - 1)), 8.0)


# ============================================================
# Public API
# ============================================================


async def isolated_query(
    sdk: ClaudeSDKManager,
    *,
    prompt: str,
    system_prompt: str,
    cwd: str = ".",
    session_id: str | None = None,
    max_turns: int = 10,
    max_budget_usd: float = 2.0,
    allowed_tools: list[str] | None = None,
    on_stream: Callable[..., Any] | None = None,
    agent_role: str | None = None,
    timeout: float = 300.0,
) -> SDKResponse:
    """Run an SDK query in a thread-isolated event loop.

    Drop-in replacement for ``sdk.query()`` that provides event-loop
    isolation.  The caller's event loop is never exposed to anyio's
    cancel-scope cleanup.

    Calls ``sdk._run_query()`` directly (bypasses the loop-bound semaphore).
    The thread pool size acts as the concurrency limiter instead.

    Retries transient / rate-limit errors up to ``_MAX_RETRIES`` times with
    exponential backoff.  Auth, budget, and permanent errors are returned
    immediately.
    """
    request_id = f"iso_{int(time.monotonic() * 1000) % 100000}"
    logger.info(
        "[%s] Starting isolated query: role=%s, max_turns=%d, budget=$%.2f",
        request_id,
        agent_role or "default",
        max_turns,
        max_budget_usd,
    )

    caller_loop = asyncio.get_running_loop()

    last_response: SDKResponse | None = None

    for attempt in range(1, _MAX_RETRIES + 2):  # 1..max_retries+1

        def _query_factory() -> Awaitable[SDKResponse]:
            """Create the coroutine to run in the isolated loop.

            Calls sdk._run_query() directly -- bypasses the loop-bound
            semaphore.  The thread pool size limits concurrency.
            """
            return sdk._run_query(
                prompt=prompt,
                system_prompt=system_prompt,
                cwd=cwd,
                session_id=session_id,
                max_turns=max_turns,
                max_budget_usd=max_budget_usd,
                allowed_tools=allowed_tools,
                on_stream=None,  # No streaming in isolated mode
                request_id=request_id,
            )

        try:
            result: SDKResponse = await caller_loop.run_in_executor(
                _executor,
                _run_in_fresh_loop,
                _query_factory,
            )
        except TimeoutError:
            logger.warning("[%s] Isolated query timed out (attempt %d)", request_id, attempt)
            result = SDKResponse(
                text=f"Error: Agent timed out after {timeout}s",
                is_error=True,
                error_message=f"Timeout after {timeout}s",
                error_category=ErrorCategory.TRANSIENT,
            )
        except RuntimeError as exc:
            if "cancel scope" in str(exc):
                logger.warning("[%s] anyio cancel scope on attempt %d", request_id, attempt)
                result = SDKResponse(
                    text="Agent interrupted (anyio error).",
                    is_error=True,
                    error_message="anyio cancel scope error",
                    error_category=ErrorCategory.TRANSIENT,
                )
            else:
                raise

        if not result.is_error:
            logger.info(
                "[%s] Isolated query succeeded: cost=$%.4f, tokens_in=%d, tokens_out=%d",
                request_id,
                result.cost_usd,
                result.input_tokens,
                result.output_tokens,
            )
            return result

        # Classify and decide whether to retry
        category = result.error_category or classify_error(result.error_message)
        result.error_category = category
        last_response = result

        if category in (ErrorCategory.AUTH, ErrorCategory.BUDGET, ErrorCategory.PERMANENT):
            logger.warning(
                "[%s] Non-retryable error (%s): %s",
                request_id,
                category.value,
                result.error_message,
            )
            return result

        if attempt > _MAX_RETRIES:
            logger.warning(
                "[%s] All %d retries exhausted: %s",
                request_id,
                _MAX_RETRIES,
                result.error_message,
            )
            return result

        delay = _backoff_seconds(category, attempt)
        logger.info(
            "[%s] Retryable error (%s), attempt %d/%d, backoff %.1fs: %s",
            request_id,
            category.value,
            attempt,
            _MAX_RETRIES + 1,
            delay,
            result.error_message,
        )
        await asyncio.sleep(delay)

    # Should not reach here, but safety net
    return last_response or SDKResponse(
        text="Error: All retry attempts failed",
        is_error=True,
        error_message="All retries exhausted",
        error_category=ErrorCategory.UNKNOWN,
    )
