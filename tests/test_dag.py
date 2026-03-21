"""Tests for minihive.dag_executor helper functions (no SDK needed)."""

from minihive.contracts import AgentRole, TaskInput, TaskOutput, TaskStatus
from minihive.dag_executor import ExecutionResult, FileLockManager, _plan_batches, _split_writers_by_conflicts


# ---- helpers ---------------------------------------------------------------

def _make_task(tid: str, role: AgentRole = AgentRole.BACKEND_DEVELOPER, **kw) -> TaskInput:
    return TaskInput(id=tid, role=role, goal=kw.pop("goal", "Implement the full feature"), **kw)


# ---- ExecutionResult -------------------------------------------------------

def test_execution_result_summary():
    result = ExecutionResult(
        completed_tasks={
            "t1": TaskOutput(task_id="t1", status=TaskStatus.COMPLETED, summary="ok"),
            "t2": TaskOutput(task_id="t2", status=TaskStatus.FAILED, summary="nope"),
        },
        total_cost_usd=0.42,
        healing_history=[{"action": "remediation_created", "detail": "fix t2"}],
    )
    s = result.summary
    assert "1 succeeded" in s
    assert "1 failed" in s
    assert "$0.42" in s
    assert "remediation_created" in s


# ---- FileLockManager -------------------------------------------------------

def test_file_lock_manager():
    mgr = FileLockManager()
    assert hasattr(mgr, "acquire")
    assert hasattr(mgr, "release")
    # Each call creates a separate instance (no longer a singleton)
    mgr2 = FileLockManager()
    assert mgr is not mgr2


# ---- _plan_batches ---------------------------------------------------------

def test_plan_batches_separates_writers():
    w1 = _make_task("w1", role=AgentRole.BACKEND_DEVELOPER, files_scope=["src/a.py"])
    w2 = _make_task("w2", role=AgentRole.FRONTEND_DEVELOPER, files_scope=["src/a.py"])
    r1 = _make_task("r1", role=AgentRole.REVIEWER)

    batches = _plan_batches([w1, w2, r1])

    # Writers with overlap must be in separate batches
    writer_batches = [b for b in batches if any(t.role in {AgentRole.BACKEND_DEVELOPER, AgentRole.FRONTEND_DEVELOPER} for t in b)]
    w1_batch = next(i for i, b in enumerate(batches) if any(t.id == "w1" for t in b))
    w2_batch = next(i for i, b in enumerate(batches) if any(t.id == "w2" for t in b))
    assert w1_batch != w2_batch, "Overlapping writers must be in different batches"

    # Reader should be in a batch (last one, per implementation)
    reader_ids = {t.id for b in batches for t in b if t.role == AgentRole.REVIEWER}
    assert "r1" in reader_ids


# ---- _split_writers_by_conflicts -------------------------------------------

def test_split_writers_by_conflicts():
    # Overlapping scopes => separate batches
    w1 = _make_task("w1", files_scope=["src/shared.py", "src/a.py"])
    w2 = _make_task("w2", files_scope=["src/shared.py", "src/b.py"])
    batches = _split_writers_by_conflicts([w1, w2])
    assert len(batches) == 2

    # Non-overlapping scopes => same batch
    w3 = _make_task("w3", files_scope=["src/x.py"])
    w4 = _make_task("w4", files_scope=["src/y.py"])
    batches = _split_writers_by_conflicts([w3, w4])
    assert len(batches) == 1
    assert len(batches[0]) == 2
