"""Tests for minihive.contracts module."""

import pytest
from pydantic import ValidationError

from minihive.contracts import (
    AgentRole,
    FailureCategory,
    TaskGraph,
    TaskInput,
    TaskOutput,
    TaskStatus,
    WRITER_ROLES,
    READER_ROLES,
    classify_failure,
    create_remediation_task,
    extract_task_output,
)


# ---- helpers ---------------------------------------------------------------

def _make_task(tid: str, goal: str = "Implement the feature fully", **kw) -> TaskInput:
    return TaskInput(id=tid, role=kw.pop("role", AgentRole.BACKEND_DEVELOPER), goal=goal, **kw)


def _make_graph(tasks: list[TaskInput]) -> TaskGraph:
    return TaskGraph(
        project_id="test",
        user_message="test request",
        vision="Test vision statement",
        tasks=tasks,
    )


# ---- TaskInput validation --------------------------------------------------

def test_task_input_valid():
    t = _make_task("task_001", goal="Build the user login system")
    assert t.id == "task_001"
    assert t.role == AgentRole.BACKEND_DEVELOPER
    assert t.goal == "Build the user login system"
    assert t.depends_on == []
    assert t.files_scope == []


def test_task_input_invalid_id():
    with pytest.raises(ValidationError, match="Invalid task id"):
        _make_task("bad id!")


def test_task_input_short_goal():
    with pytest.raises(ValidationError, match="at least 10"):
        _make_task("ok_id", goal="short")


# ---- TaskOutput ------------------------------------------------------------

def test_task_output_successful():
    out = TaskOutput(task_id="t1", status=TaskStatus.COMPLETED, summary="All good here.")
    assert out.is_successful() is True


def test_task_output_failed():
    out = TaskOutput(task_id="t1", status=TaskStatus.FAILED, summary="Something broke.")
    assert out.is_successful() is False


# ---- TaskGraph -------------------------------------------------------------

def test_task_graph_ready_tasks():
    a = _make_task("A")
    b = _make_task("B", depends_on=["A"])
    c = _make_task("C")
    graph = _make_graph([a, b, c])

    # Nothing completed => A and C are ready (no deps)
    ready_ids = {t.id for t in graph.ready_tasks({})}
    assert ready_ids == {"A", "C"}

    # A completed => B becomes ready, C already done via completed dict
    completed = {
        "A": TaskOutput(task_id="A", status=TaskStatus.COMPLETED, summary="done"),
        "C": TaskOutput(task_id="C", status=TaskStatus.COMPLETED, summary="done"),
    }
    ready_ids = {t.id for t in graph.ready_tasks(completed)}
    assert ready_ids == {"B"}


def test_task_graph_validate_dag_cycle():
    a = _make_task("A", depends_on=["B"])
    b = _make_task("B", depends_on=["A"])
    graph = _make_graph([a, b])

    errors = graph.validate_dag()
    assert any("ycle" in e for e in errors)


def test_task_graph_validate_dag_valid():
    a = _make_task("A")
    b = _make_task("B", depends_on=["A"])
    graph = _make_graph([a, b])

    assert graph.validate_dag() == []


# ---- Failure classification ------------------------------------------------

def test_classify_failure_build_error():
    out = TaskOutput(task_id="t1", status=TaskStatus.FAILED, summary="syntax error in main.py")
    assert classify_failure(out) == FailureCategory.BUILD_ERROR


def test_classify_failure_test_failure():
    out = TaskOutput(task_id="t1", status=TaskStatus.FAILED, summary="test failed on login")
    assert classify_failure(out) == FailureCategory.TEST_FAILURE


def test_classify_failure_unknown():
    out = TaskOutput(task_id="t1", status=TaskStatus.FAILED, summary="something happened")
    assert classify_failure(out) == FailureCategory.UNKNOWN


# ---- extract_task_output ---------------------------------------------------

def test_extract_task_output_json_block():
    raw = 'Some text\n```json\n{"task_id": "t1", "status": "completed", "summary": "done"}\n```'
    out = extract_task_output(raw, "t1")
    assert out.status == TaskStatus.COMPLETED
    assert out.task_id == "t1"


def test_extract_task_output_no_json():
    raw = (
        "I've created the file src/app.py and modified src/main.py. "
        "Updated the config. Implemented the login feature. "
        "Wrote tests in tests/test_login.py. Successfully verified."
    )
    out = extract_task_output(raw, "t1")
    assert out.status == TaskStatus.COMPLETED


def test_extract_task_output_empty():
    out = extract_task_output("", "t1")
    assert out.status == TaskStatus.FAILED


# ---- Remediation -----------------------------------------------------------

def test_create_remediation_task():
    task = _make_task("task_01", goal="Build the auth module fully")
    failed_out = TaskOutput(
        task_id="task_01",
        status=TaskStatus.FAILED,
        summary="syntax error in auth.py",
    )
    rem = create_remediation_task(task, failed_out, task_counter=5)
    assert rem is not None
    assert rem.is_remediation is True
    assert rem.original_task_id == "task_01"
    assert "fix" in rem.id.lower()


def test_create_remediation_task_no_strategy():
    task = _make_task("task_02", goal="Build something unclear maybe")
    failed_out = TaskOutput(
        task_id="task_02",
        status=TaskStatus.FAILED,
        summary="unclear what to do, need clarification",
    )
    rem = create_remediation_task(task, failed_out, task_counter=6)
    assert rem is None


# ---- Role sets -------------------------------------------------------------

def test_writer_reader_roles():
    assert WRITER_ROLES & READER_ROLES == frozenset(), "Writer and reader roles must be disjoint"
    assert AgentRole.BACKEND_DEVELOPER in WRITER_ROLES
    assert AgentRole.FRONTEND_DEVELOPER in WRITER_ROLES
    assert AgentRole.REVIEWER in READER_ROLES
    assert AgentRole.TEST_ENGINEER in READER_ROLES
