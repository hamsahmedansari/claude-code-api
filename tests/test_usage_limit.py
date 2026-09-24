"""Tests for Claude subscription usage limit detection."""

import asyncio
from datetime import datetime, timezone

import pytest

from claude_code_api.core import claude_manager as cm


def test_detects_usage_limit_message():
    assert cm._is_usage_limit_error("Usage limit reached") is True


def test_detects_usage_limit_with_reset_suffix():
    error = "Claude AI usage limit reached|1774000000"
    assert cm._is_usage_limit_error(error) is True


def test_ignores_unrelated_errors():
    assert cm._is_usage_limit_error("invalid model") is False
    assert cm._is_usage_limit_error("Claude exited with code 1") is False
    assert cm._is_usage_limit_error("") is False


def test_extracts_reset_time_from_resets_at():
    reset = cm._extract_usage_limit_reset("Usage limit reached resets_at=1774000000")
    assert reset == datetime(2026, 3, 20, 9, 46, 40, tzinfo=timezone.utc)


def test_extracts_reset_time_from_pipe_suffix():
    reset = cm._extract_usage_limit_reset("Claude AI usage limit reached|1774000000")
    assert reset == datetime(2026, 3, 20, 9, 46, 40, tzinfo=timezone.utc)


def test_missing_reset_time_returns_none():
    assert cm._extract_usage_limit_reset("Usage limit reached") is None


@pytest.mark.asyncio
async def test_create_session_raises_usage_limit_error(monkeypatch, tmp_path):
    manager = cm.ClaudeManager()

    async def fake_start(self, prompt, model=None, system_prompt=None):
        self.last_error = (
            "Claude exited with code 1: Usage limit reached resets_at=1774000000"
        )
        return False

    monkeypatch.setattr(cm.ClaudeProcess, "start", fake_start)

    with pytest.raises(cm.ClaudeUsageLimitError) as exc_info:
        await manager.create_session(
            session_id="sess-limit",
            project_path=str(tmp_path),
            prompt="hello",
        )

    assert exc_info.value.reset_at == datetime(
        2026, 3, 20, 9, 46, 40, tzinfo=timezone.utc
    )


def test_out_of_range_reset_time_returns_none():
    error = "Usage limit reached resets_at=99999999999999999999"
    assert cm._extract_usage_limit_reset(error) is None


def test_reset_time_ignores_unrelated_pipe_numbers():
    error = (
        "Claude exited with code 1: trace|1234567890 | "
        "Claude AI usage limit reached|1774000000"
    )
    assert cm._extract_usage_limit_reset(error) == datetime(
        2026, 3, 20, 9, 46, 40, tzinfo=timezone.utc
    )


def test_reset_time_skips_unparsable_candidate():
    error = (
        "Claude exited with code 1: warn|1700000000123 | "
        "Claude AI usage limit reached|1774000000"
    )
    assert cm._extract_usage_limit_reset(error) == datetime(
        2026, 3, 20, 9, 46, 40, tzinfo=timezone.utc
    )


@pytest.mark.asyncio
async def test_result_error_on_stdout_is_captured(tmp_path):
    process = cm.ClaudeProcess(session_id="sess", project_path=str(tmp_path))
    process._record_output_error(
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "Usage limit reached",
        }
    )
    assert process._compose_process_error(1).endswith("Usage limit reached")


@pytest.mark.asyncio
async def test_successful_result_does_not_record_error(tmp_path):
    process = cm.ClaudeProcess(session_id="sess", project_path=str(tmp_path))
    process._record_output_error(
        {"type": "result", "subtype": "success", "is_error": False, "result": "hello"}
    )
    assert process._compose_process_error(1) == "Claude exited with code 1"


@pytest.mark.asyncio
async def test_usage_limit_on_stdout_skips_model_fallback(monkeypatch, tmp_path):
    manager = cm.ClaudeManager()
    attempts = []

    async def fake_start(self, prompt, model=None, system_prompt=None):
        attempts.append(model)
        self._record_output_error(
            {"type": "result", "is_error": True, "result": "Usage limit reached"}
        )
        self.last_error = self._compose_process_error(1)
        return False

    monkeypatch.setattr(cm.ClaudeProcess, "start", fake_start)

    with pytest.raises(cm.ClaudeUsageLimitError):
        await manager.create_session(
            session_id="sess-fallback",
            project_path=str(tmp_path),
            prompt="hello",
            model="claude-opus-4-6-20260205",
        )

    assert len(attempts) == 1


class _ExitedProcess:
    def __init__(self, returncode):
        self.returncode = returncode


@pytest.mark.asyncio
async def test_startup_fails_when_result_reports_error_with_zero_exit(tmp_path):
    process = cm.ClaudeProcess(session_id="sess", project_path=str(tmp_path))
    process.process = _ExitedProcess(0)
    process._record_output_error(
        {"type": "result", "is_error": True, "result": "Usage limit reached"}
    )

    assert await process._verify_startup() is False
    assert "Usage limit reached" in process.last_error


@pytest.mark.asyncio
async def test_startup_drains_pending_output_before_composing_error(tmp_path):
    process = cm.ClaudeProcess(session_id="sess", project_path=str(tmp_path))
    process.process = _ExitedProcess(1)

    async def late_reader():
        await asyncio.sleep(0.2)
        process._record_output_error(
            {"type": "result", "is_error": True, "result": "Usage limit reached"}
        )

    process._output_task = asyncio.create_task(late_reader())

    assert await process._verify_startup() is False
    assert "Usage limit reached" in process.last_error
