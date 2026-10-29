"""Tests for caller-supplied session ids.

A caller that owns its own conversation ids (for example a chat bot keyed on an
external thread id) should be able to send the same `session_id` on every
request without a pre-create round trip, and without breaking when the
gateway's database has been reset.
"""

import pytest
from sqlalchemy.exc import IntegrityError

from claude_code_api.api.chat import _resolve_session
from claude_code_api.models.openai import ChatCompletionRequest


class StubSessionManager:
    """Records calls so the tests can assert create-vs-reuse."""

    def __init__(self, known=()):
        self.known = set(known)
        self.created = []

    async def get_session(self, session_id):
        return {"session_id": session_id} if session_id in self.known else None

    async def create_session(
        self, project_id, model=None, system_prompt=None, session_id=None
    ):
        resolved = session_id or f"generated-{len(self.created) + 1}"
        self.created.append(
            {
                "project_id": project_id,
                "model": model,
                "system_prompt": system_prompt,
                "session_id": session_id,
            }
        )
        self.known.add(resolved)
        return resolved


def _request(session_id=None):
    return ChatCompletionRequest(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "Hello"}],
        session_id=session_id,
    )


@pytest.mark.asyncio
async def test_unknown_session_id_is_created_not_rejected():
    manager = StubSessionManager()
    resolved = await _resolve_session(
        session_manager=manager,
        request=_request("olx-thread-42"),
        project_id="proj",
        claude_model="claude-haiku-4-5-20251001",
        system_prompt="You are a seller.",
    )
    assert resolved == "olx-thread-42"
    assert len(manager.created) == 1
    assert manager.created[0]["session_id"] == "olx-thread-42"


@pytest.mark.asyncio
async def test_created_session_carries_the_model_and_system_prompt():
    manager = StubSessionManager()
    await _resolve_session(
        session_manager=manager,
        request=_request("olx-thread-7"),
        project_id="proj",
        claude_model="claude-haiku-4-5-20251001",
        system_prompt="You are terse.",
    )
    assert manager.created[0]["model"] == "claude-haiku-4-5-20251001"
    assert manager.created[0]["system_prompt"] == "You are terse."


@pytest.mark.asyncio
async def test_known_session_id_is_reused_without_creating():
    manager = StubSessionManager(known={"olx-thread-42"})
    resolved = await _resolve_session(
        session_manager=manager,
        request=_request("olx-thread-42"),
        project_id="proj",
        claude_model=None,
        system_prompt=None,
    )
    assert resolved == "olx-thread-42"
    assert manager.created == []


@pytest.mark.asyncio
async def test_same_id_twice_creates_once_then_reuses():
    """The whole point: send the same id on every turn of a conversation."""
    manager = StubSessionManager()
    first = await _resolve_session(
        session_manager=manager,
        request=_request("olx-thread-9"),
        project_id="proj",
        claude_model=None,
        system_prompt=None,
    )
    second = await _resolve_session(
        session_manager=manager,
        request=_request("olx-thread-9"),
        project_id="proj",
        claude_model=None,
        system_prompt=None,
    )
    assert first == second == "olx-thread-9"
    assert len(manager.created) == 1


@pytest.mark.asyncio
async def test_omitting_session_id_still_generates_one():
    """Unchanged behaviour for callers that do not manage ids (e.g. Lampaw)."""
    manager = StubSessionManager()
    resolved = await _resolve_session(
        session_manager=manager,
        request=_request(None),
        project_id="proj",
        claude_model=None,
        system_prompt=None,
    )
    assert resolved == "generated-1"
    assert manager.created[0]["session_id"] is None


class DeactivatedRowSessionManager:
    """A session whose row exists but is not active.

    This is the shape that broke production. `get_session` filters on
    `is_active`, so a session left inactive by an earlier failed request
    reads as unknown -- and the create that follows hits the primary key
    that is still sitting in the table.
    """

    def __init__(self):
        self.create_attempts = 0

    async def get_session(self, session_id):
        return None

    async def create_session(
        self, project_id, model=None, system_prompt=None, session_id=None
    ):
        self.create_attempts += 1
        raise IntegrityError(
            "INSERT INTO sessions", {}, Exception("UNIQUE constraint failed: sessions.id")
        )


@pytest.mark.asyncio
async def test_an_existing_but_inactive_session_is_reused_not_recreated():
    """Observed on x-server-01: every reply 500ed after a gateway restart.

    The bot keys sessions on the OLX thread id, so once a thread's row had
    been written, every later message on that thread hit this path. A 500
    here is one canned fallback per thread and then silence -- for every
    conversation the bot has ever had.
    """
    manager = DeactivatedRowSessionManager()

    resolved = await _resolve_session(
        manager, _request("olx-thread-123"), "proj", "claude-haiku-4-5-20251001", None
    )

    assert resolved == "olx-thread-123"
    assert manager.create_attempts == 1


@pytest.mark.asyncio
async def test_a_create_failure_that_is_not_a_duplicate_still_raises():
    """Only the duplicate is benign. A real database fault must not be
    swallowed into a session id that was never stored."""

    class BrokenSessionManager(DeactivatedRowSessionManager):
        async def create_session(self, *a, **kw):
            raise RuntimeError("disk is full")

    with pytest.raises(RuntimeError):
        await _resolve_session(
            BrokenSessionManager(), _request("olx-thread-9"), "proj", "m", None
        )
