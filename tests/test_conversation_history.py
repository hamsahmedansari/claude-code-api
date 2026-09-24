"""Tests for multi-turn prompt construction."""

from claude_code_api.api.chat import _extract_prompts
from claude_code_api.models.openai import ChatCompletionRequest


def _request(messages):
    return ChatCompletionRequest(
        model="claude-haiku-4-5-20251001", messages=messages
    )


def test_single_turn_prompt_is_unchanged():
    request = _request([{"role": "user", "content": "Hello"}])
    user_prompt, _ = _extract_prompts(request)
    assert user_prompt == "Hello"


def test_prior_turns_are_included_in_prompt():
    request = _request(
        [
            {"role": "user", "content": "My favourite colour is chartreuse."},
            {"role": "assistant", "content": "NOTED"},
            {"role": "user", "content": "What is my favourite colour?"},
        ]
    )
    user_prompt, _ = _extract_prompts(request)

    assert "chartreuse" in user_prompt
    assert "NOTED" in user_prompt
    assert user_prompt.rstrip().endswith("What is my favourite colour?")


def test_system_message_is_not_folded_into_history():
    request = _request(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
        ]
    )
    user_prompt, system_prompt = _extract_prompts(request)

    assert system_prompt == "You are terse."
    assert "You are terse." not in user_prompt
