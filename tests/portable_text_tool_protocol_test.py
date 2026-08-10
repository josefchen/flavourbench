from __future__ import annotations

import pytest

from flavourbench.direct_kimi_pair import _parser as kimi_parser
from flavourbench.execution_policy import (
    PORTABLE_TEXT_TOOL_PROTOCOL_V1,
    ExecutionPolicy,
)
from flavourbench.provider import (
    PORTABLE_EPICURE_TOOL_NAMES,
    PORTABLE_FINAL_CHOICE_INSTRUCTION,
    _parse_portable_tool_request,
    _portable_tool_instruction,
    system_prompt_text,
)
from flavourbench.service_cohere import (
    COHERE_PLUS_FINAL_FORMAT,
    COHERE_PLUS_PHASE_REASONING,
    COHERE_PLUS_SELECTION_FORMAT,
    _cohere_plus_phase_payload,
    _normalize_cohere_plus_phase,
)
from flavourbench.service_kimi import _anthropic_request_payload, _openai_response


def _catalog() -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "description": f"Execute {name}",
            "inputSchema": {
                "type": "object",
                "properties": {"ingredient": {"type": "string"}},
            },
        }
        for name in sorted(PORTABLE_EPICURE_TOOL_NAMES)
    ]


def test_portable_policy_requires_no_staged_planning() -> None:
    policy = ExecutionPolicy(
        matched_planning=False,
        evidence_protocol=PORTABLE_TEXT_TOOL_PROTOCOL_V1,
        epicure_on_tool_required=True,
        final_response_mode="plain_text",
    )
    policy.validate()
    assert policy.document()["schema_version"] == "flavourbench-real-execution-policy-v10"

    with pytest.raises(ValueError, match="prohibits staged planning"):
        ExecutionPolicy(
            matched_planning=True,
            evidence_protocol=PORTABLE_TEXT_TOOL_PROTOCOL_V1,
            epicure_on_tool_required=True,
        ).validate()


def test_direct_kimi_parser_allows_provider_default_reasoning() -> None:
    actions = {action.dest: action for action in kimi_parser(reasoning_required=False)._actions}
    assert actions["intermediate_reasoning_effort"].required is False
    assert actions["final_reasoning_effort"].required is False


def test_direct_kimi_uses_anthropic_messages_without_openai_only_fields() -> None:
    projected = _anthropic_request_payload(
        {
            "model": "k3",
            "messages": [
                {"role": "system", "content": "Follow the exact answer contract."},
                {"role": "user", "content": "Choose A, B, C, or D."},
            ],
            "max_tokens": 128,
            "temperature": 0,
            "top_p": 1,
            "seed": 20260810,
            "provider": {"only": ["kimi-code-direct"]},
        }
    )
    assert projected == {
        "model": "k3",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Choose A, B, C, or D."}],
        "system": "Follow the exact answer contract.",
        "temperature": 0,
    }


def test_direct_kimi_projects_anthropic_response_to_runner_shape() -> None:
    projected = _openai_response(
        {
            "type": "message",
            "id": "msg_test",
            "model": "k3",
            "content": [
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "FINAL_CHOICE: B"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 4},
        }
    )
    assert projected["model"] == "k3"
    assert projected["choices"][0]["finish_reason"] == "stop"
    assert projected["choices"][0]["message"]["content"] == "FINAL_CHOICE: B"
    assert projected["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "reasoning_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def test_cohere_plus_portable_phases_use_native_structured_outputs() -> None:
    contract = {
        "portable_tool_selection_format": COHERE_PLUS_SELECTION_FORMAT,
        "portable_final_format": COHERE_PLUS_FINAL_FORMAT,
        "portable_phase_reasoning": COHERE_PLUS_PHASE_REASONING,
    }
    base = {"model": "command-a-plus-05-2026", "messages": []}
    selection = _cohere_plus_phase_payload(base, phase="portable_tool_selection", contract=contract)
    final = _cohere_plus_phase_payload(base, phase="final", contract=contract)
    assert selection["response_format"]["json_schema"]["name"] == "epicure_tool_selection"
    assert final["response_format"]["json_schema"]["name"] == "flavourbench_choice"
    assert selection["reasoning"] == {"effort": "low", "exclude": True}
    normalized = _normalize_cohere_plus_phase(
        {
            "choices": [
                {
                    "message": {"content": '{"choice":"C"}'},
                    "finish_reason": "stop",
                }
            ]
        },
        phase="final",
    )
    assert normalized["choices"][0]["message"]["content"] == "FINAL_CHOICE: C"


def test_portable_tool_request_accepts_plain_or_fenced_exact_json() -> None:
    expected = ("pairing_score", {"a": "pear", "b": "vanilla"})
    assert (
        _parse_portable_tool_request(
            '{"name":"pairing_score","arguments":{"a":"pear","b":"vanilla"}}'
        )
        == expected
    )
    assert (
        _parse_portable_tool_request(
            '```json\n{"name":"pairing_score","arguments":{"a":"pear","b":"vanilla"}}\n```'
        )
        == expected
    )


@pytest.mark.parametrize(
    "value",
    [
        "I would call pairing_score.",
        '{"name":"unknown","arguments":{}}',
        '{"name":"neighbors","arguments":{},"extra":true}',
        '{"name":"neighbors","arguments":[]}',
    ],
)
def test_portable_tool_request_rejects_ambiguous_shapes(value: str) -> None:
    with pytest.raises(RuntimeError):
        _parse_portable_tool_request(value)


def test_portable_catalog_and_system_prompt_are_explicit() -> None:
    instruction = _portable_tool_instruction(_catalog())
    assert set(PORTABLE_EPICURE_TOOL_NAMES) == {
        "neighbors",
        "pairing_score",
        "compare_on_axis",
        "cultural_profile",
    }
    assert instruction.count('"name"') == 4
    assert "Return only one JSON object" in instruction
    assert "provider-neutral Epicure tool request" in system_prompt_text(
        "epicure_on",
        "plain_text",
        PORTABLE_TEXT_TOOL_PROTOCOL_V1,
    )
    assert "emit exactly one line" in system_prompt_text(
        "epicure_off",
        "plain_text",
        PORTABLE_TEXT_TOOL_PROTOCOL_V1,
    )
    assert PORTABLE_FINAL_CHOICE_INSTRUCTION == (
        "Answer now with no analysis or explanation. Return exactly one line in this form: "
        "FINAL_CHOICE: X, replacing X with A, B, C, or D."
    )
