"""Build and adjudicate a blinded, real-output expert calibration set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .current_frontier_task_quarantine import quarantine_binding, quarantine_task_ids

SCHEMA_VERSION = "flavourbench-expert-calibration-candidate-v11"
IDENTITY_SCHEMA_VERSION = "flavourbench-expert-calibration-identity-v11"
BALLOT_SCHEMA_VERSION = "flavourbench-expert-calibration-ballot-v1"
FROZEN_SCHEMA_VERSION = "flavourbench-expert-calibration-frozen-v1"
REVIEWER_PACK_SCHEMA_VERSION = "flavourbench-expert-calibration-reviewer-pack-v1"
SCORE_SCHEMA_VERSION = "flavourbench-expert-calibration-score-v1"
CALIBRATION_SET_ID = "affiliated-expert-calibration-v11"
# Preserve the v5 selection seed so the v6 quality correction replaces only
# newly quarantined tasks instead of resampling the entire 32-pair reserve.
SELECTION_SEED = "flavourbench-affiliated-expert-calibration-v5-20260731"
TASK_FAMILIES = ("substitution", "composition", "cookability", "evidence")
CHOICES = ("left", "right", "tie", "both_bad")
TASK_VALIDITY = ("valid", "minor_issue", "invalid")
ACCEPTED_FINAL_FINISH_REASONS = frozenset({"completed", "end_turn", "stop", "stop_sequence"})
ITEM_FLAGS = (
    "identity_leak",
    "response_truncated",
    "specialist_scope",
    "rights_or_privacy",
    "other",
)
TASK_SCOPE_QUARANTINE = frozenset(
    {
        "fb-s0-substitution-004",
        "fb-s0-substitution-008",
        "fb-s0-substitution-009",
        "fb-s0-substitution-010",
        "fb-s0-substitution-017",
        "fb-s0-substitution-019",
        "fb-s0-composition-013",
        "fb-s0-composition-021",
        "fb-s0-composition-022",
        "fb-s0-composition-027",
        "fb-s0-cookability-009",
        "fb-s0-cookability-013",
        "fb-s0-cookability-023",
        "fb-s0-cookability-026",
        "fb-s0-evidence-013",
        "fb-s0-evidence-014",
        "fb-s0-evidence-028",
    }
)
TASK_SCOPE_REVIEW_SHA256 = "87755da495db285c3388c3c91b62fc53135d7c79e5584c18a266658a036d21f0"
TASK_QUALITY_QUARANTINE = frozenset(
    {
        "fb-s0-composition-015",
        "fb-s0-composition-026",
        "fb-s0-cookability-008",
        "fb-s0-cookability-019",
        "fb-s0-evidence-010",
    }
)
TASK_QUALITY_REVIEW_SHA256 = "ec1d024ac6f27b59177fd7ecb84e93b305acdc4604a9c89e84d6b9b5aeab066b"
CURRENT_FRONTIER_TASK_QUARANTINE = quarantine_task_ids()
CURRENT_FRONTIER_TASK_QUARANTINE_BINDING = quarantine_binding()
MANUAL_RESPONSE_QUARANTINE = {
    "07ef2e5ede59c50aced61d9c73bee677d9da646d0f6b7bf8ad649f9f4c8ea6fb": (
        "internal_meta_and_tool_disclosure"
    ),
    "1fb8fe72d0418ba1ea9ff94b157846878927306557592157e8327aaae5d60532": (
        "tool_disclosure_and_dangling_reference"
    ),
    "5668282190aec48ee08b68682023aa3e632e3c4762d0d7cce069cb549296a3d0": (
        "hidden_model_and_tool_disclosure"
    ),
    "b0bfcb0c435ba10419d57049f4afa20522d627dad4be85523cc792079554ca81": ("query_disclosure"),
    "24d5584ab7d9e5beea955e270b919303106c117c127cc355752a9f13ce00c370": ("tool_map_disclosure"),
    "6da5eae5c875b7344a8ed04d06125e43fc94a1f560c14eda114d18533982defd": ("tool_disclosure"),
    "35d6eb901f118e39f66635d2065dcfa9b4cda1d0c1ae8679f772c2e4630d6f32": ("semantic_incompletion"),
    "cfa7f5b06dd12f6a5c1d486e7c6639b8b8cd9a361a1cef2aca8f74e7c090e1b1": (
        "reasoning_and_drafting_preamble_disclosure"
    ),
    "e1b50c015e2725c86a0aad2f70732dee1c701f20e24a3ced6b518d60e841bce6": (
        "reasoning_and_drafting_preamble_disclosure"
    ),
    "33ac23abb43820a494e9927094cdb0a02d45710ad28f549a571c803e53501560": ("hidden_model_disclosure"),
    "ac53240b6cc835f41ae7e8e527877292e643239debc7c972e1ca66e95115e533": (
        "tool_and_drafting_preamble_disclosure"
    ),
    "827828a735d3c4ea7348192113784b557fdce79f81dda790e15511331652fc38": ("semantic_incompletion"),
}
RESPONSE_CONTENT_REVIEW_SHA256 = "48aa11295cf67f41f5ba23e3d581c3033dd84655f7cc4b90d658dd19c128486e"

# These strings disclose a model, provider, or the experimental condition. The
# source answer is never edited: a leaking pair is excluded in full.
IDENTITY_LEAK_PATTERN = re.compile(
    r"\b(?:"
    r"epicure|openrouter|bedrock|anthropic|claude|openai|"
    r"gpt(?:[- ]?5(?:\.\d+)?)?|gemini|qwen|mistral|devstral|"
    r"minimax|nova 2|fable 5"
    r")\b",
    re.IGNORECASE,
)
BLINDING_LEAK_PATTERN = re.compile(
    r"(?:<\s*/?\s*(?:reasoning|analysis|thinking|scratchpad)\s*>|\b(?:"
    r"pairing[- ](?:score[- ]?)?(?:graph|tool|map|query)|"
    r"neighbou?rs? tool|flavou?r (?:model|tools?)|"
    r"(?:the |these )?(?:pairing|flavou?r) tools?|tool data|these tools model|"
    r"using (?:the )?tool (?:data|output|results?)|"
    r"tools? (?:doesn't|does not|don't|do not)\b|"
    r"the (?:flavou?r )?model (?:backs(?: up)?|confirms|shows|rates)"
    r")\b|(?:^|\n)\s*(?:"
    r"now (?:we need to )?answer|"
    r"(?:now|let['’]?s)\s+craft (?:the )?(?:final )?answer|"
    r"we need to craft (?:the )?(?:final )?answer"
    r"))",
    re.IGNORECASE,
)
BLINDING_LEAK_PATTERN_SHA256 = hashlib.sha256(BLINDING_LEAK_PATTERN.pattern.encode()).hexdigest()


class ExpertCalibrationError(RuntimeError):
    """Calibration evidence cannot be built or frozen safely."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpertCalibrationError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ExpertCalibrationError(f"expected a JSON object: {path}")
    return value


def _artifact_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "artifact_sha256"}


def _verify_artifact_document(path: Path, record: Mapping[str, Any]) -> str:
    stored = record.get("artifact_sha256")
    if not isinstance(stored, str) or not re.fullmatch(r"[0-9a-f]{64}", stored):
        raise ExpertCalibrationError(f"artifact has no valid digest: {path}")
    if sha256_json(_artifact_payload(record)) != stored:
        raise ExpertCalibrationError(f"artifact digest mismatch: {path}")
    return stored


def _review_artifact(
    filename: str,
    *,
    expected_sha256: str,
    expected_schema: str,
) -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "expert-calibration"
        / "governance"
        / filename
    )
    record = _load_object(path)
    if (
        _verify_artifact_document(path, record) != expected_sha256
        or record.get("schema_version") != expected_schema
    ):
        raise ExpertCalibrationError(f"governance review contract mismatch: {path}")
    return record


def _assert_governance_review_contracts(items: Sequence[Mapping[str, Any]]) -> None:
    scope = _review_artifact(
        "specialist-scope-review-v1.json",
        expected_sha256=TASK_SCOPE_REVIEW_SHA256,
        expected_schema="flavourbench-specialist-scope-review-v1",
    )
    quality = _review_artifact(
        "task-quality-review-v1.json",
        expected_sha256=TASK_QUALITY_REVIEW_SHA256,
        expected_schema="flavourbench-task-quality-review-v1",
    )
    response = _review_artifact(
        "response-content-review-v1.json",
        expected_sha256=RESPONSE_CONTENT_REVIEW_SHA256,
        expected_schema="flavourbench-response-content-review-v1",
    )
    task_coordinates = sorted(
        (
            {
                "family": str(item["family"]),
                "task_id": str(item["task_id"]),
                "task_sha256": str(item["task_sha256"]),
            }
            for item in items
        ),
        key=lambda item: item["task_id"],
    )
    answer_coordinates = sorted(
        (
            {
                "task_id": str(item["task_id"]),
                "left_answer_sha256": str(_mapping(item["left"], label="left")["answer_sha256"]),
                "right_answer_sha256": str(_mapping(item["right"], label="right")["answer_sha256"]),
            }
            for item in items
        ),
        key=lambda item: item["task_id"],
    )
    answer_set = sorted(
        answer_sha256
        for item in answer_coordinates
        for answer_sha256 in (
            item["left_answer_sha256"],
            item["right_answer_sha256"],
        )
    )
    selected_task_ids = {item["task_id"] for item in task_coordinates}
    if selected_task_ids.intersection(CURRENT_FRONTIER_TASK_QUARANTINE):
        raise ExpertCalibrationError(
            "candidate coordinates include a current frontier quarantined task"
        )
    scope_quarantine = {
        str(item.get("task_id"))
        for item in scope.get("quarantine_decisions", [])
        if isinstance(item, Mapping)
    }
    quality_quarantine = {
        str(item.get("task_id"))
        for item in quality.get("quarantine_decisions", [])
        if isinstance(item, Mapping)
    }
    known_bad = {
        str(item.get("answer_sha256")): str(item.get("reason_code"))
        for item in response.get("known_bad_answer_decisions", [])
        if isinstance(item, Mapping)
    }
    method = response.get("review_method")
    if not isinstance(method, Mapping):
        raise ExpertCalibrationError("response-content review has no method contract")
    if (
        scope.get("selected_task_coordinate_sha256") != sha256_json(task_coordinates)
        or quality.get("selected_task_coordinate_sha256") != sha256_json(task_coordinates)
        or set(scope.get("selected_task_ids", [])) != selected_task_ids
        or scope_quarantine != set(TASK_SCOPE_QUARANTINE)
        or quality_quarantine != set(TASK_QUALITY_QUARANTINE)
        or response.get("selected_answer_coordinate_sha256") != sha256_json(answer_coordinates)
        or response.get("selected_answer_set_sha256") != sha256_json(answer_set)
        or response.get("selected_answers") != len(answer_set)
        or known_bad != MANUAL_RESPONSE_QUARANTINE
        or method.get("blinding_leak_pattern_sha256") != BLINDING_LEAK_PATTERN_SHA256
        or response.get("selected_responses_passed_content_screen") is not True
    ):
        raise ExpertCalibrationError("candidate coordinates differ from governance reviews")


def _assert_current_frontier_quarantine_binding(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every newly frozen gold set to the current task-admission overlay.

    Historical candidate packs predate this overlay. They remain usable only when their
    immutable item coordinates are disjoint from the held task set; the new frozen output
    records the overlay digest. Newly built packs also carry the binding in their selection
    policy and must match it exactly.
    """

    items = candidate.get("items")
    if not isinstance(items, list) or any(not isinstance(item, Mapping) for item in items):
        raise ExpertCalibrationError("candidate pack items are malformed")
    task_ids = {str(item.get("task_id") or "") for item in items}
    held = task_ids.intersection(CURRENT_FRONTIER_TASK_QUARANTINE)
    if held:
        raise ExpertCalibrationError(
            "candidate pack includes current frontier quarantined tasks: "
            + ", ".join(sorted(held))
        )
    policy = candidate.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise ExpertCalibrationError("candidate pack selection policy is malformed")
    declared = policy.get("current_frontier_task_quarantine_artifact_sha256")
    if declared is not None and (
        policy.get("current_frontier_task_quarantine_required") is not True
        or declared != CURRENT_FRONTIER_TASK_QUARANTINE_BINDING["artifact_sha256"]
        or policy.get("current_frontier_task_quarantine_task_set_sha256")
        != CURRENT_FRONTIER_TASK_QUARANTINE_BINDING["task_set_sha256"]
        or set(policy.get("current_frontier_task_quarantine_task_ids") or [])
        != set(CURRENT_FRONTIER_TASK_QUARANTINE)
    ):
        raise ExpertCalibrationError("candidate pack task-quarantine binding drifted")
    return dict(CURRENT_FRONTIER_TASK_QUARANTINE_BINDING)


def _verify_arm(path: Path, arm: Mapping[str, Any]) -> str:
    stored = _verify_artifact_document(path, arm)
    if not path.name.endswith(f"-{stored}.json"):
        raise ExpertCalibrationError(f"arm filename is not content-addressed: {path}")
    return stored


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExpertCalibrationError(f"arm has no {label} object")
    return value


def _answer(arm: Mapping[str, Any]) -> str:
    result = _mapping(arm.get("result"), label="result")
    answer = result.get("answer_markdown")
    if not isinstance(answer, str) or not answer.strip():
        raise ExpertCalibrationError("successful arm has no answer text")
    return answer.strip()


def _successful_tool_calls(arm: Mapping[str, Any]) -> int:
    result = _mapping(arm.get("result"), label="result")
    trace = result.get("tool_trace")
    if not isinstance(trace, list):
        raise ExpertCalibrationError("successful arm has no MCP trace list")
    declared = result.get("real_epicure_calls")
    if not isinstance(declared, int) or declared != len(trace):
        raise ExpertCalibrationError("Epicure call count does not match the MCP trace")
    successful = 0
    for call in trace:
        if not isinstance(call, Mapping):
            raise ExpertCalibrationError("MCP trace contains a non-object call")
        required = {
            "name",
            "arguments",
            "arguments_sha256",
            "is_error",
            "latency_ms",
            "model_visible_result_sha256",
            "result_sha256",
            "round_index",
        }
        if not required.issubset(call):
            raise ExpertCalibrationError("MCP trace is missing immutable call evidence")
        if call.get("is_error") is False:
            successful += 1
    return successful


def _real_provider_evidence(arm: Mapping[str, Any]) -> bool:
    result = _mapping(arm.get("result"), label="result")
    provider_calls = result.get("provider_calls")
    request_ids = result.get("request_id_sha256s")
    payload_hashes = result.get("request_payload_sha256s")
    returned_model_ids = result.get("returned_model_ids")
    actual_provider = result.get("actual_provider_name")
    usage = result.get("usage")
    request_identity_present = bool(
        isinstance(request_ids, list)
        and len(request_ids) >= 1
        and all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in request_ids
        )
    )
    returned_identity_present = bool(
        isinstance(returned_model_ids, list)
        and len(returned_model_ids) >= 1
        and all(isinstance(value, str) and value for value in returned_model_ids)
        and isinstance(actual_provider, str)
        and actual_provider
    )
    return bool(
        isinstance(provider_calls, int)
        and provider_calls >= 1
        and (request_identity_present or returned_identity_present)
        and isinstance(payload_hashes, list)
        and len(payload_hashes) >= 1
        and all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in payload_hashes
        )
        and isinstance(usage, Mapping)
        and isinstance(usage.get("total_tokens"), int)
        and usage["total_tokens"] > 0
    )


def _pair_key(arm: Mapping[str, Any]) -> tuple[str, str, str]:
    task = _mapping(arm.get("task"), label="task")
    model = _mapping(arm.get("model"), label="model")
    family = task.get("family")
    task_id = task.get("task_id")
    model_id = model.get("season_model_id")
    if family not in TASK_FAMILIES:
        raise ExpertCalibrationError(f"unsupported task family: {family!r}")
    if not all(isinstance(value, str) and value for value in (task_id, model_id)):
        raise ExpertCalibrationError("arm is missing task or model identity")
    return str(family), str(task_id), str(model_id)


def _eligible_arm(arm: Mapping[str, Any]) -> bool:
    result = arm.get("result")
    finish_reason = (
        str(result.get("finish_reason") or "").strip().lower()
        if isinstance(result, Mapping)
        else ""
    )
    return bool(
        arm.get("phase") == "scored"
        and arm.get("status") == "success"
        and arm.get("synthetic") is False
        and arm.get("rank_eligible") is True
        and arm.get("condition") in {"epicure_on", "epicure_off"}
        and finish_reason in ACCEPTED_FINAL_FINISH_REASONS
    )


def _pair_sort_key(pair: Mapping[str, Any], family: str) -> str:
    return sha256_text(
        "|".join(
            (
                SELECTION_SEED,
                family,
                str(pair["task_id"]),
                str(pair["model_id"]),
                str(pair["on"]["artifact_sha256"]),
                str(pair["off"]["artifact_sha256"]),
            )
        )
    )


def _select_family_pairs(
    candidates: Sequence[dict[str, Any]],
    *,
    family: str,
    count: int,
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_model[str(candidate["model_id"])].append(candidate)
    for rows in by_model.values():
        rows.sort(key=lambda row: _pair_sort_key(row, family))

    model_order = sorted(
        by_model,
        key=lambda model_id: sha256_text(f"{SELECTION_SEED}|{family}|{model_id}"),
    )

    # First solve a deterministic maximum bipartite matching so a greedy early
    # task choice cannot displace a model that has only one usable task.
    matched_by_task: dict[str, dict[str, Any]] = {}

    def augment(model_id: str, seen_tasks: set[str]) -> bool:
        for row in by_model[model_id]:
            task_id = str(row["task_id"])
            if task_id in seen_tasks:
                continue
            seen_tasks.add(task_id)
            previous = matched_by_task.get(task_id)
            if previous is None or augment(str(previous["model_id"]), seen_tasks):
                matched_by_task[task_id] = row
                return True
        return False

    for model_id in model_order:
        augment(model_id, set())
    maximum_model_matching = sorted(
        matched_by_task.values(),
        key=lambda row: _pair_sort_key(row, family),
    )
    selected = maximum_model_matching[:count]
    selected_tasks = {str(row["task_id"]) for row in selected}
    model_counts = Counter(str(row["model_id"]) for row in selected)

    # If the family has fewer distinct tool-using models than reserve slots,
    # fill the balance with new tasks and the least-used model first.
    while len(selected) < count:
        available = [
            row
            for row in candidates
            if str(row["task_id"]) not in selected_tasks and row not in selected
        ]
        if not available:
            break
        row = min(
            available,
            key=lambda value: (
                model_counts[str(value["model_id"])],
                _pair_sort_key(value, family),
            ),
        )
        selected.append(row)
        selected_tasks.add(str(row["task_id"]))
        model_counts[str(row["model_id"])] += 1
    if len(selected) != count:
        raise ExpertCalibrationError(
            f"{family} has only {len(selected)} selectable real tool-use pairs; {count} required"
        )
    return selected


def _orientation(pair: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    orientation_hash = sha256_text(
        "|".join(
            (
                SELECTION_SEED,
                str(pair["task_id"]),
                str(pair["model_id"]),
                "orientation",
            )
        )
    )
    if int(orientation_hash[0], 16) % 2 == 0:
        return pair["on"], pair["off"]
    return pair["off"], pair["on"]


def _item_id(pair: Mapping[str, Any]) -> str:
    digest = sha256_text(
        "|".join(
            (
                SELECTION_SEED,
                str(pair["family"]),
                str(pair["task_id"]),
                str(pair["model_id"]),
                str(pair["on"]["artifact_sha256"]),
                str(pair["off"]["artifact_sha256"]),
            )
        )
    )
    return f"fb-cal-{digest[:20]}"


def build_candidate_payload(
    arms_dir: Path,
    *,
    candidates_per_family: int = 8,
    frozen_target_per_family: int = 5,
    require_governance_reviews: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if candidates_per_family < frozen_target_per_family:
        raise ExpertCalibrationError("candidate reserve cannot be smaller than the frozen target")
    arm_paths = sorted(arms_dir.glob("arm-*.json"))
    if not arm_paths:
        raise ExpertCalibrationError(f"no arm artifacts found in {arms_dir}")

    all_arm_hashes: list[dict[str, str]] = []
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    source_counts: Counter[str] = Counter()
    for path in arm_paths:
        arm = _load_object(path)
        artifact_sha256 = _verify_arm(path, arm)
        all_arm_hashes.append({"filename": path.name, "artifact_sha256": artifact_sha256})
        source_counts["verified_arm_artifacts"] += 1
        if arm.get("synthetic") is False:
            source_counts["non_synthetic_arms"] += 1
        if arm.get("status") == "success":
            source_counts["successful_arms"] += 1
            result = arm.get("result")
            finish_reason = (
                str(result.get("finish_reason") or "").strip().lower()
                if isinstance(result, Mapping)
                else ""
            )
            if finish_reason in ACCEPTED_FINAL_FINISH_REASONS:
                source_counts["normal_final_completion_arms"] += 1
            else:
                source_counts["excluded_non_normal_final_completion_arms"] += 1
        if not _eligible_arm(arm):
            continue
        if not _real_provider_evidence(arm):
            continue
        key = _pair_key(arm)
        condition = str(arm["condition"])
        if condition in grouped[key]:
            raise ExpertCalibrationError(f"duplicate eligible coordinate: {key!r} {condition}")
        grouped[key][condition] = arm

    candidates: dict[str, list[dict[str, Any]]] = {family: [] for family in TASK_FAMILIES}
    excluded_identity_leaks = 0
    excluded_without_successful_tool_call = 0
    excluded_scope_quarantine = 0
    excluded_quality_quarantine = 0
    excluded_current_frontier_quarantine = 0
    excluded_manual_response_content = 0
    excluded_blinding_leaks = 0
    for (family, task_id, model_id), conditions in grouped.items():
        if task_id in CURRENT_FRONTIER_TASK_QUARANTINE:
            excluded_current_frontier_quarantine += 1
            continue
        if task_id in TASK_SCOPE_QUARANTINE:
            excluded_scope_quarantine += 1
            continue
        if task_id in TASK_QUALITY_QUARANTINE:
            excluded_quality_quarantine += 1
            continue
        if set(conditions) != {"epicure_on", "epicure_off"}:
            continue
        on = conditions["epicure_on"]
        off = conditions["epicure_off"]
        if _successful_tool_calls(on) < 1:
            excluded_without_successful_tool_call += 1
            continue
        if _successful_tool_calls(off) != 0:
            raise ExpertCalibrationError("Epicure-off arm contains a successful MCP call")
        on_answer = _answer(on)
        off_answer = _answer(off)
        if on_answer == off_answer:
            continue
        if {
            sha256_text(on_answer),
            sha256_text(off_answer),
        }.intersection(MANUAL_RESPONSE_QUARANTINE):
            excluded_manual_response_content += 1
            continue
        if IDENTITY_LEAK_PATTERN.search(on_answer) or IDENTITY_LEAK_PATTERN.search(off_answer):
            excluded_identity_leaks += 1
            continue
        if BLINDING_LEAK_PATTERN.search(on_answer) or BLINDING_LEAK_PATTERN.search(off_answer):
            excluded_blinding_leaks += 1
            continue
        on_task = _mapping(on.get("task"), label="task")
        off_task = _mapping(off.get("task"), label="task")
        if (
            on_task.get("task_sha256") != off_task.get("task_sha256")
            or on_task.get("prompt_sha256") != off_task.get("prompt_sha256")
            or on_task.get("prompt") != off_task.get("prompt")
        ):
            raise ExpertCalibrationError("paired arms do not share an immutable task")
        candidates[family].append(
            {
                "family": family,
                "task_id": task_id,
                "model_id": model_id,
                "on": on,
                "off": off,
            }
        )

    selected: list[dict[str, Any]] = []
    for family in TASK_FAMILIES:
        selected.extend(
            _select_family_pairs(
                candidates[family],
                family=family,
                count=candidates_per_family,
            )
        )

    public_items: list[dict[str, Any]] = []
    identity_items: list[dict[str, Any]] = []
    selected_provider_calls = 0
    selected_epicure_calls = 0
    selected_successful_epicure_calls = 0
    selected_model_ids: set[str] = set()
    selected_task_ids: set[str] = set()
    for pair in selected:
        left, right = _orientation(pair)
        item_id = _item_id(pair)
        task = _mapping(left.get("task"), label="task")
        left_answer = _answer(left)
        right_answer = _answer(right)
        public_items.append(
            {
                "calibration_item_id": item_id,
                "family": pair["family"],
                "task_id": pair["task_id"],
                "task_sha256": task["task_sha256"],
                "prompt_sha256": task["prompt_sha256"],
                "prompt": task["prompt"],
                "left": {
                    "answer_markdown": left_answer,
                    "answer_sha256": sha256_text(left_answer),
                },
                "right": {
                    "answer_markdown": right_answer,
                    "answer_sha256": sha256_text(right_answer),
                },
            }
        )
        identity_items.append(
            {
                "calibration_item_id": item_id,
                "family": pair["family"],
                "task_id": pair["task_id"],
                "model": {
                    "season_model_id": pair["model_id"],
                    "canonical_model_id": left["model"]["canonical_model_id"],
                    "display_name": left["model"]["display_name"],
                    "provider": left["model"]["provider"],
                    "requested_endpoint_id": left["model"]["requested_endpoint_id"],
                },
                "left": {
                    "arm_id": left["arm_id"],
                    "artifact_sha256": left["artifact_sha256"],
                    "condition": left["condition"],
                    "answer_sha256": sha256_text(left_answer),
                },
                "right": {
                    "arm_id": right["arm_id"],
                    "artifact_sha256": right["artifact_sha256"],
                    "condition": right["condition"],
                    "answer_sha256": sha256_text(right_answer),
                },
            }
        )
        selected_model_ids.add(str(pair["model_id"]))
        selected_task_ids.add(str(pair["task_id"]))
        for arm in (left, right):
            result = _mapping(arm.get("result"), label="result")
            selected_provider_calls += int(result["provider_calls"])
            selected_epicure_calls += int(result["real_epicure_calls"])
            selected_successful_epicure_calls += _successful_tool_calls(arm)

    public_items.sort(
        key=lambda item: sha256_text(f"{SELECTION_SEED}|item-order|{item['calibration_item_id']}")
    )
    identity_items.sort(key=lambda item: str(item["calibration_item_id"]))
    if require_governance_reviews:
        _assert_governance_review_contracts(public_items)
    identity_core = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "calibration_set_id": CALIBRATION_SET_ID,
        "selection_seed_sha256": sha256_text(SELECTION_SEED),
        "items": identity_items,
    }
    identity_commitment = sha256_json(identity_core)
    source_inventory_sha256 = sha256_json(all_arm_hashes)
    family_counts = Counter(str(item["family"]) for item in public_items)
    family_model_counts = {
        family: len({str(pair["model_id"]) for pair in selected if pair["family"] == family})
        for family in TASK_FAMILIES
    }
    family_task_counts = {
        family: len({str(pair["task_id"]) for pair in selected if pair["family"] == family})
        for family in TASK_FAMILIES
    }
    candidate_payload = {
        "schema_version": SCHEMA_VERSION,
        "calibration_set_id": CALIBRATION_SET_ID,
        "status": "candidate_pending_independent_gold_adjudication",
        "created_from": {
            "source_class": "paid_real_legacy_pilot_quarantined_from_season1",
            "source_arm_inventory_sha256": source_inventory_sha256,
            "source_arm_files": len(all_arm_hashes),
            "verified_arm_artifacts": source_counts["verified_arm_artifacts"],
            "non_synthetic_arms": source_counts["non_synthetic_arms"],
            "successful_arms": source_counts["successful_arms"],
            "normal_final_completion_arms": source_counts["normal_final_completion_arms"],
            "excluded_non_normal_final_completion_arms": source_counts[
                "excluded_non_normal_final_completion_arms"
            ],
        },
        "selection_policy": {
            "seed_sha256": sha256_text(SELECTION_SEED),
            "paired_same_model_same_task": True,
            "conditions": ["epicure_on", "epicure_off"],
            "real_provider_evidence_required": True,
            "successful_real_epicure_call_required_in_on_arm": True,
            "complete_mcp_trace_required": True,
            "normal_final_completion_required": True,
            "accepted_final_finish_reasons": sorted(ACCEPTED_FINAL_FINISH_REASONS),
            "raw_answers_edited": False,
            "explicit_identity_leaks_excluded": True,
            "unique_tasks_within_family": True,
            "model_diversity_maximized_within_family": True,
            "left_right_deterministically_randomized": True,
            "excluded_identity_leak_pairs": excluded_identity_leaks,
            "blinding_leak_pattern_sha256": BLINDING_LEAK_PATTERN_SHA256,
            "excluded_blinding_leak_pairs": excluded_blinding_leaks,
            "manual_response_content_quarantine_required": True,
            "manual_response_quarantine_answer_sha256s": sorted(MANUAL_RESPONSE_QUARANTINE),
            "response_content_review_sha256": RESPONSE_CONTENT_REVIEW_SHA256,
            "excluded_manual_response_content_pairs": excluded_manual_response_content,
            "excluded_pairs_without_successful_epicure_call": (
                excluded_without_successful_tool_call
            ),
            "specialist_scope_quarantine_required": True,
            "specialist_scope_quarantine_task_ids": sorted(TASK_SCOPE_QUARANTINE),
            "specialist_scope_review_sha256": TASK_SCOPE_REVIEW_SHA256,
            "excluded_specialist_scope_pairs": excluded_scope_quarantine,
            "task_quality_quarantine_required": True,
            "task_quality_quarantine_task_ids": sorted(TASK_QUALITY_QUARANTINE),
            "task_quality_review_sha256": TASK_QUALITY_REVIEW_SHA256,
            "excluded_task_quality_pairs": excluded_quality_quarantine,
            "current_frontier_task_quarantine_required": True,
            "current_frontier_task_quarantine_artifact_sha256": (
                CURRENT_FRONTIER_TASK_QUARANTINE_BINDING["artifact_sha256"]
            ),
            "current_frontier_task_quarantine_task_set_sha256": (
                CURRENT_FRONTIER_TASK_QUARANTINE_BINDING["task_set_sha256"]
            ),
            "current_frontier_task_quarantine_task_ids": sorted(
                CURRENT_FRONTIER_TASK_QUARANTINE
            ),
            "excluded_current_frontier_task_pairs": (
                excluded_current_frontier_quarantine
            ),
            "selection_seed_continued_from_candidate_v5": True,
        },
        "target": {
            "candidate_pairs": candidates_per_family * len(TASK_FAMILIES),
            "candidate_pairs_per_family": candidates_per_family,
            "frozen_pairs": frozen_target_per_family * len(TASK_FAMILIES),
            "frozen_pairs_per_family": frozen_target_per_family,
            "minimum_independent_gold_adjudicators": 2,
        },
        "observed": {
            "candidate_pairs": len(public_items),
            "candidate_pairs_by_family": dict(sorted(family_counts.items())),
            "unique_tasks": len(selected_task_ids),
            "unique_tasks_by_family": family_task_counts,
            "unique_models": len(selected_model_ids),
            "unique_models_by_family": family_model_counts,
            "source_arms": len(public_items) * 2,
            "real_provider_calls": selected_provider_calls,
            "real_epicure_calls": selected_epicure_calls,
            "successful_real_epicure_calls": selected_successful_epicure_calls,
            "synthetic_arms": 0,
            "quality_judgments": 0,
        },
        "blinding": {
            "model_provider_and_condition_removed": True,
            "identity_commitment_sha256": identity_commitment,
            "condition_may_be_inferable_from_evidence_language": True,
            "identity_lookup_prohibited_during_adjudication": True,
        },
        "use_policy": {
            "calibration_only": True,
            "rank_eligible": False,
            "official_season_items": False,
            "benchmark_result_use": "prohibited",
            "gold_labels": "absent_pending_two_independent_human_ballots",
        },
        "items": public_items,
    }
    return candidate_payload, identity_core


def _artifact_document(payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    digest = sha256_json(payload)
    return {**payload, "artifact_sha256": digest}, digest


def _atomic_write_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    temporary.replace(path)


def ballot_template(candidate: Mapping[str, Any], candidate_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": BALLOT_SCHEMA_VERSION,
        "candidate_pack_sha256": candidate_sha256,
        "role": "",
        "reviewer": {
            "reviewer_code": "",
            "affiliation": "",
        },
        "attestations": {
            "worked_independently": False,
            "no_model_or_condition_identity_lookup": False,
            "reviewed_complete_unedited_answers": False,
            "independent_of_epicure_and_model_providers": False,
            "product_affiliation_disclosed": False,
        },
        "items": [
            {
                "calibration_item_id": item["calibration_item_id"],
                "task_validity": "",
                "choice": "",
                "confidence": None,
                "rationale": "",
                "flags": [],
            }
            for item in candidate["items"]
        ],
    }


def _render_html(
    candidate: Mapping[str, Any],
    candidate_sha256: str,
    *,
    title: str,
    subtitle: str,
    allowed_roles: Sequence[str],
) -> str:
    role_labels = {
        "independent_gold_adjudicator": "Independent gold adjudicator",
        "affiliated_reviewer_calibration": "Affiliated reviewer calibration",
    }
    invalid_roles = [role for role in allowed_roles if role not in role_labels]
    if invalid_roles or not allowed_roles:
        raise ExpertCalibrationError("workspace has unsupported calibration roles")
    role_options = "\n".join(
        f'<option value="{role}">{role_labels[role]}</option>' for role in allowed_roles
    )
    embedded = json.dumps(
        {
            "candidatePackSha256": candidate_sha256,
            "calibrationSetId": candidate["calibration_set_id"],
            "items": candidate["items"],
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light; --ink:#17202a; --muted:#5e6872; --line:#cbd2d9;
  --paper:#fff; --wash:#f4f6f7; --accent:#173f5f; --warn:#8a4b08; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--wash); color:var(--ink);
  font:15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }}
header {{ background:var(--paper); border-bottom:1px solid var(--line); }}
.masthead, main {{ width:min(1440px, calc(100% - 40px)); margin:0 auto; }}
.masthead {{ padding:28px 0 24px; display:grid; grid-template-columns:1fr auto; gap:24px; }}
h1, h2, h3 {{ font-family:Georgia, "Times New Roman", serif; font-weight:600; }}
h1 {{ margin:0 0 5px; font-size:30px; }}
h2 {{ margin:0; font-size:22px; }}
h3 {{ margin:0 0 10px; font-size:17px; }}
p {{ margin:0 0 12px; }}
.meta {{ color:var(--muted); font-size:13px; }}
.hash {{ font:12px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; }}
.progress {{ text-align:right; align-self:center; }}
.progress strong {{ display:block; font:600 24px Georgia, serif; color:var(--accent); }}
main {{ padding:24px 0 80px; }}
.instructions, .identity, article {{ background:var(--paper); border:1px solid var(--line); }}
.instructions, .identity {{ padding:20px; margin-bottom:18px; }}
.instructions ol {{ margin:8px 0 0; padding-left:22px; }}
.identity-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; }}
label {{ display:block; font-weight:600; }}
input[type=text], select, textarea {{ width:100%; border:1px solid #9da7b1; border-radius:2px;
  background:#fff; color:var(--ink); font:inherit; padding:9px 10px; }}
textarea {{ min-height:88px; resize:vertical; }}
.attestations {{ display:grid; grid-template-columns:1fr 1fr; gap:8px 20px; margin-top:16px; }}
.attestations label {{ font-weight:400; }}
article {{ margin:18px 0; }}
.item-head {{ padding:14px 18px; border-bottom:1px solid var(--line);
  display:flex; justify-content:space-between; gap:20px; align-items:baseline; }}
.family {{ text-transform:uppercase; letter-spacing:.08em; color:var(--accent); font-size:11px; font-weight:700; }}
.prompt {{ padding:18px; border-bottom:1px solid var(--line); }}
.prompt pre, .answer pre {{ white-space:pre-wrap; overflow-wrap:anywhere; margin:0;
  font:14px/1.55 ui-sans-serif, system-ui, sans-serif; }}
.answers {{ display:grid; grid-template-columns:1fr 1fr; }}
.answer {{ padding:18px; min-width:0; }}
.answer + .answer {{ border-left:1px solid var(--line); }}
.answer h3 {{ padding-bottom:9px; border-bottom:2px solid var(--ink); }}
.decision {{ border-top:1px solid var(--line); padding:18px; background:#fafbfb; }}
.choice-row {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin:8px 0 16px; }}
.choice-row label {{ font-weight:500; }}
.decision-grid {{ display:grid; grid-template-columns:180px 150px 1fr; gap:14px; }}
.flags {{ display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:12px; }}
.flags label {{ font-weight:400; }}
.toolbar {{ position:sticky; bottom:0; z-index:4; border-top:1px solid var(--line);
  background:rgba(255,255,255,.97); padding:12px 0; }}
.toolbar-inner {{ width:min(1440px, calc(100% - 40px)); margin:auto; display:flex;
  align-items:center; justify-content:space-between; gap:18px; }}
button {{ border:1px solid var(--accent); background:var(--accent); color:#fff;
  padding:10px 15px; font:600 14px inherit; cursor:pointer; }}
button.secondary {{ background:#fff; color:var(--accent); }}
.notice {{ color:var(--warn); font-weight:600; }}
@media (max-width:900px) {{
  .masthead, .identity-grid, .answers, .decision-grid {{ grid-template-columns:1fr; }}
  .answer + .answer {{ border-left:0; border-top:1px solid var(--line); }}
  .attestations {{ grid-template-columns:1fr; }}
}}
@media print {{
  body {{ background:#fff; }} .toolbar {{ display:none; }}
  .masthead, main {{ width:100%; }} article {{ break-inside:avoid; }}
}}
</style>
</head>
<body>
<header>
  <div class="masthead">
    <div>
      <h1>{title}</h1>
      <p class="meta">{subtitle}</p>
      <div class="hash">Candidate pack {candidate_sha256}</div>
    </div>
    <div class="progress"><strong id="progress">0 / {len(candidate["items"])}</strong><span>complete</span></div>
  </div>
</header>
<main>
  <section class="instructions">
    <h2>Decision protocol</h2>
    <ol>
      <li>Work independently. Do not search for model, provider, or Epicure condition identity.</li>
      <li>Judge the culinary task first. Mark invalid tasks instead of forcing a preference.</li>
      <li>Read both complete, unedited responses. Choose overall quality only after comparison.</li>
      <li>Use <em>tie</em> for practical equivalence and <em>both bad</em> when neither is acceptable.</li>
      <li>Give a concrete rationale. The downloaded ballot contains no answer text or identity key.</li>
    </ol>
  </section>
  <section class="identity">
    <div class="identity-grid">
      <label>Role
        <select id="role">
          <option value="">Select role</option>
          {role_options}
        </select>
      </label>
      <label>Reviewer code<input id="reviewer-code" type="text" autocomplete="off"></label>
      <label>Affiliation<input id="affiliation" type="text" autocomplete="organization"></label>
    </div>
    <div class="attestations">
      <label><input type="checkbox" data-attest="worked_independently"> I worked independently.</label>
      <label><input type="checkbox" data-attest="no_model_or_condition_identity_lookup"> I did not seek answer identity.</label>
      <label><input type="checkbox" data-attest="reviewed_complete_unedited_answers"> I reviewed the complete presented answers.</label>
      <label data-role-attestation="independent_gold_adjudicator"><input type="checkbox" data-attest="independent_of_epicure_and_model_providers"> I am independent of Epicure and the evaluated providers.</label>
      <label data-role-attestation="affiliated_reviewer_calibration"><input type="checkbox" data-attest="product_affiliation_disclosed"> I have disclosed my Epicure or provider affiliation.</label>
    </div>
  </section>
  <div id="items"></div>
</main>
<div class="toolbar">
  <div class="toolbar-inner">
    <span id="notice" class="notice">Complete every field before exporting.</span>
    <div>
      <button class="secondary" id="import-button" type="button">Import draft</button>
      <input id="import-file" type="file" accept="application/json" hidden>
      <button id="download-button" type="button">Download ballot JSON</button>
    </div>
  </div>
</div>
<script>
const PACK = {embedded};
const choices = [
  ["left", "Answer A"], ["right", "Answer B"], ["tie", "Tie"], ["both_bad", "Both bad"]
];
const validity = [["valid","Valid"],["minor_issue","Minor issue"],["invalid","Invalid"]];
const flags = [
  ["identity_leak","Identity leak"],["response_truncated","Response truncated"],
  ["specialist_scope","Specialist scope"],["rights_or_privacy","Rights or privacy"],["other","Other"]
];
const esc = (value) => String(value).replace(/[&<>"']/g, (char) => (
  {{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[char]
));
const itemHtml = (item, index) => `
<article data-item="${{esc(item.calibration_item_id)}}">
  <div class="item-head"><h2>Comparison ${{index + 1}}</h2>
    <span class="family">${{esc(item.family)}}</span></div>
  <div class="prompt"><h3>Culinary task</h3><pre>${{esc(item.prompt)}}</pre></div>
  <div class="answers">
    <section class="answer"><h3>Answer A</h3><pre>${{esc(item.left.answer_markdown)}}</pre></section>
    <section class="answer"><h3>Answer B</h3><pre>${{esc(item.right.answer_markdown)}}</pre></section>
  </div>
  <div class="decision">
    <label>Overall preference</label>
    <div class="choice-row">${{choices.map(([value,label]) =>
      `<label><input type="radio" name="choice-${{index}}" value="${{value}}"> ${{label}}</label>`
    ).join("")}}</div>
    <div class="decision-grid">
      <label>Task validity<select data-field="task_validity">
        <option value="">Select</option>${{validity.map(([value,label]) =>
          `<option value="${{value}}">${{label}}</option>`).join("")}}</select></label>
      <label>Confidence<select data-field="confidence">
        <option value="">Select</option>${{[1,2,3,4,5].map(value =>
          `<option value="${{value}}">${{value}} / 5</option>`).join("")}}</select></label>
      <label>Comparative rationale<textarea data-field="rationale"
        placeholder="State the decisive culinary evidence (minimum 20 characters)."></textarea></label>
    </div>
    <div class="flags">${{flags.map(([value,label]) =>
      `<label><input type="checkbox" data-flag="${{value}}"> ${{label}}</label>`).join("")}}</div>
  </div>
</article>`;
document.getElementById("items").innerHTML = PACK.items.map(itemHtml).join("");

function ballot() {{
  return {{
    schema_version: "{BALLOT_SCHEMA_VERSION}",
    candidate_pack_sha256: PACK.candidatePackSha256,
    role: document.getElementById("role").value,
    reviewer: {{
      reviewer_code: document.getElementById("reviewer-code").value.trim(),
      affiliation: document.getElementById("affiliation").value.trim()
    }},
    attestations: Object.fromEntries([...document.querySelectorAll("[data-attest]")]
      .map(node => [node.dataset.attest, node.checked])),
    items: PACK.items.map((item, index) => {{
      const root = document.querySelector(`[data-item="${{item.calibration_item_id}}"]`);
      return {{
        calibration_item_id: item.calibration_item_id,
        task_validity: root.querySelector("[data-field=task_validity]").value,
        choice: root.querySelector(`input[name=choice-${{index}}]:checked`)?.value || "",
        confidence: Number(root.querySelector("[data-field=confidence]").value) || null,
        rationale: root.querySelector("[data-field=rationale]").value.trim(),
        flags: [...root.querySelectorAll("[data-flag]:checked")].map(node => node.dataset.flag)
      }};
    }})
  }};
}}
function validItem(item) {{
  return validity.some(([value]) => value === item.task_validity)
    && choices.some(([value]) => value === item.choice)
    && Number.isInteger(item.confidence) && item.confidence >= 1 && item.confidence <= 5
    && item.rationale.length >= 20;
}}
function requiredAttestations(role) {{
  const common = [
    "worked_independently",
    "no_model_or_condition_identity_lookup",
    "reviewed_complete_unedited_answers"
  ];
  return role === "independent_gold_adjudicator"
    ? [...common, "independent_of_epicure_and_model_providers"]
    : [...common, "product_affiliation_disclosed"];
}}
function updateRoleAttestations() {{
  const role = document.getElementById("role").value;
  document.querySelectorAll("[data-role-attestation]").forEach(node => {{
    node.hidden = node.dataset.roleAttestation !== role;
  }});
}}
function updateProgress() {{
  const value = ballot();
  const completed = value.items.filter(validItem).length;
  document.getElementById("progress").textContent = `${{completed}} / ${{value.items.length}}`;
  document.getElementById("notice").textContent = completed === value.items.length
    ? "All comparisons complete. Confirm identity and attestations."
    : `${{value.items.length - completed}} comparisons remain incomplete.`;
}}
document.addEventListener("change", updateProgress);
document.addEventListener("input", updateProgress);
document.getElementById("role").addEventListener("change", updateRoleAttestations);
document.getElementById("download-button").addEventListener("click", () => {{
  const value = ballot();
  const attestations = requiredAttestations(value.role)
    .every(name => value.attestations[name] === true);
  if (!value.role || value.reviewer.reviewer_code.length < 3
      || value.reviewer.affiliation.length < 2 || !attestations
      || !value.items.every(validItem)) {{
    document.getElementById("notice").textContent =
      "Export blocked: complete identity, attestations, choices, confidence, and rationales.";
    return;
  }}
  const blob = new Blob([JSON.stringify(value, null, 2) + "\\n"], {{type:"application/json"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `flavourbench-calibration-${{value.reviewer.reviewer_code}}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}});
document.getElementById("import-button").addEventListener("click", () =>
  document.getElementById("import-file").click());
document.getElementById("import-file").addEventListener("change", async (event) => {{
  const file = event.target.files?.[0]; if (!file) return;
  const value = JSON.parse(await file.text());
  if (value.candidate_pack_sha256 !== PACK.candidatePackSha256) {{
    document.getElementById("notice").textContent = "Draft belongs to another candidate pack.";
    return;
  }}
  document.getElementById("role").value = value.role || "";
  document.getElementById("reviewer-code").value = value.reviewer?.reviewer_code || "";
  document.getElementById("affiliation").value = value.reviewer?.affiliation || "";
  document.querySelectorAll("[data-attest]").forEach(node => {{
    node.checked = Boolean(value.attestations?.[node.dataset.attest]);
  }});
  const byId = Object.fromEntries((value.items || []).map(item => [item.calibration_item_id,item]));
  PACK.items.forEach((item,index) => {{
    const saved = byId[item.calibration_item_id]; if (!saved) return;
    const root = document.querySelector(`[data-item="${{item.calibration_item_id}}"]`);
    root.querySelector("[data-field=task_validity]").value = saved.task_validity || "";
    root.querySelector("[data-field=confidence]").value = saved.confidence || "";
    root.querySelector("[data-field=rationale]").value = saved.rationale || "";
    root.querySelectorAll(`input[name=choice-${{index}}]`).forEach(node => {{
      node.checked = node.value === saved.choice;
    }});
    root.querySelectorAll("[data-flag]").forEach(node => {{
      node.checked = (saved.flags || []).includes(node.dataset.flag);
    }});
  }});
  updateProgress();
}});
updateRoleAttestations();
updateProgress();
</script>
</body>
</html>
"""


def build_candidate_artifacts(
    *,
    arms_dir: Path,
    output_dir: Path,
    identity_path: Path,
    candidates_per_family: int = 8,
    frozen_target_per_family: int = 5,
) -> dict[str, Any]:
    candidate_payload, identity_core = build_candidate_payload(
        arms_dir,
        candidates_per_family=candidates_per_family,
        frozen_target_per_family=frozen_target_per_family,
        require_governance_reviews=True,
    )
    candidate_document, candidate_sha256 = _artifact_document(candidate_payload)
    candidate_path = output_dir / f"candidate-pack-{candidate_sha256}.json"
    template_payload = ballot_template(candidate_document, candidate_sha256)
    template_document, template_sha256 = _artifact_document(template_payload)
    template_path = output_dir / f"ballot-template-{template_sha256}.json"
    html_path = output_dir / f"adjudication-workspace-{candidate_sha256}.html"
    identity_document = {
        **identity_core,
        "identity_commitment_sha256": sha256_json(identity_core),
        "candidate_pack_sha256": candidate_sha256,
    }
    _atomic_write_json(candidate_path, candidate_document)
    _atomic_write_json(template_path, template_document)
    _atomic_write_json(identity_path, identity_document, mode=0o600)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        _render_html(
            candidate_document,
            candidate_sha256,
            title="FlavourBench gold adjudication",
            subtitle=("Real paid model outputs, real Epicure MCP traces, identities sealed."),
            allowed_roles=("independent_gold_adjudicator",),
        ),
        encoding="utf-8",
    )
    os.chmod(html_path, 0o644)
    return {
        "candidate_pack": str(candidate_path),
        "candidate_pack_sha256": candidate_sha256,
        "ballot_template": str(template_path),
        "ballot_template_sha256": template_sha256,
        "adjudication_workspace": str(html_path),
        "identity_key": str(identity_path),
        "identity_commitment_sha256": identity_document["identity_commitment_sha256"],
        "observed": candidate_document["observed"],
    }


def _validate_ballot(
    ballot: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    expected_role: str,
) -> dict[str, dict[str, Any]]:
    if ballot.get("schema_version") != BALLOT_SCHEMA_VERSION:
        raise ExpertCalibrationError("ballot schema version is unsupported")
    if ballot.get("candidate_pack_sha256") != candidate.get("artifact_sha256"):
        raise ExpertCalibrationError("ballot is bound to another candidate pack")
    if ballot.get("role") != expected_role:
        raise ExpertCalibrationError(f"ballot role must be {expected_role}")
    reviewer = _mapping(ballot.get("reviewer"), label="reviewer")
    if (
        not isinstance(reviewer.get("reviewer_code"), str)
        or len(reviewer["reviewer_code"].strip()) < 3
    ):
        raise ExpertCalibrationError("ballot has no reviewer code")
    if not isinstance(reviewer.get("affiliation"), str) or len(reviewer["affiliation"].strip()) < 2:
        raise ExpertCalibrationError("ballot has no reviewer affiliation")
    attestations = _mapping(ballot.get("attestations"), label="attestations")
    required_attestations = {
        "worked_independently",
        "no_model_or_condition_identity_lookup",
        "reviewed_complete_unedited_answers",
    }
    if expected_role == "independent_gold_adjudicator":
        required_attestations.add("independent_of_epicure_and_model_providers")
    elif expected_role == "affiliated_reviewer_calibration":
        required_attestations.add("product_affiliation_disclosed")
    if any(attestations.get(name) is not True for name in required_attestations):
        raise ExpertCalibrationError("ballot attestations are incomplete")
    rows = ballot.get("items")
    if not isinstance(rows, list):
        raise ExpertCalibrationError("ballot items are missing")
    expected_ids = {str(item["calibration_item_id"]) for item in candidate.get("items", [])}
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ExpertCalibrationError("ballot contains a non-object item")
        item_id = row.get("calibration_item_id")
        if not isinstance(item_id, str) or item_id not in expected_ids:
            raise ExpertCalibrationError("ballot contains an unknown item")
        if item_id in by_id:
            raise ExpertCalibrationError("ballot contains a duplicate item")
        if row.get("task_validity") not in TASK_VALIDITY:
            raise ExpertCalibrationError(f"{item_id} has no valid task assessment")
        if row.get("choice") not in CHOICES:
            raise ExpertCalibrationError(f"{item_id} has no valid preference")
        confidence = row.get("confidence")
        if (
            not isinstance(confidence, int)
            or isinstance(confidence, bool)
            or not 1 <= confidence <= 5
        ):
            raise ExpertCalibrationError(f"{item_id} has no valid confidence")
        rationale = row.get("rationale")
        if not isinstance(rationale, str) or len(rationale.strip()) < 20:
            raise ExpertCalibrationError(f"{item_id} has no adequate rationale")
        flags = row.get("flags")
        if (
            not isinstance(flags, list)
            or len(flags) != len(set(flags))
            or any(flag not in ITEM_FLAGS for flag in flags)
        ):
            raise ExpertCalibrationError(f"{item_id} has invalid flags")
        by_id[item_id] = row
    if set(by_id) != expected_ids:
        raise ExpertCalibrationError("ballot does not cover the complete candidate pack")
    return by_id


def freeze_gold_set(
    *,
    candidate_path: Path,
    ballot_paths: Sequence[Path],
    output_path: Path,
    reviewer_pack_path: Path,
    reviewer_workspace_path: Path,
    reviewer_ballot_template_path: Path,
    adjudication_ballot_path: Path | None = None,
    target_per_family: int = 5,
) -> dict[str, Any]:
    candidate = _load_object(candidate_path)
    _verify_artifact_document(candidate_path, candidate)
    if candidate.get("schema_version") != SCHEMA_VERSION:
        raise ExpertCalibrationError("candidate pack schema version is unsupported")
    current_frontier_quarantine = _assert_current_frontier_quarantine_binding(candidate)
    if len(ballot_paths) < 2:
        raise ExpertCalibrationError("two independent gold ballots are required")
    ballots = [_load_object(path) for path in ballot_paths]
    reviewer_codes = [
        str(_mapping(ballot.get("reviewer"), label="reviewer")["reviewer_code"])
        for ballot in ballots
    ]
    if len(reviewer_codes) != len(set(reviewer_codes)):
        raise ExpertCalibrationError("gold adjudicators must have distinct reviewer codes")
    reviewed = [
        _validate_ballot(
            ballot,
            candidate=candidate,
            expected_role="independent_gold_adjudicator",
        )
        for ballot in ballots
    ]
    adjudicated = None
    if adjudication_ballot_path is not None:
        adjudication_ballot = _load_object(adjudication_ballot_path)
        adjudicated = _validate_ballot(
            adjudication_ballot,
            candidate=candidate,
            expected_role="independent_gold_adjudicator",
        )
        adjudicator_code = str(adjudication_ballot["reviewer"]["reviewer_code"])
        if adjudicator_code in reviewer_codes:
            raise ExpertCalibrationError(
                "disagreement adjudicator must be distinct from primary adjudicators"
            )

    candidate_by_id = {str(item["calibration_item_id"]): item for item in candidate["items"]}
    resolved: dict[str, dict[str, Any]] = {}
    disagreements: list[str] = []
    excluded: dict[str, str] = {}
    for item_id, item in candidate_by_id.items():
        rows = [ballot[item_id] for ballot in reviewed]
        if any(
            row["task_validity"] == "invalid"
            or "identity_leak" in row["flags"]
            or "response_truncated" in row["flags"]
            or "rights_or_privacy" in row["flags"]
            for row in rows
        ):
            excluded[item_id] = "primary_adjudicator_exclusion"
            continue
        labels = {str(row["choice"]) for row in rows}
        if len(labels) == 1:
            label = next(iter(labels))
            resolution = "unanimous_primary_adjudicators"
        else:
            disagreements.append(item_id)
            if adjudicated is None:
                continue
            decision = adjudicated[item_id]
            if (
                decision["task_validity"] == "invalid"
                or "identity_leak" in decision["flags"]
                or "response_truncated" in decision["flags"]
                or "rights_or_privacy" in decision["flags"]
            ):
                excluded[item_id] = "disagreement_adjudicator_exclusion"
                continue
            label = str(decision["choice"])
            resolution = "third_independent_adjudicator"
        resolved[item_id] = {
            "calibration_item_id": item_id,
            "family": item["family"],
            "choice": label,
            "resolution": resolution,
        }
    if disagreements and adjudicated is None:
        raise ExpertCalibrationError(
            f"{len(disagreements)} preference disagreements require a third ballot"
        )

    frozen_items: list[dict[str, Any]] = []
    for family in TASK_FAMILIES:
        rows = sorted(
            (row for row in resolved.values() if row["family"] == family),
            key=lambda row: sha256_text(
                f"{candidate['artifact_sha256']}|freeze|{row['calibration_item_id']}"
            ),
        )
        if len(rows) < target_per_family:
            raise ExpertCalibrationError(
                f"{family} has only {len(rows)} resolved valid candidates; "
                f"{target_per_family} required"
            )
        frozen_items.extend(rows[:target_per_family])
    frozen_items.sort(
        key=lambda row: sha256_text(
            f"{candidate['artifact_sha256']}|frozen-order|{row['calibration_item_id']}"
        )
    )
    reviewer_items = [candidate_by_id[str(row["calibration_item_id"])] for row in frozen_items]
    gold_answer_commitment = sha256_json(
        {
            "candidate_pack_sha256": candidate["artifact_sha256"],
            "items": frozen_items,
        }
    )
    reviewer_pack_payload = {
        "schema_version": REVIEWER_PACK_SCHEMA_VERSION,
        "calibration_set_id": candidate["calibration_set_id"],
        "status": "frozen_blinded_reviewer_calibration",
        "candidate_pack_sha256": candidate["artifact_sha256"],
        "gold_answer_commitment_sha256": gold_answer_commitment,
        "target": {
            "items": len(reviewer_items),
            "items_per_family": target_per_family,
            "pass_threshold": 0.8,
        },
        "blinding": {
            "model_provider_condition_and_gold_removed": True,
            "identity_lookup_prohibited": True,
        },
        "use_policy": {
            "reviewer_admission_only": True,
            "rank_eligible": False,
            "benchmark_result_use": "prohibited",
        },
        "current_frontier_task_quarantine": current_frontier_quarantine,
        "items": reviewer_items,
    }
    reviewer_pack_document, reviewer_pack_sha256 = _artifact_document(reviewer_pack_payload)
    reviewer_ballot_payload = ballot_template(
        reviewer_pack_document,
        reviewer_pack_sha256,
    )
    reviewer_ballot_document, reviewer_ballot_template_sha256 = _artifact_document(
        reviewer_ballot_payload
    )
    frozen_payload = {
        "schema_version": FROZEN_SCHEMA_VERSION,
        "calibration_set_id": candidate["calibration_set_id"],
        "status": "frozen_human_gold",
        "candidate_pack_sha256": candidate["artifact_sha256"],
        "reviewer_pack_sha256": reviewer_pack_sha256,
        "gold_answer_commitment_sha256": gold_answer_commitment,
        "independent_gold_adjudicators": len(ballots),
        "disagreement_adjudicator_used": adjudicated is not None,
        "disagreements": len(disagreements),
        "excluded_candidates": excluded,
        "score_rule": {
            "metric": "exact_preference_accuracy",
            "pass_threshold": 0.8,
            "ties": "exact_match",
            "both_bad": "exact_match",
        },
        "use_policy": {
            "reviewer_admission_only": True,
            "rank_eligible": False,
            "benchmark_result_use": "prohibited",
        },
        "current_frontier_task_quarantine": current_frontier_quarantine,
        "items": frozen_items,
    }
    frozen_document, frozen_sha256 = _artifact_document(frozen_payload)
    _atomic_write_json(reviewer_pack_path, reviewer_pack_document)
    _atomic_write_json(
        reviewer_ballot_template_path,
        reviewer_ballot_document,
    )
    reviewer_workspace_path.parent.mkdir(parents=True, exist_ok=True)
    reviewer_workspace_path.write_text(
        _render_html(
            reviewer_pack_document,
            reviewer_pack_sha256,
            title="FlavourBench reviewer calibration",
            subtitle=(
                "Twenty frozen real-output comparisons. Gold labels and identities remain sealed."
            ),
            allowed_roles=("affiliated_reviewer_calibration",),
        ),
        encoding="utf-8",
    )
    os.chmod(reviewer_workspace_path, 0o644)
    _atomic_write_json(output_path, frozen_document, mode=0o600)
    return {
        "frozen_set": str(output_path),
        "frozen_set_sha256": frozen_sha256,
        "reviewer_pack": str(reviewer_pack_path),
        "reviewer_pack_sha256": reviewer_pack_sha256,
        "reviewer_workspace": str(reviewer_workspace_path),
        "reviewer_ballot_template": str(reviewer_ballot_template_path),
        "reviewer_ballot_template_sha256": reviewer_ballot_template_sha256,
        "items": len(frozen_items),
        "items_by_family": dict(sorted(Counter(row["family"] for row in frozen_items).items())),
        "independent_gold_adjudicators": len(ballots),
        "disagreements": len(disagreements),
        "excluded_candidates": len(excluded),
    }


def score_reviewer_ballot(
    *,
    reviewer_pack_path: Path,
    frozen_path: Path,
    ballot_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    reviewer_pack = _load_object(reviewer_pack_path)
    frozen = _load_object(frozen_path)
    ballot = _load_object(ballot_path)
    _verify_artifact_document(reviewer_pack_path, reviewer_pack)
    _verify_artifact_document(frozen_path, frozen)
    if frozen.get("schema_version") != FROZEN_SCHEMA_VERSION:
        raise ExpertCalibrationError("frozen calibration schema version is unsupported")
    if reviewer_pack.get("schema_version") != REVIEWER_PACK_SCHEMA_VERSION:
        raise ExpertCalibrationError("reviewer calibration pack schema is unsupported")
    if frozen.get("reviewer_pack_sha256") != reviewer_pack.get("artifact_sha256"):
        raise ExpertCalibrationError("frozen key belongs to another reviewer pack")
    reviewed = _validate_ballot(
        ballot,
        candidate=reviewer_pack,
        expected_role="affiliated_reviewer_calibration",
    )
    frozen_items = {str(row["calibration_item_id"]): row for row in frozen.get("items", [])}
    correct = {
        item_id: reviewed[item_id]["choice"] == gold["choice"]
        for item_id, gold in frozen_items.items()
    }
    total = len(correct)
    if total != 20:
        raise ExpertCalibrationError("frozen calibration set must contain exactly 20 items")
    correct_count = sum(correct.values())
    accuracy = correct_count / total
    score_payload = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "calibration_set_id": frozen["calibration_set_id"],
        "candidate_pack_sha256": frozen["candidate_pack_sha256"],
        "reviewer_pack_sha256": reviewer_pack["artifact_sha256"],
        "frozen_set_sha256": frozen["artifact_sha256"],
        "reviewer_code": ballot["reviewer"]["reviewer_code"],
        "items": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "pass_threshold": 0.8,
        "passed": accuracy >= 0.8,
        "per_family": {
            family: {
                "items": sum(row["family"] == family for row in frozen_items.values()),
                "correct": sum(
                    correct[item_id]
                    for item_id, row in frozen_items.items()
                    if row["family"] == family
                ),
            }
            for family in TASK_FAMILIES
        },
        "result_use": "reviewer admission only; excluded from benchmark results",
        "rank_eligible": False,
    }
    score_document, score_sha256 = _artifact_document(score_payload)
    _atomic_write_json(output_path, score_document, mode=0o600)
    return {
        "score_artifact": str(output_path),
        "score_sha256": score_sha256,
        "items": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "passed": accuracy >= 0.8,
    }


def _default_paths() -> tuple[Path, Path, Path]:
    repository = Path(__file__).resolve().parents[2]
    arms = repository / "artifacts" / "season0" / "scored-v1" / "arms"
    output = repository / "artifacts" / "expert-calibration" / "candidate-v11"
    identity = repository / ".private" / "flavourbench-calibration-identity-key-v11.json"
    return arms, output, identity


def run() -> None:
    default_arms, default_output, default_identity = _default_paths()
    parser = argparse.ArgumentParser(
        description="Build, freeze, or score the real-output expert calibration set."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--arms-dir", type=Path, default=default_arms)
    build.add_argument("--output-dir", type=Path, default=default_output)
    build.add_argument("--identity-path", type=Path, default=default_identity)
    build.add_argument("--candidates-per-family", type=int, default=8)
    build.add_argument("--frozen-target-per-family", type=int, default=5)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--candidate-pack", type=Path, required=True)
    freeze.add_argument("--gold-ballot", type=Path, action="append", required=True)
    freeze.add_argument("--adjudication-ballot", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--reviewer-pack-output", type=Path, required=True)
    freeze.add_argument("--reviewer-workspace-output", type=Path, required=True)
    freeze.add_argument("--reviewer-ballot-template-output", type=Path, required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--reviewer-pack", type=Path, required=True)
    score.add_argument("--frozen-set", type=Path, required=True)
    score.add_argument("--reviewer-ballot", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_candidate_artifacts(
            arms_dir=args.arms_dir,
            output_dir=args.output_dir,
            identity_path=args.identity_path,
            candidates_per_family=args.candidates_per_family,
            frozen_target_per_family=args.frozen_target_per_family,
        )
    elif args.command == "freeze":
        result = freeze_gold_set(
            candidate_path=args.candidate_pack,
            ballot_paths=args.gold_ballot,
            output_path=args.output,
            reviewer_pack_path=args.reviewer_pack_output,
            reviewer_workspace_path=args.reviewer_workspace_output,
            reviewer_ballot_template_path=args.reviewer_ballot_template_output,
            adjudication_ballot_path=args.adjudication_ballot,
        )
    else:
        result = score_reviewer_ballot(
            reviewer_pack_path=args.reviewer_pack,
            frozen_path=args.frozen_set,
            ballot_path=args.reviewer_ballot,
            output_path=args.output,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
