from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .provider import FINAL_SCHEMA
from .security import contains_identity_leak

VALIDATOR_VERSION = "1.3.0"
NORMAL_FINISH_REASONS = frozenset({"completed", "end_turn", "stop", "stop_sequence"})
_CONSTRAINT = re.compile(
    r"(?:\b(?:without|avoid|exclude|must not contain|no added|free[- ]from|vegan|vegetarian|"
    r"under \d+|within \d+|no more than)\b|\b[\w-]+-free\b)",
    re.IGNORECASE,
)
_ANSWER_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
_DANGLING_END = re.compile(r"(?:[:,;]|(?:^|\s)(?:and|or|because|including|such as))\s*$", re.I)
_MARKDOWN_HEADING_END = re.compile(r"(?:^|\n)\s{0,3}#{1,6}\s+[^\n]+\s*$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_EVIDENCE_SOURCE = re.compile(
    r"\b(?:data|dataset|score|scores|similarity|similarities|embedding|embeddings|"
    r"network|networks|pairing|pairings|neighbou?r|neighbou?rs|affinity|affinities)\b",
    re.IGNORECASE,
)
_DIRECT_CERTAINTY = re.compile(
    r"\b(?:proves?|proof|confirms?|demonstrates?|establishes?|guarantees?)\b",
    re.IGNORECASE,
)
_CAUSAL_OR_MECHANISTIC = re.compile(
    r"\b(?:caus(?:e|es|ed|al|ation)|mechanis(?:m|tic)|because of|responsible for)\b",
    re.IGNORECASE,
)
_FUNCTIONAL_OR_SAFETY_PROPERTY = re.compile(
    r"\b(?:bind(?:er|ing)|thicken(?:er|ing)|emulsif(?:y|ier|ication)|gell?(?:ing|ant)|"
    r"sweeter|sweetness|more acidic|acidity|pH|safe|safety|food-safe|toxic|toxicity|"
    r"shelf[- ]life|antimicrobial)\b",
    re.IGNORECASE,
)
_ASSERTIVE_LINK = re.compile(
    r"\b(?:therefore|thus|hence|which means|means that|shows that|indicates that|"
    r"confirms|proves|demonstrates|establishes)\b",
    re.IGNORECASE,
)
_CLAIM_NEGATION = re.compile(
    r"\b(?:does not|do not|did not|cannot|can not|can't|is not|are not|was not|"
    r"were not|not enough to|insufficient to|should not be used to|doesn't)\b",
    re.IGNORECASE,
)
_EXTERNAL_CONTEXT = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_VISUAL_DEPENDENCY = re.compile(
    r"\b(?:image|images|photo|photos|picture|pictures|diagram|diagrams|attached)\b",
    re.IGNORECASE,
)


def semantic_completion_detail(
    answer: str, *, finish_reason: str | None = None
) -> dict[str, Any]:
    """Detect only strong, provider-independent signs of an unfinished answer.

    This is a conservative shape gate, not a culinary-quality grader. Human review
    remains responsible for answers that are complete in form but incomplete in
    substance.
    """

    normalized = answer.strip()
    word_count = len(_ANSWER_WORD.findall(normalized))
    reasons: list[str] = []
    if len(normalized) < 120:
        reasons.append("fewer_than_120_characters")
    if word_count < 24:
        reasons.append("fewer_than_24_words")
    if normalized.count("```") % 2:
        reasons.append("unclosed_code_fence")
    if _DANGLING_END.search(normalized):
        reasons.append("dangling_terminal_clause")
    if _MARKDOWN_HEADING_END.search(normalized):
        reasons.append("ends_with_markdown_heading")
    normalized_finish_reason = str(finish_reason or "").strip().lower()
    if finish_reason is not None and normalized_finish_reason not in NORMAL_FINISH_REASONS:
        reasons.append("non_normal_finish_reason")
    return {
        "character_count": len(normalized),
        "word_count": word_count,
        "finish_reason": normalized_finish_reason or None,
        "accepted_finish_reasons": sorted(NORMAL_FINISH_REASONS),
        "failure_reasons": reasons,
        "scope": (
            "Conservative response-shape gate only; expert review determines "
            "substantive completeness."
        ),
    }


def evidence_claim_boundary_detail(answer: str) -> dict[str, Any]:
    """Flag association-to-causation/property leaps for targeted human audit.

    This deliberately conservative lexical screen is not a culinary truth judge.
    It stores only rule identifiers and sentence hashes, never answer excerpts.
    """

    findings: list[dict[str, Any]] = []
    for sentence_index, sentence in enumerate(_SENTENCE_BOUNDARY.split(answer.strip())):
        normalized = " ".join(sentence.split())
        if not normalized or not _EVIDENCE_SOURCE.search(normalized):
            continue
        if _CLAIM_NEGATION.search(normalized):
            continue
        rule_ids: list[str] = []
        if _DIRECT_CERTAINTY.search(normalized):
            rule_ids.append("association_presented_as_proof")
        if _CAUSAL_OR_MECHANISTIC.search(normalized) and _ASSERTIVE_LINK.search(normalized):
            rule_ids.append("association_presented_as_mechanism")
        if _FUNCTIONAL_OR_SAFETY_PROPERTY.search(normalized) and (
            _ASSERTIVE_LINK.search(normalized) or _DIRECT_CERTAINTY.search(normalized)
        ):
            rule_ids.append("association_used_for_functional_or_safety_property")
        if rule_ids:
            findings.append(
                {
                    "sentence_index": sentence_index,
                    "sentence_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    "rule_ids": sorted(set(rule_ids)),
                }
            )
    return {
        "warning_count": len(findings),
        "findings": findings,
        "scope": (
            "Lexical audit trigger only. A qualified reviewer determines whether the claim is "
            "actually unsupported; this validator is never ground truth or a ranking score."
        ),
        "raw_answer_excerpt_stored": False,
    }


def task_surface_integrity_detail(prompt: str) -> dict[str, Any]:
    """Expose obvious dependencies that make a text-only task non-self-contained."""

    reasons: list[str] = []
    if _EXTERNAL_CONTEXT.search(prompt):
        reasons.append("external_url_dependency_signal")
    if _VISUAL_DEPENDENCY.search(prompt):
        reasons.append("visual_context_dependency_signal")
    return {
        "failure_reasons": reasons,
        "scope": (
            "High-recall surface triage only; human answer-blind validation and criterion packs "
            "remain required for confirmatory task admission."
        ),
    }


@dataclass(frozen=True)
class Validation:
    name: str
    status: str
    score_milli: int | None
    detail: dict[str, Any]


def validate_output(
    *,
    prompt: str,
    output: dict[str, Any],
    answer: str,
    model_name: str,
    tool_errors: int,
    tool_calls: int,
    finish_reason: str | None = None,
) -> list[Validation]:
    required = set(FINAL_SCHEMA["required"])
    schema_ok = required.issubset(output) and all(
        isinstance(output.get(key), list)
        for key in ("ingredient_mentions", "constraints_addressed", "uncertainties")
    )
    identity_ok = not contains_identity_leak(answer, model_name, prompt)
    semantic_detail = semantic_completion_detail(answer, finish_reason=finish_reason)
    semantic_ok = not semantic_detail["failure_reasons"]
    evidence_detail = evidence_claim_boundary_detail(answer)
    task_surface_detail = task_surface_integrity_detail(prompt)
    has_explicit_constraint = bool(_CONSTRAINT.search(prompt))
    addressed = bool(output.get("constraints_addressed"))
    tool_score = 1000 if tool_calls == 0 else round(1000 * (tool_calls - tool_errors) / tool_calls)
    return [
        Validation(
            name="structured_response",
            status="pass" if schema_ok else "fail",
            score_milli=1000 if schema_ok else 0,
            detail={"required_fields": sorted(required)},
        ),
        Validation(
            name="identity_blinding",
            status="pass" if identity_ok else "fail",
            score_milli=1000 if identity_ok else 0,
            detail={
                "identity_marker_detected": not identity_ok,
                "gate": "identity-blinding-v2",
            },
        ),
        Validation(
            name="semantic_completion",
            status="pass" if semantic_ok else "fail",
            score_milli=1000 if semantic_ok else 0,
            detail=semantic_detail,
        ),
        Validation(
            name="evidence_claim_boundary",
            status="warn" if evidence_detail["warning_count"] else "pass",
            score_milli=0 if evidence_detail["warning_count"] else 1000,
            detail=evidence_detail,
        ),
        Validation(
            name="task_surface_integrity",
            status="warn" if task_surface_detail["failure_reasons"] else "pass",
            score_milli=0 if task_surface_detail["failure_reasons"] else 1000,
            detail=task_surface_detail,
        ),
        Validation(
            name="constraint_acknowledgement",
            status=("pass" if addressed else "warn")
            if has_explicit_constraint
            else "not_applicable",
            score_milli=(1000 if addressed else 0) if has_explicit_constraint else None,
            detail={
                "explicit_constraint_signal": has_explicit_constraint,
                "reported_constraints": output.get("constraints_addressed", []),
                "scope": "acknowledgement only; expert review determines substantive compliance",
            },
        ),
        Validation(
            name="tool_execution",
            status="pass" if tool_errors == 0 else "warn",
            score_milli=tool_score,
            detail={"tool_calls": tool_calls, "tool_errors": tool_errors},
        ),
    ]
