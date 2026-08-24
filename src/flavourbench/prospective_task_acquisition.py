"""Acquire recent, licensed, human-authored culinary task candidates.

This module deliberately stops before task validation.  It copies public question
text from Seasoned Advice, records the exact source revision and licence, applies
only deterministic HTML-to-text and narrow contact-field redactions, and emits
provisional rule-based screening signals.  It never requests answer bodies and it
never calls a model or Epicure.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import tempfile
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

import httpx

API_BASE_URL = "https://api.stackexchange.com/2.3"
API_SITE = "cooking"
SOURCE_CORPUS = "Seasoned Advice (Stack Exchange)"
LICENSING_URL = "https://stackoverflow.com/help/licensing"
ACCEPTED_LICENSES = frozenset({"CC BY-SA 4.0"})
DEFAULT_FROMDATE = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())
FAMILIES = ("substitution", "composition", "cookability", "evidence")

BUNDLE_SCHEMA = "flavourbench-public-human-task-candidate-bundle-v1"
ASSIGNMENT_SCHEMA = "flavourbench-public-human-task-review-assignment-v1"
RECEIPT_SCHEMA = "flavourbench-public-api-acquisition-receipt-v1"
SCREEN_POLICY_VERSION = "flavourbench-public-question-provisional-screens-v1"
NORMALIZATION_VERSION = "deterministic-html-to-text-v1"
DEIDENTIFICATION_VERSION = "narrow-direct-contact-redaction-v1"

_CANDIDATE_NAMESPACE = uuid.UUID("6581ce61-01a4-5ab0-8c3e-540438366b56")

_SPECIALIST_TAGS = frozenset(
    {
        "allergy",
        "allergens",
        "cultural-difference",
        "diet",
        "food-safety",
        "health",
        "kosher",
        "medical",
        "nutrition",
        "nutrient-composition",
        "raw-meat",
        "defrosting",
        "mold",
        "religion",
        "storage-lifetime",
        "storage-method",
    }
)

_SPECIALIST_PATTERNS: dict[str, re.Pattern[str]] = {
    "formal_allergen": re.compile(
        r"\b(?:allerg(?:y|ies|ic|en|ens)|anaphyl(?:axis|actic)|celiac|coeliac|"
        r"intoleran(?:ce|t))\b",
        re.IGNORECASE,
    ),
    "formal_nutrition": re.compile(
        r"\b(?:nutrition(?:al)?|calorie(?:s)?|kilocalorie(?:s)?|macros?|"
        r"recommended daily (?:allowance|intake)|glyc(?:a?emic|emic) index)\b",
        re.IGNORECASE,
    ),
    "food_safety": re.compile(
        r"\b(?:safe to (?:eat|consume)|food poisoning|botulism|pathogen(?:s|ic)?|"
        r"salmonella|listeria|e\.?\s*coli|spoiled|spoilage|shelf[- ]life|"
        r"danger zone|water[- ]bath canning|pressure canning)\b",
        re.IGNORECASE,
    ),
    "cultural_authenticity": re.compile(
        r"\b(?:authentic(?:ity|ally)?|culturally correct|real (?:italian|chinese|"
        r"indian|japanese|mexican|thai|french) way)\b",
        re.IGNORECASE,
    ),
    "medical_or_legal": re.compile(
        r"\b(?:medical advice|doctor|physician|diagnos(?:e|is)|prescription|"
        r"legal advice|legally|lawful|regulation compliance)\b",
        re.IGNORECASE,
    ),
}

_FAMILY_TAG_WEIGHTS: dict[str, dict[str, int]] = {
    "substitution": {
        "substitutions": 8,
        "ingredient-selection": 3,
        "vegan": 3,
        "vegetarian": 3,
        "gluten-free": 3,
        "dairy-free": 3,
        "sugar-free": 3,
    },
    "composition": {
        "flavor": 8,
        "spices": 5,
        "seasoning": 5,
        "pairing": 7,
        "menu-planning": 5,
        "ingredients": 3,
        "sauce": 2,
        "marinade": 3,
        "herbs": 3,
        "salt": 2,
        "vinegar": 2,
    },
    "cookability": {
        "baking": 3,
        "equipment": 4,
        "texture": 3,
        "recipe-scaling": 6,
        "cooking-time": 5,
        "measurements": 3,
        "dough": 3,
        "oven": 4,
        "technique": 4,
    },
    "evidence": {
        "food-science": 9,
        "chemistry": 9,
        "molecular-gastronomy": 8,
        "fermentation": 3,
        "emulsion": 5,
        "temperature": 2,
    },
}

_FAMILY_TEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "substitution": re.compile(
        r"\b(?:substitut(?:e|es|ed|ing|ion)|replac(?:e|ement|ing)|alternative to|"
        r"instead of|swap(?:ping)?|without|omit(?:ting)?|leave (?:it|them) out)\b",
        re.IGNORECASE,
    ),
    "composition": re.compile(
        r"\b(?:flavou?r|taste|season(?:ing)?|spic(?:e|es|y)|herb|pair(?:ing)?|"
        r"combin(?:e|ation)|complement|balance|blend|marinade|aroma|umami)\b",
        re.IGNORECASE,
    ),
    "cookability": re.compile(
        r"\b(?:cook(?:ing)?|bake|baking|oven|pan|equipment|method|technique|"
        r"temperature|scale|batch|dough|texture|consistency|prepare|recipe)\b",
        re.IGNORECASE,
    ),
    "evidence": re.compile(
        r"\b(?:why|science|chemistry|reaction|mechanism|molecule|starch|gluten|"
        r"emulsi(?:on|fy|fied)|ferment(?:ation|ing)?|carameli[sz]|maillard|"
        r"crystal|acid|alkali|pH|curdl|coagulat|denatur|gelati[sz])\b",
        re.IGNORECASE,
    ),
}

_IMAGE_CONTEXT_PATTERN = re.compile(
    r"<(?:img|figure|iframe)\b|\b(?:see|shown in|attached|pictured in) (?:the )?"
    r"(?:image|photo|picture|screenshot)\b",
    re.IGNORECASE,
)
_LINK_CONTEXT_PATTERN = re.compile(
    r"\b(?:this|the|following|linked) (?:recipe|page|video|post|article|link)\b|"
    r"\b(?:recipe|instructions) (?:here|at this link)\b",
    re.IGNORECASE,
)
_SELF_RESOLUTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:edit|update|solved)\b[^\n]{0,180}\b(?:figured (?:it|this) out|"
    r"turns out|the (?:answer|solution) (?:is|was)|fixed (?:it|this)|what worked)\b|"
    r"\bI (?:eventually )?(?:figured (?:it|this) out|solved (?:it|this))\b",
    re.IGNORECASE,
)
_SOURCE_ANSWER_LEAKAGE_PATTERN = re.compile(
    r"\b(?:accepted answer|answer (?:below|above)|as (?:the|an) answer (?:says|notes)|"
    r"from the answer|the answer on (?:this|that) (?:post|question))\b",
    re.IGNORECASE,
)

_CONTACT_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "email_address",
        re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"),
        "[redacted-email]",
    ),
    (
        "labelled_phone_number",
        re.compile(
            r"(?i)\b(?:phone|tel(?:ephone)?|mobile|whats\s*app)\s*[:=-]?\s*"
            r"(?:\+?\d[\d().\s-]{6,}\d)"
        ),
        "[redacted-phone]",
    ),
    (
        "social_contact_handle",
        re.compile(
            r"(?i)\b(?:instagram|telegram|twitter|contact me (?:on|at)|message me (?:on|at))"
            r"\s*[:=-]?\s*@[A-Za-z0-9_.-]{2,}"
        ),
        "[redacted-social-contact]",
    ),
    (
        "declared_name",
        re.compile(
            r"(?im)^(?:my name is|I am)\s+[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){0,2}"
            r"\s*[,.]?\s*$"
        ),
        "[redacted-declared-name]",
    ),
)


class AcquisitionError(RuntimeError):
    """The public-source acquisition could not be reproduced safely."""


class _TextParser(HTMLParser):
    _BLOCK_TAGS = {
        "address",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "li",
        "ol",
        "p",
        "pre",
        "table",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def html_to_text_exact(value: str) -> str:
    """Apply one frozen, deterministic HTML-to-text normalization."""

    parser = _TextParser()
    parser.feed(value)
    parser.close()
    text = html.unescape("".join(parser.parts)).replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def deidentify_direct_contacts(value: str, *, field: str) -> tuple[str, list[dict[str, Any]]]:
    """Redact only narrow, direct identity/contact strings and log every rule."""

    transformed = value
    log: list[dict[str, Any]] = []
    for rule_id, pattern, replacement in _CONTACT_PATTERNS:
        before_sha256 = text_sha256(transformed)
        transformed, count = pattern.subn(replacement, transformed)
        if count:
            log.append(
                {
                    "field": field,
                    "rule_id": rule_id,
                    "replacement": replacement,
                    "occurrences": count,
                    "input_sha256": before_sha256,
                    "output_sha256": text_sha256(transformed),
                }
            )
    return transformed, log


@dataclass(frozen=True)
class AcquisitionPolicy:
    fromdate: int = DEFAULT_FROMDATE
    todate: int | None = None
    target_per_family: int = 30
    review_reserve_per_family: int = 45
    min_prompt_chars: int = 80
    max_prompt_chars: int = 4_000
    maximum_candidates_per_source_author: int = 4
    maximum_candidates_per_source_author_family: int = 2
    near_duplicate_jaccard: float = 0.82


class PublicStackExchangeClient:
    """Small read-only client that captures a hash receipt for every response."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        inter_request_delay_seconds: float = 0.25,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=API_BASE_URL,
            timeout=timeout_seconds,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": "FlavourBench/0.1 public-human-task-acquisition",
            },
        )
        self._sleep = sleep
        self._inter_request_delay_seconds = inter_request_delay_seconds
        self.request_receipts: list[dict[str, Any]] = []

    def __enter__(self) -> PublicStackExchangeClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self._owns_client:
            self._client.close()

    def _get(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        response: httpx.Response | None = None
        payload: dict[str, Any] | None = None
        for attempt in range(4):
            response = self._client.get(path, params=params)
            try:
                candidate_payload = response.json()
            except json.JSONDecodeError as exc:
                raise AcquisitionError("Stack Exchange API returned invalid JSON") from exc
            if not isinstance(candidate_payload, dict):
                raise AcquisitionError("Stack Exchange API returned a non-object payload")
            if candidate_payload.get("error_id") == 502 and attempt < 3:
                self._sleep(float(5 * (attempt + 1)))
                continue
            payload = candidate_payload
            break
        assert response is not None and payload is not None
        response.raise_for_status()
        raw = response.content
        if payload.get("error_id"):
            raise AcquisitionError(
                f"Stack Exchange API error {payload.get('error_id')}: {payload.get('error_name')}"
            )
        receipt = {
            "ordinal": len(self.request_receipts) + 1,
            "method": "GET",
            "path": path,
            "parameters": {
                key: params[key] for key in sorted(params) if key not in {"key", "access_token"}
            },
            "authentication": "public_unauthenticated",
            "status_code": response.status_code,
            "response_body_sha256": hashlib.sha256(raw).hexdigest(),
            "response_bytes": len(raw),
            "item_count": len(payload.get("items", []))
            if isinstance(payload.get("items"), list)
            else None,
            "quota_max": payload.get("quota_max"),
            "quota_remaining": payload.get("quota_remaining"),
            "backoff_seconds": payload.get("backoff"),
        }
        self.request_receipts.append(receipt)
        backoff = payload.get("backoff")
        if isinstance(backoff, int) and backoff > 0:
            self._sleep(float(backoff + 1))
        elif self._inter_request_delay_seconds > 0:
            self._sleep(self._inter_request_delay_seconds)
        return payload

    def fetch_questions(self, *, fromdate: int, todate: int) -> list[dict[str, Any]]:
        questions: dict[int, dict[str, Any]] = {}
        page = 1
        while True:
            payload = self._get(
                "/questions",
                {
                    "site": API_SITE,
                    "fromdate": fromdate,
                    "todate": todate,
                    "page": page,
                    "pagesize": 100,
                    "order": "asc",
                    "sort": "creation",
                    "filter": "withbody",
                },
            )
            items = payload.get("items")
            if not isinstance(items, list):
                raise AcquisitionError("question response has no item list")
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("question_id"), int):
                    raise AcquisitionError("question response contains an invalid item")
                questions[int(item["question_id"])] = item
            if not payload.get("has_more"):
                break
            page += 1
        return [questions[key] for key in sorted(questions)]

    def fetch_revisions(self, question_ids: Sequence[int]) -> dict[int, list[dict[str, Any]]]:
        revisions: dict[int, list[dict[str, Any]]] = defaultdict(list)
        # The endpoint accepts 100 post IDs, but a 100-post batch can itself
        # contain more than one hundred revisions.  The API has returned a 400
        # for the second page of such a request in production.  Twenty IDs keep
        # each response below the page limit for this corpus while remaining
        # well inside the public request quota.
        for start in range(0, len(question_ids), 20):
            batch = question_ids[start : start + 20]
            page = 1
            while True:
                payload = self._get(
                    f"/posts/{';'.join(str(value) for value in batch)}/revisions",
                    {
                        "site": API_SITE,
                        "page": page,
                        "pagesize": 100,
                        "filter": "withbody",
                    },
                )
                items = payload.get("items")
                if not isinstance(items, list):
                    raise AcquisitionError("revision response has no item list")
                for item in items:
                    if not isinstance(item, dict) or not isinstance(item.get("post_id"), int):
                        raise AcquisitionError("revision response contains an invalid item")
                    revisions[int(item["post_id"])].append(item)
                if not payload.get("has_more"):
                    break
                page += 1
        return {key: revisions.get(key, []) for key in question_ids}


def _latest_content_revision(revisions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    content_revisions = [
        revision
        for revision in revisions
        if revision.get("revision_type") == "single_user"
        and isinstance(revision.get("revision_number"), int)
        and isinstance(revision.get("revision_guid"), str)
    ]
    if not content_revisions:
        return None
    return max(
        content_revisions,
        key=lambda revision: (
            int(revision["revision_number"]),
            int(revision.get("creation_date") or 0),
            str(revision["revision_guid"]),
        ),
    )


def _latest_revision_field(
    revisions: Sequence[Mapping[str, Any]], field: str
) -> Mapping[str, Any] | None:
    candidates = [
        revision
        for revision in revisions
        if revision.get("revision_type") == "single_user"
        and isinstance(revision.get("revision_number"), int)
        and isinstance(revision.get(field), str)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda revision: (
            int(revision["revision_number"]),
            int(revision.get("creation_date") or 0),
            str(revision.get("revision_guid") or ""),
        ),
    )


def _source_author(question: Mapping[str, Any]) -> dict[str, Any]:
    owner = question.get("owner")
    if not isinstance(owner, Mapping):
        return {
            "display_name": "community_or_unavailable",
            "profile_url": None,
            "source_user_id": None,
            "attribution_basis": "source_page_and_corpus",
        }
    return {
        "display_name": html.unescape(str(owner.get("display_name") or "unknown")),
        "profile_url": owner.get("link") if isinstance(owner.get("link"), str) else None,
        "source_user_id": owner.get("user_id") if isinstance(owner.get("user_id"), int) else None,
        "attribution_basis": "license_required_source_attribution",
    }


def _author_key(author: Mapping[str, Any]) -> str:
    user_id = author.get("source_user_id")
    if isinstance(user_id, int):
        return f"stackexchange-user:{user_id}"
    return "unavailable:" + canonical_sha256(
        {
            "display_name": author.get("display_name"),
            "profile_url": author.get("profile_url"),
        }
    )


def _family_scores(question: Mapping[str, Any], prompt: str) -> dict[str, int]:
    tags = {str(tag).casefold() for tag in question.get("tags", [])}
    title = html_to_text_exact(str(question.get("title") or ""))
    scores: dict[str, int] = {}
    for family in FAMILIES:
        tag_score = sum(
            weight for tag, weight in _FAMILY_TAG_WEIGHTS[family].items() if tag in tags
        )
        title_score = 3 if _FAMILY_TEXT_PATTERNS[family].search(title) else 0
        body_score = 1 if _FAMILY_TEXT_PATTERNS[family].search(prompt) else 0
        scores[family] = tag_score + title_score + body_score
    return scores


def _provisional_family(scores: Mapping[str, int]) -> tuple[str | None, list[str]]:
    maximum = max(scores.values(), default=0)
    if maximum <= 0:
        return None, []
    tied = sorted(family for family, score in scores.items() if score == maximum)
    # A tie remains unresolved rather than being silently broken by family order.
    return (tied[0], tied) if len(tied) == 1 else (None, tied)


def _screen(
    screen_id: str,
    decision: Literal["pass", "review", "exclude"],
    signals: Sequence[str],
) -> dict[str, Any]:
    return {
        "screen_id": screen_id,
        "decision": decision,
        "signals": sorted(set(signals)),
        "basis": "deterministic_rule",
        "human_ground_truth": False,
    }


def _content_words(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(re.findall(r"[a-z0-9]+", normalized))


def _trigrams(value: str) -> set[tuple[str, ...]]:
    words = _content_words(value)
    if not words:
        return set()
    width = min(3, len(words))
    return {tuple(words[index : index + width]) for index in range(max(1, len(words) - width + 1))}


def _jaccard(first: set[tuple[str, ...]], second: set[tuple[str, ...]]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _base_candidate(
    question: Mapping[str, Any],
    revisions: Sequence[Mapping[str, Any]],
    *,
    policy: AcquisitionPolicy,
) -> dict[str, Any]:
    question_id = int(question["question_id"])
    revision = _latest_content_revision(revisions)
    raw_title = str(question.get("title") or "")
    raw_body = str(question.get("body") or "")
    title_text = html_to_text_exact(raw_title)
    body_text = html_to_text_exact(raw_body)
    title_deidentified, title_log = deidentify_direct_contacts(title_text, field="title")
    body_deidentified, body_log = deidentify_direct_contacts(body_text, field="body")
    transformation_log = title_log + body_log
    prompt = f"{title_deidentified}\n\n{body_deidentified}".strip()
    family_scores = _family_scores(question, prompt)
    family, tied_families = _provisional_family(family_scores)
    tags = sorted(str(tag).casefold() for tag in question.get("tags", []))

    specialist_signals = [tag for tag in tags if tag in _SPECIALIST_TAGS]
    specialist_signals.extend(
        name for name, pattern in _SPECIALIST_PATTERNS.items() if pattern.search(prompt)
    )
    external_links = re.findall(r"<a\b[^>]*\bhref=", raw_body, re.IGNORECASE)
    context_signals: list[str] = []
    if _IMAGE_CONTEXT_PATTERN.search(raw_body) or _IMAGE_CONTEXT_PATTERN.search(prompt):
        context_signals.append("image_or_visual_dependency")
    if external_links and _LINK_CONTEXT_PATTERN.search(prompt):
        context_signals.append("linked_context_dependency")
    self_resolution_signals = (
        ["embedded_resolution_language"] if _SELF_RESOLUTION_PATTERN.search(prompt) else []
    )
    answer_leakage_signals = (
        ["source_answer_referenced_in_question"]
        if _SOURCE_ANSWER_LEAKAGE_PATTERN.search(prompt)
        else []
    )

    revision_signals: list[str] = []
    if revision is None:
        revision_signals.append("no_content_revision_returned")
        revision_guid = None
        revision_number = None
        revision_created = None
        revision_license = None
    else:
        revision_guid = str(revision["revision_guid"])
        revision_number = int(revision["revision_number"])
        revision_created = _utc_iso(int(revision["creation_date"]))
        license_revision = _latest_revision_field(revisions, "content_license")
        revision_license = (
            license_revision.get("content_license") if license_revision is not None else None
        )
        body_revision = _latest_revision_field(revisions, "body")
        title_revision = _latest_revision_field(revisions, "title")
        if body_revision is None or html_to_text_exact(str(body_revision["body"])) != body_text:
            revision_signals.append("current_body_does_not_match_terminal_revision")
        if title_revision is None or html_to_text_exact(str(title_revision["title"])) != title_text:
            revision_signals.append("current_title_does_not_match_terminal_revision")

    question_license = str(question.get("content_license") or "") or None
    revision_license = str(revision_license or "") or None
    observed_licenses = {value for value in (question_license, revision_license) if value}
    license_signals = [
        value for value in sorted(observed_licenses) if value not in ACCEPTED_LICENSES
    ]
    if not observed_licenses:
        license_signals.append("no_api_reported_content_license")
    effective_license = (
        next(iter(observed_licenses))
        if len(observed_licenses) == 1 and not license_signals
        else None
    )
    length_signals: list[str] = []
    if len(prompt) < policy.min_prompt_chars:
        length_signals.append("prompt_too_short")
    if len(prompt) > policy.max_prompt_chars:
        length_signals.append("prompt_exceeds_current_ui_limit")
    lifecycle_signals = ["closed_question"] if question.get("closed_date") is not None else []
    family_signals = []
    if family is None:
        family_signals.append(
            "family_score_tie:" + ",".join(tied_families) if tied_families else "no_family_signal"
        )

    screens = [
        _screen(
            "redistributable_license",
            "pass" if not license_signals else "exclude",
            license_signals,
        ),
        _screen(
            "specialist_track_exclusion",
            "pass" if not specialist_signals else "exclude",
            specialist_signals,
        ),
        _screen(
            "self_contained_visual_and_link_context",
            "pass" if not context_signals else "review",
            context_signals,
        ),
        _screen(
            "embedded_self_resolution",
            "pass" if not self_resolution_signals else "review",
            self_resolution_signals,
        ),
        _screen(
            "source_answer_leakage",
            "pass" if not answer_leakage_signals else "review",
            answer_leakage_signals,
        ),
        _screen(
            "source_revision_integrity",
            "pass" if not revision_signals else "exclude",
            revision_signals,
        ),
        _screen(
            "current_reviewer_ui_length",
            "pass" if not length_signals else "exclude",
            length_signals,
        ),
        _screen(
            "question_lifecycle",
            "pass" if not lifecycle_signals else "review",
            lifecycle_signals,
        ),
        _screen(
            "provisional_family_assignment",
            "pass" if not family_signals else "review",
            family_signals,
        ),
        _screen("within_bundle_duplicate", "pass", []),
    ]
    author = _source_author(question)
    candidate_id = str(
        uuid.uuid5(
            _CANDIDATE_NAMESPACE,
            f"{API_SITE}:{question_id}:{revision_guid or 'unresolved'}",
        )
    )
    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "source_origin": "exact_public_human_question",
        "synthetic": False,
        "model_authored": False,
        "model_rewritten": False,
        "prompt": prompt,
        "prompt_sha256": text_sha256(prompt),
        "provisional_family": family,
        "provisional_family_candidates": tied_families,
        "family_scores": family_scores,
        "source": {
            "corpus": SOURCE_CORPUS,
            "site": API_SITE,
            "question_id": question_id,
            "url": str(question.get("link") or ""),
            "created_utc": _utc_iso(int(question["creation_date"])),
            "last_activity_utc": _utc_iso(int(question["last_activity_date"])),
            "last_edit_utc": _utc_iso(int(question["last_edit_date"]))
            if isinstance(question.get("last_edit_date"), int)
            else None,
            "revision_guid": revision_guid,
            "revision_number": revision_number,
            "revision_created_utc": revision_created,
            "content_license": effective_license,
            "question_content_license": question_license,
            "revision_content_license": revision_license,
            "licensing_url": LICENSING_URL,
            "attribution": author,
            "tags": tags,
            "score_at_acquisition": int(question.get("score") or 0),
            "answer_count_at_acquisition": int(question.get("answer_count") or 0),
            "has_accepted_answer_at_acquisition": isinstance(
                question.get("accepted_answer_id"), int
            ),
            "source_answer_ids_stored": False,
            "source_answer_payload_requested": False,
            "title_html_source_sha256": text_sha256(raw_title),
            "body_html_source_sha256": text_sha256(raw_body),
            "title_text_source_sha256": text_sha256(title_text),
            "body_text_source_sha256": text_sha256(body_text),
            "title_text_deidentified": title_deidentified,
            "body_text_deidentified": body_deidentified,
        },
        "transformations": {
            "html_to_text": NORMALIZATION_VERSION,
            "deidentification": DEIDENTIFICATION_VERSION,
            "log": transformation_log,
            "unlogged_rewriting_performed": False,
        },
        "provisional_screens": screens,
        "claim_boundary": {
            "screen_is_human_task_validation": False,
            "family_is_human_confirmed": False,
            "contamination_free_claimed": False,
            "public_source_may_have_entered_model_training": True,
            "rank_eligible": False,
        },
    }
    record["candidate_record_sha256"] = canonical_sha256(record)
    return record


def _replace_duplicate_screen(
    candidate: dict[str, Any],
    *,
    decision: Literal["pass", "review", "exclude"],
    signals: Sequence[str],
) -> None:
    screens = candidate["provisional_screens"]
    index = next(
        index
        for index, screen in enumerate(screens)
        if screen["screen_id"] == "within_bundle_duplicate"
    )
    screens[index] = _screen("within_bundle_duplicate", decision, signals)
    candidate.pop("candidate_record_sha256", None)
    candidate["candidate_record_sha256"] = canonical_sha256(candidate)


def _mark_duplicates(candidates: list[dict[str, Any]], *, threshold: float) -> None:
    # Keep the newest source revision and flag older matches for human inspection.
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate["source"]["created_utc"],
            candidate["source"]["question_id"],
        ),
        reverse=True,
    )
    accepted: list[tuple[dict[str, Any], set[tuple[str, ...]]]] = []
    exact_seen: dict[tuple[str, ...], str] = {}
    for candidate in ordered:
        words = _content_words(candidate["prompt"])
        exact = exact_seen.get(words)
        if exact is not None:
            _replace_duplicate_screen(
                candidate,
                decision="exclude",
                signals=[f"exact_normalized_duplicate_of:{exact}"],
            )
            continue
        features = _trigrams(candidate["prompt"])
        near: tuple[str, float] | None = None
        if len(features) >= 4:
            for prior, prior_features in accepted:
                similarity = _jaccard(features, prior_features)
                if similarity > threshold and (near is None or similarity > near[1]):
                    near = (prior["candidate_id"], similarity)
        if near is not None:
            _replace_duplicate_screen(
                candidate,
                decision="review",
                signals=[f"near_duplicate_of:{near[0]}:jaccard={near[1]:.6f}"],
            )
        else:
            exact_seen[words] = candidate["candidate_id"]
            accepted.append((candidate, features))


def _provisional_pass(candidate: Mapping[str, Any]) -> bool:
    return all(
        screen.get("decision") == "pass" for screen in candidate.get("provisional_screens", [])
    )


def _eligible_for_blind_family_review(candidate: Mapping[str, Any]) -> bool:
    """Allow unresolved family labels, but no other review or exclusion flag."""

    family_candidates = candidate.get("provisional_family_candidates")
    if not isinstance(family_candidates, list) or not family_candidates:
        return False
    return all(
        screen.get("decision") == "pass"
        or screen.get("screen_id") == "provisional_family_assignment"
        for screen in candidate.get("provisional_screens", [])
    )


def build_candidate_bundle(
    *,
    questions: Sequence[Mapping[str, Any]],
    revisions_by_question: Mapping[int, Sequence[Mapping[str, Any]]],
    policy: AcquisitionPolicy,
    retrieved_utc: str,
    request_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = [
        _base_candidate(
            question,
            revisions_by_question.get(int(question["question_id"]), []),
            policy=policy,
        )
        for question in sorted(questions, key=lambda value: int(value["question_id"]))
    ]
    _mark_duplicates(candidates, threshold=policy.near_duplicate_jaccard)
    candidates.sort(key=lambda candidate: int(candidate["source"]["question_id"]))
    provisional_pass = [candidate for candidate in candidates if _provisional_pass(candidate)]
    blind_family_review_eligible = [
        candidate for candidate in candidates if _eligible_for_blind_family_review(candidate)
    ]
    family_counts = Counter(str(candidate["provisional_family"]) for candidate in provisional_pass)
    screen_counts: Counter[tuple[str, str]] = Counter()
    for candidate in candidates:
        for screen in candidate["provisional_screens"]:
            screen_counts[(screen["screen_id"], screen["decision"])] += 1
    unique_authors = {_author_key(candidate["source"]["attribution"]) for candidate in candidates}
    bundle: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA,
        "artifact_role": "prospective_human_authored_task_candidates_not_validation",
        "benchmark": "FlavourBench",
        "retrieved_utc": retrieved_utc,
        "source_policy": {
            "exact_human_authored_public_questions_only": True,
            "model_authored_or_paraphrased_tasks": 0,
            "model_outputs_used_for_screening_or_validation": False,
            "source_answer_endpoints_requested": False,
            "source_answer_text_in_bundle": False,
            "accepted_content_licenses": sorted(ACCEPTED_LICENSES),
            "license_attribution_retained": True,
            "normalization_version": NORMALIZATION_VERSION,
            "deidentification_version": DEIDENTIFICATION_VERSION,
            "contact_redactions_only": True,
        },
        "acquisition": {
            "api_base_url": API_BASE_URL,
            "site": API_SITE,
            "question_endpoint": "/questions",
            "revision_endpoint": "/posts/{ids}/revisions",
            "answer_endpoint_requests": 0,
            "fromdate_utc": _utc_iso(policy.fromdate),
            "todate_utc": _utc_iso(int(policy.todate or 0)),
            "request_receipts_sha256": canonical_sha256(list(request_receipts)),
            "request_count": len(request_receipts),
        },
        "screen_policy": {
            "version": SCREEN_POLICY_VERSION,
            "screens_are_provisional": True,
            "screens_are_human_ground_truth": False,
            "specialist_tags": sorted(_SPECIALIST_TAGS),
            "specialist_patterns": {
                name: pattern.pattern for name, pattern in _SPECIALIST_PATTERNS.items()
            },
            "near_duplicate_jaccard_threshold": policy.near_duplicate_jaccard,
            "family_tag_weights": _FAMILY_TAG_WEIGHTS,
            "family_text_patterns": {
                family: pattern.pattern for family, pattern in _FAMILY_TEXT_PATTERNS.items()
            },
        },
        "target": {
            "tasks": policy.target_per_family * len(FAMILIES),
            "per_family": policy.target_per_family,
            "candidate_review_reserve_per_family": policy.review_reserve_per_family,
        },
        "counts": {
            "questions_retrieved": len(questions),
            "candidate_records": len(candidates),
            "provisional_screen_pass": len(provisional_pass),
            "eligible_for_blind_family_review": len(blind_family_review_eligible),
            "provisional_screen_pass_by_family": {
                family: family_counts[family] for family in FAMILIES
            },
            "unique_attributed_source_authors": len(unique_authors),
            "synthetic_tasks": 0,
            "model_authored_tasks": 0,
            "source_answer_payloads": 0,
            "sealed_human_task_validations": 0,
            "screen_decisions": {
                screen_id: {
                    decision: screen_counts[(screen_id, decision)]
                    for decision in ("pass", "review", "exclude")
                }
                for screen_id in sorted({key[0] for key in screen_counts})
            },
        },
        "candidates": candidates,
        "claim_boundary": {
            "official_task_bank": False,
            "rank_eligible": False,
            "human_validated": False,
            "contamination_free": False,
            "public_source_contamination_risk_remains": True,
            "automated_screen_failures_require_human_confirmation": True,
        },
    }
    bundle["artifact_sha256"] = canonical_sha256(bundle)
    return bundle


def _review_selection(
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: AcquisitionPolicy,
) -> list[tuple[Mapping[str, Any], str]]:
    eligible = [
        candidate for candidate in candidates if _eligible_for_blind_family_review(candidate)
    ]
    eligible.sort(
        key=lambda candidate: (
            str(candidate["source"]["created_utc"]),
            int(candidate["source"]["question_id"]),
        ),
        reverse=True,
    )
    selected: list[tuple[Mapping[str, Any], str]] = []
    selected_ids: set[str] = set()
    total_by_author: Counter[str] = Counter()
    family_by_author: Counter[tuple[str, str]] = Counter()
    family_counts: Counter[str] = Counter()
    # Tied automated labels remain ties. The allocated family is only a
    # scheduling stratum for qualification-matched blind review.
    while True:
        open_families = [
            family
            for family in FAMILIES
            if family_counts[family] < policy.review_reserve_per_family
        ]
        if not open_families:
            break
        open_families.sort(key=lambda family: (family_counts[family], FAMILIES.index(family)))
        made_progress = False
        for family in open_families:
            for candidate in eligible:
                if candidate["candidate_id"] in selected_ids:
                    continue
                if family not in candidate["provisional_family_candidates"]:
                    continue
                author = _author_key(candidate["source"]["attribution"])
                if total_by_author[author] >= policy.maximum_candidates_per_source_author:
                    continue
                if (
                    family_by_author[(author, family)]
                    >= policy.maximum_candidates_per_source_author_family
                ):
                    continue
                selected.append((candidate, family))
                selected_ids.add(str(candidate["candidate_id"]))
                family_counts[family] += 1
                total_by_author[author] += 1
                family_by_author[(author, family)] += 1
                made_progress = True
                break
        if not made_progress:
            break
    return sorted(
        selected,
        key=lambda item: (
            FAMILIES.index(item[1]),
            str(item[0]["source"]["created_utc"]),
            int(item[0]["source"]["question_id"]),
        ),
    )


def build_assignment_artifact(
    bundle: Mapping[str, Any], *, policy: AcquisitionPolicy
) -> dict[str, Any]:
    selected = _review_selection(bundle["candidates"], policy=policy)
    selected_counts = Counter(allocation_family for _, allocation_family in selected)
    target_supported = all(
        selected_counts[family] >= policy.target_per_family for family in FAMILIES
    )
    assignment_rows: list[dict[str, Any]] = []
    event_templates: list[dict[str, Any]] = []
    for ordinal, (candidate, family) in enumerate(selected, start=1):
        assignment_rows.append(
            {
                "assignment_ordinal": ordinal,
                "candidate_id": candidate["candidate_id"],
                "phase": "blind_validity",
                "prompt": candidate["prompt"],
                "prompt_sha256": candidate["prompt_sha256"],
                "allocation_family_hidden_from_blind_reviewer": family,
                "provisional_family_candidates_hidden_from_blind_reviewer": candidate[
                    "provisional_family_candidates"
                ],
                "source_metadata_visible_after_blind_decision": candidate["source"],
                "model_outputs_visible": False,
                "source_answer_text_visible": False,
                "rank_eligible": False,
            }
        )
        event_payload = {
            "schema_version": "flavourbench-licensed-public-task-candidate-v1",
            "family": family,
            "prompt": candidate["prompt"],
            "prompt_sha256": candidate["prompt_sha256"],
            "construct_blueprint_sha256": None,
            "construct_cell_id": None,
            "difficulty_tier": None,
            "subskills": [],
            "explicit_constraints": [],
            "unacceptable_outcomes": [],
            "acceptable_solution_outline": None,
            "objective_validator_possible": None,
            "validator_notes": None,
            "rights_basis": "public_cc_by_sa_4_0",
            "human_authorship_attestation": None,
            "no_personal_data_attestation": None,
            "research_use_consent": None,
            "author_reviewer_id": None,
            "licensed_source_candidate": True,
            "source_candidate_record_sha256": candidate["candidate_record_sha256"],
            "source": candidate["source"],
            "candidate_record_sha256": canonical_sha256(
                {
                    "source_candidate_record_sha256": candidate["candidate_record_sha256"],
                    "prompt_sha256": candidate["prompt_sha256"],
                    "family": family,
                    "rights_basis": "public_cc_by_sa_4_0",
                }
            ),
            "rank_eligible": False,
        }
        event_templates.append(
            {
                "entity_type": "task_candidate",
                "entity_id": candidate["candidate_id"],
                "event_type": "task_candidate_submitted",
                "payload_json": event_payload,
                "database_mutation_authorized": False,
            }
        )

    reserve_tasks = len(selected)
    target_tasks = policy.target_per_family * len(FAMILIES)
    workload = {
        "candidate_source_validation_ballots_remaining": reserve_tasks * 2,
        "candidate_adjudications_remaining": reserve_tasks,
        "final_task_validator_contract_reviews_remaining": target_tasks,
        "final_task_contamination_audits_remaining": target_tasks,
        "smallest_viable_role_assignments_if_no_candidate_attrition": target_tasks * 5,
        "total_role_assignments_remaining_if_target_is_reached": (
            reserve_tasks * 3 + target_tasks * 2
        ),
        "operational_reserve_plan_role_assignments": reserve_tasks * 3 + target_tasks * 2,
        "distinct_people_required_on_each_admitted_task": 5,
        "smallest_viable_distinct_enrolled_reviewers_with_cross_task_rotation": 5,
        "recommended_minimum_distinct_enrolled_reviewers": 12,
    }
    artifact: dict[str, Any] = {
        "schema_version": ASSIGNMENT_SCHEMA,
        "artifact_role": "source_only_blind_validation_import_and_assignment",
        "source_candidate_bundle_sha256": bundle["artifact_sha256"],
        "selection_policy": {
            "candidate_review_reserve_per_family": policy.review_reserve_per_family,
            "target_per_family": policy.target_per_family,
            "newest_source_revision_first": True,
            "maximum_candidates_per_source_author": (policy.maximum_candidates_per_source_author),
            "maximum_candidates_per_source_author_family": (
                policy.maximum_candidates_per_source_author_family
            ),
            "automated_family_labels_are_provisional": True,
            "allocation_family_is_review_scheduling_not_ground_truth": True,
        },
        "counts": {
            "selected_for_blind_human_review": len(selected),
            "selected_by_allocation_family": {
                family: selected_counts[family] for family in FAMILIES
            },
            "target_tasks": target_tasks,
            "target_supported_by_provisional_reserve": target_supported,
            "human_validation_ballots_recorded": 0,
            "synthetic_tasks": 0,
            "model_outputs": 0,
        },
        "assignment_rows": assignment_rows,
        "run_event_import_templates": event_templates,
        "human_workload_remaining": workload,
        "reviewer_identity_compatibility": {
            "season_scoped_identity_binding_required": True,
            "season_scoped_family_admission_required": True,
            "person_commitments_must_be_distinct_within_task": True,
            "source_author_attribution_is_not_a_0030_identity_binding": True,
            "two_source_validators_required": True,
            "one_adjudicator_required": True,
            "one_validator_contract_reviewer_required": True,
            "one_contamination_auditor_required": True,
        },
        "ui_compatibility": {
            "existing_blind_task_candidate_view": "compatible",
            "existing_source_reconciliation_view": "requires_public_source_metadata_adapter",
            "existing_confirmatory_bank_import": "fail_closed_incompatible",
            "reason": (
                "The current bank import requires an enrolled original contributor and an "
                "original-author protocol acceptance. A licensed public source author must not "
                "be represented as that contributor."
            ),
        },
        "claim_boundary": {
            "assignment_is_human_validation_evidence": False,
            "source_answers_used": False,
            "model_outputs_used": False,
            "database_import_authorized": False,
            "official": False,
            "rank_eligible": False,
        },
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def build_acquisition_receipt(
    *,
    bundle: Mapping[str, Any],
    assignment: Mapping[str, Any],
    request_receipts: Sequence[Mapping[str, Any]],
    retrieved_utc: str,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "acquisition_mode": "real_public_api",
        "provider": "Stack Exchange API 2.3",
        "site": API_SITE,
        "authentication": "public_unauthenticated",
        "retrieved_utc": retrieved_utc,
        "candidate_bundle_sha256": bundle["artifact_sha256"],
        "assignment_artifact_sha256": assignment["artifact_sha256"],
        "requests": list(request_receipts),
        "request_count": len(request_receipts),
        "question_endpoint_requests": sum(
            receipt.get("path") == "/questions" for receipt in request_receipts
        ),
        "revision_endpoint_requests": sum(
            str(receipt.get("path", "")).endswith("/revisions") for receipt in request_receipts
        ),
        "answer_endpoint_requests": 0,
        "paid_provider_calls": 0,
        "epicure_calls": 0,
        "model_calls": 0,
        "synthetic_tasks": 0,
        "claim_boundary": {
            "receipt_proves_api_transfer": True,
            "receipt_proves_human_task_validity": False,
            "receipt_proves_decontamination": False,
            "rank_eligible": False,
        },
    }
    receipt["artifact_sha256"] = canonical_sha256(receipt)
    return receipt


def verify_artifact(document: Mapping[str, Any], *, schema_version: str) -> None:
    if document.get("schema_version") != schema_version:
        raise AcquisitionError(f"unexpected artifact schema: {document.get('schema_version')}")
    embedded = document.get("artifact_sha256")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if embedded != canonical_sha256(payload):
        raise AcquisitionError("content-addressed artifact hash does not verify")


def _write_content_addressed(document: Mapping[str, Any], *, output_dir: Path, prefix: str) -> Path:
    verify_artifact(document, schema_version=str(document["schema_version"]))
    digest = str(document["artifact_sha256"])
    rendered = (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{prefix}-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise AcquisitionError("content-addressed destination has conflicting bytes")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def acquire(
    *,
    policy: AcquisitionPolicy,
    output_dir: Path,
    now: datetime | None = None,
    client: PublicStackExchangeClient | None = None,
) -> tuple[Path, Path, Path]:
    observed_now = (now or datetime.now(UTC)).astimezone(UTC)
    todate = policy.todate or int(observed_now.timestamp())
    resolved_policy = AcquisitionPolicy(
        fromdate=policy.fromdate,
        todate=todate,
        target_per_family=policy.target_per_family,
        review_reserve_per_family=policy.review_reserve_per_family,
        min_prompt_chars=policy.min_prompt_chars,
        max_prompt_chars=policy.max_prompt_chars,
        maximum_candidates_per_source_author=policy.maximum_candidates_per_source_author,
        maximum_candidates_per_source_author_family=(
            policy.maximum_candidates_per_source_author_family
        ),
        near_duplicate_jaccard=policy.near_duplicate_jaccard,
    )
    retrieved_utc = observed_now.isoformat().replace("+00:00", "Z")
    if client is None:
        with PublicStackExchangeClient() as owned_client:
            questions = owned_client.fetch_questions(
                fromdate=resolved_policy.fromdate,
                todate=todate,
            )
            revisions = owned_client.fetch_revisions(
                [int(question["question_id"]) for question in questions]
            )
            request_receipts = list(owned_client.request_receipts)
    else:
        questions = client.fetch_questions(
            fromdate=resolved_policy.fromdate,
            todate=todate,
        )
        revisions = client.fetch_revisions([int(question["question_id"]) for question in questions])
        request_receipts = list(client.request_receipts)
    bundle = build_candidate_bundle(
        questions=questions,
        revisions_by_question=revisions,
        policy=resolved_policy,
        retrieved_utc=retrieved_utc,
        request_receipts=request_receipts,
    )
    assignment = build_assignment_artifact(bundle, policy=resolved_policy)
    receipt = build_acquisition_receipt(
        bundle=bundle,
        assignment=assignment,
        request_receipts=request_receipts,
        retrieved_utc=retrieved_utc,
    )
    bundle_path = _write_content_addressed(
        bundle,
        output_dir=output_dir,
        prefix="public-human-task-candidates",
    )
    assignment_path = _write_content_addressed(
        assignment,
        output_dir=output_dir,
        prefix="public-human-task-review-assignment",
    )
    receipt_path = _write_content_addressed(
        receipt,
        output_dir=output_dir,
        prefix="public-api-acquisition-receipt",
    )
    return bundle_path, assignment_path, receipt_path


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/season1/prospective-task-acquisition-v1"),
    )
    parser.add_argument("--fromdate", type=int, default=DEFAULT_FROMDATE)
    parser.add_argument("--todate", type=int)
    parser.add_argument("--target-per-family", type=int, default=30)
    parser.add_argument("--review-reserve-per-family", type=int, default=45)
    args = parser.parse_args(argv)
    policy = AcquisitionPolicy(
        fromdate=args.fromdate,
        todate=args.todate,
        target_per_family=args.target_per_family,
        review_reserve_per_family=args.review_reserve_per_family,
    )
    bundle_path, assignment_path, receipt_path = acquire(
        policy=policy,
        output_dir=args.output_dir,
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "bundle": str(bundle_path),
                "bundle_sha256": bundle["artifact_sha256"],
                "assignment": str(assignment_path),
                "assignment_sha256": assignment["artifact_sha256"],
                "receipt": str(receipt_path),
                "counts": bundle["counts"],
                "review_counts": assignment["counts"],
                "human_workload_remaining": assignment["human_workload_remaining"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
