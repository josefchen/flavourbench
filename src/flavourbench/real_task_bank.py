from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx

STACKEXCHANGE_API = "https://api.stackexchange.com/2.3"
STACKEXCHANGE_SITE = "cooking"
STACKEXCHANGE_TERMS_URL = "https://stackoverflow.com/help/licensing"
DEFAULT_FROMDATE = 1_672_531_200  # 2023-01-01T00:00:00Z
RECENT_FROMDATE = 1_735_689_600  # 2025-01-01T00:00:00Z
OBSERVED_THROUGH = 1_784_160_000  # 2026-07-16T00:00:00Z

FAMILY_TAGS: dict[str, tuple[str, ...]] = {
    "substitution": (
        "substitutions",
        "ingredient-selection",
        "vegan",
        "vegetarian",
        "gluten-free",
        "dairy-free",
        "sugar-free",
    ),
    "composition": (
        "flavor",
        "spices",
        "seasoning",
        "pairing",
        "menu-planning",
        "ingredients",
        "sauce",
        "marinade",
        "herbs",
        "salt",
        "vinegar",
        "stock",
        "wine",
    ),
    "cookability": (
        "baking",
        "equipment",
        "texture",
        "recipe-scaling",
        "cooking-time",
        "measurements",
        "dough",
        "oven",
    ),
    "evidence": (
        "food-science",
        "chemistry",
        "molecular-gastronomy",
        "fermentation",
        "emulsion",
        "temperature",
    ),
}

FAMILY_TITLE_PATTERNS: dict[str, str] = {
    "substitution": (
        r"\b(substitut(?:e|es|ed|ing|ion)|replac(?:e|ement|ing)|alternative to|"
        r"instead of|swap(?:ping)?|without|omit(?:ting)?|leave (?:it|them) out|"
        r"vegan|vegetarian|gluten[- ]free|dairy[- ]free)\b"
    ),
    "composition": (
        r"\b(flavo(?:u)?r|taste|season(?:ing)?|spic(?:e|es|y)|herb|pair(?:ing)?|"
        r"combin(?:e|ation)|complement|balance|blend|marinade|sauce|aroma|"
        r"savoury|savory|sweetness|acidity|bitter|umami)\b"
    ),
    "cookability": (
        r"\b(recipe|cook(?:ing)?|bake|baking|oven|pan|equipment|method|technique|"
        r"time|temperature|scale|batch|dough|texture|consistency|process|prepare)\b"
    ),
    "evidence": (
        r"\b(why|science|chemistr(?:y|ies)|reaction|mechanism|molecule|protein|"
        r"starch|gluten|emulsi(?:on|fy|fied)|ferment(?:ation|ing)?|temperature|"
        r"carameli[sz]|maillard|crystal|acid|alkali|pH|curdl|coagulat|denatur|gelati)"
    ),
}

EXCLUDED_CONTENT_PATTERN = re.compile(
    r"\b(allerg(?:y|ies|ic|en|ens)|intoleran(?:ce|t)|anaphyla(?:xis|ctic)|celiac|coeliac|"
    r"medical|nutrition(?:al)?|calorie|safe to eat|food safety|food poisoning|"
    r"botulism|pathogen|authentic(?:ity)?|religious|kosher|halal)\b",
    re.IGNORECASE,
)

EXCLUDED_TAGS = frozenset(
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


class _PlainTextParser(HTMLParser):
    _BLOCK_TAGS = {
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


def html_to_text(value: str) -> str:
    parser = _PlainTextParser()
    parser.feed(value)
    parser.close()
    text = html.unescape("".join(parser.parts)).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SelectionPolicy:
    per_family: int = 30
    min_recent_per_family: int = 10
    min_question_chars: int = 40
    max_question_chars: int = 6_000
    min_reference_chars: int = 40
    fromdate: int = DEFAULT_FROMDATE
    recent_fromdate: int = RECENT_FROMDATE
    observed_through: int = OBSERVED_THROUGH


class StackExchangeClient:
    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self._client = httpx.Client(
            base_url=STACKEXCHANGE_API,
            timeout=timeout_seconds,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": "FlavourBench/0.1 real-user-task-import",
            },
        )

    def __enter__(self) -> StackExchangeClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self._client.close()

    def _get(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        response = self._client.get(path, params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Stack Exchange API returned a non-object payload")
        backoff = payload.get("backoff")
        if isinstance(backoff, int) and backoff > 0:
            time.sleep(backoff)
        if payload.get("error_id"):
            raise RuntimeError(
                f"Stack Exchange API error {payload.get('error_id')}: {payload.get('error_name')}"
            )
        return payload

    def fetch_questions(self, *, tag: str | None = None, fromdate: int) -> list[dict[str, Any]]:
        page = 1
        questions: list[dict[str, Any]] = []
        while True:
            params: dict[str, Any] = {
                "site": STACKEXCHANGE_SITE,
                "fromdate": fromdate,
                "page": page,
                "pagesize": 100,
                "order": "desc",
                "sort": "creation",
                "filter": "withbody",
            }
            if tag:
                params["tagged"] = tag
            payload = self._get(
                "/questions",
                params,
            )
            items = payload.get("items", [])
            if not isinstance(items, list):
                raise ValueError("Stack Exchange question items must be a list")
            questions.extend(item for item in items if isinstance(item, dict))
            if not payload.get("has_more"):
                return questions
            page += 1

    def fetch_answers(self, answer_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        answers: dict[int, dict[str, Any]] = {}
        for start in range(0, len(answer_ids), 100):
            batch = answer_ids[start : start + 100]
            payload = self._get(
                f"/answers/{';'.join(str(answer_id) for answer_id in batch)}",
                {
                    "site": STACKEXCHANGE_SITE,
                    "pagesize": 100,
                    "filter": "withbody",
                },
            )
            items = payload.get("items", [])
            if not isinstance(items, list):
                raise ValueError("Stack Exchange answer items must be a list")
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("answer_id"), int):
                    answers[item["answer_id"]] = item
        return answers


def _owner(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("owner")
    if not isinstance(raw, Mapping):
        return {"display_name": "unknown", "user_id": None, "profile_url": None}
    return {
        "display_name": html.unescape(str(raw.get("display_name") or "unknown")),
        "user_id": raw.get("user_id") if isinstance(raw.get("user_id"), int) else None,
        "profile_url": raw.get("link") if isinstance(raw.get("link"), str) else None,
    }


def _eligible_question(
    question: Mapping[str, Any], answer: Mapping[str, Any] | None, policy: SelectionPolicy
) -> bool:
    tags = {str(tag) for tag in question.get("tags", [])}
    if tags & EXCLUDED_TAGS:
        return False
    if answer is None or not question.get("is_answered"):
        return False
    if not question.get("content_license") or not answer.get("content_license"):
        return False
    if question.get("closed_date") is not None:
        return False
    question_text = html_to_text(str(question.get("body") or ""))
    title = html.unescape(str(question.get("title") or ""))
    if EXCLUDED_CONTENT_PATTERN.search(f"{title}\n{question_text}"):
        return False
    answer_text = html_to_text(str(answer.get("body") or ""))
    if not policy.min_question_chars <= len(question_text) <= policy.max_question_chars:
        return False
    if len(answer_text) < policy.min_reference_chars:
        return False
    if not isinstance(question.get("creation_date"), int):
        return False
    return True


def _family_match_score(question: Mapping[str, Any], family: str) -> int:
    tags = {str(tag) for tag in question.get("tags", [])}
    family_tags = set(FAMILY_TAGS[family])
    title = html.unescape(str(question.get("title") or ""))
    body = html_to_text(str(question.get("body") or ""))
    pattern = FAMILY_TITLE_PATTERNS[family]
    return (
        3 * len(tags & family_tags)
        + 2 * bool(re.search(pattern, title, re.IGNORECASE))
        + bool(re.search(pattern, body, re.IGNORECASE))
    )


def _quality_score(
    question: Mapping[str, Any], answer: Mapping[str, Any], family: str, policy: SelectionPolicy
) -> float:
    question_score = max(0, int(question.get("score") or 0))
    answer_score = max(0, int(answer.get("score") or 0))
    created = int(question["creation_date"])
    observed_through = max(policy.recent_fromdate + 1, policy.observed_through)
    recency = max(
        0.0,
        min(1.0, (created - policy.fromdate) / (observed_through - policy.fromdate)),
    )
    return (
        3.0 * _family_match_score(question, family)
        + math.log1p(question_score)
        + math.log1p(answer_score)
        + 2.0 * recency
    )


def _task_record(
    question: Mapping[str, Any],
    answer: Mapping[str, Any],
    *,
    family: str,
    ordinal: int,
    policy: SelectionPolicy,
) -> dict[str, Any]:
    question_id = int(question["question_id"])
    answer_id = int(answer["answer_id"])
    title = html.unescape(str(question.get("title") or "")).strip()
    question_text = html_to_text(str(question.get("body") or ""))
    answer_text = html_to_text(str(answer.get("body") or ""))
    prompt = f"{title}\n\n{question_text}".strip()
    record: dict[str, Any] = {
        "task_id": f"fb-s0-{family[:4]}-{ordinal:03d}",
        "family": family,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "source": {
            "corpus": "Seasoned Advice (Stack Exchange)",
            "human_origin": True,
            "question_id": question_id,
            "url": str(question["link"]),
            "created_utc": utc_iso(int(question["creation_date"])),
            "last_activity_utc": utc_iso(int(question["last_activity_date"])),
            "license": str(question.get("content_license") or "unknown"),
            "author": _owner(question),
            "score_at_import": int(question.get("score") or 0),
            "answer_count_at_import": int(question.get("answer_count") or 0),
            "tags": sorted(str(tag) for tag in question.get("tags", [])),
        },
        "human_reference": {
            "accepted_answer_id": answer_id,
            "text": answer_text,
            "text_sha256": sha256_text(answer_text),
            "url": f"https://cooking.stackexchange.com/a/{answer_id}",
            "created_utc": utc_iso(int(answer["creation_date"])),
            "last_activity_utc": utc_iso(int(answer["last_activity_date"])),
            "license": str(answer.get("content_license") or "unknown"),
            "author": _owner(answer),
            "score_at_import": int(answer.get("score") or 0),
            "use": "hidden_reference_not_automatic_ground_truth",
        },
        "selection": {
            "method": "deterministic_tag_quality_recency_v1",
            "family_tag_matches": _family_match_score(question, family),
            "quality_score": round(_quality_score(question, answer, family, policy), 6),
            "recent": int(question["creation_date"]) >= policy.recent_fromdate,
            "review_status": "source_verified_pending_culinary_review",
        },
        "specialist_track_excluded": True,
    }
    record["record_sha256"] = sha256_json(record)
    return record


def select_tasks(
    questions: Iterable[Mapping[str, Any]],
    answers: Mapping[int, Mapping[str, Any]],
    *,
    policy: SelectionPolicy,
) -> list[dict[str, Any]]:
    question_map: dict[int, Mapping[str, Any]] = {}
    for question in questions:
        question_id = question.get("question_id")
        answer_id = question.get("accepted_answer_id")
        if not isinstance(question_id, int) or not isinstance(answer_id, int):
            continue
        answer = answers.get(answer_id)
        if _eligible_question(question, answer, policy):
            question_map[question_id] = question

    selected_question_ids: set[int] = set()
    selected: list[dict[str, Any]] = []
    # Narrower semantic families are allocated before the broad cookability pool.
    family_order = ("substitution", "evidence", "composition", "cookability")
    for family in family_order:
        candidates = [
            question
            for question in question_map.values()
            if int(question["question_id"]) not in selected_question_ids
            and _family_match_score(question, family) > 0
        ]
        candidates.sort(
            key=lambda question: (
                _quality_score(
                    question,
                    answers[int(question["accepted_answer_id"])],
                    family,
                    policy,
                ),
                int(question["creation_date"]),
                int(question["question_id"]),
            ),
            reverse=True,
        )
        recent = [
            question
            for question in candidates
            if int(question["creation_date"]) >= policy.recent_fromdate
        ][: policy.min_recent_per_family]
        if len(recent) != policy.min_recent_per_family:
            raise ValueError(
                f"family {family!r} has only {len(recent)} recent eligible unique tasks; "
                f"{policy.min_recent_per_family} required"
            )
        chosen_ids = {int(question["question_id"]) for question in recent}
        chosen = list(recent)
        chosen.extend(
            question for question in candidates if int(question["question_id"]) not in chosen_ids
        )
        chosen = chosen[: policy.per_family]
        if len(chosen) != policy.per_family:
            raise ValueError(
                f"family {family!r} has {len(chosen)} eligible unique tasks; "
                f"{policy.per_family} required"
            )
        for ordinal, question in enumerate(chosen, start=1):
            question_id = int(question["question_id"])
            answer = answers[int(question["accepted_answer_id"])]
            selected_question_ids.add(question_id)
            selected.append(
                _task_record(
                    question,
                    answer,
                    family=family,
                    ordinal=ordinal,
                    policy=policy,
                )
            )
    return sorted(selected, key=lambda task: (task["family"], task["task_id"]))


def build_task_bank(*, policy: SelectionPolicy) -> dict[str, Any]:
    questions_by_id: dict[int, dict[str, Any]] = {}
    with StackExchangeClient() as client:
        for question in client.fetch_questions(fromdate=policy.fromdate):
            question_id = question.get("question_id")
            if isinstance(question_id, int):
                questions_by_id[question_id] = question
        answer_ids = sorted(
            {
                int(question["accepted_answer_id"])
                for question in questions_by_id.values()
                if isinstance(question.get("accepted_answer_id"), int)
            }
        )
        answers = client.fetch_answers(answer_ids)

    tasks = select_tasks(questions_by_id.values(), answers, policy=policy)
    family_counts = {
        family: sum(task["family"] == family for task in tasks) for family in FAMILY_TAGS
    }
    recent_counts = {
        family: sum(
            task["family"] == family and bool(task["selection"]["recent"]) for task in tasks
        )
        for family in FAMILY_TAGS
    }
    bank: dict[str, Any] = {
        "schema_version": "flavourbench-real-user-task-bank-v1",
        "benchmark": "FlavourBench Season 0",
        "source_class": "real_human_authored_public_questions",
        "synthetic_tasks": 0,
        "source": {
            "api": STACKEXCHANGE_API,
            "site": STACKEXCHANGE_SITE,
            "corpus": "Seasoned Advice",
            "licensing_information": STACKEXCHANGE_TERMS_URL,
            "fromdate_utc": utc_iso(policy.fromdate),
            "retrieved_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "candidate_questions_retrieved": len(questions_by_id),
            "accepted_answers_retrieved": len(answers),
        },
        "selection_policy": {
            "per_family": policy.per_family,
            "minimum_recent_per_family": policy.min_recent_per_family,
            "recent_fromdate_utc": utc_iso(policy.recent_fromdate),
            "observed_through_utc": utc_iso(policy.observed_through),
            "excluded_tags": sorted(EXCLUDED_TAGS),
            "family_tags": {family: list(tags) for family, tags in FAMILY_TAGS.items()},
            "family_title_patterns": FAMILY_TITLE_PATTERNS,
            "specialist_content_pattern": EXCLUDED_CONTENT_PATTERN.pattern,
            "candidate_retrieval": "complete_site_corpus_since_fromdate",
            "accepted_human_answer_required": True,
            "closed_questions_excluded": True,
            "specialist_tracks_excluded": True,
        },
        "counts": {
            "tasks": len(tasks),
            "by_family": family_counts,
            "recent_by_family": recent_counts,
            "human_references": len(tasks),
        },
        "tasks": tasks,
    }
    bank["content_sha256"] = sha256_json(
        {"tasks": tasks, "selection_policy": bank["selection_policy"]}
    )
    return bank


def write_task_bank(bank: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = str(bank["content_sha256"])
    output_path = output_dir / f"stackexchange-real-user-task-bank-{digest}.json"
    output_path.write_bytes(json.dumps(bank, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
    return output_path


def run() -> None:
    parser = argparse.ArgumentParser(description="Build a real-user FlavourBench task bank")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/season0/source"),
        help="content-addressed output directory",
    )
    parser.add_argument("--per-family", type=int, default=30)
    parser.add_argument("--min-recent-per-family", type=int, default=10)
    args = parser.parse_args()
    policy = SelectionPolicy(
        per_family=args.per_family,
        min_recent_per_family=args.min_recent_per_family,
    )
    bank = build_task_bank(policy=policy)
    output_path = write_task_bank(bank, args.output_dir)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "content_sha256": bank["content_sha256"],
                "counts": bank["counts"],
                "synthetic_tasks": bank["synthetic_tasks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
