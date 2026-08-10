"""Build a licensed historical real-task supplement from the Stack Exchange dump."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .real_task_bank import SelectionPolicy, _eligible_question, _family_match_score, sha256_json
from .task_curation import PREPARE_SCHEMA, _candidate_record

HISTORICAL_FROMDATE = 1_514_764_800  # 2018-01-01T00:00:00Z
HISTORICAL_TODATE = 1_672_531_199  # 2022-12-31T23:59:59Z
TARGET_FAMILIES = ("substitution", "composition")


class StackExchangeDumpError(RuntimeError):
    """The public data dump could not produce a verifiable source pool."""


def _timestamp(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise StackExchangeDumpError(f"invalid Stack Exchange timestamp {value!r}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def _integer(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _tags(value: str) -> list[str]:
    if "|" in value:
        return sorted(tag for tag in value.strip("|").split("|") if tag)
    return sorted(re.findall(r"<([^>]+)>", value))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_posts(
    posts_path: Path, *, fromdate: int, todate: int
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], set[int]]:
    questions: dict[int, dict[str, Any]] = {}
    answers: dict[int, dict[str, Any]] = {}
    accepted_ids: set[int] = set()
    owner_ids: set[int] = set()
    for _event, element in ET.iterparse(posts_path, events=("end",)):
        if element.tag != "row":
            element.clear()
            continue
        attributes = element.attrib
        post_type = attributes.get("PostTypeId")
        post_id = _integer(attributes.get("Id"), -1)
        if post_type == "1":
            created = _timestamp(attributes.get("CreationDate") or "")
            accepted_answer_id = _integer(attributes.get("AcceptedAnswerId"), -1)
            if fromdate <= created <= todate and post_id > 0 and accepted_answer_id > 0:
                owner_id = _integer(attributes.get("OwnerUserId"), 0)
                if owner_id:
                    owner_ids.add(owner_id)
                questions[post_id] = {
                    "question_id": post_id,
                    "accepted_answer_id": accepted_answer_id,
                    "is_answered": True,
                    "title": attributes.get("Title") or "",
                    "body": attributes.get("Body") or "",
                    "creation_date": created,
                    "last_activity_date": _timestamp(
                        attributes.get("LastActivityDate") or attributes.get("CreationDate") or ""
                    ),
                    "link": f"https://cooking.stackexchange.com/questions/{post_id}",
                    "content_license": attributes.get("ContentLicense") or "",
                    "owner_user_id": owner_id or None,
                    "owner_display_name": attributes.get("OwnerDisplayName") or "",
                    "score": _integer(attributes.get("Score")),
                    "answer_count": _integer(attributes.get("AnswerCount")),
                    "tags": _tags(attributes.get("Tags") or ""),
                }
                accepted_ids.add(accepted_answer_id)
        elif post_type == "2" and post_id in accepted_ids:
            owner_id = _integer(attributes.get("OwnerUserId"), 0)
            if owner_id:
                owner_ids.add(owner_id)
            answers[post_id] = {
                "answer_id": post_id,
                "body": attributes.get("Body") or "",
                "creation_date": _timestamp(attributes.get("CreationDate") or ""),
                "last_activity_date": _timestamp(
                    attributes.get("LastActivityDate") or attributes.get("CreationDate") or ""
                ),
                "content_license": attributes.get("ContentLicense") or "",
                "owner_user_id": owner_id or None,
                "owner_display_name": attributes.get("OwnerDisplayName") or "",
                "score": _integer(attributes.get("Score")),
            }
        element.clear()
    return questions, answers, owner_ids


def _read_users(users_path: Path, wanted_ids: set[int]) -> dict[int, dict[str, Any]]:
    users: dict[int, dict[str, Any]] = {}
    for _event, element in ET.iterparse(users_path, events=("end",)):
        if element.tag != "row":
            element.clear()
            continue
        user_id = _integer(element.attrib.get("Id"), 0)
        if user_id in wanted_ids:
            users[user_id] = {
                "display_name": element.attrib.get("DisplayName") or "unknown",
                "user_id": user_id,
                "link": f"https://cooking.stackexchange.com/users/{user_id}",
            }
        element.clear()
    return users


def _attach_owner(record: dict[str, Any], users: Mapping[int, Mapping[str, Any]]) -> None:
    owner_id = record.pop("owner_user_id", None)
    display_name = str(record.pop("owner_display_name", "") or "")
    known = users.get(owner_id) if isinstance(owner_id, int) else None
    record["owner"] = (
        dict(known)
        if known is not None
        else {
            "display_name": display_name or "unknown",
            "user_id": owner_id,
            "link": (f"https://cooking.stackexchange.com/users/{owner_id}" if owner_id else None),
        }
    )


def build_historical_candidate_pool(
    *,
    posts_path: Path,
    users_path: Path,
    archive_path: Path,
    fromdate: int = HISTORICAL_FROMDATE,
    todate: int = HISTORICAL_TODATE,
    per_family: int = 240,
) -> dict[str, Any]:
    if not fromdate < todate:
        raise StackExchangeDumpError("historical source window must be increasing")
    if per_family <= 0:
        raise StackExchangeDumpError("historical per-family candidate count must be positive")
    questions, answers, owner_ids = _read_posts(posts_path, fromdate=fromdate, todate=todate)
    users = _read_users(users_path, owner_ids)
    policy = SelectionPolicy(fromdate=fromdate)
    candidates: list[dict[str, Any]] = []
    for question_id in sorted(questions):
        question = questions[question_id]
        answer = answers.get(int(question["accepted_answer_id"]))
        if not _eligible_question(question, answer, policy):
            continue
        if not any(_family_match_score(question, family) > 0 for family in TARGET_FAMILIES):
            continue
        assert answer is not None
        _attach_owner(question, users)
        _attach_owner(answer, users)
        candidates.append(_candidate_record(question, answer, policy))

    selected: list[dict[str, Any]] = []
    used_question_ids: set[int] = set()
    for family in TARGET_FAMILIES:
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                int(candidate["heuristics"]["family_match_scores"][family]),
                math.log1p(max(0, int(candidate["source"]["score_at_import"]))),
                math.log1p(max(0, int(candidate["human_reference"]["score_at_import"]))),
                min(len(str(candidate["prompt"])), 2_000),
                int(candidate["question_id"]),
            ),
            reverse=True,
        )
        chosen = [
            candidate
            for candidate in ranked
            if int(candidate["question_id"]) not in used_question_ids
            and int(candidate["heuristics"]["family_match_scores"][family]) > 0
        ][:per_family]
        if len(chosen) != per_family:
            raise StackExchangeDumpError(
                f"historical {family} pool has only {len(chosen)} unique candidates"
            )
        selected.extend(chosen)
        used_question_ids.update(int(candidate["question_id"]) for candidate in chosen)
    candidates = sorted(selected, key=lambda candidate: int(candidate["question_id"]))

    source = {
        "site": "cooking",
        "corpus": "Seasoned Advice public Stack Exchange data dump",
        "archive_url": "https://archive.org/download/stackexchange/cooking.stackexchange.com.7z",
        "archive_sha256": _file_sha256(archive_path),
        "posts_sha256": _file_sha256(posts_path),
        "users_sha256": _file_sha256(users_path),
        "fromdate_utc": datetime.fromtimestamp(fromdate, UTC).isoformat().replace("+00:00", "Z"),
        "todate_utc": datetime.fromtimestamp(todate, UTC).isoformat().replace("+00:00", "Z"),
        "recent_fromdate_utc": datetime.fromtimestamp(policy.recent_fromdate, UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "observed_through_utc": datetime.fromtimestamp(policy.observed_through, UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "questions_in_window_with_accepted_answer": len(questions),
        "accepted_answers_retrieved": len(answers),
        "targeted_families": list(TARGET_FAMILIES),
        "candidate_selection": {
            "method": "family_match_then_question_answer_score_v1",
            "per_family": per_family,
            "unique_across_targeted_families": True,
        },
    }
    pool: dict[str, Any] = {
        "schema_version": PREPARE_SCHEMA,
        "benchmark": "FlavourBench Season 0 historical real-user supplement",
        "source_class": "real_human_authored_public_questions",
        "synthetic_tasks": 0,
        "curation_answers_visible_to_models": False,
        "source": source,
        "counts": {
            "eligible_candidates": len(candidates),
            "recent_candidates": 0,
            "accepted_human_references": len(candidates),
        },
        "candidates": candidates,
    }
    pool["content_sha256"] = sha256_json(
        {
            "schema_version": PREPARE_SCHEMA,
            "source": {
                key: source[key]
                for key in (
                    "site",
                    "fromdate_utc",
                    "recent_fromdate_utc",
                    "observed_through_utc",
                )
            },
            "candidates": candidates,
        }
    )
    return pool


def write_historical_pool(pool: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"real-task-historical-supplement-{pool['content_sha256']}.json"
    rendered = json.dumps(pool, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    if path.exists() and path.read_bytes() != rendered:
        raise StackExchangeDumpError("historical-pool content-address conflict")
    path.write_bytes(rendered)
    return path


def run() -> None:
    parser = argparse.ArgumentParser(description="Build a historical real-task source supplement")
    parser.add_argument("--posts", type=Path, required=True)
    parser.add_argument("--users", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, default=Path("data/season0/source"))
    args = parser.parse_args()
    pool = build_historical_candidate_pool(
        posts_path=args.posts,
        users_path=args.users,
        archive_path=args.archive,
        per_family=args.per_family,
    )
    path = write_historical_pool(pool, args.output_dir)
    print(
        json.dumps(
            {
                "output": str(path),
                "content_sha256": pool["content_sha256"],
                "counts": pool["counts"],
                "synthetic_tasks": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
