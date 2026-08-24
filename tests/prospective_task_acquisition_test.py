from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx

from flavourbench.prospective_task_acquisition import (
    API_BASE_URL,
    ASSIGNMENT_SCHEMA,
    BUNDLE_SCHEMA,
    AcquisitionPolicy,
    PublicStackExchangeClient,
    build_assignment_artifact,
    build_candidate_bundle,
    deidentify_direct_contacts,
    html_to_text_exact,
    verify_artifact,
)


def _question(
    question_id: int,
    *,
    title: str,
    body: str,
    tags: list[str],
    created: int = 1_735_689_600,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "title": title,
        "body": body,
        "tags": tags,
        "creation_date": created + question_id,
        "last_activity_date": created + question_id + 60,
        "last_edit_date": created + question_id,
        "content_license": "CC BY-SA 4.0",
        "link": f"https://cooking.stackexchange.com/questions/{question_id}/fixture",
        "owner": {
            "user_id": question_id,
            "display_name": f"Cook {question_id}",
            "link": f"https://cooking.stackexchange.com/users/{question_id}/fixture",
        },
        "score": 2,
        "answer_count": 1,
        "accepted_answer_id": 10_000 + question_id,
        "is_answered": True,
    }


def _revision(question: dict[str, object]) -> dict[str, object]:
    return {
        "post_id": question["question_id"],
        "post_type": "question",
        "revision_type": "single_user",
        "revision_number": 1,
        "revision_guid": f"fixture-revision-{question['question_id']}",
        "creation_date": question["creation_date"],
        "content_license": "CC BY-SA 4.0",
        "title": question["title"],
        "body": question["body"],
    }


def test_normalization_and_contact_redaction_are_narrow_and_logged() -> None:
    value = html_to_text_exact(
        "<p>Why did my sauce split?</p><p>Email cook@example.test or use 180 C.</p>"
    )
    assert value == "Why did my sauce split?\n\nEmail cook@example.test or use 180 C."
    redacted, log = deidentify_direct_contacts(value, field="body")
    assert redacted == "Why did my sauce split?\n\nEmail [redacted-email] or use 180 C."
    assert [row["rule_id"] for row in log] == ["email_address"]
    assert log[0]["occurrences"] == 1
    assert "cook@example.test" not in json.dumps(log)


def test_public_client_paginates_questions_fetches_revisions_and_never_answers() -> None:
    first = _question(
        1,
        title="Can I replace butter in this biscuit dough?",
        body="<p>I need a dairy-free replacement that preserves a flaky texture.</p>",
        tags=["substitutions"],
    )
    second = _question(
        2,
        title="Why does this emulsion split after cooling?",
        body="<p>I want to understand the mechanism and how temperature changes it.</p>",
        tags=["food-science"],
    )
    observed_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_paths.append(request.url.path)
        if request.url.path.endswith("/questions"):
            page = int(request.url.params["page"])
            return httpx.Response(
                200,
                json={
                    "items": [first] if page == 1 else [second],
                    "has_more": page == 1,
                    "quota_max": 300,
                    "quota_remaining": 299 - page,
                },
            )
        if request.url.path.endswith("/revisions"):
            return httpx.Response(
                200,
                json={
                    "items": [_revision(first), _revision(second)],
                    "has_more": False,
                    "quota_max": 300,
                    "quota_remaining": 296,
                },
            )
        raise AssertionError(f"unexpected endpoint: {request.url}")

    http_client = httpx.Client(
        base_url=API_BASE_URL,
        transport=httpx.MockTransport(handler),
    )
    with PublicStackExchangeClient(client=http_client, sleep=lambda _: None) as client:
        questions = client.fetch_questions(fromdate=1, todate=2_000_000_000)
        revisions = client.fetch_revisions([1, 2])

    assert [row["question_id"] for row in questions] == [1, 2]
    assert set(revisions) == {1, 2}
    assert len(client.request_receipts) == 3
    assert all("answer" not in path for path in observed_paths)
    assert all(
        receipt["authentication"] == "public_unauthenticated"
        for receipt in client.request_receipts
    )


def test_bundle_keeps_exact_revision_provenance_and_provisional_screens() -> None:
    valid = _question(
        10,
        title="What can replace butter in this biscuit dough?",
        body=(
            "<p>I need a dairy-free substitute that preserves distinct layers and can be "
            "worked with ordinary kitchen equipment.</p>"
        ),
        tags=["substitutions", "baking"],
    )
    specialist = _question(
        11,
        title="Is this cooked rice safe to eat?",
        body=(
            "<p>It sat at room temperature and I need a food safety determination before "
            "serving it.</p>"
        ),
        tags=["food-safety"],
    )
    visual = _question(
        12,
        title="Why is the pictured loaf shaped this way?",
        body=(
            "<p>Please inspect the attached image before explaining the crumb.</p>"
            "<img src='https://example.test/crumb.png'>"
        ),
        tags=["bread", "food-science"],
    )
    bundle = build_candidate_bundle(
        questions=[valid, specialist, visual],
        revisions_by_question={
            int(question["question_id"]): [_revision(question)]
            for question in (valid, specialist, visual)
        },
        policy=AcquisitionPolicy(
            fromdate=1_735_689_600,
            todate=1_800_000_000,
            target_per_family=1,
            review_reserve_per_family=1,
        ),
        retrieved_utc="2026-08-08T00:00:00Z",
        request_receipts=[{"path": "/questions"}],
    )

    verify_artifact(bundle, schema_version=BUNDLE_SCHEMA)
    assert bundle["counts"]["synthetic_tasks"] == 0
    assert bundle["counts"]["source_answer_payloads"] == 0
    by_id = {row["source"]["question_id"]: row for row in bundle["candidates"]}
    assert by_id[10]["source"]["revision_guid"] == "fixture-revision-10"
    assert by_id[10]["source"]["body_html_source_sha256"]
    assert by_id[10]["source"]["source_answer_payload_requested"] is False
    specialist_screen = next(
        row
        for row in by_id[11]["provisional_screens"]
        if row["screen_id"] == "specialist_track_exclusion"
    )
    assert specialist_screen["decision"] == "exclude"
    context_screen = next(
        row
        for row in by_id[12]["provisional_screens"]
        if row["screen_id"] == "self_contained_visual_and_link_context"
    )
    assert context_screen["decision"] == "review"
    assert all(
        screen["human_ground_truth"] is False
        for candidate in bundle["candidates"]
        for screen in candidate["provisional_screens"]
    )


def test_terminal_revision_can_change_only_one_field() -> None:
    question = _question(
        13,
        title="Can I replace butter while keeping pastry flaky?",
        body=(
            "<p>I need a substitute that preserves layers and works with ordinary kitchen "
            "equipment.</p>"
        ),
        tags=["substitutions"],
    )
    initial = _revision(question)
    initial["title"] = "Can I replace butter?"
    title_edit = {
        "post_id": 13,
        "post_type": "question",
        "revision_type": "single_user",
        "revision_number": 2,
        "revision_guid": "fixture-revision-13-title-edit",
        "creation_date": int(question["creation_date"]) + 10,
        "content_license": "CC BY-SA 4.0",
        "title": question["title"],
    }
    bundle = build_candidate_bundle(
        questions=[question],
        revisions_by_question={13: [initial, title_edit]},
        policy=AcquisitionPolicy(
            fromdate=1_735_689_600,
            todate=1_800_000_000,
            target_per_family=1,
            review_reserve_per_family=1,
        ),
        retrieved_utc="2026-08-08T00:00:00Z",
        request_receipts=[],
    )
    candidate = bundle["candidates"][0]
    assert candidate["source"]["revision_number"] == 2
    revision_screen = next(
        screen
        for screen in candidate["provisional_screens"]
        if screen["screen_id"] == "source_revision_integrity"
    )
    assert revision_screen["decision"] == "pass"


def test_duplicate_screen_and_assignment_fail_closed_without_human_reviews() -> None:
    questions: list[dict[str, object]] = []
    revisions: dict[int, list[dict[str, object]]] = {}
    for question_id in range(1, 9):
        family_index = (question_id - 1) % 4
        family_data = (
            (
                "Can I substitute butter while preserving pastry layers?",
                "I need a replacement that works in an ordinary home kitchen and keeps the "
                "dough flaky without introducing a strong new flavour.",
                ["substitutions"],
            ),
            (
                "How should I balance acid and spice in a bean stew?",
                "I want the aromatics, acidity, and toasted spices to remain distinct while "
                "forming one coherent flavour profile for dinner.",
                ["flavor", "spices"],
            ),
            (
                "How can I sequence two oven stages for this bread?",
                "I have one conventional oven and need a practical timing method that produces "
                "a crisp crust without drying the crumb.",
                ["baking", "oven"],
            ),
            (
                "Why does an emulsion split when its temperature falls?",
                "I want a mechanism-based explanation of what changes during cooling and which "
                "observations would distinguish the plausible causes.",
                ["food-science", "emulsion"],
            ),
        )[family_index]
        # Preserve family signal while making each fixture genuinely distinct.
        question = _question(
            question_id,
            title=family_data[0],
            body=f"<p>{family_data[1]} Case {question_id}.</p>",
            tags=family_data[2],
            created=1_740_000_000,
        )
        questions.append(question)
        revisions[question_id] = [_revision(question)]

    # Add one exact duplicate with a later source timestamp; the older one is excluded.
    duplicate = dict(questions[0])
    duplicate["question_id"] = 99
    duplicate["creation_date"] = 1_750_000_000
    duplicate["last_activity_date"] = 1_750_000_060
    duplicate["last_edit_date"] = 1_750_000_000
    duplicate["link"] = "https://cooking.stackexchange.com/questions/99/fixture"
    questions.append(duplicate)
    revisions[99] = [_revision(duplicate)]

    policy = AcquisitionPolicy(
        fromdate=1_735_689_600,
        todate=1_800_000_000,
        target_per_family=1,
        review_reserve_per_family=1,
        maximum_candidates_per_source_author=8,
        maximum_candidates_per_source_author_family=8,
    )
    bundle = build_candidate_bundle(
        questions=questions,
        revisions_by_question=revisions,
        policy=policy,
        retrieved_utc="2026-08-08T00:00:00Z",
        request_receipts=[],
    )
    duplicate_screens = [
        next(
            screen
            for screen in candidate["provisional_screens"]
            if screen["screen_id"] == "within_bundle_duplicate"
        )
        for candidate in bundle["candidates"]
    ]
    assert any(screen["decision"] == "exclude" for screen in duplicate_screens)

    assignment = build_assignment_artifact(bundle, policy=policy)
    verify_artifact(assignment, schema_version=ASSIGNMENT_SCHEMA)
    assert assignment["counts"]["human_validation_ballots_recorded"] == 0
    assert assignment["claim_boundary"]["database_import_authorized"] is False
    assert assignment["ui_compatibility"]["existing_blind_task_candidate_view"] == (
        "compatible"
    )
    assert assignment["ui_compatibility"]["existing_confirmatory_bank_import"] == (
        "fail_closed_incompatible"
    )
    assert all(
        row["model_outputs_visible"] is False
        and row["source_answer_text_visible"] is False
        for row in assignment["assignment_rows"]
    )
    assert all(
        event["database_mutation_authorized"] is False
        and event["payload_json"]["rank_eligible"] is False
        for event in assignment["run_event_import_templates"]
    )


def test_artifact_hash_changes_when_source_text_changes() -> None:
    question = _question(
        21,
        title="Why does starch thicken this sauce?",
        body="<p>I want to understand the mechanism under ordinary cooking conditions.</p>",
        tags=["food-science"],
    )
    policy = AcquisitionPolicy(
        fromdate=1_735_689_600,
        todate=1_800_000_000,
        target_per_family=1,
        review_reserve_per_family=1,
    )
    first = build_candidate_bundle(
        questions=[question],
        revisions_by_question={21: [_revision(question)]},
        policy=policy,
        retrieved_utc=datetime(2026, 8, 8, tzinfo=UTC).isoformat(),
        request_receipts=[],
    )
    changed = dict(question)
    changed["body"] = (
        "<p>I want to understand the mechanism under ordinary cooking conditions and "
        "compare two starches.</p>"
    )
    second = build_candidate_bundle(
        questions=[changed],
        revisions_by_question={21: [_revision(changed)]},
        policy=policy,
        retrieved_utc=datetime(2026, 8, 8, tzinfo=UTC).isoformat(),
        request_receipts=[],
    )
    assert first["artifact_sha256"] != second["artifact_sha256"]
