from flavourbench.tasks import candidate_tasks


def test_season_zero_has_30_unique_candidates_per_family() -> None:
    tasks = candidate_tasks()
    assert len(tasks) == 120
    assert len({task.public_id for task in tasks}) == 120
    assert len({task.prompt_sha256 for task in tasks}) == 120
    for family in {task.family for task in tasks}:
        family_tasks = [task for task in tasks if task.family == family]
        assert len(family_tasks) == 30
        assert all(task.split == "pilot" for task in family_tasks)
        assert all(task.review_status == "candidate" for task in family_tasks)


def test_legacy_ids_are_not_reused() -> None:
    assert all(not task.public_id.startswith(("q-", "legacy-")) for task in candidate_tasks())
