from __future__ import annotations

import hashlib
from pathlib import Path

from flavourbench.stackexchange_dump import build_historical_candidate_pool


def _write(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def test_build_historical_pool_keeps_real_accepted_answer_and_attribution(tmp_path: Path) -> None:
    posts = tmp_path / "Posts.xml"
    users = tmp_path / "Users.xml"
    archive = tmp_path / "cooking.7z"
    _write(
        posts,
        """<?xml version="1.0" encoding="utf-8"?>
<posts>
<row Id="100" PostTypeId="1" AcceptedAnswerId="101"
 CreationDate="2020-01-01T00:00:00.000" LastActivityDate="2020-01-02T00:00:00.000"
 Score="5" Body="&lt;p&gt;What can replace eggs in this vegan cake
 while preserving structure?&lt;/p&gt;"
 OwnerUserId="7" Title="Vegan egg substitution for cake" Tags="|substitutions|vegan|baking|"
 AnswerCount="1" ContentLicense="CC BY-SA 4.0" />
<row Id="101" PostTypeId="2" ParentId="100" CreationDate="2020-01-01T01:00:00.000"
 LastActivityDate="2020-01-01T01:00:00.000" Score="8"
 Body="&lt;p&gt;Use aquafaba for foam and starch for binding in this specific cake.&lt;/p&gt;"
 OwnerUserId="8" ContentLicense="CC BY-SA 4.0" />
<row Id="102" PostTypeId="1" AcceptedAnswerId="103"
 CreationDate="2020-02-01T00:00:00.000" LastActivityDate="2020-02-02T00:00:00.000"
 Score="6" Body="&lt;p&gt;How can I balance acid, sweetness, and herbs in tomato sauce?&lt;/p&gt;"
 OwnerUserId="7" Title="Balancing tomato sauce" Tags="|flavor|sauce|herbs|"
 AnswerCount="1" ContentLicense="CC BY-SA 4.0" />
<row Id="103" PostTypeId="2" ParentId="102" CreationDate="2020-02-01T01:00:00.000"
 LastActivityDate="2020-02-01T01:00:00.000" Score="9"
 Body="&lt;p&gt;Adjust salt first, then acid and sweetness, tasting between changes.&lt;/p&gt;"
 OwnerUserId="8" ContentLicense="CC BY-SA 4.0" />
</posts>""",
    )
    _write(
        users,
        """<?xml version="1.0" encoding="utf-8"?>
<users><row Id="7" DisplayName="Question Cook"/><row Id="8" DisplayName="Answer Cook"/></users>""",
    )
    archive.write_bytes(b"fixture")

    pool = build_historical_candidate_pool(
        posts_path=posts,
        users_path=users,
        archive_path=archive,
        per_family=1,
    )

    assert pool["synthetic_tasks"] == 0
    assert pool["counts"]["eligible_candidates"] == 2
    candidate = next(item for item in pool["candidates"] if item["question_id"] == 100)
    assert candidate["source"]["author"]["display_name"] == "Question Cook"
    assert candidate["human_reference"]["author"]["display_name"] == "Answer Cook"
    assert candidate["human_reference"]["accepted"] is True
    assert pool["source"]["archive_sha256"] == hashlib.sha256(b"fixture").hexdigest()
