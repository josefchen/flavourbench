from __future__ import annotations

from pathlib import Path

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode


class _ComposeLoader(yaml.SafeLoader):
    pass


def _construct_reset(loader: _ComposeLoader, node: yaml.Node):
    if isinstance(node, MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, ScalarNode) and node.value == "null":
        return None
    return loader.construct_scalar(node)


_ComposeLoader.add_constructor("!reset", _construct_reset)


def _compose() -> dict:
    projects_root = Path(__file__).resolve().parents[4]
    path = projects_root / "epicure" / "compose.yaml"
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=_ComposeLoader)
    assert isinstance(value, dict)
    return value


def _rollback_compose() -> dict:
    projects_root = Path(__file__).resolve().parents[4]
    path = (
        projects_root
        / "epicure"
        / "deployment"
        / "flavourbench"
        / "rollback-legacy-volume.compose.yaml"
    )
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=_ComposeLoader)
    assert isinstance(value, dict)
    return value


def test_task_validation_replay_inputs_are_read_only_and_api_only() -> None:
    services = _compose()["services"]
    api = services["flavourbench-api"]
    worker = services["flavourbench-worker"]
    api_environment = api["environment"]
    worker_environment = worker["environment"]

    expected_targets = {
        "/run/flavourbench/task-validation-candidate-bundle-v6.json",
        "/run/flavourbench/task-validation-assignment-v6.json",
        "/run/flavourbench/task-validation-acquisition-receipt-v6.json",
        "/run/flavourbench/task-validation-campaign-v6.json",
        "/run/flavourbench/task-validation-quality-v6.json",
        "/run/flavourbench/task-validation-readiness-v6.json",
        "/run/flavourbench/task-validation-automated-replay-v1.json",
    }
    task_mounts = [
        mount
        for mount in api["volumes"]
        if "/run/flavourbench/task-validation-" in mount
        and "task-validation-packet.json" not in mount
    ]
    assert len(task_mounts) == len(expected_targets)
    assert all(mount.endswith(":ro") for mount in task_mounts)
    assert {mount.rsplit(":", 1)[0].rsplit(":", 1)[1] for mount in task_mounts} == (
        expected_targets
    )
    assert "volumes" not in worker

    expected_api_paths = {
        "FLAVOURBENCH_TASK_VALIDATION_CANDIDATE_BUNDLE_PATH": (
            "/run/flavourbench/task-validation-candidate-bundle-v6.json"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_ASSIGNMENT_PATH": (
            "/run/flavourbench/task-validation-assignment-v6.json"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_ACQUISITION_RECEIPT_PATH": (
            "/run/flavourbench/task-validation-acquisition-receipt-v6.json"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_CAMPAIGN_PATH": (
            "/run/flavourbench/task-validation-campaign-v6.json"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_QUALITY_REPORT_PATH": (
            "/run/flavourbench/task-validation-quality-v6.json"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_READINESS_PATH": (
            "/run/flavourbench/task-validation-readiness-v6.json"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_AUTOMATED_REPLAY_PATH": (
            "/run/flavourbench/task-validation-automated-replay-v1.json"
        ),
    }
    assert {name: api_environment[name] for name in expected_api_paths} == expected_api_paths
    expected_api_hashes = {
        "FLAVOURBENCH_DEVELOPMENT_TASK_VALIDATION_PACKET_SHA256": (
            "c45023aee6cf8ff91437c08c16ae20498b2d025e9a0155aedd44898de1d7fbb1"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_CAMPAIGN_SHA256": (
            "76b248477b3adc81b6eb198666a93538534db8e945567e2a99fc69085f709709"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_CANDIDATE_BUNDLE_SHA256": (
            "b13ab30bfb391e57a24c81d0398dc98e408d88e5a0bf21c4e758bf9271724cc3"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_ASSIGNMENT_SHA256": (
            "631932c0560ec417e47ff4c3ea94814ca9c944253252d3f4adcee8bd595221f9"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_ACQUISITION_RECEIPT_SHA256": (
            "847a95f7159ba778281fd5c20f0489a75f4655fc08a0de8075a0cba950259045"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_QUALITY_REPORT_SHA256": (
            "292492bce348a260eedf17b3cb04b041d4cc7c35aca47bc59bdf17d675a48ea8"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_READINESS_SHA256": (
            "449df377dd8de515a46a80d36dffc80f1734f3e86a60a53651675fb75c9d82c0"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_AUTOMATED_REPLAY_SHA256": (
            "89f6dede2826e27bcd69eb764e32bd7a203b371f0098831c78c1077383383157"
        ),
        "FLAVOURBENCH_TASK_VALIDATION_AUTOMATED_REPLAY_PHYSICAL_SHA256": (
            "ced66727597192342ddb978f7f48153a8fe82b0d5808d17f89c6ce42aabaaab9"
        ),
    }
    assert {name: api_environment[name] for name in expected_api_hashes} == (expected_api_hashes)
    for name in expected_api_paths:
        assert worker_environment[name] == ""
    assert worker_environment["FLAVOURBENCH_TASK_VALIDATION_CAMPAIGN_ENABLED"] == "false"
    for name, value in worker_environment.items():
        if name.startswith("FLAVOURBENCH_TASK_VALIDATION_") and name != (
            "FLAVOURBENCH_TASK_VALIDATION_CAMPAIGN_ENABLED"
        ):
            assert value == ""


def test_worker_does_not_inherit_task_review_credentials_or_replay_settings() -> None:
    worker_environment = _compose()["services"]["flavourbench-worker"]["environment"]
    for name in (
        "FLAVOURBENCH_SERVICE_TOKEN",
        "FLAVOURBENCH_ADMIN_TOKEN",
        "FLAVOURBENCH_EXPERT_TOKEN",
        "FLAVOURBENCH_TASK_VALIDATOR_IDENTITY_HMAC_SECRET",
        "FLAVOURBENCH_REVIEWER_IDENTITY_HMAC_SECRET",
        "FLAVOURBENCH_REVIEWER_CREDENTIAL_HMAC_SECRET",
        "FLAVOURBENCH_ACTIVE_EXPERT_CONSENT_SHA256S",
        "FLAVOURBENCH_EXPERT_CONSENT_DOCUMENTS_DIR",
        "FLAVOURBENCH_CONTAMINATION_SCAN_BUNDLE_PATH",
        "FLAVOURBENCH_CONTAMINATION_SCAN_BUNDLE_SHA256",
        "FLAVOURBENCH_VALIDATOR_CALIBRATION_ARTIFACT_PATH",
        "FLAVOURBENCH_VALIDATOR_CALIBRATION_ARTIFACT_SHA256",
        "FLAVOURBENCH_CONTAMINATION_CALIBRATION_ARTIFACT_PATH",
        "FLAVOURBENCH_CONTAMINATION_CALIBRATION_ARTIFACT_SHA256",
        "FLAVOURBENCH_TASK_VALIDATION_AUTOMATED_REPLAY_SHA256",
        "FLAVOURBENCH_TASK_VALIDATION_AUTOMATED_REPLAY_PHYSICAL_SHA256",
    ):
        assert worker_environment[name] in {"", "[]"}


def test_human_study_activation_record_is_read_only_api_only_and_fail_closed() -> None:
    projects_root = Path(__file__).resolve().parents[4]
    epicure_root = projects_root / "epicure"
    services = _compose()["services"]
    api = services["flavourbench-api"]
    worker = services["flavourbench-worker"]
    path_name = "FLAVOURBENCH_HUMAN_STUDY_ACTIVATION_MANIFEST_PATH"
    digest_name = "FLAVOURBENCH_HUMAN_STUDY_ACTIVATION_MANIFEST_SHA256"
    target = "/run/flavourbench/human-study-activation.json"

    activation_mounts = [mount for mount in api["volumes"] if mount.rsplit(":", 2)[-2] == target]
    assert len(activation_mounts) == 1
    assert activation_mounts[0].endswith(f":{target}:ro")
    assert "human-study-activation-current-v1.json" in activation_mounts[0]
    assert api["environment"][path_name] == target
    assert api["environment"][digest_name] == (
        "${FLAVOURBENCH_HUMAN_STUDY_ACTIVATION_MANIFEST_SHA256:-}"
    )
    assert worker["environment"][path_name] == ""
    assert worker["environment"][digest_name] == ""
    assert "volumes" not in worker

    env_example = (epicure_root / ".env.example").read_text(encoding="utf-8")
    assert f"{path_name}={target}" in env_example
    assert f"{digest_name}=\n" in env_example


def test_api_and_snapshot_worker_use_system_arrow_allocator_as_defense_in_depth() -> None:
    services = _compose()["services"]
    assert services["flavourbench-api"]["environment"]["ARROW_DEFAULT_MEMORY_POOL"] == "system"
    assert services["flavourbench-worker"]["environment"]["ARROW_DEFAULT_MEMORY_POOL"] == "system"


def test_clean_operational_database_uses_a_new_explicit_volume_and_ordered_bootstrap() -> None:
    compose = _compose()
    services = compose["services"]
    assert services["flavourbench-db"]["volumes"] == [
        "flavourbench-db-data-v2:/var/lib/postgresql/data"
    ]
    assert set(compose["volumes"]) == {"flavourbench-db-data-v2"}
    volume = compose["volumes"]["flavourbench-db-data-v2"]
    assert volume == {
        "name": "epicure_flavourbench-db-data-v2",
        "labels": {
            "ai.flavourbench.data-plane": "operational-v2",
            "ai.flavourbench.legacy-imported": "false",
        },
    }
    assert services["flavourbench-db-bootstrap"]["depends_on"] == {
        "flavourbench-db": {"condition": "service_healthy"}
    }
    assert services["flavourbench-migrate"]["depends_on"] == {
        "flavourbench-db-bootstrap": {"condition": "service_completed_successfully"}
    }
    assert services["flavourbench-db-grants"]["depends_on"] == {
        "flavourbench-migrate": {"condition": "service_completed_successfully"}
    }
    assert services["flavourbench-api"]["depends_on"] == {
        "flavourbench-db-grants": {"condition": "service_completed_successfully"}
    }


def test_retrospective_pilot_import_is_not_in_clean_database_profile() -> None:
    services = _compose()["services"]
    importer = services["flavourbench-current-pilot-review-import"]
    assert importer["profiles"] == ["flavourbench-pilot-import"]
    assert (
        "flavourbench-current-pilot-review-import" not in services["flavourbench-api"]["depends_on"]
    )


def test_worker_requires_an_explicit_generation_profile() -> None:
    services = _compose()["services"]
    assert services["flavourbench-worker"]["profiles"] == ["flavourbench-worker"]

    projects_root = Path(__file__).resolve().parents[4]
    unit_path = projects_root / "epicure" / "deployment" / "epicure.service"
    unit_text = unit_path.read_text(encoding="utf-8")
    assert "Environment=COMPOSE_PROFILES=" in unit_text
    assert "--profile flavourbench-worker" not in unit_text
    assert unit_text.count("docker update --restart=no epicure-flavourbench-worker-1") == 3
    assert unit_text.count("docker rm -f epicure-flavourbench-worker-1") == 3

    for path in (projects_root / "epicure" / "deployment" / "deploy_cluster.sh",):
        text = path.read_text(encoding="utf-8")
        assert "--profile flavourbench-worker" not in text

    readme = (projects_root / "epicure" / "README.md").read_text(encoding="utf-8")
    assert "--profile flavourbench --profile flavourbench-worker" in readme


def test_api_control_plane_and_worker_generation_authority_are_separate() -> None:
    services = _compose()["services"]
    api_environment = services["flavourbench-api"]["environment"]
    worker_environment = services["flavourbench-worker"]["environment"]

    assert api_environment["FLAVOURBENCH_SERVICE_ROLE"] == "api"
    assert api_environment["FLAVOURBENCH_ENVIRONMENT"] == "production"
    assert api_environment["FLAVOURBENCH_EXECUTION_MODE"] == "live"
    assert api_environment["FLAVOURBENCH_LIVE_AUTHORIZED"] == "true"
    assert api_environment["FLAVOURBENCH_PUBLIC_ARENA_ENABLED"] == "false"
    assert api_environment["FLAVOURBENCH_CATALOG_SYNC_ENABLED"] == "false"
    assert api_environment["FLAVOURBENCH_TASK_VALIDATION_CAMPAIGN_ENABLED"] == "true"
    assert api_environment["FLAVOURBENCH_TASK_VALIDATION_SEASON_SLUG"] == "season-1"
    for name in (
        "FLAVOURBENCH_CONTAMINATION_SCAN_BUNDLE_PATH",
        "FLAVOURBENCH_CONTAMINATION_SCAN_BUNDLE_SHA256",
        "FLAVOURBENCH_VALIDATOR_CALIBRATION_ARTIFACT_PATH",
        "FLAVOURBENCH_VALIDATOR_CALIBRATION_ARTIFACT_SHA256",
        "FLAVOURBENCH_CONTAMINATION_CALIBRATION_ARTIFACT_PATH",
        "FLAVOURBENCH_CONTAMINATION_CALIBRATION_ARTIFACT_SHA256",
    ):
        assert api_environment[name] == ""
    for name in (
        "FLAVOURBENCH_OPENROUTER_API_KEY",
        "FLAVOURBENCH_KIMI_API_KEY",
        "FLAVOURBENCH_COHERE_API_KEY",
        "FLAVOURBENCH_QWENCLOUD_API_KEY",
        "FLAVOURBENCH_MCP_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        assert name not in api_environment

    assert worker_environment["FLAVOURBENCH_SERVICE_ROLE"] == "worker"
    assert worker_environment["FLAVOURBENCH_ENVIRONMENT"] == "production"
    assert worker_environment["FLAVOURBENCH_EXECUTION_MODE"] == "live"
    assert worker_environment["FLAVOURBENCH_LIVE_AUTHORIZED"] == (
        "${FLAVOURBENCH_WORKER_LIVE_AUTHORIZED:-false}"
    )
    assert worker_environment["FLAVOURBENCH_CATALOG_SYNC_ENABLED"] == (
        "${FLAVOURBENCH_WORKER_CATALOG_SYNC_ENABLED:-false}"
    )
    assert worker_environment["FLAVOURBENCH_PUBLIC_ARENA_ENABLED"] == "false"
    assert worker_environment["FLAVOURBENCH_TASK_VALIDATION_CAMPAIGN_ENABLED"] == "false"
    assert worker_environment["FLAVOURBENCH_TASK_VALIDATION_SEASON_SLUG"] == ""

    projects_root = Path(__file__).resolve().parents[4]
    helper = (projects_root / "epicure" / "deployment" / "configure_flavourbench_env.sh").read_text(
        encoding="utf-8"
    )
    assert "set_value FLAVOURBENCH_LIVE_AUTHORIZED true" not in helper
    assert "set_value FLAVOURBENCH_WORKER_LIVE_AUTHORIZED false" in helper
    assert "set_value FLAVOURBENCH_WORKER_CATALOG_SYNC_ENABLED false" in helper


def test_legacy_volume_is_available_only_through_explicit_local_rollback_override() -> None:
    rollback = _rollback_compose()
    legacy = rollback["volumes"]["flavourbench-db-data-legacy"]
    assert legacy == {
        "external": True,
        "name": "epicure_flavourbench-db-data",
    }
    database = rollback["services"]["flavourbench-db"]
    assert database["volumes"] == ["flavourbench-db-data-legacy:/var/lib/postgresql/data"]
    assert database["healthcheck"]["test"] == [
        "CMD-SHELL",
        "pg_isready -U flavourbench -d flavourbench",
    ]
    api = rollback["services"]["flavourbench-api"]
    assert api["pull_policy"] == "never"
    assert api["build"] is None
    assert api["depends_on"] == {}
    assert "FLAVOURBENCH_ROLLBACK_API_IMAGE:?" in api["image"]
    assert "FLAVOURBENCH_DB_LEGACY_PASSWORD:?" in api["environment"]["FLAVOURBENCH_DATABASE_URL"]
    projects_root = Path(__file__).resolve().parents[4]
    rollback_text = (
        projects_root
        / "epicure"
        / "deployment"
        / "flavourbench"
        / "rollback-legacy-volume.compose.yaml"
    ).read_text(encoding="utf-8")
    assert "build: !reset null" in rollback_text
    assert "depends_on: !reset {}" in rollback_text
