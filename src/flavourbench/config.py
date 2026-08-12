from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FLAVOURBENCH_",
        env_file=(".env.local", ".env"),
        extra="ignore",
    )

    environment: str = "development"
    service_role: str = "api"
    database_url: str = "sqlite:///./flavourbench-dev.sqlite3"
    auto_create_schema: bool = True
    service_token: str = "development-only-service-token"
    admin_token: str = ""
    expert_token: str = ""
    active_expert_consent_sha256s: list[str] = Field(default_factory=list)
    expert_consent_documents_dir: str = ""
    human_study_activation_manifest_path: str = ""
    human_study_activation_manifest_sha256: str = ""
    development_task_validation_packet_path: str = ""
    development_task_validation_packet_sha256: str = ""
    task_validation_campaign_enabled: bool = False
    task_validation_season_slug: str = "season-1"
    task_validation_campaign_path: str = ""
    task_validation_campaign_sha256: str = ""
    task_validation_candidate_bundle_path: str = ""
    task_validation_candidate_bundle_sha256: str = ""
    task_validation_assignment_path: str = ""
    task_validation_assignment_sha256: str = ""
    task_validation_acquisition_receipt_path: str = ""
    task_validation_acquisition_receipt_sha256: str = ""
    task_validation_quality_report_path: str = ""
    task_validation_quality_report_sha256: str = ""
    task_validation_readiness_path: str = ""
    task_validation_readiness_sha256: str = ""
    task_validation_automated_replay_path: str = ""
    task_validation_automated_replay_sha256: str = ""
    task_validation_automated_replay_physical_sha256: str = ""
    contamination_scan_bundle_path: str = ""
    contamination_scan_bundle_sha256: str = ""
    contamination_calibration_artifact_path: str = ""
    contamination_calibration_artifact_sha256: str = ""
    validator_calibration_artifact_path: str = ""
    validator_calibration_artifact_sha256: str = ""
    pseudonym_secret: str = "development-only-pseudonym-secret"
    task_validator_identity_hmac_secret: str = (
        "development-only-task-validator-identity-hmac-secret"
    )
    reviewer_identity_hmac_secret: str = "development-only-reviewer-identity-hmac-secret"
    reviewer_identity_hmac_key_id: str = "primary"
    reviewer_credential_hmac_secret: str = "development-only-reviewer-credential-hmac-secret"
    reviewer_credential_hmac_key_id: str = "primary"
    reviewer_session_max_uses: int = Field(default=128, ge=4, le=256)
    reviewer_session_ttl_seconds: int = Field(default=86_400, ge=300, le=604_800)
    expert_output_comparison_quorum: int = Field(default=2, ge=2, le=10)
    organization_api_key_hmac_secret: str = "development-only-organization-api-key-hmac-secret"
    organization_api_key_hmac_key_id: str = "primary"
    organization_api_key_hmac_verification_keys: dict[str, str] = Field(default_factory=dict)
    run_card_signing_secret: str = "development-only-run-card-signing-secret"
    run_card_signing_key_id: str = "primary"
    run_card_verification_keys: dict[str, str] = Field(default_factory=dict)
    budget_authorization_signing_secret: str = (
        "development-only-budget-authorization-signing-secret"
    )
    budget_authorization_signing_key_id: str = "primary"
    budget_authorization_verification_keys: dict[str, str] = Field(default_factory=dict)
    build_image_digest: str = "unresolved"
    research_archive_directory: str = "artifacts/season1/research-releases"
    research_archive_signing_private_key_path: str = ""
    research_archive_signing_key_id: str = ""
    public_arena_enabled: bool = False

    execution_mode: str = "mock"
    live_authorized: bool = False
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Generation traffic may be proxied through Cloudflare, but OpenRouter's
    # historical accounting endpoint must be queried directly. Keeping these
    # URLs separate also prevents gateway failures from erasing billable work.
    openrouter_accounting_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_http_referer: str = "https://epicure.kaikaku.ai/flavourbench"
    openrouter_title: str = "Epicure FlavourBench"
    openrouter_timeout_seconds: int = Field(default=180, ge=10, le=600)
    openrouter_zdr: bool = False
    openrouter_accounting_attempts: int = Field(default=6, ge=1, le=12)
    openrouter_accounting_initial_delay_seconds: float = Field(default=0.5, ge=0, le=10)
    openrouter_max_prompt_price_per_mtok: float | None = Field(default=None, ge=0)
    openrouter_max_completion_price_per_mtok: float | None = Field(default=None, ge=0)
    cloudflare_ai_gateway_token: str = ""

    # Kimi Code is a distinct managed endpoint. It is never sent through
    # OpenRouter and therefore has its own credential, timeout, and catalog
    # identity. The API currently returns token usage but no per-generation
    # charge, so downstream records keep rate-card estimates separate from
    # provider-reconciled cost.
    kimi_api_key: str = ""
    kimi_base_url: str = "https://api.kimi.com/coding"
    kimi_timeout_seconds: int = Field(default=300, ge=10, le=600)

    # Cohere V2 is an exact-model route with independent credentials. It
    # returns token usage and generation IDs but no per-generation charge
    # lookup, so cost records retain their frozen rate-card basis.
    cohere_api_key: str = ""
    cohere_base_url: str = "https://api.cohere.com"
    cohere_timeout_seconds: int = Field(default=300, ge=10, le=600)

    # Alibaba documents DASHSCOPE_API_KEY as the canonical server-side name
    # for QwenCloud pay-as-you-go credentials.  Accept the prefixed deployment
    # spelling as well, while never exposing either variable to the API process.
    qwencloud_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("FLAVOURBENCH_QWENCLOUD_API_KEY", "DASHSCOPE_API_KEY"),
    )
    qwencloud_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    qwencloud_timeout_seconds: int = Field(default=300, ge=10, le=600)

    # Mirrors the explicit Bedrock lane switch read by bedrock_auth.py. The
    # credential itself remains in the AWS SDK environment and is never copied
    # into application settings or API processes.
    bedrock_enabled: bool = False

    mcp_url: str = "http://mcp:8080/mcp"
    mcp_token: str = Field(
        default="",
        validation_alias=AliasChoices("FLAVOURBENCH_MCP_TOKEN", "MCP_API_TOKEN"),
    )
    mcp_timeout_seconds: int = Field(default=15, ge=2, le=60)
    epicure_release_id: str = "unresolved-1790-development-only"
    epicure_bundle_sha256: str = "unresolved"
    epicure_application_sha256: str = "unresolved"
    epicure_tool_schema_sha256: str = "unresolved"

    default_season_slug: str = "season-0"
    model_track_percent: int = Field(default=70, ge=0, le=100)
    max_tool_rounds: int = Field(default=8, ge=1, le=8)
    # OpenRouter's standard tool-calling contract permits a model to request
    # multiple independent functions in one assistant turn. Bound the fan-out
    # explicitly instead of rejecting otherwise valid frontier-model output.
    max_tool_calls_per_round: int = Field(default=4, ge=1, le=16)
    max_tool_calls_total: int = Field(default=16, ge=1, le=64)
    # One network attempt is the safe default. Read timeouts have uncertain
    # delivery semantics and are never retried automatically.
    max_provider_attempts: int = Field(default=1, ge=1, le=3)
    # The retrospective pilot's shorter ceiling produced four provider-declared
    # length completions across three reviewed pairs. The prospective service
    # uses a larger symmetric ceiling and still rejects every non-normal stop.
    max_output_tokens: int = Field(default=4096, ge=128, le=16384)
    max_intermediate_tokens: int = Field(default=700, ge=64, le=8192)
    decoding_temperature: float = Field(default=0.2, ge=0, le=2)
    decoding_top_p: float = Field(default=0.95, gt=0, le=1)
    decoding_seed: int = 20260715
    max_tool_result_bytes: int = Field(default=32_768, ge=1_024, le=262_144)
    max_cumulative_tool_result_bytes: int = Field(default=98_304, ge=1_024, le=1_048_576)
    admission_window_seconds: int = Field(default=60, ge=10, le=3600)
    admission_max_battles: int = Field(default=5, ge=1, le=100)
    admission_network_multiplier: int = Field(default=10, ge=1, le=100)
    retention_days: int = Field(default=30, ge=1, le=365)
    worker_poll_seconds: float = Field(default=1.0, ge=0.1, le=30)
    worker_claim_timeout_seconds: int = Field(default=600, ge=60, le=3600)
    catalog_sync_enabled: bool = False
    catalog_sync_hours: int = Field(default=24, ge=1, le=168)

    @model_validator(mode="after")
    def validate_execution_boundary(self) -> Settings:
        if self.environment not in {"development", "test", "production"}:
            raise ValueError("environment must be development, test, or production")
        if self.service_role not in {"api", "worker", "migration"}:
            raise ValueError("service_role must be api, worker, or migration")
        if self.execution_mode not in {"mock", "live"}:
            raise ValueError("execution_mode must be mock or live")
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in self.active_expert_consent_sha256s
        ):
            raise ValueError("active expert consent hashes must be lowercase sha256 digests")
        activation_manifest_sha256 = self.human_study_activation_manifest_sha256
        if activation_manifest_sha256 and (
            len(activation_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in activation_manifest_sha256)
        ):
            raise ValueError("human-study activation manifest hash must be lowercase sha256")
        validation_packet_sha256 = self.development_task_validation_packet_sha256
        if validation_packet_sha256 and (
            len(validation_packet_sha256) != 64
            or any(character not in "0123456789abcdef" for character in validation_packet_sha256)
        ):
            raise ValueError("development task validation packet hash must be lowercase sha256")
        for label, digest in (
            ("task-validation campaign", self.task_validation_campaign_sha256),
            ("task-validation candidate bundle", self.task_validation_candidate_bundle_sha256),
            ("task-validation assignment", self.task_validation_assignment_sha256),
            (
                "task-validation acquisition receipt",
                self.task_validation_acquisition_receipt_sha256,
            ),
            ("task-validation quality report", self.task_validation_quality_report_sha256),
            ("task-validation readiness decision", self.task_validation_readiness_sha256),
            ("task-validation automated replay", self.task_validation_automated_replay_sha256),
            (
                "task-validation automated replay physical",
                self.task_validation_automated_replay_physical_sha256,
            ),
        ):
            if digest and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{label} hash must be lowercase sha256")
        if self.task_validation_campaign_enabled and not all(
            (
                self.task_validation_season_slug,
                self.task_validation_campaign_path,
                self.task_validation_campaign_sha256,
                self.task_validation_candidate_bundle_path,
                self.task_validation_candidate_bundle_sha256,
                self.task_validation_assignment_path,
                self.task_validation_assignment_sha256,
                self.task_validation_acquisition_receipt_path,
                self.task_validation_acquisition_receipt_sha256,
                self.task_validation_quality_report_path,
                self.task_validation_quality_report_sha256,
                self.task_validation_readiness_path,
                self.task_validation_readiness_sha256,
                self.task_validation_automated_replay_path,
                self.task_validation_automated_replay_sha256,
                self.task_validation_automated_replay_physical_sha256,
            )
        ):
            raise ValueError(
                "enabled task-validation campaign requires every pinned artifact and hash"
            )
        contamination_bundle_sha256 = self.contamination_scan_bundle_sha256
        if contamination_bundle_sha256 and (
            len(contamination_bundle_sha256) != 64
            or any(character not in "0123456789abcdef" for character in contamination_bundle_sha256)
        ):
            raise ValueError("contamination scan bundle hash must be lowercase sha256")
        contamination_calibration_sha256 = self.contamination_calibration_artifact_sha256
        if contamination_calibration_sha256 and (
            len(contamination_calibration_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in contamination_calibration_sha256
            )
        ):
            raise ValueError("contamination calibration artifact hash must be lowercase sha256")
        validator_calibration_sha256 = self.validator_calibration_artifact_sha256
        if validator_calibration_sha256 and (
            len(validator_calibration_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in validator_calibration_sha256
            )
        ):
            raise ValueError("validator calibration artifact hash must be lowercase sha256")
        key_id = self.budget_authorization_signing_key_id
        if (
            not key_id
            or len(key_id) > 64
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in key_id
            )
        ):
            raise ValueError("budget-authorization signing key id is invalid")
        for reviewer_key_id in (
            self.reviewer_identity_hmac_key_id,
            self.reviewer_credential_hmac_key_id,
        ):
            if (
                not reviewer_key_id
                or len(reviewer_key_id) > 64
                or any(
                    character
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                    for character in reviewer_key_id
                )
            ):
                raise ValueError("reviewer HMAC key id is invalid")
        organization_hmac_key_id = self.organization_api_key_hmac_key_id
        if (
            not organization_hmac_key_id
            or len(organization_hmac_key_id) > 64
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in organization_hmac_key_id
            )
        ):
            raise ValueError("organization API-key HMAC key id is invalid")
        for (
            verification_key_id,
            verification_secret,
        ) in self.organization_api_key_hmac_verification_keys.items():
            if (
                not verification_key_id
                or len(verification_key_id) > 64
                or any(
                    character
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                    for character in verification_key_id
                )
                or not verification_secret
            ):
                raise ValueError("organization API-key HMAC keyring is invalid")
        configured_organization_key = self.organization_api_key_hmac_verification_keys.get(
            organization_hmac_key_id
        )
        if (
            configured_organization_key is not None
            and configured_organization_key != self.organization_api_key_hmac_secret
        ):
            raise ValueError(
                "current organization API-key HMAC key conflicts with the verification keyring"
            )
        run_card_key_id = self.run_card_signing_key_id
        if (
            not run_card_key_id
            or len(run_card_key_id) > 64
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in run_card_key_id
            )
        ):
            raise ValueError("run-card signing key id is invalid")
        for verification_key_id, verification_secret in self.run_card_verification_keys.items():
            if (
                not verification_key_id
                or len(verification_key_id) > 64
                or any(
                    character
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                    for character in verification_key_id
                )
                or not verification_secret
            ):
                raise ValueError("run-card verification keyring is invalid")
        configured_run_card_key = self.run_card_verification_keys.get(run_card_key_id)
        if (
            configured_run_card_key is not None
            and configured_run_card_key != self.run_card_signing_secret
        ):
            raise ValueError("current run-card signing key conflicts with the verification keyring")
        for (
            verification_key_id,
            verification_secret,
        ) in self.budget_authorization_verification_keys.items():
            if (
                not verification_key_id
                or len(verification_key_id) > 64
                or any(
                    character
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                    for character in verification_key_id
                )
                or not verification_secret
            ):
                raise ValueError("budget-authorization verification keyring is invalid")
        configured_current = self.budget_authorization_verification_keys.get(key_id)
        if (
            configured_current is not None
            and configured_current != self.budget_authorization_signing_secret
        ):
            raise ValueError(
                "current budget-authorization key id conflicts with the verification keyring"
            )
        if self.execution_mode == "live":
            if not self.live_authorized:
                raise ValueError("live execution requires FLAVOURBENCH_LIVE_AUTHORIZED=true")
            if self.service_role == "worker" and not any(
                (
                    self.openrouter_api_key,
                    self.kimi_api_key,
                    self.cohere_api_key,
                    self.qwencloud_api_key,
                    self.bedrock_enabled,
                )
            ):
                raise ValueError(
                    "live worker execution requires at least one provider credential: "
                    "an OpenRouter, Kimi, Cohere, or QwenCloud API key, or enabled Bedrock identity"
                )
        if self.environment == "production":
            if self.service_role != "migration" and (
                self.execution_mode != "live" or not self.live_authorized
            ):
                raise ValueError("production requires explicitly authorized live execution")
            if not self.database_url.startswith(("postgresql://", "postgresql+")):
                raise ValueError("production requires PostgreSQL")
            if self.auto_create_schema:
                raise ValueError("production requires migration-managed schema")
            secrets_by_name = (
                {
                    "service token": self.service_token,
                    "admin token": self.admin_token,
                    "expert token": self.expert_token,
                    "pseudonym secret": self.pseudonym_secret,
                    "task-validator identity HMAC secret": (
                        self.task_validator_identity_hmac_secret
                    ),
                    "reviewer identity HMAC secret": self.reviewer_identity_hmac_secret,
                    "reviewer credential HMAC secret": self.reviewer_credential_hmac_secret,
                    "organization API-key HMAC secret": (self.organization_api_key_hmac_secret),
                    "run-card signing secret": self.run_card_signing_secret,
                    "budget-authorization signing secret": (
                        self.budget_authorization_signing_secret
                    ),
                }
                if self.service_role == "api"
                else {
                    "budget-authorization signing secret": (
                        self.budget_authorization_signing_secret
                    )
                }
                if self.service_role == "worker"
                else {}
            )
            defaults = {
                "development-only-service-token",
                "development-only-pseudonym-secret",
                "development-only-task-validator-identity-hmac-secret",
                "development-only-reviewer-identity-hmac-secret",
                "development-only-reviewer-credential-hmac-secret",
                "development-only-organization-api-key-hmac-secret",
                "development-only-run-card-signing-secret",
                "development-only-budget-authorization-signing-secret",
            }
            for name, value in secrets_by_name.items():
                if len(value) < 32 or value in defaults:
                    raise ValueError(f"production requires a unique {name} of 32+ characters")
            if any(
                len(value) < 32 for value in self.budget_authorization_verification_keys.values()
            ):
                raise ValueError(
                    "production budget-authorization verification keys must be 32+ characters"
                )
            if any(
                len(value) < 32
                for value in self.organization_api_key_hmac_verification_keys.values()
            ):
                raise ValueError(
                    "production organization API-key HMAC verification keys must be 32+ characters"
                )
            if any(len(value) < 32 for value in self.run_card_verification_keys.values()):
                raise ValueError("production run-card verification keys must be 32+ characters")
            if len(set(secrets_by_name.values())) != len(secrets_by_name):
                raise ValueError("production authentication secrets must be distinct")
            if self.service_role == "api" and any(
                (
                    self.openrouter_api_key,
                    self.kimi_api_key,
                    self.cohere_api_key,
                    self.qwencloud_api_key,
                    self.mcp_token,
                    self.cloudflare_ai_gateway_token,
                )
            ):
                raise ValueError("production API must not receive provider or MCP credentials")
            if self.service_role == "worker" and not self.mcp_token:
                raise ValueError("production worker requires an authenticated Epicure MCP token")
            if self.service_role == "worker" and (
                "gateway.ai.cloudflare.com" in self.openrouter_base_url
                and not self.cloudflare_ai_gateway_token
            ):
                raise ValueError("production Cloudflare routing requires a gateway token")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def budget_authorization_verification_keyring(
    settings: Settings | None = None,
) -> Mapping[str, str]:
    """Return current and retained historical budget-signature verification keys."""

    configured = settings or get_settings()
    keys = dict(getattr(configured, "budget_authorization_verification_keys", {}))
    key_id = getattr(configured, "budget_authorization_signing_key_id", "primary")
    keys[key_id] = configured.budget_authorization_signing_secret
    return keys


def organization_api_key_hmac_keyring(
    settings: Settings | None = None,
) -> Mapping[str, str]:
    """Return the active and retained organization API-key HMAC peppers."""

    configured = settings or get_settings()
    keys = dict(configured.organization_api_key_hmac_verification_keys)
    keys[configured.organization_api_key_hmac_key_id] = configured.organization_api_key_hmac_secret
    return keys


def run_card_verification_keyring(
    settings: Settings | None = None,
) -> Mapping[str, str]:
    """Return current and retained run-card HMAC verification keys."""

    configured = settings or get_settings()
    keys = dict(configured.run_card_verification_keys)
    keys[configured.run_card_signing_key_id] = configured.run_card_signing_secret
    return keys
