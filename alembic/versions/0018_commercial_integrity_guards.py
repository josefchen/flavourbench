"""Enforce commercial-record immutability and tenant integrity in PostgreSQL.

Revision ID: 0018_commercial_integrity_guards
Revises: 0017_model_company_onboarding
"""

from __future__ import annotations

from alembic import op

revision = "0018_commercial_integrity_guards"
down_revision = "0017_model_company_onboarding"
branch_labels = None
depends_on = None


UPDATE_GUARD_FUNCTION = "flavourbench_commercial_update_guard"
DELETE_GUARD_FUNCTION = "flavourbench_commercial_delete_guard"
TENANT_GUARD_FUNCTION = "flavourbench_commercial_tenant_guard"
INSERT_GUARD_FUNCTION = "flavourbench_commercial_insert_guard"
ARM_BINDING_GUARD_FUNCTION = "flavourbench_commercial_arm_binding_guard"


def _postgres_upgrade() -> None:
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION public.{UPDATE_GUARD_FUNCTION}()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            immutable_old jsonb;
            immutable_new jsonb;
            permitted_fields text[];
        BEGIN
            IF TG_TABLE_NAME = 'organizations' THEN
                immutable_old := to_jsonb(OLD) - ARRAY[
                    'status', 'verified_at', 'suspended_at', 'closed_at'
                ];
                immutable_new := to_jsonb(NEW) - ARRAY[
                    'status', 'verified_at', 'suspended_at', 'closed_at'
                ];
                IF immutable_old IS DISTINCT FROM immutable_new THEN
                    RAISE EXCEPTION 'organization contract is immutable';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status AND NOT (
                    (OLD.status = 'pending_verification' AND NEW.status IN ('active', 'closed')) OR
                    (OLD.status = 'active' AND NEW.status IN ('suspended', 'closed')) OR
                    (OLD.status = 'suspended' AND NEW.status = 'closed')
                ) THEN
                    RAISE EXCEPTION 'organization lifecycle transition is invalid';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status THEN
                    permitted_fields := CASE
                        WHEN OLD.status = 'pending_verification' AND NEW.status = 'active'
                            THEN ARRAY['status', 'verified_at']
                        WHEN NEW.status = 'suspended'
                            THEN ARRAY['status', 'suspended_at']
                        ELSE ARRAY['status', 'closed_at']
                    END;
                    IF to_jsonb(OLD) - permitted_fields IS DISTINCT FROM
                       to_jsonb(NEW) - permitted_fields THEN
                        RAISE EXCEPTION 'organization lifecycle evidence is write-once';
                    END IF;
                END IF;
                IF OLD.status IS NOT DISTINCT FROM NEW.status AND (
                    OLD.verified_at IS DISTINCT FROM NEW.verified_at OR
                    OLD.suspended_at IS DISTINCT FROM NEW.suspended_at OR
                    OLD.closed_at IS DISTINCT FROM NEW.closed_at
                ) THEN
                    RAISE EXCEPTION 'organization lifecycle timestamp requires transition';
                END IF;
                IF NEW.status = 'active' AND NEW.verified_at IS NULL THEN
                    RAISE EXCEPTION 'active organization requires verification evidence';
                END IF;
                IF NEW.status = 'suspended' AND NEW.suspended_at IS NULL THEN
                    RAISE EXCEPTION 'suspended organization requires timestamp';
                END IF;
                IF NEW.status = 'closed' AND NEW.closed_at IS NULL THEN
                    RAISE EXCEPTION 'closed organization requires timestamp';
                END IF;

            ELSIF TG_TABLE_NAME = 'organization_api_keys' THEN
                immutable_old := to_jsonb(OLD) - ARRAY['status', 'last_used_at', 'revoked_at'];
                immutable_new := to_jsonb(NEW) - ARRAY['status', 'last_used_at', 'revoked_at'];
                IF immutable_old IS DISTINCT FROM immutable_new THEN
                    RAISE EXCEPTION 'organization API-key contract is immutable';
                END IF;
                IF OLD.last_used_at IS NOT NULL AND (
                    NEW.last_used_at IS NULL OR NEW.last_used_at < OLD.last_used_at
                ) THEN
                    RAISE EXCEPTION 'organization API-key last-used time cannot move backward';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status AND NOT (
                    OLD.status = 'active' AND NEW.status = 'revoked' AND NEW.revoked_at IS NOT NULL
                ) THEN
                    RAISE EXCEPTION 'organization API key may only transition active to revoked';
                END IF;
                IF OLD.status IS NOT DISTINCT FROM NEW.status AND
                   OLD.revoked_at IS DISTINCT FROM NEW.revoked_at THEN
                    RAISE EXCEPTION 'organization API-key revocation requires status transition';
                END IF;

            ELSIF TG_TABLE_NAME = 'model_submissions' THEN
                immutable_old := to_jsonb(OLD) - ARRAY[
                    'status', 'catalog_model_id', 'decision_reference_sha256',
                    'submitted_at', 'decided_at', 'suspended_at'
                ];
                immutable_new := to_jsonb(NEW) - ARRAY[
                    'status', 'catalog_model_id', 'decision_reference_sha256',
                    'submitted_at', 'decided_at', 'suspended_at'
                ];
                IF immutable_old IS DISTINCT FROM immutable_new THEN
                    RAISE EXCEPTION 'model submission content is immutable';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status AND NOT (
                    (OLD.status = 'draft' AND NEW.status IN ('submitted', 'withdrawn')) OR
                    (OLD.status = 'submitted' AND NEW.status IN (
                        'approved', 'changes_requested', 'rejected', 'withdrawn'
                    )) OR
                    (OLD.status = 'approved' AND NEW.status IN ('suspended', 'retired')) OR
                    (OLD.status = 'changes_requested' AND NEW.status = 'withdrawn') OR
                    (OLD.status = 'suspended' AND NEW.status = 'retired')
                ) THEN
                    RAISE EXCEPTION 'model submission lifecycle transition is invalid';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status THEN
                    permitted_fields := CASE
                        WHEN NEW.status = 'submitted'
                            THEN ARRAY['status', 'submitted_at']
                        WHEN NEW.status = 'approved'
                            THEN ARRAY[
                                'status', 'catalog_model_id', 'decision_reference_sha256',
                                'decided_at'
                            ]
                        WHEN NEW.status IN ('changes_requested', 'rejected')
                            THEN ARRAY['status', 'decision_reference_sha256', 'decided_at']
                        WHEN NEW.status = 'suspended'
                            THEN ARRAY['status', 'suspended_at']
                        ELSE ARRAY['status']
                    END;
                    IF to_jsonb(OLD) - permitted_fields IS DISTINCT FROM
                       to_jsonb(NEW) - permitted_fields THEN
                        RAISE EXCEPTION 'model submission lifecycle evidence is write-once';
                    END IF;
                END IF;
                IF OLD.status IS NOT DISTINCT FROM NEW.status AND (
                    OLD.catalog_model_id IS DISTINCT FROM NEW.catalog_model_id OR
                    OLD.decision_reference_sha256 IS DISTINCT FROM NEW.decision_reference_sha256 OR
                    OLD.submitted_at IS DISTINCT FROM NEW.submitted_at OR
                    OLD.decided_at IS DISTINCT FROM NEW.decided_at OR
                    OLD.suspended_at IS DISTINCT FROM NEW.suspended_at
                ) THEN
                    RAISE EXCEPTION 'model submission metadata requires transition';
                END IF;
                IF NEW.status = 'submitted' AND NEW.submitted_at IS NULL THEN
                    RAISE EXCEPTION 'submitted model requires timestamp';
                END IF;
                IF NEW.status IN ('approved', 'changes_requested', 'rejected') AND (
                    NEW.decided_at IS NULL OR NEW.decision_reference_sha256 IS NULL OR
                    (NEW.status = 'approved' AND NEW.catalog_model_id IS NULL)
                ) THEN
                    RAISE EXCEPTION 'model decision evidence is incomplete';
                END IF;

            ELSIF TG_TABLE_NAME = 'model_route_revisions' THEN
                immutable_old := to_jsonb(OLD) - ARRAY[
                    'status', 'approved_contract_test_id', 'approved_season_id',
                    'approved_season_manifest_sha256', 'approved_endpoint_contract_sha256',
                    'valid_until', 'submitted_at', 'approved_at', 'suspended_at', 'retired_at'
                ];
                immutable_new := to_jsonb(NEW) - ARRAY[
                    'status', 'approved_contract_test_id', 'approved_season_id',
                    'approved_season_manifest_sha256', 'approved_endpoint_contract_sha256',
                    'valid_until', 'submitted_at', 'approved_at', 'suspended_at', 'retired_at'
                ];
                IF immutable_old IS DISTINCT FROM immutable_new THEN
                    RAISE EXCEPTION 'model route content is immutable';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status AND NOT (
                    (OLD.status = 'draft' AND NEW.status IN ('submitted', 'withdrawn')) OR
                    (OLD.status = 'submitted' AND NEW.status IN (
                        'contract_testing', 'approved', 'changes_requested', 'rejected', 'withdrawn'
                    )) OR
                    (OLD.status = 'contract_testing' AND NEW.status IN (
                        'approved', 'changes_requested', 'rejected'
                    )) OR
                    (OLD.status = 'approved' AND NEW.status IN ('suspended', 'retired')) OR
                    (OLD.status = 'changes_requested' AND NEW.status = 'withdrawn') OR
                    (OLD.status = 'suspended' AND NEW.status = 'retired')
                ) THEN
                    RAISE EXCEPTION 'model route lifecycle transition is invalid';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status THEN
                    permitted_fields := CASE
                        WHEN NEW.status = 'submitted'
                            THEN ARRAY['status', 'submitted_at']
                        WHEN NEW.status = 'approved'
                            THEN ARRAY[
                                'status', 'approved_contract_test_id', 'approved_season_id',
                                'approved_season_manifest_sha256',
                                'approved_endpoint_contract_sha256', 'valid_until', 'approved_at'
                            ]
                        WHEN NEW.status = 'suspended'
                            THEN ARRAY['status', 'suspended_at']
                        WHEN NEW.status = 'retired'
                            THEN ARRAY['status', 'retired_at']
                        ELSE ARRAY['status']
                    END;
                    IF to_jsonb(OLD) - permitted_fields IS DISTINCT FROM
                       to_jsonb(NEW) - permitted_fields THEN
                        RAISE EXCEPTION 'model route approval evidence is write-once';
                    END IF;
                END IF;
                IF OLD.status IS NOT DISTINCT FROM NEW.status AND (
                    OLD.approved_contract_test_id IS DISTINCT FROM NEW.approved_contract_test_id OR
                    OLD.approved_season_id IS DISTINCT FROM NEW.approved_season_id OR
                    OLD.approved_season_manifest_sha256 IS DISTINCT FROM
                        NEW.approved_season_manifest_sha256 OR
                    OLD.approved_endpoint_contract_sha256 IS DISTINCT FROM
                        NEW.approved_endpoint_contract_sha256 OR
                    OLD.valid_until IS DISTINCT FROM NEW.valid_until OR
                    OLD.submitted_at IS DISTINCT FROM NEW.submitted_at OR
                    OLD.approved_at IS DISTINCT FROM NEW.approved_at OR
                    OLD.suspended_at IS DISTINCT FROM NEW.suspended_at OR
                    OLD.retired_at IS DISTINCT FROM NEW.retired_at
                ) THEN
                    RAISE EXCEPTION 'model route metadata requires transition';
                END IF;
                IF NEW.status = 'submitted' AND NEW.submitted_at IS NULL THEN
                    RAISE EXCEPTION 'submitted model route requires timestamp';
                END IF;
                IF NEW.status = 'approved' AND (
                    NEW.approved_at IS NULL OR NEW.approved_season_id IS NULL OR
                    NEW.approved_season_manifest_sha256 IS NULL OR
                    NEW.approved_endpoint_contract_sha256 IS NULL
                ) THEN
                    RAISE EXCEPTION 'model route approval evidence is incomplete';
                END IF;

            ELSIF TG_TABLE_NAME = 'route_contract_tests' THEN
                immutable_old := to_jsonb(OLD) - ARRAY[
                    'status', 'request_sha256', 'response_sha256', 'tool_trace_sha256',
                    'structured_output_sha256', 'observed_model_id', 'observed_provider_slug',
                    'check_results_json', 'check_results_sha256', 'generation_id', 'usage_json',
                    'cost_micros', 'cost_accounting_basis', 'latency_ms', 'failure_code',
                    'incident_id', 'started_at', 'completed_at', 'valid_until'
                ];
                immutable_new := to_jsonb(NEW) - ARRAY[
                    'status', 'request_sha256', 'response_sha256', 'tool_trace_sha256',
                    'structured_output_sha256', 'observed_model_id', 'observed_provider_slug',
                    'check_results_json', 'check_results_sha256', 'generation_id', 'usage_json',
                    'cost_micros', 'cost_accounting_basis', 'latency_ms', 'failure_code',
                    'incident_id', 'started_at', 'completed_at', 'valid_until'
                ];
                IF immutable_old IS DISTINCT FROM immutable_new THEN
                    RAISE EXCEPTION 'route contract-test contract is immutable';
                END IF;
                IF OLD.status IN ('passed', 'failed', 'inconclusive', 'cancelled') THEN
                    RAISE EXCEPTION 'terminal route contract tests are immutable';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status AND NOT (
                    (OLD.status = 'queued' AND NEW.status IN ('running', 'cancelled')) OR
                    (OLD.status = 'running' AND NEW.status IN (
                        'passed', 'failed', 'inconclusive', 'cancelled'
                    ))
                ) THEN
                    RAISE EXCEPTION 'route contract-test lifecycle transition is invalid';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status THEN
                    permitted_fields := CASE
                        WHEN NEW.status = 'running'
                            THEN ARRAY['status', 'request_sha256', 'started_at']
                        ELSE ARRAY[
                            'status', 'response_sha256', 'tool_trace_sha256',
                            'structured_output_sha256', 'observed_model_id',
                            'observed_provider_slug', 'check_results_json',
                            'check_results_sha256', 'generation_id', 'usage_json',
                            'cost_micros', 'cost_accounting_basis', 'latency_ms',
                            'failure_code', 'incident_id', 'completed_at', 'valid_until'
                        ]
                    END;
                    IF to_jsonb(OLD) - permitted_fields IS DISTINCT FROM
                       to_jsonb(NEW) - permitted_fields THEN
                        RAISE EXCEPTION 'route contract-test evidence is write-once';
                    END IF;
                END IF;
                IF OLD.status IS NOT DISTINCT FROM NEW.status AND
                   to_jsonb(OLD) IS DISTINCT FROM to_jsonb(NEW) THEN
                    RAISE EXCEPTION 'route contract-test evidence requires transition';
                END IF;
                IF NEW.status = 'running' AND NEW.started_at IS NULL THEN
                    RAISE EXCEPTION 'running route contract test requires timestamp';
                END IF;
                IF NEW.status IN ('passed', 'failed', 'inconclusive', 'cancelled') AND
                   NEW.completed_at IS NULL THEN
                    RAISE EXCEPTION 'terminal route contract test requires timestamp';
                END IF;
                IF NEW.status = 'passed' AND (
                    NEW.request_sha256 IS NULL OR NEW.response_sha256 IS NULL OR
                    NEW.tool_trace_sha256 IS NULL OR NEW.structured_output_sha256 IS NULL OR
                    NEW.observed_model_id IS NULL OR NEW.observed_provider_slug IS NULL OR
                    NEW.check_results_sha256 IS NULL OR NEW.generation_id IS NULL OR
                    NEW.valid_until IS NULL
                ) THEN
                    RAISE EXCEPTION 'passed route contract-test evidence is incomplete';
                END IF;

            ELSIF TG_TABLE_NAME = 'evaluation_orders' THEN
                immutable_old := to_jsonb(OLD) - ARRAY[
                    'status', 'billing_status', 'publication_status',
                    'quote_reference_sha256', 'submitted_at', 'approved_at', 'started_at',
                    'completed_at', 'delivered_at'
                ];
                immutable_new := to_jsonb(NEW) - ARRAY[
                    'status', 'billing_status', 'publication_status',
                    'quote_reference_sha256', 'submitted_at', 'approved_at', 'started_at',
                    'completed_at', 'delivered_at'
                ];
                IF immutable_old IS DISTINCT FROM immutable_new THEN
                    RAISE EXCEPTION 'evaluation-order contract is immutable';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status AND NOT (
                    (OLD.status = 'draft' AND NEW.status IN ('submitted', 'cancelled')) OR
                    (OLD.status = 'submitted' AND NEW.status IN (
                        'approved', 'rejected', 'cancelled'
                    )) OR
                    (OLD.status = 'approved' AND NEW.status IN ('provisioning', 'cancelled')) OR
                    (OLD.status = 'provisioning' AND NEW.status IN (
                        'ready', 'failed', 'cancelling'
                    )) OR
                    (OLD.status = 'ready' AND NEW.status IN ('running', 'cancelled')) OR
                    (OLD.status = 'running' AND NEW.status IN (
                        'collection_complete', 'cancelling', 'failed'
                    )) OR
                    (OLD.status = 'collection_complete' AND NEW.status IN (
                        'analysis_complete', 'failed'
                    )) OR
                    (OLD.status = 'analysis_complete' AND NEW.status IN ('delivered', 'failed')) OR
                    (OLD.status = 'cancelling' AND NEW.status IN ('cancelled', 'failed'))
                ) THEN
                    RAISE EXCEPTION 'evaluation-order lifecycle transition is invalid';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status THEN
                    permitted_fields := CASE
                        WHEN NEW.status = 'submitted'
                            THEN ARRAY['status', 'submitted_at']
                        WHEN NEW.status = 'approved'
                            THEN ARRAY[
                                'status', 'billing_status', 'quote_reference_sha256', 'approved_at'
                            ]
                        WHEN NEW.status IN ('rejected', 'cancelled')
                            THEN ARRAY['status', 'billing_status']
                        WHEN NEW.status = 'provisioning'
                            THEN ARRAY['status']
                        WHEN NEW.status = 'running'
                            THEN ARRAY['status', 'started_at']
                        WHEN NEW.status = 'analysis_complete'
                            THEN ARRAY['status', 'completed_at']
                        WHEN NEW.status = 'delivered'
                            THEN ARRAY['status', 'delivered_at']
                        ELSE ARRAY['status']
                    END;
                    IF to_jsonb(OLD) - permitted_fields IS DISTINCT FROM
                       to_jsonb(NEW) - permitted_fields THEN
                        RAISE EXCEPTION 'evaluation-order lifecycle evidence is write-once';
                    END IF;
                ELSIF OLD.publication_status IS NOT DISTINCT FROM NEW.publication_status OR
                      to_jsonb(OLD) - 'publication_status' IS DISTINCT FROM
                          to_jsonb(NEW) - 'publication_status' THEN
                    RAISE EXCEPTION 'evaluation-order metadata requires transition';
                END IF;
                IF OLD.publication_status IS DISTINCT FROM NEW.publication_status AND NOT (
                    (OLD.publication_status = 'private' AND
                     NEW.publication_status = 'authorized') OR
                    (OLD.publication_status = 'authorized' AND NEW.publication_status IN (
                        'published', 'withdrawn'
                    )) OR
                    (OLD.publication_status = 'published' AND NEW.publication_status = 'withdrawn')
                ) THEN
                    RAISE EXCEPTION 'evaluation-order publication transition is invalid';
                END IF;
                IF NEW.status = 'submitted' AND NEW.submitted_at IS NULL THEN
                    RAISE EXCEPTION 'submitted evaluation order requires timestamp';
                END IF;
                IF NEW.status = 'approved' AND (
                    NEW.approved_at IS NULL OR NEW.billing_status <> 'authorized' OR
                    NEW.quote_reference_sha256 IS NULL
                ) THEN
                    RAISE EXCEPTION 'evaluation-order approval evidence is incomplete';
                END IF;
                IF NEW.status = 'provisioning' AND NOT EXISTS (
                    SELECT 1 FROM public.controlled_runs
                    WHERE evaluation_order_id = NEW.id
                ) THEN
                    RAISE EXCEPTION 'provisioning order requires controlled run';
                END IF;

            ELSIF TG_TABLE_NAME = 'evidence_bundles' THEN
                immutable_old := to_jsonb(OLD) - ARRAY[
                    'status', 'archive_sha256', 'storage_object_key', 'size_bytes',
                    'signature_algorithm', 'signing_key_id', 'signature_base64',
                    'sealed_at', 'available_until', 'revoked_at'
                ];
                immutable_new := to_jsonb(NEW) - ARRAY[
                    'status', 'archive_sha256', 'storage_object_key', 'size_bytes',
                    'signature_algorithm', 'signing_key_id', 'signature_base64',
                    'sealed_at', 'available_until', 'revoked_at'
                ];
                IF immutable_old IS DISTINCT FROM immutable_new THEN
                    RAISE EXCEPTION 'evidence-bundle manifest is immutable';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status AND NOT (
                    (OLD.status = 'building' AND NEW.status IN ('sealed', 'failed')) OR
                    (OLD.status = 'sealed' AND NEW.status IN (
                        'available', 'superseded', 'revoked'
                    )) OR
                    (OLD.status = 'available' AND NEW.status IN ('superseded', 'revoked'))
                ) THEN
                    RAISE EXCEPTION 'evidence-bundle lifecycle transition is invalid';
                END IF;
                IF OLD.status IS DISTINCT FROM NEW.status THEN
                    permitted_fields := CASE
                        WHEN NEW.status = 'sealed'
                            THEN ARRAY[
                                'status', 'archive_sha256', 'storage_object_key', 'size_bytes',
                                'signature_algorithm', 'signing_key_id', 'signature_base64',
                                'sealed_at', 'available_until'
                            ]
                        WHEN NEW.status = 'revoked'
                            THEN ARRAY['status', 'revoked_at']
                        ELSE ARRAY['status']
                    END;
                    IF to_jsonb(OLD) - permitted_fields IS DISTINCT FROM
                       to_jsonb(NEW) - permitted_fields THEN
                        RAISE EXCEPTION 'evidence-bundle archive evidence is write-once';
                    END IF;
                ELSE
                    RAISE EXCEPTION 'evidence-bundle archive metadata requires transition';
                END IF;
                IF NEW.status IN ('sealed', 'available', 'superseded', 'revoked') AND (
                    NEW.archive_sha256 IS NULL OR NEW.storage_object_key IS NULL OR
                    NEW.size_bytes IS NULL OR NEW.signature_algorithm IS NULL OR
                    NEW.signing_key_id IS NULL OR NEW.signature_base64 IS NULL OR
                    NEW.sealed_at IS NULL
                ) THEN
                    RAISE EXCEPTION 'sealed evidence-bundle evidence is incomplete';
                END IF;
                IF NEW.status = 'revoked' AND NEW.revoked_at IS NULL THEN
                    RAISE EXCEPTION 'revoked evidence bundle requires timestamp';
                END IF;

            ELSE
                RAISE EXCEPTION
                    'commercial update guard attached to unsupported table %', TG_TABLE_NAME;
            END IF;
            RETURN NEW;
        END;
        $$;

        CREATE OR REPLACE FUNCTION public.{DELETE_GUARD_FUNCTION}()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION '% records are append-only', TG_TABLE_NAME;
        END;
        $$;

        CREATE OR REPLACE FUNCTION public.{INSERT_GUARD_FUNCTION}()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF TG_TABLE_NAME = 'organizations' THEN
                IF NEW.status NOT IN ('pending_verification', 'active') OR
                   (NEW.status = 'active' AND NEW.verified_at IS NULL) OR
                   (NEW.status = 'pending_verification' AND NEW.verified_at IS NOT NULL) OR
                   NEW.suspended_at IS NOT NULL OR NEW.closed_at IS NOT NULL THEN
                    RAISE EXCEPTION 'organization insert state is invalid';
                END IF;
            ELSIF TG_TABLE_NAME = 'organization_api_keys' THEN
                IF NEW.status <> 'active' OR NEW.revoked_at IS NOT NULL OR
                   NEW.expires_at <= NEW.not_before OR NOT EXISTS (
                       SELECT 1 FROM public.organizations
                       WHERE id = NEW.organization_id AND status = 'active'
                   ) THEN
                    RAISE EXCEPTION 'organization API-key insert state is invalid';
                END IF;
            ELSIF TG_TABLE_NAME = 'governance_acceptances' THEN
                IF NEW.status <> 'active' OR NEW.revoked_at IS NOT NULL OR
                   NEW.accepted_at > CURRENT_TIMESTAMP OR
                   (NEW.expires_at IS NOT NULL AND NEW.expires_at <= NEW.accepted_at) THEN
                    RAISE EXCEPTION 'governance-acceptance insert state is invalid';
                END IF;
            ELSIF TG_TABLE_NAME = 'model_submissions' THEN
                IF NEW.status <> 'draft' OR NEW.catalog_model_id IS NOT NULL OR
                   NEW.decision_reference_sha256 IS NOT NULL OR
                   NEW.submitted_at IS NOT NULL OR NEW.decided_at IS NOT NULL OR
                   NEW.suspended_at IS NOT NULL THEN
                    RAISE EXCEPTION 'model-submission insert state is invalid';
                END IF;
            ELSIF TG_TABLE_NAME = 'model_route_revisions' THEN
                IF NEW.status <> 'draft' OR NEW.approved_contract_test_id IS NOT NULL OR
                   NEW.approved_season_id IS NOT NULL OR
                   NEW.approved_season_manifest_sha256 IS NOT NULL OR
                   NEW.approved_endpoint_contract_sha256 IS NOT NULL OR
                   NEW.valid_until IS NOT NULL OR NEW.submitted_at IS NOT NULL OR
                   NEW.approved_at IS NOT NULL OR NEW.suspended_at IS NOT NULL OR
                   NEW.retired_at IS NOT NULL THEN
                    RAISE EXCEPTION 'model-route insert state is invalid';
                END IF;
            ELSIF TG_TABLE_NAME = 'route_contract_tests' THEN
                IF NEW.status <> 'queued' OR NEW.request_sha256 IS NOT NULL OR
                   NEW.response_sha256 IS NOT NULL OR NEW.tool_trace_sha256 IS NOT NULL OR
                   NEW.structured_output_sha256 IS NOT NULL OR
                   NEW.observed_model_id IS NOT NULL OR
                   NEW.observed_provider_slug IS NOT NULL OR
                   NEW.check_results_sha256 IS NOT NULL OR NEW.generation_id IS NOT NULL OR
                   NEW.cost_micros <> 0 OR NEW.latency_ms IS NOT NULL OR
                   NEW.failure_code IS NOT NULL OR NEW.incident_id IS NOT NULL OR
                   NEW.started_at IS NOT NULL OR NEW.completed_at IS NOT NULL OR
                   NEW.valid_until IS NOT NULL THEN
                    RAISE EXCEPTION 'route contract-test insert state is invalid';
                END IF;
            ELSIF TG_TABLE_NAME = 'evaluation_orders' THEN
                IF NEW.status <> 'draft' OR NEW.billing_status <> 'unquoted' OR
                   NEW.publication_status <> 'private' OR
                   NEW.quote_reference_sha256 IS NOT NULL OR NEW.submitted_at IS NOT NULL OR
                   NEW.approved_at IS NOT NULL OR NEW.started_at IS NOT NULL OR
                   NEW.completed_at IS NOT NULL OR NEW.delivered_at IS NOT NULL THEN
                    RAISE EXCEPTION 'evaluation-order insert state is invalid';
                END IF;
            ELSIF TG_TABLE_NAME = 'evidence_bundles' THEN
                IF NEW.status <> 'building' OR NEW.archive_sha256 IS NOT NULL OR
                   NEW.storage_object_key IS NOT NULL OR NEW.size_bytes IS NOT NULL OR
                   NEW.signature_algorithm IS NOT NULL OR NEW.signing_key_id IS NOT NULL OR
                   NEW.signature_base64 IS NOT NULL OR NEW.sealed_at IS NOT NULL OR
                   NEW.available_until IS NOT NULL OR NEW.revoked_at IS NOT NULL THEN
                    RAISE EXCEPTION 'evidence-bundle insert state is invalid';
                END IF;
            ELSIF TG_TABLE_NAME = 'api_idempotency_keys' THEN
                IF NEW.expires_at <= NEW.created_at OR
                   NEW.response_status < 100 OR NEW.response_status > 599 THEN
                    RAISE EXCEPTION 'API idempotency insert state is invalid';
                END IF;
            ELSE
                RAISE EXCEPTION
                    'commercial insert guard attached to unsupported table %', TG_TABLE_NAME;
            END IF;
            RETURN NEW;
        END;
        $$;

        CREATE OR REPLACE FUNCTION public.{TENANT_GUARD_FUNCTION}()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            expected_organization_id varchar(36);
        BEGIN
            IF TG_TABLE_NAME = 'governance_acceptances' THEN
                IF NEW.model_submission_id IS NOT NULL THEN
                    SELECT organization_id INTO expected_organization_id
                    FROM public.model_submissions WHERE id = NEW.model_submission_id;
                ELSIF NEW.route_revision_id IS NOT NULL THEN
                    SELECT submission.organization_id INTO expected_organization_id
                    FROM public.model_route_revisions AS route
                    JOIN public.model_submissions AS submission
                      ON submission.id = route.model_submission_id
                    WHERE route.id = NEW.route_revision_id;
                ELSIF NEW.evaluation_order_id IS NOT NULL THEN
                    SELECT organization_id INTO expected_organization_id
                    FROM public.evaluation_orders WHERE id = NEW.evaluation_order_id;
                ELSE
                    expected_organization_id := NEW.organization_id;
                END IF;
            ELSIF TG_TABLE_NAME = 'evaluation_orders' THEN
                SELECT submission.organization_id INTO expected_organization_id
                FROM public.model_submissions AS submission
                JOIN public.model_route_revisions AS route
                  ON route.id = NEW.route_revision_id
                 AND route.model_submission_id = submission.id
                WHERE submission.id = NEW.model_submission_id;
            ELSIF TG_TABLE_NAME = 'controlled_runs' THEN
                IF TG_OP = 'UPDATE' THEN
                    IF OLD.organization_id IS DISTINCT FROM NEW.organization_id OR
                       OLD.evaluation_order_id IS DISTINCT FROM NEW.evaluation_order_id OR
                       OLD.route_revision_id IS DISTINCT FROM NEW.route_revision_id OR
                       OLD.endpoint_descriptor_sha256 IS DISTINCT FROM
                           NEW.endpoint_descriptor_sha256 THEN
                        RAISE EXCEPTION 'commercial controlled-run binding is immutable';
                    END IF;
                    RETURN NEW;
                END IF;
                IF NEW.organization_id IS NULL AND NEW.evaluation_order_id IS NULL AND
                   NEW.route_revision_id IS NULL AND
                   NEW.endpoint_descriptor_sha256 IS NULL THEN
                    RETURN NEW;
                END IF;
                IF NEW.organization_id IS NULL OR NEW.evaluation_order_id IS NULL OR
                   NEW.route_revision_id IS NULL OR
                   NEW.endpoint_descriptor_sha256 IS NULL THEN
                    RAISE EXCEPTION 'commercial controlled-run binding is incomplete';
                END IF;
                SELECT orders.organization_id INTO expected_organization_id
                FROM public.evaluation_orders AS orders
                JOIN public.model_route_revisions AS route
                  ON route.id = orders.route_revision_id
                 AND route.id = NEW.route_revision_id
                 AND route.descriptor_sha256 = NEW.endpoint_descriptor_sha256
                 AND route.status = 'approved'
                 AND route.approved_season_id = NEW.season_id
                JOIN public.model_submissions AS submission
                  ON submission.id = orders.model_submission_id
                 AND submission.id = route.model_submission_id
                 AND submission.status = 'approved'
                 AND submission.catalog_model_id = NEW.submitted_endpoint_model_id
                 AND submission.model_card_sha256 = NEW.submitted_model_card_sha256
                JOIN public.seasons AS season
                  ON season.id = orders.season_id
                 AND season.id = NEW.season_id
                 AND season.status = 'active'
                 AND season.official IS TRUE
                 AND season.frozen_at IS NOT NULL
                 AND season.manifest_sha256 = route.approved_season_manifest_sha256
                WHERE orders.id = NEW.evaluation_order_id
                  AND orders.organization_id = NEW.organization_id
                  AND orders.status = 'approved'
                  AND orders.billing_status = 'authorized'
                  AND orders.quote_reference_sha256 IS NOT NULL
                  AND orders.rater_plan_sha256 = NEW.rater_plan_sha256
                  AND orders.analysis_plan_sha256 = NEW.analysis_plan_sha256
                  AND orders.budget_cap_micros = NEW.budget_cap_micros
                  AND route.data_policy_sha256 = NEW.data_policy_sha256;
            ELSIF TG_TABLE_NAME = 'evidence_bundles' THEN
                SELECT organization_id INTO expected_organization_id
                FROM public.evaluation_orders WHERE id = NEW.evaluation_order_id;
            ELSIF TG_TABLE_NAME = 'api_idempotency_keys' THEN
                SELECT organization_id INTO expected_organization_id
                FROM public.organization_api_keys WHERE id = NEW.api_key_id;
            ELSE
                RAISE EXCEPTION 'tenant guard attached to unsupported table %', TG_TABLE_NAME;
            END IF;
            IF expected_organization_id IS NULL OR
               expected_organization_id <> NEW.organization_id THEN
                RAISE EXCEPTION 'cross-tenant commercial reference is forbidden';
            END IF;
            RETURN NEW;
        END;
        $$;
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION public.{ARM_BINDING_GUARD_FUNCTION}()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            expected_route_revision_id varchar(36);
            expected_descriptor_sha256 varchar(64);
        BEGIN
            IF TG_OP = 'UPDATE' THEN
                IF OLD.route_revision_id IS DISTINCT FROM NEW.route_revision_id OR
                   OLD.endpoint_descriptor_sha256 IS DISTINCT FROM
                       NEW.endpoint_descriptor_sha256 THEN
                    RAISE EXCEPTION 'response-arm commercial binding is immutable';
                END IF;
                RETURN NEW;
            END IF;
            SELECT run.route_revision_id, run.endpoint_descriptor_sha256
              INTO expected_route_revision_id, expected_descriptor_sha256
            FROM public.battles AS battle
            JOIN public.controlled_runs AS run ON run.id = battle.controlled_run_id
            JOIN public.evaluation_orders AS orders ON orders.id = run.evaluation_order_id
            JOIN public.model_route_revisions AS route
              ON route.id = run.route_revision_id
             AND route.id = orders.route_revision_id
             AND route.descriptor_sha256 = run.endpoint_descriptor_sha256
             AND route.status = 'approved'
             AND route.approved_season_id = run.season_id
            JOIN public.model_submissions AS submission
              ON submission.id = orders.model_submission_id
             AND submission.id = route.model_submission_id
             AND submission.status = 'approved'
             AND submission.catalog_model_id = run.submitted_endpoint_model_id
            JOIN public.season_models AS slot
              ON slot.season_id = run.season_id
             AND slot.model_id = run.submitted_endpoint_model_id
             AND slot.eligible IS TRUE
             AND slot.endpoint_contract_sha256 = route.approved_endpoint_contract_sha256
            WHERE battle.id = NEW.battle_id
              AND orders.status IN ('ready', 'running')
              AND NEW.model_id = run.submitted_endpoint_model_id;
            IF expected_route_revision_id IS NOT NULL THEN
                IF NEW.route_revision_id IS DISTINCT FROM expected_route_revision_id OR
                   NEW.endpoint_descriptor_sha256 IS DISTINCT FROM
                       expected_descriptor_sha256 THEN
                    RAISE EXCEPTION 'submitted commercial arm has wrong route binding';
                END IF;
            ELSIF NEW.route_revision_id IS NOT NULL OR
                  NEW.endpoint_descriptor_sha256 IS NOT NULL THEN
                RAISE EXCEPTION 'non-submitted arm cannot claim commercial route';
            END IF;
            RETURN NEW;
        END;
        $$;

        CREATE TRIGGER trg_response_arms_commercial_binding_insert_guard
        BEFORE INSERT
        ON public.response_arms
        FOR EACH ROW
        EXECUTE FUNCTION public.{ARM_BINDING_GUARD_FUNCTION}();

        CREATE TRIGGER trg_response_arms_commercial_binding_update_guard
        BEFORE UPDATE OF route_revision_id, endpoint_descriptor_sha256
        ON public.response_arms
        FOR EACH ROW
        EXECUTE FUNCTION public.{ARM_BINDING_GUARD_FUNCTION}();
        """
    )

    update_tables = (
        "organizations",
        "organization_api_keys",
        "model_submissions",
        "model_route_revisions",
        "route_contract_tests",
        "evaluation_orders",
        "evidence_bundles",
    )
    append_only_tables = (
        *update_tables,
        "governance_acceptances",
        "api_idempotency_keys",
    )
    tenant_tables = (
        "governance_acceptances",
        "evaluation_orders",
        "controlled_runs",
        "evidence_bundles",
        "api_idempotency_keys",
    )
    insert_tables = append_only_tables
    strict_append_only_updates = (
        "governance_acceptances",
        "api_idempotency_keys",
    )
    for table in insert_tables:
        op.execute(
            f"CREATE TRIGGER trg_{table}_commercial_insert_guard "
            f"BEFORE INSERT ON public.{table} FOR EACH ROW "
            f"EXECUTE FUNCTION public.{INSERT_GUARD_FUNCTION}()"
        )
    for table in update_tables:
        op.execute(
            f"CREATE TRIGGER trg_{table}_commercial_update_guard "
            f"BEFORE UPDATE ON public.{table} FOR EACH ROW "
            f"EXECUTE FUNCTION public.{UPDATE_GUARD_FUNCTION}()"
        )
    for table in strict_append_only_updates:
        op.execute(
            f"CREATE TRIGGER trg_{table}_commercial_update_guard "
            f"BEFORE UPDATE ON public.{table} FOR EACH ROW "
            f"EXECUTE FUNCTION public.{DELETE_GUARD_FUNCTION}()"
        )
    for table in append_only_tables:
        op.execute(
            f"CREATE TRIGGER trg_{table}_commercial_delete_guard "
            f"BEFORE DELETE ON public.{table} FOR EACH ROW "
            f"EXECUTE FUNCTION public.{DELETE_GUARD_FUNCTION}()"
        )
    for table in tenant_tables:
        op.execute(
            f"CREATE TRIGGER trg_{table}_commercial_tenant_guard "
            f"BEFORE INSERT OR UPDATE ON public.{table} FOR EACH ROW "
            f"EXECUTE FUNCTION public.{TENANT_GUARD_FUNCTION}()"
        )

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'flavourbench_api') THEN
                REVOKE UPDATE ON TABLE
                    public.organizations,
                    public.organization_api_keys,
                    public.governance_acceptances,
                    public.model_submissions,
                    public.model_route_revisions,
                    public.route_contract_tests,
                    public.evaluation_orders,
                    public.evidence_bundles,
                    public.api_idempotency_keys
                FROM flavourbench_api;
                REVOKE INSERT ON TABLE
                    public.route_contract_tests,
                    public.evidence_bundles
                FROM flavourbench_api;
                GRANT UPDATE (status, last_used_at, revoked_at)
                    ON TABLE public.organization_api_keys TO flavourbench_api;
                GRANT UPDATE (
                    status, catalog_model_id, decision_reference_sha256,
                    submitted_at, decided_at, suspended_at
                ) ON TABLE public.model_submissions TO flavourbench_api;
                GRANT UPDATE (
                    status, approved_contract_test_id, approved_season_id,
                    approved_season_manifest_sha256, approved_endpoint_contract_sha256, valid_until,
                    submitted_at, approved_at, suspended_at, retired_at
                ) ON TABLE public.model_route_revisions TO flavourbench_api;
                GRANT UPDATE (
                    status, billing_status, publication_status,
                    quote_reference_sha256, submitted_at, approved_at,
                    started_at, completed_at, delivered_at
                ) ON TABLE public.evaluation_orders TO flavourbench_api;
            END IF;
            IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'flavourbench_worker') THEN
                GRANT INSERT ON TABLE public.evidence_bundles TO flavourbench_worker;
                GRANT UPDATE (
                    status, archive_sha256, storage_object_key, size_bytes,
                    signature_algorithm, signing_key_id, signature_base64,
                    sealed_at, available_until, revoked_at
                ) ON TABLE public.evidence_bundles TO flavourbench_worker;
            END IF;
        END;
        $$;
        """
    )


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        _postgres_upgrade()


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    tables = (
        "organizations",
        "organization_api_keys",
        "governance_acceptances",
        "model_submissions",
        "model_route_revisions",
        "route_contract_tests",
        "evaluation_orders",
        "controlled_runs",
        "evidence_bundles",
        "api_idempotency_keys",
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_response_arms_commercial_binding_insert_guard "
        "ON public.response_arms"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_response_arms_commercial_binding_update_guard "
        "ON public.response_arms"
    )
    tenant_tables = {
        "governance_acceptances",
        "evaluation_orders",
        "controlled_runs",
        "evidence_bundles",
        "api_idempotency_keys",
    }
    update_tables = set(tables) - {"governance_acceptances", "api_idempotency_keys"}
    for table in tables:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_commercial_insert_guard ON public.{table}")
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_commercial_delete_guard ON public.{table}")
        if table in update_tables or table in {"governance_acceptances", "api_idempotency_keys"}:
            op.execute(
                f"DROP TRIGGER IF EXISTS trg_{table}_commercial_update_guard ON public.{table}"
            )
        if table in tenant_tables:
            op.execute(
                f"DROP TRIGGER IF EXISTS trg_{table}_commercial_tenant_guard ON public.{table}"
            )
    op.execute(f"DROP FUNCTION IF EXISTS public.{TENANT_GUARD_FUNCTION}()")
    op.execute(f"DROP FUNCTION IF EXISTS public.{ARM_BINDING_GUARD_FUNCTION}()")
    op.execute(f"DROP FUNCTION IF EXISTS public.{INSERT_GUARD_FUNCTION}()")
    op.execute(f"DROP FUNCTION IF EXISTS public.{DELETE_GUARD_FUNCTION}()")
    op.execute(f"DROP FUNCTION IF EXISTS public.{UPDATE_GUARD_FUNCTION}()")
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'flavourbench_api') THEN
                GRANT SELECT, INSERT, UPDATE ON TABLE
                    public.organizations,
                    public.organization_api_keys,
                    public.governance_acceptances,
                    public.model_submissions,
                    public.model_route_revisions,
                    public.route_contract_tests,
                    public.evaluation_orders,
                    public.evidence_bundles,
                    public.api_idempotency_keys
                TO flavourbench_api;
            END IF;
        END;
        $$;
        """
    )
