"""Add immutable, snapshot-bound research release archive metadata.

Revision ID: 0027_research_release_archives
Revises: 0026_retention_basis
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0027_research_release_archives"
down_revision = "0026_retention_basis"
branch_labels = None
depends_on = None

FUNCTION_NAME = "flavourbench_research_release_archive_append_only"
TRIGGER_NAME = "trg_research_release_archives_append_only"
SQLITE_UPDATE_TRIGGER = "trg_research_release_archives_no_update"
SQLITE_DELETE_TRIGGER = "trg_research_release_archives_no_delete"


def upgrade() -> None:
    op.create_table(
        "research_release_archives",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("season_id", sa.String(length=36), sa.ForeignKey("seasons.id"), nullable=False),
        sa.Column("archive_class", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.String(length=96), nullable=False),
        sa.Column("snapshot_ids_json", sa.JSON(), nullable=False),
        sa.Column("snapshot_set_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_json", sa.JSON(), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("archive_sha256", sa.String(length=64), nullable=False),
        sa.Column("storage_object_key", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.Column("source_date_epoch", sa.BigInteger(), nullable=False),
        sa.Column("requirements_lock_sha256", sa.String(length=64), nullable=False),
        sa.Column("build_image_digest", sa.String(length=71), nullable=False),
        sa.Column("signature_algorithm", sa.String(length=32), nullable=False),
        sa.Column("signing_key_id", sa.String(length=160), nullable=False),
        sa.Column("public_key_pem", sa.Text(), nullable=False),
        sa.Column("public_key_sha256", sa.String(length=64), nullable=False),
        sa.Column("signature_base64", sa.Text(), nullable=False),
        sa.Column("privacy_review_artifact_sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "supersedes_archive_id",
            sa.String(length=36),
            sa.ForeignKey("research_release_archives.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "archive_class IN ('internal_official', 'sanitized_public')",
            name="ck_research_release_archives_class",
        ),
        sa.CheckConstraint(
            "signature_algorithm = 'Ed25519'",
            name="ck_research_release_archives_signature_algorithm",
        ),
        sa.CheckConstraint(
            "size_bytes > 0 AND member_count > 0 AND source_date_epoch = 0",
            name="ck_research_release_archives_nonempty",
        ),
        sa.CheckConstraint(
            "length(snapshot_set_sha256) = 64 AND length(manifest_sha256) = 64 "
            "AND length(archive_sha256) = 64 AND length(requirements_lock_sha256) = 64 "
            "AND length(public_key_sha256) = 64 AND length(build_image_digest) = 71",
            name="ck_research_release_archives_digest_lengths",
        ),
        sa.CheckConstraint(
            "archive_class <> 'sanitized_public' OR privacy_review_artifact_sha256 IS NOT NULL",
            name="ck_research_release_archives_public_privacy_review",
        ),
        sa.UniqueConstraint("snapshot_set_sha256"),
        sa.UniqueConstraint("manifest_sha256"),
        sa.UniqueConstraint("archive_sha256"),
        sa.UniqueConstraint("supersedes_archive_id"),
    )
    for column in (
        "season_id",
        "archive_class",
        "snapshot_set_sha256",
        "manifest_sha256",
        "archive_sha256",
    ):
        op.create_index(
            f"ix_research_release_archives_{column}",
            "research_release_archives",
            [column],
        )
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        op.execute(
            f"""
            CREATE OR REPLACE FUNCTION public.{FUNCTION_NAME}()
            RETURNS trigger
            LANGUAGE plpgsql
            SECURITY INVOKER
            SET search_path = pg_catalog, public
            AS $$
            BEGIN
                RAISE EXCEPTION 'research release archives are append-only';
            END;
            $$;
            CREATE TRIGGER {TRIGGER_NAME}
            BEFORE UPDATE OR DELETE ON public.research_release_archives
            FOR EACH ROW EXECUTE FUNCTION public.{FUNCTION_NAME}();
            """
        )
    elif dialect == "sqlite":
        op.execute(
            f"""
            CREATE TRIGGER {SQLITE_UPDATE_TRIGGER}
            BEFORE UPDATE ON research_release_archives
            BEGIN
                SELECT RAISE(ABORT, 'research release archives are append-only');
            END
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER {SQLITE_DELETE_TRIGGER}
            BEFORE DELETE ON research_release_archives
            BEGIN
                SELECT RAISE(ABORT, 'research release archives are append-only');
            END
            """
        )
    else:
        raise RuntimeError(f"unsupported database dialect for 0027: {dialect}")


def downgrade() -> None:
    raise RuntimeError("downgrade across research-release archive policy is prohibited")
