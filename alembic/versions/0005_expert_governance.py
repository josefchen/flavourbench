"""Add qualification, conflict, consent, and cohort evidence for experts.

Revision ID: 0005_expert_governance
Revises: 0004_frozen_endpoint_contracts
Create Date: 2026-07-16
"""

import sqlalchemy as sa

from alembic import op

revision = "0005_expert_governance"
down_revision = "0004_frozen_endpoint_contracts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("expert_reviewers")}
    json_object_default = (
        sa.text("'{}'::json") if bind.dialect.name == "postgresql" else sa.text("'{}'")
    )
    additions = {
        "qualification_verified": sa.Column(
            "qualification_verified", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        "cohort": sa.Column(
            "cohort", sa.String(length=48), nullable=False, server_default="expert_independent"
        ),
        "profile_json": sa.Column(
            "profile_json", sa.JSON(), nullable=False, server_default=json_object_default
        ),
        "batch_reveal_only": sa.Column(
            "batch_reveal_only", sa.Boolean(), nullable=False, server_default=sa.true()
        ),
    }
    for name, column in additions.items():
        if name not in columns:
            op.add_column("expert_reviewers", column)
    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("expert_reviewers")}
    if "ix_expert_reviewers_qualification_verified" not in indexes:
        op.create_index(
            "ix_expert_reviewers_qualification_verified",
            "expert_reviewers",
            ["qualification_verified"],
            unique=False,
        )
    if "ix_expert_reviewers_cohort" not in indexes:
        op.create_index(
            "ix_expert_reviewers_cohort", "expert_reviewers", ["cohort"], unique=False
        )


def downgrade() -> None:
    bind = op.get_bind()
    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("expert_reviewers")}
    if "ix_expert_reviewers_cohort" in indexes:
        op.drop_index("ix_expert_reviewers_cohort", table_name="expert_reviewers")
    if "ix_expert_reviewers_qualification_verified" in indexes:
        op.drop_index(
            "ix_expert_reviewers_qualification_verified", table_name="expert_reviewers"
        )
    columns = {column["name"] for column in sa.inspect(bind).get_columns("expert_reviewers")}
    with op.batch_alter_table("expert_reviewers") as batch:
        for name in (
            "batch_reveal_only",
            "profile_json",
            "cohort",
            "qualification_verified",
        ):
            if name in columns:
                batch.drop_column(name)
