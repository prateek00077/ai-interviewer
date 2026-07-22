"""resumes and embedded resume chunks

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-22

Adds the vector extension, the two tables, and an HNSW index over cosine
distance. The RLS half is generated from app.db.rls, including the new
candidate-writable policy shape that lets a candidate record their own upload.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op
from app.db.rls import disable_rls, drop_policy, enable_rls, policy_for
from app.models.resume import EMBEDDING_DIMENSIONS

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RESUME_STATUS = ("PENDING", "UPLOADED", "PARSING", "READY", "FAILED")
NEW_TABLES = ("resumes", "resume_chunks")


def upgrade() -> None:
    # The `vector` extension is created by scripts/bootstrap_db.sql, not here.
    # Unlike citext and pgcrypto it is not a trusted extension, so only a
    # superuser can install it and Alembic connects as app_owner. Failing with a
    # clear message beats "permission denied to create extension".
    if not op.get_bind().exec_driver_sql(
        "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
    ).scalar():
        raise RuntimeError(
            "The `vector` extension is missing. Run scripts/bootstrap_db.sql as a "
            "superuser before migrating."
        )

    resume_status = postgresql.ENUM(*RESUME_STATUS, name="resume_status", create_type=False)
    resume_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "resumes",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("s3_key", sa.String(512), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(120), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("status", resume_status, server_default=sa.text("'PENDING'"), nullable=False),
        sa.Column(
            "parsed", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_resumes"),
        sa.ForeignKeyConstraint(
            ["org_id"],
            ["organizations.id"],
            name="fk_resumes_org_id_organizations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["candidates.id"],
            name="fk_resumes_candidate_id_candidates",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("s3_key", name="uq_resumes_s3_key"),
        sa.CheckConstraint(
            "size_bytes IS NULL OR size_bytes > 0", name="ck_resumes_size_bytes_positive"
        ),
    )
    op.create_index("ix_resumes_org_id", "resumes", ["org_id"])
    op.create_index("ix_resumes_candidate_id", "resumes", ["candidate_id"])
    op.create_index(
        "ix_resumes_candidate_id_created_at", "resumes", ["candidate_id", "created_at"]
    )

    op.create_table(
        "resume_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("resume_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(64), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_resume_chunks"),
        sa.ForeignKeyConstraint(
            ["org_id"],
            ["organizations.id"],
            name="fk_resume_chunks_org_id_organizations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["resume_id"],
            ["resumes.id"],
            name="fk_resume_chunks_resume_id_resumes",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("resume_id", "ordinal", name="uq_resume_chunks_resume_id_ordinal"),
    )
    op.create_index("ix_resume_chunks_org_id", "resume_chunks", ["org_id"])
    op.create_index("ix_resume_chunks_resume_id", "resume_chunks", ["resume_id"])

    # HNSW, not IVFFlat: IVFFlat needs training data present at build time and
    # this table is empty right now. HNSW builds incrementally and needs no
    # retraining as rows arrive.
    #
    # vector_cosine_ops must match the operator the retriever uses (<=>). An
    # index built for L2 is silently ignored by a cosine query -- it does not
    # error, it just scans.
    op.execute(
        "CREATE INDEX ix_resume_chunks_embedding_hnsw ON resume_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    for table in NEW_TABLES:
        for statement in enable_rls(table):
            op.execute(statement)
        op.execute(drop_policy(table))
        op.execute(policy_for(table))


def downgrade() -> None:
    for table in NEW_TABLES:
        op.execute(drop_policy(table))
        for statement in disable_rls(table):
            op.execute(statement)

    op.drop_table("resume_chunks")
    op.drop_table("resumes")
    postgresql.ENUM(name="resume_status").drop(op.get_bind(), checkfirst=True)
    # The vector extension is left installed. It is owned by the bootstrap step,
    # dropping it would break any other vector column, and app_owner could not
    # drop it anyway.
