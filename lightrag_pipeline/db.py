import logging
import os

import psycopg2
import psycopg2.extras
from flask import g

logger = logging.getLogger(__name__)

DB_DSN = (
    f"host={os.environ['POSTGRES_HOST']} "
    f"port={os.environ['POSTGRES_PORT']} "
    f"dbname={os.environ['POSTGRES_DATABASE']} "
    f"user={os.environ['POSTGRES_USER']} "
    f"password={os.environ['POSTGRES_PASSWORD']}"
)


def get_conn():
    return psycopg2.connect(DB_DSN, cursor_factory=psycopg2.extras.RealDictCursor)


def init_db():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id          UUID PRIMARY KEY,
                user_email  TEXT NOT NULL,
                title       TEXT NOT NULL DEFAULT 'Нова розмова',
                mode        TEXT NOT NULL DEFAULT 'hybrid',
                messages    JSONB NOT NULL DEFAULT '[]',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
    logger.info("chat_sessions table ready")


def fetch_session(session_id: str):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM chat_sessions WHERE id = %s AND user_email = %s",
            (session_id, g.user_email),
        )
        return cur.fetchone()


def row_to_session(row) -> dict:
    return {
        "id": str(row["id"]),
        "title": row["title"],
        "mode": row["mode"],
        "messages": row["messages"],
        "createdAt": row["created_at"].isoformat(),
        "updatedAt": row["updated_at"].isoformat(),
    }
