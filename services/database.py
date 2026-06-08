import structlog
from supabase import Client, create_client

from core.config import settings

logger = structlog.get_logger()


class SupabaseManager:
    """Manages singleton initialization for cloud database access."""

    _client: Client | None = None

    @classmethod
    def get_client(cls) -> Client:
        if cls._client is None:
            if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
                logger.error(
                    "supabase_configuration_missing",
                    url=bool(settings.SUPABASE_URL),
                    key=bool(settings.SUPABASE_KEY),
                )
                raise ValueError(
                    "Database initialization stalled: Missing Supabase credentials in .env."
                )

            cls._client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            logger.info(
                "supabase_client_instantiated_successfully",
                base_url=settings.SUPABASE_URL,
            )
        return cls._client


async def verify_db_lifecycle() -> bool:
    """Run an isolated write/delete smoke test against the documents table."""
    try:
        supabase = SupabaseManager.get_client()
        test_payload = {
            "session_id": "smoke-test-uuid",
            "filename": "lifecycle_probe.csv",
            "file_type": "text/csv",
            "total_chunks": 0,
        }

        insert_res = supabase.table("documents").insert(test_payload).execute()
        if not insert_res.data:
            raise RuntimeError(
                "Database write probe execution dropped without recorded response row data."
            )

        record_id = insert_res.data[0]["id"]
        supabase.table("documents").delete().eq("id", record_id).execute()
        return True
    except Exception as exc:
        logger.error("database_lifecycle_probe_failed", exception=str(exc))
        return False
