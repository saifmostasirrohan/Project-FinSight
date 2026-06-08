import structlog

from services.database import SupabaseManager
from services.embedder import IngestionPipeline

logger = structlog.get_logger()


class FinancialRetriever:
    @classmethod
    async def semantic_search(
        cls,
        query: str,
        session_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Convert a query to an embedding and match it against stored session chunks.
        """
        structlog.contextvars.bind_contextvars(session_id=session_id)
        logger.info("initiating_semantic_retrieval_query", query_length=len(query))

        try:
            query_vector = await IngestionPipeline.compute_embeddings([query])
            target_embedding = query_vector[0]

            supabase = SupabaseManager.get_client()
            rpc_params = {
                "query_embedding": target_embedding,
                "match_session_id": session_id,
                "match_count": top_k,
            }

            db_res = supabase.rpc("match_chunks", rpc_params).execute()

            refined_results = []
            for record in db_res.data:
                similarity = float(record.get("similarity", 0.0))

                if similarity < 0.40:
                    logger.debug(
                        "discarding_low_relevance_chunk",
                        chunk_id=record.get("id"),
                        score=similarity,
                    )
                    continue

                refined_results.append(
                    {
                        "chunk_id": record.get("id"),
                        "content": record.get("content"),
                        "metadata": record.get("metadata"),
                        "similarity_score": round(similarity, 4),
                    }
                )

            logger.info(
                "semantic_retrieval_complete",
                matches_returned=len(refined_results),
            )
            return refined_results
        except Exception as exc:
            logger.error("semantic_search_subsystem_fault", error=str(exc))
            raise RuntimeError(f"Retrieval Engine Error: {str(exc)}") from exc
