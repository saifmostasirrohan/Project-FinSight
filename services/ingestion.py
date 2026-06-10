import structlog

from services.database import SupabaseManager
from services.embedder import IngestionPipeline
from services.parser import DocumentParser

logger = structlog.get_logger()


class IngestionCoordinator:
    @classmethod
    async def ingest_file(
        cls,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        session_id: str,
    ) -> dict:
        """Coordinate parsing, chunking, embedding, and database persistence."""
        structlog.contextvars.bind_contextvars(session_id=session_id, file=filename)
        logger.info("initiating_coordinated_file_ingestion_pipeline")

        if content_type == "application/pdf":
            extracted_blocks = DocumentParser.parse_pdf(file_bytes)
            file_format = "PDF"
        else:
            extracted_blocks = DocumentParser.parse_csv(file_bytes)
            file_format = "CSV"

        if not extracted_blocks:
            raise ValueError(
                "Document payload is void of valid parseable structural strings."
            )

        supabase = SupabaseManager.get_client()

        doc_record = {
            "session_id": session_id,
            "filename": filename,
            "file_type": file_format,
            "total_chunks": 0,
        }
        db_doc = supabase.table("documents").insert(doc_record).execute()
        if not db_doc.data:
            raise RuntimeError(
                "Cloud ledger registration failure occurred during insertion phase."
            )
        document_id = db_doc.data[0]["id"]

        chunk_blueprints = IngestionPipeline.chunk_document(
            extracted_blocks,
            document_id,
            session_id,
        )
        pure_strings_list = [chunk["content"] for chunk in chunk_blueprints]
        computed_vectors = await IngestionPipeline.compute_embeddings(pure_strings_list)

        if len(computed_vectors) != len(chunk_blueprints):
            raise RuntimeError(
                "Embedding vector count mismatch occurred during ingestion phase."
            )

        final_insert_payload = []
        for index, blueprint in enumerate(chunk_blueprints):
            final_insert_payload.append(
                {
                    "document_id": blueprint["document_id"],
                    "session_id": blueprint["session_id"],
                    "content": blueprint["content"],
                    "chunk_index": blueprint["chunk_index"],
                    "embedding": computed_vectors[index],
                    "metadata": blueprint["metadata"],
                }
            )

        logger.info(
            "dispatching_bulk_chunk_vector_insert_to_supabase",
            records_count=len(final_insert_payload),
        )
        supabase.table("chunks").upsert(
            final_insert_payload,
            on_conflict="session_id,content_hash",
        ).execute()
        supabase.table("documents").update(
            {"total_chunks": len(final_insert_payload)}
        ).eq("id", document_id).execute()

        logger.info(
            "file_ingestion_pipeline_completed_successfully",
            doc_id=document_id,
            chunks=len(final_insert_payload),
        )
        return {
            "document_id": document_id,
            "total_chunks_ingested": len(final_insert_payload),
            "status": "ingested_and_indexed",
        }
