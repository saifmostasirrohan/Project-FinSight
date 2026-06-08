import httpx
import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from core.config import settings

logger = structlog.get_logger()


class LocalTransformerSingleton:
    """Load the MiniLM CPU model once, only when cloud embedding is unavailable."""

    _model = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            logger.info("loading_local_cpu_transformer_model_into_ram")
            cls._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            logger.info("local_cpu_transformer_model_loaded_successfully")
        return cls._model


class IngestionPipeline:
    @staticmethod
    def chunk_document(
        parsed_elements: list[dict],
        document_id: str,
        session_id: str,
    ) -> list[dict]:
        """
        Split long-form documents while preserving CSV row sentences as atomic chunks.
        """
        if not parsed_elements:
            logger.info("chunking_sequence_complete", generated_chunks_count=0)
            return []

        logger.info(
            "starting_document_chunking_sequence",
            file_type=parsed_elements[0]["metadata"]["file_type"],
        )
        processed_chunks = []
        chunk_idx = 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )

        for element in parsed_elements:
            file_type = element["metadata"]["file_type"]

            if file_type == "text/csv":
                chunk_idx += 1
                processed_chunks.append(
                    {
                        "document_id": document_id,
                        "session_id": session_id,
                        "content": element["content"],
                        "chunk_index": chunk_idx,
                        "metadata": {
                            **element["metadata"],
                            "source_row": element["source_location"],
                        },
                    }
                )
            else:
                sub_chunks = text_splitter.split_text(element["content"])
                for sub_text in sub_chunks:
                    chunk_idx += 1
                    processed_chunks.append(
                        {
                            "document_id": document_id,
                            "session_id": session_id,
                            "content": sub_text,
                            "chunk_index": chunk_idx,
                            "metadata": {
                                **element["metadata"],
                                "source_page": element["source_location"],
                            },
                        }
                    )

        logger.info(
            "chunking_sequence_complete",
            generated_chunks_count=len(processed_chunks),
        )
        return processed_chunks

    @classmethod
    async def compute_embeddings(cls, text_chunks: list[str]) -> list[list[float]]:
        """Vectorize chunks with Hugging Face Inference API, falling back to local CPU."""
        if not text_chunks:
            return []

        provider = settings.EMBEDDING_PROVIDER.lower().strip()

        if provider == "huggingface" and settings.HF_API_KEY:
            try:
                api_url = (
                    "https://router.huggingface.co/hf-inference/models/"
                    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
                )
                headers = {"Authorization": f"Bearer {settings.HF_API_KEY}"}

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        api_url,
                        headers=headers,
                        json={
                            "inputs": text_chunks,
                            "options": {"wait_for_model": True},
                        },
                    )

                if response.status_code == 200:
                    return response.json()

                logger.warning(
                    "hf_inference_api_non_200_response",
                    status_code=response.status_code,
                )
            except Exception as exc:
                logger.error("hf_inference_api_network_exception", error=str(exc))

        logger.info("routing_vectorization_to_local_cpu_fallback_layer")
        local_model = LocalTransformerSingleton.get_model()
        embeddings = local_model.encode(text_chunks, batch_size=32)
        return embeddings.tolist()
