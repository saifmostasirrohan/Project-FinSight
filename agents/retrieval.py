import structlog
from langchain_core.messages import AIMessage, SystemMessage

from agents.state import FinSightState
from services.llm_service import LLMEngineFactory
from services.search import FinancialRetriever

logger = structlog.get_logger()


async def retrieval_node(state: FinSightState) -> dict:
    """
    Retrieve semantic context from Supabase and generate a factual answer using
    only matched document chunks.
    """
    logger.info("entering_retrieval_agent_node_processing")

    last_user_message = state["messages"][-1].content if state["messages"] else ""
    session_id = state.get("session_id", "default_session")

    try:
        matched_chunks = await FinancialRetriever.semantic_search(
            query=str(last_user_message),
            session_id=session_id,
            top_k=5,
        )
    except Exception as err:
        logger.error("retrieval_node_vector_search_fault", error=str(err))
        matched_chunks = []

    if not matched_chunks:
        logger.info("retrieval_node_yielded_zero_context_matches")
        msg_fail = AIMessage(
            content=(
                "I searched your uploaded financial records but could not locate "
                "any relevant transaction entries matching your specific query "
                "parameter thresholds. Please confirm that the file ingestion "
                "sequence completed successfully."
            )
        )
        return {
            "messages": [msg_fail],
            "retrieved_chunks": [],
            "current_agent": "retrieval_agent",
        }

    context_accumulator = ""
    for idx, chunk in enumerate(matched_chunks):
        context_accumulator += f"[Document Section {idx + 1}]: {chunk['content']}\n"

    system_prompt = (
        "You are the specialized FinSight Retrieval Agent.\n"
        "Your task is to answer the user's question using ONLY the provided document context block below.\n\n"
        "CONTEXT BLOCK:\n"
        f"{context_accumulator}\n"
        "INSTRUCTIONS:\n"
        "- Base your answer entirely on the facts explicitly stated in the context block.\n"
        "- Cite relevant file identifiers, transaction numbers, currencies, dates, and amounts.\n"
        "- If the context does not contain enough information to answer definitively, say so clearly."
    )

    messages = [
        SystemMessage(content=system_prompt),
        AIMessage(content=f"User query constraint to resolve: {last_user_message}"),
    ]

    model = LLMEngineFactory.get_model()
    response = await model.ainvoke(messages)

    return {
        "messages": [response],
        "retrieved_chunks": matched_chunks,
        "current_agent": "retrieval_agent",
    }
