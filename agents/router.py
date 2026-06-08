import structlog
from langchain_core.messages import SystemMessage

from agents.state import FinSightState
from services.llm_service import LLMEngineFactory

logger = structlog.get_logger()


def router_node(state: FinSightState) -> dict:
    """
    Evaluate the latest user message and record a deterministic intent key
    to drive downstream graph routing.
    """
    logger.info("executing_router_agent_intent_classification")

    last_message = state["messages"][-1].content if state["messages"] else ""
    model = LLMEngineFactory.get_model()

    system_prompt = (
        "You are the master financial routing coordinator for FinSight.\n"
        "Your absolute job is to classify the user's message into exactly one category token word.\n\n"
        "CATEGORIES:\n"
        "- SEARCH: Request to look up specific files, values, transactions, vendors, dates, or basic context entries.\n"
        "- ANALYZE: Request for pattern detection, analytics, audit trends, statistical comparisons, or duplicate checks.\n"
        "- COMPLY: Request to scan transactions against company spending policies or regulatory guidelines.\n"
        "- ACTION: Request to flag items, approve findings, write official audit trail records, or generate final summaries.\n"
        "- CHAT: General greetings, casual queries, generic questions, or system capability explanations.\n\n"
        "CRITICAL: Output ONLY the uppercase word token (SEARCH, ANALYZE, COMPLY, ACTION, or CHAT). "
        "Do not include periods, spaces, preambles, or explanations."
    )

    messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=f"User Message to Classify: '{last_message}'"),
    ]

    response = model.invoke(messages)
    classified_intent = str(response.content).strip().upper()

    valid_intents = {"SEARCH", "ANALYZE", "COMPLY", "ACTION", "CHAT"}
    if classified_intent not in valid_intents:
        logger.warning(
            "router_returned_invalid_format_falling_back_to_chat",
            token=classified_intent,
        )
        classified_intent = "CHAT"

    logger.info(
        "router_intent_classification_resolved",
        resolved_intent=classified_intent,
    )
    return {"intent": classified_intent}
