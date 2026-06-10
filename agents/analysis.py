import asyncio

import structlog
from langchain_core.messages import SystemMessage

from agents.state import FinSightState
from services.analysis_engine import ForensicAnalytics
from services.llm_service import LLMEngineFactory

logger = structlog.get_logger()


async def analysis_node(state: FinSightState) -> dict:
    """Run deterministic forensic checks and summarize the resulting findings."""
    logger.info("entering_analysis_agent_node_processing")
    session_id = state.get("session_id", "default_session")
    user_message = (
        str(state["messages"][-1].content).lower() if state["messages"] else ""
    )

    duplicates = []
    outliers = []

    requests_duplicates = any(
        keyword in user_message for keyword in ("duplicate", "repeat")
    )
    requests_outliers = any(
        keyword in user_message
        for keyword in ("outlier", "suspicious", "anomal")
    )

    if requests_duplicates and not requests_outliers:
        duplicates = await asyncio.to_thread(
            ForensicAnalytics.detect_duplicates,
            session_id,
        )
    elif requests_outliers and not requests_duplicates:
        outliers = await asyncio.to_thread(
            ForensicAnalytics.detect_outliers,
            session_id,
        )
    else:
        duplicate_result, outlier_result = await asyncio.gather(
            asyncio.to_thread(ForensicAnalytics.detect_duplicates, session_id),
            asyncio.to_thread(ForensicAnalytics.detect_outliers, session_id),
        )
        duplicates = duplicate_result
        outliers = outlier_result

    compiled_findings = {
        "duplicate_groups": duplicates,
        "statistical_outliers": outliers,
    }

    system_prompt = (
        "You are the specialized FinSight Forensic Analysis Agent.\n"
        "Review the structured data findings below and explain them to a corporate "
        "compliance supervisor.\n\n"
        "STRUCTURAL FINDINGS METRICS:\n"
        f"{compiled_findings}\n\n"
        "INSTRUCTIONS:\n"
        "- Summarize all findings clearly in plain English using clean Markdown formatting.\n"
        "- Explicitly list risk scores, transaction values, currencies, vendor identities, "
        "transaction IDs, and date markers when present.\n"
        "- Do not invent findings that are absent from the metrics.\n"
        "- If both lists are empty, state exactly: "
        "'NO ANOMALOUS FOOTPRINTS LOCATED WITHIN THE CURRENT SESSION METRICS'."
    )

    model = LLMEngineFactory.get_model()
    response = await model.ainvoke([SystemMessage(content=system_prompt)])

    return {
        "messages": [response],
        "audit_findings": [compiled_findings],
        "current_agent": "analysis_agent",
    }
