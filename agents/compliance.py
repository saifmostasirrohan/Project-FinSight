import asyncio
import re
from decimal import Decimal, InvalidOperation

import structlog
from langchain_core.messages import SystemMessage

from agents.state import FinSightState
from services.analysis_engine import ForensicAnalytics
from services.database import SupabaseManager
from services.llm_service import LLMEngineFactory

logger = structlog.get_logger()


def _load_policies(session_id: str) -> list[dict]:
    supabase = SupabaseManager.get_client()
    result = (
        supabase.table("company_policies")
        .select("*")
        .eq("session_id", session_id)
        .execute()
    )
    return result.data or []


def _parse_amount(value: object) -> Decimal | None:
    normalized = re.sub(r"[^\d.-]", "", str(value))
    if not normalized:
        return None

    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


async def compliance_node(state: FinSightState) -> dict:
    """Evaluate transaction entries against active company spending rules."""
    logger.info("entering_compliance_agent_node_processing")
    session_id = state.get("session_id", "default_session")

    transactions, policies = await asyncio.gather(
        asyncio.to_thread(ForensicAnalytics.get_raw_transactions, session_id),
        asyncio.to_thread(_load_policies, session_id),
    )

    violations = []
    for transaction in transactions:
        amount = _parse_amount(transaction.get("amount", "0"))
        if amount is None:
            logger.warning(
                "compliance_transaction_amount_invalid",
                txn_id=transaction.get("txn_id", "UNKNOWN"),
            )
            continue

        for policy in policies:
            maximum = _parse_amount(policy.get("max_allowable_amount"))
            if maximum is None or maximum <= 0 or amount <= maximum:
                continue

            violations.append(
                {
                    "txn_id": transaction.get("txn_id", "UNKNOWN"),
                    "date": transaction.get("date"),
                    "vendor": transaction.get("vendor", ""),
                    "amount": transaction.get("amount"),
                    "policy_violated": policy.get("rule_text"),
                    "requires_approval": bool(policy.get("requires_approval")),
                    "severity": "HIGH" if amount > maximum * 2 else "MEDIUM",
                }
            )

    logger.info(
        "compliance_policy_evaluation_complete",
        policies_checked=len(policies),
        transactions_checked=len(transactions),
        violations_found=len(violations),
    )

    system_prompt = (
        "You are the specialized FinSight Compliance Auditing Agent.\n"
        "Review the structured rule violations below and write a formal report "
        "for management review.\n\n"
        f"VIOLATIONS METRICS:\n{violations}\n\n"
        "INSTRUCTIONS:\n"
        "- Explain every rule breach and name the transaction ID, date, vendor, "
        "amount, severity, and approval requirement.\n"
        "- Provide a professional recommendation for each flagged variance.\n"
        "- Do not invent violations absent from the metrics.\n"
        "- If the list is empty, state exactly: "
        "'NO SPENDING POLICY BREACHES DETECTED WITHIN METRIC THRESHOLDS'."
    )

    model = LLMEngineFactory.get_model()
    response = await model.ainvoke([SystemMessage(content=system_prompt)])

    return {
        "messages": [response],
        "compliance_violations": violations,
        "current_agent": "compliance_agent",
    }
