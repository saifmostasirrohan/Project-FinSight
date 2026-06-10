import structlog
from langchain_core.messages import AIMessage

from agents.state import FinSightState

logger = structlog.get_logger()


async def action_node(state: FinSightState) -> dict:
    """Compile current graph findings into a staged audit trail report."""
    logger.info("entering_action_agent_node_processing")

    findings = state.get("audit_findings", [])
    violations = state.get("compliance_violations", [])
    requires_review = bool(findings or violations)

    report_summary = (
        "### OFFICIAL FINSIGHT AUDIT TRAIL REPORT\n\n"
        "The multi-agent pipeline has staged the currently available findings:\n\n"
        f"- **Pattern Finding Groups:** {len(findings)}\n"
        f"- **Spending Policy Violations:** {len(violations)}\n"
        f"- **Supervisor Review Required:** {'Yes' if requires_review else 'No'}\n\n"
        "This report is staged in the active workflow response. No database audit "
        "record has been written by this action."
    )

    return {
        "messages": [AIMessage(content=report_summary)],
        "current_agent": "action_agent",
    }
