from typing import Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage


class FinSightState(TypedDict):
    """
    Unified state channel payload definition matrix.
    Tracks context vectors throughout multi-agent state execution.
    """

    messages: List[BaseMessage]
    session_id: str
    current_agent: str
    documents_loaded: bool
    audit_findings: List[Dict[str, Any]]
    user_confirmed: bool
