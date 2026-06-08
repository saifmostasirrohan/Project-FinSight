from typing import Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage


class FinSightState(TypedDict):
    """
    Unified state channel tracking layout matrix for Project FinSight.
    Manages multi-agent execution context variables between node handoffs.
    """

    messages: List[BaseMessage]
    session_id: str
    current_agent: str
    documents_loaded: bool
    retrieved_chunks: List[Dict[str, Any]]
    audit_findings: List[Dict[str, Any]]
    compliance_violations: List[Dict[str, Any]]
    user_confirmed: bool
    intent: str
