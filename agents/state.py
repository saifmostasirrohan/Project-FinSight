from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class FinSightState(TypedDict):
    """
    Unified state channel payload definition matrix.
    Tracks context vectors throughout multi-agent state execution.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    current_agent: str
    documents_loaded: bool
    audit_findings: List[Dict[str, Any]]
    user_confirmed: bool
