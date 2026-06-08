import structlog
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from agents.router import router_node
from agents.state import FinSightState

logger = structlog.get_logger()


async def retrieval_placeholder(state: FinSightState) -> dict:
    logger.info("placeholder_node_hit", node="retrieval")
    return {"current_agent": "retrieval_agent"}


async def analysis_placeholder(state: FinSightState) -> dict:
    logger.info("placeholder_node_hit", node="analysis")
    return {"current_agent": "analysis_agent"}


async def compliance_placeholder(state: FinSightState) -> dict:
    logger.info("placeholder_node_hit", node="compliance")
    return {"current_agent": "compliance_agent"}


async def action_placeholder(state: FinSightState) -> dict:
    logger.info("placeholder_node_hit", node="action")
    return {"current_agent": "action_agent"}


async def chat_placeholder(state: FinSightState) -> dict:
    logger.info("placeholder_node_hit", node="chat")
    return {"current_agent": "chat_agent"}


async def response_placeholder(state: FinSightState) -> dict:
    """Format route validation state into a user-visible response."""
    logger.info("entering_response_agent_egress_formatting")

    active_intent = state.get("intent", "UNKNOWN")
    active_agent = state.get("current_agent", "unknown")

    msg_out = AIMessage(
        content=(
            "FinSight Routing Validation Successful. "
            f"Path Routed via: Router -> {active_agent} -> Response Node. "
            f"Intent Category confirmed as: '{active_intent}'."
        )
    )
    return {"messages": [msg_out]}


def route_by_intent(state: FinSightState) -> str:
    """Evaluate state intent values to drive conditional routing paths."""
    intent_token = state.get("intent", "CHAT")
    mapping = {
        "SEARCH": "retrieval_node",
        "ANALYZE": "analysis_node",
        "COMPLY": "compliance_node",
        "ACTION": "action_node",
        "CHAT": "chat_node",
    }
    target_node = mapping.get(intent_token, "chat_node")
    logger.info("graph_routing_decision_dispatched", route_target=target_node)
    return target_node


workflow = StateGraph(FinSightState)

workflow.add_node("router_node", router_node)
workflow.add_node("retrieval_node", retrieval_placeholder)
workflow.add_node("analysis_node", analysis_placeholder)
workflow.add_node("compliance_node", compliance_placeholder)
workflow.add_node("action_node", action_placeholder)
workflow.add_node("chat_node", chat_placeholder)
workflow.add_node("response_node", response_placeholder)

workflow.set_entry_point("router_node")

workflow.add_conditional_edges(
    "router_node",
    route_by_intent,
    {
        "retrieval_node": "retrieval_node",
        "analysis_node": "analysis_node",
        "compliance_node": "compliance_node",
        "action_node": "action_node",
        "chat_node": "chat_node",
    },
)

workflow.add_edge("retrieval_node", "response_node")
workflow.add_edge("analysis_node", "response_node")
workflow.add_edge("compliance_node", "response_node")
workflow.add_edge("action_node", "response_node")
workflow.add_edge("chat_node", "response_node")

workflow.add_edge("response_node", END)

compiled_graph = workflow.compile()
