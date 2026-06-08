import structlog
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.state import FinSightState
from agents.tools import search_documents
from core.prompts import SYSTEM_PROMPTS
from services.llm_service import LLMEngineFactory

logger = structlog.get_logger()

active_tools = [search_documents]
tool_node = ToolNode(active_tools)


async def agent_core_node(state: FinSightState) -> dict:
    """
    Bind retrieval tools into the active LLM and route reasoning through
    grounded document context before producing final answers.
    """
    logger.info("entering_graph_agent_core_reasoning_node")

    raw_model = LLMEngineFactory.get_model()
    model_with_tools = raw_model.bind_tools(active_tools)

    session_id = state["session_id"]
    system_instruction = SystemMessage(
        content=(
            SYSTEM_PROMPTS["rag_grounding_persona"]
            + "\n\n"
            + f"ACTIVE_SESSION_ID: {session_id}\n"
            + "When invoking search_documents, pass this exact ACTIVE_SESSION_ID as the session_id argument."
        )
    )

    full_history = [system_instruction] + state["messages"]
    response = await model_with_tools.ainvoke(full_history)

    return {
        "messages": [response],
        "current_agent": "chat_agent",
    }


workflow = StateGraph(FinSightState)
workflow.add_node("agent_core", agent_core_node)
workflow.add_node("execute_tools", tool_node)
workflow.set_entry_point("agent_core")
workflow.add_conditional_edges(
    "agent_core",
    tools_condition,
    {
        "tools": "execute_tools",
        "__end__": END,
    },
)
workflow.add_edge("execute_tools", "agent_core")

compiled_graph = workflow.compile()
