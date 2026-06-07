from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from agents.state import FinSightState
from core.prompts import SYSTEM_PROMPTS
from services.llm_service import LLMEngineFactory


async def chat_node(state: FinSightState) -> dict:
    """
    Functional execution node layer that intercepts memory payloads,
    forces persona alignment, and tracks active agent IDs.
    """
    model_instance = LLMEngineFactory.get_model()

    system_instruction = SystemMessage(content=SYSTEM_PROMPTS["finsight_core"])
    full_message_context = [system_instruction] + state["messages"]
    response = await model_instance.ainvoke(full_message_context)

    return {
        "messages": [response],
        "current_agent": "chat_agent",
    }


workflow = StateGraph(FinSightState)
workflow.add_node("chat_agent", chat_node)
workflow.set_entry_point("chat_agent")
workflow.add_edge("chat_agent", END)

compiled_graph = workflow.compile()
