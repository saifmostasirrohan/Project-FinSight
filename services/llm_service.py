from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from core.config import settings


class LLMEngineFactory:
    @staticmethod
    def get_model() -> BaseChatModel:
        provider = settings.LLM_PROVIDER.lower().strip()

        if provider == "groq":
            if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "mock_key":
                raise ValueError(
                    "Inference failed: GROQ_API_KEY environment configuration is invalid."
                )
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                api_key=settings.GROQ_API_KEY,
            )
        if provider == "ollama":
            return ChatOpenAI(
                base_url=settings.OLLAMA_BASE_URL,
                api_key="ollama_passthrough",
                model="llama3.2",
                temperature=0.1,
            )
        raise ValueError(
            f"Unsupported provider definition type string caught: {provider}"
        )


async def get_response(message: str, system_prompt_key: str = "finsight_core") -> str:
    from core.prompts import SYSTEM_PROMPTS

    system_instruction = SYSTEM_PROMPTS.get(
        system_prompt_key, SYSTEM_PROMPTS["finsight_core"]
    )

    try:
        model_instance = LLMEngineFactory.get_model()
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=message),
        ]
        response = await model_instance.ainvoke(messages)
        return str(response.content)
    except Exception as e:
        raise RuntimeError(f"LLM Infrastructure Layer Fault Caught: {str(e)}") from e
