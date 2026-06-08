import asyncio
import threading

from langchain_core.tools import tool

from services.search import FinancialRetriever


def _run_async_retrieval(query: str, session_id: str) -> list[dict]:
    coroutine = FinancialRetriever.semantic_search(query=query, session_id=session_id)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result_container: dict[str, list[dict] | BaseException] = {}

    def run_in_thread() -> None:
        try:
            result_container["result"] = asyncio.run(coroutine)
        except BaseException as exc:
            result_container["error"] = exc

    worker = threading.Thread(target=run_in_thread, daemon=True)
    worker.start()
    worker.join()

    if "error" in result_container:
        raise result_container["error"]

    return result_container.get("result", [])


@tool
def search_documents(query: str, session_id: str) -> str:
    """
    Search uploaded financial documents, bank statements, ledger entries, and
    corporate expense sheets for relevant contextual data metrics. Use this tool
    whenever the user asks about explicit transactions, vendors, currency
    amounts, calendar dates, or document auditing.
    """
    results = _run_async_retrieval(query=query, session_id=session_id)

    if not results:
        return (
            "No relevant matching financial ledger context located within "
            "uploaded system files for this parameter."
        )

    formatted_context = "--- START RETRIEVED FINANCIAL DOCUMENT CONTEXT ---\n\n"
    for idx, match in enumerate(results):
        formatted_context += (
            f"Context Match [{idx + 1}] "
            f"(Relevance Score: {match['similarity_score']}):\n"
            f"Data Payload: {match['content']}\n\n"
        )
    formatted_context += "--- END RETRIEVED FINANCIAL DOCUMENT CONTEXT ---"
    return formatted_context
