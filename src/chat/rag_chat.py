"""RAG Chat module with Azure AI Search integration and AutoGen agents."""
import asyncio
from typing import Any, AsyncGenerator, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import ModelClientStreamingChunkEvent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AsyncOpenAI, OpenAIError
from typing_extensions import Annotated

from src.config.configuration import get_config
from src.evaluation.run_evals import send_to_openai
from ..models import RagLog
from ..services import RagLogService
# Constants for chat behavior (not externalized to config)
SEARCH_TOP_K = 3
VECTOR_K_NEAREST_NEIGHBORS = 3
VECTOR_FIELD_NAME = "Vector"
MAX_GROUP_CHAT_TURNS = 8
MAX_TERMINATION_MESSAGES = 10
CHAT_TEMPERATURE = 0


class AzureSearchError(Exception):
    """Custom exception for Azure Search operations."""
    pass


class EmbeddingError(Exception):
    """Custom exception for embedding generation failures."""
    pass


def _create_openai_client() -> OpenAIChatCompletionClient:
    """Create OpenAI chat completion client."""
    config = get_config()
    return OpenAIChatCompletionClient(
        model=config.openai.model,
        temperature=CHAT_TEMPERATURE,
        api_key=config.openai.api_key,
    )


def _create_async_openai_client() -> AsyncOpenAI:
    """Create async OpenAI client for embeddings."""
    config = get_config()
    return AsyncOpenAI(api_key=config.openai.api_key)


def _create_azure_search_client() -> AsyncSearchClient:
    """Create Azure AI Search async client for product index."""
    config = get_config()
    return AsyncSearchClient(
        endpoint=config.azure_ai_search.endpoint,
        index_name=config.azure_ai_search.index_name,
        credential=AzureKeyCredential(config.azure_ai_search.api_key),
    )


def _create_pdf_search_client() -> AsyncSearchClient:
    """Create Azure AI Search async client for PDF document index."""
    config = get_config()
    return AsyncSearchClient(
        endpoint=config.azure_ai_search.endpoint,
        index_name=config.azure_ai_search.pdf_index_name,
        credential=AzureKeyCredential(config.azure_ai_search.api_key),
    )


# Module-level clients (lazy initialization)
_async_openai_client: AsyncOpenAI | None = None


def _get_async_openai_client() -> AsyncOpenAI:
    """Get or create singleton async OpenAI client."""
    global _async_openai_client
    if _async_openai_client is None:
        _async_openai_client = _create_async_openai_client()
    return _async_openai_client


async def get_query_embedding(query: str) -> List[float]:
    """
    Generate embedding vector for a query string.

    Args:
        query: The text to embed.

    Returns:
        List of floats representing the embedding vector.

    Raises:
        EmbeddingError: If embedding generation fails.
    """
    try:
        config = get_config()
        client = _get_async_openai_client()
        embedding_response = await client.embeddings.create(
            input=[query],
            model=config.openai.embedding_model,
            dimensions=config.openai.embedding_dimensions,
        )
        return embedding_response.data[0].embedding
    except OpenAIError as e:
        raise EmbeddingError(f"Failed to generate embedding: {e}") from e


async def search_product_documents(
        search_term: Annotated[str, "Search term to search for."]
) -> List[Dict[str, Any]]:
    """
    Search Azure AI Search Index containing product documents.

    Searches for product documents like sales catalogs and user manuals
    using both text and vector search.

    Args:
        search_term: The search query string.

    Returns:
        List of search result documents.

    Raises:
        AzureSearchError: If the search operation fails.
    """
    try:
        query_embedding = await get_query_embedding(search_term)
    except EmbeddingError as e:
        raise AzureSearchError(f"Failed to prepare search: {e}") from e

    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=VECTOR_K_NEAREST_NEIGHBORS,
        fields=VECTOR_FIELD_NAME,
    )

    search_client = _create_azure_search_client()

    try:
        async with search_client:
            results = await search_client.search(
                search_text=search_term,
                vector_queries=[vector_query],
                top=SEARCH_TOP_K,
            )
            return [result async for result in results]
    except AzureError as e:
        raise AzureSearchError(f"Azure Search operation failed: {e}") from e


async def search_pdf_documents(
        search_term: Annotated[str, "Search term to search for in PDF documents."]
) -> List[Dict[str, Any]]:
    """
    Search Azure AI Search Index containing PDF documents.

    Searches for PDF documents like receipts, invoices, tickets, and bills
    using both text and vector search.

    Args:
        search_term: The search query string.

    Returns:
        List of search result documents with fields:
        - document_name: Name of the PDF file
        - page_number: Page number within the document
        - page_content: Extracted text content
        - source_dataset: Origin dataset identifier

    Raises:
        AzureSearchError: If the search operation fails.
    """
    try:
        query_embedding = await get_query_embedding(search_term)
    except EmbeddingError as e:
        raise AzureSearchError(f"Failed to prepare search: {e}") from e

    # PDF index uses lowercase "vector" field name
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=VECTOR_K_NEAREST_NEIGHBORS,
        fields="vector",
    )

    search_client = _create_pdf_search_client()

    try:
        async with search_client:
            results = await search_client.search(
                search_text=search_term,
                vector_queries=[vector_query],
                top=SEARCH_TOP_K,
            )
            return [result async for result in results]
    except AzureError as e:
        raise AzureSearchError(f"Azure Search operation failed: {e}") from e


SEARCH_ASSISTANT_SYSTEM_MESSAGE = """
You are a helpful assistant for a company called Products, Inc.
You have access to an Azure AI Search Index containing product records, and you may search them.
The correct syntax for a search is: "what you want to search for".
Please use the search function to find enough information to answer the user's question.
DO NOT rely on your own knowledge, ONLY use the information retrieved from the search.
If you only find one document, that is ok. If the document contains very little information, that is ok.
Please be honest about what you find. I am not looking for perfection, just the truth.
You are amazing and you can do this.
I will pay you $200 for an excellent result, but only if you follow all instructions exactly.


When enough information has been retrieved to answer the user's question to full satisfaction,
please return "TERMINATE" to end the conversation. If more information must be collected, please return CONTINUE.
"""

DOCUMENT_SEARCH_ASSISTANT_SYSTEM_MESSAGE = """
You are a helpful assistant that searches through personal documents.
You have access to an Azure AI Search Index containing PDF documents like receipts, invoices, tickets, and bills.
The correct syntax for a search is: "what you want to search for".
Please use the search function to find the relevant documents for the user's question.
DO NOT rely on your own knowledge, ONLY use the information retrieved from the search.
If you find a receipt or ticket, extract the relevant information (amounts, dates, items).
Please be honest about what you find. I am not looking for perfection, just the truth.
You are amazing and you can do this.
I will pay you $200 for an excellent result, but only if you follow all instructions exactly.

When enough information has been retrieved to answer the user's question to full satisfaction,
please return "TERMINATE" to end the conversation. If more information must be collected, please return CONTINUE.
"""

WRITER_ASSISTANT_SYSTEM_MESSAGE = """You are a helpful assistant for a company called Products, Inc.
Your job is to answer the user's question using the provided information.
DO NOT rely on your own knowledge, ONLY use the provided info.
If you don't know the answer, just say you don't know.
You are amazing and you can do this. I will pay you $200 for an excellent result, but only if you follow all instructions exactly."""


def create_search_agent(client: OpenAIChatCompletionClient) -> AssistantAgent:
    """Create the product search assistant agent."""
    return AssistantAgent(
        name="search_assistant",
        model_client=client,
        system_message=SEARCH_ASSISTANT_SYSTEM_MESSAGE,
        tools=[search_product_documents],
    )


def create_document_search_agent(client: OpenAIChatCompletionClient) -> AssistantAgent:
    """Create the document search assistant agent for PDFs."""
    return AssistantAgent(
        name="document_search_assistant",
        model_client=client,
        system_message=DOCUMENT_SEARCH_ASSISTANT_SYSTEM_MESSAGE,
        tools=[search_pdf_documents],
    )


def create_writer_agent(client: OpenAIChatCompletionClient) -> AssistantAgent:
    """Create the writer assistant agent."""
    return AssistantAgent(
        name="writer_assistant",
        model_client=client,
        system_message=WRITER_ASSISTANT_SYSTEM_MESSAGE,
        model_client_stream=True,
    )


def create_group_chat(
        agents: List[AssistantAgent], max_turns: int = MAX_GROUP_CHAT_TURNS
) -> RoundRobinGroupChat:
    """
    Create a round-robin group chat with the given agents.

    Args:
        agents: List of agents to include in the chat.
        max_turns: Maximum number of turns before termination.

    Returns:
        Configured RoundRobinGroupChat instance.
    """
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(
        MAX_TERMINATION_MESSAGES
    )
    return RoundRobinGroupChat(
        agents, termination_condition=termination, max_turns=max_turns
    )


def check_language(user_question: str, answer: str) -> str:
    """
    Verify answer language matches question language, translating if needed.

    Args:
        user_question: The original question from the user.
        answer: The generated answer to verify/translate.

    Returns:
        "LANGUAGE VERIFIED" if languages match, otherwise the translated answer.
    """
    check_language_prompt = f"""
You are an expert at languages and translation.
Please make sure the answer is written in the same language as the user's question.
If the answer and the question are both written in the same language, return
TRUE.
Otherwise, return the answer translated into the same language as the user's question.

For example, if the user's question is in German but the answer is in English, please
return the answer, translated into German.

User question: {user_question}

Answer: {answer}"""

    result = send_to_openai(check_language_prompt)

    if "TRUE" in result:
        return "LANGUAGE VERIFIED"
    return result


openai_client = _create_openai_client()
azure_ai_search_agent = create_search_agent(openai_client)
document_search_agent = create_document_search_agent(openai_client)
writer_agent = create_writer_agent(openai_client)
rag_log_service = RagLogService()

async def RAGChat_streaming(
    chat_history: str,
    user_question: str,
    user_email: str
) -> AsyncGenerator[str, None]:
    """
    Streaming version of RAGChat - yields response chunks as they arrive.

    Uses triage to determine whether to search products or documents.

    Args:
        chat_history: Previous conversation context
        user_question: The user's question
        user_email: User email for logging

    Yields:
        str: Response chunks as they are generated
    """
    from src.chat.question_triage import triage

    # ---- PHASE 0: Triage - determine which index to search ----
    category, triage_result = triage(user_question, chat_history)

    if category == "CLARIFY":
        yield triage_result
        return

    # ---- PHASE 1: Search (non-streaming, we need complete results) ----
    SEARCH_PROMPT = f"""Please search and find enough information to answer the user's question.
    User Question: {user_question}"""

    # Select the appropriate search agent based on triage
    if category == "DOCUMENT":
        search_agent = document_search_agent
    else:  # PRODUCT
        search_agent = azure_ai_search_agent

    search_result: TaskResult = await search_agent.run(task=SEARCH_PROMPT)

    # Extract retrieved data
    retrieved_data_parts = []
    for msg in search_result.messages:
        if hasattr(msg, 'content'):
            content = msg.content
            if isinstance(content, list):
                for item in content:
                    if hasattr(item, 'content'):
                        retrieved_data_parts.append(str(item.content))
                    else:
                        retrieved_data_parts.append(str(item))
            elif isinstance(content, str) and content:
                retrieved_data_parts.append(f"{getattr(msg, 'source', 'unknown')}: {content}")

    retrieved_data = "\n".join(retrieved_data_parts)

    # ---- PHASE 2: Writer Agent with Streaming ----
    WRITER_PROMPT = f"""Please write the final answer to the user's question: \n{user_question}\n\n
          You may use the chat history to help you write the answer. \n {chat_history}\n\n
        The information retrieved from the search agents is:
        {retrieved_data}. I will tip you $200 for an excellent result."""

    # Collect full answer for logging and language check
    final_answer_parts = []

    # Stream the response
    async for event in writer_agent.run_stream(task=WRITER_PROMPT):
        if isinstance(event, ModelClientStreamingChunkEvent):
            chunk = event.content
            final_answer_parts.append(chunk)
            yield chunk

    # Reconstruct full answer for post-processing
    final_answer = "".join(final_answer_parts)

    # Language check (after streaming complete)
    translated_answer = check_language(user_question, final_answer)
    if translated_answer != "LANGUAGE VERIFIED":
        # If translation was needed, yield a notice and the translated version
        yield "\n\n[Translated to match question language:]\n"
        yield translated_answer
        final_answer = translated_answer

    # ---- LOGGING ----
    rag_log = RagLog(
        user_email=user_email,
        user_question=user_question,
        agents_search_results=str(retrieved_data),
        final_answer=final_answer
    )
    rag_log_service.store_answer(rag_log)


# ============================================================================
# TODO: FastAPI Streaming Endpoint (uncomment when needed)
# ============================================================================
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
#
# app = FastAPI()
#
# class ChatRequest(BaseModel):
#     chat_history: str
#     question: str
#     user_email: str
#
# @app.post("/chat/stream")
# async def chat_stream(request: ChatRequest):
#     """Server-Sent Events endpoint for streaming chat responses."""
#     async def generate():
#         async for chunk in RAGChat_streaming(
#             request.chat_history,
#             request.question,
#             request.user_email
#         ):
#             # SSE format: data: <content>\n\n
#             yield f"data: {chunk}\n\n"
#         yield "data: [DONE]\n\n"
#
#     return StreamingResponse(
#         generate(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#         }
#     )
# ============================================================================
