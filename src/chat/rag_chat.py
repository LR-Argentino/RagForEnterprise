"""RAG Chat module with Azure AI Search integration and AutoGen agents."""
import asyncio
from typing import Any, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
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
    """Create Azure AI Search async client."""
    config = get_config()
    return AsyncSearchClient(
        endpoint=config.azure_ai_search.endpoint,
        index_name=config.azure_ai_search.index_name,
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

WRITER_ASSISTANT_SYSTEM_MESSAGE = """You are a helpful assistant for a company called Products, Inc.
Your job is to answer the user's question using the provided information.
DO NOT rely on your own knowledge, ONLY use the provided info.
If you don't know the answer, just say you don't know.
You are amazing and you can do this. I will pay you $200 for an excellent result, but only if you follow all instructions exactly."""


def create_search_agent(client: OpenAIChatCompletionClient) -> AssistantAgent:
    """Create the search assistant agent."""
    return AssistantAgent(
        name="search_assistant",
        model_client=client,
        system_message=SEARCH_ASSISTANT_SYSTEM_MESSAGE,
        tools=[search_product_documents],
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


# Module-level instances for backward compatibility
openai_client = _create_openai_client()
azure_ai_search_agent = create_search_agent(openai_client)
writer_agent = create_writer_agent(openai_client)
rag_log_service = RagLogService()

# group_chat = create_group_chat([azure_ai_search_agent, writer_agent])


async def RAGChat(chat_history: str, user_question: str, user_email: str):
    # ---- PHASE 1: Produkt Suche ----
    # product_db_groupchat = create_group_chat([azure_ai_search_agent])

    SEARCH_PROMPT = f"""Please search and find enough information to answer the user's question.
    User Question: {user_question}"""

    # When running in script use asyncio
    search_result: TaskResult = await azure_ai_search_agent.run(task=SEARCH_PROMPT)

    # DEBUG: Print message structure to understand AutoGen's response format
    print("\n=== DEBUG: search_result.messages ===")
    for i, msg in enumerate(search_result.messages):
        print(f"\n[{i}] Type: {type(msg).__name__}")
        print(f"    Source: {getattr(msg, 'source', 'N/A')}")
        if hasattr(msg, 'content'):
            content = msg.content
            print(f"    Content type: {type(content).__name__}")
            if isinstance(content, list):
                for j, item in enumerate(content):
                    print(f"      [{j}] Item type: {type(item).__name__}")
                    print(f"          Item: {item}")
            else:
                print(f"    Content: {content[:500] if isinstance(content, str) else content}")
    print("=== END DEBUG ===\n")

    # Extract retrieved data - handle both text messages and function execution results
    retrieved_data_parts = []
    for msg in search_result.messages:
        if hasattr(msg, 'content'):
            content = msg.content
            # Handle FunctionExecutionResultMessage (tool results are in a list)
            if isinstance(content, list):
                for item in content:
                    if hasattr(item, 'content'):
                        retrieved_data_parts.append(str(item.content))
                    else:
                        retrieved_data_parts.append(str(item))
            elif isinstance(content, str) and content:
                retrieved_data_parts.append(f"{getattr(msg, 'source', 'unknown')}: {content}")

    retrieved_data = "\n".join(retrieved_data_parts)

    # ---- PHASE 2: Writer Agent ----
    WRITER_PROMPT = f"""Please write the final answer to the user's question: \n{user_question}\n\n
          You may use the chat history to help you write the answer. \n {chat_history}\n\n
        The information retrieved from the search agents is:
        {retrieved_data}. I will tip you $200 for an excellent result."""

    # When running in script use asyncio
    writer_result: TaskResult = await writer_agent.run(task=WRITER_PROMPT)

    # When running outside of script TODO: Change this if this running outside of script
    #writer_result: TaskResult = await writer_agent.run(task=WRITER_PROMPT)

    final_answer = ""
    for msg in reversed(writer_result.messages):
        if hasattr(msg, 'content') and len(str(msg.content)) > 2:
            final_answer = str(msg.content)
            break

    translated_answer = check_language(user_question, final_answer)

    # DEBUG: check_language result
    print(f"\n=== DEBUG check_language ===")
    print(f"User question: {user_question}")
    print(f"Final answer (before check): {final_answer[:200]}...")
    print(f"check_language result: {translated_answer[:200] if len(translated_answer) > 200 else translated_answer}")
    print(f"=== END DEBUG ===\n")

    if translated_answer != "LANGUAGE VERIFIED":
        final_answer = translated_answer

    # ---- LOGGING ----
    rag_log = RagLog(user_email=user_email,
                     user_question=user_question,
                     agents_search_results=str(retrieved_data),
                     final_answer=final_answer)

    rag_log_service.store_answer(rag_log)
    # store_answer_info(user_email, user_question, str(retrieved_data), final_answer)
    return final_answer
