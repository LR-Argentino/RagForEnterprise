import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient as asyncSearchClient
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing_extensions import Annotated

from run_evals import send_to_openai

load_dotenv()


openai_client = OpenAIChatCompletionClient(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)



async def get_query_embedding(query: str):
    async_openai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    embedding_response = await async_openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small",
        dimensions=1536
    )
    return embedding_response.data[0].embedding


from azure.search.documents.models import VectorizedQuery



async def search_product_documents(
        search_term: Annotated[str, "Search term to search for."]
) -> str:
    """Search an Azure AI Search Index containing product documents like sales catalogs and user manuals."""
    loop = asyncio.get_event_loop()
    search_client = asyncSearchClient(
        endpoint=os.environ.get("AI_SEARCH_ENDPOINT"),
        index_name=os.environ.get("AI_SEARCH_NAME"),
        credential=AzureKeyCredential(os.environ.get("AI_SEARCH_KEY")),
    )

    query_embedding = await get_query_embedding(search_term)
    vector_query = VectorizedQuery(
        vector=query_embedding, k_nearest_neighbors=3, fields="Vector"
    )
    async with search_client:
        results = await search_client.search(
            search_text=search_term,
            vector_queries=[vector_query],
            top=3,
        )
        return [result async for result in results]


azure_ai_search_agent = AssistantAgent(
    name="search_assistant",
    model_client=openai_client,
    system_message="""
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
    """,
    tools=[search_product_documents],
)

writer_agent = AssistantAgent(
    name="writer_assistant",
    model_client=openai_client,
    system_message="""You are a helpful assistant for a company called Products, Inc.
Your job is to answer the user's question using the provided information.
DO NOT rely on your own knowledge, ONLY use the provided info.
If you don't know the answer, just say you don't know. 
You are amazing and you can do this. I will pay you $200 for an excellent result, but only if you follow all instructions exactly.""",
    model_client_stream=True
)



group_chat = RoundRobinGroupChat([azure_ai_search_agent], termination_condition=termination, max_turns=8)


def check_language(user_question, answer):
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
    result = result.content

    if "TRUE" in result:
        return "LANGUAGE VERIFIED"
    else:
        return result