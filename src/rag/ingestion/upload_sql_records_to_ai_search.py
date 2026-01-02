"""Upload SQL records to Azure AI Search with embeddings."""

import sqlite3
from contextlib import contextmanager
from datetime import date
from typing import Generator, List, Tuple

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import OpenAI

from src.rag.config.configuration import get_config


def _create_openai_client() -> OpenAI:
    """Create OpenAI client using configuration."""
    config = get_config()
    return OpenAI(api_key=config.openai.api_key)


def _create_search_client() -> SearchClient:
    """Create Azure AI Search client using configuration."""
    config = get_config()
    return SearchClient(
        endpoint=config.azure_ai_search.endpoint,
        index_name=config.azure_ai_search.index_name,
        credential=AzureKeyCredential(config.azure_ai_search.api_key),
    )


@contextmanager
def _get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections with proper cleanup."""
    config = get_config()
    conn = sqlite3.connect(config.database.path)
    try:
        yield conn
    finally:
        conn.close()


def summarize_sql_record(sql_record: Tuple) -> str:
    """
    Create a summary string from a SQL product record.

    Args:
        sql_record: Tuple containing (id, name, description, specs, manufacturer).

    Returns:
        Formatted summary string for embedding.
    """
    record_summary = f"""Product_Name: {sql_record[1]}
Product Description: {sql_record[2]}
Technical Specs: {sql_record[3]}
Manufacturer: {sql_record[4]}"""

    print(f"Summarized record for {sql_record[1]}")
    return record_summary


def clean_record_summary(record_summary: str) -> str:
    """
    Clean the record summary for embedding.

    Args:
        record_summary: Raw summary string.

    Returns:
        Cleaned summary string.
    """
    cleaned = record_summary.replace("--", "")
    return cleaned


def embed_text(openai_client: OpenAI, text: str) -> List[float]:
    """
    Generate embedding vector for text using OpenAI API.

    Args:
        openai_client: OpenAI client instance.
        text: Text to embed.

    Returns:
        List of floats representing the embedding vector.
    """
    config = get_config()
    response = openai_client.embeddings.create(
        input=text,
        model=config.openai.embedding_model,
        dimensions=config.openai.embedding_dimensions,
    )
    return response.data[0].embedding


def upload_to_ai_search(
    search_client: SearchClient,
    record: Tuple,
    record_summary: str,
    embedded_summary: List[float],
) -> None:
    """
    Upload a product record to Azure AI Search.

    Args:
        search_client: Azure Search client instance.
        record: Original SQL record tuple.
        record_summary: Cleaned summary string.
        embedded_summary: Embedding vector.
    """
    date_string = date.today().strftime("%Y-%m-%d")

    document = {
        "id": str(record[0]),
        "Summary": record_summary,
        "Vector": embedded_summary,
        "ProductName": record[1],
        "ProductDescription": record[2],
        "TechnicalSpecifications": record[3],
        "Manufacturer": record[4],
        "DateModified": date_string,
    }

    search_client.merge_or_upload_documents(documents=[document])
    print(f"Record for {record[1]} processed and uploaded!")


def summarize_embed_upload(
    openai_client: OpenAI,
    search_client: SearchClient,
    sql_record: Tuple,
) -> None:
    """
    Process a single record: summarize, embed, and upload.

    Args:
        openai_client: OpenAI client instance.
        search_client: Azure Search client instance.
        sql_record: SQL record tuple to process.
    """
    summarized = summarize_sql_record(sql_record)
    cleaned = clean_record_summary(summarized)
    embedding = embed_text(openai_client, cleaned)
    upload_to_ai_search(search_client, sql_record, cleaned, embedding)


def ingest_all_records(sql_query: str, batch_size: int = 100) -> int:
    """
    Ingest all records matching the query into Azure AI Search.

    Args:
        sql_query: SQL query to select records.
        batch_size: Number of records to process per batch.

    Returns:
        Total number of records processed.
    """
    openai_client = _create_openai_client()
    search_client = _create_search_client()

    total_records = 0

    with _get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql_query)

        while True:
            records = cursor.fetchmany(size=batch_size)
            if not records:
                break

            for record in records:
                print(f"Processing record with ID {record[0]}")
                summarize_embed_upload(openai_client, search_client, record)

            total_records += len(records)
            print(f"Total records uploaded: {total_records}")

        cursor.close()

    return total_records


if __name__ == "__main__":
    ingest_all_records("SELECT * FROM Products LIMIT 10")
