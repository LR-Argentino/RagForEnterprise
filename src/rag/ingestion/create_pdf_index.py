"""Create Azure AI Search index for PDF documents."""

import logging

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    VectorSearch,
    VectorSearchProfile,
)

from src.rag.config.configuration import get_config

logger = logging.getLogger(__name__)


def create_pdf_document_index() -> None:
    """
    Create Azure AI Search index for PDF document chunks.

    The index supports hybrid search (text + vector) with the following fields:
    - id: Unique identifier for each chunk
    - document_name: Name of the source PDF document
    - page_number: Page number within the document
    - page_content: Full text content of the page
    - vector: Embedding vector for semantic search
    - source_dataset: Dataset identifier for filtering
    - total_pages: Total number of pages in source document
    - date_ingested: Timestamp of ingestion

    Raises:
        AzureError: If index creation fails.
    """
    config = get_config()

    index_client = SearchIndexClient(
        endpoint=config.azure_ai_search.endpoint,
        credential=AzureKeyCredential(config.azure_ai_search.api_key),
    )

    index_name = config.azure_ai_search.pdf_index_name

    fields = [
        SearchField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchField(
            name="document_name",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            sortable=True,
        ),
        SearchField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SearchField(
            name="page_content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchField(
            name="vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=config.openai.embedding_dimensions,
            vector_search_profile_name="pdf-hnsw-profile",
        ),
        SearchField(
            name="source_dataset",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="total_pages",
            type=SearchFieldDataType.Int32,
            filterable=True,
        ),
        SearchField(
            name="date_ingested",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="pdf-hnsw-algorithm"),
        ],
        profiles=[
            VectorSearchProfile(
                name="pdf-hnsw-profile",
                algorithm_configuration_name="pdf-hnsw-algorithm",
            ),
        ],
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
    )

    try:
        result = index_client.create_or_update_index(index)
        logger.info(f"Index '{result.name}' created/updated successfully")
        print(f"Index '{result.name}' created/updated successfully")
    except AzureError as e:
        logger.error(f"Failed to create index: {e}")
        raise


def delete_pdf_document_index() -> None:
    """
    Delete the PDF document index.

    Raises:
        AzureError: If index deletion fails.
    """
    config = get_config()

    index_client = SearchIndexClient(
        endpoint=config.azure_ai_search.endpoint,
        credential=AzureKeyCredential(config.azure_ai_search.api_key),
    )

    index_name = config.azure_ai_search.pdf_index_name

    try:
        index_client.delete_index(index_name)
        logger.info(f"Index '{index_name}' deleted successfully")
        print(f"Index '{index_name}' deleted successfully")
    except AzureError as e:
        logger.error(f"Failed to delete index: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_pdf_document_index()
