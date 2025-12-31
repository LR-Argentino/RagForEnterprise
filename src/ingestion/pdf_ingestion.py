"""PDF ingestion pipeline for Azure AI Search."""

import hashlib
import io
import logging
from datetime import datetime, timezone
from typing import List, Tuple

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from azure.search.documents import SearchClient
from datasets import Dataset, load_dataset
from openai import OpenAI, OpenAIError

from src.clients.document_intelligence_client import (
    DocumentIntelligenceError,
    extract_pages_from_pdf,
)
from src.config.configuration import get_config
from src.models.pdf_chunk import PDFChunk, PDFChunkWithEmbedding

logger = logging.getLogger(__name__)


class PDFIngestionError(Exception):
    """Custom exception for PDF ingestion pipeline failures."""

    pass


class EmbeddingError(Exception):
    """Custom exception for embedding generation failures."""

    pass


class AzureSearchError(Exception):
    """Custom exception for Azure Search upload failures."""

    pass


# --- Client Factory Functions ---


def _create_openai_client() -> OpenAI:
    """Create OpenAI client using configuration."""
    config = get_config()
    return OpenAI(api_key=config.openai.api_key)


def _create_pdf_search_client() -> SearchClient:
    """Create Azure AI Search client for PDF index."""
    config = get_config()
    return SearchClient(
        endpoint=config.azure_ai_search.endpoint,
        index_name=config.azure_ai_search.pdf_index_name,
        credential=AzureKeyCredential(config.azure_ai_search.api_key),
    )


# --- Core Pipeline Functions ---


def load_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
) -> Dataset:
    """
    Load PDF dataset from Hugging Face.

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split to load.

    Returns:
        Loaded dataset object.

    Raises:
        PDFIngestionError: If dataset loading fails.
    """
    logger.info(f"Loading dataset: {dataset_name} (split: {split})")
    try:
        dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
        return dataset
    except Exception as e:
        raise PDFIngestionError(f"Failed to load dataset {dataset_name}: {e}") from e


def extract_pdf_bytes_from_sample(sample: dict, sample_index: int) -> Tuple[bytes, str]:
    """
    Extract PDF bytes and generate document name from dataset sample.

    The dataset contains pdfplumber PDF objects. This function extracts
    the raw bytes from the underlying file stream.

    Args:
        sample: Single sample from Hugging Face dataset.
        sample_index: Index of the sample in the dataset.

    Returns:
        Tuple of (pdf_bytes, document_name).

    Raises:
        PDFIngestionError: If PDF extraction fails.
    """
    try:
        pdf_obj = sample.get("pdf")
        if pdf_obj is None:
            raise PDFIngestionError(f"Sample {sample_index} has no 'pdf' field")

        # pdfplumber PDF object has a stream attribute for the underlying file
        if hasattr(pdf_obj, "stream"):
            stream = pdf_obj.stream
            stream.seek(0)
            pdf_bytes = stream.read()
            stream.seek(0)
        elif hasattr(pdf_obj, "path"):
            # If it's a file path
            with open(pdf_obj.path, "rb") as f:
                pdf_bytes = f.read()
        elif isinstance(pdf_obj, bytes):
            pdf_bytes = pdf_obj
        else:
            raise PDFIngestionError(
                f"Unsupported PDF object type: {type(pdf_obj)} for sample {sample_index}"
            )

        # Generate document name from content hash for uniqueness
        content_hash = hashlib.md5(pdf_bytes[:1024]).hexdigest()[:8]
        document_name = f"receipt_{sample_index}_{content_hash}.pdf"

        return pdf_bytes, document_name

    except PDFIngestionError:
        raise
    except Exception as e:
        raise PDFIngestionError(
            f"Failed to extract PDF from sample {sample_index}: {e}"
        ) from e


def embed_chunk(
    openai_client: OpenAI,
    chunk: PDFChunk,
) -> PDFChunkWithEmbedding:
    """
    Generate embedding for a PDF chunk.

    Args:
        openai_client: OpenAI client instance.
        chunk: PDFChunk to embed.

    Returns:
        PDFChunkWithEmbedding with vector.

    Raises:
        EmbeddingError: If embedding generation fails.
    """
    config = get_config()

    try:
        response = openai_client.embeddings.create(
            input=chunk.page_content,
            model=config.openai.embedding_model,
            dimensions=config.openai.embedding_dimensions,
        )
        embedding = response.data[0].embedding

        return PDFChunkWithEmbedding(
            chunk=chunk,
            embedding=embedding,
            date_ingested=datetime.now(timezone.utc),
        )

    except OpenAIError as e:
        raise EmbeddingError(
            f"Failed to embed chunk from {chunk.document_name} "
            f"page {chunk.page_number}: {e}"
        ) from e


def upload_chunk_to_search(
    search_client: SearchClient,
    chunk_with_embedding: PDFChunkWithEmbedding,
) -> None:
    """
    Upload a single embedded chunk to Azure AI Search.

    Args:
        search_client: Azure Search client for PDF index.
        chunk_with_embedding: Chunk with embedding to upload.

    Raises:
        AzureSearchError: If upload fails.
    """
    chunk = chunk_with_embedding.chunk

    # Create unique ID from document name and page number
    doc_id = f"{chunk.document_name}_page_{chunk.page_number}"
    doc_id = hashlib.md5(doc_id.encode()).hexdigest()

    document = {
        "id": doc_id,
        "document_name": chunk.document_name,
        "page_number": chunk.page_number,
        "page_content": chunk.page_content,
        "vector": chunk_with_embedding.embedding,
        "source_dataset": chunk.source_dataset,
        "total_pages": chunk.total_pages,
        "date_ingested": chunk_with_embedding.date_ingested.isoformat(),
    }

    try:
        search_client.merge_or_upload_documents(documents=[document])
        logger.debug(
            f"Uploaded chunk: {chunk.document_name} page {chunk.page_number}"
        )
    except AzureError as e:
        raise AzureSearchError(
            f"Failed to upload chunk {chunk.document_name} "
            f"page {chunk.page_number}: {e}"
        ) from e


def process_single_pdf(
    openai_client: OpenAI,
    search_client: SearchClient,
    pdf_bytes: bytes,
    document_name: str,
    source_dataset: str,
) -> int:
    """
    Process a single PDF: extract, chunk, embed, upload.

    Args:
        openai_client: OpenAI client instance.
        search_client: Azure Search client for PDF index.
        pdf_bytes: Raw PDF bytes.
        document_name: Document identifier.
        source_dataset: Dataset source identifier.

    Returns:
        Number of chunks (pages) processed.

    Raises:
        PDFIngestionError: If processing fails.
    """
    try:
        # Extract pages using Document Intelligence
        chunks = extract_pages_from_pdf(
            pdf_bytes=pdf_bytes,
            document_name=document_name,
            source_dataset=source_dataset,
        )

        if not chunks:
            logger.warning(f"No content extracted from {document_name}")
            return 0

        # Embed and upload each chunk
        for chunk in chunks:
            chunk_with_embedding = embed_chunk(openai_client, chunk)
            upload_chunk_to_search(search_client, chunk_with_embedding)

        logger.info(f"Processed {len(chunks)} pages from {document_name}")
        return len(chunks)

    except (DocumentIntelligenceError, EmbeddingError, AzureSearchError) as e:
        raise PDFIngestionError(f"Failed to process {document_name}: {e}") from e


def ingest_pdf_dataset(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 10,
    max_documents: int | None = None,
) -> int:
    """
    Ingest entire PDF dataset into Azure AI Search.

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split to process.
        batch_size: Documents to process before logging progress.
        max_documents: Optional limit on documents to process.

    Returns:
        Total number of chunks ingested.

    Raises:
        PDFIngestionError: If ingestion fails.
    """
    logger.info(f"Starting PDF ingestion from {dataset_name}")

    dataset = load_huggingface_dataset(dataset_name, split)
    openai_client = _create_openai_client()
    search_client = _create_pdf_search_client()

    total_chunks = 0
    documents_processed = 0
    documents_to_process = min(len(dataset), max_documents) if max_documents else len(dataset)

    for idx in range(documents_to_process):
        sample = dataset[idx]

        try:
            pdf_bytes, document_name = extract_pdf_bytes_from_sample(sample, idx)

            chunks_count = process_single_pdf(
                openai_client=openai_client,
                search_client=search_client,
                pdf_bytes=pdf_bytes,
                document_name=document_name,
                source_dataset=dataset_name,
            )

            total_chunks += chunks_count
            documents_processed += 1

            if documents_processed % batch_size == 0:
                logger.info(
                    f"Progress: {documents_processed}/{documents_to_process} documents, "
                    f"{total_chunks} total chunks"
                )

        except PDFIngestionError as e:
            logger.error(f"Failed to process document {idx}: {e}")
            continue

    logger.info(
        f"Ingestion complete: {documents_processed} documents, "
        f"{total_chunks} chunks uploaded"
    )
    return total_chunks


# --- Entry Point ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    total_chunks = ingest_pdf_dataset(
        dataset_name="prithivMLmods/Openpdf-MultiReceipt-1K",
        split="train",
        max_documents=5,
    )
    print(f"Ingestion complete. Total chunks: {total_chunks}")
