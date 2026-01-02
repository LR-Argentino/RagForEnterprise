"""PDF ingestion pipeline for Azure AI Search.

Includes document tracking for deduplication:
- Skip exact duplicates (same content hash)
- Replace outdated versions (delete old chunks, upload new)
- Track all ingested documents with audit trail
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Tuple

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from azure.search.documents import SearchClient
from datasets import Dataset, load_dataset
from openai import OpenAI, OpenAIError

from src.rag.clients.document_intelligence_client import (
    DocumentIntelligenceError,
    extract_pages_from_pdf,
)
from src.rag.config.configuration import get_config
from src.rag.models.pdf_chunk import PDFChunk, PDFChunkWithEmbedding
from src.rag.services.document_tracking_service import DocumentTrackingService

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


# --- Helper Functions ---


def compute_content_hash(pdf_bytes: bytes) -> str:
    """Compute MD5 hash of entire PDF content for deduplication.

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        MD5 hash as hexadecimal string.
    """
    return hashlib.md5(pdf_bytes).hexdigest()


def generate_chunk_id(document_name: str, page_number: int) -> str:
    """Generate deterministic chunk ID for Azure Search.

    Args:
        document_name: Document identifier.
        page_number: Page number within document.

    Returns:
        MD5 hash as chunk ID.
    """
    doc_id = f"{document_name}_page_{page_number}"
    return hashlib.md5(doc_id.encode()).hexdigest()


def delete_chunks_from_search(
    search_client: SearchClient,
    chunk_ids: List[str],
) -> int:
    """Delete chunks from Azure AI Search by their IDs.

    Args:
        search_client: Azure Search client for PDF index.
        chunk_ids: List of chunk IDs to delete.

    Returns:
        Number of chunks successfully deleted.

    Raises:
        AzureSearchError: If deletion fails.
    """
    if not chunk_ids:
        return 0

    try:
        documents_to_delete = [{"id": chunk_id} for chunk_id in chunk_ids]
        result = search_client.delete_documents(documents=documents_to_delete)

        deleted_count = sum(1 for r in result if r.succeeded)
        logger.info(f"Deleted {deleted_count}/{len(chunk_ids)} chunks from search index")
        return deleted_count

    except AzureError as e:
        raise AzureSearchError(f"Failed to delete chunks: {e}") from e


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
) -> str:
    """
    Upload a single embedded chunk to Azure AI Search.

    Args:
        search_client: Azure Search client for PDF index.
        chunk_with_embedding: Chunk with embedding to upload.

    Returns:
        The chunk ID that was uploaded.

    Raises:
        AzureSearchError: If upload fails.
    """
    chunk = chunk_with_embedding.chunk

    # Create unique ID from document name and page number
    doc_id = generate_chunk_id(chunk.document_name, chunk.page_number)

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
        return doc_id
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
) -> Tuple[int, List[str]]:
    """
    Process a single PDF: extract, chunk, embed, upload.

    Args:
        openai_client: OpenAI client instance.
        search_client: Azure Search client for PDF index.
        pdf_bytes: Raw PDF bytes.
        document_name: Document identifier.
        source_dataset: Dataset source identifier.

    Returns:
        Tuple of (number of chunks processed, list of chunk IDs uploaded).

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
            return 0, []

        # Embed and upload each chunk, collecting chunk IDs
        chunk_ids = []
        for chunk in chunks:
            chunk_with_embedding = embed_chunk(openai_client, chunk)
            chunk_id = upload_chunk_to_search(search_client, chunk_with_embedding)
            chunk_ids.append(chunk_id)

        logger.info(f"Processed {len(chunks)} pages from {document_name}")
        return len(chunks), chunk_ids

    except (DocumentIntelligenceError, EmbeddingError, AzureSearchError) as e:
        raise PDFIngestionError(f"Failed to process {document_name}: {e}") from e


def ingest_pdf_dataset(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 10,
    max_documents: int | None = None,
    tracking_db_path: str = "document_tracking.db",
) -> dict:
    """
    Ingest entire PDF dataset into Azure AI Search with deduplication.

    Uses document tracking to:
    - Skip exact duplicates (same content hash)
    - Replace outdated versions (delete old chunks, upload new)
    - Track all ingested documents with audit trail

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split to process.
        batch_size: Documents to process before logging progress.
        max_documents: Optional limit on documents to process.
        tracking_db_path: Path to SQLite tracking database.

    Returns:
        Dictionary with ingestion statistics:
        - total_chunks: Number of chunks uploaded
        - documents_processed: Number of documents processed
        - documents_skipped: Number of duplicates skipped
        - documents_updated: Number of documents updated (replaced)

    Raises:
        PDFIngestionError: If ingestion fails.
    """
    logger.info(f"Starting PDF ingestion from {dataset_name}")

    dataset = load_huggingface_dataset(dataset_name, split)
    openai_client = _create_openai_client()
    search_client = _create_pdf_search_client()

    # Statistics
    total_chunks = 0
    documents_processed = 0
    documents_skipped = 0
    documents_updated = 0
    documents_to_process = min(len(dataset), max_documents) if max_documents else len(dataset)

    with DocumentTrackingService(tracking_db_path) as tracking_service:
        for idx in range(documents_to_process):
            sample = dataset[idx]

            try:
                pdf_bytes, document_name = extract_pdf_bytes_from_sample(sample, idx)

                # Compute full content hash for deduplication
                content_hash = compute_content_hash(pdf_bytes)

                # Check document status
                status = tracking_service.check_document_status(
                    document_id=document_name,
                    content_hash=content_hash,
                )

                if status.is_duplicate:
                    logger.debug(f"Skipping duplicate: {document_name}")
                    documents_skipped += 1
                    continue

                if status.needs_update:
                    # Delete old chunks before uploading new ones
                    logger.info(f"Updating document: {document_name}")
                    delete_chunks_from_search(search_client, status.existing_chunk_ids)
                    documents_updated += 1

                # Process and upload the PDF
                chunks_count, chunk_ids = process_single_pdf(
                    openai_client=openai_client,
                    search_client=search_client,
                    pdf_bytes=pdf_bytes,
                    document_name=document_name,
                    source_dataset=dataset_name,
                )

                if chunks_count > 0:
                    # Register or update tracking
                    if status.needs_update:
                        tracking_service.update_document(
                            document_id=document_name,
                            content_hash=content_hash,
                            chunk_ids=chunk_ids,
                            total_pages=chunks_count,
                        )
                    else:
                        tracking_service.register_document(
                            document_id=document_name,
                            content_hash=content_hash,
                            chunk_ids=chunk_ids,
                            source_dataset=dataset_name,
                            total_pages=chunks_count,
                        )

                total_chunks += chunks_count
                documents_processed += 1

                if documents_processed % batch_size == 0:
                    logger.info(
                        f"Progress: {documents_processed}/{documents_to_process} documents, "
                        f"{total_chunks} chunks, {documents_skipped} skipped, "
                        f"{documents_updated} updated"
                    )

            except PDFIngestionError as e:
                logger.error(f"Failed to process document {idx}: {e}")
                continue

    logger.info(
        f"Ingestion complete: {documents_processed} processed, "
        f"{documents_skipped} skipped (duplicates), "
        f"{documents_updated} updated, "
        f"{total_chunks} chunks uploaded"
    )

    return {
        "total_chunks": total_chunks,
        "documents_processed": documents_processed,
        "documents_skipped": documents_skipped,
        "documents_updated": documents_updated,
    }


# --- Entry Point ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = ingest_pdf_dataset(
        dataset_name="prithivMLmods/Openpdf-MultiReceipt-1K",
        split="train",
        max_documents=15,
    )
    print(f"Ingestion complete:")
    print(f"  - Documents processed: {result['documents_processed']}")
    print(f"  - Documents skipped (duplicates): {result['documents_skipped']}")
    print(f"  - Documents updated: {result['documents_updated']}")
    print(f"  - Total chunks: {result['total_chunks']}")
