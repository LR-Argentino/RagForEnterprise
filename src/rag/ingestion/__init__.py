"""Data ingestion module for Azure AI Search."""

from src.rag.ingestion.upload_sql_records_to_ai_search import (
    clean_record_summary,
    embed_text,
    ingest_all_records,
    summarize_embed_upload,
    summarize_sql_record,
    upload_to_ai_search,
)
from src.rag.ingestion.pdf_ingestion import (
    PDFIngestionError,
    EmbeddingError,
    AzureSearchError,
    load_huggingface_dataset,
    extract_pdf_bytes_from_sample,
    embed_chunk,
    upload_chunk_to_search,
    process_single_pdf,
    ingest_pdf_dataset,
)
from src.rag.ingestion.create_pdf_index import (
    create_pdf_document_index,
    delete_pdf_document_index,
)

__all__ = [
    # SQL ingestion
    "clean_record_summary",
    "embed_text",
    "ingest_all_records",
    "summarize_embed_upload",
    "summarize_sql_record",
    "upload_to_ai_search",
    # PDF ingestion
    "PDFIngestionError",
    "EmbeddingError",
    "AzureSearchError",
    "load_huggingface_dataset",
    "extract_pdf_bytes_from_sample",
    "embed_chunk",
    "upload_chunk_to_search",
    "process_single_pdf",
    "ingest_pdf_dataset",
    # Index management
    "create_pdf_document_index",
    "delete_pdf_document_index",
]
