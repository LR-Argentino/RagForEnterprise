"""Data ingestion module for Azure AI Search."""

from src.ingestion.upload_sql_records_to_ai_search import (
    clean_record_summary,
    embed_text,
    ingest_all_records,
    summarize_embed_upload,
    summarize_sql_record,
    upload_to_ai_search,
)

__all__ = [
    "clean_record_summary",
    "embed_text",
    "ingest_all_records",
    "summarize_embed_upload",
    "summarize_sql_record",
    "upload_to_ai_search",
]
