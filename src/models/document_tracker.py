"""Document tracking models for deduplication and version management."""

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass(frozen=True)
class TrackedDocument:
    """Represents a tracked document in the ingestion system.

    Used to detect duplicates and manage document updates in Azure AI Search.
    """

    document_id: str  # External identifier (e.g., filename or sample index)
    content_hash: str  # MD5 hash of the entire document content
    chunk_ids: List[str]  # List of Azure Search chunk IDs for this document
    source_dataset: str  # Dataset source identifier
    total_pages: int  # Number of pages/chunks in the document
    ingested_at: datetime  # First ingestion timestamp
    updated_at: datetime  # Last update timestamp


@dataclass(frozen=True)
class DocumentStatus:
    """Result of checking a document's status before ingestion."""

    exists: bool  # Whether document already exists in tracking
    is_duplicate: bool  # True if content hash matches (exact duplicate)
    needs_update: bool  # True if exists but content changed
    existing_chunk_ids: List[str]  # Chunk IDs to delete if updating
