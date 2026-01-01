"""Document tracking service for deduplication and version management.

Tracks ingested documents to enable:
- Duplicate detection (skip re-uploading identical content)
- Version updates (delete old chunks when content changes)
- Audit trail (when documents were ingested/updated)
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from ..clients import SqliteClient
from ..models import TrackedDocument, DocumentStatus

logger = logging.getLogger(__name__)

# SQL statements
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_tracking (
    document_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    chunk_ids TEXT NOT NULL,
    source_dataset TEXT NOT NULL,
    total_pages INTEGER NOT NULL,
    ingested_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_content_hash ON document_tracking(content_hash)
"""


class DocumentTrackingService:
    """Service for tracking ingested documents and managing deduplication."""

    def __init__(self, db_path: str = "document_tracking.db"):
        """Initialize the document tracking service.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = db_path
        self._sqlite_client = SqliteClient(db_path)
        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        """Create the tracking table if it doesn't exist."""
        self._sqlite_client.execute_query(CREATE_TABLE_SQL)
        self._sqlite_client.execute_query(CREATE_INDEX_SQL)
        logger.debug("Document tracking table initialized")

    def check_document_status(
        self,
        document_id: str,
        content_hash: str,
    ) -> DocumentStatus:
        """Check the status of a document before ingestion.

        Args:
            document_id: External identifier for the document.
            content_hash: MD5 hash of the document content.

        Returns:
            DocumentStatus indicating whether to skip, update, or ingest.
        """
        result = self._sqlite_client.execute_query(
            "SELECT content_hash, chunk_ids FROM document_tracking WHERE document_id = ?",
            (document_id,),
        )

        if not result:
            # Document doesn't exist - new ingestion
            return DocumentStatus(
                exists=False,
                is_duplicate=False,
                needs_update=False,
                existing_chunk_ids=[],
            )

        existing_hash, chunk_ids_json = result[0]
        existing_chunk_ids = json.loads(chunk_ids_json)

        if existing_hash == content_hash:
            # Exact duplicate - skip
            return DocumentStatus(
                exists=True,
                is_duplicate=True,
                needs_update=False,
                existing_chunk_ids=existing_chunk_ids,
            )

        # Content changed - needs update
        return DocumentStatus(
            exists=True,
            is_duplicate=False,
            needs_update=True,
            existing_chunk_ids=existing_chunk_ids,
        )

    def register_document(
        self,
        document_id: str,
        content_hash: str,
        chunk_ids: list[str],
        source_dataset: str,
        total_pages: int,
    ) -> TrackedDocument:
        """Register a newly ingested document.

        Args:
            document_id: External identifier for the document.
            content_hash: MD5 hash of the document content.
            chunk_ids: List of Azure Search chunk IDs.
            source_dataset: Dataset source identifier.
            total_pages: Number of pages in the document.

        Returns:
            The registered TrackedDocument.
        """
        now = datetime.now(timezone.utc)
        chunk_ids_json = json.dumps(chunk_ids)

        self._sqlite_client.execute_query(
            """INSERT INTO document_tracking
               (document_id, content_hash, chunk_ids, source_dataset, total_pages, ingested_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                content_hash,
                chunk_ids_json,
                source_dataset,
                total_pages,
                now.isoformat(),
                now.isoformat(),
            ),
        )

        logger.info(f"Registered new document: {document_id} with {total_pages} pages")

        return TrackedDocument(
            document_id=document_id,
            content_hash=content_hash,
            chunk_ids=chunk_ids,
            source_dataset=source_dataset,
            total_pages=total_pages,
            ingested_at=now,
            updated_at=now,
        )

    def update_document(
        self,
        document_id: str,
        content_hash: str,
        chunk_ids: list[str],
        total_pages: int,
    ) -> TrackedDocument:
        """Update an existing document's tracking information.

        Args:
            document_id: External identifier for the document.
            content_hash: New MD5 hash of the document content.
            chunk_ids: New list of Azure Search chunk IDs.
            total_pages: New number of pages in the document.

        Returns:
            The updated TrackedDocument.
        """
        now = datetime.now(timezone.utc)
        chunk_ids_json = json.dumps(chunk_ids)

        # Get the original ingestion timestamp
        result = self._sqlite_client.execute_query(
            "SELECT ingested_at, source_dataset FROM document_tracking WHERE document_id = ?",
            (document_id,),
        )

        if not result:
            raise ValueError(f"Document {document_id} not found for update")

        ingested_at_str, source_dataset = result[0]
        ingested_at = datetime.fromisoformat(ingested_at_str)

        self._sqlite_client.execute_query(
            """UPDATE document_tracking
               SET content_hash = ?, chunk_ids = ?, total_pages = ?, updated_at = ?
               WHERE document_id = ?""",
            (content_hash, chunk_ids_json, total_pages, now.isoformat(), document_id),
        )

        logger.info(f"Updated document: {document_id} with {total_pages} pages")

        return TrackedDocument(
            document_id=document_id,
            content_hash=content_hash,
            chunk_ids=chunk_ids,
            source_dataset=source_dataset,
            total_pages=total_pages,
            ingested_at=ingested_at,
            updated_at=now,
        )

    def get_document(self, document_id: str) -> Optional[TrackedDocument]:
        """Get a tracked document by its ID.

        Args:
            document_id: External identifier for the document.

        Returns:
            TrackedDocument if found, None otherwise.
        """
        result = self._sqlite_client.execute_query(
            """SELECT document_id, content_hash, chunk_ids, source_dataset,
                      total_pages, ingested_at, updated_at
               FROM document_tracking WHERE document_id = ?""",
            (document_id,),
        )

        if not result:
            return None

        row = result[0]
        return TrackedDocument(
            document_id=row[0],
            content_hash=row[1],
            chunk_ids=json.loads(row[2]),
            source_dataset=row[3],
            total_pages=row[4],
            ingested_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
        )

    def delete_document(self, document_id: str) -> bool:
        """Delete a document from tracking.

        Args:
            document_id: External identifier for the document.

        Returns:
            True if document was deleted, False if not found.
        """
        # Check if exists first
        existing = self.get_document(document_id)
        if not existing:
            return False

        self._sqlite_client.execute_query(
            "DELETE FROM document_tracking WHERE document_id = ?",
            (document_id,),
        )

        logger.info(f"Deleted document tracking: {document_id}")
        return True

    def close(self) -> None:
        """Close the database connection."""
        self._sqlite_client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
        return False
