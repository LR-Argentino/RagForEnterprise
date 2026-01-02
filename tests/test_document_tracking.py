"""Tests for document tracking service.

These tests verify:
- Document registration and retrieval
- Duplicate detection (exact content match)
- Update detection (same ID, different content)
- Chunk ID tracking for deletion
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from src.rag.models.document_tracker import DocumentStatus, TrackedDocument
from src.rag.services.document_tracking_service import DocumentTrackingService
from src.rag.ingestion.pdf_ingestion import compute_content_hash, generate_chunk_id


class TestDocumentTrackingService:
    """Test DocumentTrackingService functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file for testing."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

    @pytest.fixture
    def tracking_service(self, temp_db_path):
        """Create a DocumentTrackingService with temporary database."""
        service = DocumentTrackingService(temp_db_path)
        yield service
        service.close()

    def test_service_initialization_creates_table(self, temp_db_path):
        """Test that service initialization creates the tracking table."""
        with DocumentTrackingService(temp_db_path) as service:
            # Table should exist - verify by checking a document
            result = service.get_document("nonexistent")
            assert result is None

        print(f"Database created at: {temp_db_path}")
        print("Tracking table initialized successfully")

    def test_check_document_status_new_document(self, tracking_service):
        """Test status check for a document that doesn't exist."""
        status = tracking_service.check_document_status(
            document_id="new_doc.pdf",
            content_hash="abc123",
        )

        assert status.exists is False
        assert status.is_duplicate is False
        assert status.needs_update is False
        assert status.existing_chunk_ids == []

        print(f"New document status: exists={status.exists}, "
              f"duplicate={status.is_duplicate}, needs_update={status.needs_update}")

    def test_register_document(self, tracking_service):
        """Test registering a new document."""
        chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]

        tracked = tracking_service.register_document(
            document_id="test_doc.pdf",
            content_hash="abc123def456",
            chunk_ids=chunk_ids,
            source_dataset="test_dataset",
            total_pages=3,
        )

        assert isinstance(tracked, TrackedDocument)
        assert tracked.document_id == "test_doc.pdf"
        assert tracked.content_hash == "abc123def456"
        assert tracked.chunk_ids == chunk_ids
        assert tracked.source_dataset == "test_dataset"
        assert tracked.total_pages == 3
        assert tracked.ingested_at is not None
        assert tracked.updated_at is not None

        print(f"Registered document: {tracked.document_id}")
        print(f"  Content hash: {tracked.content_hash}")
        print(f"  Chunk IDs: {tracked.chunk_ids}")
        print(f"  Ingested at: {tracked.ingested_at}")

    def test_check_document_status_exact_duplicate(self, tracking_service):
        """Test status check for exact duplicate (same content hash)."""
        chunk_ids = ["chunk_1", "chunk_2"]

        # Register original document
        tracking_service.register_document(
            document_id="original.pdf",
            content_hash="same_hash_123",
            chunk_ids=chunk_ids,
            source_dataset="test",
            total_pages=2,
        )

        # Check status with same content hash
        status = tracking_service.check_document_status(
            document_id="original.pdf",
            content_hash="same_hash_123",  # Same hash = duplicate
        )

        assert status.exists is True
        assert status.is_duplicate is True
        assert status.needs_update is False
        assert status.existing_chunk_ids == chunk_ids

        print(f"Duplicate detected: exists={status.exists}, "
              f"duplicate={status.is_duplicate}")
        print(f"Existing chunk IDs: {status.existing_chunk_ids}")

    def test_check_document_status_needs_update(self, tracking_service):
        """Test status check for document with changed content."""
        original_chunks = ["old_chunk_1", "old_chunk_2"]

        # Register original document
        tracking_service.register_document(
            document_id="changing_doc.pdf",
            content_hash="original_hash",
            chunk_ids=original_chunks,
            source_dataset="test",
            total_pages=2,
        )

        # Check status with different content hash
        status = tracking_service.check_document_status(
            document_id="changing_doc.pdf",
            content_hash="new_hash_different",  # Different hash = needs update
        )

        assert status.exists is True
        assert status.is_duplicate is False
        assert status.needs_update is True
        assert status.existing_chunk_ids == original_chunks

        print(f"Update needed: exists={status.exists}, "
              f"needs_update={status.needs_update}")
        print(f"Old chunk IDs to delete: {status.existing_chunk_ids}")

    def test_update_document(self, tracking_service):
        """Test updating an existing document."""
        original_chunks = ["old_1", "old_2"]
        new_chunks = ["new_1", "new_2", "new_3"]

        # Register original
        original = tracking_service.register_document(
            document_id="update_test.pdf",
            content_hash="original_hash",
            chunk_ids=original_chunks,
            source_dataset="test",
            total_pages=2,
        )

        # Update with new content
        updated = tracking_service.update_document(
            document_id="update_test.pdf",
            content_hash="new_hash",
            chunk_ids=new_chunks,
            total_pages=3,
        )

        assert updated.document_id == "update_test.pdf"
        assert updated.content_hash == "new_hash"
        assert updated.chunk_ids == new_chunks
        assert updated.total_pages == 3
        assert updated.ingested_at == original.ingested_at  # Original timestamp preserved
        assert updated.updated_at > original.updated_at  # Updated timestamp changed

        print(f"Document updated: {updated.document_id}")
        print(f"  Old hash: {original.content_hash} -> New hash: {updated.content_hash}")
        print(f"  Old pages: {original.total_pages} -> New pages: {updated.total_pages}")

    def test_get_document(self, tracking_service):
        """Test retrieving a tracked document."""
        tracking_service.register_document(
            document_id="retrieve_test.pdf",
            content_hash="test_hash",
            chunk_ids=["c1", "c2"],
            source_dataset="test",
            total_pages=2,
        )

        retrieved = tracking_service.get_document("retrieve_test.pdf")

        assert retrieved is not None
        assert retrieved.document_id == "retrieve_test.pdf"
        assert retrieved.content_hash == "test_hash"

        print(f"Retrieved document: {retrieved.document_id}")

    def test_get_document_not_found(self, tracking_service):
        """Test retrieving a non-existent document."""
        retrieved = tracking_service.get_document("nonexistent.pdf")

        assert retrieved is None

        print("Non-existent document correctly returns None")

    def test_delete_document(self, tracking_service):
        """Test deleting a tracked document."""
        tracking_service.register_document(
            document_id="delete_test.pdf",
            content_hash="test_hash",
            chunk_ids=["c1"],
            source_dataset="test",
            total_pages=1,
        )

        # Verify it exists
        assert tracking_service.get_document("delete_test.pdf") is not None

        # Delete it
        deleted = tracking_service.delete_document("delete_test.pdf")
        assert deleted is True

        # Verify it's gone
        assert tracking_service.get_document("delete_test.pdf") is None

        print("Document deleted successfully")

    def test_delete_nonexistent_document(self, tracking_service):
        """Test deleting a document that doesn't exist."""
        deleted = tracking_service.delete_document("nonexistent.pdf")
        assert deleted is False

        print("Delete of non-existent document correctly returns False")

    def test_context_manager(self, temp_db_path):
        """Test service works correctly as context manager."""
        with DocumentTrackingService(temp_db_path) as service:
            service.register_document(
                document_id="context_test.pdf",
                content_hash="test",
                chunk_ids=["c1"],
                source_dataset="test",
                total_pages=1,
            )
            result = service.get_document("context_test.pdf")
            assert result is not None

        print("Context manager works correctly")


class TestHelperFunctions:
    """Test helper functions for hashing and ID generation."""

    def test_compute_content_hash(self):
        """Test content hash computation."""
        content = b"This is test PDF content"
        hash1 = compute_content_hash(content)

        # Same content = same hash
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

        # Different content = different hash
        different_content = b"Different content here"
        hash3 = compute_content_hash(different_content)
        assert hash1 != hash3

        # Hash is 32 characters (MD5 hex)
        assert len(hash1) == 32

        print(f"Content hash: {hash1}")
        print(f"Same content produces same hash: {hash1 == hash2}")
        print(f"Different content produces different hash: {hash1 != hash3}")

    def test_generate_chunk_id(self):
        """Test chunk ID generation."""
        chunk_id1 = generate_chunk_id("doc.pdf", 1)
        chunk_id2 = generate_chunk_id("doc.pdf", 2)
        chunk_id3 = generate_chunk_id("other.pdf", 1)

        # Same doc + page = same ID
        assert chunk_id1 == generate_chunk_id("doc.pdf", 1)

        # Same doc, different page = different ID
        assert chunk_id1 != chunk_id2

        # Different doc, same page = different ID
        assert chunk_id1 != chunk_id3

        # ID is 32 characters (MD5 hex)
        assert len(chunk_id1) == 32

        print(f"Chunk ID for doc.pdf page 1: {chunk_id1}")
        print(f"Chunk ID for doc.pdf page 2: {chunk_id2}")
        print(f"Chunk ID for other.pdf page 1: {chunk_id3}")


class TestDocumentStatusModel:
    """Test DocumentStatus dataclass."""

    def test_document_status_new(self):
        """Test DocumentStatus for new document."""
        status = DocumentStatus(
            exists=False,
            is_duplicate=False,
            needs_update=False,
            existing_chunk_ids=[],
        )

        assert not status.exists
        assert not status.is_duplicate
        assert not status.needs_update
        assert status.existing_chunk_ids == []

        print(f"New document status: {status}")

    def test_document_status_duplicate(self):
        """Test DocumentStatus for duplicate."""
        status = DocumentStatus(
            exists=True,
            is_duplicate=True,
            needs_update=False,
            existing_chunk_ids=["chunk1", "chunk2"],
        )

        assert status.exists
        assert status.is_duplicate
        assert not status.needs_update

        print(f"Duplicate status: {status}")

    def test_document_status_update_needed(self):
        """Test DocumentStatus for document needing update."""
        status = DocumentStatus(
            exists=True,
            is_duplicate=False,
            needs_update=True,
            existing_chunk_ids=["old_chunk1"],
        )

        assert status.exists
        assert not status.is_duplicate
        assert status.needs_update

        print(f"Update needed status: {status}")


class TestTrackedDocumentModel:
    """Test TrackedDocument dataclass."""

    def test_tracked_document_creation(self):
        """Test TrackedDocument dataclass creation."""
        now = datetime.now(timezone.utc)

        doc = TrackedDocument(
            document_id="test.pdf",
            content_hash="abc123",
            chunk_ids=["c1", "c2"],
            source_dataset="test",
            total_pages=2,
            ingested_at=now,
            updated_at=now,
        )

        assert doc.document_id == "test.pdf"
        assert doc.content_hash == "abc123"
        assert doc.chunk_ids == ["c1", "c2"]
        assert doc.source_dataset == "test"
        assert doc.total_pages == 2
        assert doc.ingested_at == now
        assert doc.updated_at == now

        print(f"TrackedDocument created: {doc}")

    def test_tracked_document_is_frozen(self):
        """Test that TrackedDocument is immutable."""
        now = datetime.now(timezone.utc)

        doc = TrackedDocument(
            document_id="test.pdf",
            content_hash="abc123",
            chunk_ids=["c1"],
            source_dataset="test",
            total_pages=1,
            ingested_at=now,
            updated_at=now,
        )

        with pytest.raises(AttributeError):
            doc.content_hash = "new_hash"

        print("TrackedDocument is correctly frozen (immutable)")
