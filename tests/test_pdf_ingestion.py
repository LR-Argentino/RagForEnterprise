"""Tests for PDF ingestion pipeline.

These tests require:
- Valid Azure Document Intelligence credentials
- Valid Azure AI Search credentials
- Valid OpenAI API key
- Network connectivity to Azure and OpenAI services
"""

import pytest

from src.rag.config import get_config
from src.rag.ingestion.pdf_ingestion import (
    _create_openai_client,
    _create_pdf_search_client,
    embed_chunk,
    load_huggingface_dataset,
)
from src.rag.ingestion import create_pdf_document_index
from src.rag.models import PDFChunk, PDFChunkWithEmbedding


class TestConfiguration:
    """Test configuration loading for PDF ingestion."""

    def test_config_loads_document_intelligence(self):
        """Test that Document Intelligence config is properly loaded."""
        config = get_config()

        assert config.document_intelligence is not None
        assert config.document_intelligence.api_key is not None
        assert len(config.document_intelligence.api_key) > 0
        assert config.document_intelligence.endpoint is not None
        assert config.document_intelligence.model_id == "prebuilt-layout"

        print(f"Document Intelligence endpoint: {config.document_intelligence.endpoint}")
        print(f"Document Intelligence model: {config.document_intelligence.model_id}")

    def test_config_loads_pdf_index_name(self):
        """Test that PDF index name is properly configured."""
        config = get_config()

        assert config.azure_ai_search.pdf_index_name is not None
        assert config.azure_ai_search.pdf_index_name == "pdf-document-index"

        print(f"PDF index name: {config.azure_ai_search.pdf_index_name}")


class TestClientCreation:
    """Test client factory functions."""

    def test_create_openai_client(self):
        """Test OpenAI client creation."""
        client = _create_openai_client()

        assert client is not None
        print("OpenAI client created successfully")

    def test_create_pdf_search_client(self):
        """Test Azure Search client creation for PDF index."""
        client = _create_pdf_search_client()

        assert client is not None
        print("PDF Search client created successfully")


class TestDatasetLoading:
    """Test Hugging Face dataset loading."""

    def test_load_huggingface_dataset_small(self):
        """Test loading a small subset of the dataset."""
        dataset = load_huggingface_dataset(
            dataset_name="prithivMLmods/Openpdf-MultiReceipt-1K",
            split="train[:2]",
        )

        assert dataset is not None
        assert len(dataset) == 2
        assert "pdf" in dataset.column_names

        print(f"Dataset loaded with {len(dataset)} samples")
        print(f"Columns: {dataset.column_names}")

    def test_dataset_sample_has_pdf(self):
        """Test that dataset samples contain PDF objects."""
        dataset = load_huggingface_dataset(
            dataset_name="prithivMLmods/Openpdf-MultiReceipt-1K",
            split="train[:1]",
        )

        sample = dataset[0]
        pdf_obj = sample.get("pdf")

        assert pdf_obj is not None
        print(f"PDF object type: {type(pdf_obj)}")

        # Check if it has pages (pdfplumber object)
        if hasattr(pdf_obj, "pages"):
            print(f"PDF has {len(pdf_obj.pages)} pages")


class TestEmbedding:
    """Test embedding generation."""

    def test_embed_chunk(self):
        """Test embedding generation for a PDF chunk."""
        openai_client = _create_openai_client()
        config = get_config()

        test_chunk = PDFChunk(
            document_name="test_doc.pdf",
            page_number=1,
            page_content="This is test content for embedding generation. "
            "It contains sample text that will be converted to a vector.",
            total_pages=1,
            source_dataset="test",
        )

        result = embed_chunk(openai_client, test_chunk)

        assert isinstance(result, PDFChunkWithEmbedding)
        assert result.chunk == test_chunk
        assert len(result.embedding) == config.openai.embedding_dimensions
        assert result.date_ingested is not None

        print(f"Embedding dimensions: {len(result.embedding)}")
        print(f"First 5 values: {result.embedding[:5]}")
        print(f"Date ingested: {result.date_ingested}")


class TestIndexCreation:
    """Test Azure Search index creation."""

    def test_create_pdf_index(self):
        """Test creating the PDF document index."""
        # This will create or update the index
        create_pdf_document_index()

        # Verify by creating a client and checking index exists
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents.indexes import SearchIndexClient

        config = get_config()
        index_client = SearchIndexClient(
            endpoint=config.azure_ai_search.endpoint,
            credential=AzureKeyCredential(config.azure_ai_search.api_key),
        )

        index = index_client.get_index(config.azure_ai_search.pdf_index_name)

        assert index is not None
        assert index.name == config.azure_ai_search.pdf_index_name

        # Verify fields exist
        field_names = [f.name for f in index.fields]
        expected_fields = [
            "id",
            "document_name",
            "page_number",
            "page_content",
            "vector",
            "source_dataset",
            "total_pages",
            "date_ingested",
        ]

        for field in expected_fields:
            assert field in field_names, f"Field '{field}' not found in index"

        print(f"Index '{index.name}' verified successfully")
        print(f"Fields: {field_names}")


class TestPDFChunkModel:
    """Test PDF chunk data models."""

    def test_pdf_chunk_creation(self):
        """Test PDFChunk dataclass creation."""
        chunk = PDFChunk(
            document_name="test.pdf",
            page_number=1,
            page_content="Test content",
            total_pages=5,
            source_dataset="test_dataset",
        )

        assert chunk.document_name == "test.pdf"
        assert chunk.page_number == 1
        assert chunk.page_content == "Test content"
        assert chunk.total_pages == 5
        assert chunk.source_dataset == "test_dataset"

        print(f"PDFChunk created: {chunk}")

    def test_pdf_chunk_is_frozen(self):
        """Test that PDFChunk is immutable."""
        chunk = PDFChunk(
            document_name="test.pdf",
            page_number=1,
            page_content="Test content",
            total_pages=5,
            source_dataset="test_dataset",
        )

        with pytest.raises(AttributeError):
            chunk.page_number = 2

        print("PDFChunk is correctly frozen (immutable)")
