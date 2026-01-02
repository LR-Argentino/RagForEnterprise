"""PDF chunk models for document ingestion."""

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass(frozen=True)
class PDFChunk:
    """Represents a single page chunk from a PDF document."""

    document_name: str
    page_number: int
    page_content: str
    total_pages: int
    source_dataset: str


@dataclass(frozen=True)
class PDFChunkWithEmbedding:
    """PDF chunk with its embedding vector."""

    chunk: PDFChunk
    embedding: List[float]
    date_ingested: datetime
