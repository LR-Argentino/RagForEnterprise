"""Azure Document Intelligence client for PDF text extraction."""

import logging
from typing import List

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

from src.config.configuration import get_config
from src.models.pdf_chunk import PDFChunk

logger = logging.getLogger(__name__)


class DocumentIntelligenceError(Exception):
    """Custom exception for Document Intelligence operations."""

    pass


def _create_document_intelligence_client() -> DocumentIntelligenceClient:
    """Create Azure Document Intelligence client using configuration."""
    config = get_config()
    return DocumentIntelligenceClient(
        endpoint=config.document_intelligence.endpoint,
        credential=AzureKeyCredential(config.document_intelligence.api_key),
    )


def _extract_page_text(page) -> str:
    """
    Extract all text content from a single page.

    Args:
        page: Azure Document Intelligence page object from AnalyzeResult.

    Returns:
        Concatenated text content from all lines on the page.
    """
    if not page.lines:
        return ""
    return "\n".join(line.content for line in page.lines)


def extract_pages_from_pdf(
    pdf_bytes: bytes,
    document_name: str,
    source_dataset: str,
) -> List[PDFChunk]:
    """
    Extract text content from each page of a PDF using Azure Document Intelligence.

    Args:
        pdf_bytes: Raw PDF file bytes.
        document_name: Name identifier for the document.
        source_dataset: Dataset source identifier.

    Returns:
        List of PDFChunk objects, one per page.

    Raises:
        DocumentIntelligenceError: If extraction fails.
    """
    logger.info(f"Extracting pages from document: {document_name}")

    try:
        client = _create_document_intelligence_client()
        config = get_config()

        poller = client.begin_analyze_document(
            model_id=config.document_intelligence.model_id,
            body=pdf_bytes,
        )
        result: AnalyzeResult = poller.result()

        if not result.pages:
            logger.warning(f"No pages found in document: {document_name}")
            return []

        total_pages = len(result.pages)
        chunks: List[PDFChunk] = []

        for page in result.pages:
            page_text = _extract_page_text(page)

            if not page_text.strip():
                logger.debug(
                    f"Skipping empty page {page.page_number} in {document_name}"
                )
                continue

            chunk = PDFChunk(
                document_name=document_name,
                page_number=page.page_number,
                page_content=page_text,
                total_pages=total_pages,
                source_dataset=source_dataset,
            )
            chunks.append(chunk)

        logger.info(
            f"Extracted {len(chunks)} non-empty pages from {document_name} "
            f"(total pages: {total_pages})"
        )
        return chunks

    except AzureError as e:
        raise DocumentIntelligenceError(
            f"Failed to extract pages from {document_name}: {e}"
        ) from e
