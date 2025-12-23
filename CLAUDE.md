# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest -s

# Run single test
pytest tests/test_automated_evals.py::test_run_RAG -s
```

## Required Environment Variables

Create a `.env` file with:
- `OPENAI_API_KEY` - OpenAI API key
- `AI_SEARCH_KEY` - Azure AI Search API key
- `AI_SEARCH_NAME` - Azure AI Search service name

## Architecture

RAG (Retrieval Augmented Generation) system using Azure AI Search for document retrieval and OpenAI for embeddings/generation.

### Module Responsibilities

- **src/chat/rag_chat.py** - Core RAG orchestration with AutoGen agents (Search Agent + Writer Agent)
- **src/chat/question_triage.py** - Categorizes questions into PRODUCT, ORDER, or CLARIFY
- **src/ingestion/** - Data pipeline: embed product records and upload to Azure AI Search
- **src/clients/sqlite_client.py** - SQLite database abstraction
- **src/config/configuration.py** - Centralized config loading (`get_config()`)
- **src/services/rag_log_service.py** - Logs RAG interactions to SQLite

### Configuration

- **config.yaml** - Non-sensitive settings (models, endpoints, database paths)
- **.env** - Secrets only

### Key Patterns

- Async/await for all external API calls (AsyncOpenAI, AsyncSearchClient)
- Context managers for resource cleanup (database connections, search clients)
- Dataclass models for typed configuration and data objects
- Factory functions for client creation (`_create_openai_client`, `_create_azure_search_client`)

### Entry Points

- `RAGChat()` - Main async function for Q&A
- `src/ingestion/upload_sql_records_to_ai_search.py` - Runnable script for data ingestion
