# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run with development config (SQLite backend)
APP_ENV=dev python src/main.py

# Run with test config (CosmosDB backend)
APP_ENV=test python src/main.py

# Run all tests
pytest -s

# Run single test
pytest tests/test_automated_evals.py::test_run_RAG -s
```

## Environment Configuration

Set `APP_ENV` to select config file:
- `APP_ENV=dev` → `config_dev.yaml` (SQLite, local, free)
- `APP_ENV=test` → `config_test.yaml` (CosmosDB, production-like)
- Default → `config.yaml`

## Required Environment Variables

Create a `.env` file with:
- `OPENAI_API_KEY` - OpenAI API key
- `AI_SEARCH_KEY` - Azure AI Search API key
- `AI_SEARCH_NAME` - Azure AI Search service name
- `DOCUMENT_INTELLIGENCE_KEY` - Azure Document Intelligence key

Only for `APP_ENV=test` (CosmosDB):
- `COSMOSDB_ENDPOINT` - Azure Cosmos DB account endpoint
- `COSMOSDB_KEY` - Azure Cosmos DB account key

## Architecture

RAG (Retrieval Augmented Generation) system using Azure AI Search for document retrieval and OpenAI for embeddings/generation.

### Module Responsibilities

- **src/chat/rag_chat.py** - Core RAG orchestration with AutoGen agents (Search Agent + Writer Agent)
- **src/chat/question_triage.py** - Categorizes questions into PRODUCT, ORDER, or CLARIFY
- **src/ingestion/** - Data pipeline: embed product records and upload to Azure AI Search
- **src/clients/sqlite_client.py** - SQLite database abstraction
- **src/clients/cosmosdb_client.py** - Azure Cosmos DB client for audit logs
- **src/config/configuration.py** - Centralized config loading (`get_config()`)
- **src/services/rag_log_service.py** - Logs RAG interactions to Azure Cosmos DB

### Configuration

Environment-specific config files:
- **config_dev.yaml** - Development (SQLite backend)
- **config_test.yaml** - Test/Staging (CosmosDB backend)
- **config.yaml** - Default fallback
- **.env** - Secrets only (API keys)

### Key Patterns

- Async/await for all external API calls (AsyncOpenAI, AsyncSearchClient)
- Context managers for resource cleanup (database connections, search clients)
- Dataclass models for typed configuration and data objects
- Factory functions for client creation (`_create_openai_client`, `_create_azure_search_client`)

### Entry Points

- `RAGChat()` - Main async function for Q&A
- `src/ingestion/upload_sql_records_to_ai_search.py` - Runnable script for data ingestion
