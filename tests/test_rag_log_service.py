"""Tests for RagLogService with SQLite backend.

These tests verify:
- RagLogService initialization with SQLite backend
- Log storage and retrieval
- Table auto-creation
- Backend toggle functionality
"""

import os
import tempfile
import uuid

import pytest

from src.models import RagLog


class TestRagLogServiceSQLite:
    """Test RagLogService with SQLite backend."""

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
    def sqlite_config(self, temp_db_path, monkeypatch):
        """Configure SQLite backend for testing."""
        # Create a temporary config.yaml content
        import yaml
        from pathlib import Path

        # Set up minimal environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AI_SEARCH_KEY", "test-key")
        monkeypatch.setenv("DOCUMENT_INTELLIGENCE_KEY", "test-key")

        # Create temp config file
        config_content = {
            "openai": {"model": "gpt-4o", "embedding_model": {"name": "test", "dimensions": 1536}},
            "ai_search": {
                "index_name": "test",
                "pdf_index_name": "test",
                "endpoint": "https://test.search.windows.net",
            },
            "document_intelligence": {"endpoint": "https://test.cognitiveservices.azure.com/", "model_id": "test"},
            "database": {"path": "test.db", "document_tracking_path": "test.db"},
            "sqlalchemy": {"sqlite_engine_url": "sqlite:///test.db"},
            "logging": {"level": "INFO"},
            "audit_log": {"backend": "sqlite", "sqlite_path": temp_db_path},
        }

        fd, config_path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        # Monkeypatch the config path
        from src.config import configuration

        original_get_project_root = configuration._get_project_root

        def mock_get_project_root():
            return Path(config_path).parent

        monkeypatch.setattr(configuration, "_get_project_root", mock_get_project_root)

        # Also patch the config.yaml filename
        original_load = configuration._load_yaml_config

        def mock_load():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

        monkeypatch.setattr(configuration, "_load_yaml_config", mock_load)

        # Reset the config singleton
        configuration._config = None

        yield temp_db_path

        # Cleanup
        configuration._config = None
        if os.path.exists(config_path):
            os.remove(config_path)

    @pytest.fixture
    def rag_log_service(self, sqlite_config):
        """Create a RagLogService with SQLite backend."""
        from src.services.rag_log_service import RagLogService

        service = RagLogService()
        yield service
        # Cleanup handled by context manager or manual close

    def test_service_initialization_sqlite(self, rag_log_service):
        """Test that service initializes with SQLite backend."""
        assert rag_log_service._backend == "sqlite"
        assert rag_log_service._sqlite_client is not None
        assert rag_log_service._cosmosdb_client is None
        assert not rag_log_service.is_async

        print("RagLogService initialized with SQLite backend")
        print(f"  Backend: {rag_log_service._backend}")
        print(f"  Is async: {rag_log_service.is_async}")

    @pytest.mark.asyncio
    async def test_store_answer_sqlite(self, rag_log_service):
        """Test storing a RAG log in SQLite."""
        test_email = f"test-{uuid.uuid4().hex[:8]}@example.com"
        rag_log = RagLog(
            user_email=test_email,
            user_question="What is the price?",
            agents_search_results="[{'product': 'Test', 'price': 10}]",
            final_answer="The price is $10.",
        )

        result = await rag_log_service.store_answer(rag_log)

        # SQLite returns None
        assert result is None

        print(f"Stored RAG log for user: {test_email}")

    @pytest.mark.asyncio
    async def test_get_logs_by_user_sqlite(self, rag_log_service):
        """Test retrieving logs for a user from SQLite."""
        test_email = f"query-{uuid.uuid4().hex[:8]}@example.com"

        # Store multiple logs
        for i in range(3):
            await rag_log_service.store_answer(
                RagLog(
                    user_email=test_email,
                    user_question=f"Question {i}",
                    agents_search_results=f"Results {i}",
                    final_answer=f"Answer {i}",
                )
            )

        # Retrieve logs
        logs = await rag_log_service.get_logs_by_user(test_email)

        assert len(logs) == 3
        questions = [log["user_question"] for log in logs]
        assert all(f"Question {i}" in questions for i in range(3))

        # Verify all fields are present
        for log in logs:
            assert "id" in log
            assert "user_email" in log
            assert "user_question" in log
            assert "agents_search_results" in log
            assert "final_answer" in log
            assert "timestamp" in log

        print(f"Retrieved {len(logs)} logs for user {test_email}")
        for log in logs:
            print(f"  - {log['user_question']}: {log['final_answer']}")

    @pytest.mark.asyncio
    async def test_logs_ordered_by_timestamp_desc(self, rag_log_service):
        """Test that logs are returned in descending timestamp order."""
        import time

        test_email = f"order-{uuid.uuid4().hex[:8]}@example.com"

        # Store logs with slight delay to ensure different timestamps
        for i in range(3):
            await rag_log_service.store_answer(
                RagLog(
                    user_email=test_email,
                    user_question=f"Question {i}",
                    agents_search_results="",
                    final_answer=f"Answer {i}",
                )
            )
            time.sleep(0.01)  # Small delay to ensure unique timestamps

        logs = await rag_log_service.get_logs_by_user(test_email)

        # Most recent should be first (Question 2)
        assert logs[0]["user_question"] == "Question 2"
        assert logs[2]["user_question"] == "Question 0"

        print("Logs correctly ordered by timestamp (newest first)")

    @pytest.mark.asyncio
    async def test_get_logs_empty_user(self, rag_log_service):
        """Test retrieving logs for a user with no logs."""
        logs = await rag_log_service.get_logs_by_user("nonexistent@example.com")

        assert logs == []

        print("Empty user correctly returns empty list")

    @pytest.mark.asyncio
    async def test_context_manager_sqlite(self, sqlite_config):
        """Test service works as async context manager with SQLite."""
        from src.services.rag_log_service import RagLogService

        test_email = f"context-{uuid.uuid4().hex[:8]}@example.com"

        async with RagLogService() as service:
            await service.store_answer(
                RagLog(
                    user_email=test_email,
                    user_question="Context test",
                    agents_search_results="",
                    final_answer="Test answer",
                )
            )

            logs = await service.get_logs_by_user(test_email)
            assert len(logs) == 1

        print("Context manager works correctly with SQLite")

    @pytest.mark.asyncio
    async def test_large_content_storage(self, rag_log_service):
        """Test storing logs with large content."""
        test_email = f"large-{uuid.uuid4().hex[:8]}@example.com"
        large_results = str([{"product": f"Item {i}", "data": "x" * 1000} for i in range(50)])

        await rag_log_service.store_answer(
            RagLog(
                user_email=test_email,
                user_question="Large content test",
                agents_search_results=large_results,
                final_answer="Processed large content",
            )
        )

        logs = await rag_log_service.get_logs_by_user(test_email)
        assert len(logs) == 1
        assert len(logs[0]["agents_search_results"]) > 50000

        print(f"Stored log with {len(large_results)} bytes of content")


class TestBackendToggle:
    """Test backend toggle functionality."""

    def test_invalid_backend_raises_error(self, monkeypatch):
        """Test that invalid backend raises ValueError."""
        import yaml
        import tempfile
        from pathlib import Path

        # Set up minimal environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AI_SEARCH_KEY", "test-key")
        monkeypatch.setenv("DOCUMENT_INTELLIGENCE_KEY", "test-key")

        config_content = {
            "openai": {"model": "gpt-4o", "embedding_model": {"name": "test", "dimensions": 1536}},
            "ai_search": {"index_name": "test", "pdf_index_name": "test", "endpoint": "https://test.search.windows.net"},
            "document_intelligence": {"endpoint": "https://test.cognitiveservices.azure.com/", "model_id": "test"},
            "database": {"path": "test.db", "document_tracking_path": "test.db"},
            "sqlalchemy": {"sqlite_engine_url": "sqlite:///test.db"},
            "logging": {"level": "INFO"},
            "audit_log": {"backend": "invalid_backend", "sqlite_path": "test.db"},
        }

        fd, config_path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        from src.config import configuration

        def mock_load():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

        monkeypatch.setattr(configuration, "_load_yaml_config", mock_load)
        configuration._config = None

        try:
            from src.services.rag_log_service import RagLogService

            with pytest.raises(ValueError, match="Unknown audit_log backend"):
                RagLogService()

            print("Invalid backend correctly raises ValueError")
        finally:
            configuration._config = None
            if os.path.exists(config_path):
                os.remove(config_path)

    def test_cosmosdb_without_credentials_raises(self, monkeypatch):
        """Test that CosmosDB backend without credentials raises error."""
        import yaml
        import tempfile

        # Set up minimal environment variables (without CosmosDB)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AI_SEARCH_KEY", "test-key")
        monkeypatch.setenv("DOCUMENT_INTELLIGENCE_KEY", "test-key")
        # Explicitly unset CosmosDB vars
        monkeypatch.delenv("COSMOSDB_ENDPOINT", raising=False)
        monkeypatch.delenv("COSMOSDB_KEY", raising=False)

        config_content = {
            "openai": {"model": "gpt-4o", "embedding_model": {"name": "test", "dimensions": 1536}},
            "ai_search": {"index_name": "test", "pdf_index_name": "test", "endpoint": "https://test.search.windows.net"},
            "document_intelligence": {"endpoint": "https://test.cognitiveservices.azure.com/", "model_id": "test"},
            "database": {"path": "test.db", "document_tracking_path": "test.db"},
            "sqlalchemy": {"sqlite_engine_url": "sqlite:///test.db"},
            "logging": {"level": "INFO"},
            "audit_log": {"backend": "cosmosdb", "sqlite_path": "test.db"},
        }

        fd, config_path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        from src.config import configuration
        from src.config.configuration import ConfigurationError

        def mock_load():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

        monkeypatch.setattr(configuration, "_load_yaml_config", mock_load)
        configuration._config = None

        try:
            # Should raise ConfigurationError when trying to load config
            # because COSMOSDB_ENDPOINT is required when backend is cosmosdb
            with pytest.raises(ConfigurationError):
                from src.config.configuration import load_config

                load_config()

            print("CosmosDB backend without credentials correctly raises ConfigurationError")
        finally:
            configuration._config = None
            if os.path.exists(config_path):
                os.remove(config_path)
