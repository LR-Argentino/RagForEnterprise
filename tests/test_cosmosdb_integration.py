"""Integration tests for Cosmos DB client and RAG log service.

These tests require actual Cosmos DB credentials and connectivity.
They verify:
- CosmosDBClient connection and CRUD operations
- RagLogService integration with Cosmos DB
- Database and container auto-creation
- Query functionality with partition keys
"""

import os
import uuid
from datetime import datetime, timezone

import pytest

from src.clients.cosmosdb_client import CosmosDBClient
from src.config.configuration import get_config, ConfigurationError
from src.models import RagLog
from src.services.rag_log_service import RagLogService


def cosmos_credentials_available() -> bool:
    """Check if Cosmos DB credentials are available."""
    try:
        config = get_config()
        return bool(config.cosmosdb.endpoint and config.cosmosdb.key)
    except ConfigurationError:
        return False


# Skip all tests if credentials not available
pytestmark = pytest.mark.skipif(
    not cosmos_credentials_available(),
    reason="Cosmos DB credentials not configured (COSMOSDB_ENDPOINT, COSMOSDB_KEY)",
)


class TestCosmosDBClient:
    """Test CosmosDBClient CRUD operations."""

    @pytest.fixture
    def test_container_name(self):
        """Generate unique container name for test isolation."""
        return f"test-container-{uuid.uuid4().hex[:8]}"

    @pytest.fixture
    async def cosmos_client(self, test_container_name):
        """Create a CosmosDBClient for testing."""
        config = get_config()
        client = CosmosDBClient(
            endpoint=config.cosmosdb.endpoint,
            key=config.cosmosdb.key,
            database_name=config.cosmosdb.database_name,
            container_name=test_container_name,
            partition_key_path="/test_partition",
        )
        await client.connect()
        yield client
        # Cleanup: delete test container
        try:
            if client._container:
                await client._database.delete_container(test_container_name)
        except Exception as e:
            print(f"Cleanup warning: {e}")
        await client.close()

    @pytest.mark.asyncio
    async def test_client_connection(self, cosmos_client):
        """Test that client connects successfully and creates container."""
        assert cosmos_client._client is not None
        assert cosmos_client._database is not None
        assert cosmos_client._container is not None

        print("Cosmos DB client connected successfully")
        print(f"  Database: {cosmos_client._database_name}")
        print(f"  Container: {cosmos_client._container_name}")

    @pytest.mark.asyncio
    async def test_upsert_item_creates_new(self, cosmos_client):
        """Test upserting a new item."""
        test_item = {
            "test_partition": "user@test.com",
            "data": "test value",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        result = await cosmos_client.upsert_item(test_item)

        assert "id" in result
        assert result["test_partition"] == "user@test.com"
        assert result["data"] == "test value"
        assert "_ts" in result  # Cosmos DB timestamp
        assert "_etag" in result  # Cosmos DB etag

        print(f"Created item with id: {result['id']}")
        print(f"  Partition: {result['test_partition']}")
        print(f"  System timestamp: {result['_ts']}")

    @pytest.mark.asyncio
    async def test_upsert_item_with_custom_id(self, cosmos_client):
        """Test upserting with custom ID."""
        custom_id = f"custom-{uuid.uuid4().hex[:8]}"
        test_item = {
            "id": custom_id,
            "test_partition": "user@test.com",
            "data": "custom id test",
        }

        result = await cosmos_client.upsert_item(test_item)

        assert result["id"] == custom_id
        print(f"Created item with custom id: {result['id']}")

    @pytest.mark.asyncio
    async def test_upsert_item_updates_existing(self, cosmos_client):
        """Test upserting updates an existing item."""
        item_id = f"update-test-{uuid.uuid4().hex[:8]}"
        original_item = {
            "id": item_id,
            "test_partition": "user@test.com",
            "version": 1,
        }

        # Create
        await cosmos_client.upsert_item(original_item)

        # Update
        updated_item = {
            "id": item_id,
            "test_partition": "user@test.com",
            "version": 2,
        }
        result = await cosmos_client.upsert_item(updated_item)

        assert result["id"] == item_id
        assert result["version"] == 2

        print(f"Updated item {item_id} from version 1 to 2")

    @pytest.mark.asyncio
    async def test_read_item(self, cosmos_client):
        """Test reading a single item."""
        item_id = f"read-test-{uuid.uuid4().hex[:8]}"
        partition_key = "reader@test.com"
        test_item = {
            "id": item_id,
            "test_partition": partition_key,
            "data": "read test",
        }

        await cosmos_client.upsert_item(test_item)
        result = await cosmos_client.read_item(item_id, partition_key)

        assert result["id"] == item_id
        assert result["data"] == "read test"

        print(f"Read item: {result['id']}")

    @pytest.mark.asyncio
    async def test_query_items(self, cosmos_client):
        """Test querying items."""
        partition_key = f"query-user-{uuid.uuid4().hex[:8]}@test.com"

        # Create multiple items
        for i in range(3):
            await cosmos_client.upsert_item({
                "test_partition": partition_key,
                "sequence": i,
            })

        # Query with partition key
        query = "SELECT * FROM c WHERE c.test_partition = @partition"
        parameters = [{"name": "@partition", "value": partition_key}]

        results = await cosmos_client.query_items(
            query=query,
            parameters=parameters,
            partition_key=partition_key,
        )

        assert len(results) == 3
        sequences = [r["sequence"] for r in results]
        assert set(sequences) == {0, 1, 2}

        print(f"Queried {len(results)} items for partition {partition_key}")

    @pytest.mark.asyncio
    async def test_delete_item(self, cosmos_client):
        """Test deleting an item."""
        item_id = f"delete-test-{uuid.uuid4().hex[:8]}"
        partition_key = "delete@test.com"
        test_item = {
            "id": item_id,
            "test_partition": partition_key,
        }

        await cosmos_client.upsert_item(test_item)
        await cosmos_client.delete_item(item_id, partition_key)

        # Verify deleted
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        with pytest.raises(CosmosResourceNotFoundError):
            await cosmos_client.read_item(item_id, partition_key)

        print(f"Deleted item: {item_id}")

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client works as async context manager."""
        config = get_config()
        container_name = f"context-test-{uuid.uuid4().hex[:8]}"

        async with CosmosDBClient(
            endpoint=config.cosmosdb.endpoint,
            key=config.cosmosdb.key,
            database_name=config.cosmosdb.database_name,
            container_name=container_name,
            partition_key_path="/pk",
        ) as client:
            result = await client.upsert_item({"pk": "test", "data": "context manager test"})
            assert "id" in result

            # Cleanup
            await client._database.delete_container(container_name)

        print("Context manager works correctly")


class TestRagLogService:
    """Test RagLogService with Cosmos DB backend."""

    @pytest.fixture
    async def rag_log_service(self):
        """Create a RagLogService for testing."""
        service = RagLogService()
        yield service
        await service.close()

    @pytest.mark.asyncio
    async def test_store_answer(self, rag_log_service):
        """Test storing a RAG interaction log."""
        test_email = f"test-{uuid.uuid4().hex[:8]}@example.com"
        rag_log = RagLog(
            user_email=test_email,
            user_question="What is the price of Widget X?",
            agents_search_results="[{'product': 'Widget X', 'price': 29.99}]",
            final_answer="The price of Widget X is $29.99.",
        )

        result = await rag_log_service.store_answer(rag_log)

        assert "id" in result
        assert result["user_email"] == test_email
        assert result["user_question"] == "What is the price of Widget X?"
        assert "timestamp" in result

        print(f"Stored RAG log with id: {result['id']}")
        print(f"  User: {result['user_email']}")
        print(f"  Timestamp: {result['timestamp']}")

    @pytest.mark.asyncio
    async def test_get_logs_by_user(self, rag_log_service):
        """Test retrieving logs for a specific user."""
        test_email = f"query-test-{uuid.uuid4().hex[:8]}@example.com"

        # Store multiple logs
        for i in range(3):
            await rag_log_service.store_answer(RagLog(
                user_email=test_email,
                user_question=f"Question {i}",
                agents_search_results=f"Results {i}",
                final_answer=f"Answer {i}",
            ))

        # Retrieve logs
        logs = await rag_log_service.get_logs_by_user(test_email)

        assert len(logs) == 3
        questions = [log["user_question"] for log in logs]
        assert all(f"Question {i}" in questions for i in range(3))

        print(f"Retrieved {len(logs)} logs for user {test_email}")

    @pytest.mark.asyncio
    async def test_store_answer_with_long_content(self, rag_log_service):
        """Test storing logs with large search results."""
        test_email = f"long-content-{uuid.uuid4().hex[:8]}@example.com"
        long_results = str([{"product": f"Item {i}", "data": "x" * 1000} for i in range(10)])

        rag_log = RagLog(
            user_email=test_email,
            user_question="Search for products",
            agents_search_results=long_results,
            final_answer="Found 10 products matching your query.",
        )

        result = await rag_log_service.store_answer(rag_log)

        assert "id" in result
        assert len(result["agents_search_results"]) > 10000

        print(f"Stored log with {len(long_results)} bytes of search results")

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test service works as async context manager."""
        async with RagLogService() as service:
            test_email = f"context-{uuid.uuid4().hex[:8]}@example.com"
            result = await service.store_answer(RagLog(
                user_email=test_email,
                user_question="Context manager test",
                agents_search_results="[]",
                final_answer="Test answer",
            ))
            assert "id" in result

        print("RagLogService context manager works correctly")


class TestCosmosDBClientErrors:
    """Test error handling in CosmosDBClient."""

    @pytest.mark.asyncio
    async def test_upsert_without_connection_raises(self):
        """Test that operations without connection raise RuntimeError."""
        config = get_config()
        client = CosmosDBClient(
            endpoint=config.cosmosdb.endpoint,
            key=config.cosmosdb.key,
            database_name=config.cosmosdb.database_name,
            container_name="test-container",
        )
        # Don't call connect()

        with pytest.raises(RuntimeError, match="not connected"):
            await client.upsert_item({"test": "data"})

        print("Correctly raises RuntimeError when not connected")

    @pytest.mark.asyncio
    async def test_query_without_connection_raises(self):
        """Test that query without connection raises RuntimeError."""
        config = get_config()
        client = CosmosDBClient(
            endpoint=config.cosmosdb.endpoint,
            key=config.cosmosdb.key,
            database_name=config.cosmosdb.database_name,
            container_name="test-container",
        )

        with pytest.raises(RuntimeError, match="not connected"):
            await client.query_items("SELECT * FROM c")

        print("Correctly raises RuntimeError for query when not connected")

    @pytest.mark.asyncio
    async def test_read_nonexistent_item_raises(self):
        """Test that reading non-existent item raises CosmosResourceNotFoundError."""
        config = get_config()
        container_name = f"error-test-{uuid.uuid4().hex[:8]}"

        async with CosmosDBClient(
            endpoint=config.cosmosdb.endpoint,
            key=config.cosmosdb.key,
            database_name=config.cosmosdb.database_name,
            container_name=container_name,
            partition_key_path="/pk",
        ) as client:
            from azure.cosmos.exceptions import CosmosResourceNotFoundError
            with pytest.raises(CosmosResourceNotFoundError):
                await client.read_item("nonexistent-id", "nonexistent-partition")

            # Cleanup
            await client._database.delete_container(container_name)

        print("Correctly raises CosmosResourceNotFoundError for missing item")
