"""Azure Cosmos DB client for audit log storage."""

import uuid
from typing import Any, Optional

from azure.cosmos.aio import CosmosClient
from azure.cosmos.aio._container import ContainerProxy
from azure.cosmos.aio._database import DatabaseProxy
from azure.cosmos.exceptions import CosmosResourceNotFoundError


class CosmosDBClient:
    """Async Cosmos DB client with connection management.

    Uses the NoSQL API for storing and querying audit logs.
    Supports async context manager pattern for proper resource cleanup.
    """

    def __init__(
        self,
        endpoint: str,
        key: str,
        database_name: str,
        container_name: str,
        partition_key_path: str = "/user_email",
    ):
        """Initialize the Cosmos DB client.

        Args:
            endpoint: Cosmos DB account endpoint URL
            key: Cosmos DB account key
            database_name: Name of the database to use
            container_name: Name of the container to use
            partition_key_path: Path to the partition key field (default: /user_email)
        """
        self._endpoint = endpoint
        self._key = key
        self._database_name = database_name
        self._container_name = container_name
        self._partition_key_path = partition_key_path

        self._client: Optional[CosmosClient] = None
        self._database: Optional[DatabaseProxy] = None
        self._container: Optional[ContainerProxy] = None

    async def connect(self) -> None:
        """Establish connection and ensure database/container exist."""
        self._client = CosmosClient(url=self._endpoint, credential=self._key)
        await self._client.__aenter__()

        # Get or create database
        try:
            self._database = self._client.get_database_client(self._database_name)
            # Verify database exists by reading it
            await self._database.read()
        except CosmosResourceNotFoundError:
            self._database = await self._client.create_database(self._database_name)

        # Get or create container
        try:
            self._container = self._database.get_container_client(self._container_name)
            # Verify container exists by reading it
            await self._container.read()
        except CosmosResourceNotFoundError:
            self._container = await self._database.create_container(
                id=self._container_name,
                partition_key={"paths": [self._partition_key_path], "kind": "Hash"},
            )

    async def close(self) -> None:
        """Close the Cosmos DB connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._database = None
            self._container = None

    async def __aenter__(self) -> "CosmosDBClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit with cleanup."""
        await self.close()
        return False

    async def upsert_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Insert or update an item in the container.

        Args:
            item: Dictionary containing the item data. Must include 'id' field
                  or one will be generated. Must include the partition key field.

        Returns:
            The upserted item with any system-generated fields.

        Raises:
            RuntimeError: If client is not connected.
        """
        if self._container is None:
            raise RuntimeError("CosmosDB client not connected. Call connect() first.")

        # Ensure item has an id
        if "id" not in item:
            item["id"] = str(uuid.uuid4())

        result = await self._container.upsert_item(body=item)
        return dict(result)

    async def query_items(
        self,
        query: str,
        parameters: Optional[list[dict[str, Any]]] = None,
        partition_key: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Query items from the container.

        Args:
            query: SQL query string
            parameters: Optional query parameters as list of {"name": "@param", "value": value}
            partition_key: Optional partition key to scope the query

        Returns:
            List of matching items.

        Raises:
            RuntimeError: If client is not connected.
        """
        if self._container is None:
            raise RuntimeError("CosmosDB client not connected. Call connect() first.")

        query_options = {}
        if partition_key is not None:
            query_options["partition_key"] = partition_key

        items = []
        async for item in self._container.query_items(
            query=query,
            parameters=parameters,
            **query_options,
        ):
            items.append(dict(item))

        return items

    async def read_item(self, item_id: str, partition_key: str) -> dict[str, Any]:
        """Read a single item by id and partition key.

        Args:
            item_id: The item's id
            partition_key: The partition key value

        Returns:
            The item data.

        Raises:
            RuntimeError: If client is not connected.
            CosmosResourceNotFoundError: If item not found.
        """
        if self._container is None:
            raise RuntimeError("CosmosDB client not connected. Call connect() first.")

        result = await self._container.read_item(item=item_id, partition_key=partition_key)
        return dict(result)

    async def delete_item(self, item_id: str, partition_key: str) -> None:
        """Delete an item by id and partition key.

        Args:
            item_id: The item's id
            partition_key: The partition key value

        Raises:
            RuntimeError: If client is not connected.
            CosmosResourceNotFoundError: If item not found.
        """
        if self._container is None:
            raise RuntimeError("CosmosDB client not connected. Call connect() first.")

        await self._container.delete_item(item=item_id, partition_key=partition_key)
