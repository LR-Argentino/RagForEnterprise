"""Service for storing RAG interaction logs.

Supports two backends:
- SQLite: Local, free, for development
- CosmosDB: Azure cloud, paid, for production

Backend is configured via config.yaml: audit_log.backend
"""

import datetime
from typing import Optional, Union

from ..clients.cosmosdb_client import CosmosDBClient
from ..clients.sqlite_client import SqliteClient
from ..config.configuration import get_config
from ..models import RagLog


class RagLogService:
    """Service for storing and retrieving RAG interaction logs.

    Supports both SQLite (sync) and CosmosDB (async) backends.
    Backend selection is based on config.audit_log.backend.
    """

    def __init__(self):
        """Initialize the RAG log service based on configured backend."""
        config = get_config()
        self._backend = config.audit_log.backend

        self._sqlite_client: Optional[SqliteClient] = None
        self._cosmosdb_client: Optional[CosmosDBClient] = None
        self._connected = False

        if self._backend == "sqlite":
            self._sqlite_client = SqliteClient(config.audit_log.sqlite_path)
            self._init_sqlite_table()
        elif self._backend == "cosmosdb":
            if config.cosmosdb is None:
                raise ValueError(
                    "CosmosDB backend selected but cosmosdb config is missing. "
                    "Set COSMOSDB_ENDPOINT and COSMOSDB_KEY environment variables."
                )
            self._cosmosdb_client = CosmosDBClient(
                endpoint=config.cosmosdb.endpoint,
                key=config.cosmosdb.key,
                database_name=config.cosmosdb.database_name,
                container_name=config.cosmosdb.container_name,
                partition_key_path=config.cosmosdb.partition_key_path,
            )
        else:
            raise ValueError(f"Unknown audit_log backend: {self._backend}. Use 'sqlite' or 'cosmosdb'.")

    def _init_sqlite_table(self) -> None:
        """Create the raglogs table if it doesn't exist."""
        if self._sqlite_client is None:
            return

        self._sqlite_client.execute_query("""
            CREATE TABLE IF NOT EXISTS raglogs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                user_question TEXT NOT NULL,
                agents_search_results TEXT,
                final_answer TEXT,
                timestamp TEXT NOT NULL
            )
        """)

    @property
    def is_async(self) -> bool:
        """Check if this service uses async operations."""
        return self._backend == "cosmosdb"

    async def _ensure_cosmosdb_connected(self) -> None:
        """Ensure the Cosmos DB client is connected."""
        if not self._connected and self._cosmosdb_client is not None:
            await self._cosmosdb_client.connect()
            self._connected = True

    async def store_answer(self, rag_log: RagLog) -> Union[dict, None]:
        """Store a RAG interaction log.

        Args:
            rag_log: The RagLog dataclass containing interaction details.

        Returns:
            For CosmosDB: The stored document with metadata.
            For SQLite: None.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if self._backend == "sqlite":
            return self._store_answer_sqlite(rag_log, timestamp)
        else:
            return await self._store_answer_cosmosdb(rag_log, timestamp)

    def _store_answer_sqlite(self, rag_log: RagLog, timestamp: str) -> None:
        """Store log in SQLite."""
        if self._sqlite_client is None:
            raise RuntimeError("SQLite client not initialized")

        self._sqlite_client.execute_query(
            """INSERT INTO raglogs
               (user_email, user_question, agents_search_results, final_answer, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (
                rag_log.user_email,
                rag_log.user_question,
                rag_log.agents_search_results,
                rag_log.final_answer,
                timestamp,
            ),
        )

    async def _store_answer_cosmosdb(self, rag_log: RagLog, timestamp: str) -> dict:
        """Store log in CosmosDB."""
        await self._ensure_cosmosdb_connected()

        if self._cosmosdb_client is None:
            raise RuntimeError("CosmosDB client not initialized")

        log_document = {
            "user_email": rag_log.user_email,
            "user_question": rag_log.user_question,
            "agents_search_results": rag_log.agents_search_results,
            "final_answer": rag_log.final_answer,
            "timestamp": timestamp,
        }

        return await self._cosmosdb_client.upsert_item(log_document)

    async def get_logs_by_user(self, user_email: str) -> list[dict]:
        """Retrieve all logs for a specific user.

        Args:
            user_email: The user's email address.

        Returns:
            List of log documents for the user.
        """
        if self._backend == "sqlite":
            return self._get_logs_by_user_sqlite(user_email)
        else:
            return await self._get_logs_by_user_cosmosdb(user_email)

    def _get_logs_by_user_sqlite(self, user_email: str) -> list[dict]:
        """Get logs from SQLite."""
        if self._sqlite_client is None:
            raise RuntimeError("SQLite client not initialized")

        rows = self._sqlite_client.execute_query(
            """SELECT id, user_email, user_question, agents_search_results,
                      final_answer, timestamp
               FROM raglogs
               WHERE user_email = ?
               ORDER BY timestamp DESC""",
            (user_email,),
        )

        return [
            {
                "id": row[0],
                "user_email": row[1],
                "user_question": row[2],
                "agents_search_results": row[3],
                "final_answer": row[4],
                "timestamp": row[5],
            }
            for row in rows
        ]

    async def _get_logs_by_user_cosmosdb(self, user_email: str) -> list[dict]:
        """Get logs from CosmosDB."""
        await self._ensure_cosmosdb_connected()

        if self._cosmosdb_client is None:
            raise RuntimeError("CosmosDB client not initialized")

        query = "SELECT * FROM c WHERE c.user_email = @user_email ORDER BY c.timestamp DESC"
        parameters = [{"name": "@user_email", "value": user_email}]

        return await self._cosmosdb_client.query_items(
            query=query,
            parameters=parameters,
            partition_key=user_email,
        )

    async def close(self) -> None:
        """Close database connections."""
        if self._backend == "sqlite" and self._sqlite_client:
            self._sqlite_client.close()
        elif self._backend == "cosmosdb" and self._cosmosdb_client and self._connected:
            await self._cosmosdb_client.close()
            self._connected = False

    async def __aenter__(self) -> "RagLogService":
        """Async context manager entry."""
        if self._backend == "cosmosdb":
            await self._ensure_cosmosdb_connected()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit."""
        await self.close()
        return False
