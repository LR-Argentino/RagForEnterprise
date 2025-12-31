import sqlite3
from sqlite3 import Connection

class SqliteClient:
    """SQLite database client with connection management."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._connection = sqlite3.connect(self.connection_string)

    @property
    def connection(self) -> Connection:
        """Get the database connection."""
        return self._connection

    def execute_query(self, query: str, params=None):
        """Execute a query and return all results."""
        cursor = self._connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        # Commit for write operations (INSERT, UPDATE, DELETE)
        if query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            self._connection.commit()

        results = cursor.fetchall()
        cursor.close()
        return results

    def close(self):
        """Close the database connection."""
        self._connection.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
        return False