import datetime

from ..clients import SqliteClient
from ..models import RagLog


class RagLogService:
    _sqlite_client: SqliteClient

    def __init__(self):
        self._sqlite_client = SqliteClient("rag_logs.db")

    def store_answer(self, rag_log: RagLog):
        timestamp = str(datetime.datetime.now())

        new_log = (
            rag_log.user_email,
            rag_log.user_question,
            rag_log.agents_search_results,
            rag_log.final_answer,
            timestamp
        )
        self._sqlite_client.execute_query(
            "INSERT INTO raglogs (user_email, user_question, agents_search_results, final_answer, timestamp) VALUES (?,?,?,?,?)",
            new_log)
