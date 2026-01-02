"""WebSocket controller for RAG chat functionality."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from src.rag.chat import RAGChat_streaming

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    """Incoming chat message from client."""

    question: str
    user_email: str
    chat_history: Optional[str] = ""


class ChatResponse(BaseModel):
    """Outgoing chat response chunk."""

    type: str  # "chunk", "done", "error"
    content: str


@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for streaming RAG chat.

    Protocol:
    1. Client connects to /chat/ws
    2. Client sends JSON: {"question": "...", "user_email": "...", "chat_history": "..."}
    3. Server streams JSON responses: {"type": "chunk", "content": "..."}
    4. Server sends final: {"type": "done", "content": ""}
    5. On error: {"type": "error", "content": "error message"}

    Client can send multiple questions on same connection.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        while True:
            raw_message = await websocket.receive_text()

            try:
                message = ChatMessage.model_validate_json(raw_message)
            except ValidationError as e:
                await websocket.send_json(
                    ChatResponse(type="error", content=f"Invalid message format: {e}").model_dump()
                )
                continue

            logger.info(f"Processing question from {message.user_email}: {message.question[:50]}...")

            try:
                async for chunk in RAGChat_streaming(
                    chat_history=message.chat_history,
                    user_question=message.question,
                    user_email=message.user_email,
                ):
                    await websocket.send_json(
                        ChatResponse(type="chunk", content=chunk).model_dump()
                    )

                await websocket.send_json(
                    ChatResponse(type="done", content="").model_dump()
                )

            except Exception as e:
                logger.exception(f"Error processing question: {e}")
                await websocket.send_json(
                    ChatResponse(type="error", content=str(e)).model_dump()
                )

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
