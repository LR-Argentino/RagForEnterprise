"""FastAPI application setup."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.controller import chat_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RAG Chat API",
        description="WebSocket API for RAG-based chat",
        version="1.0.0",
    )

    # CORS for Angular frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production: specify Angular origin
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat_router)

    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


app = create_app()
