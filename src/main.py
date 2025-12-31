import asyncio
from chat import RAGChat_streaming

async def main_streaming():
    """Streaming version - prints chunks as they arrive."""
    print("Streaming response:\n")
    async for chunk in RAGChat_streaming(
        "<no chat history>",
        "Katzen, Katzen ich liebe Katzen. Was ist das Produkt Pants?",
        "luca.argentino@icloud.com"
    ):
        print(chunk, end="", flush=True)
    print("\n")  # Final newline


if __name__ == "__main__":
    # Use streaming version for real-time output
    asyncio.run(main_streaming())

    # Or use non-streaming version:
    # asyncio.run(main())