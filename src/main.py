import asyncio
from src.rag.chat import RAGChat_streaming

USER_EMAIL = "luca.argentino@icloud.com"


async def interactive_chat():
    """Interactive chat loop with history tracking."""
    chat_history = []

    print("RAG Chat gestartet. Tippe 'exit' zum Beenden.\n")

    while True:
        user_input = input("Du: ").strip()

        if user_input.lower() in ("exit", "quit", "q"):
            print("Auf Wiedersehen!")
            break

        if not user_input:
            continue

        # Build history string from previous exchanges
        history_str = "\n".join(chat_history) if chat_history else "<no chat history>"

        # Collect response
        response_chunks = []
        print("Assistent: ", end="", flush=True)

        async for chunk in RAGChat_streaming(history_str, user_input, USER_EMAIL):
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)

        print("\n")

        # Add exchange to history
        full_response = "".join(response_chunks)
        chat_history.append(f"User: {user_input}")
        chat_history.append(f"Assistant: {full_response}")


if __name__ == "__main__":
    asyncio.run(interactive_chat())
