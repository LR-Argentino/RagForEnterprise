import asyncio
from chat import RAGChat

async def main():
    result = await RAGChat("<no chat history>", "Was kannst du mir über das Produkt Haty Pants sagen?", "luca.argentino@icloud.com")
    print(result)

asyncio.run(main())