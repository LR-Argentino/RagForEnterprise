from ..evaluation import send_to_openai

TRIAGE_PROMPT = """You are a helpful assistant responsible for categorizing user questions.

You have access to TWO data sources:
1. **Products database** - Product information, specifications, descriptions
2. **Documents database** - Receipts, invoices, tickets, bills, scanned documents

## Classification Rules

**Return *PRODUCT if the user asks about:**
- Product features, specifications, descriptions
- Product availability, pricing, comparisons
- General product questions
- Example: "What are the features of Product X?" → *PRODUCT
- Example: "Tell me about the new smartphone" → *PRODUCT

**Return *DOCUMENT if the user asks about:**
- Receipts, invoices, bills
- Tickets (train, flight, concert, parking, etc.)
- Personal documents, scanned papers
- Amounts, prices, dates from documents
- Example: "Wie viel hat mein Zugticket gekostet?" → *DOCUMENT
- Example: "Show me my restaurant receipt" → *DOCUMENT
- Example: "What was the total on my invoice?" → *DOCUMENT
- Example: "Find my parking ticket" → *DOCUMENT

**Return *CLARIFY if:**
- You cannot determine which category the question belongs to
- The question is ambiguous
- Include a clarification request in brackets

## Response Format

Return ONLY the category marker:
- *PRODUCT
- *DOCUMENT
- *CLARIFY [Your clarification request]

User Question: {0}

Chat history: {1}
"""


def triage(user_question: str, chat_history: str = "") -> tuple[str, str]:
    """
    Categorize user question into PRODUCT, DOCUMENT, or CLARIFY.

    Args:
        user_question: The user's question to categorize.
        chat_history: Previous conversation context.

    Returns:
        Tuple of (category, raw_result) where category is one of:
        - "PRODUCT" - Query product-search-index
        - "DOCUMENT" - Query pdf-document-index
        - "CLARIFY" - Need more information from user
    """
    formatted_triage_prompt = TRIAGE_PROMPT.format(user_question, chat_history)

    result = send_to_openai(formatted_triage_prompt)

    if "*PRODUCT" in result:
        return "PRODUCT", result.strip()

    elif "*DOCUMENT" in result:
        return "DOCUMENT", result.strip()

    elif "*CLARIFY" in result:
        result = result.replace("[", "")
        result = result.replace("]", "")
        result = result.replace("*CLARIFY", "")
        return "CLARIFY", result.strip()

    return "CLARIFY", "Ich konnte deine Anfrage nicht zuordnen. Suchst du nach einem Produkt oder einem Dokument (Rechnung/Ticket/Beleg)?"

