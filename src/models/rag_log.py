from dataclasses import dataclass

@dataclass
class RagLog:
    user_email: str
    user_question: str
    agents_search_results: str
    final_answer: str
