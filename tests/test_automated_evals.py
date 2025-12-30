"""Automated evaluation tests for RAG system."""

from src.evaluation.run_evals import evaluate_generated_answer


def run_RAG(user_questions):
    """Placeholder for RAG function - to be implemented."""
    return "IDKOL"


def test_run_RAG():
    """Test RAG responses against expected answers using LLM evaluation."""
    eval_questions = [
        "Why was arsenic used to treat diseases?",
        "What oil is good for you to cook with?",
        "How to treat poison ivy?",
        "How long after a catheterization done via the wrist is it safe to resume light resistance training?"
    ]

    eval_answers = [
        "Arsenic was used to treat diseases due to its antimicrobial and anti-inflammatory properties. It was also used to treat syphilis, a sexually transmitted disease. However, arsenic is now known to be a toxic substance that can cause various health problems, including cancer, heart disease, and neurological disorders.",
        "Olive oil is considered one of the healthiest oils for cooking due to its high monounsaturated fatty acid content, which can help lower bad cholesterol (LDL) levels. Canola oil is also a good option as it is low in saturated fat and high in monounsaturated fat. Other healthy cooking oils include avocado oil, coconut oil, and grapeseed oil.",
        "Poison ivy can cause an allergic reaction characterized by itchy, red rash. To treat poison ivy, wash the affected area with soap and cool water as soon as possible, then apply over-the-counter corticosteroid cream or ointment to reduce inflammation. Avoid scratching the rash to prevent infection and use calamine lotion or antihistamines for relief from itching.",
        "After a catheterization done via the wrist, it is recommended to wait for at least 24 hours before resuming any form of physical activity. This is to prevent any complications such as bleeding or infection at the site of insertion. It is also important to monitor the site for any signs of swelling, redness, or discharge. If any of these symptoms occur, it is recommended to seek medical attention immediately."
    ]
    generated_answers = []

    for question in eval_questions:
        answer = run_RAG(question)
        generated_answers.append(answer)

    for i in range(len(eval_questions)):
        result = evaluate_generated_answer(eval_answers[i], generated_answers[i])
        print(f"Question {i+1}: {eval_questions[i]}")
        print(f"Expected: {eval_answers[i][:100]}...")
        print(f"Generated: {generated_answers[i]}")
        print(f"Evaluation result: {result}")
        assert "PASS" in result
