import openai
import os
from dotenv import load_dotenv
from typing import Any, Iterable, cast

load_dotenv()

evals = [
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


def send_to_openai(message):
  openai.api_key = os.environ.get("OPENAI_API_KEY")
  completion = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": message}]
    )
  return completion.choices[0].message.content.strip()

def evaluate_generated_answer(expected_answer,generated_answer):
    prompt = f"""Please evaluate the generated answer. 
    If the generated answer provides the same information as the expected answer, 
    then return PASS. Otherwise, return FAIL. 
    Expected answer: {expected_answer} Generated answer: {generated_answer}"""
    response = send_to_openai(prompt)
    return response

if __name__ == "__main__":
    # Simple smoke test: query the API for each eval and print the answer.
    # Note: this will call the real OpenAI API and requires OPENAI_API_KEY to be set.
    for i, q in enumerate(evals):
        print(f"\nQuestion {i+1}: {q}")
        try:
            resp = send_to_openai(q)
            print("Response (first 400 chars):\n", resp[:400])
        except Exception as e:
            print("Error calling OpenAI API:", e)