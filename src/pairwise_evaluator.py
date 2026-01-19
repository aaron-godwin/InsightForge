# pairwise_evaluator.py

from typing import Dict
from langchain_groq import ChatGroq


PAIRWISE_PROMPT = """
You are an expert evaluator. Compare two answers to the same question.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Evaluate which answer is better based on:
- correctness
- completeness
- clarity
- reasoning quality
- grounding in the question

Respond ONLY in JSON with the following fields:
{{
  "winner": "A" or "B",
  "confidence": float between 0 and 1,
  "justification": "short explanation"
}}
"""


class PairwiseEvaluator:
    """
    LLM-based pairwise evaluator for InsightForge using Groq + Llama 3.1 8B.
    """

    def __init__(self, llm=None):
        # Allow dependency injection for testing
        self.llm = llm or ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

    def compare(self, question: str, answer_a: str, answer_b: str) -> Dict:
        """
        Compare two answers to the same question.

        Returns:
            {
                "winner": "A" or "B",
                "confidence": float,
                "justification": str
            }
        """

        prompt = PAIRWISE_PROMPT.format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b
        )

        response = self.llm.invoke(prompt)
        content = response.content.strip()

        import json
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {
                "winner": "A",
                "confidence": 0.0,
                "justification": "Evaluator returned malformed JSON."
            }

        return result