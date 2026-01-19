# evaluation.py

from typing import List, Dict

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


class InsightEvaluator:
    """
    RAGAS-based evaluation for InsightForge.

    Expects data in the form:
    - predictions: list of dicts with:
        {
            "query": str,
            "answer": str,
            "contexts": List[str]  # RAG + memory + stats context as text chunks
        }

    - references: list of dicts with:
        {
            "query": str,
            "answer": str  # ground-truth / expected answer
        }
    """

    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def _build_dataset(
        self,
        predictions: List[Dict[str, str]],
        references: List[Dict[str, str]],
    ) -> Dataset:
        """
        Build a HuggingFace Dataset in the format RAGAS expects.

        RAGAS expects columns:
        - "question"
        - "answer"
        - "contexts"
        - "ground_truth"
        """

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for pred, ref in zip(predictions, references):
            questions.append(pred["query"])
            answers.append(pred["answer"])
            contexts.append(pred.get("contexts", []))
            ground_truths.append(ref["answer"])

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

        return Dataset.from_dict(data)

    def evaluate(
        self,
        predictions: List[Dict[str, str]],
        references: List[Dict[str, str]],
    ):
        """
        Run RAGAS evaluation over a batch of predictions and references.

        Returns:
            A dict-like object with metric scores and per-sample details.
        """

        dataset = self._build_dataset(predictions, references)

        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
        )

        return results