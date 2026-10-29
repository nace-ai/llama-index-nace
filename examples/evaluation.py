"""Judge a RAG answer with Drex: faithfulness, relevancy and correctness as probabilities."""

from llama_index.evaluation.nace import (
    NaceCorrectnessEvaluator,
    NaceFaithfulnessEvaluator,
    NaceRelevancyEvaluator,
)

QUERY = "When was the Eiffel Tower completed?"
CONTEXTS = ["The Eiffel Tower, in Paris, was completed in March 1889 for the World's Fair."]
GOOD = "The Eiffel Tower was completed in 1889."
BAD = "The Eiffel Tower was completed in 1925 and is made of concrete."


def show(name: str, result) -> None:  # noqa: ANN001
    print(
        f"{name:<12} passing={result.passing} score={result.score:.2f} "
        f"feedback={result.feedback!r} model={result.model}"
    )


faithfulness = NaceFaithfulnessEvaluator()
relevancy = NaceRelevancyEvaluator()
correctness = NaceCorrectnessEvaluator()

for label, answer in [("good", GOOD), ("bad", BAD)]:
    print(f"--- {label} answer: {answer}")
    show("faithful", faithfulness.evaluate(query=QUERY, response=answer, contexts=CONTEXTS))
    show("relevant", relevancy.evaluate(query=QUERY, response=answer, contexts=CONTEXTS))
    show(
        "correct",
        correctness.evaluate(query=QUERY, response=answer, reference="It opened in 1889."),
    )
