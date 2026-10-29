"""Branch a LlamaIndex Workflow on a Drex routing decision, with a human-review escape hatch."""

import asyncio

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.selectors.nace import NaceLowConfidenceError, NaceSingleSelector

CHOICES = [
    "billing: invoices, refunds, charges, subscriptions and payment methods",
    "technical: errors, outages, API usage, integrations and bugs",
]


class BillingEvent(Event):
    """A billing question."""

    message: str
    probabilities: dict[str, float]


class TechEvent(Event):
    """A technical question."""

    message: str
    probabilities: dict[str, float]


class HumanReviewEvent(Event):
    """Drex was not confident enough to route automatically."""

    message: str
    probabilities: dict[str, float]


class SupportTriage(Workflow):
    """Route a support message to billing, tech or a human. No LLM is involved."""

    selector = NaceSingleSelector(min_confidence=0.6)

    @step
    async def classify(self, ev: StartEvent) -> BillingEvent | TechEvent | HumanReviewEvent:
        try:
            result = await self.selector.aselect(CHOICES, ev.message)
        except NaceLowConfidenceError as error:
            probabilities = getattr(error, "probabilities", {}) or {}
            return HumanReviewEvent(message=ev.message, probabilities=probabilities)
        selection = result.selections[0]
        probabilities = result.probabilities  # type: ignore[attr-defined]
        if selection.index == 0:
            return BillingEvent(message=ev.message, probabilities=probabilities)
        return TechEvent(message=ev.message, probabilities=probabilities)

    @step
    async def billing(self, ev: BillingEvent) -> StopEvent:
        return StopEvent(result={"route": "billing", "probabilities": ev.probabilities})

    @step
    async def tech(self, ev: TechEvent) -> StopEvent:
        return StopEvent(result={"route": "technical", "probabilities": ev.probabilities})

    @step
    async def human(self, ev: HumanReviewEvent) -> StopEvent:
        return StopEvent(result={"route": "human_review", "probabilities": ev.probabilities})


async def main() -> None:
    workflow = SupportTriage(timeout=60)
    for message in [
        "I was charged twice for my March invoice, please refund one.",
        "The API returns a 500 error whenever I upload a file over 10 MB.",
        "Thanks!",
    ]:
        result = await workflow.run(message=message)
        print(f"{message!r}\n  -> {result}")


if __name__ == "__main__":
    asyncio.run(main())
