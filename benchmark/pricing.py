"""List prices in USD per 1M tokens. Cost = reported usage tokens x these prices."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """Per-1M-token prices, where they come from, and when they were checked."""

    input_per_m: float
    output_per_m: float
    source: str
    checked: str


PRICES: dict[str, Price] = {
    # Nace docs: drex-v2.0 is billed on input tokens only; output is free.
    "drex-v2.0": Price(
        0.06, 0.0, "https://console.nace.ai/docs/drex/reference/pricing", "2026-10-28"
    ),
    # Nace docs: drex-max-v1.1 is billed on input tokens only; output is free.
    "drex-max-v1.1": Price(
        0.03, 0.0, "https://console.nace.ai/docs/drex/reference/pricing", "2026-10-29"
    ),
    # Anthropic list price.
    "claude-sonnet-5-5": Price(2.00, 10.00, "https://www.anthropic.com/pricing", "2026-10-28"),
    # Standard tier, no cached-input discount applied (the benchmark sends cold prompts).
    "gpt-5.4-nano": Price(
        0.20,
        1.25,
        "https://developers.openai.com/api/docs/pricing (standard tier); listed Deprecated",
        "2026-10-28",
    ),
}
