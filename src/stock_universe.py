"""HSI constituent universe with normalized weights for bottom-up aggregation."""

from config import HSI_CONSTITUENTS


def get_universe() -> dict:
    total = sum(HSI_CONSTITUENTS.values())
    return {k: v / total for k, v in HSI_CONSTITUENTS.items()}
