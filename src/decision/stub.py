"""Безопасная заглушка decision-layer."""

from __future__ import annotations

from src.decision.interface import DecisionContext
from src.schemas import DecisionPlan


class StubDecisionProvider:
    """Безопасный provider, который не создает торговых заявок."""

    def decide(self, context: DecisionContext) -> DecisionPlan:
        """Возвращает пустой план до подключения настоящего LLM-агента."""

        return DecisionPlan(
            orders=[],
            no_action_tickers=list(context.state.positions.keys()),
            cycle_summary="Decision stub is active; no orders are emitted",
            mode="stub",
        )
