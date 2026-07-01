"""Контракт подключения decision-layer."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from src.schemas import BotState, DecisionPlan, Position, Trade


# Команда LLM подключает здесь боевой provider: он должен принять
# DecisionContext и вернуть DecisionPlan с ордерами для Executor.
class DecisionContext(BaseModel):
    """Входной контекст для decision-provider."""

    state: BotState
    positions: list[Position] = Field(default_factory=list)
    trades_today: list[Trade] = Field(default_factory=list)


class DecisionProvider(Protocol):
    """Протокол подключаемого торгового decision-layer."""

    def decide(self, context: DecisionContext) -> DecisionPlan:
        """Возвращает торговый план для одного decision-cycle."""

        raise NotImplementedError
