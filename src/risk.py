"""Детерминированная проверка заявки перед отправкой."""

from __future__ import annotations

from pydantic import BaseModel

from src.schemas import BotState, OrderRequest, OrderSide, Position, ValidationResult


class RiskConfig(BaseModel):
    """Набор базовых лимитов для pre-trade проверки."""

    max_daily_trades: int = 1000
    max_order_cash_share: float = 0.20  # снижено с 0.30 после инцидента MOEX-лонг 25% портфеля
    min_cash_reserve: float = 10_000
    allow_shorts: bool = True


class PreTradeValidator:
    """Детерминированный risk gate перед отправкой заявки."""

    def __init__(self, config: RiskConfig):
        """Создает валидатор с фиксированными лимитами риска."""

        self.config = config

    def validate(
        self,
        request: OrderRequest,
        order_value: float,
        cash_balance: float,
        state: BotState,
        positions: list[Position],
        closing_order: bool = False,
    ) -> ValidationResult:
        """Возвращает разрешение или причину отказа для заявки."""

        if state.safe_mode and not closing_order:
            return ValidationResult(allowed=False, reason="safe_mode_enabled")

        if state.daily_trade_count >= self.config.max_daily_trades:
            return ValidationResult(allowed=False, reason="daily_trade_limit_reached")

        if order_value <= 0:
            return ValidationResult(allowed=False, reason="order_value_is_not_positive")

        if closing_order:
            return ValidationResult(allowed=True)

        if order_value > max(cash_balance, 1.0) * self.config.max_order_cash_share:
            return ValidationResult(allowed=False, reason="order_value_exceeds_cash_share_limit")

        if request.direction == OrderSide.BUY and cash_balance - order_value < self.config.min_cash_reserve:
            return ValidationResult(allowed=False, reason="cash_reserve_would_be_broken")

        if request.direction == OrderSide.SELL and not self.config.allow_shorts:
            current_position = self._position_for(request.secid, positions)
            if request.quantity > current_position:
                return ValidationResult(allowed=False, reason="shorts_are_disabled")

        return ValidationResult(allowed=True)

    @staticmethod
    def _position_for(secid: str, positions: list[Position]) -> int:
        """Возвращает текущую позицию по инструменту из snapshots ArenaGo."""

        for position in positions:
            if position.secid == secid:
                return int(position.position)
        return 0
