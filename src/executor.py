"""Исполнение DecisionPlan через ArenaGo API."""

# Закрытая часть торгового контура: этот модуль принимает готовый DecisionPlan,
# проверяет риск и исполняет заявки через ArenaGoClient.

from __future__ import annotations

from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from uuid import uuid4

from src.arenago_client import ArenaGoClient, ArenaGoError
from src.data.interface import MarketDataProvider
from src.logger import write_audit_event
from src.risk import PreTradeValidator
from src.schemas import (
    DecisionAction,
    DecisionOrder,
    DecisionPlan,
    ExecutionResult,
    ExecutionStatus,
    OrderRequest,
    OrderSide,
    Position,
    PositionSide,
    StopSpec,
)
from src.state import StateStore


class Executor:
    """Исполнитель торговых планов, полученных от decision-layer."""

    def __init__(
        self,
        client: ArenaGoClient,
        state_store: StateStore,
        risk: PreTradeValidator,
        market_data: MarketDataProvider,
        data_dir: Path,
        logger: Logger,
        trading_enabled: bool = False,
    ):
        """Инициализирует executor с API-клиентом, risk gate и источником цены."""

        self.client = client
        self.state_store = state_store
        self.risk = risk
        self.market_data = market_data
        self.data_dir = data_dir
        self.logger = logger
        self.trading_enabled = trading_enabled

    def execute_plan(self, plan: DecisionPlan) -> list[ExecutionResult]:
        """Проверяет и исполняет все заявки из одного `DecisionPlan`."""

        cash = self.client.get_cash_balance()
        positions = self.client.get_positions()
        trades_today = self.client.get_trades()
        state = self.state_store.reconcile_from_snapshots(cash, positions, trades_today)

        results: list[ExecutionResult] = []
        for order in sorted(plan.orders, key=lambda item: item.priority):
            try:
                request, reference_price = self._build_request(order, positions)
            except ValueError as error:
                result = ExecutionResult(
                    order=order,
                    status=ExecutionStatus.SKIPPED,
                    reason=str(error),
                )
                self._audit_result(result)
                results.append(result)
                continue

            validation = self.risk.validate(
                request=request,
                order_value=reference_price * request.quantity,
                cash_balance=cash,
                state=state,
                positions=positions,
                closing_order=order.action in {DecisionAction.CLOSE, DecisionAction.REDUCE},
            )
            if not validation.allowed:
                result = ExecutionResult(
                    order=order,
                    status=ExecutionStatus.REJECTED,
                    reason=validation.reason,
                    request=request,
                )
                self._audit_result(result)
                results.append(result)
                continue

            if not self.trading_enabled:
                result = ExecutionResult(
                    order=order,
                    status=ExecutionStatus.DRY_RUN,
                    reason="trading_disabled",
                    request=request,
                )
                self._audit_result(result)
                results.append(result)
                continue

            try:
                response = self.client.submit_order(
                    direction=request.direction,
                    secid=request.secid,
                    quantity=request.quantity,
                )
            except ArenaGoError as error:
                result = ExecutionResult(
                    order=order,
                    status=ExecutionStatus.FAILED,
                    reason=str(error),
                    request=request,
                )
                self._audit_result(result)
                results.append(result)
                continue

            if response.remaining_cash is not None:
                cash = response.remaining_cash

            if response.price and response.quantity:
                self._register_stop_if_needed(order, request, response.price, response.quantity)

            result = ExecutionResult(
                order=order,
                status=ExecutionStatus.EXECUTED,
                reason="ok",
                request=request,
                response=response,
            )
            self._audit_result(result)
            results.append(result)
            positions = self.client.get_positions()
            trades_today = self.client.get_trades()
            state = self.state_store.reconcile_from_snapshots(cash, positions, trades_today)

        cash = self.client.get_cash_balance()
        positions = self.client.get_positions()
        trades_today = self.client.get_trades()
        self.state_store.reconcile_from_snapshots(cash, positions, trades_today)
        return results

    def _build_request(self, order: DecisionOrder, positions: list[Position]) -> tuple[OrderRequest, float]:
        """Преобразует доменное торговое намерение в заявку ArenaGo."""

        current_position = self._position_for(order.ticker, positions)
        direction = self._direction_for(order, current_position)
        quantity = self._quantity_for(order, current_position)
        reference_price = self.market_data.get_last_price(order.ticker)
        if reference_price is None:
            raise ValueError(f"no_reference_price_for_{order.ticker}")
        return (
            OrderRequest(
                direction=direction,
                secid=order.ticker,
                quantity=quantity,
                bot=self.client.bot_name,
            ),
            reference_price,
        )

    def _direction_for(self, order: DecisionOrder, current_position: int) -> OrderSide:
        """Выводит сторону `B` или `S` по действию и текущей позиции."""

        if order.action == DecisionAction.OPEN_LONG:
            return OrderSide.BUY
        if order.action == DecisionAction.INCREASE:
            return OrderSide.SELL if current_position < 0 else OrderSide.BUY
        if order.action == DecisionAction.OPEN_SHORT:
            return OrderSide.SELL
        if order.action in {DecisionAction.CLOSE, DecisionAction.REDUCE}:
            if current_position > 0:
                return OrderSide.SELL
            if current_position < 0:
                return OrderSide.BUY
            raise ValueError(f"no_position_to_{order.action.value}_{order.ticker}")
        raise ValueError(f"unsupported_action_{order.action.value}")

    def _quantity_for(self, order: DecisionOrder, current_position: int) -> int:
        """Рассчитывает целое количество бумаг для заявки."""

        if order.action == DecisionAction.CLOSE:
            quantity = abs(current_position)
        elif order.size_lots is not None:
            quantity = order.size_lots
        elif order.size_rub is not None:
            reference_price = self.market_data.get_last_price(order.ticker)
            if reference_price is None:
                raise ValueError(f"no_reference_price_for_{order.ticker}")
            quantity = int(order.size_rub // reference_price)
        else:
            raise ValueError(f"no_size_for_{order.ticker}")

        if quantity <= 0:
            raise ValueError(f"zero_quantity_for_{order.ticker}")
        if order.action == DecisionAction.REDUCE and current_position != 0:
            quantity = min(quantity, abs(current_position))
        return quantity

    @staticmethod
    def _position_for(secid: str, positions: list[Position]) -> int:
        """Возвращает размер текущей позиции по тикеру из списка ArenaGo."""

        for position in positions:
            if position.secid == secid:
                return int(position.position)
        return 0

    def _register_stop_if_needed(
        self,
        order: DecisionOrder,
        request: OrderRequest,
        entry_price: float,
        quantity: int,
    ) -> None:
        """Создает виртуальный stop/take после подтвержденного открытия позиции."""

        if order.action not in {DecisionAction.OPEN_LONG, DecisionAction.OPEN_SHORT, DecisionAction.INCREASE}:
            return
        if order.stop_price is None and order.take_price is None:
            return

        side = PositionSide.LONG if request.direction == OrderSide.BUY else PositionSide.SHORT
        stop = StopSpec(
            stop_id=f"{order.ticker}:{uuid4().hex}",
            ticker=order.ticker,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=order.stop_price,
            take_price=order.take_price,
            opened_at=datetime.now(timezone.utc),
            reason=order.reason_summary,
        )
        self.state_store.add_stop(stop)

    def _audit_result(self, result: ExecutionResult) -> None:
        """Пишет результат обработки заявки в operational log и audit log."""

        write_audit_event(
            self.data_dir,
            "execution_result",
            result.model_dump(mode="json"),
        )
        self.logger.info(
            "execution_result",
            extra={"event": result.model_dump(mode="json")},
        )
