"""Эмуляция stop-loss и take-profit через market-заявки."""

# Закрытая часть executor/recovery: ArenaGo не дает нативных стопов, поэтому
# стопы и тейки хранятся в state и исполняются отдельным watcher-циклом.

from __future__ import annotations

from logging import Logger
from pathlib import Path

from src.arenago_client import ArenaGoClient, ArenaGoError
from src.data.interface import MarketDataProvider
from src.logger import write_audit_event
from src.schemas import OrderSide, PositionSide, StopSpec
from src.state import StateStore


class StopWatcher:
    """Проверяет виртуальные stop-loss/take-profit из checkpoint."""

    def __init__(
        self,
        client: ArenaGoClient,
        state_store: StateStore,
        market_data: MarketDataProvider,
        data_dir: Path,
        logger: Logger,
        trading_enabled: bool = False,
    ):
        """Инициализирует watcher с клиентом ArenaGo и источником последней цены."""

        self.client = client
        self.state_store = state_store
        self.market_data = market_data
        self.data_dir = data_dir
        self.logger = logger
        self.trading_enabled = trading_enabled

    def run_once(self) -> None:
        """Делает один проход по активным стопам и исполняет сработавшие уровни."""

        state = self.state_store.load()
        for stop_id, stop in list(state.active_stops.items()):
            price = self.market_data.get_last_price(stop.ticker)
            if price is None:
                continue

            trigger = self._trigger_reason(stop, price)
            if trigger is None:
                continue

            direction = OrderSide.SELL if stop.side == PositionSide.LONG else OrderSide.BUY
            if not self.trading_enabled:
                self._audit(
                    "stop_trigger_dry_run",
                    stop,
                    price,
                    trigger,
                    {
                        "request": {
                            "direction": direction.value,
                            "secid": stop.ticker,
                            "quantity": stop.quantity,
                        }
                    },
                )
                continue

            try:
                response = self.client.submit_order(
                    direction=direction,
                    secid=stop.ticker,
                    quantity=stop.quantity,
                )
            except ArenaGoError as error:
                self._audit("stop_trigger_failed", stop, price, trigger, {"error": str(error)})
                continue

            self.state_store.remove_stop(stop_id)
            self._audit(
                "stop_trigger_executed",
                stop,
                price,
                trigger,
                {"response": response.model_dump(mode="json")},
            )

    @staticmethod
    def _trigger_reason(stop: StopSpec, price: float) -> str | None:
        """Возвращает тип срабатывания стопа или `None`, если цена не дошла."""

        if stop.side == PositionSide.LONG:
            if stop.stop_price is not None and price <= stop.stop_price:
                return "stop_loss"
            if stop.take_price is not None and price >= stop.take_price:
                return "take_profit"
            return None

        if stop.stop_price is not None and price >= stop.stop_price:
            return "stop_loss"
        if stop.take_price is not None and price <= stop.take_price:
            return "take_profit"
        return None

    def _audit(self, event_type: str, stop: StopSpec, price: float, trigger: str, payload: dict) -> None:
        """Пишет audit-событие по обработке виртуального стопа."""

        event = {
            "stop": stop.model_dump(mode="json"),
            "price": price,
            "trigger": trigger,
            **payload,
        }
        write_audit_event(self.data_dir, event_type, event)
        self.logger.info(event_type, extra={"event": event})
