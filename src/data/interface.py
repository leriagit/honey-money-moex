"""Интерфейс источника рыночных данных."""

from __future__ import annotations

from typing import Protocol


# Команда data pipeline расширяет этот контракт, если executor-у и
# stop-watcher-у понадобится больше данных, чем последняя цена.
class MarketDataProvider(Protocol):
    """Протокол минимального источника последней цены."""

    def get_last_price(self, secid: str) -> float | None:
        """Возвращает последнюю цену инструмента или `None` при отсутствии данных."""

        raise NotImplementedError
